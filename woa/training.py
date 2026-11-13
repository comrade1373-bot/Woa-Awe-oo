import hashlib
import importlib.util
import math
import os
import pickle
import sys
import tempfile
import time
import threading
import uuid
from pathlib import Path

from .communication import communication_cues, prompt_guidance
from .hardware import detect_hardware, recommended_batch_size
from .model import ModelConfig, ModelDependencyError, build_model
from .tokenizer import ByteTokenizer


CHECKPOINT_FORMAT_VERSION = 1
_CHAT_SYSTEM_PROMPT = (
    "You are Woa, a helpful general assistant. Answer directly and truthfully. "
    "For broad goals, give ordered tasks, dependencies, checkpoints, risks, "
    "and success checks. Never claim unperformed work."
)


def _optimizer_checkpoint_path(checkpoint_path, checkpoint_id):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(
        f"{checkpoint_path.name}.{checkpoint_id}.optimizer.pt"
    )


def _checkpoint_id(payload):
    try:
        return str(uuid.UUID(payload["checkpoint_id"]))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Checkpoint has an invalid identifier.") from error


def _existing_checkpoint_id(torch, checkpoint_path):
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("format_version") == CHECKPOINT_FORMAT_VERSION:
            return _checkpoint_id(payload)
    except (
        OSError,
        EOFError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        AttributeError,
        pickle.UnpicklingError,
    ):
        return None
    return None


def _atomic_torch_save(torch, payload, output_path):
    descriptor, temporary_path = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _fit_message(tokenizer, role, content, token_limit):
    encoded = tokenizer.encode_message(role, content)
    if len(encoded) <= token_limit:
        return encoded
    if token_limit < 2:
        return []

    content_limit = token_limit - 2
    content_bytes = content.encode("utf-8")
    prefix = content_bytes[:content_limit]
    while prefix:
        try:
            text = prefix.decode("utf-8")
            break
        except UnicodeDecodeError as error:
            prefix = prefix[: error.start]
    else:
        text = ""
    if (
        len(content_bytes) > len(prefix)
        and len(text.encode("utf-8")) + 3 <= content_limit
    ):
        text += "..."
    return tokenizer.encode_message(role, text)


def _build_inference_prompt(messages, context_length, tokenizer=None):
    if context_length < 2:
        raise ValueError("context_length must be at least two.")
    tokenizer = tokenizer or ByteTokenizer()
    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )
    latest_user = messages[latest_user_index] if latest_user_index is not None else None
    budget = context_length - 2

    user_tokens = (
        _fit_message(
            tokenizer,
            "user",
            latest_user["content"],
            budget,
        )
        if latest_user is not None
        else []
    )
    system_budget = min(180, budget * 2 // 3, budget - len(user_tokens))
    system_contents = [_CHAT_SYSTEM_PROMPT] + [
        message["content"]
        for message in messages
        if message.get("role") == "system"
    ]
    if latest_user is not None:
        system_contents.append(
            prompt_guidance(communication_cues(latest_user["content"]))
        )

    system_tokens = []
    for content in system_contents:
        remaining = system_budget - len(system_tokens)
        if remaining < 2:
            break
        system_tokens.extend(
            _fit_message(tokenizer, "system", content, remaining)
        )

    history = []
    history_budget = budget - len(system_tokens) - len(user_tokens)
    if latest_user_index is not None:
        previous_messages = messages[:latest_user_index]
    else:
        previous_messages = messages
    history_size = 0
    for message in reversed(previous_messages):
        if message.get("role") == "system":
            continue
        remaining = history_budget - history_size
        if remaining < 2:
            break
        encoded = _fit_message(
            tokenizer,
            message["role"],
            message["content"],
            remaining,
        )
        if not encoded:
            break
        history.append(encoded)
        history_size += len(encoded)

    prompt = [
        tokenizer.BOS,
        *system_tokens,
        *(token for message in reversed(history) for token in message),
        *user_tokens,
        tokenizer.ASSISTANT,
    ]
    return prompt


def load_text_corpus(paths):
    chunks = []
    seen_documents = set()
    for path in paths:
        source = Path(path)
        if source.is_dir():
            files = sorted(
                item
                for item in source.rglob("*")
                if item.is_file() and item.suffix.lower() in {".txt", ".md"}
            )
        else:
            if source.suffix.lower() not in {".txt", ".md"}:
                raise ValueError(f"Only .txt and .md files are supported: {source}")
            files = [source]
        for item in files:
            raw = item.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"Training source is not UTF-8 text: {item}") from error
            fingerprint = hashlib.sha256(raw).digest()
            if fingerprint not in seen_documents and text.strip():
                seen_documents.add(fingerprint)
                chunks.append(text)
    corpus = "\n\n".join(chunk for chunk in chunks if chunk.strip())
    if len(corpus.encode("utf-8")) < 2:
        raise ValueError("Training corpus must contain at least two UTF-8 bytes.")
    return corpus


def learning_rate(step, total_steps, base_lr, warmup_steps):
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    paths,
    output_path,
    *,
    steps=1000,
    config=None,
    batch_size=None,
    learning_rate_value=3e-4,
    warmup_steps=50,
    seed=1337,
    on_step=None,
    resume_from=None,
    checkpoint_interval=100,
    examples=None,
    initialize_from=None,
):
    try:
        import torch
    except ImportError as error:
        raise ModelDependencyError(
            "PyTorch is required for training. Install it with "
            '`pip install -e ".[train]"`.'
        ) from error
    if steps < 1:
        raise ValueError("steps must be at least one.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be at least one.")
    if learning_rate_value <= 0 or not math.isfinite(learning_rate_value):
        raise ValueError("learning_rate must be a positive finite number.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = ByteTokenizer()
    if examples is None:
        corpus = load_text_corpus(paths)
        token_ids = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
        corpus_bytes = len(corpus.encode("utf-8"))
    else:
        if not examples:
            raise ValueError("Supervised fine-tuning needs at least one example.")
        corpus = None
        token_ids = None
        corpus_bytes = sum(len(example["input_ids"]) for example in examples)
    resume_checkpoint = None
    resume_optimizer = None
    initial_checkpoint = None
    if resume_from is not None:
        try:
            resume_checkpoint = torch.load(
                resume_from, map_location="cpu", weights_only=True
            )
            if resume_checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
                raise ValueError("Unsupported checkpoint format version.")
            resumed_config = ModelConfig(**resume_checkpoint["config"])
            checkpoint_id = _checkpoint_id(resume_checkpoint)
            optimizer_path = _optimizer_checkpoint_path(resume_from, checkpoint_id)
            resume_optimizer = torch.load(
                optimizer_path, map_location="cpu", weights_only=True
            )
            if (
                resume_optimizer.get("checkpoint_id")
                != checkpoint_id
            ):
                raise ValueError("Optimizer state does not match model checkpoint.")
        except (
            OSError,
            EOFError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            AttributeError,
            pickle.UnpicklingError,
        ) as error:
            raise RuntimeError(f"Could not read resume checkpoint {resume_from}.") from error
        if config is not None and config != resumed_config:
            raise ValueError(
                "Resume checkpoint architecture does not match the requested config."
            )
        config = resumed_config
    else:
        if initialize_from is not None:
            try:
                initial_checkpoint = torch.load(
                    initialize_from, map_location="cpu", weights_only=True
                )
                if (
                    initial_checkpoint.get("format_version")
                    != CHECKPOINT_FORMAT_VERSION
                    or initial_checkpoint.get("tokenizer") != "byte-v1"
                ):
                    raise ValueError("Unsupported base checkpoint format.")
                base_config = ModelConfig(**initial_checkpoint["config"])
            except (
                OSError,
                EOFError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                AttributeError,
                pickle.UnpicklingError,
            ) as error:
                raise RuntimeError(
                    f"Could not read base checkpoint {initialize_from}."
                ) from error
            if config is not None and config != base_config:
                raise ValueError(
                    "Base checkpoint architecture does not match the requested config."
                )
            config = base_config
        else:
            config = config or ModelConfig()
    if resume_from is not None and initialize_from is not None:
        raise ValueError("Choose either resume_from or initialize_from, not both.")
    if config.vocab_size != tokenizer.VOCAB_SIZE:
        raise ValueError(
            f"Model vocab_size must match the byte tokenizer ({tokenizer.VOCAB_SIZE})."
        )
    if examples is None:
        if token_ids.numel() < config.context_length + 1:
            raise ValueError(
                f"Corpus needs at least {config.context_length + 1} tokens for this "
                "context length."
            )
    else:
        for index, example in enumerate(examples, start=1):
            input_ids = example.get("input_ids")
            targets = example.get("targets")
            if (
                not isinstance(input_ids, list)
                or not isinstance(targets, list)
                or not input_ids
                or len(input_ids) != len(targets)
                or len(input_ids) > config.context_length
                or not any(target != -100 for target in targets)
            ):
                raise ValueError(f"Invalid or overlong supervised example {index}.")

    profile = detect_hardware()
    device = torch.device(profile["device"])
    batch_size = recommended_batch_size(profile, batch_size)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_value)
    completed_steps = 0
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_optimizer["optimizer"])
        completed_steps = int(
            resume_checkpoint.get("training", {}).get("completed_steps", 0)
        )
    elif initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model"])
    torch.set_num_threads(min(8, max(1, profile["cpu_count"])))
    started = time.monotonic()
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    model.train()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_checkpoint_id = (
        _checkpoint_id(resume_checkpoint)
        if resume_checkpoint is not None
        else _existing_checkpoint_id(torch, output_path)
    )
    window_offsets = torch.arange(config.context_length + 1)

    def save_checkpoint(completed):
        nonlocal previous_checkpoint_id
        checkpoint_id = str(uuid.uuid4())
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "tokenizer": "byte-v1",
            "checkpoint_id": checkpoint_id,
            "config": config.to_dict(),
            "model": {
                name: tensor.detach().to("cpu")
                for name, tensor in model.state_dict().items()
            },
            "training": {
                "steps": steps,
                "completed_steps": completed_steps + completed,
                "seed": seed,
                "corpus_bytes": corpus_bytes,
                "device": str(device),
                "stage": "sft" if examples is not None else "pretraining",
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            },
        }
        _atomic_torch_save(
            torch,
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "checkpoint_id": checkpoint_id,
                "optimizer": optimizer.state_dict(),
            },
            _optimizer_checkpoint_path(output_path, checkpoint_id),
        )
        _atomic_torch_save(torch, payload, output_path)
        if previous_checkpoint_id and previous_checkpoint_id != checkpoint_id:
            old_optimizer_path = _optimizer_checkpoint_path(
                output_path, previous_checkpoint_id
            )
            try:
                old_optimizer_path.unlink()
            except FileNotFoundError:
                pass
        previous_checkpoint_id = checkpoint_id

    for step in range(steps):
        if examples is None:
            indices = torch.randint(
                0, token_ids.numel() - config.context_length, (batch_size,)
            )
            windows = token_ids[indices.unsqueeze(1) + window_offsets].to(device)
            inputs = windows[:, :-1]
            targets = windows[:, 1:]
        else:
            indices = torch.randint(0, len(examples), (batch_size,)).tolist()
            batch_length = max(len(examples[index]["input_ids"]) for index in indices)
            inputs = torch.zeros((batch_size, batch_length), dtype=torch.long)
            targets = torch.full((batch_size, batch_length), -100, dtype=torch.long)
            for row, index in enumerate(indices):
                example = examples[index]
                length = len(example["input_ids"])
                inputs[row, :length] = torch.tensor(example["input_ids"])
                targets[row, :length] = torch.tensor(example["targets"])
            inputs = inputs.to(device)
            targets = targets.to(device)
        current_lr = learning_rate(
            step,
            steps,
            learning_rate_value,
            min(warmup_steps, max(0, steps - 1)),
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            _, loss = model(inputs, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Training loss became non-finite at step {step + 1}.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        checkpoint_saved = (step + 1) % checkpoint_interval == 0 or step + 1 == steps
        if checkpoint_saved:
            save_checkpoint(step + 1)
        if on_step is not None:
            on_step(
                {
                    "step": completed_steps + step + 1,
                    "total_steps": completed_steps + steps,
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": current_lr,
                    "device": str(device),
                    "elapsed_seconds": time.monotonic() - started,
                    "checkpoint_saved": checkpoint_saved,
                }
            )
    return output_path


class ModelRuntime:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded_mtime = None
        self._context_length = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def _dependency_available(self, module_name):
        if sys.modules.get(module_name) is not None:
            return True
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            return False

    def _readiness(self):
        if not self.checkpoint_path.exists():
            return (
                False,
                f"No local model was found at {self.checkpoint_path}. Set "
                "WOA_MODEL_PATH to a local instruction/chat model directory, or "
                "train one with `python -m woa train --data PATH`.",
            )
        if self.checkpoint_path.is_dir():
            try:
                files = {
                    path.name
                    for path in self.checkpoint_path.iterdir()
                    if path.is_file()
                }
            except OSError:
                return (
                    False,
                    f"Could not read local model directory at {self.checkpoint_path}.",
                )
            missing = []
            if "config.json" not in files:
                missing.append("config.json")
            if not any(
                name.endswith((".safetensors", ".bin")) for name in files
            ):
                missing.append("model weights (.safetensors or .bin)")
            if not files.intersection(
                {
                    "tokenizer.json",
                    "tokenizer.model",
                    "spiece.model",
                    "vocab.json",
                    "vocab.txt",
                    "sentencepiece.bpe.model",
                }
            ):
                missing.append("tokenizer files")
            if missing:
                return (
                    False,
                    "Local model directory is incomplete; missing "
                    + ", ".join(missing)
                    + ".",
                )
        elif not self.checkpoint_path.is_file():
            return (
                False,
                f"Local model path is not a file or directory: {self.checkpoint_path}.",
            )

        if not self._dependency_available("torch"):
            return (
                False,
                "PyTorch is required for local model inference. "
                'Install it with `pip install -e ".[train]"`.',
            )
        if self.checkpoint_path.is_dir() and not self._dependency_available(
            "transformers"
        ):
            return (
                False,
                "Transformers is required for a local pretrained model. "
                'Install it with `pip install -e ".[local-model]"`.',
            )
        return (
            True,
            "Local model files and runtime dependencies are present; model loading "
            "will be verified on first use.",
        )

    @property
    def readiness_message(self):
        return self._readiness()[1]

    @property
    def available(self):
        return self._readiness()[0]

    @property
    def backend(self):
        return (
            "local_pretrained"
            if self.checkpoint_path.is_dir()
            else "project_owned"
        )

    def _path_mtime(self):
        if self.checkpoint_path.is_file():
            return self.checkpoint_path.stat().st_mtime_ns
        files = [
            path
            for path in self.checkpoint_path.iterdir()
            if path.is_file()
            and (
                path.name.endswith(".json")
                or path.name.endswith(".safetensors")
                or path.name.endswith(".bin")
            )
        ]
        if not files:
            return self.checkpoint_path.stat().st_mtime_ns
        return max(path.stat().st_mtime_ns for path in files)

    def _load(self):
        if not self.available:
            raise RuntimeError(self.readiness_message)
        try:
            import torch
        except ImportError as error:
            raise ModelDependencyError(
                "PyTorch is required for model inference. "
                'Install it with `pip install -e ".[train]"`.'
            ) from error
        with self._load_lock:
            try:
                checkpoint_mtime = self._path_mtime()
            except OSError as error:
                raise RuntimeError(
                    f"Could not access model checkpoint at {self.checkpoint_path}."
                ) from error
            if self._model is not None and checkpoint_mtime == self._loaded_mtime:
                return
            profile = detect_hardware()
            device = torch.device(profile["device"])
            tokenizer = None
            if self.checkpoint_path.is_dir():
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except ImportError as error:
                    raise ModelDependencyError(
                        "Transformers is required for a local pretrained model. "
                        'Install it with `pip install -e ".[local-model]"`.'
                    ) from error
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.checkpoint_path,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        self.checkpoint_path,
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                    config = model.config
                except (OSError, ValueError, AttributeError) as error:
                    raise RuntimeError(
                        "Could not load a local pretrained model from "
                        f"{self.checkpoint_path}. Check that it contains a "
                        "compatible tokenizer and causal language model."
                    ) from error
                context_length = next(
                    (
                        value
                        for value in (
                            getattr(config, "max_position_embeddings", None),
                            getattr(config, "n_positions", None),
                            getattr(config, "seq_length", None),
                        )
                        if isinstance(value, int) and value > 0
                    ),
                    4096,
                )
            else:
                try:
                    checkpoint = torch.load(
                        self.checkpoint_path, map_location="cpu", weights_only=True
                    )
                    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
                        raise ValueError("Unsupported checkpoint format version.")
                    if checkpoint.get("tokenizer") != "byte-v1":
                        raise ValueError("Unsupported checkpoint tokenizer.")
                    config = ModelConfig(**checkpoint["config"])
                    if config.vocab_size != ByteTokenizer.VOCAB_SIZE:
                        raise ValueError(
                            "Checkpoint vocabulary does not match byte-v1."
                        )
                    model = build_model(config)
                    model.load_state_dict(checkpoint["model"])
                except (
                    OSError,
                    KeyError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    AttributeError,
                    pickle.UnpicklingError,
                ) as error:
                    raise RuntimeError(
                        f"Could not load model checkpoint at {self.checkpoint_path}."
                    ) from error
                context_length = config.context_length

            previous_model = self._model
            previous_device = self._device
            if previous_model is not None:
                previous_model.to("cpu")
                if previous_device.type == "cuda":
                    torch.cuda.empty_cache()
            try:
                model.to(device)
            except RuntimeError as error:
                if previous_model is not None:
                    previous_model.to(previous_device)
                raise RuntimeError(
                    f"Insufficient memory to load model on {device}."
                ) from error
            model.eval()
            self._device = device
            self._model = model
            self._tokenizer = tokenizer
            self._loaded_mtime = checkpoint_mtime
            self._context_length = context_length

    def generate(self, messages, max_new_tokens=None):
        with self._generation_lock:
            self._load()
            import torch

            if max_new_tokens is None:
                max_new_tokens = 256 if self.backend == "local_pretrained" else 128

            if self.backend == "local_pretrained":
                tokenizer = self._tokenizer
                prompt_messages = list(messages)
                max_new_tokens = min(
                    max_new_tokens, max(1, self._context_length - 1)
                )
                input_limit = self._context_length - max_new_tokens

                def encode_prompt():
                    if getattr(tokenizer, "chat_template", None):
                        encoded = tokenizer.apply_chat_template(
                            prompt_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        )
                    else:
                        prompt = "\n".join(
                            f"{message['role'].capitalize()}: {message['content']}"
                            for message in prompt_messages
                        )
                        prompt += "\nAssistant:"
                        encoded = tokenizer(prompt, return_tensors="pt")["input_ids"]
                    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
                        encoded = encoded["input_ids"]
                    if encoded.ndim == 1:
                        encoded = encoded.unsqueeze(0)
                    return encoded

                input_ids = encode_prompt()
                while input_ids.shape[-1] > input_limit:
                    removable = next(
                        (
                            index
                            for index, message in enumerate(prompt_messages)
                            if message.get("role") != "system"
                            and index != len(prompt_messages) - 1
                        ),
                        None,
                    )
                    if removable is None:
                        input_ids = input_ids[:, -input_limit:]
                        break
                    del prompt_messages[removable]
                    input_ids = encode_prompt()

                input_ids = input_ids.to(self._device)
                attention_mask = torch.ones_like(input_ids)
                output = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id,
                )[0, input_ids.shape[-1] :]
                reply = tokenizer.decode(
                    output, skip_special_tokens=True
                ).strip()
                if not reply:
                    raise RuntimeError(
                        "The local pretrained model returned an empty response."
                    )
                return reply

            tokenizer = ByteTokenizer()
            prompt = _build_inference_prompt(
                messages,
                self._context_length,
                tokenizer=tokenizer,
            )
            input_ids = torch.tensor([prompt], dtype=torch.long, device=self._device)
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=40,
                eos_token_id=tokenizer.EOS,
                min_new_tokens=1,
                forbidden_token_ids=[
                    tokenizer.BOS,
                    tokenizer.SYSTEM,
                    tokenizer.USER,
                    tokenizer.ASSISTANT,
                ],
            )[0, len(prompt) :].tolist()
            reply = tokenizer.decode(output).strip()
            if not reply:
                raise RuntimeError(
                    "The project-trained model returned an empty response."
                )
            return reply


def evaluate_checkpoint(checkpoint_path, paths):
    try:
        import torch
        from torch.nn import functional as functional
    except ImportError as error:
        raise ModelDependencyError(
            "PyTorch is required to evaluate a model. "
            'Install it with `pip install -e ".[train]"`.'
        ) from error
    checkpoint_path = Path(checkpoint_path)
    corpus = load_text_corpus(paths)
    token_ids = torch.tensor(ByteTokenizer().encode(corpus), dtype=torch.long)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION
            or checkpoint.get("tokenizer") != "byte-v1"
        ):
            raise ValueError("Unsupported model checkpoint format.")
        config = ModelConfig(**checkpoint["config"])
        if config.vocab_size != ByteTokenizer.VOCAB_SIZE:
            raise ValueError("Checkpoint vocabulary does not match byte-v1.")
        model = build_model(config)
        model.load_state_dict(checkpoint["model"])
    except (
        OSError,
        EOFError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        AttributeError,
        pickle.UnpicklingError,
    ) as error:
        raise RuntimeError(f"Could not load model checkpoint {checkpoint_path}.") from error

    profile = detect_hardware()
    device = torch.device(profile["device"])
    model.to(device).eval()
    token_ids = token_ids.to(device)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, token_ids.numel() - 1, config.context_length):
            inputs = token_ids[start : start + config.context_length]
            targets = token_ids[start + 1 : start + config.context_length + 1]
            if inputs.numel() != targets.numel():
                inputs = inputs[:-1]
            logits = model(inputs.unsqueeze(0))
            loss_sum = functional.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                targets,
                reduction="sum",
            )
            total_loss += float(loss_sum.cpu())
            total_tokens += targets.numel()
    if not total_tokens:
        raise ValueError("Evaluation corpus contains no target tokens.")
    mean_loss = total_loss / total_tokens
    return {
        "tokens": total_tokens,
        "mean_cross_entropy": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80)),
        "device": str(device),
    }
