import hashlib
import math
import os
import pickle
import tempfile
import time
import threading
import uuid
from pathlib import Path

from .hardware import detect_hardware, recommended_batch_size
from .model import ModelConfig, ModelDependencyError, build_model
from .tokenizer import ByteTokenizer


CHECKPOINT_FORMAT_VERSION = 1


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

    corpus = load_text_corpus(paths)
    tokenizer = ByteTokenizer()
    token_ids = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    resume_checkpoint = None
    resume_optimizer = None
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
        config = config or ModelConfig()
    if config.vocab_size != tokenizer.VOCAB_SIZE:
        raise ValueError(
            f"Model vocab_size must match the byte tokenizer ({tokenizer.VOCAB_SIZE})."
        )
    if token_ids.numel() < config.context_length + 1:
        raise ValueError(
            f"Corpus needs at least {config.context_length + 1} tokens for this "
            "context length."
        )

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
                "corpus_bytes": len(corpus.encode("utf-8")),
                "device": str(device),
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
        indices = torch.randint(
            0, token_ids.numel() - config.context_length, (batch_size,)
        )
        inputs = torch.stack(
            [
                token_ids[index : index + config.context_length]
                for index in indices.tolist()
            ]
        ).to(device)
        targets = torch.stack(
            [
                token_ids[index + 1 : index + config.context_length + 1]
                for index in indices.tolist()
            ]
        ).to(device)
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
        self._device = None
        self._loaded_mtime = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @property
    def available(self):
        return self.checkpoint_path.is_file()

    def _load(self):
        if not self.available:
            raise RuntimeError(
                "No project-trained model checkpoint found. Train one with "
                "`python -m woa train --data PATH`."
            )
        try:
            import torch
        except ImportError as error:
            raise ModelDependencyError(
                "PyTorch is required for model inference. "
                'Install it with `pip install -e ".[train]"`.'
            ) from error
        with self._load_lock:
            try:
                checkpoint_mtime = self.checkpoint_path.stat().st_mtime_ns
            except OSError as error:
                raise RuntimeError(
                    f"Could not access model checkpoint at {self.checkpoint_path}."
                ) from error
            if self._model is not None and checkpoint_mtime == self._loaded_mtime:
                return
            profile = detect_hardware()
            device = torch.device(profile["device"])
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
                    raise ValueError("Checkpoint vocabulary does not match byte-v1.")
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
            self._loaded_mtime = checkpoint_mtime

    def generate(self, messages, max_new_tokens=128):
        with self._generation_lock:
            self._load()
            import torch

            tokenizer = ByteTokenizer()
            prompt = [
                tokenizer.BOS,
                *tokenizer.encode_message(
                    "system",
                    "You are Woa-Awe-oo. Be helpful, clear, and honest about uncertainty.",
                ),
            ]
            for message in messages:
                prompt.extend(
                    tokenizer.encode_message(message["role"], message["content"])
                )
            prompt.append(tokenizer.ASSISTANT)
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
