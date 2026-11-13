import math
import os
import pickle
import tempfile
import time
import uuid
from pathlib import Path

from .hardware import detect_hardware, recommended_batch_size
from .model import ModelConfig, ModelDependencyError, build_model
from .training import (
    CHECKPOINT_FORMAT_VERSION,
    _atomic_torch_save,
    _checkpoint_id,
    _optimizer_checkpoint_path,
)


def _read_checkpoint(torch, path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION
            or checkpoint.get("tokenizer") != "byte-v1"
        ):
            raise ValueError("Unsupported model checkpoint format.")
        return checkpoint
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
        raise RuntimeError(f"Could not read model checkpoint {path}.") from error


def _batch(examples, indices, device):
    import torch

    max_length = max(len(examples[index]["input_ids"]) for index in indices)
    batch_size = len(indices)
    inputs = torch.zeros((batch_size, max_length), dtype=torch.long)
    targets = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row, index in enumerate(indices):
        example = examples[index]
        length = len(example["input_ids"])
        inputs[row, :length] = torch.tensor(example["input_ids"])
        targets[row, :length] = torch.tensor(example["targets"])
        mask[row, :length] = torch.tensor(example["response_mask"])
    return inputs.to(device), targets.to(device), mask.to(device)


def _response_log_probabilities(model, inputs, targets, response_mask):
    import torch
    from torch.nn import functional as functional

    logits = model(inputs)
    token_log_probabilities = functional.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probabilities * response_mask).sum(dim=-1)


def train_dpo(
    examples,
    reference_path,
    output_path,
    *,
    steps=500,
    batch_size=None,
    learning_rate=1e-5,
    beta=0.1,
    checkpoint_interval=100,
    resume_from=None,
    on_step=None,
):
    try:
        import torch
        from torch.nn import functional as functional
    except ImportError as error:
        raise ModelDependencyError(
            "PyTorch is required for preference optimization. "
            'Install it with `pip install -e ".[train]"`.'
        ) from error
    if not examples:
        raise ValueError("DPO requires at least one preference example.")
    if steps < 1 or checkpoint_interval < 1:
        raise ValueError("steps and checkpoint_interval must be at least one.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite.")
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("beta must be positive.")
    if Path(reference_path).resolve() == Path(output_path).resolve():
        raise ValueError("DPO output must not overwrite its frozen reference checkpoint.")
    reference_checkpoint = _read_checkpoint(torch, reference_path)
    reference_checkpoint_id = _checkpoint_id(reference_checkpoint)
    config = ModelConfig(**reference_checkpoint["config"])
    profile = detect_hardware()
    device = torch.device(profile["device"])
    reference = build_model(config)
    reference.load_state_dict(reference_checkpoint["model"])
    reference_device = torch.device("cpu")
    if device.type == "cuda":
        try:
            free_memory, _ = torch.cuda.mem_get_info(device)
            parameter_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in reference.parameters()
            )
            if free_memory >= 2 * parameter_bytes:
                reference_device = device
        except RuntimeError:
            reference_device = torch.device("cpu")
    reference.to(reference_device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    start_checkpoint = (
        _read_checkpoint(torch, resume_from)
        if resume_from is not None
        else reference_checkpoint
    )
    if start_checkpoint["config"] != reference_checkpoint["config"]:
        raise ValueError("DPO policy and reference architectures must match.")
    if resume_from is not None:
        training_metadata = start_checkpoint.get("training", {})
        if training_metadata.get("stage") != "dpo":
            raise ValueError("Resume checkpoint was not produced by DPO.")
        if training_metadata.get("reference_checkpoint_id") != reference_checkpoint_id:
            raise ValueError("DPO resume requires the original frozen reference model.")
    policy = build_model(config)
    policy.load_state_dict(start_checkpoint["model"])
    policy.to(device).train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    completed_steps = 0
    if resume_from is not None:
        try:
            checkpoint_id = _checkpoint_id(start_checkpoint)
            optimizer_checkpoint = torch.load(
                _optimizer_checkpoint_path(resume_from, checkpoint_id),
                map_location="cpu",
                weights_only=True,
            )
            if optimizer_checkpoint.get("checkpoint_id") != checkpoint_id:
                raise ValueError("DPO optimizer state does not match its checkpoint.")
            optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
            completed_steps = int(
                start_checkpoint.get("training", {}).get("completed_steps", 0)
            )
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
            raise RuntimeError(f"Could not resume DPO checkpoint {resume_from}.") from error

    batch_size = recommended_batch_size(profile, batch_size)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_examples = [example["chosen"] for example in examples]
    rejected_examples = [example["rejected"] for example in examples]
    started = time.monotonic()
    previous_checkpoint_id = (
        _checkpoint_id(start_checkpoint)
        if resume_from is not None
        else None
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
                for name, tensor in policy.state_dict().items()
            },
            "training": {
                "stage": "dpo",
                "completed_steps": completed_steps + completed,
                "device": str(device),
                "reference_checkpoint": str(reference_path),
                "reference_checkpoint_id": reference_checkpoint_id,
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
        if previous_checkpoint_id:
            previous_sidecar = _optimizer_checkpoint_path(
                output_path, previous_checkpoint_id
            )
            try:
                previous_sidecar.unlink()
            except FileNotFoundError:
                pass
        previous_checkpoint_id = checkpoint_id

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16
    for step in range(steps):
        indices = torch.randint(0, len(examples), (batch_size,)).tolist()
        chosen = _batch(chosen_examples, indices, device)
        rejected = _batch(rejected_examples, indices, device)
        reference_chosen = _batch(chosen_examples, indices, reference_device)
        reference_rejected = _batch(rejected_examples, indices, reference_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if reference_device.type == "cuda":
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    ref_chosen = _response_log_probabilities(
                        reference, *reference_chosen
                    )
                    ref_rejected = _response_log_probabilities(
                        reference, *reference_rejected
                    )
            else:
                ref_chosen = _response_log_probabilities(
                    reference, *reference_chosen
                )
                ref_rejected = _response_log_probabilities(
                    reference, *reference_rejected
                )
            ref_chosen = ref_chosen.to(device)
            ref_rejected = ref_rejected.to(device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            policy_chosen = _response_log_probabilities(policy, *chosen)
            policy_rejected = _response_log_probabilities(policy, *rejected)
            log_ratio = (policy_chosen - ref_chosen) - (
                policy_rejected - ref_rejected
            )
            loss = -functional.logsigmoid(beta * log_ratio).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"DPO loss became non-finite at step {step + 1}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        checkpoint_saved = (
            (step + 1) % checkpoint_interval == 0 or step + 1 == steps
        )
        if checkpoint_saved:
            save_checkpoint(step + 1)
        if on_step is not None:
            on_step(
                {
                    "step": completed_steps + step + 1,
                    "total_steps": completed_steps + steps,
                    "loss": float(loss.detach().cpu()),
                    "device": str(device),
                    "elapsed_seconds": time.monotonic() - started,
                    "checkpoint_saved": checkpoint_saved,
                }
            )
    return output_path
