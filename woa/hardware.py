import os
import platform
from pathlib import Path


def _system_memory_bytes():
    limits = []
    if not hasattr(os, "sysconf"):
        physical_memory = None
    else:
        try:
            physical_memory = (
                os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            )
        except (OSError, ValueError):
            physical_memory = None
    if physical_memory:
        limits.append(physical_memory)
    for limit_path, current_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        try:
            raw_limit = Path(limit_path).read_text(encoding="ascii").strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
            current = int(Path(current_path).read_text(encoding="ascii").strip())
            if 0 < limit < 2**60:
                limits.append(max(1, limit - current))
        except (OSError, ValueError):
            continue
    return min(limits) if limits else None


def detect_hardware():
    """Report actual local compute; never claim or emulate unavailable devices."""
    system_memory = _system_memory_bytes()
    try:
        import torch
    except ImportError:
        return {
            "device": "cpu",
            "accelerator": None,
            "cpu_count": os.cpu_count() or 1,
            "platform": platform.platform(),
            "torch_available": False,
            "total_memory_bytes": None,
            "system_memory_bytes": system_memory,
        }

    device = "cpu"
    accelerator = None
    total_memory = None
    if torch.cuda.is_available():
        device = "cuda"
        accelerator = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        accelerator = "Apple Metal"

    return {
        "device": device,
        "accelerator": accelerator,
        "cpu_count": os.cpu_count() or 1,
        "platform": platform.platform(),
        "torch_available": True,
        "total_memory_bytes": total_memory,
        "system_memory_bytes": system_memory,
    }


def recommended_batch_size(profile, requested=None):
    if requested is not None:
        if requested < 1:
            raise ValueError("Batch size must be at least one.")
        return requested
    memory = profile.get("total_memory_bytes")
    if profile["device"] == "cuda" and memory:
        return max(1, min(32, int(memory // (2 * 1024**3))))
    if profile["device"] == "mps":
        system_memory = profile.get("system_memory_bytes")
        return max(1, min(8, int(system_memory // (8 * 1024**3)))) if system_memory else 2
    system_memory = profile.get("system_memory_bytes")
    if system_memory:
        return max(1, min(4, int(system_memory // (8 * 1024**3))))
    return 2


def recommended_model_settings(profile):
    """Conservative starting points, not guarantees of available memory."""
    memory = profile.get("total_memory_bytes")
    if profile["device"] == "cuda" and memory and memory >= 24 * 1024**3:
        return {
            "context_length": 512,
            "d_model": 256,
            "n_layers": 8,
            "n_heads": 8,
            "batch_size": 4,
        }
    if profile["device"] == "cuda" and memory and memory >= 12 * 1024**3:
        return {
            "context_length": 512,
            "d_model": 192,
            "n_layers": 6,
            "n_heads": 6,
            "batch_size": 2,
        }
    return {
        "context_length": 256,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "batch_size": recommended_batch_size(profile),
    }
