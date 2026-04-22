from __future__ import annotations

import torch


def resolve_torch_device(device: str, cuda_device_index: int = 0) -> tuple[torch.device, int | None]:
    """Map ``SATOR_DEVICE`` and ``SATOR_CUDA_DEVICE`` to a :class:`torch.device`.

    Returns ``(device, cuda_index_used)`` where *cuda_index_used* is the resolved
    integer index on CUDA, or ``None`` when running on CPU.
    """
    if device == "cuda" and torch.cuda.is_available():
        n = int(torch.cuda.device_count())
        if n <= 0:
            return torch.device("cpu"), None
        idx = int(cuda_device_index)
        if idx < 0 or idx >= n:
            idx = 0
        return torch.device(f"cuda:{idx}"), idx
    return torch.device("cpu"), None
