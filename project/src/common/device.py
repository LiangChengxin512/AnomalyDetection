"""Hardware-agnostic device selection (copied from the TranAD repository).

Supports auto / cuda / mps / cpu with an environment override
(``PROJECT_DEVICE``) and a CLI ``--device`` flag everywhere it is used.
"""
from __future__ import annotations

import os

import torch

DEVICE_ENV_VAR = "PROJECT_DEVICE"


def _mps_usable() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        if not backend.is_available():
            return False
        _ = torch.zeros(1, device="mps").sum().item()
        return True
    except Exception:
        return False


def get_best_device(requested: str = "auto", model_name: str | None = None) -> torch.device:
    env = os.environ.get(DEVICE_ENV_VAR)
    requested = (env or requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device '{requested}'. Use auto, cpu, cuda, or mps.")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    elif requested == "mps":
        if not _mps_usable():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
        device = torch.device("mps")
    elif requested == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif _mps_usable():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # float64 is poorly supported on MPS; model code uses float32 anyway.
    if device.type == "mps":
        torch.set_default_dtype(torch.float32)
    return device


def get_dtype_for_device(device: torch.device) -> torch.dtype:
    return torch.float32


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device,
                             dtype: torch.dtype | None = None) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                if dtype is not None and value.is_floating_point():
                    state[key] = value.to(device=device, dtype=dtype)
                else:
                    state[key] = value.to(device=device)
