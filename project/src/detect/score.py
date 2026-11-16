"""Channel-residual normalization and score aggregation.

For channel ``j`` the per-timestamp evidence is
``e_{t,j} = |x_{t,j} - xhat_{t,j}| / s_j`` where ``s_j`` is the
train-normal residual scale (std or MAD, config ``scaler.residual_scale``).
``score_t`` is a configurable aggregation: mean | max | p95 | topk.
"""
from __future__ import annotations

import numpy as np


def residual_scale(residuals: np.ndarray, method: str = "std") -> np.ndarray:
    """Per-channel residual scale s_j from train-normal residuals [N, D]."""
    residuals = np.asarray(residuals, dtype=np.float64)
    if method == "mad":
        med = np.median(residuals, axis=0)
        s = np.median(np.abs(residuals - med), axis=0) * 1.4826
    else:
        s = residuals.std(axis=0)
    return np.where(s < 1e-8, 1e-8, s)


def normalize_residuals(residuals: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.abs(residuals) / (np.asarray(scale) + 1e-8)


def aggregate(e: np.ndarray, method: str = "topk", topk: int = 3) -> np.ndarray:
    """Aggregate per-channel evidence [N, D] -> score [N]."""
    if method == "mean":
        return e.mean(axis=1)
    if method == "max":
        return e.max(axis=1)
    if method == "p95":
        return np.percentile(e, 95, axis=1)
    if method == "topk":
        k = max(1, min(int(topk), e.shape[1]))
        part = np.partition(e, -k, axis=1)[:, -k:]
        return part.mean(axis=1)
    raise ValueError(f"Unsupported score aggregation: {method}")
