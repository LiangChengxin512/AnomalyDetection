"""Model-evidence attribution for detected events.

Two complementary, cheap methods:

* ``input_gradient`` — gradient of the aggregated anomaly score w.r.t.
  the window input, read off at the last timestep (optionally with
  SmoothGrad noise averaging).
* ``occlusion`` — replace a channel's window with its train median and
  measure the score drop.

Attribution runs only on detected events (bounded by
``attribution.max_events``), never on the whole test set. When no
checkpoint is available (mock mode) a residual-correlation fallback is
used so the downstream pipeline can still be exercised.
"""
from __future__ import annotations

import numpy as np
import torch

from src.detect.score import aggregate, normalize_residuals
from src.models.tranad_adapter import predict_batch_grad


def _score_of(model, windows, scale, agg, topk):
    """windows: torch tensor [B,W,D] (requires grad handled by caller)."""
    x1, x2 = predict_batch_grad(model, windows)
    resid = torch.abs(windows[:, -1, :] - x2)
    s = torch.as_tensor(scale, dtype=resid.dtype, device=resid.device)
    e = resid / (s + 1e-8)
    if agg == "mean":
        return e.mean(dim=1)
    if agg == "max":
        return e.max(dim=1).values
    if agg == "p95":
        return torch.quantile(e, 0.95, dim=1)
    k = max(1, min(int(topk), e.shape[1]))
    return e.topk(k, dim=1).values.mean(dim=1)


def input_gradient_scores(model, windows_np, scale, agg, topk,
                          smoothgrad_samples=1, noise_std=0.0, device="cpu"):
    """Per-channel |d score / d x_last| attribution for a batch of windows."""
    base = torch.as_tensor(windows_np, dtype=torch.float32, device=device)
    grads = torch.zeros_like(base[:, -1, :])
    n = max(1, int(smoothgrad_samples))
    for _ in range(n):
        noisy = base
        if noise_std > 0:
            noisy = base + torch.randn_like(base) * noise_std
        noisy = noisy.clone().detach().requires_grad_(True)
        model.zero_grad()
        s = _score_of(model, noisy, scale, agg, topk)
        s.sum().backward()
        grads += noisy.grad[:, -1, :].abs().detach()
    return (grads / n).cpu().numpy()


def occlusion_scores(model, windows_np, scale, agg, topk, fill_values,
                     device="cpu", batch_limit=64):
    """Score drop when each channel is replaced by its normal fill value.

    ``fill_values``: [D] per-channel fill (train median, scaled space).
    """
    B, W, D = windows_np.shape
    base_score = _score_eval(model, windows_np, scale, agg, topk, device)
    out = np.zeros((B, D), dtype=np.float64)
    fill = np.broadcast_to(np.asarray(fill_values, dtype=np.float32), (W, D)).copy()
    for j in range(D):
        occl = windows_np.copy()
        occl[:, :, j] = fill[:, j]
        s = _score_eval(model, occl, scale, agg, topk, device)
        out[:, j] = np.maximum(0.0, base_score - s)  # drop = evidence for j
    return out


@torch.no_grad()
def _score_eval(model, windows_np, scale, agg, topk, device):
    w = torch.as_tensor(windows_np, dtype=torch.float32, device=device)
    return _score_of(model, w, scale, agg, topk).cpu().numpy()


def residual_fallback_scores(residual_evidence_event: np.ndarray) -> np.ndarray:
    """When no model is available: use mean normalized residual as proxy."""
    e = np.asarray(residual_evidence_event, dtype=np.float64)
    if e.ndim == 1:
        e = e[None, :]
    return np.abs(e)
