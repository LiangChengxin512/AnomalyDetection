"""Adapter between the data pipeline and the TranAD-family models.

The data pipeline works with windows shaped ``[batch, W, D]``; TranAD uses
``[W, batch, D]`` internally and predicts the last row of the window. This
module keeps that convention in one place so callers do not need to know
about the permute.
"""
from __future__ import annotations

import hashlib

import torch

from src.common.device import get_best_device, get_dtype_for_device
from src.models.tranad_lib import build_tranad


def build_model(name: str, n_feats: int, n_window: int, device: torch.device,
                dtype: torch.dtype, lr: float = 1e-4, seed: int | None = None):
    model = build_tranad(name, n_feats, n_window, device=device, dtype=dtype,
                         lr=lr, seed=seed)
    return model


def model_state_hash(model) -> str:
    """Hash actual tensor contents, including persistent buffers."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    h.update(model.name.encode())
    return h.hexdigest()[:12]


def predict_batch(model, batch_bwd: torch.Tensor):
    """Run inference on a ``[B, W, D]`` batch.

    Returns ``(x1, x2)`` each ``[B, D]`` — the two-phase prediction of the
    last window element (TranAD convention).
    """
    if batch_bwd.ndim != 3 or batch_bwd.shape[1:] != (model.n_window, model.n_feats):
        raise ValueError("Expected [B, W, D] matching the model construction")
    window = batch_bwd.permute(1, 0, 2)              # [W, B, D]
    elem = window[-1, :, :].unsqueeze(0)             # [1, B, D]
    x1, x2 = model(window, elem)
    expected = (1, batch_bwd.shape[0], batch_bwd.shape[2])
    if x1.shape != expected or x2.shape != expected:
        raise ValueError(f"Reconstruction shape mismatch: {x1.shape}, {x2.shape}, expected {expected}")
    return x1[0], x2[0]


def predict_batch_grad(model, batch_bwd: torch.Tensor):
    """Same as :func:`predict_batch` but keeps the graph for attribution."""
    return predict_batch(model, batch_bwd)


def attention_summary(model) -> dict:
    """Collect captured attention weights after a forward pass.

    Requires ``model.set_capture_attention(True)`` before the forward.
    """
    out = {}
    enc = model.transformer_encoder.layers[-1].last_attn_weights
    if enc is not None:
        out['encoder_last'] = enc.detach().cpu()
    for i, dec in enumerate((model.transformer_decoder1, model.transformer_decoder2), 1):
        w = dec.layers[-1].last_cross_attn_weights
        if w is not None:
            out[f'decoder{i}_cross'] = w.detach().cpu()
    return out


def load_for_inference(checkpoint_path, device_requested="auto"):
    """Load a checkpoint saved by src.models.load_checkpoint for inference."""
    from src.models.load_checkpoint import load_checkpoint
    ckpt = load_checkpoint(checkpoint_path, device_requested=device_requested)
    return ckpt
