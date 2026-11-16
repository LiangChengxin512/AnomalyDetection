"""Checkpoint save/load with model version + config provenance."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from src.common.device import get_best_device, move_optimizer_to_device
from src.common.io_utils import git_commit


def checkpoint_dir_for(root, model_name, dataset_version):
    p = Path(root) / f"{model_name}_{dataset_version}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config,
                    extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": config,
        "model_name": model.name,
        "n_feats": model.n_feats,
        "n_window": model.n_window,
        "git_commit": git_commit(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path, model=None, optimizer=None, scheduler=None,
                    device_requested="auto"):
    """Load a checkpoint.

    If ``model`` is None, one is rebuilt from the saved metadata. Returns
    ``(model, payload)`` or just ``payload`` fields when a model is given.
    """
    device = get_best_device(device_requested)
    payload = torch.load(path, map_location=device, weights_only=False)
    if model is None:
        from src.models.tranad_adapter import build_model
        model = build_model(payload["model_name"], payload["n_feats"],
                            payload["n_window"], device, torch.float32)
    else:
        model = model.to(device)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        move_optimizer_to_device(optimizer, device)
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return model, payload
