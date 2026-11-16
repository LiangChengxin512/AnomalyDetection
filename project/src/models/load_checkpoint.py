"""Atomic CPU checkpoints; construction, feature order and masks are mandatory."""
from __future__ import annotations
import os
from pathlib import Path
import torch
from src.common.device import get_best_device, move_optimizer_to_device


def checkpoint_dir_for(root, model_name, dataset_version):
    p = Path(root) / f'{model_name}_{dataset_version}'
    p.mkdir(parents=True, exist_ok=True)
    return p


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(v) for v in value)
    return value


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(cpu_state(payload), tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config, extra=None):
    extra = extra or {}
    payload = dict(schema_version='2.0', epoch=epoch, model_state_dict=model.state_dict(), metrics=metrics,
                   config=config, model_name=model.name, n_feats=model.n_feats, n_window=model.n_window, extra=extra)
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler_state_dict'] = scheduler.state_dict()
    atomic_save(path, payload)


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, device_requested=None,
                    expected=None, allow_legacy=False):
    # Checkpoints are trusted local artifacts, not untrusted downloaded pickle files.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    meta = payload.get('extra', {})
    if not allow_legacy and (not meta.get('feature_order') or 'continuous_mask' not in meta):
        raise ValueError('Legacy checkpoint lacks feature order/mask provenance; audit it, then retrain under the frozen protocol')
    if expected:
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f'Checkpoint mismatch for {key}')
    from src.models.tranad_adapter import build_model
    device = get_best_device(device_requested)
    if model is None:
        model = build_model(payload['model_name'], payload['n_feats'], payload['n_window'], torch.device('cpu'), torch.float32)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if 'continuous_mask' in meta:
        if len(meta['continuous_mask']) != payload['n_feats'] or len(meta['feature_order']) != payload['n_feats']:
            raise ValueError('Checkpoint feature metadata dimensions disagree')
        model.set_continuous_mask(meta['continuous_mask'])
    model = model.to(device=device, dtype=torch.float32)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        move_optimizer_to_device(optimizer, device)
    if scheduler is not None:
        scheduler.load_state_dict(payload['scheduler_state_dict'])
    return model, payload
