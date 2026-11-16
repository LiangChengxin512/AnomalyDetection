"""Normal-only training, effective batches, resumable early stopping and best restore."""
from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.common.device import get_best_device
from src.common.io_utils import read_json, write_json, sha256_file
from src.common.provenance import digest
from src.data.prepare_factory import scaled_frames
from src.data.window import build_index
from src.models.early_stopping import EarlyStopping
from src.models.load_checkpoint import load_checkpoint, save_checkpoint
from src.models.tranad_adapter import build_model, predict_batch


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def train_epoch(model, idx, order, optimizer, physical_batch, effective_batch, epoch, device):
    model.train()
    total = recon_total = 0.
    for start in range(0, len(order), effective_batch):
        group = order[start:start+effective_batch]
        optimizer.zero_grad(set_to_none=True)
        for lo in range(0, len(group), physical_batch):
            ids = group[lo:lo+physical_batch]
            batch = torch.as_tensor(idx.gather(ids), dtype=torch.float32, device=device)
            x1, x2 = predict_batch(model, batch)
            target = batch[:, -1, :]
            recon = ((x1-target).square()/epoch + (1-1/epoch)*(x2-target).square()).mean()
            aux = sum(model.auxiliary_losses().values(), target.new_tensor(0.))
            loss = recon + aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss in epoch {epoch}')
            (loss * len(ids) / len(group)).backward()
            total += float(loss.detach().cpu())*len(ids)
            recon_total += float(recon.detach().cpu())*len(ids)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
    return total/len(order), recon_total/len(order)


@torch.no_grad()
def validation_loss(model, idx, physical_batch, device):
    model.eval()
    total = 0.
    for lo in range(0, len(idx), physical_batch):
        ids = np.arange(lo, min(lo+physical_batch, len(idx)))
        batch = torch.as_tensor(idx.gather(ids), dtype=torch.float32, device=device)
        _, pred = predict_batch(model, batch)
        total += float((pred-batch[:, -1, :]).square().mean().cpu()) * len(ids)
    return total/len(idx)


def train_model(cfg, prepared, output, model_name, seed, device, run_id, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    scaler = read_json(prepared / 'scaler.json')
    index = read_json(prepared / 'index.json')
    training = cfg['training']
    W = cfg['window']['length']
    # Exclude padding from training/monitoring; replay retains and labels warmup.
    ti = build_index(scaled_frames(prepared, 'train'), W, training['train_stride'], 'skip')
    vi = build_index(scaled_frames(prepared, 'val_model'), W, 1, 'skip')
    if not len(ti) or not len(vi):
        raise ValueError('Empty training or validation windows')
    physical = training.get('physical_batch_size') or (16 if device.type == 'mps' else 128)
    effective = training['batch_size']
    if min(physical, effective, training['max_epochs']) <= 0:
        raise ValueError('Batches and total epoch budget must be positive')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(model_name, len(scaler['tags']), W, device, torch.float32, seed=seed)
    model.set_continuous_mask(scaler['continuous_mask'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['model']['lr'], weight_decay=cfg['model']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, cfg['model']['lr_step'], cfg['model']['lr_gamma'])
    stopping_cfg = {k:v for k,v in training['early_stopping'].items() if k in ['patience','min_epochs','min_delta_abs','min_delta_rel','enabled']}
    stopper = EarlyStopping(**stopping_cfg)
    binding = dict(feature_order=scaler['tags'], continuous_mask=scaler['continuous_mask'], seed=seed,
                   data_fingerprint=index['dataset_fingerprint'], scaler_hash=digest(scaler),
                   config_hash=digest({'training': training, 'model': cfg['model'], 'W': W, 'name': model_name}))
    history, start, resume_count = [], 0, 0
    latest = output / 'latest.ckpt'
    if resume and latest.exists():
        model, saved = load_checkpoint(latest, model, opt, scheduler, str(device), expected=binding)
        state = saved['extra']
        stopper = EarlyStopping(**state['early_stopping'])
        history, start = state['history'], saved['epoch']
        resume_count = state['resume_count']+1
        rng.bit_generator.state = state['sampler_rng']
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        if device.type == 'cuda' and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if device.type == 'mps' and state.get('mps_rng') is not None:
            torch.mps.set_rng_state(state['mps_rng'])
    for epoch in range(start+1, training['max_epochs']+1):
        if stopper.stopped:
            break
        sync(device); began = time.perf_counter()
        loss, recon = train_epoch(model, ti, rng.permutation(len(ti)), opt, physical, effective, epoch, device)
        val = validation_loss(model, vi, physical, device)
        best, reference = stopper.update(val, epoch)
        lr = opt.param_groups[0]['lr']
        scheduler.step(); sync(device)
        row = dict(schema_version='2.0', run_id=run_id, model_id=model_name, seed=seed, epoch=epoch,
                   train_loss=loss, train_recon_loss=recon, val_recon_mse=val, learning_rate=lr,
                   improved_best=best, improved_reference=reference, bad_epochs=stopper.bad_epochs,
                   epoch_seconds=time.perf_counter()-began, train_windows=len(ti), val_windows=len(vi),
                   device=str(device), physical_batch=physical, effective_batch=effective)
        history.append(row)
        extra = dict(binding, early_stopping=stopper.state_dict(), history=history,
                     sampler_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                     mps_rng=torch.mps.get_rng_state() if device.type == 'mps' else None,
                     resume_count=resume_count)
        save_checkpoint(latest, model, opt, scheduler, epoch, row, cfg['model'], extra)
        if best:
            save_checkpoint(output / 'best.ckpt', model, None, None, epoch, row, cfg['model'], extra)
        pd.DataFrame(history).to_csv(output / 'training_history.csv', index=False)
        print(f'[{model_name}/seed={seed}] epoch={epoch} val={val:.6g} bad_epochs={stopper.bad_epochs} seconds={row["epoch_seconds"]:.2f}', flush=True)
    # A successful return always uses the exact best state, including an early-stop resume.
    model, payload = load_checkpoint(output / 'best.ckpt', device_requested=str(device), expected=binding)
    model.eval()
    summary = dict(schema_version='2.0', run_id=run_id, model_id=model_name,
                   planned_epochs=training['max_epochs'], completed_epochs=len(history), best_epoch=stopper.best_epoch,
                   best_metric=stopper.best_metric, stop_reason='early_stopping' if stopper.stopped else 'max_epochs',
                   best_checkpoint='best.ckpt', checkpoint_sha256=sha256_file(output/'best.ckpt'),
                   total_train_seconds=sum(r['epoch_seconds'] for r in history), resume_count=resume_count, **binding)
    pd.DataFrame([{k:v for k,v in summary.items() if not isinstance(v, list)}]).to_csv(output/'training_summary.csv', index=False)
    write_json(output / 'training_summary.json', summary)
    return summary


def main():
    # The training wrapper uses the same run isolation, data preparation and protocol.
    from scripts.reproduce_all import main as pipeline_main
    import sys
    return pipeline_main(['--train-only'] + sys.argv[1:])

if __name__ == '__main__':
    raise SystemExit(main())
