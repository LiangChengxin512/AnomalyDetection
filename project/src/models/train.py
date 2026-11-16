"""Training engine for TranAD-family models on HAI normal data.

Trains only on the ``train`` split (normal operation). Checkpoints land in
``checkpoints/<model>_<dataset_version>/`` with config + metrics embedded.
Designed to be launched by ``scripts/run_train.sh`` on CUDA / MPS / CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.device import get_best_device, get_dtype_for_device  # noqa: E402
from src.common.io_utils import read_json, write_json  # noqa: E402
from src.data.fit_scaler import apply_scaler  # noqa: E402
from src.data.window import build_index  # noqa: E402
from src.models.load_checkpoint import checkpoint_dir_for, save_checkpoint  # noqa: E402
from src.models.tranad_adapter import build_model, model_state_hash, predict_batch  # noqa: E402


def _aux_total(model, weights) -> torch.Tensor:
    losses = getattr(model, "auxiliary_losses", lambda: {})()
    total = next(model.parameters()).new_tensor(0.0)
    for name, val in losses.items():
        total = total + weights.get(name, weights.get("default", 1.0)) * val
    return total


def train_epoch(model, idx, order, optimizer, criterion, weights,
                batch_size: int, epoch: int, device, log_every: int = 200):
    model.train()
    tot, rec, aux = [], [], []
    t0 = time.time()
    for bi, lo in enumerate(range(0, len(order), batch_size)):
        sel = order[lo:lo + batch_size]
        batch = torch.as_tensor(idx.gather(sel), device=device)
        window = batch.permute(1, 0, 2)
        elem = window[-1, :, :].unsqueeze(0)
        x1, x2 = model(window, elem)
        n = epoch + 1
        loss1 = (1.0 / n) * criterion(x1, elem[0])
        loss2 = (1.0 - 1.0 / n) * criterion(x2, elem[0])
        recon = (loss1 + loss2).mean()
        a = _aux_total(model, weights)
        loss = recon + a
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot.append(float(loss.detach().cpu()))
        rec.append(float(recon.detach().cpu()))
        aux.append(float(a.detach().cpu()))
        if log_every and bi % log_every == 0:
            print(f"  batch {bi:6d}/{len(order)//batch_size} loss={np.mean(tot):.5f}",
                  flush=True)
    return {"train_loss": float(np.mean(tot)), "recon_loss": float(np.mean(rec)),
            "aux_loss": float(np.mean(aux)), "epoch_time_s": round(time.time() - t0, 2)}


@torch.no_grad()
def val_recon_loss(model, idx, order, criterion, batch_size, device) -> float:
    model.eval()
    losses, cnt = 0.0, 0
    for lo in range(0, len(order), batch_size):
        sel = order[lo:lo + batch_size]
        batch = torch.as_tensor(idx.gather(sel), device=device)
        x1, x2 = predict_batch(model, batch)
        elem = batch[:, -1, :]
        losses += float(((elem - x2) ** 2).mean().cpu()) * len(sel)
        cnt += len(sel)
    return losses / max(1, cnt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="override config model.name")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-train-windows", type=int, default=0,
                    help="cap windows per train file (smoke tests)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mname = args.model or cfg["model"]["name"]
    epochs = args.epochs or cfg["model"]["epochs"]
    batch_size = args.batch_size or cfg["model"]["batch_size"]
    lr = args.lr or cfg["model"]["lr"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 1)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = get_best_device(args.device)
    dtype = get_dtype_for_device(device)
    print(f"[train] model={mname} device={device} dtype={dtype} "
          f"epochs={epochs} batch={batch_size} lr={lr} seed={seed}")

    # data -------------------------------------------------------------
    out_dir = resolve(cfg, "processed_dir")
    scaler = read_json(out_dir / "scaler.json")
    tags = scaler["tags"]
    index = read_json(out_dir / "index.json")
    W = cfg["window"]["length"]
    stride = cfg["window"]["stride"]

    def frames_for(split):
        frames = []
        for rec in index["splits"][split]:
            df = pd.read_parquet(rec["parquet"], columns=tags)
            frames.append(apply_scaler(df[tags].to_numpy(np.float32), scaler))
        return frames

    train_frames = frames_for("train")
    val_frames = frames_for("val")
    train_idx = build_index(train_frames, W, stride=stride,
                            pad_mode=cfg["window"]["pad_mode"],
                            max_windows_per_segment=args.max_train_windows)
    val_idx = build_index(val_frames, W, stride=1,
                          pad_mode=cfg["window"]["pad_mode"],
                          max_windows_per_segment=args.max_train_windows)
    print(f"[train] train windows={len(train_idx)} val windows={len(val_idx)} W={W}")

    # model ------------------------------------------------------------
    model = build_model(mname, len(tags), cfg["model"]["n_window"], device, dtype,
                        lr=lr, seed=seed)
    if hasattr(model, "set_continuous_mask"):
        model.set_continuous_mask(torch.as_tensor(scaler["continuous_mask"],
                                                  device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg["model"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, cfg["model"]["lr_step"], cfg["model"]["lr_gamma"])
    criterion = nn.MSELoss(reduction="none")
    aux_weights = {"default": cfg["model"]["aux_weight"],
                   "association": cfg["model"]["assoc_weight"],
                   "memory_entropy": cfg["model"]["memory_weight"]}

    ckpt_dir = checkpoint_dir_for(resolve(cfg, "checkpoint_dir"), mname,
                                  cfg["dataset_version"])
    log_dir = Path(args.log_dir) if args.log_dir else resolve(cfg, "artifacts_dir") / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / f"config_{mname}.json",
               {"args": vars(args), "model": cfg["model"], "window": cfg["window"],
                "device": str(device), "model_version": model_state_hash(model)})

    start_epoch, best_val = 0, np.inf
    latest = ckpt_dir / "latest.ckpt"
    if args.resume and latest.exists():
        from src.models.load_checkpoint import load_checkpoint
        _, payload = load_checkpoint(latest, model, optimizer, scheduler,
                                     device_requested=args.device)
        start_epoch = payload["epoch"] + 1
        best_val = payload.get("metrics", {}).get("val_recon_loss", np.inf)
        print(f"[train] resumed at epoch {start_epoch}")

    hist = []
    rng = np.random.default_rng(seed)
    order_all = np.arange(len(train_idx))
    val_order = np.arange(len(val_idx))
    for epoch in range(start_epoch, start_epoch + epochs):
        order = rng.permutation(order_all)
        m = train_epoch(model, train_idx, order, optimizer, criterion,
                        aux_weights, batch_size, epoch, device)
        scheduler.step()
        vl = val_recon_loss(model, val_idx, val_order, criterion,
                            batch_size, device)
        m["val_recon_loss"] = vl
        m["lr"] = optimizer.param_groups[0]["lr"]
        m["epoch"] = epoch
        hist.append(m)
        print(f"[train] epoch {epoch}: {m}")
        save_checkpoint(latest, model, optimizer, scheduler, epoch, m,
                        {"name": mname, "cfg_model": cfg["model"]})
        if vl < best_val:
            best_val = vl
            save_checkpoint(ckpt_dir / "best.ckpt", model, optimizer,
                            scheduler, epoch, m, {"name": mname})
        pd.DataFrame(hist).to_csv(log_dir / f"metrics_{mname}.csv", index=False)
    print(f"[train] done. best_val_recon={best_val:.6f} ckpt_dir={ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
