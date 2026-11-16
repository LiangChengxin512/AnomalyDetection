"""Sliding-window inference replay.

Runs a trained TranAD-family checkpoint over prepared splits and writes
``scores_<split>_<model>.parquet`` with per-timestamp score, per-channel
normalized evidence, threshold (if calibrated) and provenance columns.

Also computes the train residual scale ``s_j`` (used to normalize
evidence) and stores it in ``residual_scale.json`` — fitted on the train
split only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.device import get_best_device  # noqa: E402
from src.common.io_utils import read_json, write_json, write_parquet  # noqa: E402
from src.data.window import build_index  # noqa: E402
from src.detect.score import aggregate, normalize_residuals, residual_scale  # noqa: E402
from src.models.load_checkpoint import load_checkpoint  # noqa: E402
from src.models.tranad_adapter import model_state_hash, predict_batch  # noqa: E402


def _split_frames(cfg, split: str, tags: list[str]) -> tuple[list[np.ndarray], list[pd.DataFrame]]:
    out_dir = resolve(cfg, "processed_dir")
    scaler = read_json(out_dir / "scaler.json")
    index = read_json(out_dir / "index.json")
    from src.data.fit_scaler import apply_scaler
    frames, metas = [], []
    for rec in index["splits"][split]:
        df = pd.read_parquet(rec["parquet"])
        metas.append(df[["timestamp", "label_attack", "event_id", "source_file"]])
        frames.append(apply_scaler(df[tags].to_numpy(np.float32), scaler))
    return frames, metas


def replay_split(model, frames, cfg, batch_size: int, device, stride: int = 1,
                 pad_mode: str = "replicate", max_windows: int = 0):
    """Returns (scores, evidence[N,D], end_rows) for all windows."""
    idx = build_index(frames, cfg["window"]["length"], stride=stride,
                      pad_mode=pad_mode, max_windows_per_segment=max_windows)
    n = len(idx)
    D = frames[0].shape[1]
    scores = np.empty(n, dtype=np.float64)
    evid = np.empty((n, D), dtype=np.float32)
    order = np.arange(n)
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for lo in range(0, n, batch_size):
            sel = order[lo:lo + batch_size]
            batch = torch.as_tensor(idx.gather(sel), device=device)
            _, x2 = predict_batch(model, batch)
            x_last = batch[:, -1, :]
            resid = (x_last - x2).detach().cpu().numpy()
            evid[sel] = resid
            scores[sel] = np.abs(resid).mean(axis=1)  # placeholder, replaced below
    return scores, evid, idx


def finalize_scores(raw_resid: np.ndarray, scale: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    e = normalize_residuals(raw_resid, scale)
    s = aggregate(e, cfg["model"]["score_agg"], cfg["model"]["score_topk"])
    return s, e


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a checkpoint over prepared splits.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--tag", default=None, help="artifact tag (default: model name)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tags = read_json(resolve(cfg, "processed_dir") / "scaler.json")["tags"]
    device = get_best_device(args.device)

    model, payload = load_checkpoint(args.checkpoint, device_requested=args.device)
    mver = f"{model.name}-{model_state_hash(model)}"
    tag = args.tag or model.name
    print(f"[replay] model={mver} ckpt={args.checkpoint} device={device}")

    # Pass 1: train residuals -> residual scale s_j (train split only).
    frames, _ = _split_frames(cfg, "train", tags)
    _, resid, _ = replay_split(model, frames, cfg, args.batch_size, device,
                               stride=args.stride, max_windows=args.max_windows)
    scale = residual_scale(resid, read_json(resolve(cfg, "processed_dir")
                                            / "scaler.json")["residual_scale_method"])
    write_json(resolve(cfg, "artifacts_dir") / "residual_scale.json",
               {"tags": tags, "scale": scale.tolist(), "model_version": mver})
    print(f"[replay] residual scale fitted on {resid.shape[0]} train windows")

    art = resolve(cfg, "artifacts_dir")
    for split in args.splits.split(","):
        split = split.strip()
        frames, metas = _split_frames(cfg, split, tags)
        _, resid, idx = replay_split(model, frames, cfg, args.batch_size,
                                     device, stride=args.stride,
                                     max_windows=args.max_windows)
        scores, evid = finalize_scores(resid, scale, cfg)
        # map each window end row back to its timestamp/label
        rec = {"timestamp": [], "score": scores, "label_attack": [],
               "event_id": [], "source_file": [], "end_row": idx.ends,
               "seg": idx.seg}
        for si, meta in enumerate(metas):
            m = idx.seg == si
            rec["timestamp"].append(meta["timestamp"].iloc[idx.ends[m]].to_numpy())
            rec["label_attack"].append(meta["label_attack"].iloc[idx.ends[m]].to_numpy())
            rec["event_id"].append(meta["event_id"].iloc[idx.ends[m]].to_numpy())
            rec["source_file"].append(meta["source_file"].iloc[idx.ends[m]].to_numpy())
        df = pd.DataFrame({
            "timestamp": np.concatenate(rec["timestamp"]),
            "score": scores,
            "label_attack": np.concatenate(rec["label_attack"]),
            "event_id": np.concatenate(rec["event_id"]),
            "source_file": np.concatenate(rec["source_file"]),
            "end_row": idx.ends, "seg": idx.seg,
        })
        for j, t in enumerate(tags):
            df[f"e_{t}"] = evid[:, j]
        df["model_version"] = mver
        df["dataset_version"] = cfg["dataset_version"]
        out = art / f"scores_{split}_{tag}.parquet"
        write_parquet(df, out)
        print(f"[replay] {split}: {len(df)} windows -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
