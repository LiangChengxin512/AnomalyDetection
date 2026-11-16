"""Deterministic mock scorer for end-to-end pipeline checks.

Produces artifact-compatible ``scores_<split>_MOCK.parquet`` files WITHOUT
a trained model. The predictor is a simple per-channel exponential moving
average — a deliberately weak baseline so downstream stages can be tested
without GPU training.

Use ``--inject-signal`` ONLY for smoke tests: it nudges the score upward
inside labelled events so RCA/policy stages see detections. Mock outputs
are tagged model_version="MOCK-*" and must never appear in reported
metrics tables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import read_json, write_json, write_parquet  # noqa: E402
from src.data.fit_scaler import apply_scaler  # noqa: E402
from src.detect.score import aggregate, normalize_residuals, residual_scale  # noqa: E402


def ema_residual(x: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Residual of a per-channel EMA predictor, aligned to row t."""
    pred = np.empty_like(x)
    pred[0] = x[0]
    for i in range(1, len(x)):
        pred[i] = alpha * x[i - 1] + (1 - alpha) * pred[i - 1]
    resid = np.empty_like(x)
    resid[0] = 0.0
    resid[1:] = np.abs(x[1:] - pred[1:])
    return resid


def run_split(cfg, split: str, tags: list[str], scale: np.ndarray | None,
              inject: bool, seed: int):
    out_dir = resolve(cfg, "processed_dir")
    scaler = read_json(out_dir / "scaler.json")
    rng = np.random.default_rng(seed)
    frames, metas = [], []
    for rec in read_json(out_dir / "index.json")["splits"][split]:
        df = pd.read_parquet(rec["parquet"])
        metas.append(df)
        frames.append(apply_scaler(df[tags].to_numpy(np.float32), scaler))
    frames = [ema_residual(f) for f in frames]
    resid = np.concatenate(frames, axis=0)
    if scale is None:
        scale = residual_scale(resid, "std")
    e = normalize_residuals(resid, scale)
    score = aggregate(e, cfg["model"]["score_agg"], cfg["model"]["score_topk"])
    if inject:
        label = np.concatenate([m["label_attack"].to_numpy() for m in metas])
        score = score + label * (score.std() * 6.0 + 1e-6)
        e = e + label[:, None] * 5.0
    score = score + rng.normal(0, score.std() * 0.01 + 1e-9, len(score))
    df = pd.DataFrame({
        "timestamp": np.concatenate([m["timestamp"].to_numpy() for m in metas]),
        "score": score,
        "label_attack": np.concatenate([m["label_attack"].to_numpy() for m in metas]),
        "event_id": np.concatenate([m["event_id"].to_numpy() for m in metas]),
        "source_file": np.concatenate([m["source_file"].to_numpy() for m in metas]),
    })
    for j, t in enumerate(tags):
        df[f"e_{t}"] = e[:, j]
    df["model_version"] = f"MOCK-ema-{seed}"
    df["dataset_version"] = cfg["dataset_version"]
    df["end_row"] = np.arange(len(df))
    df["seg"] = np.repeat(np.arange(len(metas)), [len(m) for m in metas])
    return df, scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--inject-signal", action="store_true",
                    help="SMOKE TESTS ONLY: inflate scores inside labelled events")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tags = read_json(resolve(cfg, "processed_dir") / "scaler.json")["tags"]
    art = resolve(cfg, "artifacts_dir")

    scale = None
    for split in args.splits.split(","):
        split = split.strip()
        df, scale = run_split(cfg, split, tags, scale, args.inject_signal, args.seed)
        out = art / f"scores_{split}_MOCK.parquet"
        write_parquet(df, out)
        print(f"[mock] {split}: {len(df)} rows -> {out.name} (inject={args.inject_signal})")
    write_json(art / "residual_scale.json",
               {"tags": tags, "scale": scale.tolist(), "model_version": "MOCK"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
