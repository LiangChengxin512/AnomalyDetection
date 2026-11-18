"""Fit the input scaler on train-normal data only.

Convention (kept identical to upstream TranAD preprocessing): per-channel
min-max scaling into [0,1], fit on the TRAIN split only and applied
everywhere else. Discrete tags are min-max scaled the same way (their
0/1 values map to 0/1) but are flagged in ``continuous_mask`` so the
FeatureGate and downstream scoring can treat them differently.
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
from src.common.io_utils import write_json  # noqa: E402


def fit_scaler(cfg) -> dict:
    schema = yaml.safe_load(open(resolve(cfg, "tag_schema"), encoding="utf-8"))
    tags = [t["name"] for t in schema["tags"]]
    discrete = np.array([bool(t["discrete"]) for t in schema["tags"]])
    out_dir = resolve(cfg, "processed_dir")

    mins = np.full(len(tags), np.inf)
    maxs = np.full(len(tags), -np.inf)
    for rec in resolve(cfg, "processed_dir").glob("train__*.parquet"):
        df = pd.read_parquet(rec, columns=tags)
        v = df[tags].to_numpy(dtype=np.float64)
        mins = np.minimum(mins, np.nanmin(v, axis=0))
        maxs = np.maximum(maxs, np.nanmax(v, axis=0))
    span = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    scaler = {
        "tags": tags,
        "method": "minmax_0_1_train_only",
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "span": span.tolist(),
        "discrete": discrete.tolist(),
        "continuous_mask": (~discrete).tolist(),
        "clip_z": float(cfg["scaler"].get("clip_z", 10.0)),
        "residual_scale_method": cfg["scaler"].get("residual_scale", "std"),
    }
    return scaler


def apply_scaler(arr: np.ndarray, scaler: dict) -> np.ndarray:
    mins = np.asarray(scaler["min"], dtype=np.float32)
    span = np.asarray(scaler["span"], dtype=np.float32)
    out = (arr - mins) / span
    return np.clip(out, -0.5, 1.5).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scaler = fit_scaler(cfg)
    path = resolve(cfg, "processed_dir") / "scaler.json"
    write_json(path, scaler)
    n = len(scaler["tags"])
    print(f"[scaler] fit on train-normal; {n} tags "
          f"({sum(scaler['discrete'])} discrete) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
