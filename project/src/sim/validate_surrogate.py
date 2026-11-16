"""Validate the surrogate on the held-out normal validation split.

Reports per-tag RMSE and the share of validation residuals inside the
+-2*s training-residual band, then writes ``surrogate_card.json`` with a
usage gate and an ``uncertainty_level`` consumed by the shadow policy's
abstain rule. All results are surrogate-only approximations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import read_json, write_json  # noqa: E402
from src.data.fit_scaler import apply_scaler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("schema_version") == "2.0":
        raise RuntimeError("This historical v1 CLI is disabled for protocol v2. Use scripts/reproduce_all.py; see docs/EXPERIMENTS.md.")
    scaler = read_json(resolve(cfg, "processed_dir") / "scaler.json")
    tags = scaler["tags"]
    index = read_json(resolve(cfg, "processed_dir") / "index.json")
    model = read_json(resolve(cfg, "artifacts_dir") / "surrogate_model.json")
    C = np.asarray(model["coef"]); b = np.asarray(model["intercept"])

    def predict(x):
        return np.clip(x @ C.T + b, -0.5, 1.5)

    # train residual band for coverage
    rng = np.random.default_rng(0)
    train_resid = []
    for rec in index["splits"]["train"][:1]:
        df = pd.read_parquet(rec["parquet"], columns=tags)
        X = apply_scaler(df[tags].to_numpy(np.float32), scaler)
        sub = X[:: max(1, len(X) // 50000)]
        pred = predict(sub[:-1])
        train_resid.append(sub[1:] - pred)
    train_resid = np.concatenate(train_resid, axis=0)
    band = np.std(train_resid, axis=0) * cfg["surrogate"]["val_coverage_z"] + 1e-8

    per_tag, cover = [], []
    for rec in index["splits"]["val"]:
        df = pd.read_parquet(rec["parquet"], columns=tags)
        X = apply_scaler(df[tags].to_numpy(np.float32), scaler)
        pred = predict(X[:-1])
        resid = X[1:] - pred
        rmse = np.sqrt((resid ** 2).mean(axis=0))
        tag_std = X.std(axis=0) + 1e-9
        in_band = (np.abs(resid) < band).mean(axis=0)
        per_tag.append((rmse, tag_std, in_band))
    rmse = np.mean([p[0] for p in per_tag], axis=0)
    tag_std = np.mean([p[1] for p in per_tag], axis=0)
    cov = np.mean([p[2] for p in per_tag], axis=0)

    ratio = rmse / tag_std
    gate = cfg["surrogate"]["gate"]
    usable = bool(np.median(ratio) <= gate["max_rmse_ratio"]
                  and np.median(cov) >= gate["min_coverage"])
    med_ratio = float(np.median(ratio))
    if med_ratio <= 0.8:
        uncert = "low"
    elif med_ratio <= gate["max_rmse_ratio"]:
        uncert = "medium"
    else:
        uncert = "high"

    card = {
        "model": model["model"],
        "val_split": [r["file"] for r in index["splits"]["val"]],
        "median_rmse_over_std": med_ratio,
        "median_coverage_pm2s": float(np.median(cov)),
        "uncertainty_level": uncert,
        "usable_for_policy_eval": usable,
        "gate": gate,
        "coef": model["coef"], "intercept": model["intercept"],
        "tags": tags,
        "note": ("Surrogate-only: linear one-step approximation fitted on "
                 "normal data. Attack/action effects are NOT validated "
                 "against the real plant."),
        "per_tag": [{"tag": t, "rmse": float(r), "std": float(s),
                     "coverage": float(c)} for t, r, s, c in
                    zip(tags, rmse, tag_std, cov)],
    }
    out = resolve(cfg, "artifacts_dir") / "surrogate_card.json"
    write_json(out, card)
    print(f"[surrogate] val: median RMSE/std={med_ratio:.3f} "
          f"coverage={np.median(cov):.3f} usable={usable} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
