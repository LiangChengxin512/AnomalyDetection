"""Fit the one-step-ahead surrogate used ONLY for shadow-policy evaluation.

Model: per-output ridge regression ``x_{t+1} = C x_t + b`` in scaled
space, fitted on train-normal rows. Discrete tags are clamped to {0,1}
predictions. The surrogate is an explicit approximation — its card
(surrogate_card.json) records the validation error and a usage gate;
policy outcomes computed under it are labelled "surrogate-only".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import read_json, write_json  # noqa: E402
from src.data.fit_scaler import apply_scaler  # noqa: E402


def fit(X: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """X: [N,D] consecutive scaled rows. Returns (coef [D,D], intercept [D])."""
    Xin, Xout = X[:-1], X[1:]
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xin, Xout)
    return model.coef_, model.intercept_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scaler = read_json(resolve(cfg, "processed_dir") / "scaler.json")
    tags = scaler["tags"]
    index = read_json(resolve(cfg, "processed_dir") / "index.json")

    rows = []
    max_rows = cfg["surrogate"]["fit_max_rows"]
    for rec in index["splits"]["train"]:
        df = pd.read_parquet(rec["parquet"], columns=tags)
        X = apply_scaler(df[tags].to_numpy(np.float32), scaler)
        rows.append(X)
    X = np.concatenate(rows, axis=0)
    if len(X) > max_rows:
        sel = np.linspace(0, len(X) - 1, max_rows).astype(int)
        X = X[sel]
    coef, intercept = fit(X, cfg["surrogate"]["alpha"])
    out = resolve(cfg, "artifacts_dir") / "surrogate_model.json"
    write_json(out, {"model": "ridge_one_step", "alpha": cfg["surrogate"]["alpha"],
                     "tags": tags, "coef": coef.tolist(),
                     "intercept": intercept.tolist(),
                     "fit_rows": int(len(X)),
                     "fit_split": "train",
                     "space": "minmax_0_1"})
    print(f"[surrogate] ridge one-step fit on {len(X)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
