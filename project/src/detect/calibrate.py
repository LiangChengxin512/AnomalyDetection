"""Threshold calibration on validation-normal scores only.

The chosen threshold is locked into ``thresholds.json`` together with its
provenance (method, validation file list, quantile/FAR target). It must
never be re-fit on test data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import write_json  # noqa: E402


def calibrate(val_scores: np.ndarray, method: str, quantile: float = 0.999,
              far_per_hour: float = 0.5) -> dict:
    val_scores = np.asarray(val_scores, dtype=np.float64)
    val_scores = val_scores[np.isfinite(val_scores)]
    if method == "quantile":
        thr = float(np.quantile(val_scores, quantile))
        achieved_far = float((val_scores > thr).mean())
    elif method == "far_per_hour":
        # 1 Hz scores: FAR per hour = rate of exceedances * 3600
        q = 1.0 - far_per_hour / 3600.0
        thr = float(np.quantile(val_scores, max(0.0, min(1.0, q))))
        achieved_far = float((val_scores > thr).mean() * 3600.0)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return {"threshold": thr, "method": method, "quantile": quantile,
            "far_per_hour_target": far_per_hour,
            "n_val_scores": int(len(val_scores)),
            "achieved_val_exceed_rate": achieved_far}


def load_thresholds(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scores", required=True,
                    help="val scores parquet (from replay.py or mock_scores.py)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    import pandas as pd
    df = pd.read_parquet(args.scores, columns=["score"])
    res = calibrate(df["score"].to_numpy(),
                    method=cfg["threshold"]["method"],
                    quantile=cfg["threshold"]["quantile"],
                    far_per_hour=cfg["threshold"]["far_per_hour"])
    res["scores_file"] = str(args.scores)
    res["val_files"] = cfg["data"]["val_files"]
    out = resolve(cfg, "artifacts_dir") / "thresholds.json"
    write_json(out, res)
    print(f"[calibrate] threshold={res['threshold']:.6f} "
          f"({res['method']}, n={res['n_val_scores']}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
