"""Event-level bootstrap confidence intervals for detection + RCA metrics.

Resamples labelled events (not single points) so the interval reflects
per-event variability. Output: ``artifacts/metrics_bootstrap.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.events import binary_to_ranges  # noqa: E402
from src.common.io_utils import read_json, read_jsonl  # noqa: E402


def _event_event_recall(pred, label_ranges):
    if not label_ranges:
        return 0.0
    pred_runs = binary_to_ranges(pred)
    det = sum(1 for s, e in label_ranges
              if any(not (e < ps or s > pe) for ps, pe in pred_runs))
    return det / len(label_ranges)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scores", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("schema_version") == "2.0":
        raise RuntimeError("This historical v1 CLI is disabled for protocol v2. Use scripts/reproduce_all.py; see docs/EXPERIMENTS.md.")
    B = cfg["eval"]["bootstrap_samples"]
    alpha = cfg["eval"]["bootstrap_alpha"]
    rng = np.random.default_rng(0)

    df = pd.read_parquet(args.scores)
    s = df["score"].to_numpy()
    y = df["label_attack"].to_numpy()
    val_path = resolve(cfg, "artifacts_dir") / "thresholds.json"
    thr = read_json(val_path)["threshold"]
    pred = (s > thr).astype(int)

    label_runs = binary_to_ranges(y)
    if not label_runs:
        print("[bootstrap] no labelled events; nothing to do")
        return 0
    recalls = []
    delays = []
    for _ in range(B):
        samp = rng.choice(len(label_runs), len(label_runs), replace=True)
        # event-level recall across resampled events
        det, dly = 0, []
        for i in samp:
            s_, e_ = label_runs[i]
            hits = np.flatnonzero(pred[s_:e_ + 1])
            if hits.size:
                det += 1
                dly.append(int(hits[0]))
        recalls.append(det / len(label_runs))
        delays.append(np.mean(dly) if dly else np.nan)
    lo, hi = np.nanpercentile(recalls, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    dlo, dhi = np.nanpercentile(delays, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    rows = [
        {"metric": "event_recall_boot_ci", "mean": float(np.nanmean(recalls)),
         "ci_lo": float(lo), "ci_hi": float(hi), "n_events": len(label_runs)},
        {"metric": "detection_delay_boot_ci", "mean": float(np.nanmean(delays)),
         "ci_lo": float(dlo), "ci_hi": float(dhi), "n_events": len(label_runs)},
    ]
    out = resolve(cfg, "artifacts_dir") / "metrics_bootstrap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[bootstrap] wrote {out}: recall CI=[{lo:.3f},{hi:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
