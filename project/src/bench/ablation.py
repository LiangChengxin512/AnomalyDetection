"""Ablation and negative-control driver (plan table A1-A9).

Each ablation reuses existing artifacts where possible and marks clearly
which results are protocol-clean vs diagnostic. Output:
``artifacts/metrics_ablation.csv`` + a markdown summary appended to
reports/results.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import read_json, read_jsonl  # noqa: E402
from src.bench.metrics import (event_metrics, full_detection_report, mrr,  # noqa: E402
                               pa_f1, topk_hit)
from src.detect.calibrate import calibrate  # noqa: E402
from src.explain.rootcause import rank_event  # noqa: E402


def _load(cfg, name):
    return pd.read_parquet(resolve(cfg, "artifacts_dir") / f"scores_test_{name}.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--primary", default="MOCK",
                    help="score artifact tag for the primary model")
    ap.add_argument("--base", default=None,
                    help="score artifact tag for the base TranAD comparison (A1)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("schema_version") == "2.0":
        raise RuntimeError("This historical v1 CLI is disabled for protocol v2. Use scripts/reproduce_all.py; see docs/EXPERIMENTS.md.")
    art = resolve(cfg, "artifacts_dir")
    tags = read_json(resolve(cfg, "processed_dir") / "scaler.json")["tags"]

    df = _load(cfg, args.primary)
    y = df["label_attack"].to_numpy()
    s = df["score"].to_numpy()
    val_path = art / f"scores_val_{args.primary}.parquet"
    thr = calibrate(pd.read_parquet(val_path)["score"].to_numpy(),
                    cfg["threshold"]["method"], cfg["threshold"]["quantile"],
                    cfg["threshold"]["far_per_hour"])["threshold"] if val_path.exists() else None
    pred = (s > thr).astype(int) if thr is not None else (s > s.mean()).astype(int)
    rows = []

    # A6: PA vs no-PA comparison (diagnostic of metric bias)
    ev = event_metrics(pred, y, cfg["detection"]["min_event_gap_s"])
    rows.append({"ablation": "A6_pa_vs_noPA", "metric": "event_f1",
                 "value": ev["event_f1"], "note": "protocol metric"})
    rows.append({"ablation": "A6_pa_vs_noPA", "metric": "pa_f1",
                 "value": pa_f1(pred, y), "note": "biased; appendix only"})

    # A5: event-start shuffle negative control for detection
    rng = np.random.default_rng(0)
    y_sh = rng.permutation(y)
    ev_sh = event_metrics(pred, y_sh, cfg["detection"]["min_event_gap_s"])
    rows.append({"ablation": "A5_label_shuffle", "metric": "event_f1",
                 "value": ev_sh["event_f1"],
                 "note": "should be ~chance; else leakage suspected"})

    # A1: base TranAD comparison, if its score artifact exists
    if args.base:
        try:
            bdf = _load(cfg, args.base)
            bval = art / f"scores_val_{args.base}.parquet"
            bthr = calibrate(pd.read_parquet(bval)["score"].to_numpy(),
                             cfg["threshold"]["method"],
                             cfg["threshold"]["quantile"],
                             cfg["threshold"]["far_per_hour"])["threshold"]
            bev = event_metrics((bdf["score"] > bthr).astype(int),
                                bdf["label_attack"].to_numpy(),
                                cfg["detection"]["min_event_gap_s"])
            rows.append({"ablation": "A1_base_model", "metric": "event_f1",
                         "value": bev["event_f1"],
                         "note": f"base={args.base}"})
        except FileNotFoundError:
            print(f"[ablation] A1 skipped: scores_test_{args.base}.parquet missing")

    # A2/A3/A4/A9 on root-cause: recompute rankings under evidence removal
    rc_path = art / "rootcause.jsonl"
    if rc_path.exists():
        recs = read_jsonl(rc_path)
        events = {e["event_id"]: e for e in read_json(
            resolve(cfg, "processed_dir") / "events.json")["events"]}
        e_cols = [f"e_{t}" for t in tags]
        E = df[e_cols].to_numpy(np.float64)
        lab_ev = df["event_id"].to_numpy()
        graph = read_json(resolve(cfg, "process_graph"))
        variants = {
            "full": dict(cfg["rootcause"]["weights"]),
            "A2_residual_only": {"residual": 1.0, "attribution": 0.0,
                                 "onset": 0.0, "graph": 0.0, "stability": 0.0},
            "A3_no_attribution": {**cfg["rootcause"]["weights"],
                                  "attribution": 0.0},
            "A3_no_onset": {**cfg["rootcause"]["weights"], "onset": 0.0},
            "A4_no_graph": {**cfg["rootcause"]["weights"], "graph": 0.0},
        }
        # A4 rand graph built once
        rand_edges = [{"source": rng.choice(tags), "target": rng.choice(tags)}
                      for _ in graph["edges"]]
        for vname, w in variants.items():
            hits, mrrs = [], []
            cfg_v = dict(cfg)
            cfg_v["rootcause"] = dict(cfg["rootcause"], weights=w)
            for rec in recs:
                s_, e_ = int(rec["row_start"]), int(rec["row_end"])
                seg = lab_ev[s_:e_ + 1]
                seg = seg[seg != ""]
                if not len(seg):
                    continue
                gt = events.get(seg[0])
                if not gt:
                    continue
                gt_tags = set(gt["observable_tags"])
                cands = rank_event(E[s_:e_ + 1],
                                   np.zeros((e_ - s_ + 1, len(tags))),
                                   tags, graph["edges"], cfg_v)
                hits.append(topk_hit(cands, gt_tags, 3))
                mrrs.append(mrr(cands, gt_tags))
            rows.append({"ablation": vname, "metric": "top3_hit",
                         "value": float(np.mean(hits)) if hits else None,
                         "note": f"n={len(hits)} labelled events"})
            rows.append({"ablation": vname, "metric": "mrr",
                         "value": float(np.mean(mrrs)) if mrrs else None})
        # A4 random graph
        hits = []
        cfg_v = dict(cfg)
        for rec in recs:
            s_, e_ = int(rec["row_start"]), int(rec["row_end"])
            seg = lab_ev[s_:e_ + 1]; seg = seg[seg != ""]
            if not len(seg):
                continue
            gt = events.get(seg[0])
            if not gt:
                continue
            cands = rank_event(E[s_:e_ + 1],
                               np.zeros((e_ - s_ + 1, len(tags))),
                               tags, rand_edges, cfg)
            hits.append(topk_hit(cands, set(gt["observable_tags"]), 3))
        rows.append({"ablation": "A4_random_graph", "metric": "top3_hit",
                     "value": float(np.mean(hits)) if hits else None,
                     "note": "negative control"})

    out = art / "metrics_ablation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[ablation] wrote {out}")
    for r in rows:
        print(f"  {r['ablation']:24s} {r['metric']:12s} {r['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
