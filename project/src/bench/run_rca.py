"""Root-cause benchmark: Top-k hit / MRR against the attack-target table.

Compares ranking strategies per detected event:

* ``full``       — the fused multi-evidence ranking (rootcause.jsonl)
* ``residual``   — mean channel residual only (strong baseline)
* ``attribution``— attribution channel only
* ``graph_rand`` — full fusion with edge-shuffled graph (negative control)
* ``target_perm``— fused ranking scored against permuted attack targets
                   (label-leakage negative control)

An event is scored when a detected event overlaps a labelled attack. GT
comes from ``observable_tags`` in the attack table. Output:
``artifacts/metrics_rca.csv``.
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
from src.bench.metrics import mrr, topk_hit  # noqa: E402
from src.explain.rootcause import rank_event  # noqa: E402


def _strategies(cfg, tags, edges, rng):
    def full(e, attr):
        return rank_event(e, attr, tags, edges, cfg)

    def residual(e, attr):
        m = e.mean(axis=0)
        order = np.argsort(-m)
        return [{"tag": t, "rank": i + 1} for i, t in
                enumerate(np.asarray(tags)[order])]

    def attr_only(e, attr):
        a = attr.mean(axis=0) if attr.ndim == 2 else attr
        order = np.argsort(-a)
        return [{"tag": t, "rank": i + 1} for i, t in
                enumerate(np.asarray(tags)[order])]

    def graph_rand(e, attr):
        # negative control: keep node set, shuffle edges
        rnd = [{"source": rng.choice(tags), "target": rng.choice(tags)}
               for _ in edges]
        return rank_event(e, attr, tags, rnd, cfg)
    return {"full": full, "residual": residual,
            "attribution": attr_only, "graph_rand": graph_rand}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--rootcause", default=None)
    ap.add_argument("--topk", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("schema_version") == "2.0":
        raise RuntimeError("This historical v1 CLI is disabled for protocol v2. Use scripts/reproduce_all.py; see docs/EXPERIMENTS.md.")
    topk_list = args.topk or cfg["eval"]["rca_topk"]
    tags = read_json(resolve(cfg, "processed_dir") / "scaler.json")["tags"]
    graph = read_json(resolve(cfg, "process_graph"))
    events = {e["event_id"]: e for e in
              read_json(resolve(cfg, "processed_dir") / "events.json")["events"]}

    df = pd.read_parquet(args.scores)
    e_cols = [f"e_{t}" for t in tags]
    E = df[e_cols].to_numpy(np.float64)
    recs = read_jsonl(args.rootcause or
                      str(resolve(cfg, "artifacts_dir") / "rootcause.jsonl"))
    rng = np.random.default_rng(0)
    strats = _strategies(cfg, tags, graph["edges"], rng)

    # map each detected event to a labelled GT event via overlap
    rows = []
    lab_ev = df["event_id"].to_numpy()
    for rec in recs:
        s, e = int(rec["row_start"]), int(rec["row_end"])
        seg_eids = lab_ev[s:e + 1]
        seg_eids = seg_eids[seg_eids != ""]
        gt_eid = seg_eids[0] if len(seg_eids) else None
        gt = events.get(gt_eid) if gt_eid else None
        gt_tags = set(gt["observable_tags"]) if gt else set()
        e_seg = E[s:e + 1]
        attr = np.asarray([c["evidence"]["attribution"] for c in
                           rec["root_cause_candidates"]])
        # rebuild attr vector aligned to tag order (use stored evidence)
        attr_vec = np.zeros(len(tags))
        for c in rec["root_cause_candidates"]:
            if c["tag"] in tags:
                attr_vec[tags.index(c["tag"])] = c["evidence"]["attribution"]
        for sname, fn in strats.items():
            cands = fn(e_seg, attr_vec[None, :].repeat(len(e_seg), axis=0))
            row = {"event_id": rec["event_id"], "gt_event_id": gt_eid,
                   "attack_id": gt["attack_id"] if gt else None,
                   "strategy": sname, "n_gt_tags": len(gt_tags)}
            for k in topk_list:
                row[f"top{k}_hit"] = (topk_hit(cands, gt_tags, k)
                                      if gt_tags else None)
            row["mrr"] = mrr(cands, gt_tags) if gt_tags else None
            rows.append(row)
        # negative control: permuted GT tags
        if gt_tags:
            all_tags = set(tags)
            perm = set(rng.choice(sorted(all_tags), size=len(gt_tags),
                                  replace=False))
            cands = rec["root_cause_candidates"]
            row = {"event_id": rec["event_id"], "gt_event_id": gt_eid,
                   "attack_id": gt["attack_id"], "strategy": "target_perm",
                   "n_gt_tags": len(gt_tags)}
            for k in topk_list:
                row[f"top{k}_hit"] = topk_hit(cands, perm, k)
            row["mrr"] = mrr(cands, perm)
            rows.append(row)

    out = resolve(cfg, "artifacts_dir") / "metrics_rca.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    df_out = pd.DataFrame(rows)
    for sname in df_out["strategy"].unique():
        sub = df_out[df_out["strategy"] == sname]
        vals = {k: np.nanmean(sub[f"top{k}_hit"]) for k in topk_list}
        print(f"[rca] {sname}: " + " ".join(f"top{k}={v:.3f}" for k, v in vals.items())
              + f" mrr={np.nanmean(sub['mrr']):.3f}")
    print(f"[rca] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
