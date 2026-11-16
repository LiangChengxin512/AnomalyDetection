"""Multi-evidence root-cause ranking.

For each detected event, candidate tags are scored on five normalized
evidence channels and fused with fixed, documented weights (config
``rootcause.weights``):

1. ``residual``    — mean normalized channel residual over the event.
2. ``attribution`` — mean |input-gradient| (or occlusion / residual
                     fallback) over the event.
3. ``onset``       — how early the channel's residual first exceeds its
                     validation-normal quantile, relative to event start.
4. ``graph``       — process-graph support: anomalous downstream tags the
                     candidate can explain within ``graph_hops``.
5. ``stability``   — rank stability across noise-perturbed re-ranking.

The fused ranking is a ``root_cause_candidate`` — model evidence, NOT a
proven physical cause. Outputs ``artifacts/rootcause.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import (read_json, read_jsonl, write_json,  # noqa: E402
                                 write_jsonl)
from src.explain.attribution import residual_fallback_scores  # noqa: E402


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    m = v.max() if v.size else 0.0
    return v / m if m > 0 else v


def _graph_support(tag: str, anomalous: set, edges, hops: int) -> float:
    """Fraction of other anomalous tags reachable downstream within hops."""
    adj = {}
    for e in edges:
        adj.setdefault(e["source"], []).append(e["target"])
    reached = set()
    frontier = [tag]
    for _ in range(hops):
        nxt = []
        for f in frontier:
            for t in adj.get(f, []):
                if t not in reached:
                    reached.add(t)
                    nxt.append(t)
        frontier = nxt
    others = anomalous - {tag}
    if not others:
        return 0.0
    return len(reached & others) / len(others)


def _onset_ranks(e: np.ndarray, onset_q: float, start_i: int,
                 event_len: int) -> np.ndarray:
    """Earlier first-exceed => higher score. Channels that never exceed
    get 0."""
    D = e.shape[1]
    onset_q_val = np.quantile(e, onset_q, axis=0)
    first = np.full(D, np.inf)
    for j in range(D):
        hits = np.flatnonzero(e[:, j] > max(onset_q_val[j], 1e-9))
        if hits.size:
            first[j] = hits[0]
    if not np.isfinite(first).any():
        return np.zeros(D)
    # invert: earliest -> 1.0, latest -> small positive
    finite = np.isfinite(first)
    order = np.zeros(D)
    if finite.sum() == 1:
        order[finite] = 1.0
    else:
        span = first[finite].max() - first[finite].min() + 1e-9
        order[finite] = 1.0 - (first[finite] - first[finite].min()) / span * 0.9
    return order


def rank_event(e: np.ndarray, attr: np.ndarray, tags: list[str], edges,
               cfg, onset_q_val: np.ndarray | None = None,
               stability_attr: np.ndarray | None = None) -> list[dict]:
    """Fuse evidence and return ranked candidates for one event.

    ``e``: [T,D] normalized residual evidence. ``attr``: [T,D] or [D].
    """
    w = cfg["rootcause"]["weights"]
    resid = _norm(e.mean(axis=0))
    a = np.asarray(attr, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=0)
    attr_n = _norm(a)

    onset = _onset_ranks(e, cfg["rootcause"]["onset_quantile"], 0, e.shape[0])
    anomalous = set(np.flatnonzero(resid > np.quantile(resid, 0.9))
                    if (resid > 0).any() else [])
    anomalous = {int(i) for i in anomalous}
    graph = np.array([_graph_support(tags[j], {tags[i] for i in anomalous},
                                     edges, cfg["rootcause"]["graph_hops"])
                      for j in range(len(tags))])
    graph = _norm(graph)

    if stability_attr is not None:
        stab = _norm(stability_attr)
        stab = stab * 0 + np.corrcoef(resid, stab)[0, 1] if resid.std() and stab.std() else stab * 0 + 0.5
    else:
        stab = np.full(len(tags), 0.5)  # unknown when no perturbation run

    fused = (w["residual"] * resid + w["attribution"] * attr_n
             + w["onset"] * onset + w["graph"] * graph + w["stability"] * stab)
    order = np.argsort(-fused)
    cands = []
    for r, j in enumerate(order):
        cands.append({"tag": tags[j], "rank": r + 1,
                      "score": float(fused[j]),
                      "evidence": {"residual": float(resid[j]),
                                   "attribution": float(attr_n[j]),
                                   "onset": float(onset[j]),
                                   "graph": float(graph[j]),
                                   "stability": float(stab[j])}})
    return cands


def detect_events(scores: np.ndarray, threshold: float, min_gap: int):
    from src.common.events import binary_to_ranges
    return binary_to_ranges(scores > threshold, min_gap=min_gap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scores", required=True, help="test scores parquet")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--attribution", default=None,
                    help="optional attribution parquet (attr_<tag> cols)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tags = read_json(resolve(cfg, "processed_dir") / "scaler.json")["tags"]

    df = pd.read_parquet(args.scores)
    thr_path = args.thresholds or str(resolve(cfg, "artifacts_dir") / "thresholds.json")
    thr = read_json(thr_path)["threshold"]

    attr = None
    if args.attribution:
        adf = pd.read_parquet(args.attribution)
        attr = adf[[f"attr_{t}" for t in tags]].to_numpy()

    graph = read_json(resolve(cfg, "process_graph"))
    e_cols = [f"e_{t}" for t in tags]
    E = df[e_cols].to_numpy(np.float64)
    S = df["score"].to_numpy()

    records = []
    events = detect_events(S, thr, cfg["detection"]["min_event_gap_s"])
    label_events = df["event_id"].to_numpy()
    for k, (s, e) in enumerate(events):
        e_seg = E[s:e + 1]
        a_seg = attr[s:e + 1] if attr is not None else residual_fallback_scores(e_seg)
        cands = rank_event(e_seg, a_seg, tags, graph["edges"], cfg)
        top = cands[0]
        margin = (cands[0]["score"] - cands[1]["score"]) if len(cands) > 1 else 1.0
        gt_eid = label_events[s] if s < len(label_events) else ""
        rec = {
            "schema_version": "1.0",
            "event_id": f"det-{k:04d}",
            "row_start": int(s), "row_end": int(e),
            "start": str(df["timestamp"].iloc[s]),
            "end": str(df["timestamp"].iloc[e]),
            "gt_event_id": gt_eid or None,
            "root_cause_candidates": cands[:10],
            "confidence": float(top["score"]),
            "abstain": bool(margin < cfg["rootcause"]["abstain_margin"]),
            "abstain_reason": "low_margin" if margin < cfg["rootcause"]["abstain_margin"] else None,
            "graph_version": graph["graph_version"],
            "model_version": str(df["model_version"].iloc[0]) if "model_version" in df else "unknown",
        }
        records.append(rec)

    out = args.out or str(resolve(cfg, "artifacts_dir") / "rootcause.jsonl")
    write_jsonl(out, records)
    print(f"[rootcause] {len(records)} detected events -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
