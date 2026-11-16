"""Closed-loop replay of decision policies inside the surrogate.

Compares three policies over each detected event:

* ``no_action``   — do nothing (baseline).
* ``fixed``       — always suggest the first legal conservative action
                    (``hold_current_setpoints``).
* ``adaptive``    — the shadow policy's chosen action from advice.jsonl.

Risk is the surrogate-rollout risk already defined in the policy module.
All numbers are labelled surrogate-only: they are NOT physical plant
evidence. Output: ``artifacts/policy_replay.parquet``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import read_json, read_jsonl, write_parquet  # noqa: E402
from src.data.fit_scaler import apply_scaler  # noqa: E402
from src.decision.shadow_policy import _risk  # noqa: E402


def _rollout(x0, action_delta, predict, horizon, z, w, uncert_val):
    x = x0.copy()
    risks = []
    for k in range(horizon):
        if k == 0 and action_delta is not None:
            x = np.clip(x + action_delta, -0.5, 1.5)
        x = predict(x)
        risks.append(_risk(x, z, uncert_val, w))
    return float(np.sum(risks)), x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--advice", default=None)
    ap.add_argument("--scores", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("schema_version") == "2.0":
        raise RuntimeError("This historical v1 CLI is disabled for protocol v2. Use scripts/reproduce_all.py; see docs/EXPERIMENTS.md.")
    scaler = read_json(resolve(cfg, "processed_dir") / "scaler.json")
    tags = scaler["tags"]
    span = np.asarray(scaler["span"], dtype=np.float64)

    card = read_json(resolve(cfg, "artifacts_dir") / "surrogate_card.json")
    C = np.asarray(card["coef"]); b = np.asarray(card["intercept"])
    predict = lambda x: np.clip(x @ C.T + b, -0.5, 1.5)
    uncert = {"low": 0.25, "medium": 0.5, "high": 1.0}[card["uncertainty_level"]]
    w = cfg["policy"]["risk"]
    H = cfg["policy"]["horizon"]

    advice = {r["event_id"]: r for r in read_jsonl(
        args.advice or str(resolve(cfg, "artifacts_dir") / "advice.jsonl"))}
    from src.decision.action_catalog import load_catalog
    cat = load_catalog(str(resolve(cfg, "action_catalog")))
    cat_by_id = {a["id"]: a for a in cat["actions"]}

    scores = pd.read_parquet(args.scores)
    e_cols = [f"e_{t}" for t in tags]
    index = read_json(resolve(cfg, "processed_dir") / "index.json")
    feats = [apply_scaler(pd.read_parquet(r["parquet"], columns=tags)[tags]
                          .to_numpy(np.float32), scaler)
             for r in index["splits"]["test"]]
    X = np.concatenate(feats, axis=0)
    end_rows = scores["end_row"].to_numpy()

    rc = {r["event_id"]: r for r in read_jsonl(
        resolve(cfg, "artifacts_dir") / "rootcause.jsonl")}

    rows = []
    for eid, adv in advice.items():
        r = rc.get(eid)
        if r is None:
            continue
        er = int(end_rows[int(r["row_end"])])
        x0 = X[er] if er < len(X) else np.full(len(tags), 0.5)
        z = scores[e_cols].to_numpy()[int(r["row_start"]):int(r["row_end"]) + 1].mean(axis=0)

        def delta_for(action_id):
            a = cat_by_id.get(action_id or "")
            if not a or a["kind"] != "adjust" or not a.get("direction"):
                return None
            d = np.zeros(len(tags))
            for t in a["affects"]:
                if t in tags:
                    j = tags.index(t)
                    d[j] = a["direction"] * min(a["max_delta"], a["max_delta"]) / max(span[j], 1e-9)
            return d

        fixed_id = "hold_current_setpoints"
        risk_no, _ = _rollout(x0, None, predict, H, z, w, uncert)
        risk_fix, _ = _rollout(x0, delta_for(fixed_id), predict, H, z, w, uncert)
        chosen = adv.get("selected_action")
        risk_ad, _ = _rollout(x0, delta_for(chosen), predict, H, z, w, uncert)
        rows.append({
            "event_id": eid, "gt_event_id": adv.get("gt_event_id"),
            "selected_action": chosen, "mode": adv.get("mode"),
            "surrogate_risk_no_action": risk_no,
            "surrogate_risk_fixed": risk_fix,
            "surrogate_risk_adaptive": risk_ad,
            "surrogate_gain_vs_no_action": risk_no - risk_ad,
            "surrogate_gain_vs_fixed": risk_fix - risk_ad,
            "uncertainty_level": card["uncertainty_level"],
            "surrogate_only": True,
        })
    df = pd.DataFrame(rows)
    out = resolve(cfg, "artifacts_dir") / "policy_replay.parquet"
    write_parquet(df, out)
    print(f"[closed_loop] {len(df)} events replayed in surrogate -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
