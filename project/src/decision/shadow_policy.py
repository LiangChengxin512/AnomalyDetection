"""Constrained shadow policy with explicit abstain.

For each detected event (rootcause.jsonl), enumerate legal catalog
actions and pick the one minimizing a short-horizon objective::

    sum_k  lr*Risk(x_{t+k}) + lu*|u_k - u_{k-1}|^2 + lc*Cost(u_k)

evaluated under the surrogate model when available (otherwise an
envelope-only risk estimate). The policy abstains (``selected_action:
null, mode="abstain"``) when:

* root-cause confidence < ``policy.abstain.rc_confidence_min``
* surrogate uncertainty is above ``surrogate_uncertainty_max``
* the best action does not beat the no-action risk by ``risk_gain_min``

Outputs ``artifacts/advice.jsonl`` — one record per event with constraint
checks, expected risk and rationale. Advisory only: never writes to a PLC.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import read_json, read_jsonl, write_jsonl  # noqa: E402
from src.decision.action_catalog import enumerate_legal_actions, load_catalog  # noqa: E402


UNCERT_ORDER = {"low": 0, "medium": 1, "high": 2}


def _risk(x_row: np.ndarray, z_row: np.ndarray, uncert: float, w) -> float:
    """Scalar risk: out-of-envelope fraction + mean |z| + uncertainty."""
    out_env = float(np.mean(x_row > 1.0) + np.mean(x_row < 0.0))
    return w["w_range"] * out_env + w["w_z"] * float(np.mean(np.abs(z_row))) \
        + w["w_uncert"] * uncert


def _apply_action(x: np.ndarray, action: dict, tags: list[str],
                  scale_span: np.ndarray, direction_hint: int = 0) -> np.ndarray:
    """Apply an advisory action's nominal effect in scaled space."""
    y = x.copy()
    d = action.get("direction", 0)
    if action["kind"] == "adjust" and d != 0:
        for t in action.get("affects", []):
            if t in tags:
                j = tags.index(t)
                delta = min(action.get("max_delta", 0.0), action["max_delta"]) / max(scale_span[j], 1e-9)
                y[j] = np.clip(y[j] + direction_hint * delta, -0.5, 1.5)
    # 'hold'/'inspect'/'mode' have no nominal plant effect in the surrogate.
    return y


def evaluate_event_policy(rec: dict, ctx: dict) -> dict:
    """Decide one event. ``ctx`` holds catalog, surrogate, config, tags."""
    cfg, cat, tags = ctx["cfg"], ctx["catalog"], ctx["tags"]
    pol = cfg["policy"]
    cand_tags = [c["tag"] for c in rec.get("root_cause_candidates", [])[:5]]
    legal = enumerate_legal_actions(cat, cand_tags)
    rc_conf = float(rec.get("confidence", 0.0))
    if rec.get("abstain"):
        return _advice(rec, legal, None, "abstain", "rootcause_abstained", ctx)
    if rc_conf < pol["abstain"]["rc_confidence_min"]:
        return _advice(rec, legal, None, "abstain",
                       f"rc_confidence<{pol['abstain']['rc_confidence_min']}", ctx)

    surr = ctx.get("surrogate")
    uncert = "high" if surr is None else surr.get("uncertainty_level", "medium")
    if UNCERT_ORDER[uncert] > UNCERT_ORDER[pol["abstain"]["surrogate_uncertainty_max"]]:
        return _advice(rec, legal, None, "abstain",
                       f"surrogate_uncertainty={uncert}", ctx)

    # current state: mean scaled row over the event (from scores parquet)
    x = ctx["x_event"]  # [D] scaled features at event
    z = ctx["z_event"]  # [D] normalized residuals at event
    span = ctx["scale_span"]
    w = pol["risk"]

    def rollout(a):
        """1-step surrogate or envelope-only risk projection."""
        risks = []
        xc = x.copy()
        for k in range(pol["horizon"]):
            xc = _apply_action(xc, a, tags, span, direction_hint=-1)
            if surr is not None:
                xc = surr["predict"](xc)
            u_val = UNCERT_ORDER[uncert] / 2.0
            risks.append(_risk(xc, z, u_val, w))
        return float(np.sum(risks))

    no_action = {"id": "__no_action__", "kind": "hold", "cost": 0.0,
                 "affects": [], "direction": 0}
    risk_no = rollout(no_action)
    scored = []
    for a in legal:
        r = rollout(a)
        obj = pol["lambda_risk"] * r + pol["lambda_cost"] * float(a.get("cost", 0.1))
        scored.append((obj, r, a))
    scored.sort(key=lambda t: t[0])
    best_obj, best_risk, best = scored[0] if scored else (np.inf, risk_no, None)

    if best is None or (risk_no - best_risk) < pol["abstain"]["risk_gain_min"]:
        return _advice(rec, legal, None, "abstain",
                       "no_risk_gain_vs_no_action", ctx,
                       expected_risk=risk_no)
    return _advice(rec, legal, best["id"], "advisory",
                   "min-risk legal action", ctx, expected_risk=best_risk,
                   no_action_risk=risk_no)


def _advice(rec, legal, selected, mode, rationale, ctx,
            expected_risk=None, no_action_risk=None):
    checks = {
        "within_catalog": selected is None or any(a["id"] == selected for a in legal),
        "advisory_only": True,
        "plc_write": False,
        "surrogate_uncertainty": "high" if ctx.get("surrogate") is None
        else ctx["surrogate"].get("uncertainty_level", "medium"),
    }
    return {
        "schema_version": "1.0",
        "event_id": rec["event_id"],
        "gt_event_id": rec.get("gt_event_id"),
        "root_cause_candidates": rec.get("root_cause_candidates", [])[:3],
        "action_set": [a["id"] for a in legal],
        "selected_action": selected,
        "mode": mode,
        "expected_risk_proxy": expected_risk,
        "no_action_risk_proxy": no_action_risk,
        "constraint_checks": checks,
        "rationale": rationale,
        "note": "Advisory only; requires human confirmation.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--rootcause", default=None)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--surrogate", default=None,
                    help="surrogate_card.json (optional; absent => envelope-only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("schema_version") == "2.0":
        raise RuntimeError("This historical v1 CLI is disabled for protocol v2. Use scripts/reproduce_all.py; see docs/EXPERIMENTS.md.")
    cat = load_catalog(str(resolve(cfg, "action_catalog")))
    scaler = read_json(resolve(cfg, "processed_dir") / "scaler.json")
    tags = scaler["tags"]

    rc_path = args.rootcause or str(resolve(cfg, "artifacts_dir") / "rootcause.jsonl")
    records = read_jsonl(rc_path)
    scores = pd.read_parquet(args.scores)
    e_cols = [f"e_{t}" for t in tags]

    # scaled feature matrix for the scored test split (state at event rows)
    from src.data.fit_scaler import apply_scaler
    index = read_json(resolve(cfg, "processed_dir") / "index.json")
    feats = []
    for rec in index["splits"]["test"]:
        df = pd.read_parquet(rec["parquet"], columns=tags)
        feats.append(apply_scaler(df[tags].to_numpy(np.float32), scaler))
    X = np.concatenate(feats, axis=0)
    end_rows = scores["end_row"].to_numpy() if "end_row" in scores else np.arange(len(scores))

    surrogate = None
    if args.surrogate:
        surrogate = read_json(args.surrogate)
        # turn stored linear surrogate into a predict() closure
        coef = np.asarray(surrogate["coef"]); intercept = np.asarray(surrogate["intercept"])
        surrogate["predict"] = lambda x, C=coef, b=intercept: np.clip(C @ x + b, -0.5, 1.5)

    span = np.asarray(scaler["span"], dtype=np.float64)
    rows = []
    for rec in records:
        s, e = int(rec["row_start"]), int(rec["row_end"])
        e_seg = scores[e_cols].to_numpy()[s:e + 1]
        x_ev = X[int(end_rows[e])] if int(end_rows[e]) < len(X) else np.full(len(tags), 0.5)
        ctx = {"cfg": cfg, "catalog": cat, "tags": tags, "surrogate": surrogate,
               "x_event": np.asarray(x_ev, dtype=np.float64),
               "z_event": e_seg.mean(axis=0), "scale_span": span}
        rows.append(evaluate_event_policy(rec, ctx))

    out = args.out or str(resolve(cfg, "artifacts_dir") / "advice.jsonl")
    write_jsonl(out, rows)
    n_abs = sum(1 for r in rows if r["mode"] == "abstain")
    print(f"[policy] {len(rows)} events -> {out} (abstain={n_abs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
