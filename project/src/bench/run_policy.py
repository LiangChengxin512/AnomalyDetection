"""Policy benchmark: legality + surrogate-only outcomes vs baselines.

Reads ``advice.jsonl`` and ``policy_replay.parquet`` and reports action
legality rate, abstain coverage, suggested-action mix, surrogate risk
reduction vs no-action and vs fixed-conservative policy. Every metric is
labelled surrogate-only where applicable.

Output: ``artifacts/metrics_policy.csv``.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = resolve(cfg, "artifacts_dir")

    advice = read_jsonl(art / "advice.jsonl")
    replay_path = art / "policy_replay.parquet"
    replay = pd.read_parquet(replay_path) if replay_path.exists() else pd.DataFrame()

    n = len(advice)
    legal = sum(1 for a in advice if a["constraint_checks"].get("within_catalog", True))
    abstain = sum(1 for a in advice if a["mode"] == "abstain")
    advisory = sum(1 for a in advice if a["mode"] == "advisory")
    rows = [{
        "metric": "n_events", "value": n, "scope": "replay"},
        {"metric": "legal_action_rate", "value": legal / max(1, n), "scope": "replay"},
        {"metric": "abstain_rate", "value": abstain / max(1, n), "scope": "replay"},
        {"metric": "advisory_rate", "value": advisory / max(1, n), "scope": "replay"},
    ]
    if len(replay):
        for col, label in [
            ("surrogate_gain_vs_no_action", "adaptive_vs_no_action"),
            ("surrogate_gain_vs_fixed", "adaptive_vs_fixed")]:
            v = replay[col].dropna()
            rows.append({"metric": f"mean_{label}", "value": float(v.mean()),
                         "scope": "surrogate_only"})
            rows.append({"metric": f"frac_events_improved_{label}",
                         "value": float((v > 0).mean()), "scope": "surrogate_only"})
    out = art / "metrics_policy.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[policy-bench] wrote {out}")
    for r in rows:
        print(f"  {r['metric']}: {r['value']} ({r['scope']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
