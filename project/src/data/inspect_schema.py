"""Inspect raw HAI CSV headers and build config/tag_schema.yaml.

The schema records, for every tag: process, signal kind, unit (only where
the technical manual documents one — everything else is "unknown"),
observed value range, and whether the tag is treated as discrete.

The script FAILS on version/column mismatches instead of silently
dropping columns, per the project plan.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, project_path, resolve  # noqa: E402
from src.common.io_utils import write_json  # noqa: E402

# Units documented in the HAI technical manual / testbed descriptions.
# Tags not listed here are left unit="unknown" rather than guessed.
DOCUMENTED_UNITS = {
    "P2_VIBTR01": "um", "P2_VIBTR02": "um", "P2_VIBTR03": "um", "P2_VIBTR04": "um",
    "P2_SIT01": "rpm",
    "P4_HT_FD": "pu", "P4_ST_FD": "pu",
    "P4_HT_PS": "MW", "P4_ST_PS": "MW", "P4_ST_PO": "MW", "P4_HT_PO": "MW",
    "P4_LD": "MW", "P4_ST_LD": "MW",
    "P4_ST_PT01": "bar", "P4_ST_TT01": "degC",
}

# Name-suffix conventions for signal kind (documented HAI convention).
def classify_tag(tag: str, n_unique: int, uniques: np.ndarray) -> str:
    if tag.startswith("x") and tag.endswith("_OUT"):
        return "controller_output"
    if tag.endswith("D") and not tag.endswith(("AD", "BD", "LD", "GO", "FD")):
        return "command"
    if tag.endswith("Z"):
        return "feedback"
    if tag.endswith(("SP", "SD")) or "SETPOINT" in tag or "ASSIGN" in tag:
        return "setpoint"
    if n_unique <= 4 and set(np.unique(uniques)) <= {0, 1, 2, 3}:
        return "status"
    return "measurement"


def is_discrete(tag: str, n_unique: int, uniques: np.ndarray, kind: str) -> bool:
    if kind in {"status"}:
        return True
    if n_unique <= 3 and set(np.unique(uniques[:8])) <= {0.0, 1.0, 2.0}:
        return True
    return False


def process_of(tag: str) -> str:
    if tag.startswith("P1_"):
        return "P1_boiler"
    if tag.startswith("P2_"):
        return "P2_turbine"
    if tag.startswith("P3_"):
        return "P3_water"
    if tag.startswith("P4_"):
        return "P4_hil"
    if tag.startswith("x"):
        return "DCS_internal"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample-rows", type=int, default=50000,
                    help="rows sampled per file for range/unique stats")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir = resolve(cfg, "raw_dir")
    files = (cfg["data"]["train_files"] + cfg["data"]["val_files"]
             + cfg["data"]["test_files"])

    headers = {}
    stats = {}
    audit = ["# HAI 23.05 schema audit", ""]
    for name in files:
        path = raw_dir / name
        if not path.exists():
            print(f"[schema] MISSING {path}", file=sys.stderr)
            return 2
        head = pd.read_csv(path, nrows=0)
        headers[name] = list(head.columns)
        df = pd.read_csv(path, nrows=args.sample_rows)
        stats[name] = df
        print(f"[schema] {name}: {len(head.columns)} columns")

    ref = headers[files[0]]
    mismatches = {n: h for n, h in headers.items() if h != ref}
    audit.append(f"- reference file: `{files[0]}` ({len(ref)} columns)")
    for n, h in headers.items():
        audit.append(f"- {n}: {len(h)} columns"
                     + ("  **MISMATCH**" if n in mismatches else ""))
    if mismatches:
        audit.append("\n## Column mismatches (fatal)")
        for n, h in mismatches.items():
            extra = sorted(set(h) - set(ref))
            missing = sorted(set(ref) - set(h))
            audit.append(f"- {n}: extra={extra} missing={missing}")
        _write_audit(cfg, audit)
        print("[schema] FATAL: column mismatch between files; see reports/schema_audit.md",
              file=sys.stderr)
        return 3

    ts_col = cfg["data"]["timestamp_col"]
    tags = [c for c in ref if c != ts_col]
    audit.append(f"\n- tag count (excluding `{ts_col}`): {len(tags)}")

    schema = {"schema_version": "1.0", "dataset_version": "HAI-23.05",
              "source": "hai_dataset_technical_details.pdf v4.0 + observed data",
              "timestamp_col": ts_col, "tags": []}
    rows = []
    for tag in tags:
        cols = [stats[n][tag].dropna().to_numpy() for n in stats]
        allv = np.concatenate(cols)
        uniques = np.unique(allv)
        kind = classify_tag(tag, len(uniques), uniques)
        discrete = is_discrete(tag, len(uniques), uniques, kind)
        entry = {
            "name": tag,
            "process": process_of(tag),
            "kind": kind,
            "unit": DOCUMENTED_UNITS.get(tag, "unknown"),
            "discrete": bool(discrete),
            "n_unique": int(len(uniques)),
            "observed_min": float(np.min(allv)) if allv.size else None,
            "observed_max": float(np.max(allv)) if allv.size else None,
            "value_source": "observed_train_sample",
        }
        if discrete and len(uniques) <= 8:
            entry["categories"] = [float(u) for u in uniques]
        schema["tags"].append(entry)
        rows.append(entry)

    import yaml
    schema_path = resolve(cfg, "tag_schema")
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(schema, f, sort_keys=False, allow_unicode=True)
    print(f"[schema] wrote {schema_path} ({len(tags)} tags)")

    audit.append("\n## Tag summary")
    audit.append("| tag | process | kind | discrete | unit | observed range |")
    audit.append("|---|---|---|---|---|---|")
    for r in rows:
        audit.append(f"| {r['name']} | {r['process']} | {r['kind']} | "
                     f"{r['discrete']} | {r['unit']} | "
                     f"[{r['observed_min']:.4g}, {r['observed_max']:.4g}] |")
    _write_audit(cfg, audit)
    print("[schema] reports/schema_audit.md written")
    return 0


def _write_audit(cfg, lines):
    out = resolve(cfg, "reports_dir") / "schema_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
