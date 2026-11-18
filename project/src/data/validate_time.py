"""Validate timestamp continuity, monotonicity and duplication per file.

Writes per-file row counts and gap statistics into the audit report and
the manifest used by downstream windowing (windows must not be fabricated
across gaps or file boundaries).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import write_json  # noqa: E402


def validate_file(path: Path, ts_col: str) -> dict:
    ts = pd.read_csv(path, usecols=[ts_col])[ts_col]
    # 去掉 format="mixed"，加上 errors="coerce" 确保解析失败时转为 NaT 而不中断
    parsed = pd.to_datetime(ts, errors="coerce")
    diffs = parsed.diff().dropna()
    n_dup = int(parsed.duplicated().sum())
    n_unsorted = int((parsed.diff().dropna() < pd.Timedelta(0)).sum())
    gaps = diffs[diffs > pd.Timedelta(seconds=1)]
    return {
        "file": path.name,
        "rows": int(len(ts)),
        "start": str(parsed.iloc[0]),
        "end": str(parsed.iloc[-1]),
        "duplicates": n_dup,
        "non_monotonic_steps": n_unsorted,
        "gaps_over_1s": int(len(gaps)),
        "max_gap_s": float(diffs.max().total_seconds()) if len(diffs) else 0.0,
        "gap_positions": [int(i) for i in gaps.index[:20]],
        "parse_failures": int(parsed.isna().sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir = resolve(cfg, "raw_dir")
    files = (cfg["data"]["train_files"] + cfg["data"]["val_files"]
             + cfg["data"]["test_files"]
             + list(cfg["data"]["label_files"].values()))

    report = []
    ok = True
    for name in files:
        path = raw_dir / name
        if not path.exists():
            print(f"[time] MISSING {path}", file=sys.stderr)
            ok = False
            continue
        info = validate_file(path, cfg["data"]["timestamp_col"])
        # label-test2.csv uses minute-resolution timestamps: duplicates and
        # 60 s "gaps" are expected there, not a data defect.
        is_minute_res = (info["duplicates"] > 0 and info["max_gap_s"] <= 60.0
                         and name.startswith("label"))
        info["minute_resolution"] = bool(is_minute_res)
        report.append(info)
        clean = (info["parse_failures"] == 0 and info["non_monotonic_steps"] == 0
                 and (is_minute_res or (info["duplicates"] == 0
                                        and info["gaps_over_1s"] == 0)))
        status = "OK" if clean else "WARN"
        print(f"[time] {info['file']}: rows={info['rows']} "
              f"{info['start']}..{info['end']} dups={info['duplicates']} "
              f"gaps>1s={info['gaps_over_1s']} maxgap={info['max_gap_s']:.0f}s [{status}]")
        if status == "WARN":
            ok = False

    out = resolve(cfg, "processed_dir") / "time_validation.json"
    write_json(out, {"files": report, "all_ok": ok})
    print(f"[time] wrote {out}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
