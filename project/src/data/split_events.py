"""Build the event manifest.

Combines contiguous label runs (from the prepared test parquet), the
manual attack-target table (config/attack_targets_hai2305.yaml) and the
summary_label*.txt files. Events are matched to manual attack IDs by
time overlap; label runs with no manual entry are kept with
``attack_id: null`` so they can be reported separately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, project_path, resolve  # noqa: E402
from src.common.events import binary_to_ranges, event_id_for  # noqa: E402
from src.common.io_utils import write_json  # noqa: E402


def load_attack_table(cfg) -> dict[str, list[dict]]:
    path = project_path(cfg["paths"]["attack_targets"])
    table = yaml.safe_load(open(path, encoding="utf-8"))
    by_file: dict[str, list[dict]] = {}
    for ev in table["events"]:
        by_file.setdefault(ev["file"], []).append(ev)
    return by_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = resolve(cfg, "processed_dir")
    attack_table = load_attack_table(cfg)

    manifest = {"dataset_version": "HAI-23.05", "events": []}
    for name in cfg["data"]["test_files"]:
        df = pd.read_parquet(out_dir / f"test__{name.replace('.csv', '')}.parquet",
                             columns=["timestamp", "label_attack"])
        runs = binary_to_ranges(df["label_attack"].to_numpy())
        manual = attack_table.get(name, [])
        manual_sorted = sorted(manual, key=lambda e: e["start"])
        used = set()
        for i, (s, e) in enumerate(runs, 1):
            ts_s, ts_e = df["timestamp"].iloc[s], df["timestamp"].iloc[e]
            best, best_overlap = None, pd.Timedelta(0)
            for j, man in enumerate(manual_sorted):
                ms, me = pd.Timestamp(man["start"]), pd.Timestamp(man["end"])
                ov = min(ts_e, me) - max(ts_s, ms)
                if ov > best_overlap:
                    best, best_overlap, used_idx = man, ov, j
            rec = {
                "event_id": event_id_for(name, i),
                "file": name,
                "row_start": s, "row_end": e,
                "start": str(ts_s), "end": str(ts_e),
                "duration_s": int((ts_e - ts_s).total_seconds()) + 1,
                "attack_id": None, "scenario": None, "controller": None,
                "target_points": [], "observable_tags": [],
                "gt_confidence": None,
            }
            if best is not None:
                rec.update({"attack_id": best["attack_id"], "scenario": best["scenario"],
                            "controller": best["controller"],
                            "target_points": best["target_points"],
                            "observable_tags": best["observable_tags"],
                            "gt_confidence": best.get("confidence", "high")})
                used.add(best["attack_id"])
            manifest["events"].append(rec)
        orphans = [m["attack_id"] for m in manual if m["attack_id"] not in used]
        for oid in orphans:
            print(f"[events] WARNING: {oid} ({name}) not matched to a label run")

    path = out_dir / "events.json"
    write_json(path, manifest)
    n = len(manifest["events"])
    matched = sum(1 for e in manifest["events"] if e["attack_id"])
    print(f"[events] {n} label events, {matched} matched to manual IDs -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
