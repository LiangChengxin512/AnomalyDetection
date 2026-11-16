"""Convert raw HAI CSVs into per-split parquet files.

Output columns: ``timestamp, <86 tags>, label_attack, event_id, source_file``.
Labels are kept strictly separate from model features downstream; they are
carried here only so evaluation code can join on timestamp without
re-reading raw CSVs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.events import binary_to_ranges, event_id_for  # noqa: E402
from src.common.io_utils import write_json, write_parquet  # noqa: E402


def _verify_against_summary(summary_path: Path, df: pd.DataFrame,
                            label: np.ndarray, tol_s: int = 60) -> bool:
    """Check label runs cover the attack windows listed in summary_label."""
    txt = summary_path.read_text()
    runs = binary_to_ranges(label)
    windows = []
    for line in txt.splitlines():
        m = re.match(r"\[\s*\d+\]\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
                     r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m:
            windows.append((pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2))))
    for ws, we in windows:
        hit = any(df["timestamp"].iloc[s] <= we + pd.Timedelta(seconds=tol_s)
                  and df["timestamp"].iloc[e] >= ws - pd.Timedelta(seconds=tol_s)
                  for s, e in runs)
        if not hit:
            return False
    return True


def _load_tags(cfg) -> list[str]:
    import yaml
    schema = yaml.safe_load(open(resolve(cfg, "tag_schema"), encoding="utf-8"))
    return [t["name"] for t in schema["tags"]]


def prepare_file(cfg, name: str, tags: list[str], label_map: dict | None):
    raw_dir = resolve(cfg, "raw_dir")
    ts_col = cfg["data"]["timestamp_col"]
    df = pd.read_csv(raw_dir / name)
    if df.columns[0] != ts_col:
        raise ValueError(f"{name}: first column is {df.columns[0]!r}, expected {ts_col!r}")
    missing = [t for t in tags if t not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing tags {missing}")
    out = df[[ts_col] + tags].copy()
    # 去掉 format="mixed"，加上 errors="coerce" 确保解析失败时转为 NaT 而不中断
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.rename(columns={ts_col: "timestamp"})

    label = np.zeros(len(out), dtype=np.int64)
    if name in label_map:
        lab = pd.read_csv(raw_dir / label_map[name])
        lab.columns = ["timestamp", "label"]
        if len(lab) != len(out):
            raise ValueError(f"{name}: label rows {len(lab)} != data rows {len(out)}")
        # 去掉 format="mixed"，加上 errors="coerce" 确保解析失败时转为 NaT 而不中断
        lab_ts = pd.to_datetime(lab["timestamp"], errors="coerce")
        if (lab_ts.to_numpy() == out["timestamp"].to_numpy()).all():
            label = lab["label"].fillna(0).astype(int).to_numpy()
        else:
            # label-test2.csv stores minute-resolution timestamps
            # (60 data rows share one label timestamp). Row counts match,
            # so align by position and verify the per-second values
            # reproduce the summary attack windows within 60 s.
            label = lab["label"].fillna(0).astype(int).to_numpy()
            summary = cfg["data"].get("summary_files", {}).get(name)
            if summary:
                ok = _verify_against_summary(raw_dir / summary, out, label)
                print(f"[prepare] {name}: labels position-aligned "
                      f"(minute-resolution stamps); summary check={'OK' if ok else 'FAILED'}")
                if not ok:
                    raise ValueError(f"{name}: position-aligned labels disagree "
                                     "with summary_label attack windows")
    out["label_attack"] = label
    out["source_file"] = name

    # event ids from contiguous label runs (1 s grid, no gap merge)
    out["event_id"] = ""
    for i, (s, e) in enumerate(binary_to_ranges(label), 1):
        out.loc[s:e, "event_id"] = event_id_for(name, i)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tags = _load_tags(cfg)
    out_dir = resolve(cfg, "processed_dir")
    label_map = cfg["data"]["label_files"]

    splits = {
        "train": cfg["data"]["train_files"],
        "val": cfg["data"]["val_files"],
        "test": cfg["data"]["test_files"],
    }
    index = {"tags": tags, "splits": {}}
    for split, files in splits.items():
        split_paths = []
        for name in files:
            df = prepare_file(cfg, name, tags, label_map)
            dest = out_dir / f"{split}__{name.replace('.csv', '')}.parquet"
            write_parquet(df, dest)
            n_anom = int(df["label_attack"].sum())
            print(f"[prepare] {split}/{name}: {len(df)} rows, {n_anom} anomalous "
                  f"({100.0 * n_anom / max(1, len(df)):.2f}%) -> {dest.name}")
            split_paths.append({"file": name, "parquet": str(dest),
                                "rows": len(df), "anomalous_rows": n_anom})
        index["splits"][split] = split_paths
    write_json(out_dir / "index.json", index)
    print(f"[prepare] wrote {out_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
