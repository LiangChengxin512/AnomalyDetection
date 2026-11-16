#!/usr/bin/env python3
"""Package run artifacts + reports + config into a dated bundle."""
from __future__ import annotations

import datetime as dt
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, project_path, resolve  # noqa: E402
from src.common.io_utils import write_json  # noqa: E402


def main() -> int:
    cfg = load_config()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = project_path("artifacts", f"bundle_{stamp}")
    bundle.mkdir(parents=True, exist_ok=True)
    for d in ("artifacts", "reports", "figures"):
        src = resolve(cfg, f"{d}_dir" if d != "artifacts" else "artifacts_dir")
        dst = bundle / d
        dst.mkdir(exist_ok=True)
        for p in src.glob("*"):
            if p.is_file() and not p.name.startswith("bundle_"):
                shutil.copy2(p, dst / p.name)
    for f in ("config/experiment.yaml", "config/tag_schema.yaml",
              "config/action_catalog.yaml", "config/process_graph.json",
              "config/attack_targets_hai2305.yaml", "data/raw_manifest.json"):
        p = project_path(f)
        if p.exists():
            shutil.copy2(p, bundle / Path(f).name)
    write_json(bundle / "bundle_manifest.json", {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_version": cfg["dataset_version"],
        "note": "Raw CSV data is not bundled (CC BY-SA; re-fetch via scripts/fetch_hai.py)."})
    print(f"[package] bundle -> {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
