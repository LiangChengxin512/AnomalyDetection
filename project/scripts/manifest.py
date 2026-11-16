#!/usr/bin/env python3
"""Scan the raw data directory and (re)build data/raw_manifest.json."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import project_path  # noqa: E402
from src.common.io_utils import sha256_file, write_json  # noqa: E402


def main() -> int:
    raw_root = project_path("data", "raw")
    manifest_path = project_path("data", "raw_manifest.json")
    files = []
    for p in sorted(raw_root.rglob("*")):
        if p.is_file() and p.name != "raw_manifest.json":
            files.append({
                "file": str(p.relative_to(project_path())),
                "size": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    manifest = {"dataset_version": "HAI-23.05",
                "scanned_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "files": files}
    write_json(manifest_path, manifest)
    print(f"[manifest] {len(files)} files hashed -> {manifest_path}")
    for f in files:
        print(f"  {f['size']:>12d}  {f['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
