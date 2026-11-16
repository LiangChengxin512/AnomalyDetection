#!/usr/bin/env python3
"""Fetch the HAI 23.05 dataset and record provenance.

Two download strategies, tried in order:

1. ``git clone`` + ``git lfs pull`` when git-lfs is installed.
2. Direct HTTPS download from GitHub's media endpoint
   (``media.githubusercontent.com``), which serves LFS objects without
   requiring git-lfs. Small files (graph JSON) come from raw.githubusercontent.

Every downloaded file is hashed (SHA256) and recorded in
``data/raw_manifest.json`` together with the source URL and fetch
timestamp. Raw data is never committed to git.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import project_path  # noqa: E402
from src.common.io_utils import sha256_file, write_json  # noqa: E402

REPO = "https://github.com/icsdataset/hai"
MEDIA = "https://media.githubusercontent.com/media/icsdataset/hai/master"
RAW = "https://raw.githubusercontent.com/icsdataset/hai/master"

HAI_2305_FILES = [
    "hai-train1.csv", "hai-train2.csv", "hai-train3.csv", "hai-train4.csv",
    "hai-test1.csv", "hai-test2.csv",
    "label-test1.csv", "label-test2.csv",
    "summary_label1.txt", "summary_label2.txt",
]
GRAPH_FILES = [
    "phy_boiler.json", "dcs_1001h.json", "dcs_1002h.json", "dcs_1003h.json",
    "dcs_1004h.json", "dcs_1010h.json", "dcs_1011h.json", "dcs_1020h.json",
]
DOC_FILES = [
    ("hai_dataset_technical_details.pdf", MEDIA),
    ("README.md", RAW),
]

LICENSE_NOTE = (
    "HAI datasets: CC BY-SA 4.0 per the official repository README. "
    "Technical manual v4.0: CC BY-SA 4.0. Do not redistribute raw data "
    "inside the code repository."
)


def _download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"Failed to download {url}: {last}")


def fetch_via_http(raw_dir: Path, graph_dir: Path) -> list[dict]:
    records = []
    for name in HAI_2305_FILES:
        dest = raw_dir / name
        url = f"{MEDIA}/hai-23.05/{name}"
        if not dest.exists() or dest.stat().st_size < 200:
            print(f"[fetch] {name} ...")
            _download(url, dest)
        records.append({"file": str(dest.relative_to(project_path())),
                        "url": url, "size": dest.stat().st_size})
    for name in GRAPH_FILES:
        dest = graph_dir / name
        url = f"{RAW}/graph/boiler/{name}"
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"[fetch] graph/{name} ...")
            _download(url, dest)
        records.append({"file": str(dest.relative_to(project_path())),
                        "url": url, "size": dest.stat().st_size})
    for name, base in DOC_FILES:
        dest = raw_dir.parent / name
        url = f"{base}/{name}"
        if not dest.exists() or dest.stat().st_size < 200:
            print(f"[fetch] {name} ...")
            _download(url, dest)
        records.append({"file": str(dest.relative_to(project_path())),
                        "url": url, "size": dest.stat().st_size})
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch HAI 23.05 raw data.")
    ap.add_argument("--out", default=str(project_path("data", "raw")),
                    help="raw data root (default: project/data/raw)")
    args = ap.parse_args()

    raw_root = Path(args.out)
    raw_dir = raw_root / "hai-23.05"
    graph_dir = raw_root / "graph" / "boiler"

    have_lfs = shutil.which("git-lfs") is not None or (
        subprocess.run(["git", "lfs", "version"], capture_output=True).returncode == 0)
    records = []
    if have_lfs and not (raw_dir / "hai-train1.csv").exists():
        try:
            print("[fetch] git-lfs detected; cloning official repository ...")
            subprocess.run(["git", "clone", "--depth", "1", REPO + ".git",
                            str(raw_root / "hai-repo")], check=True)
            subprocess.run(["git", "lfs", "pull"], cwd=raw_root / "hai-repo", check=True)
            for name in HAI_2305_FILES:
                shutil.copy2(raw_root / "hai-repo" / "hai-23.05" / name, raw_dir / name)
            for name in GRAPH_FILES:
                shutil.copy2(raw_root / "hai-repo" / "graph" / "boiler" / name, graph_dir / name)
        except subprocess.CalledProcessError as exc:
            print(f"[fetch] git-lfs path failed ({exc}); falling back to HTTPS.")

    records += fetch_via_http(raw_dir, graph_dir)

    manifest_path = project_path("data", "raw_manifest.json")
    manifest = {"dataset_version": "HAI-23.05",
                "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_repo": REPO,
                "license": LICENSE_NOTE,
                "files": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            manifest["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        except Exception:
            pass
    seen = {f["file"] for f in manifest.get("files", [])}
    for rec in records:
        rec["sha256"] = sha256_file(project_path(rec["file"]))
        if rec["file"] not in seen:
            manifest["files"].append(rec)
        else:
            for old in manifest["files"]:
                if old["file"] == rec["file"]:
                    old.update(rec)
    write_json(manifest_path, manifest)
    print(f"[fetch] manifest updated: {manifest_path} ({len(manifest['files'])} files)")
    print("[fetch] license:", LICENSE_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
