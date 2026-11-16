# Week 1 — Data acquisition and manifest

## Done
- Downloaded HAI 23.05 (no git-lfs available; used GitHub media/raw URLs):
  hai-train1..4.csv, hai-test1..2.csv, label-test1..2.csv, summary_label1..2.txt
  (~690 MB) into `data/raw/hai-23.05/`.
- `scripts/fetch_hai.py` + `scripts/manifest.py` written; manifest records URL,
  SHA-256, size for every raw file -> `data/raw_manifest.json`.
- Fetched `hai_dataset_technical_details.pdf` and NetworkX graph
  `phy_boiler.json` (41 nodes / 67 links) for reference.

## Issues
- git-lfs missing -> bypassed with direct media URLs. Raw data is not versioned.
