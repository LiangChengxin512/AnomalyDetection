# Week 3 — Preparation and event splitting

## Done
- `src/data/prepare_hai.py`: produced processed Parquet splits
  (train = train1-3, val = train4, test = test1+test2) with labels kept in
  separate columns (`label_attack`, `event_id`, `source_file`), never in
  model features. Label-test2 aligned by row position and checked against
  `summary_label2.txt` windows.
- `src/data/split_events.py`: found 52 attack events; matched all 52 to
  manual attack IDs via `config/attack_targets_hai2305.yaml`.

## Output
- `data/processed/*.parquet`, `artifacts/events_index.json`.
