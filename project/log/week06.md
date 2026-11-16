# Week 6 — Training engine, scoring, calibration

## Done
- `src/models/train.py`: AdamW, batch 128, checkpointing, per-epoch metrics,
  device-aware, normal-data only.
- `src/detect/score.py`: per-channel residual normalization by train scales
  `artifacts/residual_scale.json`; Top-K aggregation configurable.
- `src/detect/replay.py`: windowed replay -> scores.parquet.
- `src/detect/calibrate.py`: normal-validation quantile threshold ->
  `artifacts/thresholds.json` (MOCK threshold 24.811 on 198,000 val rows).
- `src/detect/mock_scores.py`: deterministic mock scores with injected
  anomaly segments for integration tests without weights.
