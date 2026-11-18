# CLI instructions

All commands run from `project/` with the shared venv:

```bash
cd /home/devin/Code/Liangchengxin/AnomalyDetection/project
PY=/home/devin/Code/Liangchengxin/.venv/bin/python
```

## Data pipeline (already run on real HAI 23.05)

```bash
$PY scripts/fetch_hai.py            # download raw files + manifest
$PY scripts/manifest.py             # SHA-256 manifest -> data/raw_manifest.json
$PY -m src.data.inspect_schema      # -> config/tag_schema.yaml, reports/schema_audit.md
$PY -m src.data.validate_time       # 1 Hz continuity + label-file checks
$PY -m src.data.prepare_hai         # -> data/processed/*.parquet
$PY -m src.data.split_events        # 52 events -> artifacts/events_index.json
$PY -m src.data.fit_scaler          # -> artifacts/scaler.json + residual_scale.json
$PY -m src.data.build_graph         # -> config/process_graph.json
```

## Training (heavy — run on GPU/MPS machine)

```bash
bash scripts/run_train.sh --device auto --epochs 20 --batch-size 128 --tag cge
# or directly:
$PY -m src.models.train --model TranAD_C_G_E --device auto --epochs 20 --batch-size 128
# quick CPU smoke: bash scripts/run_train.sh --max-train-windows 500 --epochs 1
```

Checkpoint -> `checkpoints/TranAD_C_G_E_HAI-23.05/best.ckpt`
(training logs tee'd to `artifacts/train_logs/`).

## Scoring + downstream pipeline

```bash
# full pipeline in one shot (auto-skips trained stages if checkpoint absent):
bash scripts/run_pipeline.sh --device auto

# or stage by stage:
CKPT=checkpoints/TranAD_C_G_E_HAI-23.05/best.ckpt
$PY -m src.detect.replay --checkpoint $CKPT --splits val,test --device auto
$PY -m src.detect.calibrate --scores artifacts/scores_val_*.parquet
$PY -m src.explain.rootcause --scores artifacts/scores_test_*.parquet
$PY -m src.decision.shadow_policy --scores artifacts/scores_test_*.parquet
$PY -m src.sim.fit_surrogate && $PY -m src.sim.validate_surrogate
$PY -m src.sim.closed_loop --scores artifacts/scores_test_*.parquet
$PY -m src.bench.run_detection && $PY -m src.bench.run_rca \
    --scores artifacts/scores_test_*.parquet && $PY -m src.bench.run_policy
$PY -m src.bench.ablation --primary TranAD_C_G_E && $PY -m src.bench.bootstrap \
    --scores artifacts/scores_test_*.parquet
$PY scripts/make_figures.py
$PY scripts/package_artifacts.py
```

## Mock mode (no checkpoint needed)

```bash
$PY -m src.detect.mock_scores --splits train,val,test --inject-signal
bash scripts/run_pipeline.sh --mock            # full chain via reproduce_all.py
```

`--device auto|cuda|mps|cpu` is accepted by `run_train.sh`, `run_pipeline.sh`,
`train.py`, `replay.py`.
