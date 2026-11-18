# What to do next (after this engineering pass)

The codebase for Weeks 1–12 is complete and the pipeline runs end-to-end on
real HAI 23.05 data with mock scores. To turn it into thesis results:

## 1. Train the model (blocking step)

On a GPU or Apple-silicon machine:

```bash
bash scripts/run_train.sh --device auto --epochs 20 --batch-size 128 --tag cge
```

- Uses only normal training data (train1–3); train4 is held out as validation.
- Produces `checkpoints/TranAD_C_G_E_HAI-23.05/best.ckpt`. If you already have
  a checkpoint from the parent repo, pass it via
  `scripts/run_pipeline.sh --skip-train --checkpoint <path>` —
  `load_checkpoint.py` reports any state-dict mismatches.

## 2. Re-run the full pipeline with real scores

```bash
bash scripts/run_pipeline.sh --device auto
```

This regenerates scores (REAL instead of MOCK), recalibrates the threshold on
normal validation data, and rewrites `rootcause.jsonl`, `advice.jsonl`,
all `metrics_*.csv`, figures, and `reports/results.md`.

## 3. Review before trusting numbers

- **Action catalog**: `config/action_catalog.yaml` is a draft. Have each
  action's conditions and bounds checked against the HAI technical manual /
  your supervisor before citing policy results.
- **Surrogate scope**: `metrics_policy.csv` rows marked `surrogate_only` are
  proxy metrics — never describe them as real plant effects.
- **Ablation sanity**: confirm A4 (random graph) and A5 (label shuffle)
  degrade RCA metrics; if they don't, the graph/onset evidence isn't helping
  and that negative result must be reported.
- **Threshold protocol**: `thresholds.json` must trace to validation-normal
  scores only. Never tune on test labels.

## 4. Remaining engineering gaps

- Baselines (PCA/IsolationForest/OCSVM, USAD/AnomalyTransformer/GDN) — hooks
  exist in `src/bench/run_detection.py`; only TranAD paths are exercised.
- Multi-seed runs: set `seed` in `config/experiment.yaml` per run; bootstrap
  CIs are currently per-event.
- Efficiency table (latency/throughput/memory) — log lines exist; consolidate
  into `metrics_efficiency.csv` if needed for the thesis.
- `scripts/reproduce_all.py` — verify once on a clean checkout with a real
  checkpoint.

## 5. Status map

| Stage | Real data | Mock | Needs training |
|---|---|---|---|
| data → windows | done | — | — |
| scoring → calibration | ready | done | yes |
| RCA → advice | ready | done | yes |
| surrogate → policy eval | done | done | no |
| metrics/ablation/figures | ready | done | yes |
