# Week 12 — Reproduction, packaging, reports

## Done
- `scripts/reproduce_all.py`: single-config re-run entry.
- `scripts/package_artifacts.py`: bundles configs, schemas, artifacts, logs,
  reports (excludes raw data).
- `reports/results.md`, `reports/limitations.md` written; `log/cli_instructions.md`
  and `log/next_steps.md` document exact commands and the remaining
  train-then-rerun schedule.
- `scripts/run_train.sh`, `scripts/run_pipeline.sh`: robust entry points with
  `set -euo pipefail`, device flags, logging.
