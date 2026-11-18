# Week 9 — Surrogate and closed-loop replay

## Done
- `src/sim/fit_surrogate.py`: ridge one-step transition model on 200,000
  normal rows -> `artifacts/surrogate_model.json`.
- `src/sim/validate_surrogate.py`: median RMSE/std 0.029, interval coverage
  0.998 -> `artifacts/surrogate_card.json` (usable=true, shadow scope only).
- `src/sim/closed_loop.py`: replays no-action / fixed / adaptive policies per
  event -> `artifacts/policy_replay.parquet`, all rows tagged surrogate-only.
