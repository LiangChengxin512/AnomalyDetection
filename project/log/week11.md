# Week 11 — Ablations, bootstrap, figures

## Done
- `src/bench/ablation.py`: A2 residual-only, A3 no-attribution/no-onset,
  A4 no-graph + random graph, A5 label shuffle, A6 PA vs no-PA ->
  `metrics_ablation.csv`.
- `src/bench/bootstrap.py`: per-event bootstrap CIs -> `metrics_bootstrap.csv`.
- `scripts/make_figures.py`: score timeline, detection summary, RCA strategy,
  policy risk, event case study -> `figures/*.png`
  (fixed matplotlib >=3.10 tick_labels API).
