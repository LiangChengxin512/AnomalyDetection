# Results — HAI 23.05 anomaly perception → root cause → shadow decision

> Status: pipeline fully wired and validated end-to-end with **mock scores**.
> Numbers below marked `MOCK` are integration smoke outputs, **not** research
> results. Replace by re-running `scripts/run_pipeline.sh` after training.

## 1. Detection (artifacts/metrics_detection.csv)

| metric | MOCK value | note |
|---|---|---|
| point PR-AUC | 0.981 | mock scores only |
| VUS-PR | 0.931 | mock scores only |
| event recall | 1.00 | 52 label events, all covered (mock over-alerts) |
| event precision | 0.003 | 17,077 predicted events → heavy false-alert load |
| false alarms / hour | 864.7 | mock threshold deliberately loose |
| eTaPR F1 | 0.0148 | eTaPR installed (etapr 22.6.1), works |
| PA-F1 (appendix only) | 0.143 | reported for historical comparability, not a main metric |

Threshold source: `artifacts/thresholds.json` — 99.5% quantile of **normal
validation** scores (`hai-train4`, 198,000 rows). No test labels used.

## 2. Root-cause ranking (artifacts/metrics_rca.csv)

Evaluated on the 52 labeled attack events whose detected events overlap them,
against manual attack targets (`config/attack_targets_hai2305.yaml`).

| strategy | top-1 | top-3 | MRR |
|---|---|---|---|
| full (residual+attr+onset+graph+stability) | 0.269 | 0.442 | 0.402 |
| residual only | 0.442 | 0.615 | 0.533 |
| attribution only | 0.442 | 0.615 | 0.534 |
| random graph (negative control) | 0.385 | 0.538 | 0.483 |
| target permutation (negative control) | 0.058 | 0.115 | 0.103 |

MOCK interpretation only: scores carry no real signal so residual/attribution
degenerate to the same ordering; the target-permutation floor (≈0.06 top-1)
confirms the harness itself is not leaking labels.

## 3. Shadow policy (artifacts/metrics_policy.csv)

| metric | MOCK value | scope |
|---|---|---|
| events replayed | 17,077 | — |
| legal action rate | 1.000 | hard catalog constraint |
| abstain rate | 0.454 | evidence/uncertainty gate |
| mean adaptive − no-action risk | 0.0020 | **surrogate only** |
| frac events improved vs no-action | 0.030 | **surrogate only** |

Surrogate card (`artifacts/surrogate_card.json`): ridge one-step model fit on
200,000 normal rows; median RMSE/std = 0.029; interval coverage = 0.998;
`usable = true` for shadow evaluation **only**.

## 4. Bootstrap & ablations

- `artifacts/metrics_bootstrap.csv`: per-event bootstrap CIs.
- `artifacts/metrics_ablation.csv`: A2 residual-only, A3 attribution/onset
  removal, A4 graph removal + random graph, A5 label shuffle, A6 PA vs no-PA.

## 5. Figures (figures/)

`score_timeline.png`, `detection_summary.png`, `rca_strategy.png`,
`policy_risk.png`, `event_case.png` — all generated from artifacts above.
