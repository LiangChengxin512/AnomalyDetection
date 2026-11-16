# CSV and artifact dictionary (schema 2.0)

CSV uses UTF-8, comma separators, a header, decimal points, and no DataFrame index.
Missing or undefined numeric results are empty. `status` and `reason` distinguish
missing denominators, unavailable metrics, unqualified surrogate effects and failed
tasks. A zero is a measured zero, never a substitute for unavailable data.

## Common identifiers

`run_id` identifies an isolated execution. `experiment_id` is D/R/P/B/N plus its
registered number. `variant_id` identifies a business capability variant. `model_id`
is the model name. `seed` is empty for deterministic baselines; estimator sampling
seeds are recorded separately. `dataset_fingerprint`, `config_hash`, `code_hash`,
`git_commit`, `git_dirty`, OS/Python/PyTorch versions and actual device provide
provenance. The source hash includes untracked Python source; Git commit alone is
not used as a content version.

`sample_id` is derived from dataset/source identity plus segment and original row.
`source_file + segment_id + row_in_source` is the human-readable row key. Timestamps
preserve declared source timezone semantics. `alarm_id` is generated from the first
causal trigger, independent of labels. `gt_event_id` is evaluation-only.
`threshold_id` binds the frozen threshold to model weights, scaler, residual scale,
aggregation, alarm policy and the normal calibration keys. Large detail tables use
these keys to join the run manifest rather than repeating all metadata.

## Tables

| File | Row grain | Meaning / units |
|---|---|---|
| `task_registry.csv` | Task | Dependency IDs, status, times, seconds, exit code, artifact directory |
| `training_history.csv` | Model/seed/epoch | Train/reconstruction loss, normal validation MSE, LR, patience state, physical/effective batch, epoch seconds, actual device |
| `training_summary.csv` | Model/seed | Total planned/completed epochs, best epoch/loss, stop reason, checkpoint SHA256, resume count |
| `metrics_detection.csv` | Detector/seed/metric | AP, trapezoidal PR-AUC, point and one-to-one event metrics, any-overlap coverage, FAR per effective normal hour, conditional delay in seconds |
| `metrics_rca.csv` | Method/seed/target scope/conditioning/snapshot/metric | Strict observable vs weak associated targets; snapshots at alarm + 0/5/10 seconds; top-k and MRR with explicit eligible/evaluated counts |
| `metrics_policy.csv` | Policy/seed/metric | Requests, offered/refused/suppressed advice, actual repeated offers, prevented duplicates, coverage and cost |
| `metrics_ablation.csv` | Business variant/seed/metric | Reexecuted capability changes; reference B00 |
| `metrics_negative_controls.csv` | Control/seed/metric | Execution audit and available target/flow results |
| `metrics_efficiency.csv` | Stage/device/metric | Synchronized batch-1 p50/p95 milliseconds, throughput, allocator bytes; explanation latencies separately labeled |
| `metrics_ci.csv` | Metric/seed/interval | Estimate, low/high, resampling unit, bootstrap seed/B, paired-event count |
| `event_detection.csv` | True or predicted interval | One-to-one match, alarm ID, duplicate, missed flag, delay, preexisting alarm, first trigger |
| `event_rca.csv` | True event/method/target scope/snapshot | Eligible targets, full-ranking top-k/MRR, explained flag, abstention and decision time |
| `event_policy.csv` | True event/variant/seed | Event alarm requests, offers, cost, cases and repeated offers |
| `rootcause_candidates.csv` | Alarm/time/method/variable | Full rank, normalized score, residual/gradient/onset/graph/stability components and availability |
| `policy_decisions.csv` | Decision request | Selected action or null, mode, refusal/cooldown reason, cost, legality, repeated-offer flag |
| `policy_constraints.csv` | Candidate/constraint/request | Passed flag, observed value, declared limit and unit when applicable |
| `policy_rollout.csv` | Qualified rollout step, or unavailable marker | Surrogate-only effects; no artificial zeros when qualification fails |
| `alarm_workload.csv` | Attack event/variant/seed | Alarm/case/advice workload; normal-period rates also appear in metric summaries |
| `surrogate_validation.csv` | Surrogate/tag/horizon | 1-/5-step RMSE, persistence RMSE, normal interval coverage/width, qualification |
| `threshold_sensitivity.csv` | Detector/seed/normal quantile | Candidate threshold, locked threshold ID/value, normal event FAR, point exceedance and test precision/recall/F1; analysis only |
| `coverage_risk.csv` | Seed/evidence margin gate | Request/offer/specific-offer/refusal/cooldown counts, advice coverage, specific coverage, unqualified risk as empty with reason |
| `case_actions.csv` | Operator/system action | Case, actor, timestamp, reason, old and new state |
| `summary_*.csv` | Method/metric/protocol facet | Numeric seed mean/std/count; deterministic methods count once |

`advice_coverage` counts offered advice over all decision requests, including
requests suppressed by cooldown. `n_specific_advice_abstained` separately counts
withheld specific routing when a generic inspection is still offered. Legal-rate
denominator is offered advice, not all requests. Inspection cost is a workflow
proxy, not monetary cost or physical risk reduction.

## Detail and plotting inputs

Each scoring task stores `val_threshold_scores.parquet`, `test_scores.parquet`,
`*_residual.npy`, `*_evidence.npy`, `threshold.json`, `residual_scale.json` and
`channel_thresholds.json`. NPY rows follow the exact sample order in the matching
score Parquet. The application never reads benchmark CSV files to create outputs.
Use `scripts/export_scores.py --run-dir RUN --model D07 --seed 1 --source FILE
--tag TAG --output selected.csv` to export an explicit selection.

`plot_data/` contains actual figure inputs, training curves, method/seed-specific
PR/ROC curves and score timelines. Timelines explicitly mark
`purpose=evaluation_plot` and may join labels only for analysis. Figures are PNG
and SVG; `figure_manifest.json` records source CSV and style hashes. Changing
`config/plots.yaml` and running `make_figures.py` never retrains or recalibrates.
No missing quantity is filled with zero for plotting.

Threshold sensitivity uses the same alarm machine at normal-calibration quantiles
0.99, 0.995, 0.999, 0.9995 and 1.0. Test metrics are evaluation-only and never select
the deployment threshold. Coverage analysis reexecutes policy gates at evidence
margins 0, 0.01, 0.05 and 0.1 on identical causal R05 snapshots, including cooldown
effects. `specific_coverage` excludes generic inspection. A seed with no requests
has undefined coverage, not zero. `risk` remains empty while physical action
response is unqualified. These grids describe sensitivity, not calibrated
probabilities or intervention benefit.

Run `python scripts/export_analysis.py --run-dir RUN` to regenerate these two
tables from frozen artifacts on CPU. Its report records input/output hashes and
confirms that model weights, residual scales and selected thresholds did not change.
HIL intervention studies and the full uncapped benchmark remain separate work.
