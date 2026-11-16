# Week 18 / P5 — Metrics, uncertainty and redrawable outputs

Executed on 2026-11-16 as a logical continuation milestone.

Added schema-versioned long CSV tables for training, detection, RCA, policy,
business variants, negative controls, efficiency, intervals and event details.
Large telemetry/score records use Parquet; nested RCA uses JSONL. Source keys,
configuration/source hashes, threshold IDs and checkpoint/estimator hashes bind
results to explicit run artifacts. Failed tasks and undefined values remain visible.

Event matching maximizes one-to-one matches before IoU; one persistent alarm cannot
become many true positives. Any-overlap coverage is separately named. Missed delays
are empty. Point AP uses the standard definition. eTaPR uses the author source pinned
to commit af9e7aed35cfd160cbe0d04c8ec4c102502cb677, with a documented optional-plotting
import patch and guarded empty-range edge cases. Golden examples passed. The former
VUS approximation was removed; author VUS integration remains unavailable.

Implemented source-stratified attack/normal-block bootstrap for point F1 and paired
attack-event intervals for RCA/business differences. Seed uncertainty is reported
separately as mean/std/count. The smoke fixture's tiny event count does not support
performance conclusions or meaningful generalization claims.

`plot_data` contains the actual exported plotting inputs. `make_figures.py` reads
CSV only, supports style configuration and writes PNG/SVG plus input hashes. Training
and detector plots were visually inspected. Standalone plots identify their scope.
Changing style does not train or recalibrate. See `docs/CSV_DICTIONARY.md`.

Added threshold and evidence-margin sensitivity exports. On the final frozen smoke
artifacts, five normal-calibration quantiles across 17 model/seed entries produce
85 rows; four evidence-margin gates across three seeds produce 12 rows. Alarm
states are recalculated at each candidate threshold. Policy gates and cooldowns
are reexecuted on the same causal ranking snapshots. Selected deployment thresholds
are never replaced using test metrics. All 248 model/score/surrogate input files
retained identical hashes. Zero-request coverage and unqualified physical risk stay
empty. `scripts/export_analysis.py` repeats this CPU-only analysis.

The redraw command also supports PR/ROC curves, per-source score timelines,
threshold sensitivity, generic versus specific advice coverage, efficiency, and
per-model/seed train/validation curves with best/last epochs. Missing attack classes
and unavailable effect measurements receive explicit unavailable records. Frozen
CSV files, not model execution, supply every plotted value.

Remaining research scope: the full uncapped experiment, comprehensive intervals for
all metrics, author VUS validation, and qualified intervention-effect studies.
