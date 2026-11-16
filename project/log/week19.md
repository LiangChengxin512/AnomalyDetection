# Week 19 / P6 — Bundle, runtime and factory application

Executed on 2026-11-16 as a logical continuation milestone.

Implemented portable bundles containing real CPU-loadable best weights, scaler,
feature order/type mask, residual/channel scales, normal references, locked alarm
thresholds, process graph, reviewed inspection catalog, surrogate qualification and
runtime configuration. Files are individually hashed; the loader checks structural
and preprocessing bindings. No original data path or interpreter is packaged.
The deployment model is preselected D07/seed 1, not the best test seed.

The application calls the same causal RuntimeEngine used by business experiments.
It supports continuous or step replay, speed changes, watch values, acknowledgements,
advice rejection, escalation, notes, case closure and export. SQLite preserves case
and operator actions, unique sample IDs prevent duplicate replay effects, and JSONL
and CSV provide portable audit views. End-of-input and missing-input holds are not
reported as normal recovery. Historical measurements do not change after advice.

English guides are delivered in `docs/APP_GUIDE.md`, `docs/FACTORY_ONBOARDING_EN.md`
and `docs/PLATFORM_GUIDE.md`. A factory example maps source columns, units and feature
types, with unreviewed physical actions disabled. New feature systems require
retraining. The legacy demo now delegates to this real bundle application.

Validation includes moving a real smoke bundle to a path with spaces/Unicode/`!`,
CPU loading, corrupted-file rejection, mismatched site-unit rejection, dynamic label
invariance and persistent case-workflow tests. Real application replay and the final
platform report are recorded in Week 20 and `reports/continuation_validation`.

The display is currently plain English text; explanations run synchronously and
latency is measured. Async explanation queues, live connectors and a browser UI
remain extensions rather than completed features.
