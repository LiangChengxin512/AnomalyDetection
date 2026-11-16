# Experiment protocol and execution

The authoritative CLI is `scripts/reproduce_all.py`. Its `.sh` and `.bat` wrappers
forward all arguments to that file. Default execution uses real models and real
HAI data. Historical mock and v1 artifacts are never selected by directory scans.
The old standalone v1 CLIs reject the v2 configuration with a migration message.

## Suites

| Suite | Models | Default seeds | Data budget | Meaning |
|---|---|---|---|---|
| `full` | D01–D07 | 1, 7, 21 | All declared data | Full protocol, after every required task succeeds |
| `smoke` | D01–D07 | 1 | 384 rows per source and validation role, 2 epochs | Integration only |
| `business_ablation` | D07 | 1, 7, 21 | All declared data | Business chain comparison |
| `onboarding` | Site primary D07 | 1 | All declared data | New-factory calibration and packaging; labels optional |

`--smoke-replay-rows 1900` explicitly extends only the smoke replay prefix. It never
changes the full suite. The first 1,900 HAI test1 rows include one registered attack;
this is a coverage fixture, not a representative benchmark. Its direct target is
not observable, so strict localization remains unavailable in that prefix.

`--models D01,D07`, `--experiments R01,R05,B00`, or alternate seeds declare a subset.
A subset does not receive full acceptance. Deterministic D01 and D02 are fitted
once, with an empty result seed. D03–D05 use seed-specific proportional sampling;
the exact sampled keys are in `fit_samples.csv`. Full neural training uses every
legal training window at stride 1. Test scoring always has stride 1.

## Detector registry

| ID | Method | Score and fit |
|---|---|---|
| D01 | Robust Z / MAD | Train median/MAD, scale floor, discrete normal-frequency surprise; max channel score |
| D02 | EWMA | Score before updating with alpha 0.2; reset per segment |
| D03 | PCA | Up to 20 components, up to 50,000 proportional train samples; reconstruction |
| D04 | Isolation Forest | 200 trees, max_samples 256, up to 50,000 train samples; negative score_samples |
| D05 | One-class SVM | RBF, nu 0.01, gamma scale, up to 5,000 train samples; negative decision_function |
| D06 | TranAD | Base project implementation, Top-3 aggregation |
| D07 | TranAD_C_G_E | Existing C/G/E model, Top-3 aggregation |

No neural C/G/E structure ablation is introduced. Both neural models use the same
window, normalization, effective batch and validation monitor. Architecture is
from this repository's TranAD adapter, not a claim to reproduce published tables.

## Data isolation and calibration

HAI train1–3 are normal training data. Train4 is split chronologically into 40%
model validation, 20% uncertainty calibration, and 40% threshold calibration.
W−1 excluded rows separate the roles. Windows are gathered lazily and cannot cross
files, segments, or role boundaries. Invalid timestamps stop preparation. Gaps,
duplicates and backwards timestamps start new segments. Every score carries a
source/segment/row key and a stable sample ID. The first W−1 rows are warmup and
excluded from primary metrics, including for instantaneous baselines.

The input scaler uses train data. Neural/PCA residual scales use reconstruction
residuals of the actual fitted estimator/best checkpoint on train data. Thresholds
and onset channel limits use only `val_threshold`. Missing calibration data is a
failure. The default quantile is 0.999. Optional matched event FAR uses the same
alarm machine and an explicitly enumerated candidate grid; event FAR is not point
exceedance rate. HAI label-test2 row alignment is allowed only for the registered
original file hashes, matching rows, bounded minute offsets and summary check.

Telemetry Parquet files contain no attack labels. `evaluation_labels.parquet` is
loaded only by benchmark code. Source paths in the prepared index are relative.
A bundle contains no original data paths. A factory's timestamp timezone is used
only when explicitly declared; HAI's unknown source timezone is not mislabeled UTC.

## Training and resume

Validation reconstruction MSE controls early stopping. Every strict best is saved;
patience is reset only by improvement beyond the absolute-plus-relative delta.
Nonfinite loss/gradients/validation are failures, not early stopping. Best weights
are restored before scoring. Epochs start at 1 in exported histories.

`latest.ckpt` includes optimizer/scheduler states, complete history, best/reference
state, RNG and sampler states, feature order/mask, scaler/data/config fingerprints.
An interrupted matching task resumes the remaining total budget. A completed early
stop does not restart. Physical batches accumulate to the same effective batch of
128, including correct weighting of the tail group.

Each task writes a receipt of input and output hashes. Missing or modified outputs
invalidate reuse. Failed tasks preserve errors and cause dependent tasks to be
skipped while independent tasks continue. OS-held file locks prevent concurrent
writers and are released automatically on process exit. Use a new run ID for a
new protocol; a changed resumed stage updates its outputs and downstream tasks.

## Explanations, policies and business variants

R01 uses residuals, R02 actual input gradients, R03 actual channel occlusion with
train-normal median/mode replacement, R04 fusion without graph, and R05 full
fusion. Evidence is produced at first activation and every five event-time seconds
while active. It does not read future windows or attack endpoints. All 86 HAI
features are ranked. Stability is the frequency of recent top-three membership,
not a fixed confidence number. `margin` is an evidence margin, not a probability.

Strict direct observable attack targets and weak associated-loop targets are
reported separately. End-to-end denominators include missed eligible attacks;
conditional results include detected eligible attacks. Snapshots are at actual
alarm time plus 0/5/10 seconds. Preexisting alarms are explicitly identified.

P00 offers no action; P01 routes a fixed conservative rule using current physical
values; P02 routes inspection advice using RCA; P03 adds evidence refusal and
qualification gates. Low evidence can withhold specific advice while preserving a
generic inspection workflow. B00–B07 are configured in `config/ablations/business.yaml`
and execute the affected runtime anew against the same frozen scores. B01 removes
alarm consolidation; B07 removes advice cooldown. Suppressed duplicates and actual
repeated offers are different fields. B04/B05 are offline-only switches.

The default HAI catalog has inspection actions only. Surrogates fit contiguous
one-step state/control transitions before sampling and validate 1-/5-step errors
against persistence using isolated uncertainty data. An independent Ridge setting
is retained as an evaluator. Prediction does not establish causal action response;
all effect rows are `NOT_APPLICABLE` when response qualification is absent. B05 can
therefore be equivalent to B00 in HAI, and that is reported without invented gains.

N01 rewires the graph while preserving degree; N02 permutes targets within the
observable pool; N03 permutes onset ownership; N04 reports PA-F1 only as an appendix
comparison; N05 dynamically perturbs label fields and filename on a declared
prefix. A negative control is not a proof of novelty or absence of all leakage.

## Metrics and uncertainty

Primary point metrics use unadjusted predictions and standard average precision.
PR-AUC and AP have distinct names. One-to-one interval matching maximizes match
count before IoU and deterministic tie breaking. Any-overlap attack coverage is a
separate metric. A single long alarm cannot count as many one-to-one detections.
Missed delays are empty, never zero. FAR uses effective normal hours.

The author eTaPR source is pinned under `vendor/etapr/`; only optional plotting
imports are patched. Empty-range cases are guarded because that upstream release
raises on them. Segment-weighted eTaPR columns explicitly name their aggregation.
VUS-PR remains unavailable until its author implementation is installed and
validated; the historical approximation has been removed.

Bootstrap uses a fixed seed and 2,000 resamples. Point-F1 intervals stratify attack
clusters and nonoverlapping normal blocks by source. RCA and business comparisons
use paired event differences within seed. Seed mean/std is separate. The current
implementation does not claim bootstrap intervals for every metric. Intervals
from a smoke subset are diagnostic only.

## Reuse an explicitly declared checkpoint

`--skip-train` requires a JSON checkpoint manifest. For example:

```json
{
  "D07/seed_1": {
    "checkpoint": "runs/previous/models/D07/seed_1/best.ckpt",
    "sha256": "REPLACE_WITH_ACTUAL_SHA256",
    "training_summary": "runs/previous/models/D07/seed_1/training_summary.json"
  }
}
```

Every requested neural model/seed must be listed, and feature order, mask, scaler,
data fingerprint and seed must match. A legacy checkpoint without those fields is
not silently certified. A one-model subset cannot stand in for a full matrix.
