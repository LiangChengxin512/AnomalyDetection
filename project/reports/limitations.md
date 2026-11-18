# Limitations and safety boundaries

## Data

- HAI 23.05 is a **historical recording**, not a replayable HIL environment.
  There are no action-effect labels; any downstream "risk reduction" is
  computed inside a surrogate transition model fitted on normal data.
- `label-test2.csv` stores minute-resolution timestamps repeated per second.
  It is aligned to `hai-test2.csv` **by row position** (1:1, 230,400 rows) and
  validated against `summary_label2.txt` event windows. Do not join it by
  timestamp.
- Dataset license: HAI README states CC BY 4.0; technical manual v4.0 states
  CC BY-SA 4.0. Treat under the stricter terms; raw CSVs are not committed to
  this repository and must not be copied into supplemental material without
  checking the released license file.

## Model / evaluation

- The improved TranAD used here is the repository's TranAD-C/G/E family
  (reversible norm + temporal conv adapters + gating + association discrepancy
  + sparse memory + phase blending, Top-K aggregation). It is a perception
  component, not the claimed contribution.
- Thresholds are calibrated on normal validation data only
  (`quantile 0.995` of `hai-train4` scores). Test labels never influence the
  threshold, graph edges, explainer parameters, or policy tuning.
- PA-F1 is reported in the appendix column only; event-level and eTaPR metrics
  are the primary measures.
- Attention / gradient attribution is *sensitivity evidence*, not causality.
  Output fields use `root_cause_candidates`, never `causal_root_cause`.
  Graph-randomization, onset-shuffle and target-permutation negative controls
  are part of `src/bench/ablation.py` and must accompany any claim.

## Decision layer

- All actions are **shadow / advisory only**. Nothing writes to a PLC or
  physical device. `advice.jsonl` records `mode: advisory | abstain`.
- The action catalog (`config/action_catalog.yaml`) is a draft placeholder.
  Every action, its allowed conditions and bounds must be reviewed against the
  HAI technical manual / domain supervisor before results are reported.
- Surrogate metrics (`surrogate_only` scope in `metrics_policy.csv`) must
  never be described as real plant intervention effects. The surrogate is
  fitted on *normal* one-step transitions; it has not been validated under
  attack conditions, so attack-period predictions are extrapolations.

## Known gaps to resolve before thesis claims

- Real checkpoint training on GPU/MPS is pending; all current numbers are MOCK.
- Baselines (PCA/IF/OCSVM/USAD/AnomalyTransformer/GDN) are scaffolded in
  `src/bench/run_detection.py` but only the TranAD + mock paths are exercised.
- Efficiency metrics (latency, throughput, memory) are collected by
  `train.py`/`replay.py` logs but not yet consolidated into a table.
- Multi-seed runs: config supports `seed`; bootstrap CIs are per-event, not
  yet per-seed.
