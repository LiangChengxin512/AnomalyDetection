# Week 14 / P1 — Early stopping and orchestration

Executed on 2026-11-16 as a logical continuation milestone.

Implemented independent strict-best and patience-reference state, total epoch
budgets, sample-weighted gradient accumulation, finite-value failure handling,
atomic checkpoints, best restoration and complete optimizer/scheduler/RNG/sampler
resume. Training histories are one-based and retained across resumes.

The deterministic validation sequence 0.50, 0.40, 0.39, 0.391, 0.392, 0.393 stops
at epoch 6 with best epoch 3. A real optimizer/checkpoint test exercised that
monitor sequence, and a dependent calibration task was called after successful
early stopping. An interrupted four-epoch training test resumed to exactly four
epochs and produced tensor-identical weights to uninterrupted execution. NaN
validation failed instead of being reported as early stopping.

The Python pipeline now owns task dependencies, isolated run paths, receipts,
output-hash validation, failure aggregation and exclusive OS-held run locks.
Independent tasks continue after failures. Required missing tasks remain visible.
Shell and CMD wrappers forward arguments, use the project directory and preserve
exit codes. Shell configuration equivalence and nonzero-exit propagation were
executed from an external path containing spaces, Unicode and `!`; see
`reports/continuation_validation/wrappers.json`. Windows execution remains pending.

The full registry and normal validation split protocol are serialized with each
run. Historical v1 command modules explicitly reject the v2 config instead of
mixing old artifact conventions into new runs.
