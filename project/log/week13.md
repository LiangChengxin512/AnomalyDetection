# Week 13 / P0 — Data and model correctness

Executed on 2026-11-16. Week numbers follow the continuation plan's logical
milestones; they do not represent eight elapsed calendar weeks.

Implemented a project-local dependency set and preflight with explicit interpreter,
package/build information and a real attention/convolution/top-k/gradient device
probe. CLI device selection now takes precedence over PROJECT_DEVICE. Explicit
unavailable accelerators fail. MPS required access outside the desktop sandbox;
the actual Mac supports MPS and executed real model operations.

Added stable source/segment/row keys, strict timestamp parsing, gap segmentation,
relative prepared indexes, label-free telemetry, and isolated normal validation
roles. Scaler fitting uses only training data. Input windows and surrogate
transitions never cross source or segment boundaries. Transition pairs are formed
before sampling. Physical values are retained separately from model-scaled inputs.

The adapter asserts [B,W,D] input and [B,D] reconstruction contracts for B=1 and
B>1. Checkpoints now explicitly preserve feature order and continuous/discrete
masks, load through CPU, and hash actual weight content.

The legacy best checkpoint loaded and produced finite [2,86] output, but its
metadata lacks authenticated feature/mask and validation-role bindings. Restoring
the historical scaler's mask changed the fixed-input output by up to
0.0036880374, confirming that the missing mask matters. It is not automatically
reused under v2. See `reports/continuation_validation/legacy_checkpoint.json`.

Validation: actual HAI preparation and MPS smoke training completed; temporal
boundary, shape, mask round-trip, device-priority and legacy-refusal tests passed.
Original raw data, checkpoints and historical artifacts were preserved.
