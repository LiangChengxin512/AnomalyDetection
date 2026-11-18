# Week 5 — Model adapter and checkpoint handling

## Done
- Copied the improved TranAD implementation from the parent repo into
  `src/models/tranad_lib.py` (self-contained; keeps checkpoint key
  compatibility for TranAD-C/G/E variants R,V,A,M,C,G,E).
- `src/models/tranad_adapter.py`: predict(batch) -> reconstruction +
  attention; `src/models/load_checkpoint.py`: state-dict load with version
  report and parameter hash.
- `src/common/device.py`: auto/cuda/mps/cpu selection per prompt spec
  (cuda > mps > cpu); float32 everywhere for MPS safety.
