# Week 7 — Attribution and root-cause ranking

## Done
- `src/explain/attribution.py`: residual-normalized channel evidence +
  gradient/IG hook support for real checkpoints.
- `src/explain/rootcause.py`: candidate score = calibrated mix of residual,
  attribution, onset order, graph support, cross-window stability; emits
  `artifacts/rootcause.jsonl` with full evidence decomposition,
  predecessor paths, confidence, abstain flags (17,077 detected events in
  mock run).
