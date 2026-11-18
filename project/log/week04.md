# Week 4 — Windowing, scaler, process graph

## Done
- `src/data/window.py`: sliding windows W=128, stride configurable; never
  crosses file/split boundaries; discrete-channel mask support.
- `src/data/fit_scaler.py`: feature-wise min-max [0,1] scaler fit on normal
  training data only (TranAD sigmoid convention). Reports 86 tags, 16
  discrete features. Saved to `artifacts/scaler.json`.
- `src/data/build_graph.py`: built documented process graph from the official
  NetworkX boiler graph + manual control-loop table ->
  `config/process_graph.json` (86 nodes, 56 edges, each with source/confidence).
