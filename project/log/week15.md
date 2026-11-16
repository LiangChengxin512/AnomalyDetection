# Week 15 / P2 — Detector matrix and calibration

Executed on 2026-11-16 as a logical continuation milestone.

Implemented D01 Robust-Z/MAD with discrete frequency surprise, D02 causal EWMA,
D03 PCA reconstruction, D04 Isolation Forest, D05 One-Class SVM, D06 TranAD and
D07 TranAD_C_G_E. D01/D02 fit once. The other five methods use seeds 1, 7 and 21;
ML sampling keys and budgets are exported. Both neural models share Top-3 scoring,
W=128, an effective batch of 128, early stopping and the same normal-data protocol.

Every fitted model has isolated scaler/residual/threshold bindings. Neural/PCA
residual scales come from the fitted best model's training reconstruction.
Thresholds and channel onset limits use val_threshold only. No test fallback is
allowed. Calibration exports distinguish point exceedance, exceeding samples per
hour and consolidated false events per normal hour. All models exclude the same
segment-start warmup interval from headline metrics.

Executed real MPS smoke training: two epochs for each neural model at all three
seeds, plus all statistical/ML fits. The expanded run uses 384 training/validation
rows per source/role and an explicit 1,900-row prefix from each test file. It has
17 fitted detector/seed entries and 12 actual neural training epochs. This is
integration evidence, not a full benchmark or evidence of method superiority.

Run directories: `runs/continuation_mps_matrix` and the clean-environment attack
fixture `runs/continuation_release`. The later canonical validation run,
`runs/continuation_final`, uses a 384-row replay prefix and exercises early stopping
for all six neural tasks: a six-epoch total budget, minimum five epochs, patience
one and a diagnostic absolute improvement threshold of 10. All six stop at epoch
five, with best epoch five (30 actual neural epochs in total). These diagnostic
settings are not the recommended full-training configuration. The shorter final
prefix has no attack events; the expanded fixture is retained separately.

Full uncapped training remains to be run using `docs/PLATFORM_GUIDE.md`; no full-run
completion is claimed here.
