# Recover the existing Windows CUDA run

The supplied log completed all six neural fits and their score/detection stages.
The first failure is B01_seed_1 (cuBLAS execution failure), followed by illegal
memory accesses. The log alone does not establish GPU out-of-memory as the cause.

Copy the updated scripts/reproduce_all.py to the Windows checkout. Preserve the
entire runs/hai_full_windows directory, including receipts, models and scores.
Start a fresh Python process from PowerShell in project:

```powershell
.\scripts\run_pipeline.bat --suite full --device cuda --physical-batch-size 128 --replay-physical-batch-size 48 --run-id hai_full_windows --export-bundle --resume --fail-fast
```

The training value 128 deliberately matches the completed training contract. The
new replay value 48 limits downstream occlusion batches without changing fitted
model/score task keys. Effective training batch size remains 128. Completed tasks
are reused only when their input and output hashes match. Changing the old
--physical-batch-size to 48 would invalidate fit tasks and can restart training.
For a NEW training run, --physical-batch-size 48 is appropriate; sample-weighted
accumulation retains the configured effective batch of 128.

The replay override is recorded in resolved_config.yaml and replay_execution.json.
It invalidates business replay stages whose execution setting changed, including
previously completed B00, but not completed fitting/scoring. A failed replay stage
restarts at its beginning, not at an individual row. Completed training checkpoints
and normal calibration results remain available. No partial failed CUDA step is
resumed. --fail-fast stops at the first error rather than running more stages in
the failing process.

B01 uses single-window input gradients, not batched occlusion; reducing the replay
batch cannot guarantee that B01's cuBLAS error disappears. If it fails again, start
a new process with synchronous CUDA diagnostics and save its first failure:

```powershell
$env:CUDA_LAUNCH_BLOCKING = '1'
.\scripts\run_pipeline.bat --suite full --device cuda --physical-batch-size 128 --replay-physical-batch-size 48 --run-id hai_full_windows --export-bundle --resume --fail-fast
Remove-Item Env:CUDA_LAUNCH_BLOCKING
```

Inspect runs/hai_full_windows/experiments/B01/seed_1/failure.txt (or the first
reported failing task). The synchronous traceback and GPU/driver details are
needed to distinguish memory pressure from another CUDA kernel failure. This
recovery change has CPU regression coverage; Windows CUDA execution must be
verified on the affected computer.
