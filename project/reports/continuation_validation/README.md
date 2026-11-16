# Continuation validation evidence

Recorded on 2026-11-16. This is a real-data integration and application validation
report. It does **not** certify an uncapped benchmark or Linux/Windows CUDA execution.
Week numbers refer to the plan's milestones, not elapsed calendar weeks.

## What ran

| Check | Observed result | Evidence |
|---|---|---|
| Contract/integration tests | 29 passed | [unit_tests.txt](unit_tests.txt) |
| Clean environment MPS matrix | 121 tasks succeeded, three stochastic seeds, real HAI, 384 rows per source/role | [final_pipeline.txt](final_pipeline.txt), [run summary](final_run_summary.json) |
| Actual neural early stopping | Both models × seeds 1/7/21 stopped at epoch 5 of 6; downstream tasks continued | [training history](../../runs/continuation_final/tables/training_history.csv), [training summary](../../runs/continuation_final/tables/training_summary.csv) |
| Identical-run resume | All 121 tasks reused after output-hash validation | [resume_pipeline.txt](resume_pipeline.txt), [final_resume_pipeline.txt](final_resume_pipeline.txt) |
| Final MPS exporter integration | Changed analysis stages reran successfully; all trained weights reused | [final_export_integration.txt](final_export_integration.txt) |
| CPU/MPS bundle agreement | 32 windows; score/evidence/gradient tolerance passed; zero prediction disagreements | [cpu_mps_agreement.json](cpu_mps_agreement.json) |
| MPS app workflow and resume | 560 unique frames; two recovered, closed cases | [app_mps_case_workflow.json](app_mps_case_workflow.json), [replay](app_mps_replay.txt), [resumed workflow](app_mps_workflow_resume.txt) |
| CPU app workflow | 560 frames; observed recovery; operator workflow and closure | [case_workflow_recovery.json](case_workflow_recovery.json) |
| Shell wrapper equivalence | External cwd and spaces/Unicode/`!`; error exit code preserved | [wrappers.json](wrappers.json) |
| Legacy checkpoint audit | Finite forward output, but rejected for v2 reuse due to missing authenticated provenance/mask | [legacy_checkpoint.json](legacy_checkpoint.json) |
| Bundle package | All 16 payload files hash-verified | [package_check.json](package_check.json) |
| Result package | All 900 payload files hash-verified | [results_package_check.json](results_package_check.json) |
| CPU sensitivity analysis | 85 threshold rows, 12 coverage rows; 248 protected input artifacts unchanged | [analysis_exports.json](analysis_exports.json) |
| CPU exporter integration | D01/D02 smoke pipeline, nine tasks, including new analysis and figures | [cpu_exports_pipeline.txt](cpu_exports_pipeline.txt) |
| Full acceptance guard | Smoke correctly rejected as full | [full_gate_expected_rejection.txt](full_gate_expected_rejection.txt) |

The canonical run is [continuation_final](../../runs/continuation_final/run_manifest.json),
with [task receipts and status](../../runs/continuation_final/task_registry.csv) and
an [integrity audit](../../runs/continuation_final/integrity_audit.json).
The deployment bundle is [continuation_final_primary](../../bundles/continuation_final_primary/bundle_manifest.json).
It is preselected D07/seed 1, not the best test seed. Its validation scope is smoke.

Archives: [deployment bundle ZIP](../../bundles/continuation_final_primary.zip)
and [result ZIP](../../bundles/continuation_final_results.zip). The result archive
includes nested checkpoints, experiment details, tables, plots and manifests;
original/prepared telemetry and the Python environment are excluded. Use the
project checkout and platform guide alongside the deployment bundle.

The final 384-row replay prefixes contain no attack events. Earlier real runs,
including `continuation_release`, explicitly extended the replay to 1,900 rows per
source. That fixture includes one registered attack with an unobservable strict
target. Its weak-target/PR/ROC outputs are useful for testing export paths, not for
scientific comparison. No missing strict-target score is converted to success.

## Commands used for canonical validation

Run from the repository's `project` directory with its `.venv` activated:

```bash
python -m unittest discover -s tests -q
python scripts/reproduce_all.py --suite smoke --device mps --run-id continuation_final --seeds 1,7,21 --max-epochs 6 --patience 1 --min-delta-abs 10 --physical-batch-size 8 --export-csv --export-bundle
# The same protocol, with --resume, verified reuse of every task.
python scripts/reproduce_all.py --suite smoke --device mps --run-id continuation_final --seeds 1,7,21 --max-epochs 6 --patience 1 --min-delta-abs 10 --physical-batch-size 8 --export-csv --export-bundle --resume
python scripts/validate_bundle.py --bundle bundles/continuation_final_primary --device mps --output reports/continuation_validation/cpu_mps_agreement.json
python scripts/export_analysis.py --run-dir runs/continuation_final
python scripts/make_figures.py --run-dir runs/continuation_final --plot-config config/plots.yaml
python scripts/audit_run.py --run-dir runs/continuation_final
```

`min_delta_abs=10` deliberately tests the early-stop path; use the normal defaults
in the [platform guide](../../docs/PLATFORM_GUIDE.md) for full training. The saved
training stage keeps its authenticated source/input key and frozen weights. The
run manifest now records the finalized pipeline source, which was validated by
another MPS resume and then a complete 121-task reuse check. The run summary records
the original bundle source and the latest pipeline source separately. Supplemental
CPU analysis has its own source/output inventory; no postprocessing is presented
as another training run.

The `--require-full` audit is expected to fail on this run. Its failing result was
saved separately; the ordinary integrity audit was then restored to the successful
smoke-scope result. Do not interpret this expected failure as a full experiment.

For the application validation, a real bundle replayed the first 560 HAI test1
frames. The scripted commands in [mps_case_commands.jsonl](mps_case_commands.jsonl)
were then applied after replay to the same persisted session. They are local
operator workflow records. They do not alter measurements or cause the recorded
recovery. Manual interactive commands are documented in [APP_GUIDE.md](../../docs/APP_GUIDE.md).

## Remaining gates

1. Complete the uncapped three-seed full benchmark on at least one target platform.
2. Execute real Linux CUDA and Windows CUDA smoke, transfer, resume and app checks.
   Only macOS arm64 CPU/MPS was available for execution here.
3. Integrate and validate the author's VUS implementation before reporting VUS-PR.
4. Extend statistical intervals beyond the implemented point-F1 and paired
   RCA/business-event intervals where the final research report requires them.
5. Independently qualify action response before any physical-control or
   intervention-effect study. The supplied HAI deployment remains inspection-only.

All weekly implementation notes are in [log/](../../log/), one file per Week 13–20.
Installation and launch instructions for macOS MPS, Linux CUDA and Windows CUDA
are in [PLATFORM_GUIDE.md](../../docs/PLATFORM_GUIDE.md).
