# Week 20 / P7 — Release validation and remaining acceptance

Executed on 2026-11-16 as a logical continuation milestone.

Status: implementation and local real-data validation delivered; the plan's final
full-benchmark and three-platform acceptance remains incomplete.

## Executed validation

- Created an independent project `.venv` with Python 3.10.20 and PyTorch 2.2.2.
  Direct and transitive installed versions are saved in
  `requirements/validated-macos.txt` and `requirements/macos-validation-lock.txt`.
- `runs/continuation_final` executed all 121 tasks with real HAI data on macOS MPS:
  D01–D07, R01–R05, P00–P03, B00–B07, N01–N05, preparation, qualification and
  bundle export. Its data cap is 384 rows per source/role. Seeds are 1, 7 and 21
  for stochastic methods; deterministic D01/D02 run once.
- All six neural tasks stopped successfully at epoch five of a six-epoch budget;
  downstream calibration, replay, experiments and packaging still executed. The
  deliberately large `min_delta_abs=10` is an early-stop diagnostic, not a model
  selection recommendation. An identical resume reused all 121 verified tasks.
- All 29 contract/integration tests passed, including interrupted optimizer/RNG
  resume, best-checkpoint restoration, masks/shapes, time boundaries, action gates,
  idempotent case storage, bundle relocation/corruption, and sensitivity recalculation.
- The same real bundle was evaluated on CPU and MPS for 32 fixed windows. With
  `atol=1e-4` and `rtol=1e-3`, scores, evidence and gradients agreed; maximum score
  difference was 0.0000242913, with zero threshold-prediction disagreements.
- Real MPS application replay processed 560 unique frames and two cases. Both
  cases recorded observed recovery and the acknowledge/reject/escalate/note/resolve/
  close workflow. Resuming the session did not duplicate input frames or cases.
  Recovery is an observation in historical data, not an effect of the advice.
- A bundle was relocated to a path containing spaces, Unicode and `!`; CPU loading,
  feature/unit mismatch rejection and corrupt-file rejection passed. Shell wrapper
  configuration equivalence and error exit propagation passed from an external cwd.
- The deployable bundle archive contains 16 inventoried files whose hashes were
  verified. Supplemental CPU analysis exported 85 threshold and 12 coverage rows,
  preserving all 248 protected model/score/surrogate files.
  `bundles/continuation_final_results.zip` packages 900 result files with verified
  hashes; original/prepared telemetry and the environment are excluded.
- A separate D01/D02 CPU smoke run completed all nine tasks, validating pipeline
  integration of the final analysis exporter. New figure types were rendered from
  frozen CSVs; training, coverage and PR curves were visually inspected.
- The finalized exporter/figure code also passed a real MPS pipeline resume. A
  second identical resume reused all 121 tasks. All previously trained weights
  remained unchanged through these final integration checks.
- The normal integrity audit passed. `audit_run.py --require-full` intentionally
  rejected this smoke run. The rejection is saved as evidence that a capped run
  cannot be accepted as a full benchmark.

Evidence and exact commands are indexed in
`reports/continuation_validation/README.md`. The canonical deployment is
`bundles/continuation_final_primary`, with its ZIP alongside it. Training evidence
retains its original stage source key and bundle source hash; the latest resumed
pipeline and supplemental CPU analysis record their own source hashes. Earlier
runs are retained rather than relabeled as final.

## Scope and outstanding acceptance

| Platform | Actual validation | Still required |
|---|---|---|
| macOS arm64 MPS | Clean environment; real three-seed smoke matrix; early stop/resume; attribution; CPU/MPS agreement; application workflow | Uncapped full experiment |
| Linux CUDA | Installation and launch commands supplied | Real CUDA training, attribution, transfer, resume and app execution |
| Windows CUDA | CMD wrappers statically reviewed; installation and launch commands supplied | Actual CMD/path/exit tests and real CUDA execution |

The final 384-row prefixes contain no attack events. The separate 1,900-row replay
fixture in `runs/continuation_release` includes one registered attack whose direct
target is unobservable; strict localization is therefore unavailable there too.
Neither fixture supports comparative detection or localization claims.

The full uncapped HAI benchmark and Linux/Windows CUDA hardware execution have not
been completed. VUS-PR remains unavailable pending author-implementation validation.
The HAI catalog remains inspection-only because action response is unqualified;
surrogate intervention gains are NOT_APPLICABLE. Statistical intervals currently
cover point F1 and paired RCA/business events, not every reported metric. Rich or
asynchronous display and live factory adapters remain application extensions.

Use `docs/PLATFORM_GUIDE.md` for installation, training-only, complete experiment,
resume, application and redraw commands on all three platforms. No full-benchmark
or three-platform completion is claimed by this milestone log.
