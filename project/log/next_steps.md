# Remaining acceptance work after continuation implementation

The Python pipeline, detector registry, shared runtime, bundle loader, CLI case
workflow, CSV exports, tests, and platform launchers are implemented. Weeks 13–20
record actual validation separately from scientific/platform acceptance.

1. Run the uncapped `full` suite on at least one target platform. Smoke runs do not
   satisfy this gate or support comparative research claims.
2. Execute real smoke training, resume, attribution, bundle transfer and application
   workflow on Linux CUDA and Windows CUDA. These machines are unavailable in the
   current macOS workspace. Windows wrapper behavior has only been statically reviewed.
3. Install and validate the author's VUS implementation before reporting VUS-PR.
   Until then its result remains unavailable; no approximation is substituted.
4. If physical adjustment advice or intervention gains are needed, obtain reviewed
   units/control mappings and independent action-response evidence. The HAI bundle
   remains inspection-only and its surrogate effect rows remain NOT_APPLICABLE.
5. Expand intervals to all required metrics for the final paper. Current intervals
   cover point F1 and paired event-level RCA/business comparisons. Threshold
   sensitivity and evidence-margin coverage exports now run from frozen artifacts;
   the risk axis still requires independent action-response qualification.
6. Optional application work: asynchronous explanations, live read-only adapters,
   an evaluation-only display overlay and a browser interface. Current CLI replay
   processes every row synchronously and records actual latency.

Run commands are in `docs/PLATFORM_GUIDE.md`; the final validation evidence is
summarized in `reports/continuation_validation/README.md` and `log/week20.md`.
Historical next steps are preserved as `legacy_next_steps.md`.
