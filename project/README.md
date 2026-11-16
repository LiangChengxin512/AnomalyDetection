# Industrial anomaly monitoring and advisory workflow

This project implements the continuation plan using a single Python pipeline and
shared experiment/application runtime. It trains TranAD and TranAD_C_G_E, compares
five statistical/ML baselines, evaluates causal explanation snapshots and business
variants, and exports a frozen model bundle for the English CLI monitor.

Start with [platform setup and launch commands](docs/PLATFORM_GUIDE.md).
See [experiment protocols](docs/EXPERIMENTS.md), [application commands](docs/APP_GUIDE.md),
[factory onboarding](docs/FACTORY_ONBOARDING_EN.md), and [CSV definitions](docs/CSV_DICTIONARY.md).
Implementation and actual validation are recorded separately in `log/week13.md`
through `log/week20.md`. Historical Weeks 1–12 and v1 artifacts are retained.
The [validation evidence index](reports/continuation_validation/README.md) records
29 passing tests, a real three-seed MPS smoke matrix, resume and application checks,
and the outstanding full-benchmark and CUDA hardware gates.

The default HAI catalog offers inspection workflows. Historical observations do
not change when advice is issued. A surrogate prediction does not establish an
action's causal effect. Unqualified effect metrics remain unavailable.

A smoke run tests software integration using an explicitly limited data budget.
It is not a full benchmark and must not be used to claim detection superiority.
Only a completed, uncapped three-seed full run can satisfy the full experiment gate.
