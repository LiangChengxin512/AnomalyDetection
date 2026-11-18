# Current CLI instructions

The continuation-v2 entry points are documented in
[the platform guide](../docs/PLATFORM_GUIDE.md),
[experiment protocol](../docs/EXPERIMENTS.md), and
[application guide](../docs/APP_GUIDE.md).

From `project/`, activate the intended Python environment and run:

```bash
python scripts/preflight.py --site config/sites/hai2305.yaml --device auto
python scripts/reproduce_all.py --suite smoke --device auto --run-id first_smoke --export-csv --export-bundle
python scripts/reproduce_all.py --suite full --device auto --seeds 1,7,21 --max-epochs 50 --patience 5 --run-id full_run --export-csv --export-bundle
python -m src.app.cli --bundle bundles/full_run_primary --site config/sites/hai2305.yaml --input data/raw/hai-23.05/hai-test1.csv --device auto --speed 5
```

Use the bundle path printed by your actual run. Explicit `mps` and `cuda` settings
fail if unavailable. `auto` can select CPU; it does not prove GPU execution.
Equivalent shell/CMD launchers are in `scripts/`. The old instructions are preserved
as `legacy_cli_instructions.md` and do not describe v2 execution.
