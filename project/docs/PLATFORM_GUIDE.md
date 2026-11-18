# Setup, training, experiments, and application launch

Run these commands from the repository's `project` directory. Use Python 3.10 or
3.11. The project does not require the parent repository's DGL, torchvision or
torchaudio dependencies. All three launchers forward arguments to the same Python
implementation. `PYTHON`, if set, must be an executable path without arguments.

## macOS with Apple silicon / MPS

```bash
cd /path/to/AnomalyDetection/project
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2
python -m pip install -r requirements/base.txt -r requirements/app.txt
python scripts/preflight.py --site config/sites/hai2305.yaml --device mps

# Real integration smoke, two epochs; limited data, not benchmark results.
bash scripts/run_pipeline.sh --suite smoke --device mps --run-id mps_smoke --export-csv --export-bundle

# All training + all required comparisons/ablations, without data caps.
bash scripts/run_pipeline.sh --suite full --device mps --seeds 1,7,21 --max-epochs 50 --patience 5 --physical-batch-size 16 --run-id hai_full_mps --export-csv --export-bundle

# Training only, when an application bundle is not yet needed.
bash scripts/run_train.sh --suite full --device mps --models D06,D07 --seeds 1,7,21 --max-epochs 50 --patience 5 --physical-batch-size 16 --run-id hai_train_mps

# Use the bundle path printed by the completed pipeline.
bash scripts/run_app.sh --bundle bundles/hai_full_mps_primary --site config/sites/hai2305.yaml --input data/raw/hai-23.05/hai-test1.csv --device mps --speed 5
```

The validation environment is recorded in `requirements/validated-macos.txt` and
per-run `preflight.json`. In sandboxed desktop tools, MPS may require access outside
the sandbox. A requested unavailable device fails; it never silently becomes CPU.

This workspace already contains a validated smoke bundle. To try the application
immediately after activating `.venv`, without training another model:

```bash
bash scripts/run_app.sh --bundle bundles/continuation_final_primary --site config/sites/hai2305.yaml --input data/raw/hai-23.05/hai-test1.csv --device mps --speed 20 --max-rows 560 --display-every 20
```

It is an integration/demo bundle trained on limited normal data; its manifest
explicitly identifies the smoke scope. Use a full-run bundle for later benchmark
and deployment validation.

## Linux with NVIDIA CUDA

```bash
cd /path/to/AnomalyDetection/project
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/base.txt -r requirements/app.txt
python scripts/preflight.py --site config/sites/hai2305.yaml --device cuda

bash scripts/run_pipeline.sh --suite smoke --device cuda --run-id linux_smoke --export-csv --export-bundle
bash scripts/run_pipeline.sh --suite full --device cuda --seeds 1,7,21 --max-epochs 50 --patience 5 --physical-batch-size 128 --run-id hai_full_linux --export-csv --export-bundle
bash scripts/run_train.sh --suite full --device cuda --models D06,D07 --seeds 1,7,21 --max-epochs 50 --run-id hai_train_linux
bash scripts/run_app.sh --bundle bundles/hai_full_linux_primary --site config/sites/hai2305.yaml --input data/raw/hai-23.05/hai-test1.csv --device cuda --speed 5
```

CUDA 12.1 above is a concrete PyTorch 2.2.2 wheel option, not a guarantee that every
NVIDIA driver supports it. If needed, use the official CUDA 11.8 wheel option
(`cu118`) with a compatible driver. Check `nvidia-smi` and the actual device reported
by preflight. The PyTorch [previous-version installation matrix](https://pytorch.org/get-started/previous-versions/)
provides both wheel indexes; use the [official installer](https://pytorch.org/get-started/locally/)
when selecting another version and validate it as a new environment.

## Windows with NVIDIA CUDA (CMD)

```bat
cd /d C:\path\to\AnomalyDetection\project
py -3.10 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements\base.txt -r requirements\app.txt
python scripts\preflight.py --site config\sites\hai2305.yaml --device cuda

scripts\run_pipeline.bat --suite smoke --device cuda --run-id windows_smoke --export-csv --export-bundle
scripts\run_pipeline.bat --suite full --device cuda --seeds 1,7,21 --max-epochs 50 --patience 5 --physical-batch-size 128 --run-id hai_full_windows --export-csv --export-bundle
scripts\run_train.bat --suite full --device cuda --models D06,D07 --seeds 1,7,21 --max-epochs 50 --run-id hai_train_windows
scripts\run_app.bat --bundle bundles\hai_full_windows_primary --site config\sites\hai2305.yaml --input data\raw\hai-23.05\hai-test1.csv --device cuda --speed 5
```

In PowerShell, activate `.venv\Scripts\Activate.ps1` and invoke the same `.bat`
files with `& .\scripts\run_pipeline.bat ...`, or call Python directly:
`python scripts/reproduce_all.py ...` and `python -m src.app.cli ...`.
The batch wrappers disable delayed expansion, quote the interpreter and script
paths, use their own directory, and return Python's exit code. Avoid Windows path
lengths beyond your OS configuration; long-path support has not been hardware-tested.

## Resume, budgets, and outputs

Repeat a command with the same `--run-id` and add `--resume`. Verified outputs are
reused. Changed stage inputs invalidate the stage and its dependents. Completed
early-stopped tasks are successful. `--max-epochs` is a lifetime total, not additional
epochs. An altered training contract starts a fresh training attempt; use a new run
ID to keep distinct protocols completely separate.

Physical batch size controls memory. Effective batch size remains 128 through
sample-weighted gradient accumulation. No device automatically reduces feature
count, window size, or full-run data coverage. If an OOM occurs, choose a smaller
physical batch and a new run ID; partial failed optimizer steps are not resumed.

CSV is always exported. Run outputs are under `runs/<run_id>/`; the authoritative
status is `run_manifest.json` and task details are in `task_registry.csv`. Replot:

```bash
python scripts/make_figures.py --run-dir runs/hai_full_mps --plot-config config/plots.yaml
# CPU analysis of frozen scores/rankings; leaves weights and deployment thresholds unchanged.
python scripts/export_analysis.py --run-dir runs/hai_full_mps
# Package the explicit bundle or result directory, including nested files and hashes.
python scripts/package_artifacts.py --source bundles/hai_full_mps_primary --output bundles/hai_full_mps_primary.zip
```

Use the **printed bundle path**, particularly after a changed resumed run: immutable
older bundles are retained and a new bundle may receive a hash suffix. App data is
stored under `app_runs/<session_id>/`. Use `--step` in an interactive terminal for
manual replay, or `--speed 20 --display-every 20` for a faster demonstration.

Linux CUDA and Windows CUDA require real execution on those systems before claiming
platform validation. Their commands and wrappers are supplied; macOS checks do not
substitute for those hardware tests. See `log/week20.md` for the actual test matrix.
