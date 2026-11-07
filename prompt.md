在当前目录下创建一个project目录，在project目录下严格执行thesis_system_plan.md的计划。

在project下创建一个log目录，执行完每一个Week Plan需要用一个新的md文件用于记录这个Week做了什么事情

工作语言：英语，注释和系统前台显示内容都需要简单的英语表达

我的TranAD_C_G_E+TopK模型代码在AnomalyDetection目录下，如果找不到可以先用TranAD的代码代替，需要把模型代码复制到project下，之后只在project目录下开发系统

You are an expert ML Systems Engineer. Please implement the complete engineering codebase for the 12-week Master's thesis project plan (based on the provided `thesis_system_plan.md`).

### Critical Execution Rules:
1. **End-to-End Code Generation (Do Not Halt at Week 5/6)**:
   - You MUST implement the full codebase spanning Week 1 through Week 12.
   - Do NOT stop and wait for me to train models on a GPU.
   - For components downstream of perception (Weeks 7–12: root-cause ranking, shadow policy, surrogate modeling, baseline benchmarking, and metric generation), implement a lightweight mock/dummy scoring utility (`src/detect/mock_scores.py`). This allows end-to-end integration tests and pipeline sanity checks without requiring real model weights immediately.

2. **Hardware-Agnostic Device Management (Linux CUDA / Apple Silicon MPS / CPU)**:
   - Write all PyTorch model code and training loops to dynamically auto-detect and select the execution device:
     `device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))`
   - Explicitly support setting the device via environment variable or CLI argument (e.g., `--device auto|cuda|mps|cpu`).
   - Guard against MPS-specific edge cases (e.g., float64 casting fallback if unsupported).

3. **Standalone Shell Scripts (`run_train.sh` & `run_pipeline.sh`)**:
   - Provide clean, robust bash entry points:
     - `scripts/run_train.sh`: Dedicated script for me to run the heavy training separately on either Linux (NVIDIA CUDA) or macOS (Apple Silicon MPS). Includes parameters for batch size, epochs, device selection, and logging.
     - `scripts/run_pipeline.sh`: End-to-end pipeline script to run data validation -> windowing -> inference -> root cause analysis -> shadow decision -> evaluation -> plotting.
   - Make all shell scripts POSIX-compliant with proper error handling (`set -euo pipefail`).

4. **Architecture & Deliverables to Implement**:
   - **Week 1-4**: `fetch_hai.py`, `inspect_schema.py`, `validate_time.py`, `prepare_hai.py`, `window.py`, `fit_scaler.py`, and `build_graph.py` (generating `process_graph.json` and `tag_schema.yaml`).
   - **Week 5-6**: `tranad_adapter.py`, training engine, checkpoint loading, `score.py`, `calibrate.py` (for normal-val thresholds), and `mock_scores.py`.
   - **Week 7-9**: `attribution.py`, `rootcause.py` (multi-evidence ranking), `action_catalog.py`, `shadow_policy.py` (constrained MPC / abstain mode), and `closed_loop.py` (surrogate validation).
   - **Week 10-12**: `run_detection.py`, `run_rca.py`, `run_policy.py`, `ablation.py`, `bootstrap.py`, and `make_figures.py`.

Please structure the directory cleanly as specified in the plan, write modular and well-documented Python files, and provide clear CLI instructions on how I can trigger the training script later on my GPU/MPS machines. Save the clear CLI instructions in a md file in folder project/log. Also save into a md file what I need to do next step in order to finish the plan thesis_system_plan.md schedule after I run the tranning script