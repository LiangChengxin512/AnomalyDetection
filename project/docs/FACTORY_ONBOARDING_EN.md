# Factory onboarding guide

This guide documents the continuation-v2 CLI. HAI defaults to inspection-only advice. Physical action-response evaluation is unavailable without separately qualified intervention evidence.


> **Document status:** This is the proposed user manual for the application described in this plan. The new commands and configuration files below still need to be implemented. Do not assume that the current code already supports them. After implementation, publish this appendix as `docs/FACTORY_ONBOARDING_EN.md` and check every command against the delivered software.

## A1. What the system does

The system watches measurements from your factory. It learns patterns of normal operation, raises an alarm when the incoming measurements look unusual, lists variables that may explain the alarm, and suggests a response from a registered action list.

An operator can acknowledge the alarm, reject a suggestion, request a review, record an inspection, and close the case. The system saves the whole history so that you can see what it observed and why it suggested a response.

An anomaly is not automatically a cyberattack. It may also come from a sensor fault, maintenance, a mode change, or a process problem. The root-cause list contains candidates to investigate; it does not prove a physical cause.

The standard application gives advice. It does not send commands to your PLC. A replay of a recorded file also does not change when you accept a suggestion. Any predicted result shown by a process model is a simulation and is labelled as such.

## A2. What you need to provide

Prepare these items before training a model for your factory:

| Item | What it means in everyday terms |
|---|---|
| Normal measurements | Records from periods when the plant was operating normally |
| A separate normal validation period | A later normal period used to check training and set alarm limits |
| Data to monitor or replay | A file containing measurements in time order |
| A variable list | The names, types, units, and equipment associated with each measurement |
| Sampling information | How often a complete row arrives, and which timezone the timestamps use |
| Optional process connections | Documented links between sensors, controllers, pumps, valves, and equipment |
| An action list | Suggestions that your process team has reviewed, with conditions and limits |

Normal data should cover the operating modes you want the system to recognise, including normal starts, stops, load changes, and maintenance modes where appropriate. A file containing only a few steady minutes is unlikely to represent the whole plant.

You do not need attack labels to run the monitor. Labels are useful later if you want to measure detection quality. Keep them in a separate evaluation file. Never include a label, fault name, attack ID, or expected root cause in the model input.

## A3. Choose the measurements before collecting files

Choose variables that are available during normal live operation. Keep the same meaning and units throughout a model version.

For example, you might choose:

- Boiler pressure, measured in bar.
- Water flow, measured in cubic metres per hour.
- Pump speed command, measured in percent.
- Pump running status, using 0 for stopped and 1 for running.

These are examples, not a required list. The HAI example has 86 input variables. Your factory can have a different number after training its own model.

Do not feed four variables into a model trained for 86 variables. Do not copy one sensor into several columns to make the input fit. A new variable set normally requires a new model, new scaling information, and new alarm limits.

## A4. Prepare a simple CSV file

For the first connection, use a CSV file. This makes it easy to inspect the data before adding a live connection.

Example:

```csv
timestamp,Pressure_kPa,Flow_m3h,PumpSpeed_pct,PumpRunning
2026-01-15T08:00:00Z,250.0,18.2,45.0,1
2026-01-15T08:00:01Z,250.4,18.1,45.0,1
2026-01-15T08:00:02Z,249.8,18.3,45.0,1
```

Use one row per sampling time. The example uses one row per second. The three rows above only explain the format; they are not enough for training.

Follow these rules:

1. Use a timestamp column and one column for each registered variable.
2. Use a decimal point in numbers. Keep text such as "bar" out of numeric cells.
3. Save the file as UTF-8.
4. Include a timezone offset in timestamps, or declare the source timezone in the site configuration.
5. Keep rows in time order. Report gaps and duplicate times instead of hiding them.
6. Keep missing values empty. Do not replace a missing pressure reading with zero.
7. Use the same variable meaning in training and monitoring. A pressure setpoint and a measured pressure are different inputs.

The system selects and orders features from the registered variable list. The display order of columns in your CSV does not change the model order. Missing required columns stop the run and produce a clear error.

## A5. Set the sampling rate correctly

The HAI example uses one sample per second and a 128-sample history window. At that rate, the window represents 128 seconds.

If your historian exports ten samples per second, decide how to convert them before training. Continuous measurements may use a documented average or last value; digital states normally use a last valid state. Different signals may need different rules.

If your source produces a row every five seconds, a 128-sample window represents 640 seconds. Do not label it as a 128-second window. Record the real sampling interval and train with that interval.

Training, alarm evaluation, process prediction, and live monitoring must agree on the time step. A file gap starts a new segment and a new warm-up period. The system must not join measurements across that gap as if no time had passed.

## A6. Create the site configuration

The configuration below maps your CSV names to stable internal names. All data and configuration paths in a site file are relative to the project directory, unless written as absolute paths. Exported model packages use paths relative to the package itself.

Create `config/sites/factory_a.yaml` for your site:

```yaml
schema_version: "1.0"
site_id: factory_a

input:
  format: csv
  timestamp_column: timestamp
  source_timezone: UTC
  sample_interval_seconds: 1
  missing_required_feature: stop
  gap_policy: new_segment

data:
  train_files:
    - data/factory_a/normal_train.csv
  normal_validation_files:
    - data/factory_a/normal_validation.csv
  replay_files:
    - data/factory_a/replay.csv

features:
  - name: BOILER_PRESSURE
    source_column: Pressure_kPa
    kind: continuous
    equipment_id: boiler_1
    source_unit: kPa
    unit: bar
    transform: {scale: 0.01, offset: 0.0}
    hard_bounds: {min: null, max: null}

  - name: WATER_FLOW
    source_column: Flow_m3h
    kind: continuous
    equipment_id: water_loop_1
    source_unit: m3_per_hour
    unit: m3_per_hour
    transform: {scale: 1.0, offset: 0.0}
    hard_bounds: {min: null, max: null}

  - name: PUMP_SPEED_COMMAND
    source_column: PumpSpeed_pct
    kind: continuous
    equipment_id: pump_1
    source_unit: percent
    unit: percent
    transform: {scale: 1.0, offset: 0.0}
    hard_bounds: {min: null, max: null}

  - name: PUMP_RUNNING
    source_column: PumpRunning
    kind: discrete
    equipment_id: pump_1
    source_unit: boolean
    unit: boolean
    allowed_values: [0, 1]
    transform: {scale: 1.0, offset: 0.0}

model:
  primary: TranAD_C_G_E
  window_samples: 128
  score_aggregation: topk
  score_topk: 3

process:
  graph: null

decision:
  physical_adjustment_enabled: false
  allow_inspection_advice: true
  action_catalog: config/action_catalog_factory_a.yaml
```

The pressure conversion in this example changes 250 kPa into 2.5 bar. Check every conversion with the person who owns the instrument data.

The null bounds are deliberate placeholders. They do not mean that every pressure or flow is safe. They mean that reviewed physical limits have not been supplied. Keep physical adjustment advice disabled until your process team provides and approves those limits.

The feature names and values above are illustrative. Do not copy the HAI graph or HAI action catalogue into an unrelated factory.

## A7. Add process connections and actions

You can start without a process graph. Detection still works, but the explanation should say that process connections are unavailable.

When you add a graph, each connection should have a source: a drawing, controller document, operating procedure, or an engineer's reviewed description. A connection means a documented process relationship; it does not automatically prove causality.

Start the action catalogue with inspection and review suggestions. For example:

```yaml
schema_version: "1.0"
actions:
  - id: inspect_boiler_pressure
    kind: inspect
    description_en: "Check the pressure transmitter and the boiler pressure loop."
    equipment_id: boiler_1
    applies_to: [BOILER_PRESSURE]
    required_tags: [BOILER_PRESSURE]
    control_tag: null
    approval_level: operator_review
    review_status: approved
    evidence_source: "Example only; replace with your reviewed operating procedure."
    cooldown_s: 30
    cost: 0.1
```

The approved field above illustrates the file format. Your team must replace the example procedure and review the real entry before use.

An adjustment suggestion needs more information: the real command tag, unit, allowed range, maximum change, direction rule, rate limit, required operating conditions, and rollback procedure. If a required value is missing, the system rejects the adjustment and explains why.

An inspection suggestion does not change a simulated pressure value. It creates a task for a person to check the equipment.

## A8. Install and check the environment

Use Python 3.10 or 3.11 and a separate Python environment for this project. Install the platform-specific PyTorch build first; see [PLATFORM_GUIDE.md](PLATFORM_GUIDE.md). The project documents dependency versions and platform instructions under `environment/`.

Typical environment setup:

```bash
# macOS or Linux, from the project directory
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/app.txt
```

```bat
rem Windows CMD, from the project directory
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements\base.txt
python -m pip install -r requirements\app.txt
```

Install the tested PyTorch build using the delivered instructions for your platform. CUDA builds must match the supported driver setup. A Mac with Apple silicon uses MPS rather than CUDA.

Then run the environment check:

```bash
python scripts/preflight.py --site config/sites/factory_a.yaml --device auto
```

Check the displayed Python executable, data locations, installed packages, and selected device. If you request CUDA or MPS explicitly and it is unavailable, the check should stop. Do not assume that a run used the GPU just because the computer has one.

## A9. Train and export a model for your factory

Keep the normal training period and normal validation period separate in time. The system will divide validation data into model checking, prediction uncertainty checking, and alarm calibration sections.

command for macOS or Linux:

```bash
bash scripts/run_pipeline.sh --config config/experiment.yaml --site config/sites/factory_a.yaml --suite onboarding --device auto --seeds 1 --max-epochs 50 --patience 5 --run-id factory_a_initial --export-csv --export-bundle
```

Windows command:

```bat
scripts\run_pipeline.bat --config config\experiment.yaml --site config\sites\factory_a.yaml --suite onboarding --device cuda --seeds 1 --max-epochs 50 --patience 5 --run-id factory_a_initial --export-csv --export-bundle
```

The onboarding suite trains the configured primary model and prepares its deployment package. It does not require attack labels or claim that a full HAI benchmark was completed. Metrics that require labels remain unavailable until you supply a separate evaluation set.

Training may stop before 50 epochs if normal validation reconstruction stops improving. This is expected. The pipeline loads the best checkpoint and continues with calibration and packaging.

The final message should identify:

- The run directory.
- The actual number of training epochs and the best epoch.
- The model package directory and its bundle ID.
- Which capabilities are available.
- Which prediction or action capabilities were disabled, and why.

Do not choose the package with the best test attack score after trying many seeds. Use the preselected seed or a model choice based on the declared validation protocol.

## A10. Run the monitor

Use the package path printed by the pipeline. The following path is an example:

```bash
python -m src.app.cli --bundle bundles/factory_a_primary --site config/sites/factory_a.yaml --input data/factory_a/replay.csv --device auto --speed 1
```

Use `--speed 5` for faster replay or `--step` to advance manually. Replay speed changes waiting time, not the sampling interval used by the model.

During startup the application checks that the package, features, units, window length, scaling data, and alarm limits agree. It should stop if an important file is missing or incompatible.

A normal screen should show:

- Factory and package ID.
- Current data timestamp and replay position.
- Watched variables with units and quality flags.
- Anomaly score and fixed alarm threshold.
- Alarm state and any warm-up message.
- Current candidates, advice, and case status.

Watch the raw physical measurements as well as the anomaly score. A large score does not replace an approved physical operating limit.

## A11. Handle an alarm and finish the case

When an alarm opens, read the candidate variables and the explanation. Then use the available commands:

| Command | Purpose |
|---|---|
| `acknowledge` | Record that someone has seen the alarm |
| `reject-advice` | Reject a suggestion and record the reason |
| `escalate` | Move the case to a higher review level inside the application |
| `add-note` | Record an inspection or observation |
| `close-case` | Close a reviewed case with a reason |
| `export` | Save the session tables for inspection |

Escalation in the standard CLI updates the local case record. It does not email or message someone automatically.

If measurements return to the recovery condition, the application can mark "recovery observed." This does not prove that a suggestion caused the recovery. Keep the recorded measurements and the operator's actions separate.

If the input stops or a required sensor disappears, the application should show a data-quality hold. It must not display "normal operation" simply because no new rows arrived.

## A12. Find the results and change charts

Each experiment run has its own directory under `runs/`. Each monitor session has its own directory under `app_runs/`.

Useful exports include:

| File | What you can use it for |
|---|---|
| `training_history.csv` | Plot training and validation loss |
| `metrics_detection.csv` | Compare anomaly detectors |
| `metrics_rca.csv` | Compare candidate ranking methods |
| `metrics_policy.csv` | Inspect advice coverage, rejection, and policy measures |
| `metrics_ablation.csv` | Compare business capabilities with and without one module |
| `policy_decisions.csv` | Read each suggestion and its reason |
| `rootcause_candidates.csv` | Inspect variable rankings and evidence |
| `case_actions.csv` | Review acknowledgements, notes, and case changes |
| `plot_data/*.csv` | Redraw the exact data used by the supplied figures |

Change colours, labels, chart size, or display order in `config/plots.yaml`. The redraw command is:

```bash
python scripts/make_figures.py --run-dir runs/factory_a_initial --plot-config config/plots.yaml
```

A chart-style change should not retrain the model or recalibrate alarms. If you calculate new metrics from existing scores, save them with a new analysis configuration rather than overwriting the original result.

## A13. Move the system to another computer

Copy the project code and the complete model package. Install the documented environment on the new computer. Keep the input files in an accessible local directory and update the site data paths if needed.

Do not copy only the checkpoint. The scaler, residual scales, feature order, data types, alarm limits, graph, and action rules are part of the deployed model.

The package stores CPU-loadable parameters and supports device selection for MPS, CUDA, or CPU. The validation report distinguishes platforms actually tested from pending hardware checks. Small numeric differences may occur across backends. The delivered validation report should state the tested devices and tolerances.

A package trained for Factory A is not automatically suitable for Factory B. Reusing it requires matching variable meanings and preprocessing, plus a separate normal-data check. A different feature set requires retraining.

## A14. Connect live factory data later

Start by exporting a recent historian CSV and checking the offline replay. Once the format is reliable, add a source adapter that emits the same TelemetryFrame records.

An OPC UA, MQTT, historian, or file-tail adapter must provide:

1. A timestamp for each complete sample.
2. The registered variables in their declared units.
3. Sensor quality or missing-value flags.
4. A stable equipment/site identity.
5. Correct handling of delayed, duplicate, and out-of-order records.

The model and policy should not need to know which transport carried the measurements. Live connectors are an extension after CSV onboarding; they are not promised as already implemented.

Use read access for the standard monitor. Sending commands to real equipment requires a separately designed and reviewed control integration, beyond this application's advisory workflow.

## A15. Common problems

| Message or symptom | What to check |
|---|---|
| Required feature missing | Compare source_column names with the CSV header; do not fill a missing sensor with zero |
| Feature order or package mismatch | Use the site profile that belongs to the package, or train a new package |
| Timestamps cannot be parsed | Check timestamp format, timezone, empty values, and duplicate times |
| Old absolute path not found | Regenerate processed indexes with the current data root; do not rely on another computer's /home path |
| Parquet engine missing | Install the documented pyarrow version in the Python environment shown by preflight |
| CUDA/MPS unavailable | Check the selected Python/PyTorch build and the platform instructions |
| Too many alarms | Inspect sensor quality, units, operating-mode coverage, and the normal calibration report before changing limits |
| Model stopped early | Read training_summary; early_stopping with a valid best checkpoint is a successful training outcome |
| Advice rejected | Read the recorded reason: missing input, low evidence, unreviewed action, unsupported state, or uncertain prediction |
| All actions are inspection suggestions | Physical adjustments may be disabled or lack qualified action-response evidence |
| Process-model gains are unavailable | Prediction qualification or action support failed; monitoring and case handling can still work |
| No alarm in a short demo | Keep the true model output; inspect a longer case or a documented miss rather than inserting a fake alarm |

Before updating the model, save the existing package and run report. Retrain and recalibrate as a new version using reviewed normal data. Do not let the application silently learn from unreviewed incidents or change its own alarm limits during a demonstration.



