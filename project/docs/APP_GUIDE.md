# CLI monitor and case workflow

Start a completed bundle with `python -m src.app.cli --bundle <printed-path> --input
<csv> --site config/sites/hai2305.yaml --device mps --speed 5`. The `.sh` and `.bat`
launchers call the same module. `--plain` is accepted; the current display is plain
English text and does not require Rich.

Startup verifies file hashes, feature order/type mask, units and transforms,
scaler/residual/threshold bindings, window size, graph nodes and action references.
Missing files or mismatched site profiles fail. The application does not train,
refit thresholds, use mock scores, or display attack labels/targets.

The monitor shows timestamp, replay position, score, fixed threshold, alarm state,
candidate variables, inspection advice and case ID. Add `watch TAG` to show a raw
measurement and declared unit. Unknown HAI units remain `unknown` rather than
being invented. Replay speed changes wall-clock waiting only. Every row is processed.
`--display-every N` reduces console output without skipping inputs.

Commands are entered as complete lines in an interactive terminal:

| Command | Example | Effect |
|---|---|---|
| `pause` | `pause` | Pause reading new samples |
| `resume` / `start` | `resume` | Resume automatic replay |
| `step` | `step` | Process one row, then pause |
| `speed N` | `speed 20` | Change replay speed |
| `watch TAG` | `watch P1_PIT01` | Display a raw physical value |
| `acknowledge CASE REASON` | `acknowledge alarm:... Operator reviewed the alert` | OPEN → ACKNOWLEDGED |
| `reject-advice CASE REASON` | `reject-advice alarm:... Inspection needs a specialist` | Record refusal |
| `escalate CASE REASON` | `escalate alarm:... Supervisor review needed` | Record local escalation |
| `add-note CASE REASON` | `add-note alarm:... Transmitter inspection recorded` | Enter review |
| `close-case CASE REASON` | `close-case alarm:... Review completed` | Record resolution and close |
| `export` | `export` | Export session CSV/JSONL |
| `quit` | `quit` | Stop and export |

Escalation is a local case transition; no email or message is sent. Actions never
write PLC commands. A case requires acknowledgement or review before closure, and
all operator actions require an actor and reason. Recovery observed in historical
data is recorded separately and does not establish that advice caused recovery.
End of input is displayed as DATA_QUALITY_HOLD, not NORMAL.

Use `--session-id demo_1` to persist an identifiable session. Reusing that ID checks
the bundle and input hashes. The runtime rebuilds its causal state from the prefix;
SQLite sample uniqueness prevents duplicate cases and detection records. The
append-only source of audit records is the SQLite `frames` table; `audit.jsonl` is
a reproducible export of those records. Session exports include `cases.csv` and
`case_actions.csv` under `app_runs/<session_id>/`.

For a reproducible scripted demonstration, `--commands-file file.jsonl` applies
local workflow commands after replay. Each line has `case_id`, `command`, `actor`,
and `reason`. It does not inject an alarm or alter model output. For example:

```json
{"case_id":"REPLACE_WITH_REAL_CASE_ID","command":"acknowledge","actor":"demo_operator","reason":"Reviewed recorded alarm"}
{"case_id":"REPLACE_WITH_REAL_CASE_ID","command":"add-note","actor":"demo_operator","reason":"Recorded inspection finding"}
{"case_id":"REPLACE_WITH_REAL_CASE_ID","command":"close-case","actor":"demo_operator","reason":"Workflow review complete; no causal repair claim"}
```

The current implementation computes explanations synchronously. Its latency is
measured, and no input rows are dropped to hide that cost. Asynchronous explanation
queues, live historian/OPC UA/MQTT adapters, an evaluation overlay, and a browser UI
remain future extensions. Use offline CSV replay before developing a live adapter.
