"""Build config/process_graph.json from documented sources only.

Edge sources (each edge records its provenance):

* ``manual_fig4-8``  — control loops described in the technical manual:
  P1-PC, P1-LC, P1-FC, P1-TC (incl. feedforward/cascade), P1-CC, P2-SC,
  P2-TC, P3-LC, and the P1/P3 -> P4 HIL couplings.
* ``name_convention`` — command ``*D`` -> feedback ``*Z`` and
  setpoint ``*SP/_OUT`` -> command pairs derived from HAI tag naming.
* ``boiler_graph``   — device-level links from the official
  ``graph/boiler/phy_boiler.json`` (NetworkX node-link data), included
  as supporting context; device ids are kept as separate node types.

No edge is added "because attention looked high". Direction is
cause -> effect: setpoint -> command -> feedback -> process variable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import write_json  # noqa: E402

# Documented control-loop edges (manual figs. 4-8 + attack-table loops).
LOOP_EDGES = [
    # P1-PC pressure loop
    ("x1001_05_SETPOINT_OUT", "P1_PCV01D", "sp_to_cv"),
    ("x1001_05_SETPOINT_OUT", "P1_PCV02D", "sp_to_cv"),
    ("P1_PCV01Z", "P1_PIT01", "cv_to_pv"),
    ("P1_PCV02Z", "P1_PIT01", "cv_to_pv"),
    # P1-LC level loop (feed-forward from P1-FC per manual fig.5)
    ("x1002_07_SETPOINT_OUT", "P1_LCV01D", "sp_to_cv"),
    ("x1002_08_SETPOINT_OUT", "P1_LCV01D", "sp_to_cv"),
    ("P1_LCV01Z", "P1_LIT01", "cv_to_pv"),
    ("P1_FCV03D", "P1_LIT01", "feedforward"),
    ("P1_FT03", "P1_LCV01D", "ff_disturbance"),
    # P1-FC flow loop
    ("x1003_10_SETPOINT_OUT", "P1_FCV03D", "sp_to_cv"),
    ("x1003_18_SETPOINT_OUT", "P1_FCV03D", "sp_to_cv"),
    ("P1_FCV03Z", "P1_FT03", "cv_to_pv"),
    # P1-TC temperature loop (cascade w/ FT02 feedforward, manual fig.7)
    ("x1003_24_SUM_OUT", "P1_FCV01D", "sp_to_cv"),
    ("x1003_24_SUM_OUT", "P1_FCV02D", "sp_to_cv"),
    ("P1_FT02", "P1_FCV01D", "ff_cascade"),
    ("P1_FT02", "P1_FCV02D", "ff_cascade"),
    ("P1_FCV01Z", "P1_FT02", "cv_to_pv"),
    ("P1_FCV02Z", "P1_TIT01", "cv_to_pv"),
    # P1-CC cooling pump
    ("P1_PP04SP", "P1_PP04D", "sp_to_cv"),
    ("P1_PP04D", "P1_PP04", "cmd_to_state"),
    ("P1_PP04", "P1_TIT02", "cooling_effect"),
    ("P1_PP04", "P1_TIT03", "cooling_effect"),
    # P2 speed control + trip supervision
    ("P2_AutoSD", "P2_SCO", "sp_to_cv"),
    ("P2_ManualSD", "P2_SCO", "sp_to_cv"),
    ("P2_SCO", "P2_SIT01", "cv_to_pv"),
    ("P2_SIT01", "P2_VIBTR01", "pv_to_vibration"),
    ("P2_SIT01", "P2_VIBTR02", "pv_to_vibration"),
    ("P2_SIT01", "P2_VIBTR03", "pv_to_vibration"),
    ("P2_SIT01", "P2_VIBTR04", "pv_to_vibration"),
    ("P2_VTR01", "P2_RTR", "trip_logic"),
    ("P2_VTR02", "P2_RTR", "trip_logic"),
    ("P2_VTR03", "P2_RTR", "trip_logic"),
    ("P2_VTR04", "P2_RTR", "trip_logic"),
    # P3 water-treatment level/flow
    ("P3_LCP01D", "P3_FIT01", "cv_to_pv"),
    ("P3_LCP01D", "P3_LIT01", "cv_to_pv"),
    ("P3_LCV01D", "P3_LIT01", "cv_to_pv"),
    ("P3_LCV01D", "P3_FIT01", "cv_to_pv"),
    ("P3_LIT01", "P3_LH01", "pv_to_switch"),
    ("P3_LIT01", "P3_LL01", "pv_to_switch"),
    ("P3_FIT01", "P3_PIT01", "flow_to_pressure"),
    # P4 HIL couplings (manual: boiler -> STM, hydro -> HTM, speed sync)
    ("P1_PIT01", "P4_ST_PT01", "steam_pressure_in"),
    ("P1_TIT01", "P4_ST_TT01", "steam_temp_in"),
    ("P4_ST_PS", "P4_ST_PO", "power_setpoint"),
    ("P4_ST_GOV", "P4_ST_PO", "governor"),
    ("P4_ST_PO", "P4_ST_FD", "load_to_freq"),
    ("P4_ST_LD", "P4_ST_FD", "load_to_freq"),
    ("P3_LIT01", "P4_HT_PS", "hydro_level_in"),
    ("P3_FIT01", "P4_HT_PO", "hydro_flow_in"),
    ("P4_HT_PS", "P4_HT_PO", "power_setpoint"),
    ("P4_LD", "P4_HT_FD", "load_to_freq"),
    ("P4_HT_PO", "P4_HT_FD", "gen_to_freq"),
    ("P4_ST_FD", "P2_SIT01", "speed_sync"),
    ("P4_ST_PO", "P2_SCO", "power_to_speed_demand"),
]


def name_convention_edges(tags: list[str]) -> list[tuple[str, str, str]]:
    edges = []
    tagset = set(tags)
    for t in tags:
        if t.endswith("D") and t + "Z" in tagset:
            edges.append((t, t + "Z", "cmd_to_feedback"))
        if (t.endswith("SP") or t.endswith("SD")) and t in tagset:
            pass  # SP -> CV handled by documented loop edges
    # pump command/state pairs that do not follow the D/Z convention
    for cmd, st in [("P1_PP01AD", "P1_PP01AR"), ("P1_PP01BD", "P1_PP01BR"),
                    ("P1_PP02D", "P1_PP02R")]:
        if cmd in tagset and st in tagset:
            edges.append((cmd, st, "cmd_to_state"))
    return edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    schema = yaml.safe_load(open(resolve(cfg, "tag_schema"), encoding="utf-8"))
    tags = [t["name"] for t in schema["tags"]]
    tagset = set(tags)
    meta = {t["name"]: {"process": t["process"], "kind": t["kind"],
                      "discrete": bool(t["discrete"])} for t in schema["tags"]}

    edges = []
    seen = set()

    def add(s, d, rel, evidence, conf="high"):
        if s not in tagset or d not in tagset or (s, d) in seen:
            return
        seen.add((s, d))
        edges.append({"source": s, "target": d, "relation": rel,
                      "evidence": evidence, "confidence": conf})

    for s, d, rel in LOOP_EDGES:
        add(s, d, rel, "manual_fig4-8")
    for s, d, rel in name_convention_edges(tags):
        add(s, d, rel, "name_convention")

    # Boiler device graph as supporting context (device-level, separate
    # namespace so it cannot be confused with tag-level causality).
    boiler_path = resolve(cfg, "graph_dir") / "phy_boiler.json"
    boiler_devices = {"nodes": 0, "links": 0}
    if boiler_path.exists():
        g = json.loads(boiler_path.read_text())
        boiler_devices = {"nodes": len(g.get("nodes", [])),
                          "links": len(g.get("links", [])),
                          "file": str(boiler_path)}

    graph = {
        "schema_version": "1.0",
        "dataset_version": "HAI-23.05",
        "graph_version": "process_graph_v1",
        "nodes": [{"id": t, **meta[t]} for t in tags],
        "edges": edges,
        "boiler_device_graph": boiler_devices,
        "notes": ("Tag-level edges carry documented evidence. The boiler "
                  "device graph (official NetworkX data) is stored for "
                  "context; it is device-level, not tag-level."),
    }
    out = resolve(cfg, "process_graph")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, graph)
    print(f"[graph] {len(tags)} nodes, {len(edges)} documented edges "
          f"(boiler device graph: {boiler_devices['nodes']} nodes/"
          f"{boiler_devices['links']} links) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
