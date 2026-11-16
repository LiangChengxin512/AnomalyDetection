#!/usr/bin/env python3
"""Reproduce the full pipeline from a single config.

Usage (from project/):

    python scripts/reproduce_all.py [--mock] [--skip-train] [--max-windows N]

--mock uses the deterministic EMA scorer instead of a trained checkpoint;
--skip-train assumes checkpoints/ already contains a trained model and
goes straight to replay. Every step writes into data/processed/,
artifacts/, reports/ and figures/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PY = sys.executable


def step(name: str, cmd: list[str]):
    print(f"\n=== [{name}] {' '.join(cmd)} ===", flush=True)
    r = subprocess.run(cmd, cwd=PROJECT)
    if r.returncode != 0:
        print(f"[reproduce] step {name} failed with code {r.returncode}")
        sys.exit(r.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--inject-signal", action="store_true",
                    help="mock only: inflate scores inside events (smoke test)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-windows", type=int, default=0)
    args = ap.parse_args()

    cfg = ["--config", args.config]

    step("validate_time", [PY, "-m", "src.data.validate_time"] + cfg)
    step("prepare", [PY, "-m", "src.data.prepare_hai"] + cfg)
    step("events", [PY, "-m", "src.data.split_events"] + cfg)
    step("scaler", [PY, "-m", "src.data.fit_scaler"] + cfg)
    step("graph", [PY, "-m", "src.data.build_graph"] + cfg)

    scores_tag = None
    if args.mock:
        mock_cmd = [PY, "-m", "src.detect.mock_scores", "--splits", "train,val,test"] + cfg
        if args.inject_signal:
            mock_cmd.append("--inject-signal")
        step("mock_scores", mock_cmd)
        scores_tag = "MOCK"
    else:
        ckpt = args.checkpoint
        if not args.skip_train:
            train_cmd = [PY, "-m", "src.models.train", "--device", args.device] + cfg
            if args.epochs:
                train_cmd += ["--epochs", str(args.epochs)]
            if args.max_windows:
                train_cmd += ["--max-train-windows", str(args.max_windows)]
            step("train", train_cmd)
        import yaml
        cfgd = yaml.safe_load(open(PROJECT / args.config))
        mname = cfgd["model"]["name"]
        ckpt = ckpt or str(PROJECT / "checkpoints" /
                           f"{mname}_{cfgd['dataset_version']}" / "best.ckpt")
        rep_cmd = [PY, "-m", "src.detect.replay", "--checkpoint", str(ckpt),
                   "--splits", "val,test", "--device", args.device] + cfg
        if args.max_windows:
            rep_cmd += ["--max-windows", str(args.max_windows)]
        step("replay", rep_cmd)
        scores_tag = mname

    val_scores = PROJECT / "artifacts" / f"scores_val_{scores_tag}.parquet"
    test_scores = PROJECT / "artifacts" / f"scores_test_{scores_tag}.parquet"
    step("calibrate", [PY, "-m", "src.detect.calibrate",
                       "--scores", str(val_scores)] + cfg)
    step("rootcause", [PY, "-m", "src.explain.rootcause",
                       "--scores", str(test_scores)] + cfg)
    step("surrogate_fit", [PY, "-m", "src.sim.fit_surrogate"] + cfg)
    step("surrogate_val", [PY, "-m", "src.sim.validate_surrogate"] + cfg)
    step("policy", [PY, "-m", "src.decision.shadow_policy",
                    "--scores", str(test_scores),
                    "--surrogate", str(PROJECT / "artifacts" / "surrogate_card.json")] + cfg)
    step("closed_loop", [PY, "-m", "src.sim.closed_loop",
                         "--scores", str(test_scores)] + cfg)
    step("bench_detection", [PY, "-m", "src.bench.run_detection"] + cfg)
    step("bench_rca", [PY, "-m", "src.bench.run_rca",
                       "--scores", str(test_scores)] + cfg)
    step("bench_policy", [PY, "-m", "src.bench.run_policy"] + cfg)
    step("ablation", [PY, "-m", "src.bench.ablation",
                      "--primary", scores_tag] + cfg)
    step("bootstrap", [PY, "-m", "src.bench.bootstrap",
                       "--scores", str(test_scores)] + cfg)
    step("figures", [PY, "scripts/make_figures.py"])
    print("\n[reproduce] pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
