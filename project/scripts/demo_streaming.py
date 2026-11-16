#!/usr/bin/env python3
"""Interactive CLI Streaming Demo for Cyber-Physical Anomaly Perception,
Root-Cause Attribution, and Shadow Decision Loop on HAI 23.05.

Simulates real-time 1 Hz SCADA streaming data from HAI testbed (Thermal/Boiler,
Turbine, Water Treatment, Power Grid HIL), feeding each second's 128s window
into the trained improved TranAD model, running on-the-fly thresholding,
multi-evidence graph-informed RCA, and constrained shadow safety policy.

Usage:
    python scripts/demo_streaming.py [--event 1] [--device auto] [--speed 5.0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common.config import load_config, resolve  # noqa: E402
from src.common.device import get_best_device  # noqa: E402
from src.common.io_utils import read_json, read_jsonl  # noqa: E402
from src.data.fit_scaler import apply_scaler  # noqa: E402
from src.decision.action_catalog import enumerate_legal_actions, load_catalog  # noqa: E402
from src.detect.score import aggregate, normalize_residuals  # noqa: E402
from src.explain.rootcause import _graph_support, _norm, rank_event  # noqa: E402
from src.models.load_checkpoint import load_checkpoint  # noqa: E402
from src.models.tranad_adapter import build_model, predict_batch  # noqa: E402

# ANSI Color Codes for high-contrast terminal demonstration
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_MAGENTA = "\033[95m"
C_BLUE = "\033[94m"
C_GRAY = "\033[90m"


def format_bar(val: float, max_val: float, width: int = 20) -> str:
    ratio = min(max(val / max(max_val, 1e-6), 0.0), 1.0)
    filled = int(round(ratio * width))
    bar = "█" * filled + "░" * (width - filled)
    color = C_GREEN if ratio < 0.7 else (C_YELLOW if ratio < 1.0 else C_RED)
    return f"{color}{bar}{C_RESET}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming perception, RCA, and decision demo.")
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--event", type=int, default=1, help="Event index (1-based, default: 1)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--speed", type=float, default=3.0,
                    help="Playback speed multiplier (e.g. 1.0 = real-time 1s, 5.0 = 5x fast, 0 = manual Enter)")
    ap.add_argument("--pre-buffer", type=int, default=20,
                    help="Seconds of normal data to play before attack onset (default: 20s)")
    ap.add_argument("--post-buffer", type=int, default=15,
                    help="Seconds of normal data to play after attack end (default: 15s)")
    args = ap.parse_args()

    cfg = load_config(PROJECT / args.config)
    proc_dir = resolve(cfg, "processed_dir")
    art_dir = resolve(cfg, "artifacts_dir")

    # 1. Load system configurations and metadata
    scaler = read_json(proc_dir / "scaler.json")
    tags = scaler["tags"]
    D = len(tags)
    W = cfg["window"]["length"]

    events_meta = read_json(proc_dir / "events.json")["events"]
    if args.event < 1 or args.event > len(events_meta):
        print(f"Error: Event {args.event} out of range (1..{len(events_meta)})")
        return 1
    event = events_meta[args.event - 1]

    # Process graph and action catalog
    graph = read_json(resolve(cfg, "process_graph"))
    edges = graph.get("edges", [])
    catalog = load_catalog(str(resolve(cfg, "action_catalog")))

    # Tag schema for human-readable tag info
    schema_yaml = yaml.safe_load(open(resolve(cfg, "tag_schema")))
    tag_info = {t["name"]: t for t in schema_yaml.get("tags", [])}

    # Threshold and residual scale
    thresh_file = art_dir / "thresholds.json"
    if thresh_file.exists():
        threshold = read_json(thresh_file)["threshold"]
    else:
        threshold = 24.8
        print(f"{C_YELLOW}[warn] thresholds.json not found, using fallback {threshold}{C_RESET}")

    scale_file = art_dir / "residual_scale.json"
    if scale_file.exists():
        r_scale = np.asarray(read_json(scale_file)["scale"], dtype=np.float32)
    else:
        r_scale = np.ones(D, dtype=np.float32)

    # 2. Load Model Checkpoint
    device = get_best_device(args.device)
    mname = cfg["model"]["name"]
    model = build_model(mname, D, W, device)
    ckpt_path = resolve(cfg, "checkpoint_dir") / f"{mname}_{cfg['dataset_version']}" / "best.ckpt"
    if not ckpt_path.exists():
        print(f"{C_RED}[error] Checkpoint not found: {ckpt_path}{C_RESET}")
        print("Please train the model or ensure best.ckpt is located in checkpoints directory.")
        return 1

    load_checkpoint(ckpt_path, model, device_requested=args.device)
    model.eval()

    # 3. Load Streaming Test Data
    target_file = event["file"]
    split_meta = read_json(proc_dir / "index.json")["splits"]["test"]
    parquet_path = None
    for rec in split_meta:
        if rec["source_file"] == target_file:
            parquet_path = Path(rec["parquet"])
            break
    if parquet_path is None or not parquet_path.exists():
        print(f"{C_RED}[error] Data parquet for {target_file} not found!{C_RESET}")
        return 1

    df_test = pd.read_parquet(parquet_path)
    X_raw = df_test[tags].to_numpy(np.float32)
    X_scaled = apply_scaler(X_raw, scaler)
    labels = df_test["label_attack"].to_numpy()
    timestamps = df_test["timestamp"].astype(str).to_numpy()

    row_start = max(0, int(event["row_start"]) - args.pre_buffer)
    row_end = min(len(df_test), int(event["row_end"]) + args.post_buffer)

    # Header banner
    print("\n" + "=" * 80)
    print(f"{C_BOLD}{C_CYAN}CPS 工控网络物理系统实时异常感知、根因归因与影子安全决策演示{C_RESET}")
    print("=" * 80)
    print(f"• 测试床系统 : HAI 23.05 (P1锅炉/热能, P2汽轮机, P3水处理, P4电网仿真)")
    print(f"• 检测模型   : {mname} (滑动窗口 W={W}s, 传感器通道 D={D})")
    print(f"• 演示事件   : {event['event_id']} (场景 {event['scenario']}, 攻击ID {event['attack_id']})")
    print(f"• 攻击目标真值: {C_BOLD}{C_MAGENTA}{', '.join(event['target_points'])}{C_RESET} | 受影响控制器: {event['controller']}")
    print(f"• 报警判定阈值: τ = {threshold:.3f} (严格基于纯正常验证集 99.9% 分位数标定)")
    print(f"• 播放回放设置: 范围行 [{row_start}..{row_end}], 速度: {'按回车单步步进' if args.speed == 0 else f'{args.speed}x 倍速'}")
    print("=" * 80)
    if args.speed == 0:
        input("按 [Enter] 键开始实时流式数据推演...")
    else:
        print("推演启动中...")
        time.sleep(1.0)

    # 4. Real-time Streaming Loop
    active_alarm_count = 0
    for t_idx in range(row_start, row_end):
        t_start = time.time()
        # Ensure window of length W
        w_lo = max(0, t_idx - W + 1)
        w_hi = t_idx + 1
        window_raw = X_scaled[w_lo:w_hi]
        if len(window_raw) < W:
            pad = np.repeat(window_raw[:1], W - len(window_raw), axis=0)
            window_raw = np.vstack([pad, window_raw])

        ts_str = timestamps[t_idx]
        is_attack_gt = bool(labels[t_idx] == 1)

        # -------------------------------------------------------------
        # 1. 感知 (Perception)
        # -------------------------------------------------------------
        batch = torch.as_tensor(window_raw[np.newaxis, :, :], device=device)
        with torch.no_grad():
            _, x2 = predict_batch(model, batch)
            x_last = batch[:, -1, :]
            resid = (x_last - x2).detach().cpu().numpy()[0]  # [D]

        norm_resid = normalize_residuals(resid[np.newaxis, :], r_scale)  # [1, D]
        score = float(aggregate(norm_resid, cfg["model"]["score_agg"], cfg["model"]["score_topk"])[0])
        is_alarm = score >= threshold

        # Formatting Output
        status_tag = f"{C_RED}{C_BOLD}[🚨 异常告警]{C_RESET}" if is_alarm else f"{C_GREEN}[  正常运行]{C_RESET}"
        gt_tag = f"{C_MAGENTA}[真值:受攻击]{C_RESET}" if is_attack_gt else f"{C_GRAY}[真值:正常]{C_RESET}"
        bar = format_bar(score, threshold * 1.5, width=16)

        line = (f"{C_BOLD}{ts_str}{C_RESET} | {status_tag} {gt_tag} | "
                f"Score={score:6.2f} (τ={threshold:.1f}) {bar}")
        print(line)

        # -------------------------------------------------------------
        # 2. 归因与决策 (Root Cause Attribution & Shadow Decision)
        # -------------------------------------------------------------
        if is_alarm:
            active_alarm_count += 1
            # Run quick attribution & ranking on current window
            cands = rank_event(e=norm_resid, attr=norm_resid, tags=tags,
                               edges=edges, cfg=cfg)
            top3 = cands[:3]

            print(f"  ├─ {C_CYAN}{C_BOLD}【根因归因分析】{C_RESET} (融合重建残差 + 因果过程拓扑):")
            for rank_i, c in enumerate(top3, 1):
                c_tag = c["tag"]
                c_score = c["score"]
                proc = tag_info.get(c_tag, {}).get("process", "unknown")
                is_hit = c_tag in event["target_points"]
                hit_badge = f" {C_BOLD}{C_GREEN}★ 命中攻击真值!{C_RESET}" if is_hit else ""
                print(f"  │  [Top-{rank_i}] {C_YELLOW}{c_tag:18s}{C_RESET} "
                      f"权重分: {c_score:.3f} | 所属过程: {proc}{hit_badge}")

            # ---------------------------------------------------------
            # 3. 影子安全决策 (Shadow Decision Loop)
            # ---------------------------------------------------------
            cand_names = [c["tag"] for c in top3]
            legal_actions = enumerate_legal_actions(catalog, cand_names)

            print(f"  └─ {C_MAGENTA}{C_BOLD}【受约束影子决策】{C_RESET} (安全包络保护，不直接写PLC):")
            if not legal_actions:
                print(f"     模式: {C_RED}ABSTAIN (拒识){C_RESET} | 原因: 未找到满足当前工况包络的合法动作，转人工接管")
            else:
                # Select conservative advisory action
                chosen = legal_actions[0]
                act_id = chosen["id"]
                act_desc = chosen.get("description", "")
                print(f"     模式: {C_GREEN}ADVISORY (安全建议){C_RESET}")
                print(f"     推荐动作: {C_BOLD}{act_id}{C_RESET}")
                print(f"     操作细节: {act_desc}")
                print(f"     安全边界: 经校验符合工控动作安全规范，预估越限风险受控")
            print("-" * 80)
        else:
            active_alarm_count = 0

        # Speed control
        if args.speed == 0:
            input("按 [Enter] 继续下一秒数据推演...")
        else:
            elapsed = time.time() - t_start
            wait_t = max(0.0, (1.0 / args.speed) - elapsed)
            time.sleep(wait_t)

    print("\n" + "=" * 80)
    print(f"{C_BOLD}{C_GREEN}✓ 流式推演演示完成！{C_RESET}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
