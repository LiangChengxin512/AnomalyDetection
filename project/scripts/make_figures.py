#!/usr/bin/env python3
"""Generate thesis figures from artifacts.

Figures produced (figures/):
* score_timeline.png       — score vs threshold vs attack windows
* event_case.png           — per-channel evidence heatmap for one event
* rca_strategy.png         — Top-k hit by ranking strategy
* policy_risk.png          — surrogate risk by policy
* detection_summary.png    — event F1 / delay across evaluated models
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.events import binary_to_ranges  # noqa: E402
from src.common.io_utils import read_json, read_jsonl  # noqa: E402


def main() -> int:
    cfg = load_config()
    art = resolve(cfg, "artifacts_dir")
    fig = resolve(cfg, "figures_dir")
    fig.mkdir(parents=True, exist_ok=True)

    scores = sorted(art.glob("scores_test_*.parquet"))
    if scores:
        df = pd.read_parquet(scores[0])
        s = df["score"].to_numpy()
        y = df["label_attack"].to_numpy()
        thr = read_json(art / "thresholds.json")["threshold"] if (art / "thresholds.json").exists() else np.quantile(s, .999)
        t = np.arange(len(s))
        fig1, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
        ax[0].plot(t, s, lw=0.5, label="anomaly score")
        ax[0].axhline(thr, color="r", ls="--", lw=1, label="locked threshold")
        for a, b in binary_to_ranges(y):
            ax[0].axvspan(a, b, color="orange", alpha=0.25)
        ax[0].legend(); ax[0].set_ylabel("score")
        ax[1].step(t, y, where="post", color="darkred", lw=0.8)
        ax[1].set_ylabel("attack label"); ax[1].set_xlabel("row (1 s)")
        ax[1].set_ylim(-0.05, 1.2)
        fig1.suptitle(f"{scores[0].stem}: score, threshold and attack windows")
        fig1.tight_layout()
        fig1.savefig(fig / "score_timeline.png", dpi=150)
        plt.close(fig1)

        # event evidence heatmap
        if y.any():
            s_, e_ = binary_to_ranges(y)[0]
            lo, hi = max(0, s_ - 60), min(len(df), e_ + 60)
            ecols = [c for c in df.columns if c.startswith("e_")]
            E = df[ecols].to_numpy()[lo:hi].T
            keep = np.argsort(-E.max(axis=1))[:25]
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.imshow(np.log1p(E[keep]), aspect="auto", cmap="viridis",
                       extent=[lo, hi, len(keep), 0])
            ax2.axvline(s_, color="r", lw=1); ax2.axvline(e_, color="r", lw=1)
            ax2.set_yticks(np.arange(len(keep)) + 0.5)
            ax2.set_yticklabels([ecols[i][2:] for i in keep], fontsize=6)
            ax2.set_title("top channel evidence around the first attack")
            fig2.tight_layout()
            fig2.savefig(fig / "event_case.png", dpi=150)
            plt.close(fig2)

    rca_path = art / "metrics_rca.csv"
    if rca_path.exists():
        rdf = pd.read_csv(rca_path)
        piv = rdf.groupby("strategy")["top3_hit"].mean()
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        piv.plot.bar(ax=ax3, color="steelblue")
        ax3.set_ylabel("Top-3 hit rate"); ax3.set_title("RCA strategy comparison")
        for i, v in enumerate(piv):
            ax3.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        fig3.tight_layout()
        fig3.savefig(fig / "rca_strategy.png", dpi=150)
        plt.close(fig3)

    pol_path = art / "policy_replay.parquet"
    if pol_path.exists():
        pdf = pd.read_parquet(pol_path)
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        data = [pdf["surrogate_risk_no_action"], pdf["surrogate_risk_fixed"],
                pdf["surrogate_risk_adaptive"]]
        ax4.boxplot(data, tick_labels=["no action", "fixed", "adaptive"])
        ax4.set_ylabel("surrogate risk (lower better)")
        ax4.set_title("Policy comparison in surrogate (surrogate-only)")
        fig4.tight_layout()
        fig4.savefig(fig / "policy_risk.png", dpi=150)
        plt.close(fig4)

    det_path = art / "metrics_detection.csv"
    if det_path.exists():
        ddf = pd.read_csv(det_path)
        fig5, ax5 = plt.subplots(1, 2, figsize=(12, 4))
        ax5[0].bar(ddf["model"], ddf["event_f1"])
        ax5[0].set_ylabel("event F1"); ax5[0].tick_params(axis="x", rotation=45)
        ax5[1].bar(ddf["model"], ddf["mean_detection_delay_s"].fillna(0))
        ax5[1].set_ylabel("mean detection delay (s)")
        ax5[1].tick_params(axis="x", rotation=45)
        fig5.tight_layout()
        fig5.savefig(fig / "detection_summary.png", dpi=150)
        plt.close(fig5)

    print(f"[figures] written to {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
