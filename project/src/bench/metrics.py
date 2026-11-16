"""Evaluation metrics.

Primary metrics follow the plan: point-level PR-AUC, VUS-PR, eTaPR
(official package), event-level precision/recall/F1, false alarms per
hour and first-detection delay. PA-F1 is implemented for appendix
comparability only — never a headline metric.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from src.common.events import binary_to_ranges, ranges_to_flags


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(labels, scores)
    # precision_recall_curve returns decreasing thresholds; use auc(r, p)
    return float(auc(r, p))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def vus_pr(scores: np.ndarray, labels: np.ndarray, max_samples: int = 200) -> float:
    """Unavailable until the author's VUS implementation is installed and validated.

    Never substitute the former sliding-dilation approximation under this name.
    """
    return float("nan")


def event_metrics(pred: np.ndarray, labels: np.ndarray,
                  min_gap: int = 3) -> dict:
    """Event-level metrics: an event is detected if >=1 predicted point
    falls inside it; an alert event is false if it touches no label run."""
    label_runs = binary_to_ranges(labels)
    pred_runs = binary_to_ranges(pred, min_gap=min_gap)
    tp_ev = sum(1 for s, e in label_runs
                if any(not (e < ps or s > pe) for ps, pe in pred_runs))
    fn_ev = len(label_runs) - tp_ev
    fp_ev = sum(1 for ps, pe in pred_runs
                if all(pe < s or ps > e for s, e in label_runs))
    delays = []
    for s, e in label_runs:
        inside = [ps for ps, pe in pred_runs if not (e < ps or s > pe)]
        if inside:
            delays.append(max(0, min(inside) - s))
    prec = tp_ev / max(1, tp_ev + fp_ev)
    rec = tp_ev / max(1, tp_ev + fn_ev)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"event_precision": prec, "event_recall": rec, "event_f1": f1,
            "n_label_events": len(label_runs), "n_pred_events": len(pred_runs),
            "detected_events": tp_ev, "false_alert_events": fp_ev,
            "mean_detection_delay_s": float(np.mean(delays)) if delays else None,
            "median_detection_delay_s": float(np.median(delays)) if delays else None}


def false_alarms_per_hour(pred: np.ndarray, labels: np.ndarray,
                        seconds_per_row: float = 1.0) -> float:
    pred_runs = binary_to_ranges(pred)
    label_runs = binary_to_ranges(labels)
    false_runs = [r for r in pred_runs
                  if all(r[1] < s or r[0] > e for s, e in label_runs)]
    hours = len(pred) * seconds_per_row / 3600.0
    return len(false_runs) / max(1e-9, hours)


def pa_f1(pred: np.ndarray, labels: np.ndarray) -> float:
    """Point-Adjusted F1 — appendix only, biased by construction."""
    label_runs = binary_to_ranges(labels)
    adj = pred.copy().astype(bool)
    for s, e in label_runs:
        if pred[s:e + 1].any():
            adj[s:e + 1] = True
    tp = int((adj & labels.astype(bool)).sum())
    fp = int((adj & ~labels.astype(bool)).sum())
    fn = int((~adj & labels.astype(bool)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return 2 * p * r / max(1e-9, p + r)


def etapr_metrics(pred: np.ndarray, labels: np.ndarray,
                  theta_p: float = 0.7, theta_r: float = 0.1,
                  delta: float = 0.0) -> dict:
    """Official eTaPR package (recommended by the HAI authors)."""
    # The pinned author release raises on empty range lists. Undefined
    # denominators remain unavailable; a missed nonempty attack set has recall 0.
    if not np.any(labels):
        return {"eTaPR_f1": None, "eTaP": 0.0 if np.any(pred) else None, "eTaR": None}
    if not np.any(pred):
        return {"eTaPR_f1": 0.0, "eTaP": None, "eTaR": 0.0}
    import sys
    from pathlib import Path
    vendor = Path(__file__).resolve().parents[2] / 'vendor' / 'etapr'
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    try:
        from eTaPR_pkg import etapr
    except ImportError:
        return {"eTaPR_f1": None, "eTaP": None, "eTaR": None}
    res = etapr.evaluate_w_streams(labels.astype(int).tolist(),
                                   pred.astype(int).tolist(),
                                   theta_p=theta_p, theta_r=theta_r, delta=delta)
    return {"eTaPR_f1": res.get("f1"), "eTaP": res.get("eTaP"),
            "eTaR": res.get("eTaR"), "eTaP_d": res.get("eTaPd"),
            "eTaR_d": res.get("eTaRd")}


def full_detection_report(scores: np.ndarray, pred: np.ndarray,
                          labels: np.ndarray, cfg) -> dict:
    out = {}
    out["point_pr_auc"] = pr_auc(scores, labels)
    out["point_roc_auc"] = roc_auc(scores, labels)
    out["vus_pr"] = vus_pr(scores, labels)
    out.update(event_metrics(pred, labels, cfg["detection"]["min_event_gap_s"]))
    out["false_alarms_per_hour"] = false_alarms_per_hour(pred, labels)
    out["pa_f1_appendix"] = pa_f1(pred, labels)
    out.update(etapr_metrics(pred, labels,
                             cfg["eval"]["etapr_theta_p"],
                             cfg["eval"]["etapr_theta_r"],
                             cfg["eval"]["etapr_delta"]))
    return out


# --------------------------------------------------------- RCA metrics
def topk_hit(cands: list[dict], gt_tags: set, k: int) -> float:
    top = {c["tag"] for c in cands[:k]}
    return 1.0 if top & gt_tags else 0.0


def mrr(cands: list[dict], gt_tags: set) -> float:
    for c in cands:
        if c["tag"] in gt_tags:
            return 1.0 / c["rank"]
    return 0.0
