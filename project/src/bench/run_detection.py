"""Detection benchmark: evaluate score artifacts under the locked protocol.

Evaluates one or more ``scores_test_*.parquet`` artifacts with the
threshold locked on the corresponding validation split (quantile/FAR from
config — never test-optimal). Also runs the non-model baselines
(per-channel robust-z quantile, PCA reconstruction error,
IsolationForest) directly on the prepared data.

Output: ``artifacts/metrics_detection.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.common.io_utils import read_json  # noqa: E402
from src.bench.metrics import full_detection_report  # noqa: E402
from src.data.fit_scaler import apply_scaler  # noqa: E402
from src.detect.calibrate import calibrate  # noqa: E402
from src.detect.score import aggregate, normalize_residuals, residual_scale  # noqa: E402


def _split_arrays(cfg, split, tags):
    index = read_json(resolve(cfg, "processed_dir") / "index.json")
    scaler = read_json(resolve(cfg, "processed_dir") / "scaler.json")
    X, y = [], []
    for rec in index["splits"][split]:
        df = pd.read_parquet(rec["parquet"])
        X.append(apply_scaler(df[tags].to_numpy(np.float32), scaler))
        y.append(df["label_attack"].to_numpy())
    return np.concatenate(X), np.concatenate(y)


def _eval_scores(name, scores, labels, cfg, thr_meta):
    pred = (scores > thr_meta["threshold"]).astype(int)
    rep = full_detection_report(scores, pred, labels, cfg)
    rep.update({"model": name, "threshold": thr_meta["threshold"],
                "threshold_method": thr_meta["method"]})
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--score-files", nargs="*", default=None,
                    help="test score parquets to evaluate (default: all in artifacts/)")
    ap.add_argument("--baselines", action="store_true",
                    help="also run channel-z, PCA and IsolationForest baselines")
    ap.add_argument("--baseline-max-train", type=int, default=50000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = resolve(cfg, "artifacts_dir")
    tags = read_json(resolve(cfg, "processed_dir") / "scaler.json")["tags"]

    rows = []
    score_files = args.score_files
    if not score_files:
        score_files = sorted(str(p) for p in art.glob("scores_test_*.parquet"))
    for sf in score_files:
        name = Path(sf).stem.replace("scores_test_", "")
        df = pd.read_parquet(sf)
        # val scores for the same tag name
        val_path = art / f"scores_val_{name}.parquet"
        if val_path.exists():
            vdf = pd.read_parquet(val_path, columns=["score"])
            thr = calibrate(vdf["score"].to_numpy(),
                            cfg["threshold"]["method"],
                            cfg["threshold"]["quantile"],
                            cfg["threshold"]["far_per_hour"])
        else:
            print(f"[bench] WARNING: no val scores for {name}; using test-quantile "
                  "(NOT protocol-clean, flagged)")
            thr = {"threshold": float(np.quantile(df["score"], cfg["threshold"]["quantile"])),
                   "method": "TEST_QUANTILE_FALLBACK_NOT_CLEAN"}
        rep = _eval_scores(name, df["score"].to_numpy(),
                           df["label_attack"].to_numpy(), cfg, thr)
        rep["val_scores_file"] = str(val_path) if val_path.exists() else None
        rows.append(rep)
        print(f"[bench] {name}: event_f1={rep['event_f1']:.3f} "
              f"eTaPR={rep.get('eTaPR_f1')} delay={rep['mean_detection_delay_s']}")

    if args.baselines:
        Xtr, _ = _split_arrays(cfg, "train", tags)
        Xv, _ = _split_arrays(cfg, "val", tags)
        Xte, yte = _split_arrays(cfg, "test", tags)
        rng = np.random.default_rng(0)
        sub = rng.choice(len(Xtr), min(args.baseline_max_train, len(Xtr)),
                         replace=False)

        # baseline 1: robust per-channel z, threshold on max|z|
        med = np.median(Xtr[sub], axis=0)
        mad = np.median(np.abs(Xtr[sub] - med), axis=0) * 1.4826 + 1e-6
        zv = np.abs(Xv - med) / mad
        zt = np.abs(Xte - med) / mad
        sv, st = zv.max(axis=1), zt.max(axis=1)
        thr = calibrate(sv, cfg["threshold"]["method"],
                        cfg["threshold"]["quantile"], cfg["threshold"]["far_per_hour"])
        rows.append(_eval_scores("base_channel_z", st, yte, cfg, thr))
        print("[bench] base_channel_z done")

        # baseline 2: PCA reconstruction error
        n_comp = max(2, min(20, Xtr.shape[1] // 4))
        pca = PCA(n_components=n_comp).fit(Xtr[sub])
        ev = ((Xv - pca.inverse_transform(pca.transform(Xv))) ** 2)
        et = ((Xte - pca.inverse_transform(pca.transform(Xte))) ** 2)
        sc = residual_scale(((Xtr[sub] - pca.inverse_transform(
            pca.transform(Xtr[sub]))) ** 2), "std")
        sv = aggregate(normalize_residuals(ev, sc), cfg["model"]["score_agg"],
                       cfg["model"]["score_topk"])
        st = aggregate(normalize_residuals(et, sc), cfg["model"]["score_agg"],
                       cfg["model"]["score_topk"])
        thr = calibrate(sv, cfg["threshold"]["method"],
                        cfg["threshold"]["quantile"], cfg["threshold"]["far_per_hour"])
        rows.append(_eval_scores("base_pca", st, yte, cfg, thr))
        print("[bench] base_pca done")

        # baseline 3: IsolationForest (score = -decision_function)
        iso = IsolationForest(n_estimators=100, contamination="auto",
                              random_state=0).fit(Xtr[sub])
        sv = -iso.decision_function(Xv)
        st = -iso.decision_function(Xte)
        thr = calibrate(sv, cfg["threshold"]["method"],
                        cfg["threshold"]["quantile"], cfg["threshold"]["far_per_hour"])
        rows.append(_eval_scores("base_isoforest", st, yte, cfg, thr))
        print("[bench] base_isoforest done")

    out = art / "metrics_detection.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[bench] wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
