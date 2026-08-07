#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_market_risk_radar_v1 as v1  # noqa: E402
import build_market_risk_radar_v2_adaptive_walkforward as v2  # noqa: E402

OUT = Path("data/models/market_risk_radar_v2_live")
HORIZON = 15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the evaluated 15m Market Risk Radar V2 as a live inference bundle.")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--sample-every-minutes", type=int, default=15)
    p.add_argument("--rolling-label-days", type=int, default=30)
    p.add_argument("--min-label-history", type=int, default=300)
    p.add_argument("--high-quantile", type=float, default=0.75)
    p.add_argument("--extreme-quantile", type=float, default=0.90)
    p.add_argument("--calibration-fraction", type=float, default=0.12)
    p.add_argument("--embargo-hours", type=float, default=4)
    p.add_argument("--iterations", type=int, default=450)
    p.add_argument("--max-train-rows", type=int, default=500000)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--gpu-device", default="0")
    return p.parse_args()


def serialise_calibrator(cal) -> dict:
    payload = {"method": cal.method}
    if cal.method == "platt":
        payload["coef"] = float(cal.model.coef_[0][0])
        payload["intercept"] = float(cal.model.intercept_[0])
    elif cal.method == "isotonic":
        payload["x_thresholds"] = [float(x) for x in cal.model.X_thresholds_]
        payload["y_thresholds"] = [float(x) for x in cal.model.y_thresholds_]
    return payload


def head_metrics(y, raw, calibrated) -> dict:
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "raw_roc_auc": float(roc_auc_score(y, raw)),
        "raw_pr_auc": float(average_precision_score(y, raw)),
        "raw_brier": float(brier_score_loss(y, raw)),
        "calibrated_roc_auc": float(roc_auc_score(y, calibrated)),
        "calibrated_pr_auc": float(average_precision_score(y, calibrated)),
        "calibrated_brier": float(brier_score_loss(y, calibrated)),
    }


def main() -> None:
    a = parse_args()
    if not 0 < a.calibration_fraction < 0.5:
        raise SystemExit("--calibration-fraction must be between 0 and 0.5")

    # v2 helpers expect these names on the argparse namespace.
    a.high_quantile = float(a.high_quantile)
    a.extreme_quantile = float(a.extreme_quantile)

    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading 1h topology stream (Matrix excluded)...")
    snapshots = v2.load_snapshots(a)
    print("Loading 1m market data...")
    minute_market = v1.load_minute_market()
    print("Building causal adaptive 15m labels...")
    outcomes = v1.add_forward_range(snapshots, minute_market, HORIZON)
    dataset = v2.add_causal_adaptive_labels(outcomes, HORIZON, a)
    dataset = dataset.sort_values("logged_at").reset_index(drop=True)

    times = pd.to_datetime(dataset["logged_at"], utc=True)
    cal_start = times.quantile(1.0 - a.calibration_fraction)
    embargo = pd.Timedelta(hours=a.embargo_hours)
    train = dataset[times <= cal_start - embargo].copy()
    calibration = dataset[times >= cal_start].copy()

    if len(train) > a.max_train_rows:
        train = train.sort_values("logged_at").tail(a.max_train_rows).copy()
    if train.empty or calibration.empty:
        raise RuntimeError("Production train/calibration split is empty")

    # Head-specific winners from the evaluated 15m walk-forward V2:
    # HIGH risk: static topology; EXTREME risk: dynamic topology.
    heads = {
        "high": {
            "target": "is_high_risk",
            "feature_group": "price_calendar_static_topology",
            "features": v2.STATIC_TOPOLOGY_FEATURES,
        },
        "extreme": {
            "target": "is_extreme_risk",
            "feature_group": "price_calendar_dynamic_topology",
            "features": v2.DYNAMIC_TOPOLOGY_FEATURES,
        },
    }

    manifest = {
        "product": "Matrix AI Radar",
        "engine": "market_risk_radar_v2_live",
        "horizon_minutes": HORIZON,
        "timeframe": a.timeframe,
        "sample_every_minutes": a.sample_every_minutes,
        "symbols": list(v2.SYMBOLS),
        "matrix_used_by_risk_model": False,
        "score_formula": "100 * (0.40 * P(HIGH) + 0.60 * P(EXTREME))",
        "bands": {
            "LOW RISK": [0, 25],
            "MEDIUM RISK": [25, 50],
            "HIGH RISK": [50, 75],
            "EXTREME RISK": [75, 100],
        },
        "training": {
            "rows": int(len(train)),
            "calibration_rows": int(len(calibration)),
            "train_start": str(train["logged_at"].min()),
            "train_end": str(train["logged_at"].max()),
            "calibration_start": str(calibration["logged_at"].min()),
            "calibration_end": str(calibration["logged_at"].max()),
            "rolling_label_days": a.rolling_label_days,
            "embargo_hours": a.embargo_hours,
        },
        "heads": {},
    }

    for head, cfg in heads.items():
        print(f"Training live {head.upper()} head: {cfg['feature_group']}")
        model, used, cats = v2.fit_head(train, calibration, cfg["features"], cfg["target"], a)
        raw_cal = v2.predict(model, calibration, used, cats)
        calibrator, candidates = v2.fit_calibrator(calibration[cfg["target"]].to_numpy(), raw_cal)
        calibrated = calibrator.transform(raw_cal)

        model_path = OUT / f"{head}_head.cbm"
        model.save_model(str(model_path))
        cal_payload = serialise_calibrator(calibrator)
        (OUT / f"{head}_calibrator.json").write_text(json.dumps(cal_payload, indent=2), encoding="utf-8")

        manifest["heads"][head] = {
            "feature_group": cfg["feature_group"],
            "target": cfg["target"],
            "model": model_path.name,
            "calibrator": f"{head}_calibrator.json",
            "features": list(used),
            "categorical_features": list(cats),
            "calibration_method": calibrator.method,
            "calibration_candidates": candidates,
            "calibration_metrics": head_metrics(calibration[cfg["target"]].to_numpy(), raw_cal, calibrated),
        }

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
