#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_market_risk_radar_v2_adaptive_walkforward as v2  # noqa: E402
from src.market_risk_live import _apply_calibrator  # noqa: E402

MODEL_DIR = ROOT / "data" / "models" / "market_risk_radar_v2_live"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build empirical percentile reference for Matrix AI Radar 15m live risk score.")
    p.add_argument("--quantile-points", type=int, default=101)
    return p.parse_args()


def prepare(frame: pd.DataFrame, features: list[str], cats: list[str]) -> pd.DataFrame:
    x = frame[features].copy()
    for col in features:
        if col in cats:
            x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    return x


def main() -> None:
    a = parse_args()
    manifest_path = MODEL_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing live bundle: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ns = argparse.Namespace(
        timeframe=manifest.get("timeframe", "1h"),
        sample_every_minutes=int(manifest.get("sample_every_minutes", 15)),
    )

    print("Loading topology history for calibration reference...")
    frame = v2.load_snapshots(ns)
    frame["logged_at"] = pd.to_datetime(frame["logged_at"], utc=True)

    start = pd.to_datetime(manifest["training"]["calibration_start"], utc=True)
    end = pd.to_datetime(manifest["training"]["calibration_end"], utc=True)
    cal = frame[(frame["logged_at"] >= start) & (frame["logged_at"] <= end)].copy()
    if cal.empty:
        raise RuntimeError("Calibration reference window is empty")

    probs: dict[str, np.ndarray] = {}
    for head in ("high", "extreme"):
        cfg = manifest["heads"][head]
        features = list(cfg["features"])
        cats = list(cfg["categorical_features"])
        missing = [c for c in features if c not in cal.columns]
        if missing:
            raise RuntimeError(f"Missing {head} features: {missing}")

        model = CatBoostClassifier()
        model.load_model(str(MODEL_DIR / cfg["model"]))
        x = prepare(cal, features, cats)
        raw = model.predict_proba(Pool(x, cat_features=[c for c in cats if c in x.columns]))[:, 1]
        calibrator = json.loads((MODEL_DIR / cfg["calibrator"]).read_text(encoding="utf-8"))
        probs[head] = _apply_calibrator(raw, calibrator)

    composite = np.clip(100.0 * (0.40 * probs["high"] + 0.60 * probs["extreme"]), 0.0, 100.0)
    cal = cal.reset_index(drop=True)
    cal["composite_probability_score"] = composite

    q = np.linspace(0.0, 1.0, max(11, int(a.quantile_points)))
    symbols = {}
    for symbol, g in cal.groupby("symbol", sort=True):
        values = pd.to_numeric(g["composite_probability_score"], errors="coerce").dropna().to_numpy(float)
        if len(values) < 100:
            continue
        symbols[str(symbol)] = {
            "rows": int(len(values)),
            "percentiles": [round(float(x * 100.0), 6) for x in q],
            "score_quantiles": [round(float(x), 8) for x in np.quantile(values, q)],
        }

    all_values = cal["composite_probability_score"].dropna().to_numpy(float)
    payload = {
        "method": "empirical_calibration_percentile",
        "meaning": "Risk score is the percentile rank of the current calibrated composite probability versus the model calibration period for the same symbol.",
        "calibration_start": start.isoformat(),
        "calibration_end": end.isoformat(),
        "band_thresholds": {
            "LOW RISK": [0, 50],
            "MEDIUM RISK": [50, 75],
            "HIGH RISK": [75, 90],
            "EXTREME RISK": [90, 100],
        },
        "symbols": symbols,
        "all": {
            "rows": int(len(all_values)),
            "percentiles": [round(float(x * 100.0), 6) for x in q],
            "score_quantiles": [round(float(x), 8) for x in np.quantile(all_values, q)],
        },
    }
    out = MODEL_DIR / "risk_reference.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Done: {out}")
    for symbol, ref in symbols.items():
        vals = np.asarray(ref["score_quantiles"], dtype=float)
        print(symbol, "median=", round(float(vals[len(vals)//2]), 2), "p75=", round(float(np.quantile(cal.loc[cal['symbol'].astype(str)==symbol, 'composite_probability_score'], .75)), 2), "p90=", round(float(np.quantile(cal.loc[cal['symbol'].astype(str)==symbol, 'composite_probability_score'], .90)), 2))


if __name__ == "__main__":
    main()
