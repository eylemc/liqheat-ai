#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_market_risk_radar_v1 as v1  # noqa: E402
import build_position_guardian_v4_koinvizyon_matrix as v4  # noqa: E402

OUT = Path("data/reports/market_risk_radar_v2_adaptive_walkforward")
MODEL_OUT = Path("data/models/market_risk_radar_v2_adaptive_walkforward")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT", "XAGUSDT")

BASELINE_FEATURES = list(v1.BASELINE_FEATURES)
STATIC_TOPOLOGY_FEATURES = list(dict.fromkeys(BASELINE_FEATURES + list(v4.STATIC_FEATURES)))
DYNAMIC_TOPOLOGY_FEATURES = list(
    dict.fromkeys(STATIC_TOPOLOGY_FEATURES + list(v4.DYNAMIC_FEATURES))
)
FEATURE_GROUPS = {
    "price_calendar_baseline": BASELINE_FEATURES,
    "price_calendar_static_topology": STATIC_TOPOLOGY_FEATURES,
    "price_calendar_dynamic_topology": DYNAMIC_TOPOLOGY_FEATURES,
}


@dataclass
class Calibrator:
    method: str
    model: object | None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "raw" or self.model is None:
            return p
        if self.method == "platt":
            z = np.log(p / (1 - p)).reshape(-1, 1)
            return np.clip(self.model.predict_proba(z)[:, 1], 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 1e-6, 1 - 1e-6)
        raise ValueError(self.method)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Liqheat Market Risk Radar V2: adaptive per-symbol labels and walk-forward calibration."
    )
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--sample-every-minutes", type=int, default=15)
    p.add_argument("--horizons-minutes", default="15,30,60")
    p.add_argument("--rolling-label-days", type=int, default=30)
    p.add_argument("--min-label-history", type=int, default=300)
    p.add_argument("--high-quantile", type=float, default=0.75)
    p.add_argument("--extreme-quantile", type=float, default=0.90)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--initial-train-fraction", type=float, default=0.50)
    p.add_argument("--calibration-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction-per-fold", type=float, default=0.10)
    p.add_argument("--embargo-hours", type=float, default=4)
    p.add_argument("--iterations", type=int, default=450)
    p.add_argument("--max-train-rows", type=int, default=500000)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--gpu-device", default="0")
    return p.parse_args()


def safe_metric(fn, *args, default=None, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return default


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def load_snapshots(a: argparse.Namespace) -> pd.DataFrame:
    # Matrix deliberately excluded in V2.
    topology = v4.load_topology(a)
    topology = v4.add_dynamic_features(topology)
    return topology.sort_values(["symbol", "logged_at"]).reset_index(drop=True)


def add_causal_adaptive_labels(df: pd.DataFrame, horizon: int, a: argparse.Namespace) -> pd.DataFrame:
    pieces = []
    window = f"{a.rolling_label_days}D"
    delay = pd.Timedelta(minutes=horizon)

    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("logged_at").copy()
        history = g[["logged_at", "future_range_bps"]].dropna().copy()
        history["available_at"] = history["logged_at"] + delay
        history = history.sort_values("available_at")

        s = history.set_index("available_at")["future_range_bps"]
        q50 = s.rolling(window, closed="left", min_periods=a.min_label_history).quantile(0.50)
        q75 = s.rolling(window, closed="left", min_periods=a.min_label_history).quantile(a.high_quantile)
        q90 = s.rolling(window, closed="left", min_periods=a.min_label_history).quantile(a.extreme_quantile)
        count = s.rolling(window, closed="left", min_periods=1).count()

        thresholds = pd.DataFrame({
            "threshold_available_at": q50.index,
            "adaptive_q50_bps": q50.to_numpy(),
            "adaptive_q75_bps": q75.to_numpy(),
            "adaptive_q90_bps": q90.to_numpy(),
            "adaptive_history_count": count.to_numpy(),
        }).sort_values("threshold_available_at")

        left = g.sort_values("logged_at")
        joined = pd.merge_asof(
            left,
            thresholds,
            left_on="logged_at",
            right_on="threshold_available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        joined["is_high_risk"] = (
            joined["future_range_bps"] >= joined["adaptive_q75_bps"]
        ).astype("Int8")
        joined["is_extreme_risk"] = (
            joined["future_range_bps"] >= joined["adaptive_q90_bps"]
        ).astype("Int8")
        joined["realized_risk_band"] = np.select(
            [
                joined["future_range_bps"] >= joined["adaptive_q90_bps"],
                joined["future_range_bps"] >= joined["adaptive_q75_bps"],
                joined["future_range_bps"] >= joined["adaptive_q50_bps"],
            ],
            ["EXTREME RISK", "HIGH RISK", "MEDIUM RISK"],
            default="LOW RISK",
        )
        pieces.append(joined)

    out = pd.concat(pieces, ignore_index=True)
    out = out.dropna(subset=["adaptive_q50_bps", "adaptive_q75_bps", "adaptive_q90_bps"])
    out["is_high_risk"] = out["is_high_risk"].astype("int8")
    out["is_extreme_risk"] = out["is_extreme_risk"].astype("int8")
    return out.sort_values("logged_at").reset_index(drop=True)

# Remainder of file unchanged in repository history.
