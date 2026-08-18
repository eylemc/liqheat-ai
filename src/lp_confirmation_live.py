from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data/research/liquidation_pressure/liquidation_pressure.sqlite3"

CONFIDENCE_THRESHOLD = 0.70
MIN_SAMPLES_120M = 60
MIN_PERSISTENCE_120M = 60.0


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _sign(value: float | None) -> int:
    if value is None:
        return 0
    return 1 if value > 0 else -1 if value < 0 else 0


def _bias_sign(prediction: str) -> int:
    p = str(prediction or "").upper()
    if p == "UPPER_FIRST":
        return 1
    if p == "LOWER_FIRST":
        return -1
    return 0


def _latest_features(symbol: str) -> dict[str, Any] | None:
    if not DB_PATH.exists():
        return None

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=1.0)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            """
            SELECT
                f.observed_at,
                f.signed_now,
                f.mean_30m,
                f.mean_60m,
                f.mean_120m,
                f.slope_30m,
                f.slope_60m,
                f.slope_120m,
                f.persistence_30m,
                f.persistence_60m,
                f.persistence_120m,
                f.flips_30m,
                f.flips_60m,
                f.flips_120m,
                f.peak_abs_30m,
                f.peak_abs_60m,
                f.peak_abs_120m,
                f.acceleration_2h,
                f.sample_count_30m,
                f.sample_count_60m,
                f.sample_count_120m
            FROM pressure_features f
            WHERE f.symbol = ?
            ORDER BY f.snapshot_id DESC
            LIMIT 1
            """,
            (str(symbol).upper(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def build_lp_confirmation(symbol: str, direction_model: dict[str, Any]) -> dict[str, Any]:
    prediction = str(direction_model.get("prediction") or "").upper()
    confidence = _finite(direction_model.get("confidence"))
    bias_sign = _bias_sign(prediction)

    base = {
        "available": False,
        "state": "NEUTRAL",
        "method": "LP_TEMPORAL_2H_CONSENSUS_V1",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "min_samples_120m": MIN_SAMPLES_120M,
        "min_persistence_120m": MIN_PERSISTENCE_120M,
    }

    if confidence is None or confidence < CONFIDENCE_THRESHOLD or bias_sign == 0:
        base["reason"] = "DIRECTION_BIAS_NOT_HIGH_CONFIDENCE"
        return base

    features = _latest_features(symbol)
    if not features:
        base["reason"] = "NO_PRESSURE_HISTORY"
        return base

    n120 = int(features.get("sample_count_120m") or 0)
    persistence120 = _finite(features.get("persistence_120m"))
    if n120 < MIN_SAMPLES_120M:
        base.update({
            "reason": "INSUFFICIENT_2H_HISTORY",
            "sample_count_120m": n120,
            "persistence_120m": persistence120,
        })
        return base

    current_sign = _sign(_finite(features.get("signed_now")))
    mean_sign = _sign(_finite(features.get("mean_120m")))
    slope_sign = _sign(_finite(features.get("slope_120m")))

    votes = [current_sign, mean_sign, slope_sign]
    agree_votes = sum(1 for v in votes if v == bias_sign)
    oppose_votes = sum(1 for v in votes if v == -bias_sign)
    stable = persistence120 is not None and persistence120 >= MIN_PERSISTENCE_120M

    if not stable:
        state = "NEUTRAL"
        reason = "PRESSURE_HISTORY_UNSTABLE"
    elif agree_votes >= 2:
        state = "CONFIRMED"
        reason = "TEMPORAL_PRESSURE_SUPPORTS_BIAS"
    elif oppose_votes >= 2:
        state = "CONFLICT"
        reason = "TEMPORAL_PRESSURE_OPPOSES_BIAS"
    else:
        state = "NEUTRAL"
        reason = "MIXED_PRESSURE_HISTORY"

    return {
        "available": True,
        "state": state,
        "reason": reason,
        "method": "LP_TEMPORAL_2H_CONSENSUS_V1",
        "prediction": prediction,
        "direction_confidence": confidence,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "sample_count_120m": n120,
        "persistence_120m": persistence120,
        "signed_now": _finite(features.get("signed_now")),
        "mean_30m": _finite(features.get("mean_30m")),
        "mean_60m": _finite(features.get("mean_60m")),
        "mean_120m": _finite(features.get("mean_120m")),
        "slope_30m": _finite(features.get("slope_30m")),
        "slope_60m": _finite(features.get("slope_60m")),
        "slope_120m": _finite(features.get("slope_120m")),
        "acceleration_2h": _finite(features.get("acceleration_2h")),
        "flips_120m": int(features.get("flips_120m") or 0),
        "agree_votes": agree_votes,
        "oppose_votes": oppose_votes,
        "observed_at": features.get("observed_at"),
    }
