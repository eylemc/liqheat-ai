from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data/research/liquidation_pressure/liquidation_pressure.sqlite3"

CONFIDENCE_THRESHOLD = 0.60
MIN_SAMPLES_120M = 60
MIN_PERSISTENCE_60M = 55.0
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
        "method": "LP_TEMPORAL_MULTIWINDOW_CONSENSUS_V2",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "min_samples_120m": MIN_SAMPLES_120M,
        "min_persistence_60m": MIN_PERSISTENCE_60M,
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
    persistence60 = _finite(features.get("persistence_60m"))
    persistence120 = _finite(features.get("persistence_120m"))
    if n120 < MIN_SAMPLES_120M:
        base.update({
            "reason": "INSUFFICIENT_2H_HISTORY",
            "sample_count_120m": n120,
            "persistence_60m": persistence60,
            "persistence_120m": persistence120,
        })
        return base

    mean30 = _finite(features.get("mean_30m"))
    mean60 = _finite(features.get("mean_60m"))
    mean120 = _finite(features.get("mean_120m"))
    sign30 = _sign(mean30)
    sign60 = _sign(mean60)
    sign120 = _sign(mean120)

    stable = (
        persistence60 is not None
        and persistence60 >= MIN_PERSISTENCE_60M
        and persistence120 is not None
        and persistence120 >= MIN_PERSISTENCE_120M
    )

    # Deliberately exclude signed_now and slope from the state decision.
    # A 2H confirmation should not flip because of one fresh snapshot or a noisy
    # regression slope. The 60m and 120m pressure regimes must agree before a
    # directional confirmation/conflict is emitted. 30m remains diagnostic and
    # can warn that a shorter-term transition has begun without changing state.
    if not stable:
        state = "NEUTRAL"
        reason = "PRESSURE_HISTORY_UNSTABLE"
    elif sign60 == bias_sign and sign120 == bias_sign:
        state = "CONFIRMED"
        reason = "60M_120M_PRESSURE_SUPPORT_BIAS"
    elif sign60 == -bias_sign and sign120 == -bias_sign:
        state = "CONFLICT"
        reason = "60M_120M_PRESSURE_OPPOSE_BIAS"
    else:
        state = "NEUTRAL"
        reason = "MULTIWINDOW_PRESSURE_DISAGREEMENT"

    agree_windows = sum(1 for v in (sign30, sign60, sign120) if v == bias_sign)
    oppose_windows = sum(1 for v in (sign30, sign60, sign120) if v == -bias_sign)

    return {
        "available": True,
        "state": state,
        "reason": reason,
        "method": "LP_TEMPORAL_MULTIWINDOW_CONSENSUS_V2",
        "prediction": prediction,
        "direction_confidence": confidence,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "sample_count_120m": n120,
        "persistence_60m": persistence60,
        "persistence_120m": persistence120,
        "mean_30m": mean30,
        "mean_60m": mean60,
        "mean_120m": mean120,
        "mean_sign_30m": sign30,
        "mean_sign_60m": sign60,
        "mean_sign_120m": sign120,
        "short_term_transition": sign30 != 0 and sign120 != 0 and sign30 != sign120,
        "agree_windows": agree_windows,
        "oppose_windows": oppose_windows,
        # Diagnostics only; these no longer participate in the state decision.
        "signed_now": _finite(features.get("signed_now")),
        "slope_30m": _finite(features.get("slope_30m")),
        "slope_60m": _finite(features.get("slope_60m")),
        "slope_120m": _finite(features.get("slope_120m")),
        "acceleration_2h": _finite(features.get("acceleration_2h")),
        "flips_120m": int(features.get("flips_120m") or 0),
        "observed_at": features.get("observed_at"),
    }
