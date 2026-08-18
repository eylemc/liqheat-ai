from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data/research/direction_bias/direction_bias.sqlite3"

WINDOW_MINUTES = 120
MIN_SAMPLES = 30
MIN_PERSISTENCE = 60.0


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _sign(prediction: str | None) -> int:
    p = str(prediction or "").upper()
    if p == "UPPER_FIRST":
        return 1
    if p == "LOWER_FIRST":
        return -1
    return 0


def _prediction(sign: int) -> str:
    return "UPPER_FIRST" if sign > 0 else "LOWER_FIRST" if sign < 0 else "NEUTRAL"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=2.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bias_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            snapshot_id TEXT,
            prediction TEXT NOT NULL,
            confidence REAL,
            probability_upper REAL,
            probability_lower REAL,
            signed_confidence REAL,
            UNIQUE(symbol, snapshot_id)
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_bias_symbol_time ON bias_snapshots(symbol, observed_at)"
    )
    return con


def record_direction_bias(symbol: str, snapshot_id: str | None, direction_model: dict[str, Any]) -> None:
    if not direction_model or not direction_model.get("available"):
        return
    prediction = str(direction_model.get("prediction") or "").upper()
    sign = _sign(prediction)
    if sign == 0:
        return
    confidence = _finite(direction_model.get("confidence"))
    upper = _finite(direction_model.get("probability_upper_first"))
    lower = _finite(direction_model.get("probability_lower_first"))
    observed_at = str(direction_model.get("as_of") or datetime.now(timezone.utc).isoformat())
    signed_confidence = sign * (confidence or 0.0)

    con = _connect()
    try:
        con.execute(
            """
            INSERT OR IGNORE INTO bias_snapshots(
                observed_at,symbol,snapshot_id,prediction,confidence,
                probability_upper,probability_lower,signed_confidence
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                observed_at,
                str(symbol).upper(),
                str(snapshot_id) if snapshot_id is not None else None,
                prediction,
                confidence,
                upper,
                lower,
                signed_confidence,
            ),
        )
        cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        con.execute("DELETE FROM bias_snapshots WHERE observed_at < ?", (cutoff,))
        con.commit()
    finally:
        con.close()


def build_temporal_direction_bias(symbol: str, current_model: dict[str, Any]) -> dict[str, Any]:
    raw_prediction = str((current_model or {}).get("prediction") or "").upper()
    raw_confidence = _finite((current_model or {}).get("confidence"))
    base = {
        "available": False,
        "method": "DIRECTION_BIAS_TEMPORAL_2H_V1",
        "window_minutes": WINDOW_MINUTES,
        "min_samples": MIN_SAMPLES,
        "min_persistence": MIN_PERSISTENCE,
        "raw_prediction": raw_prediction or None,
        "raw_confidence": raw_confidence,
    }
    if not (current_model or {}).get("available"):
        base["reason"] = "RAW_DIRECTION_UNAVAILABLE"
        return base
    if not DB_PATH.exists():
        base["reason"] = "NO_BIAS_HISTORY"
        return base

    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)).isoformat()
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=1.0)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT observed_at,prediction,confidence,probability_upper,probability_lower,signed_confidence
            FROM bias_snapshots
            WHERE symbol=? AND observed_at>=?
            ORDER BY observed_at ASC
            """,
            (str(symbol).upper(), cutoff),
        ).fetchall()
    finally:
        con.close()

    n = len(rows)
    base["sample_count_120m"] = n
    if n < MIN_SAMPLES:
        base["reason"] = "INSUFFICIENT_2H_HISTORY"
        return base

    signs = [_sign(r["prediction"]) for r in rows]
    confs = [float(r["confidence"] or 0.0) for r in rows]
    upper_weight = sum(c for s, c in zip(signs, confs) if s > 0)
    lower_weight = sum(c for s, c in zip(signs, confs) if s < 0)
    total_weight = upper_weight + lower_weight
    temporal_sign = 1 if upper_weight > lower_weight else -1 if lower_weight > upper_weight else 0

    dominant_count = sum(1 for s in signs if s == temporal_sign) if temporal_sign else 0
    persistence = 100.0 * dominant_count / n if n else 0.0
    weighted_confidence = max(upper_weight, lower_weight) / total_weight if total_weight > 0 else 0.5

    flips = 0
    prev = 0
    for s in signs:
        if s == 0:
            continue
        if prev and s != prev:
            flips += 1
        prev = s

    recent_n = max(1, min(10, n // 4))
    recent_signs = signs[-recent_n:]
    recent_upper = sum(1 for s in recent_signs if s > 0)
    recent_lower = sum(1 for s in recent_signs if s < 0)
    recent_sign = 1 if recent_upper > recent_lower else -1 if recent_lower > recent_upper else 0

    if persistence < MIN_PERSISTENCE:
        state = "MIXED"
    elif recent_sign and temporal_sign and recent_sign != temporal_sign:
        state = "REVERSING"
    else:
        state = "STABLE"

    return {
        "available": True,
        "method": "DIRECTION_BIAS_TEMPORAL_2H_V1",
        "prediction": _prediction(temporal_sign),
        "confidence": round(float(weighted_confidence), 6),
        "confidence_pct": round(float(weighted_confidence) * 100.0, 1),
        "state": state,
        "persistence_120m": round(float(persistence), 2),
        "sample_count_120m": n,
        "flips_120m": flips,
        "upper_weight": round(float(upper_weight), 6),
        "lower_weight": round(float(lower_weight), 6),
        "raw_prediction": raw_prediction,
        "raw_confidence": raw_confidence,
        "recent_direction": _prediction(recent_sign),
        "window_minutes": WINDOW_MINUTES,
        "min_samples": MIN_SAMPLES,
        "min_persistence": MIN_PERSISTENCE,
        "observed_at": rows[-1]["observed_at"],
    }
