#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "http://127.0.0.1:8000/radar"
DB_PATH = Path("data/research/liquidation_pressure/liquidation_pressure.sqlite3")
DEFAULT_INTERVAL_SECONDS = 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fetch_radar() -> dict[str, Any]:
    req = urllib.request.Request(API_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def pressure_value(item: dict[str, Any]) -> float | None:
    if item.get("liquidity_pressure_score") is not None:
        return finite(item.get("liquidity_pressure_score"))
    if item.get("liquidity_pressure") is not None:
        x = finite(item.get("liquidity_pressure"))
        return None if x is None else x * 100.0
    x = finite(item.get("score"))
    return None if x is None else x * 100.0


def pressure_direction(item: dict[str, Any]) -> str:
    raw = str(item.get("raw_prediction") or item.get("prediction") or "").upper()
    if raw == "SHORT_SQUEEZE":
        return "UP"
    if raw == "LONG_SQUEEZE":
        return "DOWN"
    return "N/A"


def signed_pressure(item: dict[str, Any]) -> float | None:
    value = pressure_value(item)
    if value is None:
        return None
    direction = pressure_direction(item)
    if direction == "UP":
        return abs(value)
    if direction == "DOWN":
        return -abs(value)
    return 0.0


def init_db(con: sqlite3.Connection) -> None:
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS pressure_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at TEXT NOT NULL,
            payload_generated_at TEXT,
            symbol TEXT NOT NULL,
            current_price REAL,
            pressure_value REAL,
            pressure_direction TEXT,
            signed_pressure REAL,
            raw_prediction TEXT,
            prediction TEXT,
            direction_prediction TEXT,
            direction_confidence REAL,
            direction_confidence_pct REAL,
            matrix_alignment REAL,
            matrix_1h TEXT,
            gate_signal TEXT,
            gate_risk TEXT,
            gate_score REAL,
            market_heat_score REAL,
            market_heat_band TEXT,
            age_seconds REAL,
            raw_item_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_pressure_symbol_time
        ON pressure_snapshots(symbol, observed_at);

        CREATE TABLE IF NOT EXISTS pressure_features (
            snapshot_id INTEGER PRIMARY KEY,
            observed_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signed_now REAL,
            mean_30m REAL,
            mean_60m REAL,
            mean_120m REAL,
            slope_30m REAL,
            slope_60m REAL,
            slope_120m REAL,
            persistence_30m REAL,
            persistence_60m REAL,
            persistence_120m REAL,
            flips_30m INTEGER,
            flips_60m INTEGER,
            flips_120m INTEGER,
            peak_abs_30m REAL,
            peak_abs_60m REAL,
            peak_abs_120m REAL,
            acceleration_2h REAL,
            sample_count_30m INTEGER,
            sample_count_60m INTEGER,
            sample_count_120m INTEGER,
            FOREIGN KEY(snapshot_id) REFERENCES pressure_snapshots(id)
        );
        """
    )
    con.commit()


def rows_for_window(con: sqlite3.Connection, symbol: str, minutes: int) -> list[tuple[float, str]]:
    rows = con.execute(
        """
        SELECT signed_pressure, observed_at
        FROM pressure_snapshots
        WHERE symbol = ?
          AND signed_pressure IS NOT NULL
          AND julianday(observed_at) >= julianday('now') - (? / 1440.0)
        ORDER BY observed_at
        """,
        (symbol, float(minutes)),
    ).fetchall()
    return [(float(v), str(t)) for v, t in rows]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def slope(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    sx = n * (n - 1) / 2.0
    sy = sum(values)
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    sxy = sum(i * v for i, v in enumerate(values))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom


def sign(v: float) -> int:
    return 1 if v > 0 else -1 if v < 0 else 0


def persistence(values: list[float]) -> float | None:
    nonzero = [sign(v) for v in values if sign(v) != 0]
    if not nonzero:
        return None
    ups = sum(1 for s in nonzero if s > 0)
    downs = len(nonzero) - ups
    return max(ups, downs) / len(nonzero) * 100.0


def flip_count(values: list[float]) -> int:
    signs = [sign(v) for v in values if sign(v) != 0]
    if len(signs) < 2:
        return 0
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def stats(con: sqlite3.Connection, symbol: str, minutes: int) -> dict[str, Any]:
    rows = rows_for_window(con, symbol, minutes)
    values = [v for v, _ in rows]
    return {
        "mean": mean(values),
        "slope": slope(values),
        "persistence": persistence(values),
        "flips": flip_count(values),
        "peak_abs": max((abs(v) for v in values), default=None),
        "n": len(values),
    }


def insert_item(con: sqlite3.Connection, item: dict[str, Any], generated_at: str | None) -> int:
    observed_at = utc_now()
    direction_model = item.get("direction_model") or {}
    matrix = item.get("matrix") or {}
    timeframes = matrix.get("timeframes") or {}
    one_h = timeframes.get("1h") or {}
    gate = item.get("matrix_regime_gate") or {}
    latest_flip = gate.get("latest_flip") or {}
    risk = item.get("ai_market_risk") or {}

    p_value = pressure_value(item)
    p_direction = pressure_direction(item)
    p_signed = signed_pressure(item)

    cur = con.execute(
        """
        INSERT INTO pressure_snapshots (
            observed_at, payload_generated_at, symbol, current_price,
            pressure_value, pressure_direction, signed_pressure,
            raw_prediction, prediction,
            direction_prediction, direction_confidence, direction_confidence_pct,
            matrix_alignment, matrix_1h,
            gate_signal, gate_risk, gate_score,
            market_heat_score, market_heat_band,
            age_seconds, raw_item_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            observed_at,
            generated_at,
            str(item.get("symbol") or "").upper(),
            finite(item.get("current_price")),
            p_value,
            p_direction,
            p_signed,
            item.get("raw_prediction"),
            item.get("prediction"),
            direction_model.get("prediction"),
            finite(direction_model.get("confidence")),
            finite(direction_model.get("confidence_pct")),
            finite(matrix.get("alignment_score")),
            one_h.get("trend_label"),
            latest_flip.get("side") or gate.get("matrix_trend"),
            gate.get("risk_level") or ("LOW RISK" if gate.get("status") == "VALID" else "HIGH RISK" if gate.get("available") else None),
            finite(gate.get("regime_score")),
            finite(risk.get("risk_score")) if risk.get("available") else None,
            risk.get("risk_band") if risk.get("available") else None,
            finite(item.get("age_seconds")),
            json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str),
        ),
    )
    snapshot_id = int(cur.lastrowid)

    windows = {m: stats(con, str(item.get("symbol") or "").upper(), m) for m in (30, 60, 120)}
    s30, s60, s120 = windows[30], windows[60], windows[120]
    acceleration = None
    if s30["slope"] is not None and s120["slope"] is not None:
        acceleration = float(s30["slope"] - s120["slope"])

    con.execute(
        """
        INSERT INTO pressure_features (
            snapshot_id, observed_at, symbol, signed_now,
            mean_30m, mean_60m, mean_120m,
            slope_30m, slope_60m, slope_120m,
            persistence_30m, persistence_60m, persistence_120m,
            flips_30m, flips_60m, flips_120m,
            peak_abs_30m, peak_abs_60m, peak_abs_120m,
            acceleration_2h,
            sample_count_30m, sample_count_60m, sample_count_120m
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            snapshot_id, observed_at, str(item.get("symbol") or "").upper(), p_signed,
            s30["mean"], s60["mean"], s120["mean"],
            s30["slope"], s60["slope"], s120["slope"],
            s30["persistence"], s60["persistence"], s120["persistence"],
            s30["flips"], s60["flips"], s120["flips"],
            s30["peak_abs"], s60["peak_abs"], s120["peak_abs"],
            acceleration,
            s30["n"], s60["n"], s120["n"],
        ),
    )
    return snapshot_id


def run_once(con: sqlite3.Connection) -> int:
    payload = fetch_radar()
    radar = payload.get("radar") or []
    generated_at = payload.get("generated_at")
    count = 0
    for item in radar:
        if not item.get("symbol"):
            continue
        insert_item(con, item, generated_at)
        count += 1
    con.commit()
    print(f"[{utc_now()}] liquidation-pressure logger snapshots={count}", flush=True)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Log LiqHeat liquidation-pressure history for Bias V2 research.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    init_db(con)

    if args.once:
        run_once(con)
        return 0

    while True:
        started = time.time()
        try:
            run_once(con)
        except Exception as exc:
            print(f"[{utc_now()}] liquidation-pressure logger ERROR {type(exc).__name__}: {exc}", flush=True)
        delay = max(1.0, float(args.interval) - (time.time() - started))
        time.sleep(delay)


if __name__ == "__main__":
    raise SystemExit(main())
