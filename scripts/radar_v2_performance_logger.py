#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "monitoring" / "radar_v2_performance.sqlite"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "radar_v2_performance"
DEFAULT_API_URL = os.getenv("RADAR_API_URL", "http://127.0.0.1:8000")
DEFAULT_POLL_SECONDS = int(os.getenv("RADAR_LOGGER_POLL_SECONDS", "75"))
BINANCE_URL = "https://fapi.binance.com/fapi/v1/klines"
HORIZONS = (15, 30, 60, 120, 240)
TARGETS_BPS = (25, 50, 100)
STOP_BPS = 50
ACTIONABLE = {"WATCH", "HIGH", "CRITICAL", "CONFLICT"}

STOP_REQUESTED = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def event_key(item: dict[str, Any], generated_at: str) -> str:
    matrix = item.get("matrix") or {}
    raw = "|".join(
        [
            str(item.get("symbol")),
            str(item.get("snapshot_id")),
            str(item.get("raw_prediction")),
            str(item.get("opportunity")),
            str(item.get("matrix_gate")),
            str(matrix.get("regime")),
            str(matrix.get("alignment_score")),
            str(generated_at),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    initialize_schema(connection)
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS radar_signals (
            event_key TEXT PRIMARY KEY,
            captured_at TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            snapshot_id TEXT,
            source_logged_at TEXT,
            entry_price REAL NOT NULL,
            forecast_direction INTEGER NOT NULL,
            raw_prediction TEXT NOT NULL,
            opportunity TEXT NOT NULL,
            radar_score REAL NOT NULL,
            liquidity_pressure_score REAL,
            direction_confidence REAL,
            matrix_direction INTEGER,
            matrix_direction_label TEXT,
            matrix_regime TEXT,
            matrix_alignment REAL,
            matrix_gate TEXT,
            matrix_agreement INTEGER,
            nearest_side TEXT,
            topology_imbalance REAL,
            upper_distance_pct REAL,
            lower_distance_pct REAL,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS radar_outcomes (
            event_key TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            evaluated_at TEXT NOT NULL,
            candle_count INTEGER NOT NULL,
            terminal_return_bps REAL,
            directional_return_bps REAL,
            mfe_bps REAL,
            mae_bps REAL,
            time_to_mfe_minutes INTEGER,
            time_to_mae_minutes INTEGER,
            target_25_hit INTEGER,
            target_50_hit INTEGER,
            target_100_hit INTEGER,
            target_25_minutes INTEGER,
            target_50_minutes INTEGER,
            target_100_minutes INTEGER,
            target_50_before_stop50 INTEGER,
            PRIMARY KEY (event_key, horizon_minutes),
            FOREIGN KEY (event_key) REFERENCES radar_signals(event_key)
        );

        CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
            ON radar_signals(symbol, generated_at);
        CREATE INDEX IF NOT EXISTS idx_signals_opportunity
            ON radar_signals(opportunity, generated_at);
        CREATE INDEX IF NOT EXISTS idx_outcomes_horizon
            ON radar_outcomes(horizon_minutes);
        """
    )
    connection.commit()


def fetch_radar(session: requests.Session, api_url: str) -> dict[str, Any]:
    response = session.get(f"{api_url.rstrip('/')}/radar", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("radar"), list):
        raise RuntimeError("Invalid radar payload")
    return payload


def store_payload(connection: sqlite3.Connection, payload: dict[str, Any]) -> int:
    generated_at = str(payload.get("generated_at") or utc_iso())
    captured_at = utc_iso()
    inserted = 0

    for item in payload.get("radar", []):
        symbol = str(item.get("symbol") or "").upper()
        entry_price = safe_float(item.get("current_price"))
        radar_score = safe_float(item.get("radar_score"))
        raw_prediction = str(item.get("raw_prediction") or "")
        if not symbol or entry_price is None or entry_price <= 0 or radar_score is None:
            continue
        if raw_prediction not in {"SHORT_SQUEEZE", "LONG_SQUEEZE"}:
            continue

        forecast_direction = 1 if raw_prediction == "SHORT_SQUEEZE" else -1
        matrix = item.get("matrix") or {}
        topology = item.get("topology") or {}
        key = event_key(item, generated_at)

        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO radar_signals (
                event_key, captured_at, generated_at, symbol, snapshot_id,
                source_logged_at, entry_price, forecast_direction,
                raw_prediction, opportunity, radar_score,
                liquidity_pressure_score, direction_confidence,
                matrix_direction, matrix_direction_label, matrix_regime,
                matrix_alignment, matrix_gate, matrix_agreement,
                nearest_side, topology_imbalance, upper_distance_pct,
                lower_distance_pct, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                captured_at,
                generated_at,
                symbol,
                item.get("snapshot_id"),
                item.get("logged_at"),
                entry_price,
                forecast_direction,
                raw_prediction,
                str(item.get("opportunity") or "NORMAL"),
                radar_score,
                safe_float(item.get("liquidity_pressure_score")),
                safe_float(item.get("direction_confidence")),
                matrix.get("direction"),
                matrix.get("direction_label"),
                matrix.get("regime"),
                safe_float(matrix.get("alignment_score")),
                item.get("matrix_gate"),
                None if item.get("matrix_agreement") is None else int(bool(item.get("matrix_agreement"))),
                topology.get("nearest_side"),
                safe_float(topology.get("topology_imbalance")),
                safe_float(topology.get("upper_distance_pct")),
                safe_float(topology.get("lower_distance_pct")),
                json.dumps(item, separators=(",", ":"), ensure_ascii=False),
            ),
        )
        inserted += int(cursor.rowcount > 0)

    connection.commit()
    return inserted


def fetch_klines(
    session: requests.Session,
    symbol: str,
    start: datetime,
    horizon_minutes: int,
) -> list[list[Any]]:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int((start + timedelta(minutes=horizon_minutes + 2)).timestamp() * 1000)
    response = session.get(
        BINANCE_URL,
        params={
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": min(horizon_minutes + 3, 1000),
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Binance response for {symbol}")
    return payload


def evaluate_path(
    entry_price: float,
    direction: int,
    klines: list[list[Any]],
    horizon_minutes: int,
) -> dict[str, Any] | None:
    candles = klines[:horizon_minutes]
    if len(candles) < horizon_minutes:
        return None

    highs = [float(row[2]) for row in candles]
    lows = [float(row[3]) for row in candles]
    closes = [float(row[4]) for row in candles]

    if direction == 1:
        favorable = [(high / entry_price - 1.0) * 10000.0 for high in highs]
        adverse = [(low / entry_price - 1.0) * 10000.0 for low in lows]
        terminal = (closes[-1] / entry_price - 1.0) * 10000.0
    else:
        favorable = [(entry_price / low - 1.0) * 10000.0 for low in lows]
        adverse = [(entry_price / high - 1.0) * 10000.0 for high in highs]
        terminal = (entry_price / closes[-1] - 1.0) * 10000.0

    mfe = max(favorable)
    mae = min(adverse)
    time_to_mfe = favorable.index(mfe) + 1
    time_to_mae = adverse.index(mae) + 1

    result: dict[str, Any] = {
        "candle_count": len(candles),
        "terminal_return_bps": terminal if direction == 1 else -((closes[-1] / entry_price - 1.0) * 10000.0),
        "directional_return_bps": terminal,
        "mfe_bps": mfe,
        "mae_bps": mae,
        "time_to_mfe_minutes": time_to_mfe,
        "time_to_mae_minutes": time_to_mae,
    }

    first_hits: dict[int, int | None] = {}
    for target in TARGETS_BPS:
        hit = next((index + 1 for index, value in enumerate(favorable) if value >= target), None)
        first_hits[target] = hit
        result[f"target_{target}_hit"] = int(hit is not None)
        result[f"target_{target}_minutes"] = hit

    stop_hit = next((index + 1 for index, value in enumerate(adverse) if value <= -STOP_BPS), None)
    target_50 = first_hits[50]
    result["target_50_before_stop50"] = int(
        target_50 is not None and (stop_hit is None or target_50 < stop_hit)
    )
    return result


def due_signals(connection: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    now = utc_now()
    rows = connection.execute(
        """
        SELECT s.*
        FROM radar_signals s
        WHERE EXISTS (
            SELECT 1
            FROM (
                SELECT 15 AS horizon UNION ALL SELECT 30 UNION ALL SELECT 60
                UNION ALL SELECT 120 UNION ALL SELECT 240
            ) h
            WHERE datetime(s.generated_at, '+' || h.horizon || ' minutes') <= datetime(?)
              AND NOT EXISTS (
                  SELECT 1 FROM radar_outcomes o
                  WHERE o.event_key = s.event_key
                    AND o.horizon_minutes = h.horizon
              )
        )
        ORDER BY s.generated_at ASC
        LIMIT ?
        """,
        (now.isoformat(), limit),
    ).fetchall()
    return rows


def evaluate_due(
    connection: sqlite3.Connection,
    session: requests.Session,
    limit: int = 100,
) -> int:
    evaluated = 0
    now = utc_now()

    for row in due_signals(connection, limit=limit):
        signal_time = parse_time(row["generated_at"])
        if signal_time is None:
            continue
        for horizon in HORIZONS:
            mature_at = signal_time + timedelta(minutes=horizon)
            if mature_at > now:
                continue
            exists = connection.execute(
                "SELECT 1 FROM radar_outcomes WHERE event_key=? AND horizon_minutes=?",
                (row["event_key"], horizon),
            ).fetchone()
            if exists:
                continue
            try:
                klines = fetch_klines(session, row["symbol"], signal_time, horizon)
                outcome = evaluate_path(
                    float(row["entry_price"]),
                    int(row["forecast_direction"]),
                    klines,
                    horizon,
                )
            except Exception as exc:
                print(
                    f"[{utc_iso()}] outcome fetch failed {row['symbol']} {horizon}m: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            if outcome is None:
                continue

            connection.execute(
                """
                INSERT OR IGNORE INTO radar_outcomes (
                    event_key, horizon_minutes, evaluated_at, candle_count,
                    terminal_return_bps, directional_return_bps, mfe_bps, mae_bps,
                    time_to_mfe_minutes, time_to_mae_minutes,
                    target_25_hit, target_50_hit, target_100_hit,
                    target_25_minutes, target_50_minutes, target_100_minutes,
                    target_50_before_stop50
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["event_key"], horizon, utc_iso(), outcome["candle_count"],
                    outcome["terminal_return_bps"], outcome["directional_return_bps"],
                    outcome["mfe_bps"], outcome["mae_bps"],
                    outcome["time_to_mfe_minutes"], outcome["time_to_mae_minutes"],
                    outcome["target_25_hit"], outcome["target_50_hit"], outcome["target_100_hit"],
                    outcome["target_25_minutes"], outcome["target_50_minutes"], outcome["target_100_minutes"],
                    outcome["target_50_before_stop50"],
                ),
            )
            evaluated += 1
        connection.commit()
    return evaluated


def write_reports(connection: sqlite3.Connection, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = connection.execute(
        """
        SELECT
            s.opportunity,
            s.matrix_gate,
            o.horizon_minutes,
            COUNT(*) AS observations,
            ROUND(AVG(s.radar_score), 4) AS mean_radar_score,
            ROUND(AVG(o.directional_return_bps), 4) AS mean_directional_return_bps,
            ROUND(AVG(o.mfe_bps), 4) AS mean_mfe_bps,
            ROUND(AVG(o.mae_bps), 4) AS mean_mae_bps,
            ROUND(AVG(o.directional_return_bps > 0), 6) AS terminal_win_rate,
            ROUND(AVG(o.target_25_hit), 6) AS target_25_hit_rate,
            ROUND(AVG(o.target_50_hit), 6) AS target_50_hit_rate,
            ROUND(AVG(o.target_100_hit), 6) AS target_100_hit_rate,
            ROUND(AVG(o.target_50_before_stop50), 6) AS target_50_before_stop50_rate
        FROM radar_signals s
        JOIN radar_outcomes o ON o.event_key = s.event_key
        GROUP BY s.opportunity, s.matrix_gate, o.horizon_minutes
        ORDER BY o.horizon_minutes, s.opportunity, s.matrix_gate
        """
    ).fetchall()

    csv_path = report_dir / "radar_v2_performance_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else [
        "opportunity", "matrix_gate", "horizon_minutes", "observations"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(dict(row))

    counts = connection.execute(
        """
        SELECT
            COUNT(*) AS signals,
            SUM(opportunity IN ('WATCH','HIGH','CRITICAL','CONFLICT')) AS actionable_signals,
            MIN(generated_at) AS first_signal,
            MAX(generated_at) AS last_signal
        FROM radar_signals
        """
    ).fetchone()
    outcome_count = connection.execute("SELECT COUNT(*) AS count FROM radar_outcomes").fetchone()["count"]

    report = {
        "status": "online",
        "generated_at": utc_iso(),
        "database": str(DEFAULT_DB),
        "signals": int(counts["signals"] or 0),
        "actionable_signals": int(counts["actionable_signals"] or 0),
        "outcomes": int(outcome_count or 0),
        "first_signal": counts["first_signal"],
        "last_signal": counts["last_signal"],
        "horizons_minutes": list(HORIZONS),
        "targets_bps": list(TARGETS_BPS),
        "summary_csv": str(csv_path),
    }
    (report_dir / "radar_v2_performance_status.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def status(connection: sqlite3.Connection) -> dict[str, Any]:
    signals = connection.execute("SELECT COUNT(*) AS count FROM radar_signals").fetchone()["count"]
    outcomes = connection.execute("SELECT COUNT(*) AS count FROM radar_outcomes").fetchone()["count"]
    pending = connection.execute(
        """
        SELECT COUNT(*) AS count FROM radar_signals s
        WHERE NOT EXISTS (
            SELECT 1 FROM radar_outcomes o
            WHERE o.event_key=s.event_key AND o.horizon_minutes=240
        )
        """
    ).fetchone()["count"]
    return {"signals": int(signals), "outcomes": int(outcomes), "pending_240m": int(pending)}


def handle_signal(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    session = requests.Session()
    session.headers.update({"User-Agent": "LiqHeat-Radar-V2-Performance-Logger/1.0"})
    connection = connect(args.database)

    print("=" * 96)
    print("LIQHEAT RADAR V2 PERFORMANCE LOGGER")
    print("=" * 96)
    print(f"API      : {args.api_url}")
    print(f"Database : {args.database}")
    print(f"Poll     : {args.poll_seconds}s")
    print(f"Horizons : {HORIZONS}")
    print()

    while not STOP_REQUESTED:
        cycle_started = time.time()
        try:
            payload = fetch_radar(session, args.api_url)
            inserted = store_payload(connection, payload)
            evaluated = evaluate_due(connection, session, limit=args.evaluate_limit)
            write_reports(connection, args.report_dir)
            current = status(connection)
            print(
                f"[{utc_iso()}] inserted={inserted} evaluated={evaluated} "
                f"signals={current['signals']} outcomes={current['outcomes']} "
                f"pending240={current['pending_240m']}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{utc_iso()}] cycle failed: {type(exc).__name__}: {exc}", flush=True)

        if args.once:
            break
        elapsed = time.time() - cycle_started
        sleep_for = max(1.0, args.poll_seconds - elapsed)
        end = time.time() + sleep_for
        while not STOP_REQUESTED and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    connection.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log and evaluate LiqHeat Radar V2 signals")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--evaluate-limit", type=int, default=100)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
