#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HORIZONS = (5, 15, 30, 60)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).isoformat()


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "LiqHeat-Risk-Research-Logger/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def matrix_label(item: dict[str, Any], timeframe: str) -> str | None:
    matrix = item.get("matrix") or {}
    tf = (matrix.get("timeframes") or {}).get(timeframe) or {}
    value = tf.get("trend_label")
    return str(value) if value else None


def scalp_signal(item: dict[str, Any]) -> str | None:
    label = matrix_label(item, "1m")
    if label == "BUY":
        return "LONG"
    if label == "SELL":
        return "SHORT"
    return None


def risk_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    risk = item.get("ai_market_risk") or {}
    available = bool(risk.get("available"))
    return {
        "risk_available": 1 if available else 0,
        "risk_score": as_float(risk.get("risk_score")),
        "raw_risk_score": as_float(risk.get("raw_risk_score")),
        "risk_band": risk.get("risk_band") if available else None,
        "p_high": as_float(risk.get("p_high")),
        "p_extreme": as_float(risk.get("p_extreme")),
    }


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    create_schema(conn)
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    horizon_columns = []
    for h in HORIZONS:
        horizon_columns.extend([
            f"price_{h}m REAL",
            f"return_{h}m_bps REAL",
            f"signed_return_{h}m_bps REAL",
            f"win_{h}m INTEGER",
            f"resolved_{h}m_at TEXT",
        ])
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS signal_events (
            event_id TEXT PRIMARY KEY,
            event_ts TEXT NOT NULL,
            event_epoch REAL NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal TEXT NOT NULL,
            entry_price REAL NOT NULL,
            matrix_1m TEXT,
            matrix_15m TEXT,
            matrix_1h TEXT,
            matrix_4h TEXT,
            matrix_1d TEXT,
            matrix_alignment REAL,
            liquidity_pressure REAL,
            risk_available INTEGER NOT NULL DEFAULT 0,
            risk_score REAL,
            raw_risk_score REAL,
            risk_band TEXT,
            p_high REAL,
            p_extreme REAL,
            radar_generated_at TEXT,
            source_age_seconds REAL,
            min_seen_price REAL,
            max_seen_price REAL,
            last_seen_price REAL,
            last_seen_at TEXT,
            {', '.join(horizon_columns)}
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_events_symbol_ts ON signal_events(symbol, event_epoch)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_events_type ON signal_events(event_type)")
    conn.commit()


def insert_event(conn: sqlite3.Connection, item: dict[str, Any], event_type: str) -> str | None:
    signal = scalp_signal(item)
    price = as_float(item.get("current_price"))
    symbol = item.get("symbol")
    if not signal or not symbol or price is None or price <= 0:
        return None

    matrix = item.get("matrix") or {}
    risk = risk_snapshot(item)
    event_id = str(uuid.uuid4())
    now = utc_now()
    alignment = as_float(matrix.get("alignment_score"))
    liquidity_pressure = as_float(item.get("liquidity_pressure_score"))
    if liquidity_pressure is None:
        lp = as_float(item.get("liquidity_pressure"))
        liquidity_pressure = lp * 100.0 if lp is not None else None

    conn.execute(
        """
        INSERT INTO signal_events (
            event_id,event_ts,event_epoch,event_type,symbol,signal,entry_price,
            matrix_1m,matrix_15m,matrix_1h,matrix_4h,matrix_1d,matrix_alignment,
            liquidity_pressure,risk_available,risk_score,raw_risk_score,risk_band,
            p_high,p_extreme,radar_generated_at,source_age_seconds,min_seen_price,
            max_seen_price,last_seen_price,last_seen_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            event_id, iso(now), now.timestamp(), event_type, symbol, signal, price,
            matrix_label(item, "1m"), matrix_label(item, "15m"), matrix_label(item, "1h"),
            matrix_label(item, "4h"), matrix_label(item, "1d"), alignment,
            liquidity_pressure, risk["risk_available"], risk["risk_score"],
            risk["raw_risk_score"], risk["risk_band"], risk["p_high"], risk["p_extreme"],
            item.get("generated_at"), as_float(item.get("age_seconds")), price, price, price, iso(now),
        ),
    )
    conn.commit()
    print(
        f"[{iso(now)}] {event_type} {symbol} {signal} @ {price:g} "
        f"risk={risk['risk_band'] or 'N/A'} score={risk['risk_score']} raw={risk['raw_risk_score']}",
        flush=True,
    )
    return event_id


def update_open_events(conn: sqlite3.Connection, prices: dict[str, float], now: datetime) -> None:
    rows = conn.execute(
        """
        SELECT * FROM signal_events
        WHERE resolved_60m_at IS NULL AND event_epoch >= ?
        """,
        (now.timestamp() - 3 * 3600,),
    ).fetchall()

    for row in rows:
        price = prices.get(row["symbol"])
        if price is None or price <= 0:
            continue

        min_seen = min(row["min_seen_price"] if row["min_seen_price"] is not None else price, price)
        max_seen = max(row["max_seen_price"] if row["max_seen_price"] is not None else price, price)
        conn.execute(
            """
            UPDATE signal_events
            SET min_seen_price=?, max_seen_price=?, last_seen_price=?, last_seen_at=?
            WHERE event_id=?
            """,
            (min_seen, max_seen, price, iso(now), row["event_id"]),
        )

        elapsed_minutes = (now.timestamp() - row["event_epoch"]) / 60.0
        for h in HORIZONS:
            if elapsed_minutes < h or row[f"resolved_{h}m_at"] is not None:
                continue
            raw_bps = (price / row["entry_price"] - 1.0) * 10000.0
            signed_bps = raw_bps if row["signal"] == "LONG" else -raw_bps
            win = 1 if signed_bps > 0 else 0
            conn.execute(
                f"""
                UPDATE signal_events
                SET price_{h}m=?, return_{h}m_bps=?, signed_return_{h}m_bps=?,
                    win_{h}m=?, resolved_{h}m_at=?
                WHERE event_id=?
                """,
                (price, raw_bps, signed_bps, win, iso(now), row["event_id"]),
            )
            print(
                f"[{iso(now)}] RESOLVE {h}m {row['symbol']} {row['signal']} "
                f"signed={signed_bps:+.1f}bps win={win} risk={row['risk_band'] or 'N/A'}",
                flush=True,
            )
    conn.commit()


def load_last_signals(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT s.symbol, s.signal
        FROM signal_events s
        JOIN (
          SELECT symbol, MAX(event_epoch) AS max_epoch
          FROM signal_events GROUP BY symbol
        ) x ON x.symbol=s.symbol AND x.max_epoch=s.event_epoch
        """
    ).fetchall()
    return {row["symbol"]: row["signal"] for row in rows}


def run(args: argparse.Namespace) -> None:
    db_path = Path(args.db).expanduser().resolve()
    conn = connect_db(db_path)
    last_signals = load_last_signals(conn)
    print(f"Signal/risk logger started: {db_path}", flush=True)
    print(f"Polling {args.url} every {args.interval:g}s", flush=True)

    while True:
        started = time.monotonic()
        try:
            payload = fetch_json(args.url, args.timeout)
            radar = payload.get("radar") or []
            now = utc_now()
            prices: dict[str, float] = {}

            for item in radar:
                symbol = item.get("symbol")
                price = as_float(item.get("current_price"))
                if symbol and price is not None:
                    prices[str(symbol)] = price

                signal = scalp_signal(item)
                if not symbol or not signal:
                    continue

                previous = last_signals.get(str(symbol))
                if previous is None:
                    insert_event(conn, item, "BOOTSTRAP")
                    last_signals[str(symbol)] = signal
                elif previous != signal:
                    insert_event(conn, item, "FLIP")
                    last_signals[str(symbol)] = signal

            update_open_events(conn, prices, now)

        except KeyboardInterrupt:
            print("Stopping logger.", flush=True)
            break
        except Exception as exc:
            print(f"[{iso()}] ERROR {type(exc).__name__}: {exc}", flush=True)

        elapsed = time.monotonic() - started
        time.sleep(max(0.5, args.interval - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Log 1m Matrix signal flips with live AI risk and forward outcomes")
    parser.add_argument("--url", default="http://127.0.0.1:8000/radar")
    parser.add_argument("--db", default="data/research/radar_signal_risk/radar_signal_risk.sqlite3")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
