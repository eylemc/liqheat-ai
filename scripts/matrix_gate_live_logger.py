#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_API = "http://127.0.0.1:8000/radar"
DEFAULT_DB = Path("data/research/matrix_gate_live/matrix_gate_live.sqlite3")
DEFAULT_INTERVAL = 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "liqheat-matrix-gate-logger/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS matrix_gate_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at TEXT NOT NULL,
            generated_at TEXT,
            symbol TEXT NOT NULL,
            timeframe TEXT,
            available INTEGER NOT NULL,
            matrix_trend TEXT,
            bars_since_flip INTEGER,
            regime_score REAL,
            threshold REAL,
            status TEXT,
            risk_level TEXT,
            latest_flip_side TEXT,
            latest_flip_close_time TEXT,
            latest_flip_score REAL,
            latest_flip_status TEXT,
            latest_flip_valid INTEGER,
            er REAL,
            channel_percentile REAL,
            normalized_displacement REAL,
            er_rank REAL,
            channel_rank REAL,
            norm_disp_rank REAL,
            research_status TEXT,
            research_spec_json TEXT,
            payload_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_matrix_gate_snapshots_symbol_time
        ON matrix_gate_snapshots(symbol, observed_at);

        CREATE TABLE IF NOT EXISTS matrix_gate_flips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT,
            side TEXT NOT NULL,
            close_time TEXT NOT NULL,
            score REAL,
            threshold REAL,
            status TEXT,
            risk_level TEXT,
            valid INTEGER,
            er REAL,
            channel_percentile REAL,
            normalized_displacement REAL,
            er_rank REAL,
            channel_rank REAL,
            norm_disp_rank REAL,
            bars_since_flip INTEGER,
            research_status TEXT,
            research_spec_json TEXT,
            source TEXT,
            UNIQUE(symbol, timeframe, side, close_time)
        );

        CREATE INDEX IF NOT EXISTS idx_matrix_gate_flips_symbol_close
        ON matrix_gate_flips(symbol, close_time);
        """
    )
    return con


def safe_bool(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def log_payload(con: sqlite3.Connection, payload: dict[str, Any]) -> tuple[int, int]:
    observed_at = utc_now()
    generated_at = payload.get("generated_at")
    rows = payload.get("radar") or []
    snapshot_count = 0
    new_flip_count = 0

    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        gate = row.get("matrix_regime_gate") or {}
        if not gate:
            continue

        flip = gate.get("latest_flip") or {}
        research_spec = gate.get("research_spec") or {}
        risk_level = gate.get("risk_level")
        if not risk_level:
            risk_level = "LOW RISK" if gate.get("status") == "VALID" else "HIGH RISK"

        con.execute(
            """
            INSERT INTO matrix_gate_snapshots (
                observed_at, generated_at, symbol, timeframe, available,
                matrix_trend, bars_since_flip, regime_score, threshold, status, risk_level,
                latest_flip_side, latest_flip_close_time, latest_flip_score,
                latest_flip_status, latest_flip_valid,
                er, channel_percentile, normalized_displacement,
                er_rank, channel_rank, norm_disp_rank,
                research_status, research_spec_json, payload_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                observed_at,
                generated_at,
                symbol,
                gate.get("timeframe"),
                1 if gate.get("available") else 0,
                gate.get("matrix_trend"),
                gate.get("bars_since_flip"),
                gate.get("regime_score"),
                gate.get("threshold"),
                gate.get("status"),
                risk_level,
                flip.get("side"),
                flip.get("close_time"),
                flip.get("score"),
                flip.get("status"),
                safe_bool(flip.get("valid")),
                flip.get("er"),
                flip.get("channel_percentile"),
                flip.get("normalized_displacement"),
                flip.get("er_rank"),
                flip.get("channel_rank"),
                flip.get("norm_disp_rank"),
                gate.get("research_status"),
                json.dumps(research_spec, sort_keys=True, separators=(",", ":")),
                json.dumps(gate, sort_keys=True, separators=(",", ":")),
            ),
        )
        snapshot_count += 1

        side = flip.get("side")
        close_time = flip.get("close_time")
        if side and close_time:
            before = con.total_changes
            con.execute(
                """
                INSERT INTO matrix_gate_flips (
                    first_seen_at, last_seen_at, symbol, timeframe, side, close_time,
                    score, threshold, status, risk_level, valid,
                    er, channel_percentile, normalized_displacement,
                    er_rank, channel_rank, norm_disp_rank, bars_since_flip,
                    research_status, research_spec_json, source
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol, timeframe, side, close_time) DO UPDATE SET
                    last_seen_at=excluded.last_seen_at,
                    score=excluded.score,
                    threshold=excluded.threshold,
                    status=excluded.status,
                    risk_level=excluded.risk_level,
                    valid=excluded.valid,
                    er=excluded.er,
                    channel_percentile=excluded.channel_percentile,
                    normalized_displacement=excluded.normalized_displacement,
                    er_rank=excluded.er_rank,
                    channel_rank=excluded.channel_rank,
                    norm_disp_rank=excluded.norm_disp_rank,
                    bars_since_flip=excluded.bars_since_flip,
                    research_status=excluded.research_status,
                    research_spec_json=excluded.research_spec_json,
                    source=excluded.source
                """,
                (
                    observed_at,
                    observed_at,
                    symbol,
                    gate.get("timeframe"),
                    side,
                    close_time,
                    flip.get("score"),
                    gate.get("threshold"),
                    flip.get("status") or gate.get("status"),
                    risk_level,
                    safe_bool(flip.get("valid")),
                    flip.get("er"),
                    flip.get("channel_percentile"),
                    flip.get("normalized_displacement"),
                    flip.get("er_rank"),
                    flip.get("channel_rank"),
                    flip.get("norm_disp_rank"),
                    gate.get("bars_since_flip"),
                    gate.get("research_status"),
                    json.dumps(research_spec, sort_keys=True, separators=(",", ":")),
                    flip.get("source"),
                ),
            )
            if con.total_changes > before:
                existing = con.execute(
                    "SELECT first_seen_at FROM matrix_gate_flips WHERE symbol=? AND timeframe=? AND side=? AND close_time=?",
                    (symbol, gate.get("timeframe"), side, close_time),
                ).fetchone()
                if existing and existing[0] == observed_at:
                    new_flip_count += 1

    con.commit()
    return snapshot_count, new_flip_count


def run_once(api_url: str, db_path: Path) -> None:
    payload = fetch_json(api_url)
    with connect_db(db_path) as con:
        snapshots, new_flips = log_payload(con, payload)
    print(f"{utc_now()} matrix-gate logger snapshots={snapshots} new_flips={new_flips}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Log LiqHeat Matrix regime-gate live research data.")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if args.once:
        run_once(args.api, db_path)
        return 0

    while True:
        try:
            run_once(args.api, db_path)
        except Exception as exc:
            print(f"{utc_now()} matrix-gate logger ERROR {type(exc).__name__}: {exc}", flush=True)
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
