#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "monitoring" / "radar_v2_performance.sqlite"
DEFAULT_API_URL = os.getenv("RADAR_API_URL", "http://127.0.0.1:8000")
DEFAULT_POLL_SECONDS = int(os.getenv("RADAR_FINAL_LOGGER_POLL_SECONDS", "75"))
BINANCE_URL = "https://fapi.binance.com/fapi/v1/klines"

MIN_DIRECTIONAL_RADAR_SCORE = 40.0
ENTER_CONFIDENCE = 0.58
EXIT_CONFIDENCE = 0.56
REVERSE_CONFIDENCE = 0.68
ENTER_CONFIRMATIONS = 3
EXIT_CONFIRMATIONS = 2
REVERSE_CONFIRMATIONS = 3
MIN_HOLD_SECONDS = 5 * 60
HORIZONS = (15, 30, 60, 120, 240)
TARGETS_BPS = (25, 50, 100)
STOP_BPS = 50

STOP_REQUESTED = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


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


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS radar_final_state (
            symbol TEXT PRIMARY KEY,
            outcome TEXT NOT NULL,
            pending TEXT,
            pending_confirmations INTEGER NOT NULL DEFAULT 0,
            weak_confirmations INTEGER NOT NULL DEFAULT 0,
            changed_at TEXT,
            last_snapshot_key TEXT,
            confidence REAL,
            upward_share REAL,
            radar_score REAL,
            score_gate TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS radar_final_observations (
            observation_key TEXT PRIMARY KEY,
            captured_at TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            snapshot_key TEXT NOT NULL,
            current_price REAL,
            raw_radar_score REAL,
            direction_confidence REAL,
            candidate_outcome TEXT NOT NULL,
            final_outcome TEXT NOT NULL,
            pending_direction TEXT,
            pending_confirmations INTEGER NOT NULL,
            weak_confirmations INTEGER NOT NULL,
            score_gate TEXT NOT NULL,
            matrix_gate TEXT,
            matrix_direction TEXT,
            matrix_regime TEXT,
            matrix_alignment REAL,
            opportunity TEXT,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS radar_final_signals (
            signal_key TEXT PRIMARY KEY,
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            symbol TEXT NOT NULL,
            direction INTEGER NOT NULL,
            outcome_label TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            entry_radar_score REAL NOT NULL,
            entry_confidence REAL,
            matrix_gate TEXT,
            matrix_direction TEXT,
            matrix_regime TEXT,
            matrix_alignment REAL,
            opportunity TEXT,
            source_snapshot_key TEXT NOT NULL,
            close_reason TEXT,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS radar_final_outcomes (
            signal_key TEXT NOT NULL,
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
            PRIMARY KEY (signal_key, horizon_minutes)
        );

        CREATE INDEX IF NOT EXISTS idx_final_obs_symbol_time
            ON radar_final_observations(symbol, generated_at);
        CREATE INDEX IF NOT EXISTS idx_final_signals_symbol_time
            ON radar_final_signals(symbol, opened_at);
        CREATE INDEX IF NOT EXISTS idx_final_outcomes_horizon
            ON radar_final_outcomes(horizon_minutes);
        """
    )
    connection.commit()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    initialize_schema(connection)
    return connection


def fetch_radar(session: requests.Session, api_url: str) -> dict[str, Any]:
    response = session.get(f"{api_url.rstrip('/')}/radar", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("radar"), list):
        raise RuntimeError("Invalid radar payload")
    return payload


def directional_evidence(item: dict[str, Any]) -> tuple[str, float, float]:
    probabilities = item.get("probabilities") or {}
    upward = safe_float(probabilities.get("short_squeeze"), 0.0) or 0.0
    downward = safe_float(probabilities.get("long_squeeze"), 0.0) or 0.0
    total = upward + downward
    if total <= 0:
        return "NEUTRAL", 0.0, 0.5
    upward_share = upward / total
    downward_share = downward / total
    candidate = "UPWARD" if upward_share >= downward_share else "DOWNWARD"
    return candidate, max(upward_share, downward_share), upward_share


def load_state(connection: sqlite3.Connection, symbol: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM radar_final_state WHERE symbol=?", (symbol,)
    ).fetchone()
    if row is None:
        return {
            "outcome": "NEUTRAL",
            "pending": None,
            "pending_confirmations": 0,
            "weak_confirmations": 0,
            "changed_at": None,
            "last_snapshot_key": None,
        }
    return dict(row)


def save_state(connection: sqlite3.Connection, symbol: str, state: dict[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO radar_final_state (
            symbol, outcome, pending, pending_confirmations, weak_confirmations,
            changed_at, last_snapshot_key, confidence, upward_share, radar_score,
            score_gate, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            outcome=excluded.outcome,
            pending=excluded.pending,
            pending_confirmations=excluded.pending_confirmations,
            weak_confirmations=excluded.weak_confirmations,
            changed_at=excluded.changed_at,
            last_snapshot_key=excluded.last_snapshot_key,
            confidence=excluded.confidence,
            upward_share=excluded.upward_share,
            radar_score=excluded.radar_score,
            score_gate=excluded.score_gate,
            updated_at=excluded.updated_at
        """,
        (
            symbol,
            state["outcome"],
            state.get("pending"),
            int(state.get("pending_confirmations", 0)),
            int(state.get("weak_confirmations", 0)),
            state.get("changed_at"),
            state.get("last_snapshot_key"),
            state.get("confidence"),
            state.get("upward_share"),
            state.get("radar_score"),
            state.get("score_gate"),
            utc_iso(),
        ),
    )


def snapshot_key(payload: dict[str, Any], item: dict[str, Any]) -> str:
    return str(item.get("logged_at") or payload.get("generated_at") or payload.get("last_success_at") or "")


def stabilize_item(
    connection: sqlite3.Connection,
    payload: dict[str, Any],
    item: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    symbol = str(item.get("symbol") or "UNKNOWN").upper()
    score = safe_float(item.get("radar_score"), 0.0) or 0.0
    candidate, confidence, upward_share = directional_evidence(item)
    previous = load_state(connection, symbol)
    current_snapshot_key = snapshot_key(payload, item)

    if current_snapshot_key and current_snapshot_key == previous.get("last_snapshot_key"):
        return previous, False

    outcome = str(previous.get("outcome") or "NEUTRAL")
    pending = previous.get("pending")
    pending_confirmations = int(previous.get("pending_confirmations") or 0)
    weak_confirmations = int(previous.get("weak_confirmations") or 0)
    changed_at = previous.get("changed_at")
    score_gate = "PASS" if score >= MIN_DIRECTIONAL_RADAR_SCORE else "BLOCK"

    if score_gate == "BLOCK":
        outcome = "NEUTRAL"
        pending = None
        pending_confirmations = 0
        weak_confirmations = 0
    else:
        now = utc_now()
        changed_dt = parse_time(changed_at)
        hold_elapsed = changed_dt is None or (now - changed_dt).total_seconds() >= MIN_HOLD_SECONDS

        if outcome == "NEUTRAL":
            weak_confirmations = 0
            if confidence >= ENTER_CONFIDENCE:
                if pending == candidate:
                    pending_confirmations += 1
                else:
                    pending = candidate
                    pending_confirmations = 1
                if pending_confirmations >= ENTER_CONFIRMATIONS:
                    outcome = candidate
                    pending = None
                    pending_confirmations = 0
            else:
                pending = None
                pending_confirmations = 0
        elif candidate == outcome and confidence >= EXIT_CONFIDENCE:
            pending = None
            pending_confirmations = 0
            weak_confirmations = 0
        elif candidate != outcome and confidence >= REVERSE_CONFIDENCE and hold_elapsed:
            if pending == candidate:
                pending_confirmations += 1
            else:
                pending = candidate
                pending_confirmations = 1
            weak_confirmations = 0
            if pending_confirmations >= REVERSE_CONFIRMATIONS:
                outcome = candidate
                pending = None
                pending_confirmations = 0
        elif confidence < EXIT_CONFIDENCE and hold_elapsed:
            weak_confirmations += 1
            pending = None
            pending_confirmations = 0
            if weak_confirmations >= EXIT_CONFIRMATIONS:
                outcome = "NEUTRAL"
                weak_confirmations = 0
        else:
            pending = None
            pending_confirmations = 0
            weak_confirmations = 0

    changed = outcome != str(previous.get("outcome") or "NEUTRAL")
    state = {
        "outcome": outcome,
        "pending": pending,
        "pending_confirmations": pending_confirmations,
        "weak_confirmations": weak_confirmations,
        "changed_at": utc_iso() if changed else changed_at,
        "last_snapshot_key": current_snapshot_key,
        "confidence": round(confidence, 6),
        "upward_share": round(upward_share, 6),
        "radar_score": round(score, 6),
        "score_gate": score_gate,
        "candidate": candidate,
    }
    save_state(connection, symbol, state)
    return state, changed


def observation_key(symbol: str, snapshot: str) -> str:
    return hashlib.sha256(f"{symbol}|{snapshot}".encode()).hexdigest()


def store_observation(
    connection: sqlite3.Connection,
    payload: dict[str, Any],
    item: dict[str, Any],
    state: dict[str, Any],
) -> None:
    symbol = str(item.get("symbol") or "UNKNOWN").upper()
    snapshot = snapshot_key(payload, item)
    matrix = item.get("matrix") or {}
    connection.execute(
        """
        INSERT OR IGNORE INTO radar_final_observations (
            observation_key, captured_at, generated_at, symbol, snapshot_key,
            current_price, raw_radar_score, direction_confidence,
            candidate_outcome, final_outcome, pending_direction,
            pending_confirmations, weak_confirmations, score_gate,
            matrix_gate, matrix_direction, matrix_regime, matrix_alignment,
            opportunity, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            observation_key(symbol, snapshot),
            utc_iso(),
            str(payload.get("generated_at") or utc_iso()),
            symbol,
            snapshot,
            safe_float(item.get("current_price")),
            safe_float(item.get("radar_score")),
            state.get("confidence"),
            state.get("candidate") or directional_evidence(item)[0],
            state.get("outcome"),
            state.get("pending"),
            int(state.get("pending_confirmations", 0)),
            int(state.get("weak_confirmations", 0)),
            state.get("score_gate"),
            item.get("matrix_gate"),
            matrix.get("direction_label"),
            matrix.get("regime"),
            safe_float(matrix.get("alignment_score")),
            item.get("opportunity"),
            json.dumps(item, separators=(",", ":"), ensure_ascii=False),
        ),
    )


def active_signal(connection: sqlite3.Connection, symbol: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM radar_final_signals WHERE symbol=? AND closed_at IS NULL ORDER BY opened_at DESC LIMIT 1",
        (symbol,),
    ).fetchone()


def sync_signal_transition(
    connection: sqlite3.Connection,
    payload: dict[str, Any],
    item: dict[str, Any],
    state: dict[str, Any],
) -> None:
    symbol = str(item.get("symbol") or "UNKNOWN").upper()
    outcome = str(state.get("outcome") or "NEUTRAL")
    current = active_signal(connection, symbol)
    price = safe_float(item.get("current_price"))
    matrix = item.get("matrix") or {}
    snapshot = snapshot_key(payload, item)

    if current is not None and current["outcome_label"] != outcome:
        connection.execute(
            "UPDATE radar_final_signals SET closed_at=?, exit_price=?, close_reason=? WHERE signal_key=?",
            (utc_iso(), price, f"OUTCOME_CHANGED_TO_{outcome}", current["signal_key"]),
        )
        current = None

    if outcome not in {"UPWARD", "DOWNWARD"} or price is None or price <= 0:
        return
    if current is not None:
        return

    direction = 1 if outcome == "UPWARD" else -1
    signal_key = hashlib.sha256(f"{symbol}|{snapshot}|{outcome}".encode()).hexdigest()
    connection.execute(
        """
        INSERT OR IGNORE INTO radar_final_signals (
            signal_key, opened_at, symbol, direction, outcome_label, entry_price,
            entry_radar_score, entry_confidence, matrix_gate, matrix_direction,
            matrix_regime, matrix_alignment, opportunity, source_snapshot_key,
            payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal_key,
            str(payload.get("generated_at") or utc_iso()),
            symbol,
            direction,
            outcome,
            price,
            safe_float(item.get("radar_score"), 0.0),
            state.get("confidence"),
            item.get("matrix_gate"),
            matrix.get("direction_label"),
            matrix.get("regime"),
            safe_float(matrix.get("alignment_score")),
            item.get("opportunity"),
            snapshot,
            json.dumps(item, separators=(",", ":"), ensure_ascii=False),
        ),
    )


def process_payload(connection: sqlite3.Connection, payload: dict[str, Any]) -> tuple[int, int]:
    observations = 0
    transitions = 0
    for item in payload.get("radar", []):
        state, changed = stabilize_item(connection, payload, item)
        if state.get("last_snapshot_key") != snapshot_key(payload, item):
            continue
        store_observation(connection, payload, item, state)
        sync_signal_transition(connection, payload, item, state)
        observations += 1
        transitions += int(changed)
    connection.commit()
    return observations, transitions


def fetch_klines(session: requests.Session, symbol: str, start: datetime, horizon: int) -> list[list[Any]]:
    response = session.get(
        BINANCE_URL,
        params={
            "symbol": symbol,
            "interval": "1m",
            "startTime": int(start.timestamp() * 1000),
            "endTime": int((start + timedelta(minutes=horizon + 2)).timestamp() * 1000),
            "limit": min(horizon + 3, 1000),
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Binance response for {symbol}")
    return data


def evaluate_path(entry: float, direction: int, klines: list[list[Any]], horizon: int) -> dict[str, Any] | None:
    candles = klines[:horizon]
    if len(candles) < horizon:
        return None
    highs = [float(row[2]) for row in candles]
    lows = [float(row[3]) for row in candles]
    closes = [float(row[4]) for row in candles]
    if direction == 1:
        favorable = [(high / entry - 1) * 10000 for high in highs]
        adverse = [(low / entry - 1) * 10000 for low in lows]
        terminal = (closes[-1] / entry - 1) * 10000
    else:
        favorable = [(entry / low - 1) * 10000 for low in lows]
        adverse = [(entry / high - 1) * 10000 for high in highs]
        terminal = (entry / closes[-1] - 1) * 10000
    mfe, mae = max(favorable), min(adverse)
    result: dict[str, Any] = {
        "candle_count": len(candles),
        "terminal_return_bps": terminal,
        "directional_return_bps": terminal,
        "mfe_bps": mfe,
        "mae_bps": mae,
        "time_to_mfe_minutes": favorable.index(mfe) + 1,
        "time_to_mae_minutes": adverse.index(mae) + 1,
    }
    first_hits: dict[int, int | None] = {}
    for target in TARGETS_BPS:
        hit = next((i + 1 for i, value in enumerate(favorable) if value >= target), None)
        first_hits[target] = hit
        result[f"target_{target}_hit"] = int(hit is not None)
        result[f"target_{target}_minutes"] = hit
    stop = next((i + 1 for i, value in enumerate(adverse) if value <= -STOP_BPS), None)
    hit50 = first_hits[50]
    result["target_50_before_stop50"] = int(hit50 is not None and (stop is None or hit50 < stop))
    return result


def evaluate_due(connection: sqlite3.Connection, session: requests.Session) -> int:
    evaluated = 0
    now = utc_now()
    signals = connection.execute("SELECT * FROM radar_final_signals ORDER BY opened_at ASC").fetchall()
    for row in signals:
        opened = parse_time(row["opened_at"])
        if opened is None:
            continue
        for horizon in HORIZONS:
            if opened + timedelta(minutes=horizon) > now:
                continue
            exists = connection.execute(
                "SELECT 1 FROM radar_final_outcomes WHERE signal_key=? AND horizon_minutes=?",
                (row["signal_key"], horizon),
            ).fetchone()
            if exists:
                continue
            try:
                outcome = evaluate_path(
                    float(row["entry_price"]),
                    int(row["direction"]),
                    fetch_klines(session, row["symbol"], opened, horizon),
                    horizon,
                )
            except Exception as exc:
                print(f"[{utc_iso()}] outcome failed {row['symbol']} {horizon}m: {type(exc).__name__}: {exc}", flush=True)
                continue
            if outcome is None:
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO radar_final_outcomes (
                    signal_key, horizon_minutes, evaluated_at, candle_count,
                    terminal_return_bps, directional_return_bps, mfe_bps, mae_bps,
                    time_to_mfe_minutes, time_to_mae_minutes,
                    target_25_hit, target_50_hit, target_100_hit,
                    target_25_minutes, target_50_minutes, target_100_minutes,
                    target_50_before_stop50
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["signal_key"], horizon, utc_iso(), outcome["candle_count"],
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


def print_status(connection: sqlite3.Connection) -> None:
    states = connection.execute(
        "SELECT symbol, outcome, pending, pending_confirmations, confidence, radar_score, score_gate FROM radar_final_state ORDER BY symbol"
    ).fetchall()
    signals = connection.execute("SELECT COUNT(*) FROM radar_final_signals").fetchone()[0]
    outcomes = connection.execute("SELECT COUNT(*) FROM radar_final_outcomes").fetchone()[0]
    print(f"final_signals={signals} evaluated_horizons={outcomes}")
    for row in states:
        print(
            f"{row['symbol']:8s} outcome={row['outcome']:10s} pending={str(row['pending']):10s} "
            f"confirmations={row['pending_confirmations']} confidence={row['confidence']} "
            f"score={row['radar_score']} gate={row['score_gate']}"
        )


def handle_signal(signum: int, frame: Any) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Persist and evaluate the exact stabilized Radar V2 outcome shown to users.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    connection = connect(args.db)
    if args.status:
        print_status(connection)
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": "LiqHeat-Radar-Final-Signal-Logger/1.0"})

    while not STOP_REQUESTED:
        started = time.monotonic()
        try:
            payload = fetch_radar(session, args.api_url)
            observations, transitions = process_payload(connection, payload)
            evaluated = evaluate_due(connection, session)
            print(
                f"[{utc_iso()}] observations={observations} transitions={transitions} evaluated={evaluated}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{utc_iso()}] logger error: {type(exc).__name__}: {exc}", flush=True)

        if args.once:
            break
        remaining = max(1.0, args.poll_seconds - (time.monotonic() - started))
        end = time.monotonic() + remaining
        while not STOP_REQUESTED and time.monotonic() < end:
            time.sleep(min(1.0, end - time.monotonic()))

    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
