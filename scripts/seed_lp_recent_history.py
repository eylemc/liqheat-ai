from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.liquidation_pressure_history_logger import (
    DB_PATH,
    flip_count,
    init_db,
    mean,
    persistence,
    slope,
    stats,
)
from scripts.test_historical_liquidation_pressure_backfill import (
    fetch_rows,
    reconstruct_batch,
)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT", "XAGUSDT"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Seed the live Liquidation Pressure history DB from recent Supabase "
            "liq_logging rows so 2H temporal confirmation does not need to wait."
        )
    )
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--timeframe", default="24h")
    p.add_argument("--limit-per-symbol", type=int, default=500)
    p.add_argument("--db", default=str(DB_PATH))
    return p.parse_args()


def already_exists(con: sqlite3.Connection, symbol: str, observed_at: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM pressure_snapshots WHERE symbol=? AND observed_at=? LIMIT 1",
        (symbol, observed_at),
    ).fetchone()
    return row is not None


def insert_seed_snapshot(con: sqlite3.Connection, row: dict[str, Any]) -> int | None:
    symbol = str(row["symbol"]).upper()
    observed_at = row["logged_at"].isoformat()
    if already_exists(con, symbol, observed_at):
        return None

    pressure = float(row["liquidation_pressure_score"])
    signed = float(row["signed_pressure"])
    raw_prediction = str(row["raw_prediction"])
    direction_confidence = float(row["direction_confidence"])

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
            observed_at,
            symbol,
            float(row["current_price"]) if row.get("current_price") is not None else None,
            pressure,
            "UP" if signed > 0 else "DOWN" if signed < 0 else "N/A",
            signed,
            raw_prediction,
            raw_prediction,
            None,
            direction_confidence,
            direction_confidence * 100.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            json.dumps(
                {
                    "seeded": True,
                    "source": "supabase_liq_logging_recent_reconstruction",
                    "snapshot_id": str(row["snapshot_id"]),
                    "symbol": symbol,
                    "logged_at": observed_at,
                    "liquidation_pressure_score": pressure,
                    "signed_pressure": signed,
                    "raw_prediction": raw_prediction,
                    "direction_confidence": direction_confidence,
                },
                separators=(",", ":"),
            ),
        ),
    )
    return int(cur.lastrowid)


def update_latest_features_now(con: sqlite3.Connection, symbol: str) -> None:
    latest = con.execute(
        """
        SELECT id, observed_at, signed_pressure
        FROM pressure_snapshots
        WHERE symbol=? AND signed_pressure IS NOT NULL
        ORDER BY observed_at DESC, id DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    if not latest:
        return

    snapshot_id = int(latest[0])
    observed_at = str(latest[1])
    signed_now = float(latest[2])

    windows = {m: stats(con, symbol, m) for m in (30, 60, 120)}
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
        ON CONFLICT(snapshot_id) DO UPDATE SET
            observed_at=excluded.observed_at,
            symbol=excluded.symbol,
            signed_now=excluded.signed_now,
            mean_30m=excluded.mean_30m,
            mean_60m=excluded.mean_60m,
            mean_120m=excluded.mean_120m,
            slope_30m=excluded.slope_30m,
            slope_60m=excluded.slope_60m,
            slope_120m=excluded.slope_120m,
            persistence_30m=excluded.persistence_30m,
            persistence_60m=excluded.persistence_60m,
            persistence_120m=excluded.persistence_120m,
            flips_30m=excluded.flips_30m,
            flips_60m=excluded.flips_60m,
            flips_120m=excluded.flips_120m,
            peak_abs_30m=excluded.peak_abs_30m,
            peak_abs_60m=excluded.peak_abs_60m,
            peak_abs_120m=excluded.peak_abs_120m,
            acceleration_2h=excluded.acceleration_2h,
            sample_count_30m=excluded.sample_count_30m,
            sample_count_60m=excluded.sample_count_60m,
            sample_count_120m=excluded.sample_count_120m
        """,
        (
            snapshot_id, observed_at, symbol, signed_now,
            s30["mean"], s60["mean"], s120["mean"],
            s30["slope"], s60["slope"], s120["slope"],
            s30["persistence"], s60["persistence"], s120["persistence"],
            s30["flips"], s60["flips"], s120["flips"],
            s30["peak_abs"], s60["peak_abs"], s120["peak_abs"],
            acceleration,
            s30["n"], s60["n"], s120["n"],
        ),
    )


def main() -> int:
    args = parse_args()
    since = datetime.now(timezone.utc) - timedelta(hours=float(args.hours))
    since_iso = since.isoformat()

    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    init_db(con)

    print("=" * 96)
    print("SEED LIVE LP HISTORY FROM RECENT SUPABASE")
    print("=" * 96)
    print("Since     :", since_iso)
    print("Symbols   :", [s.upper() for s in args.symbols])
    print("Timeframe :", args.timeframe)
    print("DB        :", db)
    print()

    total_inserted = 0
    for symbol in args.symbols:
        symbol = symbol.upper()
        rows = fetch_rows(symbol, args.timeframe, since_iso, args.limit_per_symbol)
        recon = reconstruct_batch(rows)
        inserted = 0
        if not recon.empty:
            recon = recon.sort_values("logged_at")
            for record in recon.to_dict("records"):
                if insert_seed_snapshot(con, record) is not None:
                    inserted += 1
            con.commit()
            update_latest_features_now(con, symbol)
            con.commit()

        total_inserted += inserted
        count_120 = con.execute(
            """
            SELECT COUNT(*) FROM pressure_snapshots
            WHERE symbol=? AND signed_pressure IS NOT NULL
              AND julianday(observed_at) >= julianday('now') - (120.0 / 1440.0)
            """,
            (symbol,),
        ).fetchone()[0]
        print(
            f"{symbol:8s} fetched={len(rows):4d} reconstructed={len(recon):4d} "
            f"inserted={inserted:4d} current_120m_samples={int(count_120):4d}"
        )

    con.close()
    print()
    print("Inserted total:", total_inserted)
    print("Done. Restart/refresh Radar once so lp_confirmation_v2 reads the seeded history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
