#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build LiqHeat forward-return feature store."
    )

    parser.add_argument(
        "--db",
        default="data/duckdb/liqheat_research.duckdb",
        help="DuckDB database path.",
    )

    parser.add_argument(
        "--out",
        default="data/features/liq_forward_returns_v1.parquet",
        help="Output Parquet path.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_suffix(".parquet.part")
    temp_path.unlink(missing_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB bulunamadı: {db_path}")

    started = time.time()

    con = duckdb.connect(str(db_path))

    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA memory_limit='18GB'")

    Path("data/duckdb/tmp").mkdir(
        parents=True,
        exist_ok=True,
    )

    con.execute(
        "PRAGMA temp_directory='data/duckdb/tmp'"
    )

    source_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM research.liq_features_v1
        """
    ).fetchone()[0]

    print("==========================================")
    print("LiqHeat Forward Returns V1")
    print("==========================================")
    print("Source rows:", f"{source_rows:,}")
    print("Horizons: 1m, 5m, 15m, 60m, 240m")
    print("Output:", output_path)
    print("Threads:", args.threads)
    print()
    print("Forward prices hesaplanıyor...")

    # Veriler yaklaşık dakikalık geliyor. LEAD satır offseti ile ileri
    # fiyatı alıyoruz; ayrıca gerçek dakika farkını hesaplayıp bozuk veya
    # eksik zaman aralıklarını valid_* kolonlarıyla ayırıyoruz.
    query = f"""
    COPY (
        WITH ordered AS (
            SELECT
                *,

                LEAD(logged_at, 1) OVER stream_window
                    AS future_ts_1m_raw,
                LEAD(current_price, 1) OVER stream_window
                    AS future_price_1m_raw,

                LEAD(logged_at, 5) OVER stream_window
                    AS future_ts_5m_raw,
                LEAD(current_price, 5) OVER stream_window
                    AS future_price_5m_raw,

                LEAD(logged_at, 15) OVER stream_window
                    AS future_ts_15m_raw,
                LEAD(current_price, 15) OVER stream_window
                    AS future_price_15m_raw,

                LEAD(logged_at, 60) OVER stream_window
                    AS future_ts_60m_raw,
                LEAD(current_price, 60) OVER stream_window
                    AS future_price_60m_raw,

                LEAD(logged_at, 240) OVER stream_window
                    AS future_ts_240m_raw,
                LEAD(current_price, 240) OVER stream_window
                    AS future_price_240m_raw

            FROM research.liq_features_v1

            WINDOW stream_window AS (
                PARTITION BY symbol, timeframe
                ORDER BY logged_at, id
            )
        ),

        measured AS (
            SELECT
                *,

                date_diff(
                    'second',
                    logged_at,
                    future_ts_1m_raw
                ) / 60.0 AS future_minutes_1m,

                date_diff(
                    'second',
                    logged_at,
                    future_ts_5m_raw
                ) / 60.0 AS future_minutes_5m,

                date_diff(
                    'second',
                    logged_at,
                    future_ts_15m_raw
                ) / 60.0 AS future_minutes_15m,

                date_diff(
                    'second',
                    logged_at,
                    future_ts_60m_raw
                ) / 60.0 AS future_minutes_60m,

                date_diff(
                    'second',
                    logged_at,
                    future_ts_240m_raw
                ) / 60.0 AS future_minutes_240m

            FROM ordered
        ),

        validated AS (
            SELECT
                *,

                future_minutes_1m BETWEEN 0.5 AND 2.5
                    AS valid_1m,

                future_minutes_5m BETWEEN 4.0 AND 7.0
                    AS valid_5m,

                future_minutes_15m BETWEEN 13.0 AND 18.0
                    AS valid_15m,

                future_minutes_60m BETWEEN 55.0 AND 66.0
                    AS valid_60m,

                future_minutes_240m BETWEEN 230.0 AND 251.0
                    AS valid_240m

            FROM measured
        )

        SELECT
            id,
            logged_at,
            symbol,
            timeframe,

            current_price,
            price_min,
            price_max,
            range_abs,
            range_pct,

            liquidation_count,
            total_longs,
            total_shorts,
            pressure_ratio,
            pressure_imbalance,
            max_volume,
            rows,
            cols,
            active_cells,
            source_file,

            CASE
                WHEN valid_1m
                THEN future_ts_1m_raw
            END AS future_ts_1m,

            CASE
                WHEN valid_1m
                THEN future_price_1m_raw
            END AS future_price_1m,

            future_minutes_1m,
            valid_1m,

            CASE
                WHEN valid_1m
                 AND current_price > 0
                THEN (
                    future_price_1m_raw
                    / current_price
                ) - 1.0
            END AS ret_1m,

            CASE
                WHEN valid_1m
                 AND future_price_1m_raw > current_price
                THEN 1
                WHEN valid_1m
                 AND future_price_1m_raw < current_price
                THEN -1
                WHEN valid_1m
                THEN 0
            END AS direction_1m,

            CASE
                WHEN valid_5m
                THEN future_ts_5m_raw
            END AS future_ts_5m,

            CASE
                WHEN valid_5m
                THEN future_price_5m_raw
            END AS future_price_5m,

            future_minutes_5m,
            valid_5m,

            CASE
                WHEN valid_5m
                 AND current_price > 0
                THEN (
                    future_price_5m_raw
                    / current_price
                ) - 1.0
            END AS ret_5m,

            CASE
                WHEN valid_5m
                 AND future_price_5m_raw > current_price
                THEN 1
                WHEN valid_5m
                 AND future_price_5m_raw < current_price
                THEN -1
                WHEN valid_5m
                THEN 0
            END AS direction_5m,

            CASE
                WHEN valid_15m
                THEN future_ts_15m_raw
            END AS future_ts_15m,

            CASE
                WHEN valid_15m
                THEN future_price_15m_raw
            END AS future_price_15m,

            future_minutes_15m,
            valid_15m,

            CASE
                WHEN valid_15m
                 AND current_price > 0
                THEN (
                    future_price_15m_raw
                    / current_price
                ) - 1.0
            END AS ret_15m,

            CASE
                WHEN valid_15m
                 AND future_price_15m_raw > current_price
                THEN 1
                WHEN valid_15m
                 AND future_price_15m_raw < current_price
                THEN -1
                WHEN valid_15m
                THEN 0
            END AS direction_15m,

            CASE
                WHEN valid_60m
                THEN future_ts_60m_raw
            END AS future_ts_60m,

            CASE
                WHEN valid_60m
                THEN future_price_60m_raw
            END AS future_price_60m,

            future_minutes_60m,
            valid_60m,

            CASE
                WHEN valid_60m
                 AND current_price > 0
                THEN (
                    future_price_60m_raw
                    / current_price
                ) - 1.0
            END AS ret_60m,

            CASE
                WHEN valid_60m
                 AND future_price_60m_raw > current_price
                THEN 1
                WHEN valid_60m
                 AND future_price_60m_raw < current_price
                THEN -1
                WHEN valid_60m
                THEN 0
            END AS direction_60m,

            CASE
                WHEN valid_240m
                THEN future_ts_240m_raw
            END AS future_ts_240m,

            CASE
                WHEN valid_240m
                THEN future_price_240m_raw
            END AS future_price_240m,

            future_minutes_240m,
            valid_240m,

            CASE
                WHEN valid_240m
                 AND current_price > 0
                THEN (
                    future_price_240m_raw
                    / current_price
                ) - 1.0
            END AS ret_240m,

            CASE
                WHEN valid_240m
                 AND future_price_240m_raw > current_price
                THEN 1
                WHEN valid_240m
                 AND future_price_240m_raw < current_price
                THEN -1
                WHEN valid_240m
                THEN 0
            END AS direction_240m

        FROM validated
    )
    TO '{temp_path.as_posix()}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD,
        ROW_GROUP_SIZE 100000
    )
    """

    con.execute(query)

    temp_path.replace(output_path)

    print("Parquet tamamlandı.")
    print("DuckDB view oluşturuluyor...")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW
            research.liq_forward_returns_v1
        AS
        SELECT *
        FROM read_parquet(
            '{output_path.as_posix()}'
        )
        """
    )

    validation = con.execute(
        """
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT id) AS unique_ids,

            COUNT_IF(valid_1m) AS valid_1m,
            COUNT_IF(valid_5m) AS valid_5m,
            COUNT_IF(valid_15m) AS valid_15m,
            COUNT_IF(valid_60m) AS valid_60m,
            COUNT_IF(valid_240m) AS valid_240m,

            COUNT_IF(ret_240m IS NOT NULL)
                AS non_null_ret_240m

        FROM research.liq_forward_returns_v1
        """
    ).fetchone()

    (
        row_count,
        unique_ids,
        valid_1m,
        valid_5m,
        valid_15m,
        valid_60m,
        valid_240m,
        non_null_ret_240m,
    ) = validation

    if row_count != source_rows:
        raise RuntimeError(
            f"Satır sayısı uyuşmuyor: "
            f"source={source_rows}, output={row_count}"
        )

    stream_summary = con.execute(
        """
        SELECT
            symbol,
            timeframe,
            COUNT(*) AS row_count,

            ROUND(
                100.0 * COUNT_IF(valid_5m) / COUNT(*),
                2
            ) AS valid_5m_pct,

            ROUND(
                100.0 * COUNT_IF(valid_15m) / COUNT(*),
                2
            ) AS valid_15m_pct,

            ROUND(
                100.0 * COUNT_IF(valid_60m) / COUNT(*),
                2
            ) AS valid_60m_pct,

            ROUND(
                100.0 * COUNT_IF(valid_240m) / COUNT(*),
                2
            ) AS valid_240m_pct,

            ROUND(AVG(ret_5m), 7)
                AS avg_ret_5m,

            ROUND(AVG(ret_15m), 7)
                AS avg_ret_15m,

            ROUND(AVG(ret_60m), 7)
                AS avg_ret_60m,

            ROUND(AVG(ret_240m), 7)
                AS avg_ret_240m

        FROM research.liq_forward_returns_v1

        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
        """
    ).fetchdf()

    elapsed = time.time() - started
    output_bytes = output_path.stat().st_size

    summary = {
        "status": "complete",
        "source_rows": int(source_rows),
        "output_rows": int(row_count),
        "unique_ids": int(unique_ids),
        "valid_1m": int(valid_1m),
        "valid_5m": int(valid_5m),
        "valid_15m": int(valid_15m),
        "valid_60m": int(valid_60m),
        "valid_240m": int(valid_240m),
        "non_null_ret_240m": int(non_null_ret_240m),
        "output": str(output_path),
        "output_bytes": int(output_bytes),
        "output_mb": round(output_bytes / 1024**2, 3),
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_minutes": round(elapsed / 60, 3),
    }

    Path("reports").mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        "reports/forward_returns_v1_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stream_summary.to_csv(
        "reports/forward_returns_v1_streams.csv",
        index=False,
    )

    print()
    print("==========================================")
    print("FORWARD RETURNS V1 COMPLETE")
    print("==========================================")
    print(json.dumps(summary, indent=2))
    print()
    print("=== STREAM SUMMARY ===")
    print(stream_summary.to_string(index=False))

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
