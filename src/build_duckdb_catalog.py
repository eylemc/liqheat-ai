from __future__ import annotations

import json
from pathlib import Path

import duckdb


DB_PATH = Path("data/duckdb/liqheat_research.duckdb")
PARQUET_GLOB = str(
    Path("data/research-parquet")
    / "**"
    / "*.parquet"
)

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='20GB'")
con.execute("PRAGMA temp_directory='data/duckdb/tmp'")

con.execute("CREATE SCHEMA IF NOT EXISTS research")

con.execute(
    f"""
    CREATE OR REPLACE VIEW research.liq_logging AS
    SELECT *
    FROM read_parquet(
        '{PARQUET_GLOB}',
        union_by_name = true,
        hive_partitioning = false
    )
    """
)

con.execute(
    """
    CREATE OR REPLACE VIEW research.stream_summary AS
    SELECT
        symbol,
        timeframe,
        COUNT(*) AS rows,
        MIN(logged_at) AS first_timestamp,
        MAX(logged_at) AS last_timestamp,
        COUNT(DISTINCT CAST(logged_at AS DATE)) AS active_days,
        AVG(current_price) AS average_price,
        SUM(COALESCE(liquidation_count, 0)) AS liquidation_events
    FROM research.liq_logging
    GROUP BY symbol, timeframe
    ORDER BY symbol, timeframe
    """
)

con.execute(
    """
    CREATE OR REPLACE VIEW research.daily_summary AS
    SELECT
        symbol,
        timeframe,
        CAST(logged_at AS DATE) AS day,
        COUNT(*) AS rows,
        MIN(logged_at) AS first_timestamp,
        MAX(logged_at) AS last_timestamp,
        MIN(current_price) AS min_price,
        MAX(current_price) AS max_price,
        AVG(current_price) AS average_price,
        SUM(COALESCE(liquidation_count, 0)) AS liquidation_events
    FROM research.liq_logging
    GROUP BY symbol, timeframe, CAST(logged_at AS DATE)
    """
)

total_rows = con.execute(
    "SELECT COUNT(*) FROM research.liq_logging"
).fetchone()[0]

duplicate_ids = con.execute(
    """
    SELECT COUNT(*)
    FROM (
        SELECT id
        FROM research.liq_logging
        WHERE id IS NOT NULL
        GROUP BY id
        HAVING COUNT(*) > 1
    )
    """
).fetchone()[0]

null_symbols = con.execute(
    """
    SELECT COUNT(*)
    FROM research.liq_logging
    WHERE symbol IS NULL OR timeframe IS NULL
    """
).fetchone()[0]

raw_rows = con.execute(
    """
    SELECT COUNT(*)
    FROM research.liq_logging
    WHERE raw_value IS NOT NULL
    """
).fetchone()[0]

streams = con.execute(
    """
    SELECT *
    FROM research.stream_summary
    """
).fetchdf()

summary = {
    "duckdb": str(DB_PATH.resolve()),
    "total_rows": int(total_rows),
    "duplicate_ids": int(duplicate_ids),
    "null_symbol_or_timeframe_rows": int(null_symbols),
    "raw_value_rows": int(raw_rows),
    "stream_count": int(len(streams)),
}

Path("reports/duckdb_catalog_summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("==========================================")
print("DUCKDB CATALOG READY")
print("==========================================")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print()
print(streams.to_string(index=False))

con.close()
