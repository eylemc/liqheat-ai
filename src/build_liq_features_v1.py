#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BATCH_SIZE = 10000


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def build_feature(row):
    payload = json.loads(row["payload_json"])

    agg = payload.get("aggregated", {})
    points = payload.get("dataPoints", [])

    total_longs = safe_float(agg.get("totalLongs"))
    total_shorts = safe_float(agg.get("totalShorts"))
    max_volume = safe_float(agg.get("maxVolume"))

    active_cells = 0

    for p in points:
        try:
            if p[2] or p[3]:
                active_cells += 1
        except Exception:
            pass

    pressure_ratio = None
    if total_shorts and total_shorts > 0:
        pressure_ratio = total_longs / total_shorts

    pressure_imbalance = None
    denom = (total_longs or 0) + (total_shorts or 0)

    if denom > 0:
        pressure_imbalance = (
            (total_longs or 0)
            - (total_shorts or 0)
        ) / denom

    range_abs = None
    range_pct = None

    if (
        row["price_min"] is not None
        and row["price_max"] is not None
    ):
        range_abs = row["price_max"] - row["price_min"]

    if (
        range_abs is not None
        and row["current_price"]
        and row["current_price"] > 0
    ):
        range_pct = range_abs / row["current_price"]

    return {
        "id": row["id"],
        "logged_at": row["logged_at"],
        "symbol": row["symbol"],
        "timeframe": row["timeframe"],
        "current_price": row["current_price"],
        "price_min": row["price_min"],
        "price_max": row["price_max"],
        "range_abs": range_abs,
        "range_pct": range_pct,
        "liquidation_count": row["liquidation_count"],
        "total_longs": total_longs,
        "total_shorts": total_shorts,
        "pressure_ratio": pressure_ratio,
        "pressure_imbalance": pressure_imbalance,
        "max_volume": max_volume,
        "rows": row["rows"],
        "cols": row["cols"],
        "active_cells": active_cells,
        "source_file": row["source_file"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default="data/duckdb/liqheat_research.duckdb",
    )

    parser.add_argument(
        "--out",
        default="data/features/liq_features_v1.parquet",
    )

    args = parser.parse_args()

    Path(args.out).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.time()

    con = duckdb.connect(args.db)

    total_rows = con.execute("""
        SELECT COUNT(*)
        FROM research.liq_logging
    """).fetchone()[0]

    print(f"Rows: {total_rows:,}")

    writer = None
    offset = 0
    processed = 0

    while True:

        batch = con.execute(f"""
            SELECT
                id,
                logged_at,
                symbol,
                timeframe,
                current_price,
                liquidation_count,
                price_min,
                price_max,
                rows,
                cols,
                payload_json,
                source_file
            FROM research.liq_logging
            LIMIT {BATCH_SIZE}
            OFFSET {offset}
        """).fetch_df()

        if len(batch) == 0:
            break

        features = []

        for _, row in batch.iterrows():
            features.append(build_feature(row))

        feature_df = pd.DataFrame(features)

        table = pa.Table.from_pandas(
            feature_df,
            preserve_index=False
        )

        if writer is None:
            writer = pq.ParquetWriter(
                args.out,
                table.schema,
                compression="zstd"
            )

        writer.write_table(table)

        processed += len(batch)
        offset += BATCH_SIZE

        print(
            f"{processed:,}/{total_rows:,} "
            f"({processed / total_rows:.1%})"
        )

    if writer:
        writer.close()

    elapsed = time.time() - start

    print()
    print("====================================")
    print("LIQ FEATURES V1 COMPLETE")
    print("====================================")
    print(f"Rows: {processed:,}")
    print(f"Output: {args.out}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
