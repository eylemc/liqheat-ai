#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


BATCH_SIZE = 2_000

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("logged_at", pa.timestamp("us", tz="UTC")),
        pa.field("symbol", pa.string()),
        pa.field("timeframe", pa.string()),
        pa.field("current_price", pa.float64()),

        pa.field("nearest_upper_price", pa.float64()),
        pa.field("nearest_lower_price", pa.float64()),

        pa.field("upper_distance_pct", pa.float64()),
        pa.field("lower_distance_pct", pa.float64()),
        pa.field("distance_advantage", pa.float64()),

        pa.field("nearest_side", pa.string()),

        pa.field("upper_pool_volume", pa.float64()),
        pa.field("lower_pool_volume", pa.float64()),
        pa.field("nearest_pool_volume", pa.float64()),
        pa.field("farther_pool_volume", pa.float64()),

        pa.field("pool_volume_ratio", pa.float64()),
        pa.field("distance_pressure_ratio", pa.float64()),

        pa.field("upper_active_levels", pa.int64()),
        pa.field("lower_active_levels", pa.int64()),

        pa.field("upper_total_volume", pa.float64()),
        pa.field("lower_total_volume", pa.float64()),

        pa.field("topology_imbalance", pa.float64()),
        pa.field("source_file", pa.string()),
    ]
)


def safe_float(value: Any) -> float:
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except Exception:
        pass
    return 0.0


def aggregate_by_price(points: list[Any]) -> dict[float, dict[str, float]]:
    levels: dict[float, dict[str, float]] = {}

    for point in points:
        if not isinstance(point, list) or len(point) < 4:
            continue

        price = safe_float(point[1])
        long_volume = safe_float(point[2])
        short_volume = safe_float(point[3])

        if price <= 0:
            continue

        bucket = levels.setdefault(
            price,
            {"long": 0.0, "short": 0.0},
        )

        bucket["long"] += long_volume
        bucket["short"] += short_volume

    return levels


def choose_pool(
    levels: dict[float, dict[str, float]],
    current_price: float,
    side: str,
) -> tuple[float | None, float]:
    candidates: list[tuple[float, float]] = []

    for price, volumes in levels.items():
        total = volumes["long"] + volumes["short"]

        if total <= 0:
            continue

        if side == "upper" and price > current_price:
            candidates.append((price, total))

        elif side == "lower" and price < current_price:
            candidates.append((price, total))

    if not candidates:
        return None, 0.0

    if side == "upper":
        nearest_price = min(price for price, _ in candidates)
    else:
        nearest_price = max(price for price, _ in candidates)

    nearest_volume = next(
        volume
        for price, volume in candidates
        if price == nearest_price
    )

    return nearest_price, nearest_volume


def build_feature(row: dict[str, Any]) -> dict[str, Any]:
    current_price = safe_float(row["current_price"])

    try:
        payload = json.loads(row["payload_json"])
    except Exception:
        payload = {}

    points = payload.get("dataPoints", [])
    levels = aggregate_by_price(points)

    upper_price, upper_pool_volume = choose_pool(
        levels,
        current_price,
        "upper",
    )

    lower_price, lower_pool_volume = choose_pool(
        levels,
        current_price,
        "lower",
    )

    upper_distance_pct = None
    lower_distance_pct = None

    if upper_price is not None and current_price > 0:
        upper_distance_pct = (
            upper_price - current_price
        ) / current_price

    if lower_price is not None and current_price > 0:
        lower_distance_pct = (
            current_price - lower_price
        ) / current_price

    nearest_side = None
    nearest_pool_volume = None
    farther_pool_volume = None
    distance_advantage = None
    pool_volume_ratio = None
    distance_pressure_ratio = None

    if (
        upper_distance_pct is not None
        and lower_distance_pct is not None
    ):
        if upper_distance_pct < lower_distance_pct:
            nearest_side = "UPPER"
            nearest_pool_volume = upper_pool_volume
            farther_pool_volume = lower_pool_volume
        elif lower_distance_pct < upper_distance_pct:
            nearest_side = "LOWER"
            nearest_pool_volume = lower_pool_volume
            farther_pool_volume = upper_pool_volume
        else:
            nearest_side = "TIE"
            nearest_pool_volume = max(
                upper_pool_volume,
                lower_pool_volume,
            )
            farther_pool_volume = min(
                upper_pool_volume,
                lower_pool_volume,
            )

        distance_advantage = abs(
            upper_distance_pct - lower_distance_pct
        )

        if farther_pool_volume and farther_pool_volume > 0:
            pool_volume_ratio = (
                nearest_pool_volume / farther_pool_volume
            )

        nearest_distance = min(
            upper_distance_pct,
            lower_distance_pct,
        )
        farther_distance = max(
            upper_distance_pct,
            lower_distance_pct,
        )

        if nearest_distance > 0:
            distance_ratio = farther_distance / nearest_distance

            if pool_volume_ratio is not None:
                distance_pressure_ratio = (
                    distance_ratio * pool_volume_ratio
                )

    upper_total_volume = 0.0
    lower_total_volume = 0.0
    upper_active_levels = 0
    lower_active_levels = 0

    for price, volumes in levels.items():
        total = volumes["long"] + volumes["short"]

        if total <= 0:
            continue

        if price > current_price:
            upper_total_volume += total
            upper_active_levels += 1

        elif price < current_price:
            lower_total_volume += total
            lower_active_levels += 1

    total_topology_volume = upper_total_volume + lower_total_volume

    topology_imbalance = None

    if total_topology_volume > 0:
        topology_imbalance = (
            upper_total_volume - lower_total_volume
        ) / total_topology_volume

    return {
        "id": row["id"],
        "logged_at": row["logged_at"],
        "symbol": row["symbol"],
        "timeframe": row["timeframe"],
        "current_price": current_price,

        "nearest_upper_price": upper_price,
        "nearest_lower_price": lower_price,

        "upper_distance_pct": upper_distance_pct,
        "lower_distance_pct": lower_distance_pct,
        "distance_advantage": distance_advantage,

        "nearest_side": nearest_side,

        "upper_pool_volume": upper_pool_volume,
        "lower_pool_volume": lower_pool_volume,
        "nearest_pool_volume": nearest_pool_volume,
        "farther_pool_volume": farther_pool_volume,

        "pool_volume_ratio": pool_volume_ratio,
        "distance_pressure_ratio": distance_pressure_ratio,

        "upper_active_levels": upper_active_levels,
        "lower_active_levels": lower_active_levels,

        "upper_total_volume": upper_total_volume,
        "lower_total_volume": lower_total_volume,

        "topology_imbalance": topology_imbalance,
        "source_file": row["source_file"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default="data/duckdb/liqheat_research.duckdb",
    )

    parser.add_argument(
        "--out",
        default="data/features/liq_topology_v2.parquet",
    )

    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".parquet.part")
    temporary_path.unlink(missing_ok=True)

    con = duckdb.connect(args.db, read_only=True)

    total_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM research.liq_logging
        """
    ).fetchone()[0]

    print("==========================================")
    print("LiqHeat Topology Features V2")
    print("==========================================")
    print("Rows:", f"{total_rows:,}")
    print("Batch size:", BATCH_SIZE)
    print("Output:", output_path)
    print()

    # Topology extraction satır sırasından bağımsızdır.
    # Ağır JSON kolonunu ORDER BY ile sıralamak RAM'i tükettiği için
    # doğrudan streaming RecordBatch reader kullanıyoruz.
    con.execute("SET threads=2")
    con.execute("SET memory_limit='12GB'")
    con.execute("SET preserve_insertion_order=false")

    Path("data/duckdb/tmp").mkdir(
        parents=True,
        exist_ok=True,
    )
    con.execute("SET temp_directory='data/duckdb/tmp'")

    reader = con.execute(
        """
        SELECT
            id,
            logged_at,
            symbol,
            timeframe,
            current_price,
            payload_json,
            source_file
        FROM research.liq_logging
        """
    ).fetch_record_batch(BATCH_SIZE)

    writer: pq.ParquetWriter | None = None
    processed = 0
    started = time.time()

    for batch in reader:
        if batch.num_rows == 0:
            continue

        rows = batch.to_pylist()
        features = [build_feature(row) for row in rows]

        feature_table = pa.Table.from_pylist(
            features,
            schema=SCHEMA,
        )

        if writer is None:
            writer = pq.ParquetWriter(
                temporary_path,
                SCHEMA,
                compression="zstd",
                compression_level=6,
                use_dictionary=True,
                write_statistics=True,
            )

        writer.write_table(
            feature_table,
            row_group_size=BATCH_SIZE,
        )

        processed += len(features)

        elapsed = time.time() - started
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = total_rows - processed
        eta_seconds = remaining / rate if rate > 0 else 0

        print(
            "\r"
            f"{processed:,}/{total_rows:,} "
            f"({processed / total_rows:.1%}) | "
            f"{rate:,.0f} rows/s | "
            f"ETA={eta_seconds / 60:.1f} min",
            end="",
            flush=True,
        )

    print()

    if writer is None:
        raise RuntimeError("Hiç feature üretilmedi")

    writer.close()
    temporary_path.replace(output_path)
    con.close()

    elapsed = time.time() - started

    print()
    print("==========================================")
    print("TOPOLOGY V2 COMPLETE")
    print("==========================================")
    print("Rows:", f"{processed:,}")
    print("Output:", output_path)
    print("Elapsed minutes:", round(elapsed / 60, 2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
