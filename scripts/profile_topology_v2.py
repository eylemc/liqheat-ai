from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/features/liq_topology_v2.parquet")
OUTPUT_DIR = Path("data/reports")
OUTPUT_JSON = OUTPUT_DIR / "topology_v2_profile.json"
OUTPUT_CSV = OUTPUT_DIR / "topology_v2_numeric_profile.csv"

TOPOLOGY_COLUMNS = [
    "nearest_side",
    "distance_advantage",
    "nearest_pool_volume",
    "farther_pool_volume",
    "pool_volume_ratio",
    "distance_pressure_ratio",
]

NUMERIC_COLUMNS = [
    "current_price",
    "nearest_upper_price",
    "nearest_lower_price",
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "upper_pool_volume",
    "lower_pool_volume",
    "nearest_pool_volume",
    "farther_pool_volume",
    "pool_volume_ratio",
    "distance_pressure_ratio",
    "upper_active_levels",
    "lower_active_levels",
    "upper_total_volume",
    "lower_total_volume",
    "topology_imbalance",
]


def python_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    return value


def value_counts_dict(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False)
    result = {}

    for key, value in counts.items():
        key = "<NULL>" if pd.isna(key) else str(key)
        result[key] = int(value)

    return result


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: File not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TOPOLOGY V2 PROFILE")
    print("=" * 70)
    print(f"Loading: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    row_count = len(df)
    column_count = len(df.columns)

    df["has_topology"] = (
        df[TOPOLOGY_COLUMNS].notna().all(axis=1).astype("int8")
    )

    topology_available = int(df["has_topology"].sum())
    topology_missing = int(row_count - topology_available)
    topology_available_pct = topology_available / row_count * 100
    topology_missing_pct = topology_missing / row_count * 100

    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = (
        int(df["id"].duplicated().sum())
        if "id" in df.columns
        else None
    )

    numeric_df = df[
        [column for column in NUMERIC_COLUMNS if column in df.columns]
    ]

    inf_counts = (
        np.isinf(numeric_df)
        .sum()
        .sort_values(ascending=False)
        .astype(int)
    )

    null_counts = df.isna().sum().sort_values(ascending=False).astype(int)

    numeric_profile = numeric_df.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T

    numeric_profile["null_count"] = numeric_df.isna().sum()
    numeric_profile["null_pct"] = numeric_df.isna().mean() * 100
    numeric_profile["inf_count"] = np.isinf(numeric_df).sum()

    numeric_profile.to_csv(OUTPUT_CSV)

    timestamp_min = df["logged_at"].min()
    timestamp_max = df["logged_at"].max()

    symbol_counts = value_counts_dict(df["symbol"])
    timeframe_counts = value_counts_dict(df["timeframe"])
    nearest_side_counts = value_counts_dict(df["nearest_side"])

    missing_by_symbol = (
        df.groupby("symbol", dropna=False)["has_topology"]
        .agg(rows="size", topology_rows="sum")
        .reset_index()
    )
    missing_by_symbol["missing_rows"] = (
        missing_by_symbol["rows"]
        - missing_by_symbol["topology_rows"]
    )
    missing_by_symbol["missing_pct"] = (
        missing_by_symbol["missing_rows"]
        / missing_by_symbol["rows"]
        * 100
    )

    missing_by_timeframe = (
        df.groupby("timeframe", dropna=False)["has_topology"]
        .agg(rows="size", topology_rows="sum")
        .reset_index()
    )
    missing_by_timeframe["missing_rows"] = (
        missing_by_timeframe["rows"]
        - missing_by_timeframe["topology_rows"]
    )
    missing_by_timeframe["missing_pct"] = (
        missing_by_timeframe["missing_rows"]
        / missing_by_timeframe["rows"]
        * 100
    )

    report = {
        "input_path": str(INPUT_PATH),
        "shape": {
            "rows": row_count,
            "columns": column_count,
        },
        "time_range": {
            "start": python_value(timestamp_min),
            "end": python_value(timestamp_max),
        },
        "data_quality": {
            "duplicate_rows": duplicate_rows,
            "duplicate_ids": duplicate_ids,
            "total_null_values": int(df.isna().sum().sum()),
            "total_inf_values": int(inf_counts.sum()),
        },
        "topology_coverage": {
            "available_rows": topology_available,
            "available_pct": topology_available_pct,
            "missing_rows": topology_missing,
            "missing_pct": topology_missing_pct,
        },
        "symbol_counts": symbol_counts,
        "timeframe_counts": timeframe_counts,
        "nearest_side_counts": nearest_side_counts,
        "null_counts": {
            column: int(value)
            for column, value in null_counts.items()
        },
        "inf_counts": {
            column: int(value)
            for column, value in inf_counts.items()
        },
        "missing_topology_by_symbol": (
            missing_by_symbol.to_dict(orient="records")
        ),
        "missing_topology_by_timeframe": (
            missing_by_timeframe.to_dict(orient="records")
        ),
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print()
    print(f"Shape                 : {df.shape}")
    print(f"Date range            : {timestamp_min} -> {timestamp_max}")
    print(f"Duplicate rows        : {duplicate_rows}")
    print(f"Duplicate IDs         : {duplicate_ids}")
    print(f"Infinite values       : {int(inf_counts.sum())}")

    print()
    print(
        f"Topology available    : {topology_available:,} "
        f"({topology_available_pct:.4f}%)"
    )
    print(
        f"Topology missing      : {topology_missing:,} "
        f"({topology_missing_pct:.4f}%)"
    )

    print()
    print("-" * 70)
    print("SYMBOL COUNTS")
    print("-" * 70)
    print(df["symbol"].value_counts(dropna=False).to_string())

    print()
    print("-" * 70)
    print("TIMEFRAME COUNTS")
    print("-" * 70)
    print(df["timeframe"].value_counts(dropna=False).to_string())

    print()
    print("-" * 70)
    print("NEAREST SIDE COUNTS")
    print("-" * 70)
    print(df["nearest_side"].value_counts(dropna=False).to_string())

    print()
    print("-" * 70)
    print("NULL COUNTS")
    print("-" * 70)
    print(null_counts[null_counts > 0].to_string())

    print()
    print("-" * 70)
    print("NUMERIC PROFILE")
    print("-" * 70)
    print(
        numeric_profile[
            [
                "count",
                "mean",
                "std",
                "min",
                "1%",
                "5%",
                "50%",
                "95%",
                "99%",
                "max",
                "null_count",
                "null_pct",
                "inf_count",
            ]
        ].to_string()
    )

    print()
    print("-" * 70)
    print("MISSING TOPOLOGY BY SYMBOL")
    print("-" * 70)
    print(
        missing_by_symbol.sort_values(
            "missing_pct",
            ascending=False,
        ).to_string(index=False)
    )

    print()
    print("-" * 70)
    print("MISSING TOPOLOGY BY TIMEFRAME")
    print("-" * 70)
    print(
        missing_by_timeframe.sort_values(
            "missing_pct",
            ascending=False,
        ).to_string(index=False)
    )

    print()
    print("=" * 70)
    print("PROFILE COMPLETE")
    print("=" * 70)
    print(f"JSON report : {OUTPUT_JSON}")
    print(f"CSV profile : {OUTPUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
