from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/features/liq_topology_v2.parquet")
OUTPUT_PATH = Path("data/features/liq_topology_v2_ml_features.parquet")

REQUIRED_COLUMNS = [
    "id",
    "logged_at",
    "symbol",
    "timeframe",
    "current_price",
    "nearest_upper_price",
    "nearest_lower_price",
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "nearest_side",
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

LOG1P_COLUMNS = [
    "current_price",
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
]

SIDE_MAP = {
    "LOWER": -1,
    "TIE": 0,
    "UPPER": 1,
}


def validate_input(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            "Required columns are missing: " + ", ".join(missing)
        )

    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs detected.")

    if df["logged_at"].isna().any():
        raise ValueError("Null logged_at values detected.")

    numeric = df.select_dtypes(include=[np.number])

    if np.isinf(numeric).any().any():
        raise ValueError("Infinite numeric values detected.")


def safe_log1p(series: pd.Series) -> pd.Series:
    """
    Apply log1p only to non-negative finite values.
    Existing null values remain null.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_negative = numeric < 0

    if invalid_negative.any():
        count = int(invalid_negative.sum())
        raise ValueError(
            f"{series.name}: {count:,} negative values cannot use log1p."
        )

    return np.log1p(numeric)


def main() -> int:
    started = time.time()

    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("BUILD TOPOLOGY V2 ML FEATURES")
    print("=" * 72)
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    df = pd.read_parquet(INPUT_PATH)
    validate_input(df)

    original_rows = len(df)

    # Preserve deterministic time order.
    df = df.sort_values(
        ["symbol", "timeframe", "logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Structural availability flags.
    df["has_upper_level"] = (
        df["nearest_upper_price"].notna().astype("int8")
    )
    df["has_lower_level"] = (
        df["nearest_lower_price"].notna().astype("int8")
    )
    df["has_topology"] = (
        df["nearest_side"].notna().astype("int8")
    )

    # Side representation:
    # LOWER=-1, TIE=0, UPPER=1, missing topology remains null.
    df["nearest_side_code"] = (
        df["nearest_side"]
        .map(SIDE_MAP)
        .astype("Int8")
    )

    # Additional interpretable balance features.
    df["active_level_difference"] = (
        df["upper_active_levels"] - df["lower_active_levels"]
    ).astype("int16")

    df["active_level_total"] = (
        df["upper_active_levels"] + df["lower_active_levels"]
    ).astype("int16")

    volume_denominator = (
        df["upper_total_volume"] + df["lower_total_volume"]
    )

    df["total_volume_imbalance_check"] = np.where(
        volume_denominator > 0,
        (
            df["upper_total_volume"] - df["lower_total_volume"]
        ) / volume_denominator,
        0.0,
    )

    # Stable signed form of the two-sided distance difference.
    df["signed_distance_edge"] = (
        df["lower_distance_pct"] - df["upper_distance_pct"]
    )

    # Log transformations for strongly skewed non-negative features.
    for column in LOG1P_COLUMNS:
        output_column = f"log1p_{column}"
        df[output_column] = safe_log1p(df[column])

    # Calendar context. These use only information available at observation time.
    df["hour_utc"] = df["logged_at"].dt.hour.astype("int8")
    df["day_of_week_utc"] = df["logged_at"].dt.dayofweek.astype("int8")
    df["day_of_month_utc"] = df["logged_at"].dt.day.astype("int8")
    df["month_utc"] = df["logged_at"].dt.month.astype("int8")
    df["is_weekend_utc"] = (
        df["day_of_week_utc"] >= 5
    ).astype("int8")

    # Cyclical time representations.
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour_utc"] / 24
    )
    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour_utc"] / 24
    )
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week_utc"] / 7
    )
    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week_utc"] / 7
    )

    # Compact categorical storage without one-hot encoding yet.
    df["symbol"] = df["symbol"].astype("category")
    df["timeframe"] = df["timeframe"].astype("category")
    df["nearest_side"] = df["nearest_side"].astype("category")

    # Final validation.
    if len(df) != original_rows:
        raise RuntimeError(
            f"Row count changed: {original_rows:,} -> {len(df):,}"
        )

    numeric = df.select_dtypes(include=[np.number])

    total_inf = int(np.isinf(numeric).sum().sum())

    if total_inf != 0:
        raise RuntimeError(
            f"Output contains {total_inf:,} infinite values."
        )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    elapsed = time.time() - started
    output_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)

    print(f"Rows                 : {len(df):,}")
    print(f"Columns              : {len(df.columns):,}")
    print(f"Topology available   : {int(df['has_topology'].sum()):,}")
    print(
        f"Topology unavailable : "
        f"{int((df['has_topology'] == 0).sum()):,}"
    )
    print(f"Infinite values      : {total_inf:,}")
    print(f"Output size          : {output_size_mb:,.1f} MB")
    print(f"Elapsed seconds      : {elapsed:,.1f}")
    print()
    print("New feature columns:")
    new_columns = [
        column
        for column in df.columns
        if column not in REQUIRED_COLUMNS
        and column != "source_file"
    ]
    for column in new_columns:
        print(f"  - {column}")

    print()
    print("=" * 72)
    print("ML FEATURE BUILD COMPLETE")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
