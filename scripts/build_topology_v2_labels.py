from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)
OUTPUT_PATH = Path(
    "data/features/liq_topology_v2_ml_labeled.parquet"
)

HORIZONS = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}

# Target zamandan sonra kabul edilecek maksimum gecikme.
# Örneğin 15m hedefi için 15m–20m arasındaki ilk snapshot kabul edilir.
TOLERANCES = {
    "15m": pd.Timedelta(minutes=5),
    "1h": pd.Timedelta(minutes=10),
    "4h": pd.Timedelta(minutes=20),
}

# Nötr sınıf sınırları.
# Getiri bu aralığın içindeyse direction = 0.
NEUTRAL_THRESHOLDS = {
    "15m": 0.0005,   # %0.05
    "1h": 0.0010,    # %0.10
    "4h": 0.0020,    # %0.20
}


def validate_input(df: pd.DataFrame) -> None:
    required = [
        "id",
        "logged_at",
        "symbol",
        "timeframe",
        "current_price",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs detected.")

    if df["logged_at"].isna().any():
        raise ValueError("Null logged_at values detected.")

    if (df["current_price"] <= 0).any():
        raise ValueError("Non-positive current prices detected.")


def add_forward_labels(
    df: pd.DataFrame,
    horizon_name: str,
    horizon: pd.Timedelta,
    tolerance: pd.Timedelta,
    neutral_threshold: float,
) -> pd.DataFrame:
    print(
        f"Building {horizon_name} labels "
        f"(horizon={horizon}, tolerance={tolerance})..."
    )

    target_column = f"target_time_{horizon_name}"
    future_time_column = f"future_time_{horizon_name}"
    future_price_column = f"future_price_{horizon_name}"
    delay_column = f"future_delay_seconds_{horizon_name}"
    return_column = f"forward_return_{horizon_name}"
    direction_column = f"direction_{horizon_name}"
    valid_column = f"label_valid_{horizon_name}"

    df[target_column] = df["logged_at"] + horizon

    right = df[
        [
            "symbol",
            "timeframe",
            "logged_at",
            "current_price",
        ]
    ].copy()

    right = right.rename(
        columns={
            "logged_at": future_time_column,
            "current_price": future_price_column,
        }
    )

    left = df[
        [
            "id",
            "symbol",
            "timeframe",
            target_column,
        ]
    ].copy()

    # merge_asof requires global sorting by merge key.
    left = left.sort_values(
        [
            target_column,
            "symbol",
            "timeframe",
        ],
        kind="mergesort",
    )

    right = right.sort_values(
        [
            future_time_column,
            "symbol",
            "timeframe",
        ],
        kind="mergesort",
    )

    matched = pd.merge_asof(
        left,
        right,
        left_on=target_column,
        right_on=future_time_column,
        by=["symbol", "timeframe"],
        direction="forward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )

    matched = matched[
        [
            "id",
            future_time_column,
            future_price_column,
        ]
    ]

    df = df.merge(
        matched,
        on="id",
        how="left",
        validate="one_to_one",
    )

    df[delay_column] = (
        df[future_time_column] - df[target_column]
    ).dt.total_seconds()

    df[return_column] = (
        df[future_price_column] / df["current_price"] - 1.0
    )

    valid_mask = (
        df[future_price_column].notna()
        & np.isfinite(df[return_column])
    )

    df[valid_column] = valid_mask.astype("int8")

    direction = np.select(
        [
            valid_mask
            & (df[return_column] > neutral_threshold),
            valid_mask
            & (df[return_column] < -neutral_threshold),
            valid_mask,
        ],
        [
            1,
            -1,
            0,
        ],
        default=np.nan,
    )

    df[direction_column] = pd.Series(
        direction,
        index=df.index,
    ).astype("Int8")

    return df


def print_label_summary(
    df: pd.DataFrame,
    horizon_name: str,
) -> None:
    valid_column = f"label_valid_{horizon_name}"
    direction_column = f"direction_{horizon_name}"
    return_column = f"forward_return_{horizon_name}"
    delay_column = f"future_delay_seconds_{horizon_name}"

    valid_rows = int(df[valid_column].sum())
    missing_rows = int(len(df) - valid_rows)
    valid_pct = valid_rows / len(df) * 100

    print()
    print("-" * 72)
    print(f"{horizon_name.upper()} LABEL SUMMARY")
    print("-" * 72)
    print(
        f"Valid labels    : {valid_rows:,} "
        f"({valid_pct:.4f}%)"
    )
    print(f"Missing labels  : {missing_rows:,}")

    print()
    print("Direction counts:")
    print(
        df[direction_column]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    valid_returns = df.loc[
        df[valid_column] == 1,
        return_column,
    ]

    print()
    print("Forward return profile:")
    print(
        valid_returns.describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print()
    print("Match delay profile, seconds:")
    print(
        df.loc[
            df[valid_column] == 1,
            delay_column,
        ].describe(
            percentiles=[0.50, 0.95, 0.99]
        ).to_string()
    )


def main() -> int:
    started = time.time()

    if not INPUT_PATH.exists():
        print(
            f"ERROR: Input file not found: {INPUT_PATH}",
            file=sys.stderr,
        )
        return 1

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("BUILD TOPOLOGY V2 FORWARD LABELS")
    print("=" * 72)
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    df = pd.read_parquet(INPUT_PATH)
    validate_input(df)

    original_rows = len(df)

    df = df.sort_values(
        [
            "symbol",
            "timeframe",
            "logged_at",
            "id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    for horizon_name, horizon in HORIZONS.items():
        df = add_forward_labels(
            df=df,
            horizon_name=horizon_name,
            horizon=horizon,
            tolerance=TOLERANCES[horizon_name],
            neutral_threshold=NEUTRAL_THRESHOLDS[
                horizon_name
            ],
        )

    if len(df) != original_rows:
        raise RuntimeError(
            f"Row count changed: "
            f"{original_rows:,} -> {len(df):,}"
        )

    numeric = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric).sum().sum())

    if inf_count:
        raise RuntimeError(
            f"Output contains {inf_count:,} infinite values."
        )

    # Target timestamps are temporary calculation columns.
    target_columns = [
        f"target_time_{name}"
        for name in HORIZONS
    ]
    df = df.drop(columns=target_columns)

    for horizon_name in HORIZONS:
        print_label_summary(df, horizon_name)

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    output_size_mb = (
        OUTPUT_PATH.stat().st_size / 1024 / 1024
    )
    elapsed = time.time() - started

    print()
    print("=" * 72)
    print("FORWARD LABEL BUILD COMPLETE")
    print("=" * 72)
    print(f"Rows            : {len(df):,}")
    print(f"Columns         : {len(df.columns):,}")
    print(f"Infinite values : {inf_count:,}")
    print(f"Output size     : {output_size_mb:,.1f} MB")
    print(f"Elapsed seconds : {elapsed:,.1f}")
    print(f"Output          : {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
