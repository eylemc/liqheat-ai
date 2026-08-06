from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

OUTPUT_PATH = Path(
    "data/features/liq_topology_v2_sweep_labels.parquet"
)

HORIZONS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}

# Küçük tutulması RAM kullanımını ciddi biçimde azaltır.
CHUNK_SIZE = 2_000

LABEL_INVALID = -9
LABEL_LOWER_FIRST = -1
LABEL_NONE = 0
LABEL_UPPER_FIRST = 1
LABEL_BOTH = 2

LABEL_NAMES = {
    LABEL_INVALID: "INVALID",
    LABEL_LOWER_FIRST: "LOWER_FIRST",
    LABEL_NONE: "NONE",
    LABEL_UPPER_FIRST: "UPPER_FIRST",
    LABEL_BOTH: "BOTH",
}

REQUIRED_COLUMNS = [
    "id",
    "logged_at",
    "symbol",
    "timeframe",
    "current_price",
    "nearest_upper_price",
    "nearest_lower_price",
]


def validate_input(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs detected.")

    if df["logged_at"].isna().any():
        raise ValueError("Null timestamps detected.")

    if (df["current_price"] <= 0).any():
        raise ValueError(
            "Non-positive current prices detected."
        )


def build_group_labels(
    group: pd.DataFrame,
    horizon: pd.Timedelta,
) -> dict[str, np.ndarray]:
    """
    For each row, inspect later prices inside:
        (logged_at, logged_at + horizon]

    Starting upper/lower levels remain fixed.
    """

    row_count = len(group)

    # Force nanosecond timestamps explicitly.
    # The parquet source is datetime64[us, UTC]; using astype("int64")
    # directly would return microseconds and corrupt horizon calculations.
    times_ns = (
        group["logged_at"]
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    prices = group[
        "current_price"
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    upper_levels = group[
        "nearest_upper_price"
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    lower_levels = group[
        "nearest_lower_price"
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    horizon_ns = int(horizon.value)

    labels = np.full(
        row_count,
        LABEL_INVALID,
        dtype=np.int8,
    )

    upper_hit_seconds = np.full(
        row_count,
        np.nan,
        dtype=np.float32,
    )

    lower_hit_seconds = np.full(
        row_count,
        np.nan,
        dtype=np.float32,
    )

    first_hit_seconds = np.full(
        row_count,
        np.nan,
        dtype=np.float32,
    )

    valid_levels = (
        np.isfinite(upper_levels)
        & np.isfinite(lower_levels)
        & (upper_levels >= prices)
        & (lower_levels <= prices)
    )

    valid_rows = np.flatnonzero(valid_levels)

    for chunk_start in range(
        0,
        len(valid_rows),
        CHUNK_SIZE,
    ):
        selected = valid_rows[
            chunk_start:
            chunk_start + CHUNK_SIZE
        ]

        target_end_times = (
            times_ns[selected] + horizon_ns
        )

        end_indices = np.searchsorted(
            times_ns,
            target_end_times,
            side="right",
        )

        future_lengths = (
            end_indices - selected - 1
        )

        max_steps = int(
            future_lengths.max(initial=0)
        )

        if max_steps <= 0:
            labels[selected] = LABEL_NONE
            continue

        offsets = np.arange(
            1,
            max_steps + 1,
            dtype=np.int32,
        )

        future_indices = (
            selected[:, None]
            + offsets[None, :]
        )

        valid_future = (
            future_indices
            < end_indices[:, None]
        )

        safe_indices = np.minimum(
            future_indices,
            row_count - 1,
        )

        future_prices = prices[safe_indices]

        upper_hits = (
            valid_future
            & (
                future_prices
                >= upper_levels[selected, None]
            )
        )

        lower_hits = (
            valid_future
            & (
                future_prices
                <= lower_levels[selected, None]
            )
        )

        has_upper = upper_hits.any(axis=1)
        has_lower = lower_hits.any(axis=1)

        first_upper_offset = (
            np.argmax(upper_hits, axis=1) + 1
        )

        first_lower_offset = (
            np.argmax(lower_hits, axis=1) + 1
        )

        none_mask = (
            ~has_upper
            & ~has_lower
        )

        upper_only = (
            has_upper
            & ~has_lower
        )

        lower_only = (
            ~has_upper
            & has_lower
        )

        both = (
            has_upper
            & has_lower
        )

        upper_first = (
            both
            & (
                first_upper_offset
                < first_lower_offset
            )
        )

        lower_first = (
            both
            & (
                first_lower_offset
                < first_upper_offset
            )
        )

        simultaneous = (
            both
            & (
                first_upper_offset
                == first_lower_offset
            )
        )

        chunk_labels = np.full(
            len(selected),
            LABEL_NONE,
            dtype=np.int8,
        )

        chunk_labels[upper_only] = (
            LABEL_UPPER_FIRST
        )
        chunk_labels[lower_only] = (
            LABEL_LOWER_FIRST
        )
        chunk_labels[upper_first] = (
            LABEL_UPPER_FIRST
        )
        chunk_labels[lower_first] = (
            LABEL_LOWER_FIRST
        )
        chunk_labels[simultaneous] = LABEL_BOTH
        chunk_labels[none_mask] = LABEL_NONE

        labels[selected] = chunk_labels

        upper_rows = np.flatnonzero(has_upper)

        if len(upper_rows):
            source_rows = selected[upper_rows]
            hit_indices = (
                source_rows
                + first_upper_offset[upper_rows]
            )

            upper_hit_seconds[source_rows] = (
                (
                    times_ns[hit_indices]
                    - times_ns[source_rows]
                )
                / 1_000_000_000
            ).astype(np.float32)

        lower_rows = np.flatnonzero(has_lower)

        if len(lower_rows):
            source_rows = selected[lower_rows]
            hit_indices = (
                source_rows
                + first_lower_offset[lower_rows]
            )

            lower_hit_seconds[source_rows] = (
                (
                    times_ns[hit_indices]
                    - times_ns[source_rows]
                )
                / 1_000_000_000
            ).astype(np.float32)

        first_offsets = np.full(
            len(selected),
            -1,
            dtype=np.int32,
        )

        first_offsets[upper_only] = (
            first_upper_offset[upper_only]
        )

        first_offsets[lower_only] = (
            first_lower_offset[lower_only]
        )

        first_offsets[upper_first] = (
            first_upper_offset[upper_first]
        )

        first_offsets[lower_first] = (
            first_lower_offset[lower_first]
        )

        first_offsets[simultaneous] = (
            first_upper_offset[simultaneous]
        )

        hit_rows = np.flatnonzero(
            first_offsets >= 0
        )

        if len(hit_rows):
            source_rows = selected[hit_rows]
            hit_indices = (
                source_rows
                + first_offsets[hit_rows]
            )

            first_hit_seconds[source_rows] = (
                (
                    times_ns[hit_indices]
                    - times_ns[source_rows]
                )
                / 1_000_000_000
            ).astype(np.float32)

        # Geçici büyük matrisleri chunk sonunda bırak.
        del (
            future_indices,
            valid_future,
            safe_indices,
            future_prices,
            upper_hits,
            lower_hits,
        )

    return {
        "code": labels,
        "upper_seconds": upper_hit_seconds,
        "lower_seconds": lower_hit_seconds,
        "first_seconds": first_hit_seconds,
    }


def print_summary(
    output: pd.DataFrame,
    horizon_name: str,
) -> None:
    code_column = f"sweep_code_{horizon_name}"
    label_column = f"sweep_label_{horizon_name}"

    print()
    print("-" * 76)
    print(f"{horizon_name.upper()} SWEEP LABEL SUMMARY")
    print("-" * 76)

    counts = (
        output[label_column]
        .value_counts(dropna=False)
    )

    summary = pd.DataFrame(
        {
            "rows": counts,
            "pct": counts / len(output) * 100,
        }
    )

    print(summary.to_string())

    valid_mask = (
        output[code_column] != LABEL_INVALID
    )

    hit_mask = output[code_column].isin(
        [
            LABEL_LOWER_FIRST,
            LABEL_UPPER_FIRST,
            LABEL_BOTH,
        ]
    )

    print()
    print(
        f"Valid rows : {int(valid_mask.sum()):,} "
        f"({valid_mask.mean() * 100:.4f}%)"
    )

    print(
        f"Any hit    : {int(hit_mask.sum()):,} "
        f"({hit_mask.sum() / max(valid_mask.sum(), 1) * 100:.4f}% "
        f"of valid)"
    )

    if hit_mask.any():
        print()
        print("First-hit seconds:")
        print(
            output.loc[
                hit_mask,
                f"first_hit_seconds_{horizon_name}",
            ]
            .describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ]
            )
            .to_string()
        )

    print()
    print("By symbol (%):")
    print(
        pd.crosstab(
            output["symbol"],
            output[label_column],
            normalize="index",
        )
        .mul(100)
        .round(2)
        .to_string()
    )

    print()
    print("By timeframe (%):")
    print(
        pd.crosstab(
            output["timeframe"],
            output[label_column],
            normalize="index",
        )
        .mul(100)
        .round(2)
        .to_string()
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

    print("=" * 76)
    print("BUILD TOPOLOGY V2 SWEEP LABELS — LOW MEMORY")
    print("=" * 76)
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Chunk : {CHUNK_SIZE:,}")
    print()

    print("Loading only required columns...")

    df = pd.read_parquet(
        INPUT_PATH,
        columns=REQUIRED_COLUMNS,
    )

    validate_input(df)

    df = df.sort_values(
        [
            "symbol",
            "timeframe",
            "logged_at",
            "id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    row_count = len(df)

    output = df[
        [
            "id",
            "logged_at",
            "symbol",
            "timeframe",
        ]
    ].copy()

    grouped_indices = list(
        df.groupby(
            ["symbol", "timeframe"],
            sort=False,
            observed=True,
        ).indices.items()
    )

    for horizon_name, horizon in HORIZONS.items():
        print()
        print(
            f"Building {horizon_name} sweep labels..."
        )

        all_codes = np.full(
            row_count,
            LABEL_INVALID,
            dtype=np.int8,
        )

        all_upper_seconds = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )

        all_lower_seconds = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )

        all_first_seconds = np.full(
            row_count,
            np.nan,
            dtype=np.float32,
        )

        for (
            symbol,
            timeframe,
        ), indices in grouped_indices:
            group_started = time.time()

            # Sıralanmış dataframe nedeniyle grup indeksleri
            # bitişik olmalı.
            start_index = int(indices[0])
            end_index = int(indices[-1]) + 1

            group = df.iloc[
                start_index:end_index
            ]

            print(
                f"  {symbol:<8} {timeframe:<3} "
                f"rows={len(group):,}",
                flush=True,
            )

            result = build_group_labels(
                group=group,
                horizon=horizon,
            )

            all_codes[
                start_index:end_index
            ] = result["code"]

            all_upper_seconds[
                start_index:end_index
            ] = result["upper_seconds"]

            all_lower_seconds[
                start_index:end_index
            ] = result["lower_seconds"]

            all_first_seconds[
                start_index:end_index
            ] = result["first_seconds"]

            print(
                f"    completed in "
                f"{time.time() - group_started:.1f}s",
                flush=True,
            )

        code_column = f"sweep_code_{horizon_name}"

        output[code_column] = all_codes

        output[
            f"sweep_valid_{horizon_name}"
        ] = (
            all_codes != LABEL_INVALID
        ).astype("int8")

        output[
            f"sweep_label_{horizon_name}"
        ] = pd.Categorical(
            [
                LABEL_NAMES[int(code)]
                for code in all_codes
            ],
            categories=[
                "LOWER_FIRST",
                "NONE",
                "UPPER_FIRST",
                "BOTH",
                "INVALID",
            ],
        )

        output[
            f"upper_hit_seconds_{horizon_name}"
        ] = all_upper_seconds

        output[
            f"lower_hit_seconds_{horizon_name}"
        ] = all_lower_seconds

        output[
            f"first_hit_seconds_{horizon_name}"
        ] = all_first_seconds

        print_summary(
            output,
            horizon_name,
        )

    print()
    print("Writing compact label parquet...")

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    elapsed = time.time() - started
    size_mb = (
        OUTPUT_PATH.stat().st_size
        / 1024
        / 1024
    )

    print()
    print("=" * 76)
    print("SWEEP LABEL BUILD COMPLETE")
    print("=" * 76)
    print(f"Rows            : {len(output):,}")
    print(f"Columns         : {len(output.columns):,}")
    print(f"Output size     : {size_mb:,.1f} MB")
    print(f"Elapsed seconds : {elapsed:,.1f}")
    print(f"Output          : {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
