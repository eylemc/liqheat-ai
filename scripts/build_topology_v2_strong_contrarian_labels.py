from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


FEATURE_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

SWEEP_PATH = Path(
    "data/features/liq_topology_v2_sweep_labels.parquet"
)

OUTPUT_PATH = Path(
    "data/features/liq_topology_v2_strong_contrarian_labels.parquet"
)

POST_HIT_WINDOW = pd.Timedelta(minutes=15)

THRESHOLDS = {
    "10bp": 0.0010,   # %0.10
    "25bp": 0.0025,   # %0.25
}

CHUNK_SIZE = 20_000

REQUIRED_FEATURE_COLUMNS = [
    "id",
    "logged_at",
    "symbol",
    "timeframe",
    "current_price",
    "nearest_side",
    "nearest_upper_price",
    "nearest_lower_price",
]

REQUIRED_SWEEP_COLUMNS = [
    "id",
    "sweep_code_1h",
    "first_hit_seconds_1h",
]


def validate(df: pd.DataFrame) -> None:
    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs detected.")

    if df["logged_at"].isna().any():
        raise ValueError("Null timestamps detected.")

    if (df["current_price"] <= 0).any():
        raise ValueError("Non-positive prices detected.")


def build_group_continuation(
    group: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Measure continuation after a contrarian first sweep.

    Contrarian cases:
      nearest_side=UPPER and sweep_code=-1 -> lower/farther side first
      nearest_side=LOWER and sweep_code=+1 -> upper/farther side first
    """

    row_count = len(group)

    times_ns = (
        group["logged_at"]
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )

    prices = group["current_price"].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    upper_levels = group["nearest_upper_price"].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    lower_levels = group["nearest_lower_price"].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    nearest_side = (
        group["nearest_side"]
        .astype("string")
        .fillna("<MISSING>")
        .to_numpy()
    )

    sweep_codes = group["sweep_code_1h"].to_numpy(
        dtype=np.int8,
        copy=True,
    )

    first_hit_seconds = group[
        "first_hit_seconds_1h"
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    clear_event = (
        np.isin(nearest_side, ["UPPER", "LOWER"])
        & np.isin(sweep_codes, [-1, 1])
        & np.isfinite(first_hit_seconds)
        & np.isfinite(upper_levels)
        & np.isfinite(lower_levels)
    )

    contrarian = (
        (
            (nearest_side == "UPPER")
            & (sweep_codes == -1)
        )
        |
        (
            (nearest_side == "LOWER")
            & (sweep_codes == 1)
        )
    ) & clear_event

    continuation_pct = np.full(
        row_count,
        np.nan,
        dtype=np.float32,
    )

    hit_delay_seconds = np.full(
        row_count,
        np.nan,
        dtype=np.float32,
    )

    contrarian_rows = np.flatnonzero(contrarian)

    if len(contrarian_rows) == 0:
        return {
            "eligible": clear_event.astype("int8"),
            "contrarian": contrarian.astype("int8"),
            "continuation_pct": continuation_pct,
            "hit_delay_seconds": hit_delay_seconds,
        }

    hit_times_ns = (
        times_ns[contrarian_rows]
        + np.rint(
            first_hit_seconds[contrarian_rows]
            * 1_000_000_000
        ).astype(np.int64)
    )

    # Resolve the future snapshot that generated the first-hit time.
    hit_indices = np.searchsorted(
        times_ns,
        hit_times_ns,
        side="left",
    )

    hit_indices = np.clip(
        hit_indices,
        0,
        row_count - 1,
    )

    hit_delay_seconds[contrarian_rows] = (
        (
            times_ns[hit_indices]
            - times_ns[contrarian_rows]
        )
        / 1_000_000_000
    ).astype(np.float32)

    window_ns = int(POST_HIT_WINDOW.value)

    for chunk_start in range(
        0,
        len(contrarian_rows),
        CHUNK_SIZE,
    ):
        selected_rows = contrarian_rows[
            chunk_start:
            chunk_start + CHUNK_SIZE
        ]

        selected_hit_indices = hit_indices[
            chunk_start:
            chunk_start + CHUNK_SIZE
        ]

        window_end_times = (
            times_ns[selected_hit_indices]
            + window_ns
        )

        end_indices = np.searchsorted(
            times_ns,
            window_end_times,
            side="right",
        )

        window_lengths = (
            end_indices
            - selected_hit_indices
        )

        maximum_steps = int(
            window_lengths.max(initial=0)
        )

        if maximum_steps <= 0:
            continue

        offsets = np.arange(
            maximum_steps,
            dtype=np.int32,
        )

        future_indices = (
            selected_hit_indices[:, None]
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

        # Invalid matrix cells must not affect min/max.
        future_for_max = np.where(
            valid_future,
            future_prices,
            -np.inf,
        )

        future_for_min = np.where(
            valid_future,
            future_prices,
            np.inf,
        )

        maximum_price = np.max(
            future_for_max,
            axis=1,
        )

        minimum_price = np.min(
            future_for_min,
            axis=1,
        )

        selected_nearest_side = nearest_side[
            selected_rows
        ]

        upper_contrarian = (
            selected_nearest_side == "LOWER"
        )

        lower_contrarian = (
            selected_nearest_side == "UPPER"
        )

        chunk_continuation = np.full(
            len(selected_rows),
            np.nan,
            dtype=np.float64,
        )

        # Farther upper side swept first:
        # continuation is maximum price beyond the upper level.
        chunk_continuation[upper_contrarian] = (
            maximum_price[upper_contrarian]
            / upper_levels[
                selected_rows[upper_contrarian]
            ]
            - 1.0
        )

        # Farther lower side swept first:
        # continuation is minimum price below the lower level.
        chunk_continuation[lower_contrarian] = (
            1.0
            - (
                minimum_price[lower_contrarian]
                / lower_levels[
                    selected_rows[lower_contrarian]
                ]
            )
        )

        # A sweep snapshot can land slightly inside the level.
        # Continuation cannot be negative for this label.
        chunk_continuation = np.maximum(
            chunk_continuation,
            0.0,
        )

        continuation_pct[selected_rows] = (
            chunk_continuation.astype(np.float32)
        )

        del (
            future_indices,
            valid_future,
            safe_indices,
            future_prices,
            future_for_max,
            future_for_min,
        )

    return {
        "eligible": clear_event.astype("int8"),
        "contrarian": contrarian.astype("int8"),
        "continuation_pct": continuation_pct,
        "hit_delay_seconds": hit_delay_seconds,
    }


def print_summary(output: pd.DataFrame) -> None:
    eligible = output["strong_label_eligible_1h"].eq(1)
    contrarian = output["contrarian_sweep_1h"].eq(1)

    print()
    print("-" * 78)
    print("STRONG CONTRARIAN LABEL SUMMARY")
    print("-" * 78)

    print(
        f"Eligible rows       : {int(eligible.sum()):,} "
        f"({eligible.mean() * 100:.4f}%)"
    )

    print(
        f"Contrarian sweeps   : {int(contrarian.sum()):,} "
        f"({contrarian.sum() / max(eligible.sum(), 1) * 100:.4f}% "
        f"of eligible)"
    )

    for threshold_name in THRESHOLDS:
        column = (
            f"strong_contrarian_{threshold_name}_1h"
        )

        positive = output[column].eq(1)

        print(
            f"Strong {threshold_name:<4}      : "
            f"{int(positive.sum()):,} "
            f"({positive.sum() / max(eligible.sum(), 1) * 100:.4f}% "
            f"of eligible; "
            f"{positive.sum() / max(contrarian.sum(), 1) * 100:.4f}% "
            f"of contrarian)"
        )

    valid_continuation = output.loc[
        contrarian,
        "post_hit_continuation_pct_1h",
    ]

    print()
    print("Contrarian continuation profile:")
    print(
        valid_continuation.describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print()
    print("Strong 25bp by symbol (% of eligible):")
    symbol_table = (
        output.loc[eligible]
        .groupby("symbol", observed=True)[
            "strong_contrarian_25bp_1h"
        ]
        .agg(["count", "sum", "mean"])
    )
    symbol_table["mean"] *= 100
    print(symbol_table.to_string())

    print()
    print("Strong 25bp by timeframe (% of eligible):")
    timeframe_table = (
        output.loc[eligible]
        .groupby("timeframe", observed=True)[
            "strong_contrarian_25bp_1h"
        ]
        .agg(["count", "sum", "mean"])
    )
    timeframe_table["mean"] *= 100
    print(timeframe_table.to_string())


def main() -> int:
    started = time.time()

    for path in [FEATURE_PATH, SWEEP_PATH]:
        if not path.exists():
            print(
                f"ERROR: Missing file: {path}",
                file=sys.stderr,
            )
            return 1

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("BUILD TOPOLOGY V2 STRONG CONTRARIAN LABELS")
    print("=" * 78)
    print(f"Feature input : {FEATURE_PATH}")
    print(f"Sweep input   : {SWEEP_PATH}")
    print(f"Output        : {OUTPUT_PATH}")
    print(f"Post-hit      : {POST_HIT_WINDOW}")
    print()

    print("Loading required feature columns...")

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=REQUIRED_FEATURE_COLUMNS,
    )

    print("Loading sweep columns...")

    sweeps = pd.read_parquet(
        SWEEP_PATH,
        columns=REQUIRED_SWEEP_COLUMNS,
    )

    if sweeps["id"].duplicated().any():
        raise ValueError(
            "Duplicate IDs in sweep file."
        )

    df = features.merge(
        sweeps,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(features):
        raise RuntimeError(
            f"Merge row mismatch: "
            f"{len(features):,} -> {len(df):,}"
        )

    validate(df)

    df = df.sort_values(
        [
            "symbol",
            "timeframe",
            "logged_at",
            "id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    output = df[
        [
            "id",
            "logged_at",
            "symbol",
            "timeframe",
        ]
    ].copy()

    row_count = len(df)

    all_eligible = np.zeros(
        row_count,
        dtype=np.int8,
    )

    all_contrarian = np.zeros(
        row_count,
        dtype=np.int8,
    )

    all_continuation = np.full(
        row_count,
        np.nan,
        dtype=np.float32,
    )

    all_hit_delay = np.full(
        row_count,
        np.nan,
        dtype=np.float32,
    )

    grouped = df.groupby(
        ["symbol", "timeframe"],
        sort=False,
        observed=True,
    )

    print()
    print("Building labels...")

    for (symbol, timeframe), group in grouped:
        group_started = time.time()

        start_index = int(group.index[0])
        end_index = int(group.index[-1]) + 1

        print(
            f"  {symbol:<8} {timeframe:<3} "
            f"rows={len(group):,}",
            flush=True,
        )

        result = build_group_continuation(group)

        all_eligible[start_index:end_index] = (
            result["eligible"]
        )

        all_contrarian[start_index:end_index] = (
            result["contrarian"]
        )

        all_continuation[start_index:end_index] = (
            result["continuation_pct"]
        )

        all_hit_delay[start_index:end_index] = (
            result["hit_delay_seconds"]
        )

        print(
            f"    completed in "
            f"{time.time() - group_started:.2f}s",
            flush=True,
        )

    output["strong_label_eligible_1h"] = (
        all_eligible
    )

    output["contrarian_sweep_1h"] = (
        all_contrarian
    )

    output["post_hit_continuation_pct_1h"] = (
        all_continuation
    )

    output["contrarian_hit_delay_seconds_1h"] = (
        all_hit_delay
    )

    for threshold_name, threshold in THRESHOLDS.items():
        column = (
            f"strong_contrarian_{threshold_name}_1h"
        )

        # -9 = invalid/not eligible
        labels = np.full(
            row_count,
            -9,
            dtype=np.int8,
        )

        eligible_mask = all_eligible == 1

        labels[eligible_mask] = 0

        positive_mask = (
            eligible_mask
            & (all_contrarian == 1)
            & np.isfinite(all_continuation)
            & (all_continuation >= threshold)
        )

        labels[positive_mask] = 1

        output[column] = labels

    print_summary(output)

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
    print("=" * 78)
    print("STRONG CONTRARIAN LABEL BUILD COMPLETE")
    print("=" * 78)
    print(f"Rows        : {len(output):,}")
    print(f"Columns     : {len(output.columns):,}")
    print(f"Output size : {size_mb:,.1f} MB")
    print(f"Elapsed     : {elapsed:.1f}s")
    print(f"Output      : {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
