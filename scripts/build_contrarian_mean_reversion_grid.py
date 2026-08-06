from pathlib import Path
import json
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

OUTPUT_DIR = Path(
    "data/features/mean_reversion_outcomes"
)

OUTCOMES_PATH = (
    OUTPUT_DIR
    / "contrarian_mean_reversion_outcomes.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "mean_reversion_grid_summary.csv"
)

GROUP_SUMMARY_PATH = (
    OUTPUT_DIR
    / "mean_reversion_group_summary.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "mean_reversion_grid_report.json"
)

POST_ENTRY_WINDOW = pd.Timedelta(minutes=15)

TP_BPS_VALUES = [10, 15, 20, 25, 40]
SL_BPS_VALUES = [10, 15, 20, 25, 40]

ENTRY_FEE_BPS = 5.0
EXIT_FEE_BPS = 5.0
ENTRY_SLIPPAGE_BPS = 2.0
EXIT_SLIPPAGE_BPS = 2.0

ROUND_TRIP_COST_BPS = (
    ENTRY_FEE_BPS
    + EXIT_FEE_BPS
    + ENTRY_SLIPPAGE_BPS
    + EXIT_SLIPPAGE_BPS
)

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

OUTCOME_TIME = 0
OUTCOME_TP = 1
OUTCOME_SL = -1
OUTCOME_SIMULTANEOUS = 2


def combo_name(tp_bps: int, sl_bps: int) -> str:
    return f"tp{tp_bps}_sl{sl_bps}"


def validate(df: pd.DataFrame) -> None:
    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs detected.")

    if df["logged_at"].isna().any():
        raise ValueError("Null timestamps detected.")

    if (df["current_price"] <= 0).any():
        raise ValueError(
            "Non-positive current prices detected."
        )


def safe_profit_factor(
    net_returns: np.ndarray,
) -> float | None:
    gains = net_returns[net_returns > 0].sum()
    losses = -net_returns[net_returns < 0].sum()

    if losses <= 0:
        return None

    return float(gains / losses)


def max_drawdown(
    net_returns: np.ndarray,
) -> float:
    if len(net_returns) == 0:
        return 0.0

    equity = np.cumprod(
        1.0 + net_returns
    )

    peaks = np.maximum.accumulate(equity)

    return float(
        np.min(equity / peaks - 1.0)
    )


def build_group_outcomes(
    group: pd.DataFrame,
) -> pd.DataFrame:
    row_count = len(group)

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

    nearest_side = (
        group["nearest_side"]
        .astype("string")
        .fillna("<MISSING>")
        .to_numpy()
    )

    sweep_codes = group[
        "sweep_code_1h"
    ].to_numpy(
        dtype=np.int8,
        copy=True,
    )

    first_hit_seconds = group[
        "first_hit_seconds_1h"
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    # Contrarian sweep:
    # nearest LOWER + upper swept first
    # nearest UPPER + lower swept first
    contrarian_mask = (
        (
            (nearest_side == "LOWER")
            & (sweep_codes == 1)
        )
        |
        (
            (nearest_side == "UPPER")
            & (sweep_codes == -1)
        )
    ) & np.isfinite(first_hit_seconds)

    source_rows = np.flatnonzero(
        contrarian_mask
    )

    if len(source_rows) == 0:
        return pd.DataFrame()

    expected_entry_times_ns = (
        times_ns[source_rows]
        + np.rint(
            first_hit_seconds[source_rows]
            * 1_000_000_000
        ).astype(np.int64)
    )

    entry_indices = np.searchsorted(
        times_ns,
        expected_entry_times_ns,
        side="left",
    )

    valid_entry = (
        (entry_indices > source_rows)
        & (entry_indices < row_count)
    )

    source_rows = source_rows[
        valid_entry
    ]

    entry_indices = entry_indices[
        valid_entry
    ]

    if len(source_rows) == 0:
        return pd.DataFrame()

    # Mean-reversion direction:
    # upper/far sweep -> SHORT
    # lower/far sweep -> LONG
    directions = np.where(
        nearest_side[source_rows] == "LOWER",
        -1,
        1,
    ).astype(np.int8)

    event_count = len(source_rows)

    base_data = {
        "id": group.iloc[source_rows][
            "id"
        ].to_numpy(),
        "logged_at": group.iloc[source_rows][
            "logged_at"
        ].to_numpy(),
        "symbol": group.iloc[source_rows][
            "symbol"
        ].to_numpy(),
        "timeframe": group.iloc[source_rows][
            "timeframe"
        ].to_numpy(),
        "nearest_side": nearest_side[
            source_rows
        ],
        "mean_reversion_direction_code": (
            directions
        ),
        "entry_time": pd.to_datetime(
            times_ns[entry_indices],
            utc=True,
        ),
        "entry_price": prices[entry_indices],
        "entry_delay_seconds": (
            times_ns[entry_indices]
            - times_ns[source_rows]
        ) / 1_000_000_000,
    }

    outcome_arrays = {}

    for tp_bps in TP_BPS_VALUES:
        for sl_bps in SL_BPS_VALUES:
            name = combo_name(
                tp_bps,
                sl_bps,
            )

            outcome_arrays[
                f"{name}_outcome"
            ] = np.full(
                event_count,
                OUTCOME_TIME,
                dtype=np.int8,
            )

            outcome_arrays[
                f"{name}_gross_return"
            ] = np.full(
                event_count,
                np.nan,
                dtype=np.float32,
            )

            outcome_arrays[
                f"{name}_net_return"
            ] = np.full(
                event_count,
                np.nan,
                dtype=np.float32,
            )

            outcome_arrays[
                f"{name}_holding_seconds"
            ] = np.full(
                event_count,
                np.nan,
                dtype=np.float32,
            )

    window_ns = int(
        POST_ENTRY_WINDOW.value
    )

    for chunk_start in range(
        0,
        event_count,
        CHUNK_SIZE,
    ):
        chunk_end = min(
            chunk_start + CHUNK_SIZE,
            event_count,
        )

        chunk_entry_indices = (
            entry_indices[
                chunk_start:chunk_end
            ]
        )

        chunk_directions = (
            directions[
                chunk_start:chunk_end
            ].astype(np.float64)
        )

        chunk_entry_prices = prices[
            chunk_entry_indices
        ]

        window_end_times = (
            times_ns[chunk_entry_indices]
            + window_ns
        )

        end_indices = np.searchsorted(
            times_ns,
            window_end_times,
            side="right",
        )

        lengths = (
            end_indices
            - chunk_entry_indices
        )

        max_steps = int(
            lengths.max(initial=0)
        )

        if max_steps <= 0:
            continue

        offsets = np.arange(
            max_steps,
            dtype=np.int32,
        )

        future_indices = (
            chunk_entry_indices[:, None]
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

        future_prices = prices[
            safe_indices
        ]

        directional_returns = (
            chunk_directions[:, None]
            * (
                future_prices
                / chunk_entry_prices[:, None]
                - 1.0
            )
        )

        directional_returns = np.where(
            valid_future,
            directional_returns,
            np.nan,
        )

        last_valid_offset = np.maximum(
            lengths - 1,
            0,
        )

        row_numbers = np.arange(
            chunk_end - chunk_start
        )

        time_exit_returns = (
            directional_returns[
                row_numbers,
                last_valid_offset,
            ]
        )

        for tp_bps in TP_BPS_VALUES:
            tp_return = (
                tp_bps / 10_000
            )

            tp_hits = (
                valid_future
                & (
                    directional_returns
                    >= tp_return
                )
            )

            has_tp = tp_hits.any(
                axis=1
            )

            first_tp_offset = np.argmax(
                tp_hits,
                axis=1,
            )

            for sl_bps in SL_BPS_VALUES:
                sl_return = (
                    sl_bps / 10_000
                )

                name = combo_name(
                    tp_bps,
                    sl_bps,
                )

                sl_hits = (
                    valid_future
                    & (
                        directional_returns
                        <= -sl_return
                    )
                )

                has_sl = sl_hits.any(
                    axis=1
                )

                first_sl_offset = (
                    np.argmax(
                        sl_hits,
                        axis=1,
                    )
                )

                tp_first = (
                    has_tp
                    & (
                        ~has_sl
                        | (
                            first_tp_offset
                            < first_sl_offset
                        )
                    )
                )

                sl_first = (
                    has_sl
                    & (
                        ~has_tp
                        | (
                            first_sl_offset
                            < first_tp_offset
                        )
                    )
                )

                simultaneous = (
                    has_tp
                    & has_sl
                    & (
                        first_tp_offset
                        == first_sl_offset
                    )
                )

                outcomes = np.full(
                    chunk_end - chunk_start,
                    OUTCOME_TIME,
                    dtype=np.int8,
                )

                outcomes[
                    tp_first
                ] = OUTCOME_TP

                outcomes[
                    sl_first
                ] = OUTCOME_SL

                # Conservative.
                outcomes[
                    simultaneous
                ] = OUTCOME_SIMULTANEOUS

                gross_returns = (
                    time_exit_returns.copy()
                )

                gross_returns[
                    tp_first
                ] = tp_return

                gross_returns[
                    sl_first
                ] = -sl_return

                gross_returns[
                    simultaneous
                ] = -sl_return

                exit_offsets = (
                    last_valid_offset.copy()
                )

                exit_offsets[
                    tp_first
                ] = first_tp_offset[
                    tp_first
                ]

                exit_offsets[
                    sl_first
                ] = first_sl_offset[
                    sl_first
                ]

                exit_offsets[
                    simultaneous
                ] = first_sl_offset[
                    simultaneous
                ]

                exit_indices = (
                    chunk_entry_indices
                    + exit_offsets
                )

                holding_seconds = (
                    times_ns[exit_indices]
                    - times_ns[
                        chunk_entry_indices
                    ]
                ) / 1_000_000_000

                net_returns = (
                    gross_returns
                    - (
                        ROUND_TRIP_COST_BPS
                        / 10_000
                    )
                )

                outcome_arrays[
                    f"{name}_outcome"
                ][chunk_start:chunk_end] = (
                    outcomes
                )

                outcome_arrays[
                    f"{name}_gross_return"
                ][chunk_start:chunk_end] = (
                    gross_returns.astype(
                        np.float32
                    )
                )

                outcome_arrays[
                    f"{name}_net_return"
                ][chunk_start:chunk_end] = (
                    net_returns.astype(
                        np.float32
                    )
                )

                outcome_arrays[
                    f"{name}_holding_seconds"
                ][chunk_start:chunk_end] = (
                    holding_seconds.astype(
                        np.float32
                    )
                )

        del (
            future_indices,
            valid_future,
            safe_indices,
            future_prices,
            directional_returns,
        )

    return pd.DataFrame(
        {
            **base_data,
            **outcome_arrays,
        }
    )


def summarize_combo(
    outcomes: pd.DataFrame,
    tp_bps: int,
    sl_bps: int,
) -> dict:
    name = combo_name(
        tp_bps,
        sl_bps,
    )

    codes = outcomes[
        f"{name}_outcome"
    ].to_numpy(dtype=np.int8)

    gross_returns = outcomes[
        f"{name}_gross_return"
    ].to_numpy(dtype=np.float64)

    net_returns = outcomes[
        f"{name}_net_return"
    ].to_numpy(dtype=np.float64)

    holding = outcomes[
        f"{name}_holding_seconds"
    ].to_numpy(dtype=np.float64)

    valid = np.isfinite(
        net_returns
    )

    codes = codes[valid]
    gross_returns = gross_returns[
        valid
    ]
    net_returns = net_returns[
        valid
    ]
    holding = holding[valid]

    events = len(net_returns)

    tp_count = int(
        (codes == OUTCOME_TP).sum()
    )

    sl_count = int(
        (
            (codes == OUTCOME_SL)
            | (
                codes
                == OUTCOME_SIMULTANEOUS
            )
        ).sum()
    )

    time_count = int(
        (codes == OUTCOME_TIME).sum()
    )

    net_win_bps = (
        tp_bps
        - ROUND_TRIP_COST_BPS
    )

    net_loss_bps = (
        sl_bps
        + ROUND_TRIP_COST_BPS
    )

    break_even_tp_rate = (
        net_loss_bps
        / (
            net_win_bps
            + net_loss_bps
        )
        if net_win_bps > 0
        else None
    )

    return {
        "tp_bps": tp_bps,
        "sl_bps": sl_bps,
        "events": events,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "time_count": time_count,
        "tp_rate": (
            tp_count / events
            if events
            else 0.0
        ),
        "positive_net_rate": float(
            (net_returns > 0).mean()
        ) if events else 0.0,
        "gross_expectancy_bps": float(
            gross_returns.mean()
            * 10_000
        ),
        "net_expectancy_bps": float(
            net_returns.mean()
            * 10_000
        ),
        "median_net_bps": float(
            np.median(net_returns)
            * 10_000
        ),
        "profit_factor": (
            safe_profit_factor(
                net_returns
            )
        ),
        "max_drawdown": (
            max_drawdown(
                net_returns
            )
        ),
        "mean_holding_seconds": float(
            holding.mean()
        ),
        "break_even_tp_rate_simple": (
            break_even_tp_rate
        ),
        "net_win_bps": net_win_bps,
        "net_loss_bps": net_loss_bps,
        "round_trip_cost_bps": (
            ROUND_TRIP_COST_BPS
        ),
    }


def build_group_summary(
    outcomes: pd.DataFrame,
    best_tp: int,
    best_sl: int,
) -> pd.DataFrame:
    name = combo_name(
        best_tp,
        best_sl,
    )

    rows = []

    for group_type, columns in [
        ("symbol", ["symbol"]),
        ("timeframe", ["timeframe"]),
        (
            "symbol_timeframe",
            ["symbol", "timeframe"],
        ),
    ]:
        for key, group in outcomes.groupby(
            columns,
            observed=True,
            dropna=False,
        ):
            if not isinstance(
                key,
                tuple,
            ):
                key = (key,)

            net_returns = group[
                f"{name}_net_return"
            ].to_numpy(
                dtype=np.float64
            )

            rows.append(
                {
                    "group_type": (
                        group_type
                    ),
                    "group_value": " / ".join(
                        str(value)
                        for value in key
                    ),
                    "tp_bps": best_tp,
                    "sl_bps": best_sl,
                    "events": int(
                        len(group)
                    ),
                    "positive_net_rate": float(
                        (
                            net_returns > 0
                        ).mean()
                    ),
                    "mean_net_bps": float(
                        net_returns.mean()
                        * 10_000
                    ),
                    "median_net_bps": float(
                        np.median(
                            net_returns
                        )
                        * 10_000
                    ),
                    "profit_factor": (
                        safe_profit_factor(
                            net_returns
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    started = time.time()

    for path in [
        FEATURE_PATH,
        SWEEP_PATH,
    ]:
        if not path.exists():
            print(
                f"ERROR: Missing file: "
                f"{path}",
                file=sys.stderr,
            )
            return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 84)
    print(
        "CONTRARIAN SWEEP — "
        "MEAN-REVERSION OUTCOME GRID"
    )
    print("=" * 84)
    print(f"Feature input : {FEATURE_PATH}")
    print(f"Sweep input   : {SWEEP_PATH}")
    print(f"Output        : {OUTCOMES_PATH}")
    print(f"Window        : {POST_ENTRY_WINDOW}")
    print(f"TP grid       : {TP_BPS_VALUES}")
    print(f"SL grid       : {SL_BPS_VALUES}")
    print(
        f"Round-trip cost: "
        f"{ROUND_TRIP_COST_BPS:.1f} bps"
    )
    print()

    print("Loading features...")

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=(
            REQUIRED_FEATURE_COLUMNS
        ),
    )

    print("Loading sweep labels...")

    sweeps = pd.read_parquet(
        SWEEP_PATH,
        columns=(
            REQUIRED_SWEEP_COLUMNS
        ),
    )

    if sweeps["id"].duplicated().any():
        raise ValueError(
            "Duplicate IDs in sweep data."
        )

    df = features.merge(
        sweeps,
        on="id",
        how="inner",
        validate="one_to_one",
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

    output_frames = []

    print()
    print("Building mean-reversion outcomes...")

    grouped = df.groupby(
        ["symbol", "timeframe"],
        sort=False,
        observed=True,
    )

    for (
        symbol,
        timeframe,
    ), group in grouped:
        group_started = time.time()

        print(
            f"  {symbol:<8} "
            f"{timeframe:<3} "
            f"rows={len(group):,}",
            flush=True,
        )

        result = build_group_outcomes(
            group.reset_index(
                drop=True
            )
        )

        if not result.empty:
            output_frames.append(
                result
            )

        print(
            f"    events={len(result):,} "
            f"completed in "
            f"{time.time() - group_started:.2f}s",
            flush=True,
        )

    if not output_frames:
        raise RuntimeError(
            "No mean-reversion events generated."
        )

    outcomes = pd.concat(
        output_frames,
        ignore_index=True,
    )

    outcomes = outcomes.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    print()
    print(
        f"Mean-reversion events: "
        f"{len(outcomes):,}"
    )

    print("Writing outcome parquet...")

    outcomes.to_parquet(
        OUTCOMES_PATH,
        index=False,
        compression="zstd",
    )

    summary_rows = []

    for tp_bps in TP_BPS_VALUES:
        for sl_bps in SL_BPS_VALUES:
            summary_rows.append(
                summarize_combo(
                    outcomes,
                    tp_bps,
                    sl_bps,
                )
            )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "net_expectancy_bps",
            "profit_factor",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    best_row = summary.iloc[0]

    best_tp = int(
        best_row["tp_bps"]
    )

    best_sl = int(
        best_row["sl_bps"]
    )

    group_summary = build_group_summary(
        outcomes,
        best_tp,
        best_sl,
    )

    group_summary.to_csv(
        GROUP_SUMMARY_PATH,
        index=False,
    )

    positive = summary.loc[
        summary[
            "net_expectancy_bps"
        ] > 0
    ]

    report = {
        "strategy": (
            "Fade the farther-side "
            "contrarian liquidity sweep"
        ),
        "events": int(len(outcomes)),
        "post_entry_window_minutes": (
            POST_ENTRY_WINDOW
            .total_seconds()
            / 60
        ),
        "tp_bps_values": (
            TP_BPS_VALUES
        ),
        "sl_bps_values": (
            SL_BPS_VALUES
        ),
        "costs": {
            "entry_fee_bps": (
                ENTRY_FEE_BPS
            ),
            "exit_fee_bps": (
                EXIT_FEE_BPS
            ),
            "entry_slippage_bps": (
                ENTRY_SLIPPAGE_BPS
            ),
            "exit_slippage_bps": (
                EXIT_SLIPPAGE_BPS
            ),
            "round_trip_cost_bps": (
                ROUND_TRIP_COST_BPS
            ),
        },
        "best_unfiltered": (
            best_row.to_dict()
        ),
        "positive_combinations": (
            positive.to_dict(
                orient="records"
            )
        ),
        "summary": summary.to_dict(
            orient="records"
        ),
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print()
    print("=" * 84)
    print("MEAN-REVERSION GRID SUMMARY")
    print("=" * 84)

    print(
        summary[
            [
                "tp_bps",
                "sl_bps",
                "events",
                "tp_rate",
                "positive_net_rate",
                "gross_expectancy_bps",
                "net_expectancy_bps",
                "profit_factor",
                "mean_holding_seconds",
                "break_even_tp_rate_simple",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )

    print()

    if positive.empty:
        print(
            "No TP/SL combination has "
            "positive unfiltered net expectancy."
        )
    else:
        print(
            "Positive unfiltered combinations:"
        )

        print(
            positive[
                [
                    "tp_bps",
                    "sl_bps",
                    "net_expectancy_bps",
                    "profit_factor",
                    "tp_rate",
                    "positive_net_rate",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"Best combination by net expectancy: "
        f"TP={best_tp}bp / SL={best_sl}bp"
    )

    print()
    print("Best combination by symbol:")
    print(
        group_summary.loc[
            group_summary[
                "group_type"
            ].eq("symbol")
        ]
        .sort_values(
            "mean_net_bps",
            ascending=False,
        )
        .to_string(index=False)
    )

    elapsed = time.time() - started

    size_mb = (
        OUTCOMES_PATH.stat().st_size
        / 1024
        / 1024
    )

    print()
    print("=" * 84)
    print(
        "MEAN-REVERSION OUTCOME "
        "BUILD COMPLETE"
    )
    print("=" * 84)
    print(f"Events       : {len(outcomes):,}")
    print(f"Outcomes     : {OUTCOMES_PATH}")
    print(f"Summary      : {SUMMARY_PATH}")
    print(f"Groups       : {GROUP_SUMMARY_PATH}")
    print(f"Report       : {REPORT_PATH}")
    print(f"Output size  : {size_mb:,.1f} MB")
    print(f"Elapsed      : {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
