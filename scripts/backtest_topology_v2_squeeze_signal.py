from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import json
import math
import sys
import time

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

FEATURE_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

EXPERIMENT_DIR = Path(
    "data/research/topology_v2_squeeze_grid/"
    "tf_1h__future_60m__precursor_5m__q_0p975"
)

PREDICTIONS_PATH = (
    EXPERIMENT_DIR
    / "walk_forward_predictions.parquet"
)

OUTPUT_DIR = Path(
    "data/backtests/topology_v2_squeeze_signal"
)

SIGNALS_PATH = (
    OUTPUT_DIR
    / "generated_signals.parquet"
)

TRADES_PATH = (
    OUTPUT_DIR
    / "all_trades.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "backtest_summary.csv"
)

TOP_CONFIGS_PATH = (
    OUTPUT_DIR
    / "top_configurations.csv"
)

FOLD_METRICS_PATH = (
    OUTPUT_DIR
    / "fold_metrics.csv"
)

SYMBOL_METRICS_PATH = (
    OUTPUT_DIR
    / "symbol_metrics.csv"
)

BEST_TRADES_PATH = (
    OUTPUT_DIR
    / "best_configuration_trades.parquet"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "report.json"
)


# =============================================================================
# DATA AND MODEL SEMANTICS
# =============================================================================

TOPOLOGY_TIMEFRAME = "1h"

# Model class:
#   -1 = LONG_SQUEEZE  = downside move = SHORT trade
#    0 = NO_EVENT
#    1 = SHORT_SQUEEZE = upside move   = LONG trade
LONG_SQUEEZE_CLASS = -1
NO_EVENT_CLASS = 0
SHORT_SQUEEZE_CLASS = 1

TRADE_LONG = 1
TRADE_SHORT = -1


# =============================================================================
# EXECUTION COSTS
# =============================================================================

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


# =============================================================================
# SIGNAL GRID
# =============================================================================

# Absolute squeeze probabilities.
SQUEEZE_THRESHOLDS = [
    0.70,
    0.80,
    0.90,
    0.95,
]

# A new alarm is armed only after score falls below:
# threshold - REARM_GAP.
REARM_GAP = 0.15

DIRECTION_CONFIDENCE_VALUES = [
    0.55,
    0.60,
    0.65,
    0.70,
]

COOLDOWN_MINUTES_VALUES = [
    60,
    120,
    240,
]


# =============================================================================
# EXECUTION GRID
# =============================================================================

# 0 = first available signal snapshot.
# 73 seconds approximates one LiqHeat refresh.
ENTRY_DELAY_SECONDS_VALUES = [
    0,
    73,
]

MAX_HOLD_MINUTES_VALUES = [
    30,
    60,
    120,
]

TARGET_MODES = [
    "nearest_liquidity",
    "fixed_50bp",
    "fixed_100bp",
]

STOP_MODES = [
    "opposite_liquidity",
    "fixed_50bp",
    "fixed_100bp",
]

# Do not trade when the first liquidation reward is too close.
MIN_TARGET_DISTANCE_BPS_VALUES = [
    0,
    25,
    50,
]

# Same symbol cannot have overlapping positions.
ONE_POSITION_PER_SYMBOL = True

# For realistic ranking.
MIN_TRADES_FOR_RANKING = 30

# Used for equity and drawdown reporting.
# This is not leverage; it is the fraction of equity allocated to each trade.
NOTIONAL_FRACTION_PER_TRADE = 0.10


# =============================================================================
# OUTPUT COLUMNS
# =============================================================================

FEATURE_COLUMNS = [
    "id",
    "logged_at",
    "symbol",
    "timeframe",
    "current_price",
    "nearest_upper_price",
    "nearest_lower_price",
    "upper_distance_pct",
    "lower_distance_pct",
    "upper_pool_volume",
    "lower_pool_volume",
    "upper_total_volume",
    "lower_total_volume",
    "topology_imbalance",
]


@dataclass(frozen=True)
class SignalConfig:
    squeeze_threshold: float
    direction_confidence: float
    cooldown_minutes: int


@dataclass(frozen=True)
class ExecutionConfig:
    entry_delay_seconds: int
    max_hold_minutes: int
    target_mode: str
    stop_mode: str
    min_target_distance_bps: int


def safe_float(value) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def profit_factor(returns: np.ndarray) -> float | None:
    positive = returns[returns > 0].sum()
    negative = -returns[returns < 0].sum()

    if negative <= 0:
        return None

    return float(positive / negative)


def compounded_equity(
    returns: np.ndarray,
) -> np.ndarray:
    if len(returns) == 0:
        return np.array([], dtype=np.float64)

    allocated_returns = (
        returns
        * NOTIONAL_FRACTION_PER_TRADE
    )

    return np.cumprod(
        1.0 + allocated_returns
    )


def max_drawdown(
    equity: np.ndarray,
) -> float:
    if len(equity) == 0:
        return 0.0

    peaks = np.maximum.accumulate(equity)

    drawdowns = (
        equity / peaks
        - 1.0
    )

    return float(drawdowns.min())


def load_data() -> tuple[
    pd.DataFrame,
    dict[tuple[str, str], dict],
]:
    for path in [
        FEATURE_PATH,
        PREDICTIONS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    print("Loading out-of-sample predictions...")

    predictions = pd.read_parquet(
        PREDICTIONS_PATH
    )

    required_prediction_columns = {
        "id",
        "logged_at",
        "symbol",
        "fold",
        "target_event",
        "prediction",
        "probability_long_squeeze",
        "probability_no_event",
        "probability_short_squeeze",
        "squeeze_probability",
    }

    missing_predictions = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_predictions:
        raise ValueError(
            "Prediction file is missing columns: "
            f"{sorted(missing_predictions)}"
        )

    print("Loading 1h topology and price history...")

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=FEATURE_COLUMNS,
        filters=[
            (
                "timeframe",
                "==",
                TOPOLOGY_TIMEFRAME,
            )
        ],
    )

    if features.empty:
        raise RuntimeError(
            "No 1h topology rows found."
        )

    if features["id"].duplicated().any():
        raise ValueError(
            "Duplicate feature IDs detected."
        )

    if predictions["id"].duplicated().any():
        raise ValueError(
            "Duplicate prediction IDs detected."
        )

    merged = predictions.merge(
        features,
        on=[
            "id",
            "logged_at",
            "symbol",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=("_prediction", "_feature"),
    )

    # Normalize the walk-forward fold column after merge.
    if "fold" not in merged.columns:
        fold_candidates = [
            column
            for column in [
                "fold_prediction",
                "fold_x",
                "fold_feature",
                "fold_y",
            ]
            if column in merged.columns
        ]

        if not fold_candidates:
            raise ValueError(
                "No fold column found after prediction/feature merge. "
                f"Available columns: {sorted(merged.columns.tolist())}"
            )

        merged["fold"] = merged[fold_candidates[0]]

    if merged["fold"].isna().any():
        raise ValueError("Null fold values detected after merge.")

    if merged.empty:
        raise RuntimeError(
            "Prediction/feature merge produced no rows."
        )

    merged = merged.sort_values(
        [
            "fold",
            "symbol",
            "logged_at",
            "id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    direction_probability_sum = (
        merged[
            "probability_long_squeeze"
        ]
        + merged[
            "probability_short_squeeze"
        ]
    )

    merged[
        "direction_confidence"
    ] = np.where(
        direction_probability_sum > 0,
        np.maximum(
            merged[
                "probability_long_squeeze"
            ],
            merged[
                "probability_short_squeeze"
            ],
        )
        / direction_probability_sum,
        0.5,
    )

    merged[
        "predicted_squeeze_class"
    ] = np.where(
        merged[
            "probability_short_squeeze"
        ]
        >= merged[
            "probability_long_squeeze"
        ],
        SHORT_SQUEEZE_CLASS,
        LONG_SQUEEZE_CLASS,
    ).astype("int8")

    price_groups: dict[
        tuple[str, str],
        dict,
    ] = {}

    for (
        symbol,
        timeframe,
    ), group in features.groupby(
        [
            "symbol",
            "timeframe",
        ],
        sort=False,
        observed=True,
    ):
        group = group.sort_values(
            [
                "logged_at",
                "id",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        price_groups[
            (
                str(symbol),
                str(timeframe),
            )
        ] = {
            "times_ns": (
                group[
                    "logged_at"
                ]
                .to_numpy(
                    dtype="datetime64[ns]"
                )
                .astype(np.int64)
            ),
            "prices": (
                group[
                    "current_price"
                ]
                .to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            ),
        }

    print(
        f"Merged OOS rows : {len(merged):,}"
    )

    print(
        "Symbols         : "
        + ", ".join(
            sorted(
                merged[
                    "symbol"
                ].unique()
            )
        )
    )

    print(
        "Folds           : "
        + ", ".join(
            sorted(
                merged[
                    "fold"
                ].unique()
            )
        )
    )

    return merged, price_groups


def generate_signals_for_config(
    predictions: pd.DataFrame,
    config: SignalConfig,
) -> pd.DataFrame:
    """
    Stateful alarm generation.

    State per fold + symbol:

      armed=True
      score crosses threshold -> signal, armed=False
      score falls below threshold - rearm gap -> armed=True

    Cooldown additionally prevents repeated alerts.
    """

    accepted_indices: list[int] = []

    rearm_threshold = max(
        0.0,
        config.squeeze_threshold
        - REARM_GAP,
    )

    for (
        fold,
        symbol,
    ), group in predictions.groupby(
        [
            "fold",
            "symbol",
        ],
        sort=False,
        observed=True,
    ):
        group = group.sort_values(
            "logged_at",
            kind="mergesort",
        )

        armed = True
        previous_score = 0.0
        last_signal_time = None

        cooldown = pd.Timedelta(
            minutes=(
                config.cooldown_minutes
            )
        )

        for row in group.itertuples():
            score = float(
                row.squeeze_probability
            )

            confidence = float(
                row.direction_confidence
            )

            timestamp = row.logged_at

            if score <= rearm_threshold:
                armed = True

            threshold_crossed = (
                previous_score
                < config.squeeze_threshold
                and score
                >= config.squeeze_threshold
            )

            cooldown_ready = (
                last_signal_time is None
                or (
                    timestamp
                    - last_signal_time
                    >= cooldown
                )
            )

            if (
                armed
                and threshold_crossed
                and cooldown_ready
                and confidence
                >= config.direction_confidence
            ):
                accepted_indices.append(
                    row.Index
                )

                last_signal_time = timestamp
                armed = False

            previous_score = score

    if not accepted_indices:
        return predictions.iloc[
            0:0
        ].copy()

    output = (
        predictions
        .loc[accepted_indices]
        .copy()
        .sort_values(
            "logged_at",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    output[
        "signal_squeeze_threshold"
    ] = config.squeeze_threshold

    output[
        "signal_direction_confidence"
    ] = config.direction_confidence

    output[
        "signal_cooldown_minutes"
    ] = config.cooldown_minutes

    return output


def resolve_target_price(
    signal,
    direction: int,
    entry_price: float,
    target_mode: str,
) -> float | None:
    if target_mode == "nearest_liquidity":
        if direction == TRADE_LONG:
            target = safe_float(
                signal.nearest_upper_price
            )
        else:
            target = safe_float(
                signal.nearest_lower_price
            )

        if (
            target is None
            or target <= 0
        ):
            return None

        if (
            direction == TRADE_LONG
            and target <= entry_price
        ):
            return None

        if (
            direction == TRADE_SHORT
            and target >= entry_price
        ):
            return None

        return target

    if target_mode == "fixed_50bp":
        distance = 50 / 10_000

    elif target_mode == "fixed_100bp":
        distance = 100 / 10_000

    else:
        raise ValueError(
            f"Unknown target mode: {target_mode}"
        )

    return entry_price * (
        1.0
        + direction * distance
    )


def resolve_stop_price(
    signal,
    direction: int,
    entry_price: float,
    stop_mode: str,
) -> float | None:
    if stop_mode == "opposite_liquidity":
        if direction == TRADE_LONG:
            stop = safe_float(
                signal.nearest_lower_price
            )
        else:
            stop = safe_float(
                signal.nearest_upper_price
            )

        if (
            stop is None
            or stop <= 0
        ):
            return None

        if (
            direction == TRADE_LONG
            and stop >= entry_price
        ):
            return None

        if (
            direction == TRADE_SHORT
            and stop <= entry_price
        ):
            return None

        return stop

    if stop_mode == "fixed_50bp":
        distance = 50 / 10_000

    elif stop_mode == "fixed_100bp":
        distance = 100 / 10_000

    else:
        raise ValueError(
            f"Unknown stop mode: {stop_mode}"
        )

    return entry_price * (
        1.0
        - direction * distance
    )


def simulate_trade(
    signal,
    group_data: dict,
    config: ExecutionConfig,
) -> dict | None:
    times_ns = group_data[
        "times_ns"
    ]

    prices = group_data[
        "prices"
    ]

    signal_time_ns = int(
        pd.Timestamp(
            signal.logged_at
        ).to_datetime64()
        .astype(
            "datetime64[ns]"
        )
        .astype(np.int64)
    )

    desired_entry_time_ns = (
        signal_time_ns
        + config.entry_delay_seconds
        * 1_000_000_000
    )

    entry_index = int(
        np.searchsorted(
            times_ns,
            desired_entry_time_ns,
            side="left",
        )
    )

    if entry_index >= len(
        times_ns
    ):
        return None

    entry_price = float(
        prices[entry_index]
    )

    if (
        not math.isfinite(
            entry_price
        )
        or entry_price <= 0
    ):
        return None

    predicted_class = int(
        signal.predicted_squeeze_class
    )

    if (
        predicted_class
        == SHORT_SQUEEZE_CLASS
    ):
        direction = TRADE_LONG
        trade_side = "LONG"
        event_name = "SHORT_SQUEEZE"

    elif (
        predicted_class
        == LONG_SQUEEZE_CLASS
    ):
        direction = TRADE_SHORT
        trade_side = "SHORT"
        event_name = "LONG_SQUEEZE"

    else:
        return None

    target_price = resolve_target_price(
        signal=signal,
        direction=direction,
        entry_price=entry_price,
        target_mode=(
            config.target_mode
        ),
    )

    stop_price = resolve_stop_price(
        signal=signal,
        direction=direction,
        entry_price=entry_price,
        stop_mode=config.stop_mode,
    )

    if (
        target_price is None
        or stop_price is None
    ):
        return None

    target_distance = (
        direction
        * (
            target_price
            / entry_price
            - 1.0
        )
    )

    stop_distance = (
        -direction
        * (
            stop_price
            / entry_price
            - 1.0
        )
    )

    if (
        target_distance <= 0
        or stop_distance <= 0
    ):
        return None

    target_distance_bps = (
        target_distance
        * 10_000
    )

    stop_distance_bps = (
        stop_distance
        * 10_000
    )

    if (
        target_distance_bps
        < config.min_target_distance_bps
    ):
        return None

    window_end_ns = (
        times_ns[entry_index]
        + config.max_hold_minutes
        * 60
        * 1_000_000_000
    )

    exit_limit = int(
        np.searchsorted(
            times_ns,
            window_end_ns,
            side="right",
        )
    )

    exit_limit = min(
        exit_limit,
        len(times_ns),
    )

    if (
        exit_limit
        <= entry_index + 1
    ):
        return None

    exit_index = (
        exit_limit - 1
    )

    exit_reason = "TIME"

    gross_return = (
        direction
        * (
            float(
                prices[exit_index]
            )
            / entry_price
            - 1.0
        )
    )

    mfe = 0.0
    mae = 0.0

    for index in range(
        entry_index + 1,
        exit_limit,
    ):
        price = float(
            prices[index]
        )

        directional_return = (
            direction
            * (
                price
                / entry_price
                - 1.0
            )
        )

        mfe = max(
            mfe,
            directional_return,
        )

        mae = min(
            mae,
            directional_return,
        )

        target_hit = (
            price >= target_price
            if direction == TRADE_LONG
            else price <= target_price
        )

        stop_hit = (
            price <= stop_price
            if direction == TRADE_LONG
            else price >= stop_price
        )

        # Conservative rule:
        # if the same snapshot appears to cross both,
        # count the stop first.
        if stop_hit:
            exit_index = index
            exit_reason = "SL"
            gross_return = (
                -stop_distance
            )
            break

        if target_hit:
            exit_index = index
            exit_reason = "TP"
            gross_return = (
                target_distance
            )
            break

    net_return = (
        gross_return
        - ROUND_TRIP_COST_BPS
        / 10_000
    )

    actual_event = int(
        signal.target_event
    )

    event_detected = (
        actual_event != NO_EVENT_CLASS
    )

    event_direction_correct = (
        event_detected
        and actual_event
        == predicted_class
    )

    entry_time = pd.Timestamp(
        times_ns[entry_index],
        unit="ns",
        tz="UTC",
    )

    exit_time = pd.Timestamp(
        times_ns[exit_index],
        unit="ns",
        tz="UTC",
    )

    return {
        "signal_id": signal.id,
        "fold": signal.fold,
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,

        "signal_time": signal.logged_at,
        "entry_time": entry_time,
        "exit_time": exit_time,

        "predicted_event": event_name,
        "trade_side": trade_side,

        "squeeze_probability": float(
            signal.squeeze_probability
        ),

        "direction_confidence_actual": float(
            signal.direction_confidence
        ),

        "actual_event_class": (
            actual_event
        ),

        "event_detected": bool(
            event_detected
        ),

        "event_direction_correct": bool(
            event_direction_correct
        ),

        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "exit_price": float(
            prices[exit_index]
        ),

        "target_distance_bps": (
            target_distance_bps
        ),

        "stop_distance_bps": (
            stop_distance_bps
        ),

        "exit_reason": exit_reason,

        "gross_return": gross_return,
        "net_return": net_return,

        "gross_bps": (
            gross_return
            * 10_000
        ),

        "net_bps": (
            net_return
            * 10_000
        ),

        "mfe_bps": mfe * 10_000,
        "mae_bps": mae * 10_000,

        "holding_seconds": (
            times_ns[exit_index]
            - times_ns[entry_index]
        )
        / 1_000_000_000,
    }


def simulate_config(
    signals: pd.DataFrame,
    price_groups: dict,
    signal_config: SignalConfig,
    execution_config: ExecutionConfig,
) -> pd.DataFrame:
    rows = []

    last_exit_time: dict[
        str,
        pd.Timestamp,
    ] = {}

    for signal in signals.itertuples():
        symbol = str(
            signal.symbol
        )

        if ONE_POSITION_PER_SYMBOL:
            previous_exit = (
                last_exit_time.get(
                    symbol
                )
            )

            if (
                previous_exit is not None
                and signal.logged_at
                < previous_exit
            ):
                continue

        key = (
            symbol,
            TOPOLOGY_TIMEFRAME,
        )

        group_data = (
            price_groups.get(key)
        )

        if group_data is None:
            continue

        result = simulate_trade(
            signal=signal,
            group_data=group_data,
            config=execution_config,
        )

        if result is None:
            continue

        result.update(
            {
                "squeeze_threshold": (
                    signal_config
                    .squeeze_threshold
                ),

                "minimum_direction_confidence": (
                    signal_config
                    .direction_confidence
                ),

                "cooldown_minutes": (
                    signal_config
                    .cooldown_minutes
                ),

                "entry_delay_seconds": (
                    execution_config
                    .entry_delay_seconds
                ),

                "max_hold_minutes": (
                    execution_config
                    .max_hold_minutes
                ),

                "target_mode": (
                    execution_config
                    .target_mode
                ),

                "stop_mode": (
                    execution_config
                    .stop_mode
                ),

                "minimum_target_distance_bps": (
                    execution_config
                    .min_target_distance_bps
                ),

                "round_trip_cost_bps": (
                    ROUND_TRIP_COST_BPS
                ),
            }
        )

        rows.append(result)

        last_exit_time[symbol] = (
            result["exit_time"]
        )

    return pd.DataFrame(rows)


def summarize_trades(
    trades: pd.DataFrame,
) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": None,
            "mean_net_bps": None,
            "median_net_bps": None,
            "total_net_bps": None,
            "profit_factor": None,
            "compounded_return": None,
            "max_drawdown": None,
            "mean_mfe_bps": None,
            "mean_mae_bps": None,
            "mean_holding_minutes": None,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "event_rate": None,
            "event_direction_accuracy": None,
            "trades_per_day": None,
        }

    trades = trades.sort_values(
        [
            "entry_time",
            "symbol",
        ],
        kind="mergesort",
    )

    returns = trades[
        "net_return"
    ].to_numpy(
        dtype=np.float64
    )

    equity = compounded_equity(
        returns
    )

    date_span_days = max(
        1.0,
        (
            trades[
                "entry_time"
            ].max()
            - trades[
                "entry_time"
            ].min()
        ).total_seconds()
        / 86_400,
    )

    event_rows = trades.loc[
        trades[
            "event_detected"
        ]
    ]

    if event_rows.empty:
        direction_accuracy = None
    else:
        direction_accuracy = float(
            event_rows[
                "event_direction_correct"
            ].mean()
        )

    return {
        "trades": int(
            len(trades)
        ),

        "wins": int(
            (returns > 0).sum()
        ),

        "losses": int(
            (returns < 0).sum()
        ),

        "timeouts": int(
            trades[
                "exit_reason"
            ].eq("TIME").sum()
        ),

        "win_rate": float(
            (returns > 0).mean()
        ),

        "mean_net_bps": float(
            returns.mean()
            * 10_000
        ),

        "median_net_bps": float(
            np.median(returns)
            * 10_000
        ),

        "total_net_bps": float(
            returns.sum()
            * 10_000
        ),

        "profit_factor": (
            profit_factor(
                returns
            )
        ),

        "compounded_return": float(
            equity[-1] - 1.0
        )
        if len(equity)
        else None,

        "max_drawdown": (
            max_drawdown(
                equity
            )
        ),

        "mean_mfe_bps": float(
            trades[
                "mfe_bps"
            ].mean()
        ),

        "mean_mae_bps": float(
            trades[
                "mae_bps"
            ].mean()
        ),

        "mean_holding_minutes": float(
            trades[
                "holding_seconds"
            ].mean()
            / 60
        ),

        "target_hit_rate": float(
            trades[
                "exit_reason"
            ].eq("TP").mean()
        ),

        "stop_hit_rate": float(
            trades[
                "exit_reason"
            ].eq("SL").mean()
        ),

        "event_rate": float(
            trades[
                "event_detected"
            ].mean()
        ),

        "event_direction_accuracy": (
            direction_accuracy
        ),

        "trades_per_day": float(
            len(trades)
            / date_span_days
        ),
    }


def fold_consistency(
    trades: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    rows = []

    # Some grid configurations legitimately produce zero trades.
    # An empty DataFrame created from an empty row list has no columns,
    # so return an empty fold report before attempting groupby.
    if trades.empty or "fold" not in trades.columns:
        empty_fold_frame = pd.DataFrame(
            columns=[
                "fold",
                "trades",
                "wins",
                "losses",
                "timeouts",
                "win_rate",
                "mean_net_bps",
                "median_net_bps",
                "total_net_bps",
                "profit_factor",
                "compounded_return",
                "max_drawdown",
                "mean_mfe_bps",
                "mean_mae_bps",
                "mean_holding_minutes",
                "target_hit_rate",
                "stop_hit_rate",
                "event_rate",
                "event_direction_accuracy",
                "trades_per_day",
            ]
        )

        return empty_fold_frame, {
            "positive_folds": 0,
            "fold_count": 0,
            "minimum_fold_mean_bps": None,
            "minimum_fold_profit_factor": None,
        }

    for fold, group in trades.groupby(
        "fold",
        observed=True,
        sort=True,
    ):
        metrics = summarize_trades(
            group
        )

        rows.append(
            {
                "fold": fold,
                **metrics,
            }
        )

    fold_frame = pd.DataFrame(
        rows
    )

    if fold_frame.empty:
        return fold_frame, {
            "positive_folds": 0,
            "fold_count": 0,
            "minimum_fold_mean_bps": None,
            "minimum_fold_profit_factor": None,
        }

    positive_folds = int(
        (
            fold_frame[
                "mean_net_bps"
            ] > 0
        ).sum()
    )

    profit_factors = pd.to_numeric(
        fold_frame[
            "profit_factor"
        ],
        errors="coerce",
    ).dropna()

    return fold_frame, {
        "positive_folds": (
            positive_folds
        ),

        "fold_count": int(
            len(fold_frame)
        ),

        "minimum_fold_mean_bps": float(
            fold_frame[
                "mean_net_bps"
            ].min()
        ),

        "minimum_fold_profit_factor": (
            float(
                profit_factors.min()
            )
            if not profit_factors.empty
            else None
        ),
    }


def calculate_research_score(
    metrics: dict,
    consistency: dict,
) -> float:
    trades = metrics.get(
        "trades",
        0,
    )

    if trades < MIN_TRADES_FOR_RANKING:
        return -1_000_000.0

    mean_net = metrics.get(
        "mean_net_bps"
    )

    pf = metrics.get(
        "profit_factor"
    )

    win_rate = metrics.get(
        "win_rate"
    )

    event_rate = metrics.get(
        "event_rate"
    )

    direction_accuracy = metrics.get(
        "event_direction_accuracy"
    )

    max_dd = metrics.get(
        "max_drawdown"
    )

    positive_folds = consistency.get(
        "positive_folds",
        0,
    )

    fold_count = max(
        1,
        consistency.get(
            "fold_count",
            1,
        ),
    )

    minimum_fold_mean = (
        consistency.get(
            "minimum_fold_mean_bps"
        )
    )

    values = [
        mean_net,
        pf,
        win_rate,
        event_rate,
        max_dd,
    ]

    if any(
        value is None
        or not math.isfinite(
            float(value)
        )
        for value in values
    ):
        return -1_000_000.0

    direction_accuracy = (
        float(direction_accuracy)
        if direction_accuracy is not None
        else 0.0
    )

    minimum_fold_mean = (
        float(minimum_fold_mean)
        if minimum_fold_mean is not None
        else -100.0
    )

    fold_ratio = (
        positive_folds
        / fold_count
    )

    drawdown_penalty = abs(
        float(max_dd)
    ) * 100

    return float(
        float(mean_net)
        * 0.30

        + min(
            float(pf),
            5.0,
        )
        * 5.0

        + float(win_rate)
        * 10.0

        + float(event_rate)
        * 5.0

        + direction_accuracy
        * 5.0

        + fold_ratio
        * 10.0

        + minimum_fold_mean
        * 0.10

        - drawdown_penalty
        * 0.50
    )


def build_symbol_metrics(
    trades: pd.DataFrame,
    config_id: int,
) -> list[dict]:
    rows = []

    for symbol, group in trades.groupby(
        "symbol",
        observed=True,
        sort=True,
    ):
        rows.append(
            {
                "config_id": config_id,
                "symbol": symbol,
                **summarize_trades(
                    group
                ),
            }
        )

    return rows


def main() -> int:
    started = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "TOPOLOGY V2 — COMPREHENSIVE "
        "SQUEEZE SIGNAL BACKTEST"
    )
    print("=" * 100)

    print(f"Features    : {FEATURE_PATH}")
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(f"Output      : {OUTPUT_DIR}")
    print()

    print("Execution assumptions:")
    print(
        f"  Round-trip costs : "
        f"{ROUND_TRIP_COST_BPS:.1f} bps"
    )
    print(
        f"  One position/symbol: "
        f"{ONE_POSITION_PER_SYMBOL}"
    )
    print(
        f"  Notional fraction: "
        f"{NOTIONAL_FRACTION_PER_TRADE:.0%}"
    )
    print()

    predictions, price_groups = (
        load_data()
    )

    signal_configs = [
        SignalConfig(
            squeeze_threshold=threshold,
            direction_confidence=confidence,
            cooldown_minutes=cooldown,
        )
        for (
            threshold,
            confidence,
            cooldown,
        ) in itertools.product(
            SQUEEZE_THRESHOLDS,
            DIRECTION_CONFIDENCE_VALUES,
            COOLDOWN_MINUTES_VALUES,
        )
    ]

    execution_configs = [
        ExecutionConfig(
            entry_delay_seconds=delay,
            max_hold_minutes=hold,
            target_mode=target_mode,
            stop_mode=stop_mode,
            min_target_distance_bps=min_target,
        )
        for (
            delay,
            hold,
            target_mode,
            stop_mode,
            min_target,
        ) in itertools.product(
            ENTRY_DELAY_SECONDS_VALUES,
            MAX_HOLD_MINUTES_VALUES,
            TARGET_MODES,
            STOP_MODES,
            MIN_TARGET_DISTANCE_BPS_VALUES,
        )
    ]

    total_configs = (
        len(signal_configs)
        * len(execution_configs)
    )

    print(
        f"Signal configurations   : "
        f"{len(signal_configs):,}"
    )

    print(
        f"Execution configurations: "
        f"{len(execution_configs):,}"
    )

    print(
        f"Total backtests         : "
        f"{total_configs:,}"
    )

    print()

    summary_rows = []
    fold_rows = []
    symbol_rows = []
    all_trade_frames = []
    signal_frames = []

    config_id = 0
    completed = 0

    for signal_index, signal_config in enumerate(
        signal_configs,
        start=1,
    ):
        signals = generate_signals_for_config(
            predictions=predictions,
            config=signal_config,
        )

        if not signals.empty:
            signal_frames.append(
                signals
            )

        print()
        print("-" * 100)
        print(
            f"SIGNAL CONFIG "
            f"{signal_index}/"
            f"{len(signal_configs)}"
        )
        print(
            f"threshold="
            f"{signal_config.squeeze_threshold:.2f} "
            f"direction_conf="
            f"{signal_config.direction_confidence:.2f} "
            f"cooldown="
            f"{signal_config.cooldown_minutes}m "
            f"signals={len(signals):,}"
        )
        print("-" * 100)

        for execution_config in (
            execution_configs
        ):
            config_id += 1
            completed += 1

            trades = simulate_config(
                signals=signals,
                price_groups=price_groups,
                signal_config=signal_config,
                execution_config=(
                    execution_config
                ),
            )

            metrics = summarize_trades(
                trades
            )

            (
                fold_frame,
                consistency,
            ) = fold_consistency(
                trades
            )

            score = (
                calculate_research_score(
                    metrics,
                    consistency,
                )
            )

            summary_row = {
                "config_id": config_id,

                "squeeze_threshold": (
                    signal_config
                    .squeeze_threshold
                ),

                "minimum_direction_confidence": (
                    signal_config
                    .direction_confidence
                ),

                "cooldown_minutes": (
                    signal_config
                    .cooldown_minutes
                ),

                "entry_delay_seconds": (
                    execution_config
                    .entry_delay_seconds
                ),

                "max_hold_minutes": (
                    execution_config
                    .max_hold_minutes
                ),

                "target_mode": (
                    execution_config
                    .target_mode
                ),

                "stop_mode": (
                    execution_config
                    .stop_mode
                ),

                "minimum_target_distance_bps": (
                    execution_config
                    .min_target_distance_bps
                ),

                "raw_signals": int(
                    len(signals)
                ),

                **metrics,
                **consistency,

                "research_score": score,
            }

            summary_rows.append(
                summary_row
            )

            if not fold_frame.empty:
                for row in (
                    fold_frame
                    .to_dict(
                        orient="records"
                    )
                ):
                    fold_rows.append(
                        {
                            "config_id": (
                                config_id
                            ),
                            **row,
                        }
                    )

            if not trades.empty:
                trades[
                    "config_id"
                ] = config_id

                all_trade_frames.append(
                    trades
                )

                symbol_rows.extend(
                    build_symbol_metrics(
                        trades,
                        config_id,
                    )
                )

            if (
                completed % 100 == 0
                or completed
                == total_configs
            ):
                elapsed = (
                    time.time()
                    - started
                )

                average = (
                    elapsed
                    / completed
                )

                remaining = (
                    total_configs
                    - completed
                )

                eta_minutes = (
                    average
                    * remaining
                    / 60
                )

                print(
                    f"Progress "
                    f"{completed:,}/"
                    f"{total_configs:,} "
                    f"({completed / total_configs:.1%}) "
                    f"| ETA "
                    f"{eta_minutes:.1f} min",
                    flush=True,
                )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        [
            "research_score",
            "positive_folds",
            "mean_net_bps",
            "profit_factor",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    eligible = summary.loc[
        summary["trades"]
        >= MIN_TRADES_FOR_RANKING
    ].copy()

    top_configs = (
        eligible.head(100)
    )

    top_configs.to_csv(
        TOP_CONFIGS_PATH,
        index=False,
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    symbol_metrics = pd.DataFrame(
        symbol_rows
    )

    symbol_metrics.to_csv(
        SYMBOL_METRICS_PATH,
        index=False,
    )

    if signal_frames:
        all_signals = (
            pd.concat(
                signal_frames,
                ignore_index=True,
            )
            .drop_duplicates(
                subset=[
                    "id",
                    "signal_squeeze_threshold",
                    "signal_direction_confidence",
                    "signal_cooldown_minutes",
                ]
            )
        )

        all_signals.to_parquet(
            SIGNALS_PATH,
            index=False,
            compression="zstd",
        )

    if all_trade_frames:
        all_trades = pd.concat(
            all_trade_frames,
            ignore_index=True,
        )

        all_trades.to_parquet(
            TRADES_PATH,
            index=False,
            compression="zstd",
        )
    else:
        all_trades = pd.DataFrame()

    if top_configs.empty:
        print()
        print(
            "No configuration met the "
            "minimum trade requirement."
        )

        best = None

    else:
        best = top_configs.iloc[
            0
        ].to_dict()

        best_config_id = int(
            best["config_id"]
        )

        best_trades = all_trades.loc[
            all_trades[
                "config_id"
            ].eq(
                best_config_id
            )
        ].copy()

        best_trades.to_parquet(
            BEST_TRADES_PATH,
            index=False,
            compression="zstd",
        )

    report = {
        "experiment": (
            "tf_1h__future_60m__"
            "precursor_5m__q_0p975"
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

        "grid": {
            "signal_configurations": (
                len(signal_configs)
            ),
            "execution_configurations": (
                len(execution_configs)
            ),
            "total_backtests": (
                total_configs
            ),
        },

        "best_configuration": best,

        "top_20": (
            top_configs
            .head(20)
            .to_dict(
                orient="records"
            )
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 140)
    print(
        "TOP SQUEEZE SIGNAL BACKTESTS"
    )
    print("=" * 140)

    display_columns = [
        "config_id",
        "squeeze_threshold",
        "minimum_direction_confidence",
        "cooldown_minutes",
        "entry_delay_seconds",
        "max_hold_minutes",
        "target_mode",
        "stop_mode",
        "minimum_target_distance_bps",
        "trades",
        "trades_per_day",
        "win_rate",
        "mean_net_bps",
        "median_net_bps",
        "profit_factor",
        "compounded_return",
        "max_drawdown",
        "event_rate",
        "event_direction_accuracy",
        "positive_folds",
        "minimum_fold_mean_bps",
        "research_score",
    ]

    if not top_configs.empty:
        print(
            top_configs[
                display_columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

        print()
        print("=" * 140)
        print("BEST CONFIGURATION")
        print("=" * 140)

        for key in display_columns:
            print(
                f"{key:36}: "
                f"{best.get(key)}"
            )

        print()
        print(
            "Best configuration by fold:"
        )

        print(
            fold_metrics.loc[
                fold_metrics[
                    "config_id"
                ].eq(
                    int(
                        best[
                            "config_id"
                        ]
                    )
                )
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "Best configuration by symbol:"
        )

        print(
            symbol_metrics.loc[
                symbol_metrics[
                    "config_id"
                ].eq(
                    int(
                        best[
                            "config_id"
                        ]
                    )
                )
            ].to_string(
                index=False
            )
        )

    elapsed = (
        time.time()
        - started
    )

    print()
    print("=" * 100)
    print(
        "COMPREHENSIVE BACKTEST COMPLETE"
    )
    print("=" * 100)
    print(f"Summary      : {SUMMARY_PATH}")
    print(f"Top configs  : {TOP_CONFIGS_PATH}")
    print(f"Fold metrics : {FOLD_METRICS_PATH}")
    print(f"Symbols      : {SYMBOL_METRICS_PATH}")
    print(f"Signals      : {SIGNALS_PATH}")
    print(f"All trades   : {TRADES_PATH}")
    print(f"Best trades  : {BEST_TRADES_PATH}")
    print(f"Report       : {REPORT_PATH}")
    print(
        f"Elapsed      : "
        f"{elapsed / 60:.1f} minutes"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print()
        print("=" * 100)
        print("BACKTEST FAILED")
        print("=" * 100)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        raise
