from __future__ import annotations

from pathlib import Path
import json
import math
import sys
import time

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool


FEATURE_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

STRONG_LABEL_PATH = Path(
    "data/features/liq_topology_v2_strong_contrarian_labels.parquet"
)

SWEEP_LABEL_PATH = Path(
    "data/features/liq_topology_v2_sweep_labels.parquet"
)

MODEL_DIR = Path(
    "data/models/topology_v2_strong_contrarian_25bp_walk_forward"
)

OUTPUT_DIR = Path(
    "data/backtests/topology_v2_strong_contrarian_25bp"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
TRADES_PATH = OUTPUT_DIR / "trades.parquet"
FOLD_PATH = OUTPUT_DIR / "fold_metrics.csv"
REPORT_PATH = OUTPUT_DIR / "report.json"

TARGET_COLUMN = "strong_contrarian_25bp_1h"

EMBARGO = pd.Timedelta(hours=4)

# Strategy assumptions.
TP_BPS = 25.0
STOP_BPS_VALUES = [10.0, 15.0, 25.0]

POST_ENTRY_WINDOW = pd.Timedelta(minutes=15)
SIGNAL_COOLDOWN = pd.Timedelta(minutes=15)

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

# Validation'da en yüksek olasılıklı bu dilimler için eşik öğrenilir.
SIGNAL_FRACTIONS = [0.01, 0.05, 0.10]

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "symbol",
    "timeframe",
    "nearest_side",
]

NUMERIC_FEATURES = [
    "nearest_side_code",

    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",

    "log1p_upper_distance_pct",
    "log1p_lower_distance_pct",
    "log1p_distance_advantage",

    "upper_pool_volume",
    "lower_pool_volume",
    "nearest_pool_volume",
    "farther_pool_volume",

    "upper_total_volume",
    "lower_total_volume",

    "upper_active_levels",
    "lower_active_levels",

    "log1p_upper_pool_volume",
    "log1p_lower_pool_volume",
    "log1p_nearest_pool_volume",
    "log1p_farther_pool_volume",

    "log1p_upper_total_volume",
    "log1p_lower_total_volume",

    "log1p_upper_active_levels",
    "log1p_lower_active_levels",

    "pool_volume_ratio",
    "log1p_pool_volume_ratio",

    "distance_pressure_ratio",
    "log1p_distance_pressure_ratio",

    "topology_imbalance",
    "total_volume_imbalance_check",

    "active_level_difference",
    "active_level_total",

    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_utc",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

PRICE_COLUMNS = [
    "id",
    "logged_at",
    "symbol",
    "timeframe",
    "current_price",
    "nearest_side",
    "nearest_upper_price",
    "nearest_lower_price",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_FEATURES:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("<MISSING>")
            .astype(str)
        )

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


def make_folds(df: pd.DataFrame) -> list[dict]:
    q40 = df["logged_at"].quantile(0.40)
    q55 = df["logged_at"].quantile(0.55)
    q70 = df["logged_at"].quantile(0.70)
    q85 = df["logged_at"].quantile(0.85)

    return [
        {
            "name": "fold_1",
            "train_end": q40 - EMBARGO,
            "validation_start": q40 + EMBARGO,
            "validation_end": q55 - EMBARGO,
            "test_start": q55 + EMBARGO,
            "test_end": q70 - EMBARGO,
        },
        {
            "name": "fold_2",
            "train_end": q55 - EMBARGO,
            "validation_start": q55 + EMBARGO,
            "validation_end": q70 - EMBARGO,
            "test_start": q70 + EMBARGO,
            "test_end": q85 - EMBARGO,
        },
        {
            "name": "fold_3",
            "train_end": q70 - EMBARGO,
            "validation_start": q70 + EMBARGO,
            "validation_end": q85 - EMBARGO,
            "test_start": q85 + EMBARGO,
            "test_end": df["logged_at"].max(),
        },
    ]


def validation_threshold(
    probabilities: np.ndarray,
    fraction: float,
) -> float:
    return float(
        np.quantile(
            probabilities,
            1.0 - fraction,
        )
    )


def apply_symbol_cooldown(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aynı sembolde birbirine çok yakın, neredeyse aynı snapshot
    sinyallerinden yalnızca birini korur.

    Önce zaman, aynı zamanda ise yüksek olasılık seçilir.
    """
    candidates = candidates.sort_values(
        ["logged_at", "probability"],
        ascending=[True, False],
        kind="mergesort",
    )

    accepted_indices = []
    last_signal_time: dict[str, pd.Timestamp] = {}

    for row in candidates.itertuples():
        symbol = str(row.symbol)
        timestamp = row.logged_at

        previous = last_signal_time.get(symbol)

        if (
            previous is None
            or timestamp - previous >= SIGNAL_COOLDOWN
        ):
            accepted_indices.append(row.Index)
            last_signal_time[symbol] = timestamp

    return (
        candidates.loc[accepted_indices]
        .sort_values("logged_at")
        .reset_index(drop=True)
    )


def max_drawdown(compounded_curve: np.ndarray) -> float:
    if len(compounded_curve) == 0:
        return 0.0

    running_max = np.maximum.accumulate(compounded_curve)

    drawdowns = (
        compounded_curve / running_max
        - 1.0
    )

    return float(drawdowns.min())


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
            "compounded_return": None,
            "profit_factor": None,
            "max_drawdown": None,
            "mean_holding_seconds": None,
        }

    returns = trades["net_return"].to_numpy(
        dtype=np.float64
    )

    positive = returns[returns > 0]
    negative = returns[returns < 0]

    gross_profit = positive.sum()
    gross_loss = -negative.sum()

    compounded_curve = np.cumprod(
        1.0 + returns
    )

    return {
        "trades": int(len(trades)),
        "wins": int((returns > 0).sum()),
        "losses": int((returns < 0).sum()),
        "timeouts": int(
            trades["exit_reason"].eq("TIME").sum()
        ),
        "win_rate": float(
            (returns > 0).mean()
        ),
        "mean_net_bps": float(
            returns.mean() * 10_000
        ),
        "median_net_bps": float(
            np.median(returns) * 10_000
        ),
        "total_net_bps": float(
            returns.sum() * 10_000
        ),
        "compounded_return": float(
            compounded_curve[-1] - 1.0
        ),
        "profit_factor": (
            float(gross_profit / gross_loss)
            if gross_loss > 0
            else None
        ),
        "max_drawdown": max_drawdown(
            compounded_curve
        ),
        "mean_holding_seconds": float(
            trades["holding_seconds"].mean()
        ),
    }


def simulate_candidate(
    candidate,
    group_times_ns: np.ndarray,
    group_prices: np.ndarray,
    local_position: int,
    stop_bps: float,
) -> dict | None:
    """
    Strateji:
      nearest_side=LOWER -> uzak taraf UPPER -> long breakout
      nearest_side=UPPER -> uzak taraf LOWER -> short breakdown

    Yakın taraf önce alınırsa contrarian setup iptal edilir.
    Yalnız uzak taraf ilk sweep ise trade açılır.
    """

    nearest_side = str(candidate.nearest_side)
    sweep_code = int(candidate.sweep_code_1h)

    if nearest_side == "LOWER":
        direction = 1
        required_sweep_code = 1
        trigger_level = float(
            candidate.nearest_upper_price
        )
    elif nearest_side == "UPPER":
        direction = -1
        required_sweep_code = -1
        trigger_level = float(
            candidate.nearest_lower_price
        )
    else:
        return None

    # Uzak taraf ilk alınmadıysa setup geçersiz; emir iptal.
    if sweep_code != required_sweep_code:
        return None

    hit_seconds = float(
        candidate.first_hit_seconds_1h
    )

    if not math.isfinite(hit_seconds):
        return None

    signal_time_ns = int(
        candidate.logged_at.to_datetime64()
        .astype("datetime64[ns]")
        .astype(np.int64)
    )

    expected_hit_time_ns = (
        signal_time_ns
        + int(round(hit_seconds * 1_000_000_000))
    )

    entry_index = int(
        np.searchsorted(
            group_times_ns,
            expected_hit_time_ns,
            side="left",
        )
    )

    if (
        entry_index >= len(group_times_ns)
        or entry_index <= local_position
    ):
        return None

    # Giriş, seviyeye teorik dokunuş fiyatından değil,
    # elimizde gözlenen ilk snapshot fiyatından yapılır.
    entry_price = float(
        group_prices[entry_index]
    )

    if not math.isfinite(entry_price) or entry_price <= 0:
        return None

    window_end_ns = (
        group_times_ns[entry_index]
        + int(POST_ENTRY_WINDOW.value)
    )

    exit_limit = int(
        np.searchsorted(
            group_times_ns,
            window_end_ns,
            side="right",
        )
    )

    exit_limit = min(
        exit_limit,
        len(group_times_ns),
    )

    if exit_limit <= entry_index:
        return None

    tp_return = TP_BPS / 10_000
    stop_return = stop_bps / 10_000

    exit_index = exit_limit - 1
    exit_reason = "TIME"
    gross_return = (
        direction
        * (
            float(group_prices[exit_index])
            / entry_price
            - 1.0
        )
    )

    # Snapshot dizisinde ilk gözlenen TP veya SL.
    for index in range(
        entry_index + 1,
        exit_limit,
    ):
        price = float(group_prices[index])

        directional_return = (
            direction
            * (
                price / entry_price
                - 1.0
            )
        )

        if directional_return >= tp_return:
            exit_index = index
            exit_reason = "TP"
            gross_return = tp_return
            break

        if directional_return <= -stop_return:
            exit_index = index
            exit_reason = "SL"
            gross_return = -stop_return
            break

    net_return = (
        gross_return
        - ROUND_TRIP_COST_BPS / 10_000
    )

    return {
        "entry_time": pd.Timestamp(
            group_times_ns[entry_index],
            unit="ns",
            tz="UTC",
        ),
        "exit_time": pd.Timestamp(
            group_times_ns[exit_index],
            unit="ns",
            tz="UTC",
        ),
        "direction": (
            "LONG"
            if direction == 1
            else "SHORT"
        ),
        "trigger_level": trigger_level,
        "entry_price": entry_price,
        "exit_price": float(
            group_prices[exit_index]
        ),
        "exit_reason": exit_reason,
        "gross_return": gross_return,
        "net_return": net_return,
        "gross_bps": gross_return * 10_000,
        "net_bps": net_return * 10_000,
        "holding_seconds": (
            group_times_ns[exit_index]
            - group_times_ns[entry_index]
        ) / 1_000_000_000,
    }


def main() -> int:
    started = time.time()

    required_paths = [
        FEATURE_PATH,
        STRONG_LABEL_PATH,
        SWEEP_LABEL_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            print(
                f"ERROR: Missing file: {path}",
                file=sys.stderr,
            )
            return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 84)
    print(
        "TOPOLOGY V2 — STRONG CONTRARIAN "
        "25BP ECONOMIC BACKTEST"
    )
    print("=" * 84)

    print("Assumptions:")
    print(f"  Take profit          : {TP_BPS:.1f} bps")
    print(
        f"  Stop scenarios       : "
        f"{STOP_BPS_VALUES}"
    )
    print(
        f"  Post-entry window    : "
        f"{POST_ENTRY_WINDOW}"
    )
    print(
        f"  Signal cooldown      : "
        f"{SIGNAL_COOLDOWN}"
    )
    print(
        f"  Round-trip costs     : "
        f"{ROUND_TRIP_COST_BPS:.1f} bps"
    )
    print()

    selected_feature_columns = list(
        dict.fromkeys(
            PRICE_COLUMNS + FEATURE_COLUMNS
        )
    )

    print("Loading feature and price data...")

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=selected_feature_columns,
    )

    print("Loading strong labels...")

    strong_labels = pd.read_parquet(
        STRONG_LABEL_PATH,
        columns=[
            "id",
            TARGET_COLUMN,
        ],
    )

    print("Loading sweep execution labels...")

    sweep_labels = pd.read_parquet(
        SWEEP_LABEL_PATH,
        columns=[
            "id",
            "sweep_code_1h",
            "first_hit_seconds_1h",
        ],
    )

    df = (
        features
        .merge(
            strong_labels,
            on="id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            sweep_labels,
            on="id",
            how="inner",
            validate="one_to_one",
        )
    )

    df = df.loc[
        df[TARGET_COLUMN].isin([0, 1])
    ].copy()

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN].astype("int8")
    )

    df = df.sort_values(
        [
            "symbol",
            "timeframe",
            "logged_at",
            "id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # Her row'un grup içindeki local pozisyonu.
    df["local_position"] = (
        df.groupby(
            ["symbol", "timeframe"],
            observed=True,
        )
        .cumcount()
        .astype("int32")
    )

    print(
        f"Usable rows: {len(df):,}; "
        f"base event rate: "
        f"{df[TARGET_COLUMN].mean():.4f}"
    )

    # Grup fiyat dizilerini yalnız bir kez hazırla.
    price_groups = {}

    for key, group in df.groupby(
        ["symbol", "timeframe"],
        sort=False,
        observed=True,
    ):
        price_groups[key] = {
            "times_ns": (
                group["logged_at"]
                .to_numpy(dtype="datetime64[ns]")
                .astype(np.int64)
            ),
            "prices": (
                group["current_price"]
                .to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            ),
        }

    folds = make_folds(df)

    all_trade_frames = []
    fold_metric_rows = []
    summary_rows = []
    report_folds = []

    for fold_index, fold in enumerate(
        folds,
        start=1,
    ):
        fold_name = fold["name"]
        model_path = (
            MODEL_DIR / f"{fold_name}.cbm"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        print()
        print("=" * 84)
        print(f"BACKTEST {fold_name.upper()}")
        print("=" * 84)

        validation_df = df.loc[
            (
                df["logged_at"]
                >= fold["validation_start"]
            )
            & (
                df["logged_at"]
                <= fold["validation_end"]
            )
        ].copy()

        test_df = df.loc[
            (
                df["logged_at"]
                >= fold["test_start"]
            )
            & (
                df["logged_at"]
                <= fold["test_end"]
            )
        ].copy()

        model = CatBoostClassifier()
        model.load_model(str(model_path))

        validation_pool = Pool(
            prepare_features(validation_df),
            cat_features=[
                FEATURE_COLUMNS.index(column)
                for column in CATEGORICAL_FEATURES
            ],
            feature_names=FEATURE_COLUMNS,
        )

        test_pool = Pool(
            prepare_features(test_df),
            cat_features=[
                FEATURE_COLUMNS.index(column)
                for column in CATEGORICAL_FEATURES
            ],
            feature_names=FEATURE_COLUMNS,
        )

        validation_df["probability"] = (
            model.predict_proba(
                validation_pool
            )[:, 1]
        )

        test_df["probability"] = (
            model.predict_proba(
                test_pool
            )[:, 1]
        )

        fold_report = {
            "fold": fold_name,
            "validation_rows": int(
                len(validation_df)
            ),
            "test_rows": int(len(test_df)),
            "test_base_rate": float(
                test_df[TARGET_COLUMN].mean()
            ),
            "signal_sets": [],
        }

        for fraction in SIGNAL_FRACTIONS:
            threshold = validation_threshold(
                validation_df[
                    "probability"
                ].to_numpy(),
                fraction,
            )

            raw_candidates = test_df.loc[
                test_df["probability"]
                >= threshold
            ].copy()

            candidates = apply_symbol_cooldown(
                raw_candidates
            )

            print()
            print(
                f"{fold_name} validation top "
                f"{fraction * 100:.0f}% threshold="
                f"{threshold:.6f}"
            )
            print(
                f"  Raw test signals : "
                f"{len(raw_candidates):,}"
            )
            print(
                f"  After cooldown   : "
                f"{len(candidates):,}"
            )

            for stop_bps in STOP_BPS_VALUES:
                trade_rows = []

                for candidate in candidates.itertuples():
                    key = (
                        candidate.symbol,
                        candidate.timeframe,
                    )

                    group_data = price_groups[key]

                    result = simulate_candidate(
                        candidate=candidate,
                        group_times_ns=(
                            group_data["times_ns"]
                        ),
                        group_prices=(
                            group_data["prices"]
                        ),
                        local_position=int(
                            candidate.local_position
                        ),
                        stop_bps=stop_bps,
                    )

                    if result is None:
                        continue

                    trade_rows.append(
                        {
                            "fold": fold_name,
                            "signal_fraction": fraction,
                            "probability_threshold": (
                                threshold
                            ),
                            "stop_bps": stop_bps,
                            "signal_id": candidate.id,
                            "signal_time": (
                                candidate.logged_at
                            ),
                            "symbol": candidate.symbol,
                            "timeframe": (
                                candidate.timeframe
                            ),
                            "nearest_side": (
                                candidate.nearest_side
                            ),
                            "probability": (
                                candidate.probability
                            ),
                            "target_event": int(
                                getattr(
                                    candidate,
                                    TARGET_COLUMN,
                                )
                            ),
                            **result,
                        }
                    )

                trades = pd.DataFrame(
                    trade_rows
                )

                metrics = summarize_trades(
                    trades
                )

                signal_count = len(candidates)

                metrics.update(
                    {
                        "fold": fold_name,
                        "signal_fraction": fraction,
                        "probability_threshold": (
                            threshold
                        ),
                        "stop_bps": stop_bps,
                        "signals": signal_count,
                        "raw_signals": int(
                            len(raw_candidates)
                        ),
                        "trigger_rate": (
                            len(trades) / signal_count
                            if signal_count
                            else None
                        ),
                        "signal_event_rate": (
                            float(
                                candidates[
                                    TARGET_COLUMN
                                ].mean()
                            )
                            if signal_count
                            else None
                        ),
                        "round_trip_cost_bps": (
                            ROUND_TRIP_COST_BPS
                        ),
                    }
                )

                fold_metric_rows.append(
                    metrics
                )

                print(
                    f"  stop={stop_bps:>4.0f}bp "
                    f"trades={metrics['trades']:>5} "
                    f"trigger="
                    f"{(metrics['trigger_rate'] or 0):.2%} "
                    f"win="
                    f"{(metrics['win_rate'] or 0):.2%} "
                    f"mean_net="
                    f"{(metrics['mean_net_bps'] or 0):+.2f}bp "
                    f"PF="
                    f"{metrics['profit_factor']}"
                )

                if not trades.empty:
                    all_trade_frames.append(
                        trades
                    )

                fold_report[
                    "signal_sets"
                ].append(metrics)

        report_folds.append(fold_report)

    metrics_df = pd.DataFrame(
        fold_metric_rows
    )

    metrics_df.to_csv(
        FOLD_PATH,
        index=False,
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

    # Aggregate aynı fraction + stop ayarını fold'lar boyunca birleştir.
    for (
        fraction,
        stop_bps,
    ), group_metrics in metrics_df.groupby(
        ["signal_fraction", "stop_bps"],
        observed=True,
    ):
        if all_trades.empty:
            combined_trades = pd.DataFrame()
        else:
            combined_trades = all_trades.loc[
                (
                    all_trades[
                        "signal_fraction"
                    ].eq(fraction)
                )
                & (
                    all_trades[
                        "stop_bps"
                    ].eq(stop_bps)
                )
            ].sort_values(
                "entry_time"
            )

        aggregate_metrics = summarize_trades(
            combined_trades
        )

        aggregate_metrics.update(
            {
                "signal_fraction": fraction,
                "stop_bps": stop_bps,
                "signals": int(
                    group_metrics[
                        "signals"
                    ].sum()
                ),
                "raw_signals": int(
                    group_metrics[
                        "raw_signals"
                    ].sum()
                ),
                "mean_trigger_rate": float(
                    group_metrics[
                        "trigger_rate"
                    ].mean()
                ),
                "mean_signal_event_rate": float(
                    group_metrics[
                        "signal_event_rate"
                    ].mean()
                ),
                "round_trip_cost_bps": (
                    ROUND_TRIP_COST_BPS
                ),
            }
        )

        summary_rows.append(
            aggregate_metrics
        )

    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "mean_net_bps",
            "profit_factor",
        ],
        ascending=[False, False],
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    report = {
        "strategy": (
            "Strong contrarian farther-side breakout, "
            "25bp take-profit"
        ),
        "assumptions": {
            "take_profit_bps": TP_BPS,
            "stop_bps_values": STOP_BPS_VALUES,
            "post_entry_minutes": (
                POST_ENTRY_WINDOW
                .total_seconds()
                / 60
            ),
            "cooldown_minutes": (
                SIGNAL_COOLDOWN
                .total_seconds()
                / 60
            ),
            "entry_fee_bps": ENTRY_FEE_BPS,
            "exit_fee_bps": EXIT_FEE_BPS,
            "entry_slippage_bps": (
                ENTRY_SLIPPAGE_BPS
            ),
            "exit_slippage_bps": (
                EXIT_SLIPPAGE_BPS
            ),
            "round_trip_cost_bps": (
                ROUND_TRIP_COST_BPS
            ),
            "signal_fractions": (
                SIGNAL_FRACTIONS
            ),
        },
        "folds": report_folds,
        "aggregate": (
            summary_df.to_dict(
                orient="records"
            )
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
    print("ECONOMIC BACKTEST SUMMARY")
    print("=" * 84)

    display_columns = [
        "signal_fraction",
        "stop_bps",
        "signals",
        "trades",
        "mean_trigger_rate",
        "win_rate",
        "mean_net_bps",
        "median_net_bps",
        "profit_factor",
        "compounded_return",
        "max_drawdown",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("=" * 84)
    print("BACKTEST COMPLETE")
    print("=" * 84)
    print(f"Summary : {SUMMARY_PATH}")
    print(f"Folds   : {FOLD_PATH}")
    print(f"Trades  : {TRADES_PATH}")
    print(f"Report  : {REPORT_PATH}")
    print(
        f"Elapsed : "
        f"{time.time() - started:.1f}s"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
