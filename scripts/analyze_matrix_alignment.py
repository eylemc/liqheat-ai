#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_PATH = Path(
    "data/forecast_v3/"
    "matrix_topology_dataset.parquet"
)

REPORT_ROOT = Path(
    "reports/matrix_alignment"
)

HORIZONS = [15, 30, 60]

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

MATRIX_TREND_COLUMNS = {
    "1h": "matrix_1h_matrix_trend",
    "4h": "matrix_4h_matrix_trend",
    "24h": "matrix_24h_matrix_trend",
}

TOPOLOGY_SIDE_COLUMNS = {
    "1h": "tf1h_nearest_side",
    "4h": "tf4h_nearest_side",
    "24h": "tf24h_nearest_side",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def map_topology_side(
    values: pd.Series,
) -> pd.Series:
    """
    UPPER:
      Likidite hedefi fiyatın üzerinde.
      Direction +1.

    LOWER:
      Likidite hedefi fiyatın altında.
      Direction -1.
    """
    normalized = (
        values
        .astype("string")
        .str.upper()
        .str.strip()
    )

    mapped = normalized.map({
        "UPPER": 1,
        "LOWER": -1,
    })

    return pd.to_numeric(
        mapped,
        errors="coerce",
    ).astype("float32")


def add_matrix_states(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    one_hour = pd.to_numeric(
        output[
            MATRIX_TREND_COLUMNS["1h"]
        ],
        errors="coerce",
    )

    four_hour = pd.to_numeric(
        output[
            MATRIX_TREND_COLUMNS["4h"]
        ],
        errors="coerce",
    )

    daily = pd.to_numeric(
        output[
            MATRIX_TREND_COLUMNS["24h"]
        ],
        errors="coerce",
    )

    valid = (
        one_hour.isin([-1, 1])
        & four_hour.isin([-1, 1])
        & daily.isin([-1, 1])
    )

    full_bullish = (
        valid
        & (one_hour == 1)
        & (four_hour == 1)
        & (daily == 1)
    )

    full_bearish = (
        valid
        & (one_hour == -1)
        & (four_hour == -1)
        & (daily == -1)
    )

    output["matrix_fully_aligned"] = (
        full_bullish | full_bearish
    )

    output["matrix_alignment_direction"] = np.select(
        [
            full_bullish,
            full_bearish,
        ],
        [
            1,
            -1,
        ],
        default=0,
    ).astype("int8")

    # Üst zaman dilimlerinin çekirdek rejimi.
    output["matrix_24h_4h_aligned"] = (
        daily.isin([-1, 1])
        & (daily == four_hour)
    )

    output["matrix_24h_4h_direction"] = np.where(
        output["matrix_24h_4h_aligned"],
        daily,
        0,
    ).astype("int8")

    # 1h, üst rejime katılıyor mu?
    output["matrix_1h_confirms_upper"] = (
        output["matrix_24h_4h_aligned"]
        & (one_hour == daily)
    )

    output["matrix_1h_opposes_upper"] = (
        output["matrix_24h_4h_aligned"]
        & one_hour.isin([-1, 1])
        & (one_hour != daily)
    )

    output["matrix_regime"] = np.select(
        [
            full_bullish,
            full_bearish,
            output["matrix_24h_4h_aligned"]
            & (daily == 1),
            output["matrix_24h_4h_aligned"]
            & (daily == -1),
        ],
        [
            "FULL_LONG_ALIGNMENT",
            "FULL_SHORT_ALIGNMENT",
            "UPPER_LONG_1H_NOT_ALIGNED",
            "UPPER_SHORT_1H_NOT_ALIGNED",
        ],
        default="MIXED",
    )

    return output


def add_topology_direction(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    topology_directions = {}

    for timeframe, column in (
        TOPOLOGY_SIDE_COLUMNS.items()
    ):
        topology_directions[timeframe] = (
            map_topology_side(
                output[column]
            )
        )

        output[
            f"topology_{timeframe}_direction"
        ] = topology_directions[timeframe]

    direction_matrix = np.column_stack([
        topology_directions["1h"],
        topology_directions["4h"],
        topology_directions["24h"],
    ])

    valid_count = np.sum(
        np.isfinite(direction_matrix),
        axis=1,
    )

    direction_sum = np.nansum(
        direction_matrix,
        axis=1,
    )

    output[
        "topology_valid_timeframes"
    ] = valid_count.astype("int8")

    output[
        "topology_direction_sum"
    ] = direction_sum.astype("float32")

    # Majority vote:
    # +1 upper-target bias
    # -1 lower-target bias
    #  0 tie / unavailable
    output[
        "topology_direction"
    ] = np.where(
        valid_count >= 2,
        np.sign(direction_sum),
        0,
    ).astype("int8")

    output[
        "topology_full_alignment"
    ] = (
        (valid_count == 3)
        & (np.abs(direction_sum) == 3)
    )

    output[
        "topology_agreement_strength"
    ] = np.where(
        valid_count > 0,
        np.abs(direction_sum) / valid_count,
        np.nan,
    ).astype("float32")

    output[
        "topology_matrix_agree"
    ] = (
        output["matrix_fully_aligned"]
        & output[
            "topology_direction"
        ].isin([-1, 1])
        & (
            output["topology_direction"]
            == output[
                "matrix_alignment_direction"
            ]
        )
    )

    output[
        "topology_matrix_conflict"
    ] = (
        output["matrix_fully_aligned"]
        & output[
            "topology_direction"
        ].isin([-1, 1])
        & (
            output["topology_direction"]
            != output[
                "matrix_alignment_direction"
            ]
        )
    )

    return output


def non_overlapping_sample(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Forward-return pencerelerinin üst üste binmesini
    azaltmak için her symbol içinde horizon dakikalık
    zaman bucket'ından tek gözlem alır.
    """
    if frame.empty:
        return frame.copy()

    output_parts = []

    for symbol, part in frame.groupby(
        "symbol",
        observed=True,
        sort=False,
    ):
        part = (
            part
            .sort_values(
                "observation_time"
            )
            .copy()
        )

        epoch_minutes = (
            part["observation_time"]
            .astype("int64")
            // 60_000_000_000
        )

        part["_non_overlap_bucket"] = (
            epoch_minutes // horizon
        )

        sampled = (
            part
            .groupby(
                "_non_overlap_bucket",
                observed=True,
                as_index=False,
                sort=False,
            )
            .first()
        )

        sampled["symbol"] = symbol

        output_parts.append(sampled)

    if not output_parts:
        return frame.iloc[0:0].copy()

    return pd.concat(
        output_parts,
        ignore_index=True,
    )


def calculate_profit_factor(
    directional_returns: pd.Series,
) -> float:
    positive_sum = float(
        directional_returns[
            directional_returns > 0
        ].sum()
    )

    negative_sum = float(
        directional_returns[
            directional_returns < 0
        ].sum()
    )

    if negative_sum == 0:
        return float("inf") if positive_sum > 0 else np.nan

    return positive_sum / abs(negative_sum)


def summarize_returns(
    frame: pd.DataFrame,
    return_column: str,
    direction_column: str | None,
    horizon: int,
    cohort: str,
    symbol: str,
) -> dict[str, Any]:
    working = frame.copy()

    actual_return = pd.to_numeric(
        working[return_column],
        errors="coerce",
    )

    if direction_column is None:
        directional_return = actual_return
    else:
        direction = pd.to_numeric(
            working[direction_column],
            errors="coerce",
        )

        valid_direction = direction.isin(
            [-1, 1]
        )

        working = working.loc[
            valid_direction
        ].copy()

        actual_return = actual_return.loc[
            valid_direction
        ]

        direction = direction.loc[
            valid_direction
        ]

        directional_return = (
            actual_return * direction
        )

    valid_return = (
        directional_return.notna()
        & np.isfinite(directional_return)
    )

    working = working.loc[
        valid_return
    ].copy()

    directional_return = (
        directional_return.loc[
            valid_return
        ].astype(float)
    )

    actual_return = (
        actual_return.loc[
            valid_return
        ].astype(float)
    )

    if working.empty:
        return {
            "cohort": cohort,
            "symbol": symbol,
            "horizon_minutes": horizon,
            "rows": 0,
        }

    sampled = non_overlapping_sample(
        working.assign(
            _directional_return=(
                directional_return.to_numpy()
            )
        ),
        horizon,
    )

    sampled_returns = pd.to_numeric(
        sampled["_directional_return"],
        errors="coerce",
    ).dropna()

    sample_count = len(sampled_returns)

    sample_mean = (
        float(sampled_returns.mean())
        if sample_count
        else np.nan
    )

    sample_std = (
        float(sampled_returns.std(ddof=1))
        if sample_count > 1
        else np.nan
    )

    standard_error = (
        sample_std / math.sqrt(sample_count)
        if (
            sample_count > 1
            and np.isfinite(sample_std)
            and sample_std > 0
        )
        else np.nan
    )

    t_stat = (
        sample_mean / standard_error
        if (
            np.isfinite(standard_error)
            and standard_error > 0
        )
        else np.nan
    )

    gross_positive = float(
        directional_return[
            directional_return > 0
        ].sum()
    )

    gross_negative = float(
        directional_return[
            directional_return < 0
        ].sum()
    )

    return {
        "cohort": cohort,
        "symbol": symbol,
        "horizon_minutes": horizon,

        "rows": int(len(directional_return)),

        "mean_directional_return_bps": float(
            directional_return.mean()
        ),

        "median_directional_return_bps": float(
            directional_return.median()
        ),

        "win_rate": float(
            (directional_return > 0).mean()
        ),

        "loss_rate": float(
            (directional_return < 0).mean()
        ),

        "flat_rate": float(
            (directional_return == 0).mean()
        ),

        "profit_factor": float(
            calculate_profit_factor(
                directional_return
            )
        ),

        "gross_positive_bps": (
            gross_positive
        ),

        "gross_negative_bps": (
            gross_negative
        ),

        "q05_bps": float(
            directional_return.quantile(0.05)
        ),

        "q25_bps": float(
            directional_return.quantile(0.25)
        ),

        "q75_bps": float(
            directional_return.quantile(0.75)
        ),

        "q95_bps": float(
            directional_return.quantile(0.95)
        ),

        "actual_unsigned_mean_bps": float(
            actual_return.mean()
        ),

        "non_overlapping_rows": int(
            sample_count
        ),

        "non_overlapping_mean_bps": (
            sample_mean
        ),

        "non_overlapping_std_bps": (
            sample_std
        ),

        "non_overlapping_t_stat": (
            t_stat
        ),

        "minimum_time": (
            working[
                "observation_time"
            ].min().isoformat()
        ),

        "maximum_time": (
            working[
                "observation_time"
            ].max().isoformat()
        ),
    }


def build_cohorts(
    frame: pd.DataFrame,
) -> list[
    tuple[
        str,
        pd.Series,
        str | None,
    ]
]:
    return [
        (
            "MARKET_UNFILTERED",
            pd.Series(
                True,
                index=frame.index,
            ),
            None,
        ),

        (
            "MATRIX_FULL_ALIGNMENT",
            frame[
                "matrix_fully_aligned"
            ],
            "matrix_alignment_direction",
        ),

        (
            "MATRIX_24H_4H_ALIGNMENT",
            frame[
                "matrix_24h_4h_aligned"
            ],
            "matrix_24h_4h_direction",
        ),

        (
            "MATRIX_24H_4H_PLUS_1H_CONFIRMATION",
            frame[
                "matrix_1h_confirms_upper"
            ],
            "matrix_24h_4h_direction",
        ),

        (
            "MATRIX_24H_4H_WITH_1H_OPPOSITION",
            frame[
                "matrix_1h_opposes_upper"
            ],
            "matrix_24h_4h_direction",
        ),

        (
            "TOPOLOGY_ONLY",
            frame[
                "topology_direction"
            ].isin([-1, 1]),
            "topology_direction",
        ),

        (
            "TOPOLOGY_MATRIX_AGREEMENT",
            frame[
                "topology_matrix_agree"
            ],
            "matrix_alignment_direction",
        ),

        (
            "TOPOLOGY_MATRIX_CONFLICT",
            frame[
                "topology_matrix_conflict"
            ],
            "topology_direction",
        ),

        (
            "TOPOLOGY_FULL_ALIGNMENT_ONLY",
            frame[
                "topology_full_alignment"
            ]
            & frame[
                "topology_direction"
            ].isin([-1, 1]),
            "topology_direction",
        ),

        (
            "TOPOLOGY_FULL_ALIGNMENT_PLUS_MATRIX",
            frame[
                "topology_full_alignment"
            ]
            & frame[
                "topology_matrix_agree"
            ],
            "matrix_alignment_direction",
        ),
    ]


def matrix_state_distribution(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for symbol_scope in [
        "ALL",
        *SYMBOLS,
    ]:
        scoped = (
            frame
            if symbol_scope == "ALL"
            else frame[
                frame["symbol"].astype(str)
                == symbol_scope
            ]
        )

        counts = (
            scoped[
                "matrix_regime"
            ]
            .value_counts(
                dropna=False
            )
        )

        for regime, count in counts.items():
            rows.append({
                "symbol": symbol_scope,
                "matrix_regime": str(
                    regime
                ),
                "rows": int(count),
                "share": float(
                    count / len(scoped)
                ) if len(scoped) else np.nan,
            })

    return pd.DataFrame(rows)


def comparison_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    selected = metrics[
        metrics["symbol"] == "ALL"
    ].copy()

    value_columns = [
        "mean_directional_return_bps",
        "win_rate",
        "profit_factor",
        "non_overlapping_mean_bps",
        "non_overlapping_t_stat",
        "rows",
    ]

    available = [
        column
        for column in value_columns
        if column in selected.columns
    ]

    pivot = selected.pivot_table(
        index="cohort",
        columns="horizon_minutes",
        values=available,
        aggfunc="first",
    )

    pivot.columns = [
        f"{metric}_{horizon}m"
        for metric, horizon
        in pivot.columns
    ]

    return (
        pivot
        .reset_index()
        .sort_values(
            "cohort"
        )
    )


def main() -> int:
    started = time.time()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET_PATH}"
        )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print(
        "LIQHEAT — MULTI-TIMEFRAME "
        "MATRIX ALIGNMENT STUDY"
    )
    print("=" * 110)
    print("Dataset:", DATASET_PATH)
    print()

    frame = pd.read_parquet(
        DATASET_PATH
    )

    frame[
        "observation_time"
    ] = pd.to_datetime(
        frame["observation_time"],
        utc=True,
        errors="coerce",
    )

    required_columns = [
        "symbol",
        "observation_time",
        *MATRIX_TREND_COLUMNS.values(),
        *TOPOLOGY_SIDE_COLUMNS.values(),
    ]

    for horizon in HORIZONS:
        required_columns.extend([
            f"future_valid_{horizon}m",
            f"future_return_bps_{horizon}m",
        ])

    require_columns(
        frame,
        required_columns,
    )

    frame = frame[
        frame["symbol"].astype(str)
        .isin(SYMBOLS)
        & frame[
            "observation_time"
        ].notna()
    ].copy()

    frame = add_matrix_states(
        frame
    )

    frame = add_topology_direction(
        frame
    )

    print(
        "Rows:",
        f"{len(frame):,}",
    )

    print()
    print("Matrix regime distribution:")

    regime_distribution = (
        matrix_state_distribution(
            frame
        )
    )

    print(
        regime_distribution[
            regime_distribution["symbol"]
            == "ALL"
        ].to_string(
            index=False
        )
    )

    all_results = []

    cohorts = build_cohorts(
        frame
    )

    for horizon in HORIZONS:
        return_column = (
            f"future_return_bps_"
            f"{horizon}m"
        )

        valid_column = (
            f"future_valid_"
            f"{horizon}m"
        )

        horizon_frame = frame[
            (frame[valid_column] == 1)
            & frame[
                return_column
            ].notna()
        ].copy()

        print()
        print("=" * 110)
        print(
            f"{horizon} MINUTE FORWARD RETURNS"
        )
        print("=" * 110)

        for cohort_name, mask, direction in cohorts:
            cohort_mask = mask.loc[
                horizon_frame.index
            ].fillna(False)

            cohort_frame = horizon_frame.loc[
                cohort_mask
            ].copy()

            all_scope = summarize_returns(
                cohort_frame,
                return_column,
                direction,
                horizon,
                cohort_name,
                "ALL",
            )

            all_results.append(
                all_scope
            )

            print(
                f"{cohort_name:<46} "
                f"rows={all_scope.get('rows', 0):>8,} "
                f"mean={all_scope.get('mean_directional_return_bps', np.nan):>8.3f} "
                f"win={all_scope.get('win_rate', np.nan):>7.2%} "
                f"PF={all_scope.get('profit_factor', np.nan):>7.3f} "
                f"t={all_scope.get('non_overlapping_t_stat', np.nan):>7.3f}"
            )

            for symbol in SYMBOLS:
                symbol_frame = cohort_frame[
                    cohort_frame[
                        "symbol"
                    ].astype(str)
                    == symbol
                ]

                symbol_result = summarize_returns(
                    symbol_frame,
                    return_column,
                    direction,
                    horizon,
                    cohort_name,
                    symbol,
                )

                all_results.append(
                    symbol_result
                )

    metrics = pd.DataFrame(
        all_results
    )

    metrics.to_csv(
        REPORT_ROOT
        / "cohort_metrics.csv",
        index=False,
    )

    regime_distribution.to_csv(
        REPORT_ROOT
        / "matrix_regime_distribution.csv",
        index=False,
    )

    comparison = comparison_table(
        metrics
    )

    comparison.to_csv(
        REPORT_ROOT
        / "comparison_summary.csv",
        index=False,
    )

    # Agreement ile conflict'i doğrudan yan yana göster.
    agreement_comparison_rows = []

    for symbol in [
        "ALL",
        *SYMBOLS,
    ]:
        for horizon in HORIZONS:
            scoped = metrics[
                (metrics["symbol"] == symbol)
                & (
                    metrics[
                        "horizon_minutes"
                    ] == horizon
                )
            ]

            agree = scoped[
                scoped["cohort"]
                == "TOPOLOGY_MATRIX_AGREEMENT"
            ]

            conflict = scoped[
                scoped["cohort"]
                == "TOPOLOGY_MATRIX_CONFLICT"
            ]

            topology = scoped[
                scoped["cohort"]
                == "TOPOLOGY_ONLY"
            ]

            if (
                agree.empty
                or conflict.empty
                or topology.empty
            ):
                continue

            agree_row = agree.iloc[0]
            conflict_row = conflict.iloc[0]
            topology_row = topology.iloc[0]

            agreement_comparison_rows.append({
                "symbol": symbol,
                "horizon_minutes": horizon,

                "topology_only_rows": int(
                    topology_row["rows"]
                ),

                "topology_only_mean_bps": (
                    topology_row[
                        "mean_directional_return_bps"
                    ]
                ),

                "topology_only_win_rate": (
                    topology_row[
                        "win_rate"
                    ]
                ),

                "agreement_rows": int(
                    agree_row["rows"]
                ),

                "agreement_mean_bps": (
                    agree_row[
                        "mean_directional_return_bps"
                    ]
                ),

                "agreement_win_rate": (
                    agree_row["win_rate"]
                ),

                "agreement_profit_factor": (
                    agree_row[
                        "profit_factor"
                    ]
                ),

                "conflict_rows": int(
                    conflict_row["rows"]
                ),

                "conflict_mean_bps": (
                    conflict_row[
                        "mean_directional_return_bps"
                    ]
                ),

                "conflict_win_rate": (
                    conflict_row[
                        "win_rate"
                    ]
                ),

                "conflict_profit_factor": (
                    conflict_row[
                        "profit_factor"
                    ]
                ),

                "agreement_minus_topology_mean_bps": (
                    agree_row[
                        "mean_directional_return_bps"
                    ]
                    - topology_row[
                        "mean_directional_return_bps"
                    ]
                ),

                "agreement_minus_conflict_mean_bps": (
                    agree_row[
                        "mean_directional_return_bps"
                    ]
                    - conflict_row[
                        "mean_directional_return_bps"
                    ]
                ),

                "agreement_minus_conflict_win_rate": (
                    agree_row["win_rate"]
                    - conflict_row["win_rate"]
                ),
            })

    agreement_comparison = pd.DataFrame(
        agreement_comparison_rows
    )

    agreement_comparison.to_csv(
        REPORT_ROOT
        / "topology_matrix_agreement_comparison.csv",
        index=False,
    )

    final_report = {
        "status": "complete",
        "engine": (
            "liqheat-matrix-alignment-study-v1"
        ),
        "dataset": str(
            DATASET_PATH
        ),
        "rows": int(len(frame)),
        "symbols": SYMBOLS,
        "horizons_minutes": (
            HORIZONS
        ),
        "matrix_rule": (
            "24h, 4h and 1h trends must "
            "all equal +1 or all equal -1"
        ),
        "topology_rule": (
            "majority vote of nearest-side "
            "UPPER=+1 and LOWER=-1 across "
            "1h, 4h and 24h"
        ),
        "overlap_control": (
            "non-overlapping horizon buckets "
            "are used for t-statistics"
        ),
        "cohort_count": len(
            cohorts
        ),
        "elapsed_seconds": float(
            time.time() - started
        ),
        "outputs": {
            "cohort_metrics": str(
                REPORT_ROOT
                / "cohort_metrics.csv"
            ),
            "comparison_summary": str(
                REPORT_ROOT
                / "comparison_summary.csv"
            ),
            "agreement_comparison": str(
                REPORT_ROOT
                / (
                    "topology_matrix_"
                    "agreement_comparison.csv"
                )
            ),
            "matrix_regime_distribution": str(
                REPORT_ROOT
                / "matrix_regime_distribution.csv"
            ),
        },
    }

    (
        REPORT_ROOT
        / "matrix_alignment_report.json"
    ).write_text(
        json.dumps(
            json_safe(final_report),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 140)
    print("MATRIX ALIGNMENT STUDY COMPLETE")
    print("=" * 140)

    print()
    print(
        "TOPOLOGY VS MATRIX AGREEMENT"
    )

    print(
        agreement_comparison[
            agreement_comparison["symbol"]
            == "ALL"
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Outputs:",
        REPORT_ROOT,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
