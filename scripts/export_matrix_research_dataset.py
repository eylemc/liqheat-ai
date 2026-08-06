#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "reports/matrix_event_study_v3/"
    "matrix_event_observations.parquet"
)

OUTPUT_ROOT = Path(
    "data/research/matrix_events"
)

FULL_CSV_PATH = (
    OUTPUT_ROOT
    / "matrix_events_full.csv.gz"
)

COMPACT_CSV_PATH = (
    OUTPUT_ROOT
    / "matrix_events_compact.csv.gz"
)

SUMMARY_CSV_PATH = (
    OUTPUT_ROOT
    / "matrix_events_summary.csv"
)

SYMBOL_SUMMARY_CSV_PATH = (
    OUTPUT_ROOT
    / "matrix_events_symbol_summary.csv"
)

REPORT_PATH = (
    OUTPUT_ROOT
    / "matrix_events_export_report.json"
)

WINDOWS = [
    15,
    30,
    60,
    120,
    240,
]

TARGETS = [
    25,
    50,
    100,
]

FIRST_HIT_PAIRS = [
    (25, 25),
    (50, 25),
    (50, 50),
    (100, 50),
    (100, 100),
]


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


def ensure_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    return [
        column
        for column in columns
        if column in frame.columns
    ]


def create_compact_dataset(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "symbol",
        "entry_timeframe",
        "alignment",
        "signal_time",
        "entry_time",
        "direction",
        "entry_price",
    ]

    for window in WINDOWS:
        columns.extend([
            f"mfe_bps_{window}m",
            f"mae_bps_{window}m",
            f"terminal_return_bps_{window}m",
            f"time_to_mfe_minutes_{window}m",
            f"time_to_mae_minutes_{window}m",
            f"mfe_mae_ratio_{window}m",
        ])

        for target in TARGETS:
            columns.extend([
                f"hit_target_{target}bps_{window}m",
                f"time_to_target_{target}bps_{window}m",
            ])

        for target, stop in FIRST_HIT_PAIRS:
            prefix = (
                f"first_hit_t{target}_"
                f"s{stop}_{window}m"
            )

            columns.extend([
                prefix,
                f"{prefix}_target_minute",
                f"{prefix}_stop_minute",
            ])

    columns = ensure_columns(
        frame,
        columns,
    )

    return frame[
        columns
    ].copy()


def summarize_group(
    group: pd.DataFrame,
    window: int,
) -> dict[str, Any]:
    mfe_column = (
        f"mfe_bps_{window}m"
    )

    mae_column = (
        f"mae_bps_{window}m"
    )

    terminal_column = (
        f"terminal_return_bps_{window}m"
    )

    mfe = pd.to_numeric(
        group[mfe_column],
        errors="coerce",
    )

    mae = pd.to_numeric(
        group[mae_column],
        errors="coerce",
    )

    terminal = pd.to_numeric(
        group[terminal_column],
        errors="coerce",
    )

    result: dict[str, Any] = {
        "events": int(len(group)),

        "mean_mfe_bps": float(
            mfe.mean()
        ),

        "median_mfe_bps": float(
            mfe.median()
        ),

        "q25_mfe_bps": float(
            mfe.quantile(0.25)
        ),

        "q75_mfe_bps": float(
            mfe.quantile(0.75)
        ),

        "q90_mfe_bps": float(
            mfe.quantile(0.90)
        ),

        "mean_mae_bps": float(
            mae.mean()
        ),

        "median_mae_bps": float(
            mae.median()
        ),

        "q10_mae_bps": float(
            mae.quantile(0.10)
        ),

        "mean_terminal_return_bps": float(
            terminal.mean()
        ),

        "median_terminal_return_bps": float(
            terminal.median()
        ),

        "terminal_win_rate": float(
            (terminal > 0).mean()
        ),

        "terminal_loss_rate": float(
            (terminal < 0).mean()
        ),

        "mean_time_to_mfe_minutes": float(
            pd.to_numeric(
                group[
                    f"time_to_mfe_minutes_"
                    f"{window}m"
                ],
                errors="coerce",
            ).mean()
        ),

        "median_time_to_mfe_minutes": float(
            pd.to_numeric(
                group[
                    f"time_to_mfe_minutes_"
                    f"{window}m"
                ],
                errors="coerce",
            ).median()
        ),
    }

    for target in TARGETS:
        hit_column = (
            f"hit_target_{target}bps_"
            f"{window}m"
        )

        time_column = (
            f"time_to_target_{target}bps_"
            f"{window}m"
        )

        if hit_column in group.columns:
            hit_values = (
                group[hit_column]
                .fillna(False)
                .astype(bool)
            )

            result[
                f"target_{target}bps_hit_rate"
            ] = float(
                hit_values.mean()
            )

            hit_times = pd.to_numeric(
                group.loc[
                    hit_values,
                    time_column,
                ],
                errors="coerce",
            )

            result[
                f"target_{target}bps_"
                f"median_hit_minutes"
            ] = (
                float(hit_times.median())
                if len(hit_times)
                else np.nan
            )

    for target, stop in FIRST_HIT_PAIRS:
        column = (
            f"first_hit_t{target}_"
            f"s{stop}_{window}m"
        )

        if column not in group.columns:
            continue

        values = (
            group[column]
            .astype("string")
        )

        resolved = values.isin([
            "TARGET_FIRST",
            "STOP_FIRST",
        ])

        result[
            f"t{target}_s{stop}_"
            f"target_first_rate"
        ] = float(
            (
                values
                == "TARGET_FIRST"
            ).mean()
        )

        result[
            f"t{target}_s{stop}_"
            f"stop_first_rate"
        ] = float(
            (
                values
                == "STOP_FIRST"
            ).mean()
        )

        result[
            f"t{target}_s{stop}_"
            f"resolved_events"
        ] = int(
            resolved.sum()
        )

        result[
            f"t{target}_s{stop}_"
            f"resolved_target_win_rate"
        ] = (
            float(
                (
                    values.loc[resolved]
                    == "TARGET_FIRST"
                ).mean()
            )
            if resolved.any()
            else np.nan
        )

    return result


def create_summary(
    frame: pd.DataFrame,
    include_symbol: bool,
) -> pd.DataFrame:
    grouping_columns = [
        "entry_timeframe",
        "alignment",
    ]

    if include_symbol:
        grouping_columns.insert(
            0,
            "symbol",
        )

    rows = []

    for keys, group in frame.groupby(
        grouping_columns,
        observed=True,
        sort=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_values = dict(
            zip(
                grouping_columns,
                keys,
            )
        )

        for window in WINDOWS:
            required_column = (
                f"mfe_bps_{window}m"
            )

            if required_column not in group.columns:
                continue

            metrics = summarize_group(
                group,
                window,
            )

            rows.append({
                **key_values,
                "window_minutes": window,
                **metrics,
            })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                *grouping_columns,
                "window_minutes",
            ]
        )
        .reset_index(drop=True)
    )


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "MATRIX RESEARCH DATASET EXPORT"
    )
    print("=" * 100)
    print("Input:", INPUT_PATH)

    frame = pd.read_parquet(
        INPUT_PATH
    )

    for column in [
        "signal_time",
        "entry_time",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column],
                utc=True,
                errors="coerce",
            )

    frame = (
        frame
        .sort_values(
            [
                "symbol",
                "signal_time",
                "entry_timeframe",
                "alignment",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "Rows:",
        f"{len(frame):,}",
    )

    print(
        "Columns:",
        f"{len(frame.columns):,}",
    )

    # Tam dataset:
    # Büyük olacağı için gzip sıkıştırmalı CSV.
    frame.to_csv(
        FULL_CSV_PATH,
        index=False,
        compression="gzip",
    )

    compact = create_compact_dataset(
        frame
    )

    compact.to_csv(
        COMPACT_CSV_PATH,
        index=False,
        compression="gzip",
    )

    pooled_summary = create_summary(
        frame,
        include_symbol=False,
    )

    pooled_summary.to_csv(
        SUMMARY_CSV_PATH,
        index=False,
    )

    symbol_summary = create_summary(
        frame,
        include_symbol=True,
    )

    symbol_summary.to_csv(
        SYMBOL_SUMMARY_CSV_PATH,
        index=False,
    )

    report = {
        "status": "complete",

        "input": str(
            INPUT_PATH
        ),

        "rows": int(
            len(frame)
        ),

        "full_columns": int(
            len(frame.columns)
        ),

        "compact_columns": int(
            len(compact.columns)
        ),

        "symbols": sorted(
            frame[
                "symbol"
            ].astype(str).unique().tolist()
        ),

        "entry_timeframes": sorted(
            frame[
                "entry_timeframe"
            ].astype(str).unique().tolist()
        ),

        "alignments": sorted(
            frame[
                "alignment"
            ].astype(str).unique().tolist()
        ),

        "minimum_signal_time": (
            frame[
                "signal_time"
            ].min().isoformat()
        ),

        "maximum_signal_time": (
            frame[
                "signal_time"
            ].max().isoformat()
        ),

        "outputs": {
            "full_csv_gzip": str(
                FULL_CSV_PATH
            ),

            "compact_csv_gzip": str(
                COMPACT_CSV_PATH
            ),

            "pooled_summary_csv": str(
                SUMMARY_CSV_PATH
            ),

            "symbol_summary_csv": str(
                SYMBOL_SUMMARY_CSV_PATH
            ),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("EXPORT COMPLETE")
    print("=" * 100)

    print(
        "Full CSV:",
        FULL_CSV_PATH,
        f"({FULL_CSV_PATH.stat().st_size / 1024**2:.2f} MB)",
    )

    print(
        "Compact CSV:",
        COMPACT_CSV_PATH,
        f"({COMPACT_CSV_PATH.stat().st_size / 1024**2:.2f} MB)",
    )

    print(
        "Summary CSV:",
        SUMMARY_CSV_PATH,
    )

    print(
        "Symbol summary:",
        SYMBOL_SUMMARY_CSV_PATH,
    )

    print(
        "Report:",
        REPORT_PATH,
    )

    print()
    print(
        pooled_summary[
            [
                "entry_timeframe",
                "alignment",
                "window_minutes",
                "events",
                "mean_mfe_bps",
                "mean_mae_bps",
                "terminal_win_rate",
                "target_50bps_hit_rate",
                "target_100bps_hit_rate",
            ]
        ]
        .tail(30)
        .to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
