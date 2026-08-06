from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


ROOT = Path(
    "data/research/topology_v2_squeeze_grid"
)

FULL_SUMMARY_PATH = (
    ROOT / "grid_summary.csv"
)

TOP_RESULTS_PATH = (
    ROOT / "top_experiments.csv"
)

BEST_CONFIG_PATH = (
    ROOT / "best_configuration.json"
)


def parse_experiment_name(
    name: str,
) -> dict:
    output = {}

    for part in name.split("__"):
        if part.startswith("tf_"):
            output["timeframe"] = (
                part.removeprefix("tf_")
            )

        elif part.startswith("future_"):
            output["future_minutes"] = int(
                part
                .removeprefix("future_")
                .removesuffix("m")
            )

        elif part.startswith(
            "precursor_"
        ):
            output[
                "precursor_minutes"
            ] = int(
                part
                .removeprefix(
                    "precursor_"
                )
                .removesuffix("m")
            )

        elif part.startswith("q_"):
            output[
                "volatility_quantile"
            ] = float(
                part
                .removeprefix("q_")
                .replace("p", ".")
            )

    return output


def safe_mean(
    series: pd.Series,
) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.mean())


def safe_min(
    series: pd.Series,
) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.min())


def load_symbol_count(
    experiment_dir: Path,
) -> int:
    event_path = (
        experiment_dir
        / "detected_squeeze_events.parquet"
    )

    if not event_path.exists():
        return 0

    symbols = pd.read_parquet(
        event_path,
        columns=["symbol"],
    )["symbol"].dropna().unique()

    return int(len(symbols))


def score_experiment(
    row: dict,
) -> float:
    """
    Öncelik:
      1. En kötü fold'da dahi yüksek lift
      2. Ortalama top %1 ve %5 başarısı
      3. Yön başarısı
      4. Aşırı alarm frekansına ceza
    """

    values = [
        row.get(
            "mean_top_1pct_lift"
        ),
        row.get(
            "min_top_1pct_lift"
        ),
        row.get(
            "mean_top_5pct_lift"
        ),
        row.get(
            "min_top_5pct_lift"
        ),
    ]

    if any(
        value is None
        or not math.isfinite(value)
        for value in values
    ):
        return float("-inf")

    events_per_day = row[
        "events_per_day_all_symbols"
    ]

    direction_recall = (
        row.get(
            "mean_direction_recall"
        )
        or 0.0
    )

    balanced_accuracy = (
        row.get(
            "mean_balanced_accuracy"
        )
        or 0.0
    )

    # 15 event/gün üzerinde giderek artan ceza.
    frequency_penalty = max(
        0.0,
        events_per_day - 15.0,
    ) * 0.025

    # Çok az event üreten ve istatistiksel olarak
    # kırılgan olabilecek konfigürasyonlara küçük ceza.
    sparse_penalty = max(
        0.0,
        2.0 - events_per_day,
    ) * 0.10

    return float(
        row["mean_top_1pct_lift"]
        * 0.25
        + row["min_top_1pct_lift"]
        * 0.25
        + row["mean_top_5pct_lift"]
        * 0.20
        + row["min_top_5pct_lift"]
        * 0.20
        + direction_recall
        * 1.00
        + balanced_accuracy
        * 0.50
        - frequency_penalty
        - sparse_penalty
    )


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(
            f"Grid root not found: {ROOT}"
        )

    rows = []

    experiment_dirs = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_dir()
    )

    for experiment_dir in (
        experiment_dirs
    ):
        report_path = (
            experiment_dir
            / "report.json"
        )

        fold_summary_path = (
            experiment_dir
            / "walk_forward_summary.csv"
        )

        event_path = (
            experiment_dir
            / "detected_squeeze_events.parquet"
        )

        if not all(
            path.exists()
            for path in [
                report_path,
                fold_summary_path,
                event_path,
            ]
        ):
            continue

        try:
            report = json.loads(
                report_path.read_text(
                    encoding="utf-8"
                )
            )

            folds = pd.read_csv(
                fold_summary_path
            )

            symbol_count = (
                load_symbol_count(
                    experiment_dir
                )
            )

        except Exception as error:
            print(
                f"Skipping unreadable "
                f"experiment "
                f"{experiment_dir.name}: "
                f"{error}"
            )
            continue

        if (
            folds.empty
            or symbol_count <= 0
        ):
            continue

        parsed = (
            parse_experiment_name(
                experiment_dir.name
            )
        )

        event_count = int(
            report["events"]["count"]
        )

        events_per_day = float(
            report["events"][
                "events_per_day"
            ]
        )

        mean_long_recall = safe_mean(
            folds["long_recall"]
        )

        mean_short_recall = safe_mean(
            folds["short_recall"]
        )

        direction_values = [
            value
            for value in [
                mean_long_recall,
                mean_short_recall,
            ]
            if value is not None
        ]

        mean_direction_recall = (
            float(
                np.mean(
                    direction_values
                )
            )
            if direction_values
            else None
        )

        row = {
            "experiment": (
                experiment_dir.name
            ),
            **parsed,

            "symbol_count": (
                symbol_count
            ),

            "event_count": (
                event_count
            ),

            "events_per_day_all_symbols": (
                events_per_day
            ),

            "events_per_day_per_symbol": (
                events_per_day
                / symbol_count
            ),

            "mean_balanced_accuracy": (
                safe_mean(
                    folds[
                        "balanced_accuracy"
                    ]
                )
            ),

            "min_balanced_accuracy": (
                safe_min(
                    folds[
                        "balanced_accuracy"
                    ]
                )
            ),

            "mean_macro_f1": (
                safe_mean(
                    folds["macro_f1"]
                )
            ),

            "mean_long_recall": (
                mean_long_recall
            ),

            "mean_short_recall": (
                mean_short_recall
            ),

            "mean_direction_recall": (
                mean_direction_recall
            ),

            "mean_top_1pct_event_rate": (
                safe_mean(
                    folds[
                        "top_1pct_event_rate"
                    ]
                )
            ),

            "mean_top_1pct_lift": (
                safe_mean(
                    folds[
                        "top_1pct_lift"
                    ]
                )
            ),

            "min_top_1pct_lift": (
                safe_min(
                    folds[
                        "top_1pct_lift"
                    ]
                )
            ),

            "mean_top_5pct_event_rate": (
                safe_mean(
                    folds[
                        "top_5pct_event_rate"
                    ]
                )
            ),

            "mean_top_5pct_lift": (
                safe_mean(
                    folds[
                        "top_5pct_lift"
                    ]
                )
            ),

            "min_top_5pct_lift": (
                safe_min(
                    folds[
                        "top_5pct_lift"
                    ]
                )
            ),

            "mean_top_10pct_event_rate": (
                safe_mean(
                    folds[
                        "top_10pct_event_rate"
                    ]
                )
            ),

            "mean_top_10pct_lift": (
                safe_mean(
                    folds[
                        "top_10pct_lift"
                    ]
                )
            ),

            "min_top_10pct_lift": (
                safe_min(
                    folds[
                        "top_10pct_lift"
                    ]
                )
            ),

            "output_directory": (
                str(experiment_dir)
            ),
        }

        row["research_score"] = (
            score_experiment(row)
        )

        rows.append(row)

    if not rows:
        raise SystemExit(
            "No completed squeeze grid "
            "experiments found."
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    summary = summary.sort_values(
        [
            "research_score",
            "min_top_5pct_lift",
            "min_top_1pct_lift",
            "mean_direction_recall",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    summary.to_csv(
        FULL_SUMMARY_PATH,
        index=False,
    )

    top_results = (
        summary.head(30).copy()
    )

    top_results.to_csv(
        TOP_RESULTS_PATH,
        index=False,
    )

    best = summary.iloc[0].to_dict()

    BEST_CONFIG_PATH.write_text(
        json.dumps(
            best,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 120)
    print(
        "TOP SQUEEZE GRID EXPERIMENTS"
    )
    print("=" * 120)

    display_columns = [
        "experiment",
        "symbol_count",
        "events_per_day_all_symbols",
        "events_per_day_per_symbol",
        "mean_balanced_accuracy",
        "mean_direction_recall",
        "mean_top_1pct_event_rate",
        "mean_top_1pct_lift",
        "min_top_1pct_lift",
        "mean_top_5pct_event_rate",
        "mean_top_5pct_lift",
        "min_top_5pct_lift",
        "research_score",
    ]

    print(
        top_results[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("=" * 120)
    print("BEST CONFIGURATION")
    print("=" * 120)

    for key in [
        "experiment",
        "timeframe",
        "future_minutes",
        "precursor_minutes",
        "volatility_quantile",
        "symbol_count",
        "event_count",
        "events_per_day_all_symbols",
        "events_per_day_per_symbol",
        "mean_top_1pct_event_rate",
        "mean_top_1pct_lift",
        "min_top_1pct_lift",
        "mean_top_5pct_event_rate",
        "mean_top_5pct_lift",
        "min_top_5pct_lift",
        "mean_direction_recall",
        "research_score",
    ]:
        print(
            f"{key:30}: "
            f"{best.get(key)}"
        )

    print()
    print(
        f"Full summary : "
        f"{FULL_SUMMARY_PATH}"
    )
    print(
        f"Top results  : "
        f"{TOP_RESULTS_PATH}"
    )
    print(
        f"Best config  : "
        f"{BEST_CONFIG_PATH}"
    )


if __name__ == "__main__":
    main()
