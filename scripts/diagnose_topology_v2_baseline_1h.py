from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


MODEL_PATH = Path(
    "data/models/topology_v2_baseline_1h/model.joblib"
)

PREDICTIONS_PATH = Path(
    "data/models/topology_v2_baseline_1h/test_predictions.parquet"
)

OUTPUT_DIR = Path(
    "data/reports/topology_v2_baseline_1h"
)

GROUP_REPORT_PATH = OUTPUT_DIR / "group_performance.csv"
PROBABILITY_REPORT_PATH = OUTPUT_DIR / "probability_profile.csv"
CONFIDENCE_REPORT_PATH = OUTPUT_DIR / "confidence_performance.csv"


def calculate_metrics(group: pd.DataFrame) -> pd.Series:
    actual = group["direction_1h"]
    predicted = group["predicted_direction_1h"]

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[-1, 0, 1],
    )

    return pd.Series(
        {
            "rows": len(group),
            "accuracy": accuracy_score(
                actual,
                predicted,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                actual,
                predicted,
            ),
            "macro_f1": f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0,
            ),
            "actual_down": int((actual == -1).sum()),
            "actual_neutral": int((actual == 0).sum()),
            "actual_up": int((actual == 1).sum()),
            "predicted_down": int((predicted == -1).sum()),
            "predicted_neutral": int((predicted == 0).sum()),
            "predicted_up": int((predicted == 1).sum()),
            "correct_down": int(matrix[0, 0]),
            "correct_neutral": int(matrix[1, 1]),
            "correct_up": int(matrix[2, 2]),
        }
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("TOPOLOGY V2 — 1H BASELINE DIAGNOSTICS")
    print("=" * 76)

    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(PREDICTIONS_PATH)

    probability_columns = {
        -1: "probability_class_-1",
        0: "probability_class_0",
        1: "probability_class_1",
    }

    for column in probability_columns.values():
        if column not in df.columns:
            raise ValueError(
                f"Missing probability column: {column}"
            )

    df["maximum_probability"] = df[
        list(probability_columns.values())
    ].max(axis=1)

    df["probability_margin"] = (
        df[list(probability_columns.values())]
        .apply(
            lambda row: (
                np.sort(row.to_numpy())[-1]
                - np.sort(row.to_numpy())[-2]
            ),
            axis=1,
        )
    )

    print()
    print("Model classes:")
    print(
        model.named_steps["classifier"].classes_
    )

    print()
    print("-" * 76)
    print("OVERALL METRICS")
    print("-" * 76)
    print(calculate_metrics(df).to_string())

    reports = []

    overall = calculate_metrics(df).to_frame().T
    overall.insert(0, "group_type", "overall")
    overall.insert(1, "group_value", "all")
    reports.append(overall)

    for group_type, columns in [
        ("symbol", ["symbol"]),
        ("timeframe", ["timeframe"]),
        (
            "symbol_timeframe",
            ["symbol", "timeframe"],
        ),
    ]:
        grouped = (
            df.groupby(
                columns,
                observed=True,
                dropna=False,
            )
            .apply(
                calculate_metrics,
                include_groups=False,
            )
            .reset_index()
        )

        if len(columns) == 1:
            grouped["group_value"] = (
                grouped[columns[0]].astype(str)
            )
        else:
            grouped["group_value"] = (
                grouped[columns]
                .astype(str)
                .agg(" / ".join, axis=1)
            )

        grouped.insert(
            0,
            "group_type",
            group_type,
        )

        grouped = grouped.drop(
            columns=columns,
        )

        reports.append(grouped)

    group_report = pd.concat(
        reports,
        ignore_index=True,
    )

    group_report.to_csv(
        GROUP_REPORT_PATH,
        index=False,
    )

    print()
    print("-" * 76)
    print("PERFORMANCE BY SYMBOL")
    print("-" * 76)

    print(
        group_report.loc[
            group_report["group_type"] == "symbol",
            [
                "group_value",
                "rows",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "predicted_down",
                "predicted_neutral",
                "predicted_up",
            ],
        ]
        .sort_values(
            "balanced_accuracy",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print("-" * 76)
    print("PERFORMANCE BY TIMEFRAME")
    print("-" * 76)

    print(
        group_report.loc[
            group_report["group_type"] == "timeframe",
            [
                "group_value",
                "rows",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "predicted_down",
                "predicted_neutral",
                "predicted_up",
            ],
        ]
        .sort_values(
            "balanced_accuracy",
            ascending=False,
        )
        .to_string(index=False)
    )

    probability_rows = []

    for class_value, column in probability_columns.items():
        series = df[column]

        profile = series.describe(
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

        probability_rows.append(
            {
                "class": class_value,
                "mean": profile["mean"],
                "std": profile["std"],
                "min": profile["min"],
                "p01": profile["1%"],
                "p05": profile["5%"],
                "p25": profile["25%"],
                "p50": profile["50%"],
                "p75": profile["75%"],
                "p95": profile["95%"],
                "p99": profile["99%"],
                "max": profile["max"],
            }
        )

    probability_report = pd.DataFrame(
        probability_rows
    )

    probability_report.to_csv(
        PROBABILITY_REPORT_PATH,
        index=False,
    )

    print()
    print("-" * 76)
    print("PROBABILITY PROFILE")
    print("-" * 76)
    print(
        probability_report.to_string(
            index=False
        )
    )

    confidence_bins = [
        0.00,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.70,
        0.80,
        0.90,
        1.01,
    ]

    df["confidence_bin"] = pd.cut(
        df["maximum_probability"],
        bins=confidence_bins,
        right=False,
        include_lowest=True,
    )

    confidence_report = (
        df.groupby(
            "confidence_bin",
            observed=True,
        )
        .apply(
            calculate_metrics,
            include_groups=False,
        )
        .reset_index()
    )

    confidence_report[
        "coverage_pct"
    ] = (
        confidence_report["rows"]
        / len(df)
        * 100
    )

    confidence_report.to_csv(
        CONFIDENCE_REPORT_PATH,
        index=False,
    )

    print()
    print("-" * 76)
    print("PERFORMANCE BY CONFIDENCE")
    print("-" * 76)
    print(
        confidence_report[
            [
                "confidence_bin",
                "rows",
                "coverage_pct",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 76)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 76)
    print(f"Group report      : {GROUP_REPORT_PATH}")
    print(f"Probability report: {PROBABILITY_REPORT_PATH}")
    print(f"Confidence report : {CONFIDENCE_REPORT_PATH}")


if __name__ == "__main__":
    main()
