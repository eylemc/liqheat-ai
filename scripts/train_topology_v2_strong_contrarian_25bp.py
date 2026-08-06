from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURE_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

LABEL_PATH = Path(
    "data/features/liq_topology_v2_strong_contrarian_labels.parquet"
)

OUTPUT_DIR = Path(
    "data/models/topology_v2_strong_contrarian_25bp"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.parquet"

TARGET_COLUMN = "strong_contrarian_25bp_1h"

EMBARGO = pd.Timedelta(hours=4)

MAX_TRAIN_ROWS = 1_200_000
MAX_VALIDATION_ROWS = 300_000
MAX_TEST_ROWS = 300_000

RANDOM_STATE = 42

BASE_CATEGORICAL = [
    "symbol",
    "timeframe",
    "nearest_side",
]

DISTANCE_FEATURES = [
    "nearest_side_code",
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",
    "log1p_upper_distance_pct",
    "log1p_lower_distance_pct",
    "log1p_distance_advantage",
]

PURE_VOLUME_FEATURES = [
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
]

STRUCTURE_FEATURES = [
    "pool_volume_ratio",
    "log1p_pool_volume_ratio",
    "distance_pressure_ratio",
    "log1p_distance_pressure_ratio",
    "topology_imbalance",
    "total_volume_imbalance_check",
    "active_level_difference",
    "active_level_total",
]

CALENDAR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_utc",
]

EXPERIMENTS = {
    "distance_only": {
        "categorical": BASE_CATEGORICAL,
        "numeric": DISTANCE_FEATURES,
    },
    "volume_structure": {
        "categorical": BASE_CATEGORICAL,
        "numeric": PURE_VOLUME_FEATURES + STRUCTURE_FEATURES,
    },
    "full_combined": {
        "categorical": BASE_CATEGORICAL,
        "numeric": (
            DISTANCE_FEATURES
            + PURE_VOLUME_FEATURES
            + STRUCTURE_FEATURES
            + CALENDAR_FEATURES
        ),
    },
}


def all_features():
    columns = []
    for config in EXPERIMENTS.values():
        columns.extend(config["categorical"])
        columns.extend(config["numeric"])
    return list(dict.fromkeys(columns))


def sample_frame(df, maximum_rows, seed):
    if len(df) <= maximum_rows:
        return df.copy()

    return (
        df.sample(
            n=maximum_rows,
            random_state=seed,
            replace=False,
        )
        .sort_values(["logged_at", "id"])
        .reset_index(drop=True)
    )


def prepare_features(df, categorical, numeric):
    columns = categorical + numeric
    X = df[columns].copy()

    for column in categorical:
        X[column] = (
            X[column]
            .astype("string")
            .fillna("<MISSING>")
            .astype(str)
        )

    for column in numeric:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


def threshold_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype("int8")

    return {
        "threshold": float(threshold),
        "predicted_positive": int(predictions.sum()),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }


def top_fraction_metrics(y_true, probabilities):
    rows = []

    order = np.argsort(-probabilities)
    base_rate = float(np.mean(y_true))

    for fraction in [0.01, 0.02, 0.05, 0.10, 0.20]:
        count = max(
            1,
            int(len(y_true) * fraction),
        )

        selected = order[:count]
        event_rate = float(
            np.mean(
                np.asarray(y_true)[selected]
            )
        )

        rows.append(
            {
                "top_fraction": fraction,
                "rows": count,
                "event_rate": event_rate,
                "base_rate": base_rate,
                "lift": (
                    event_rate / base_rate
                    if base_rate > 0
                    else None
                ),
            }
        )

    return rows


def train_experiment(
    name,
    config,
    train_df,
    validation_df,
    test_df,
):
    categorical = config["categorical"]
    numeric = config["numeric"]
    feature_columns = categorical + numeric

    print()
    print("=" * 80)
    print(f"EXPERIMENT: {name.upper()}")
    print("=" * 80)
    print(f"Feature count: {len(feature_columns)}")

    X_train = prepare_features(
        train_df,
        categorical,
        numeric,
    )
    y_train = train_df[TARGET_COLUMN]

    X_validation = prepare_features(
        validation_df,
        categorical,
        numeric,
    )
    y_validation = validation_df[
        TARGET_COLUMN
    ]

    X_test = prepare_features(
        test_df,
        categorical,
        numeric,
    )
    y_test = test_df[TARGET_COLUMN]

    categorical_indices = [
        feature_columns.index(column)
        for column in categorical
    ]

    train_pool = Pool(
        X_train,
        label=y_train,
        cat_features=categorical_indices,
        feature_names=feature_columns,
    )

    validation_pool = Pool(
        X_validation,
        label=y_validation,
        cat_features=categorical_indices,
        feature_names=feature_columns,
    )

    test_pool = Pool(
        X_test,
        label=y_test,
        cat_features=categorical_indices,
        feature_names=feature_columns,
    )

    model = CatBoostClassifier(
        iterations=1500,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        random_strength=1.0,
        l2_leaf_reg=7.0,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        od_type="Iter",
        od_wait=150,
        use_best_model=True,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
    )

    started = time.time()

    model.fit(
        train_pool,
        eval_set=validation_pool,
    )

    training_seconds = time.time() - started

    validation_probabilities = (
        model.predict_proba(validation_pool)[:, 1]
    )

    test_probabilities = (
        model.predict_proba(test_pool)[:, 1]
    )

    # Validation setinde en yüksek F1 eşiğini seç.
    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    threshold_results = [
        threshold_metrics(
            y_validation,
            validation_probabilities,
            threshold,
        )
        for threshold in thresholds
    ]

    best_threshold_result = max(
        threshold_results,
        key=lambda row: row["f1"],
    )

    best_threshold = (
        best_threshold_result["threshold"]
    )

    test_at_05 = threshold_metrics(
        y_test,
        test_probabilities,
        0.50,
    )

    test_at_best = threshold_metrics(
        y_test,
        test_probabilities,
        best_threshold,
    )

    validation_metrics = {
        "roc_auc": float(
            roc_auc_score(
                y_validation,
                validation_probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_validation,
                validation_probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_validation,
                validation_probabilities,
            )
        ),
        "best_threshold": best_threshold,
        "best_threshold_metrics": (
            best_threshold_result
        ),
    }

    test_metrics = {
        "roc_auc": float(
            roc_auc_score(
                y_test,
                test_probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_test,
                test_probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_test,
                test_probabilities,
            )
        ),
        "threshold_0_50": test_at_05,
        "validation_selected_threshold": (
            test_at_best
        ),
        "top_fraction_metrics": (
            top_fraction_metrics(
                y_test,
                test_probabilities,
            )
        ),
    }

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.get_feature_importance(
                train_pool,
                type="FeatureImportance",
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    model.save_model(
        str(OUTPUT_DIR / f"{name}.cbm")
    )

    importance.to_csv(
        OUTPUT_DIR / f"{name}_feature_importance.csv",
        index=False,
    )

    print()
    print(
        f"Best iteration : "
        f"{model.get_best_iteration()}"
    )
    print(
        f"Training time  : "
        f"{training_seconds:.1f}s"
    )

    print()
    print("Validation:")
    print(
        f"  ROC AUC={validation_metrics['roc_auc']:.4f}  "
        f"PR AUC={validation_metrics['pr_auc']:.4f}  "
        f"best threshold={best_threshold:.3f}"
    )

    print()
    print("Test:")
    print(
        f"  ROC AUC={test_metrics['roc_auc']:.4f}  "
        f"PR AUC={test_metrics['pr_auc']:.4f}  "
        f"log loss={test_metrics['log_loss']:.4f}"
    )

    print(
        f"  best-threshold precision="
        f"{test_at_best['precision']:.4f}  "
        f"recall={test_at_best['recall']:.4f}  "
        f"f1={test_at_best['f1']:.4f}"
    )

    print()
    print("Top probability buckets:")
    for row in test_metrics[
        "top_fraction_metrics"
    ]:
        print(
            f"  top {row['top_fraction'] * 100:>4.0f}% "
            f"event_rate={row['event_rate']:.4f} "
            f"lift={row['lift']:.2f}x"
        )

    print()
    print("Top 15 features:")
    print(
        importance.head(15).to_string(
            index=False
        )
    )

    return {
        "experiment": name,
        "feature_count": len(feature_columns),
        "best_iteration": int(
            model.get_best_iteration()
        ),
        "training_seconds": training_seconds,
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_importance": (
            importance.to_dict(
                orient="records"
            )
        ),
        "test_probabilities": (
            test_probabilities
        ),
    }


def main():
    started = time.time()

    for path in [FEATURE_PATH, LABEL_PATH]:
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

    print("=" * 80)
    print("TOPOLOGY V2 — STRONG CONTRARIAN 25BP MODEL")
    print("=" * 80)

    selected_columns = list(
        dict.fromkeys(
            [
                "id",
                "logged_at",
                "symbol",
                "timeframe",
                "nearest_side",
            ]
            + all_features()
        )
    )

    print("Loading features...")

    features = pd.read_parquet(
        FEATURE_PATH,
        columns=selected_columns,
    )

    print("Loading labels...")

    labels = pd.read_parquet(
        LABEL_PATH,
        columns=[
            "id",
            TARGET_COLUMN,
        ],
    )

    df = features.merge(
        labels,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    df = df.loc[
        df[TARGET_COLUMN].isin([0, 1])
    ].copy()

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN].astype("int8")
    )

    df = df.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    print()
    print(f"Usable rows: {len(df):,}")
    print("Target distribution:")
    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Target distribution (%):")
    print(
        (
            df[TARGET_COLUMN]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).to_string()
    )

    train_boundary = df[
        "logged_at"
    ].quantile(0.70)

    validation_boundary = df[
        "logged_at"
    ].quantile(0.85)

    train_end = train_boundary - EMBARGO
    validation_start = (
        train_boundary + EMBARGO
    )
    validation_end = (
        validation_boundary - EMBARGO
    )
    test_start = (
        validation_boundary + EMBARGO
    )

    train_df = df.loc[
        df["logged_at"] <= train_end
    ].copy()

    validation_df = df.loc[
        (df["logged_at"] >= validation_start)
        & (df["logged_at"] <= validation_end)
    ].copy()

    test_df = df.loc[
        df["logged_at"] >= test_start
    ].copy()

    train_sample = sample_frame(
        train_df,
        MAX_TRAIN_ROWS,
        RANDOM_STATE,
    )

    validation_sample = sample_frame(
        validation_df,
        MAX_VALIDATION_ROWS,
        RANDOM_STATE + 1,
    )

    test_sample = sample_frame(
        test_df,
        MAX_TEST_ROWS,
        RANDOM_STATE + 2,
    )

    print()
    print("Sample sizes:")
    print(f"  Train      : {len(train_sample):,}")
    print(f"  Validation : {len(validation_sample):,}")
    print(f"  Test       : {len(test_sample):,}")

    prediction_output = test_sample[
        [
            "id",
            "logged_at",
            "symbol",
            "timeframe",
            TARGET_COLUMN,
        ]
    ].copy()

    results = []

    for name, config in EXPERIMENTS.items():
        result = train_experiment(
            name,
            config,
            train_sample,
            validation_sample,
            test_sample,
        )

        results.append(result)

        prediction_output[
            f"{name}_probability"
        ] = result[
            "test_probabilities"
        ].astype("float32")

    prediction_output.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    summary_rows = []

    for result in results:
        best_test = result["test"][
            "validation_selected_threshold"
        ]

        top_metrics = {
            row["top_fraction"]: row
            for row in result["test"][
                "top_fraction_metrics"
            ]
        }

        summary_rows.append(
            {
                "experiment": result["experiment"],
                "feature_count": result["feature_count"],
                "best_iteration": result["best_iteration"],
                "validation_roc_auc": (
                    result["validation"]["roc_auc"]
                ),
                "validation_pr_auc": (
                    result["validation"]["pr_auc"]
                ),
                "selected_threshold": (
                    result["validation"][
                        "best_threshold"
                    ]
                ),
                "test_roc_auc": (
                    result["test"]["roc_auc"]
                ),
                "test_pr_auc": (
                    result["test"]["pr_auc"]
                ),
                "test_precision": (
                    best_test["precision"]
                ),
                "test_recall": (
                    best_test["recall"]
                ),
                "test_f1": best_test["f1"],
                "top_1pct_event_rate": (
                    top_metrics[0.01][
                        "event_rate"
                    ]
                ),
                "top_1pct_lift": (
                    top_metrics[0.01]["lift"]
                ),
                "top_5pct_event_rate": (
                    top_metrics[0.05][
                        "event_rate"
                    ]
                ),
                "top_5pct_lift": (
                    top_metrics[0.05]["lift"]
                ),
                "top_10pct_event_rate": (
                    top_metrics[0.10][
                        "event_rate"
                    ]
                ),
                "top_10pct_lift": (
                    top_metrics[0.10]["lift"]
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    serializable_results = []

    for result in results:
        serializable_results.append(
            {
                key: value
                for key, value in result.items()
                if key != "test_probabilities"
            }
        )

    report = {
        "target": TARGET_COLUMN,
        "usable_rows": int(len(df)),
        "positive_rate": float(
            df[TARGET_COLUMN].mean()
        ),
        "experiments": serializable_results,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("STRONG CONTRARIAN 25BP SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    print()
    print(f"Summary     : {SUMMARY_PATH}")
    print(f"Metrics     : {METRICS_PATH}")
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(
        f"Elapsed     : "
        f"{time.time() - started:.1f}s"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
