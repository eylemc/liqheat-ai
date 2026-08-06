from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = Path(
    "data/features/liq_topology_v2_ml_labeled.parquet"
)

OUTPUT_DIR = Path("data/models/topology_v2_baseline_1h")
MODEL_PATH = OUTPUT_DIR / "model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
FEATURES_PATH = OUTPUT_DIR / "features.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.parquet"

TARGET_COLUMN = "direction_1h"
VALID_COLUMN = "label_valid_1h"

# Splitlerin birbirine future-label etkisi taşımaması için.
EMBARGO = pd.Timedelta(hours=4)

# İlk baseline için makul ve hızlı örneklem.
# Temporal dağılım korunarak rastgele örneklenir.
MAX_TRAIN_ROWS = 800_000
MAX_VALIDATION_ROWS = 300_000
MAX_TEST_ROWS = 300_000

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "symbol",
    "timeframe",
    "nearest_side",
]

NUMERIC_FEATURES = [
    "has_upper_level",
    "has_lower_level",
    "has_topology",
    "nearest_side_code",

    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "signed_distance_edge",

    "log1p_upper_distance_pct",
    "log1p_lower_distance_pct",
    "log1p_distance_advantage",

    "topology_imbalance",
    "total_volume_imbalance_check",

    "active_level_difference",
    "active_level_total",

    "log1p_upper_pool_volume",
    "log1p_lower_pool_volume",
    "log1p_nearest_pool_volume",
    "log1p_farther_pool_volume",

    "log1p_pool_volume_ratio",
    "log1p_distance_pressure_ratio",

    "log1p_upper_active_levels",
    "log1p_lower_active_levels",
    "log1p_upper_total_volume",
    "log1p_lower_total_volume",

    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_utc",
]

FORBIDDEN_PATTERNS = [
    "future_",
    "forward_return_",
    "direction_",
    "label_valid_",
    "target_time_",
]


def json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if pd.isna(value):
        return None

    return value


def make_one_hot_encoder():
    """
    Support both newer and older scikit-learn versions.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float32,
        )


def sample_frame(
    df: pd.DataFrame,
    maximum_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if len(df) <= maximum_rows:
        return df.copy()

    return (
        df.sample(
            n=maximum_rows,
            random_state=random_state,
            replace=False,
        )
        .sort_values("logged_at")
        .reset_index(drop=True)
    )


def check_features(df: pd.DataFrame) -> None:
    selected = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    missing = [
        column
        for column in selected
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing feature columns: " + ", ".join(missing)
        )

    leaked = []

    for column in selected:
        if any(
            pattern in column
            for pattern in FORBIDDEN_PATTERNS
        ):
            leaked.append(column)

    if leaked:
        raise ValueError(
            "Potential target leakage columns selected: "
            + ", ".join(leaked)
        )


def class_counts(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False).sort_index()

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def evaluate_model(
    name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict:
    result = {
        "name": name,
        "rows": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[-1, 0, 1],
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[-1, 0, 1],
            target_names=["DOWN", "NEUTRAL", "UP"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if probabilities is not None:
        result["log_loss"] = float(
            log_loss(
                y_true,
                probabilities,
                labels=[-1, 0, 1],
            )
        )

    return result


def print_metrics(metrics: dict) -> None:
    print(
        f"{metrics['name']:<22} "
        f"accuracy={metrics['accuracy']:.4f}  "
        f"balanced={metrics['balanced_accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )

    if "log_loss" in metrics:
        print(
            f"{'':<22} "
            f"log_loss={metrics['log_loss']:.4f}"
        )

    print("Confusion matrix [DOWN, NEUTRAL, UP]:")
    print(np.array(metrics["confusion_matrix"]))


def main() -> int:
    started = time.time()

    if not INPUT_PATH.exists():
        print(
            f"ERROR: Input file not found: {INPUT_PATH}",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("TOPOLOGY V2 — 1H BASELINE MODEL")
    print("=" * 76)
    print(f"Input : {INPUT_PATH}")
    print(f"Model : {MODEL_PATH}")
    print()

    selected_columns = list(
        dict.fromkeys(
            [
                "id",
                "logged_at",
                "current_price",
                TARGET_COLUMN,
                VALID_COLUMN,
            ]
            + CATEGORICAL_FEATURES
            + NUMERIC_FEATURES
        )
    )

    print("Loading selected columns...")
    df = pd.read_parquet(
        INPUT_PATH,
        columns=selected_columns,
    )

    check_features(df)

    df = df.loc[
        df[VALID_COLUMN].eq(1)
        & df[TARGET_COLUMN].notna()
    ].copy()

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype("int8")
    )

    df = df.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No valid labeled rows remain.")

    minimum_time = df["logged_at"].min()
    maximum_time = df["logged_at"].max()

    # Kronolojik yüzde 70 / 15 / 15 sınırları.
    train_boundary = df["logged_at"].quantile(0.70)
    validation_boundary = df["logged_at"].quantile(0.85)

    # Train, validation ve test arasında embargo boşlukları.
    train_end = train_boundary - EMBARGO
    validation_start = train_boundary + EMBARGO

    validation_end = validation_boundary - EMBARGO
    test_start = validation_boundary + EMBARGO

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

    excluded_embargo_rows = (
        len(df)
        - len(train_df)
        - len(validation_df)
        - len(test_df)
    )

    if min(
        len(train_df),
        len(validation_df),
        len(test_df),
    ) == 0:
        raise RuntimeError(
            "At least one temporal split is empty."
        )

    print("Temporal range:")
    print(f"  Dataset    : {minimum_time} -> {maximum_time}")
    print(f"  Train end  : {train_end}")
    print(f"  Val start  : {validation_start}")
    print(f"  Val end    : {validation_end}")
    print(f"  Test start : {test_start}")
    print(f"  Embargo    : {EMBARGO}")
    print()

    print("Full split sizes:")
    print(f"  Train      : {len(train_df):,}")
    print(f"  Validation : {len(validation_df):,}")
    print(f"  Test       : {len(test_df):,}")
    print(f"  Embargoed  : {excluded_embargo_rows:,}")
    print()

    # İlk model için kontrollü örneklem.
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

    print("Model sample sizes:")
    print(f"  Train      : {len(train_sample):,}")
    print(f"  Validation : {len(validation_sample):,}")
    print(f"  Test       : {len(test_sample):,}")
    print()

    print("Train target distribution:")
    print(
        train_sample[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .to_string()
    )
    print()

    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    X_train = train_sample[feature_columns]
    y_train = train_sample[TARGET_COLUMN]

    X_validation = validation_sample[feature_columns]
    y_validation = validation_sample[TARGET_COLUMN]

    X_test = test_sample[feature_columns]
    y_test = test_sample[TARGET_COLUMN]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(
                    with_mean=False,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                make_one_hot_encoder(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=0.0001,
        l1_ratio=0.05,
        class_weight="balanced",
        max_iter=50,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        average=True,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    majority_class = int(
        y_train.value_counts().idxmax()
    )

    validation_majority_predictions = np.full(
        len(y_validation),
        majority_class,
        dtype=np.int8,
    )

    test_majority_predictions = np.full(
        len(y_test),
        majority_class,
        dtype=np.int8,
    )

    print("Training logistic baseline...")
    fit_started = time.time()

    model.fit(X_train, y_train)

    fit_elapsed = time.time() - fit_started

    print(
        f"Training completed in "
        f"{fit_elapsed:.1f} seconds."
    )
    print()

    validation_predictions = model.predict(
        X_validation
    )
    validation_probabilities = model.predict_proba(
        X_validation
    )

    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)

    validation_majority_metrics = evaluate_model(
        "validation_majority",
        y_validation,
        validation_majority_predictions,
    )

    validation_model_metrics = evaluate_model(
        "validation_logistic",
        y_validation,
        validation_predictions,
        validation_probabilities,
    )

    test_majority_metrics = evaluate_model(
        "test_majority",
        y_test,
        test_majority_predictions,
    )

    test_model_metrics = evaluate_model(
        "test_logistic",
        y_test,
        test_predictions,
        test_probabilities,
    )

    print("=" * 76)
    print("VALIDATION RESULTS")
    print("=" * 76)
    print_metrics(validation_majority_metrics)
    print()
    print_metrics(validation_model_metrics)

    print()
    print("=" * 76)
    print("TEST RESULTS")
    print("=" * 76)
    print_metrics(test_majority_metrics)
    print()
    print_metrics(test_model_metrics)

    model_classes = [
        int(value)
        for value in model.named_steps[
            "classifier"
        ].classes_
    ]

    prediction_output = test_sample[
        [
            "id",
            "logged_at",
            "symbol",
            "timeframe",
            "current_price",
            TARGET_COLUMN,
        ]
    ].copy()

    prediction_output["predicted_direction_1h"] = (
        test_predictions.astype("int8")
    )

    for index, class_value in enumerate(model_classes):
        prediction_output[
            f"probability_class_{class_value}"
        ] = test_probabilities[:, index].astype(
            "float32"
        )

    prediction_output.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    split_report = {
        "dataset": {
            "input": str(INPUT_PATH),
            "minimum_time": json_value(minimum_time),
            "maximum_time": json_value(maximum_time),
            "valid_labeled_rows": int(len(df)),
        },
        "target": TARGET_COLUMN,
        "embargo_hours": (
            EMBARGO.total_seconds() / 3600
        ),
        "boundaries": {
            "train_boundary_raw": json_value(
                train_boundary
            ),
            "validation_boundary_raw": json_value(
                validation_boundary
            ),
            "train_end": json_value(train_end),
            "validation_start": json_value(
                validation_start
            ),
            "validation_end": json_value(
                validation_end
            ),
            "test_start": json_value(test_start),
        },
        "full_split_rows": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
            "embargoed": int(excluded_embargo_rows),
        },
        "sample_rows": {
            "train": int(len(train_sample)),
            "validation": int(
                len(validation_sample)
            ),
            "test": int(len(test_sample)),
        },
        "class_counts": {
            "train": class_counts(y_train),
            "validation": class_counts(
                y_validation
            ),
            "test": class_counts(y_test),
        },
        "majority_class": majority_class,
        "model_classes": model_classes,
        "training_seconds": fit_elapsed,
        "metrics": {
            "validation_majority": (
                validation_majority_metrics
            ),
            "validation_logistic": (
                validation_model_metrics
            ),
            "test_majority": (
                test_majority_metrics
            ),
            "test_logistic": (
                test_model_metrics
            ),
        },
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with FEATURES_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "categorical_features": (
                    CATEGORICAL_FEATURES
                ),
                "numeric_features": (
                    NUMERIC_FEATURES
                ),
                "forbidden_patterns": (
                    FORBIDDEN_PATTERNS
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    joblib.dump(model, MODEL_PATH)

    total_elapsed = time.time() - started

    print()
    print("=" * 76)
    print("BASELINE COMPLETE")
    print("=" * 76)
    print(f"Model       : {MODEL_PATH}")
    print(f"Metrics     : {METRICS_PATH}")
    print(f"Features    : {FEATURES_PATH}")
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(f"Total time  : {total_elapsed:.1f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
