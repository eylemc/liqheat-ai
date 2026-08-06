from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


FEATURE_PATH = Path(
    "data/features/liq_topology_v2_ml_features.parquet"
)

LABEL_PATH = Path(
    "data/features/liq_topology_v2_strong_contrarian_labels.parquet"
)

OUTPUT_DIR = Path(
    "data/models/topology_v2_strong_contrarian_25bp_walk_forward"
)

SUMMARY_PATH = OUTPUT_DIR / "walk_forward_summary.csv"
GROUP_PATH = OUTPUT_DIR / "walk_forward_group_metrics.csv"
METRICS_PATH = OUTPUT_DIR / "walk_forward_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "walk_forward_predictions.parquet"

TARGET_COLUMN = "strong_contrarian_25bp_1h"

EMBARGO = pd.Timedelta(hours=4)

# Fold başına kontrollü örneklem.
MAX_TRAIN_ROWS = 1_200_000
MAX_VALIDATION_ROWS = 250_000
MAX_TEST_ROWS = 250_000

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

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


def sample_frame(
    df: pd.DataFrame,
    maximum_rows: int,
    seed: int,
) -> pd.DataFrame:
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


def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
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


def safe_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def safe_pr_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float | None:
    if y_true.sum() == 0:
        return None

    return float(
        average_precision_score(
            y_true,
            probabilities,
        )
    )


def top_fraction_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    order = np.argsort(-probabilities)
    base_rate = float(np.mean(y_true))

    output = []

    for fraction in [
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
    ]:
        selected_rows = max(
            1,
            int(len(y_true) * fraction),
        )

        selected = order[:selected_rows]

        event_rate = float(
            np.mean(y_true[selected])
        )

        lift = (
            event_rate / base_rate
            if base_rate > 0
            else None
        )

        output.append(
            {
                "fraction": fraction,
                "rows": selected_rows,
                "event_rate": event_rate,
                "base_rate": base_rate,
                "lift": lift,
            }
        )

    return output


def calculate_group_metrics(
    predictions: pd.DataFrame,
    fold_name: str,
    group_type: str,
    group_columns: list[str],
) -> list[dict]:
    rows = []

    grouped = predictions.groupby(
        group_columns,
        observed=True,
        dropna=False,
    )

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        y_true = group[TARGET_COLUMN].to_numpy(
            dtype=np.int8
        )

        probabilities = group[
            "probability"
        ].to_numpy(dtype=np.float64)

        base_rate = float(
            np.mean(y_true)
        )

        top_metrics = top_fraction_metrics(
            y_true,
            probabilities,
        )

        top_map = {
            item["fraction"]: item
            for item in top_metrics
        }

        rows.append(
            {
                "fold": fold_name,
                "group_type": group_type,
                "group_value": " / ".join(
                    str(value)
                    for value in group_key
                ),
                "rows": int(len(group)),
                "positives": int(y_true.sum()),
                "base_rate": base_rate,
                "roc_auc": safe_auc(
                    y_true,
                    probabilities,
                ),
                "pr_auc": safe_pr_auc(
                    y_true,
                    probabilities,
                ),
                "top_1pct_event_rate": (
                    top_map[0.01]["event_rate"]
                ),
                "top_1pct_lift": (
                    top_map[0.01]["lift"]
                ),
                "top_5pct_event_rate": (
                    top_map[0.05]["event_rate"]
                ),
                "top_5pct_lift": (
                    top_map[0.05]["lift"]
                ),
                "top_10pct_event_rate": (
                    top_map[0.10]["event_rate"]
                ),
                "top_10pct_lift": (
                    top_map[0.10]["lift"]
                ),
            }
        )

    return rows


def make_folds(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Expanding-window walk-forward folds.

    Quantile boundaries are used so each fold has enough rows even
    when symbol/timeframe density changes over time.
    """

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


def train_fold(
    df: pd.DataFrame,
    fold: dict,
    fold_index: int,
) -> dict:
    fold_name = fold["name"]

    train_df = df.loc[
        df["logged_at"] <= fold["train_end"]
    ].copy()

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

    if min(
        len(train_df),
        len(validation_df),
        len(test_df),
    ) == 0:
        raise RuntimeError(
            f"{fold_name}: empty temporal split."
        )

    train_sample = sample_frame(
        train_df,
        MAX_TRAIN_ROWS,
        RANDOM_STATE + fold_index * 10,
    )

    validation_sample = sample_frame(
        validation_df,
        MAX_VALIDATION_ROWS,
        RANDOM_STATE + fold_index * 10 + 1,
    )

    test_sample = sample_frame(
        test_df,
        MAX_TEST_ROWS,
        RANDOM_STATE + fold_index * 10 + 2,
    )

    print()
    print("=" * 82)
    print(f"WALK-FORWARD {fold_name.upper()}")
    print("=" * 82)

    print("Date ranges:")
    print(
        f"  Train      : "
        f"{train_sample['logged_at'].min()} "
        f"-> {train_sample['logged_at'].max()}"
    )
    print(
        f"  Validation : "
        f"{validation_sample['logged_at'].min()} "
        f"-> {validation_sample['logged_at'].max()}"
    )
    print(
        f"  Test       : "
        f"{test_sample['logged_at'].min()} "
        f"-> {test_sample['logged_at'].max()}"
    )

    print()
    print("Sample rows:")
    print(
        f"  Train      : {len(train_sample):,}"
    )
    print(
        f"  Validation : "
        f"{len(validation_sample):,}"
    )
    print(
        f"  Test       : {len(test_sample):,}"
    )

    print()
    print("Positive rates:")
    print(
        f"  Train      : "
        f"{train_sample[TARGET_COLUMN].mean():.4f}"
    )
    print(
        f"  Validation : "
        f"{validation_sample[TARGET_COLUMN].mean():.4f}"
    )
    print(
        f"  Test       : "
        f"{test_sample[TARGET_COLUMN].mean():.4f}"
    )

    X_train = prepare_features(train_sample)
    y_train = train_sample[TARGET_COLUMN]

    X_validation = prepare_features(
        validation_sample
    )
    y_validation = validation_sample[
        TARGET_COLUMN
    ]

    X_test = prepare_features(test_sample)
    y_test = test_sample[TARGET_COLUMN]

    categorical_indices = [
        FEATURE_COLUMNS.index(column)
        for column in CATEGORICAL_FEATURES
    ]

    train_pool = Pool(
        X_train,
        label=y_train,
        cat_features=categorical_indices,
        feature_names=FEATURE_COLUMNS,
    )

    validation_pool = Pool(
        X_validation,
        label=y_validation,
        cat_features=categorical_indices,
        feature_names=FEATURE_COLUMNS,
    )

    test_pool = Pool(
        X_test,
        label=y_test,
        cat_features=categorical_indices,
        feature_names=FEATURE_COLUMNS,
    )

    model = CatBoostClassifier(
        iterations=1500,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=(
            RANDOM_STATE + fold_index
        ),
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
        model.predict_proba(
            validation_pool
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            test_pool
        )[:, 1]
    )

    y_validation_array = (
        y_validation.to_numpy(dtype=np.int8)
    )

    y_test_array = (
        y_test.to_numpy(dtype=np.int8)
    )

    validation_roc_auc = safe_auc(
        y_validation_array,
        validation_probabilities,
    )

    validation_pr_auc = safe_pr_auc(
        y_validation_array,
        validation_probabilities,
    )

    test_roc_auc = safe_auc(
        y_test_array,
        test_probabilities,
    )

    test_pr_auc = safe_pr_auc(
        y_test_array,
        test_probabilities,
    )

    test_log_loss = float(
        log_loss(
            y_test_array,
            test_probabilities,
            labels=[0, 1],
        )
    )

    top_metrics = top_fraction_metrics(
        y_test_array,
        test_probabilities,
    )

    top_map = {
        item["fraction"]: item
        for item in top_metrics
    }

    prediction_output = test_sample[
        [
            "id",
            "logged_at",
            "symbol",
            "timeframe",
            TARGET_COLUMN,
        ]
    ].copy()

    prediction_output["fold"] = fold_name
    prediction_output["probability"] = (
        test_probabilities.astype("float32")
    )

    model_path = (
        OUTPUT_DIR / f"{fold_name}.cbm"
    )

    importance_path = (
        OUTPUT_DIR
        / f"{fold_name}_feature_importance.csv"
    )

    model.save_model(str(model_path))

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": (
                model.get_feature_importance(
                    train_pool,
                    type="FeatureImportance",
                )
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    feature_importance.to_csv(
        importance_path,
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
        f"  ROC AUC={validation_roc_auc:.4f}  "
        f"PR AUC={validation_pr_auc:.4f}"
    )

    print()
    print("Test:")
    print(
        f"  base rate={y_test_array.mean():.4f}  "
        f"ROC AUC={test_roc_auc:.4f}  "
        f"PR AUC={test_pr_auc:.4f}  "
        f"log loss={test_log_loss:.4f}"
    )

    print()
    print("Top buckets:")

    for fraction in [
        0.01,
        0.05,
        0.10,
    ]:
        item = top_map[fraction]

        print(
            f"  top {fraction * 100:>4.0f}% "
            f"event_rate={item['event_rate']:.4f} "
            f"lift={item['lift']:.2f}x"
        )

    print()
    print("Top 10 features:")
    print(
        feature_importance
        .head(10)
        .to_string(index=False)
    )

    fold_summary = {
        "fold": fold_name,
        "train_start": (
            train_sample["logged_at"]
            .min()
            .isoformat()
        ),
        "train_end": (
            train_sample["logged_at"]
            .max()
            .isoformat()
        ),
        "validation_start": (
            validation_sample["logged_at"]
            .min()
            .isoformat()
        ),
        "validation_end": (
            validation_sample["logged_at"]
            .max()
            .isoformat()
        ),
        "test_start": (
            test_sample["logged_at"]
            .min()
            .isoformat()
        ),
        "test_end": (
            test_sample["logged_at"]
            .max()
            .isoformat()
        ),
        "train_rows": int(len(train_sample)),
        "validation_rows": int(
            len(validation_sample)
        ),
        "test_rows": int(len(test_sample)),
        "train_base_rate": float(
            y_train.mean()
        ),
        "validation_base_rate": float(
            y_validation.mean()
        ),
        "test_base_rate": float(
            y_test.mean()
        ),
        "best_iteration": int(
            model.get_best_iteration()
        ),
        "training_seconds": training_seconds,
        "validation_roc_auc": (
            validation_roc_auc
        ),
        "validation_pr_auc": (
            validation_pr_auc
        ),
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_log_loss": test_log_loss,
        "top_1pct_event_rate": (
            top_map[0.01]["event_rate"]
        ),
        "top_1pct_lift": (
            top_map[0.01]["lift"]
        ),
        "top_5pct_event_rate": (
            top_map[0.05]["event_rate"]
        ),
        "top_5pct_lift": (
            top_map[0.05]["lift"]
        ),
        "top_10pct_event_rate": (
            top_map[0.10]["event_rate"]
        ),
        "top_10pct_lift": (
            top_map[0.10]["lift"]
        ),
        "model_path": str(model_path),
        "importance_path": str(
            importance_path
        ),
    }

    group_rows = []

    group_rows.extend(
        calculate_group_metrics(
            prediction_output,
            fold_name,
            "symbol",
            ["symbol"],
        )
    )

    group_rows.extend(
        calculate_group_metrics(
            prediction_output,
            fold_name,
            "timeframe",
            ["timeframe"],
        )
    )

    group_rows.extend(
        calculate_group_metrics(
            prediction_output,
            fold_name,
            "symbol_timeframe",
            ["symbol", "timeframe"],
        )
    )

    return {
        "summary": fold_summary,
        "predictions": prediction_output,
        "group_metrics": group_rows,
        "feature_importance": (
            feature_importance
            .head(20)
            .to_dict(orient="records")
        ),
    }


def main() -> int:
    started = time.time()

    for path in [
        FEATURE_PATH,
        LABEL_PATH,
    ]:
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

    print("=" * 82)
    print(
        "TOPOLOGY V2 — STRONG CONTRARIAN "
        "25BP WALK-FORWARD"
    )
    print("=" * 82)

    selected_columns = list(
        dict.fromkeys(
            [
                "id",
                "logged_at",
                "symbol",
                "timeframe",
                "nearest_side",
            ]
            + FEATURE_COLUMNS
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

    if features["id"].duplicated().any():
        raise ValueError(
            "Duplicate IDs in feature data."
        )

    if labels["id"].duplicated().any():
        raise ValueError(
            "Duplicate IDs in label data."
        )

    df = features.merge(
        labels,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(features):
        raise RuntimeError(
            f"Merge row mismatch: "
            f"{len(features):,} -> {len(df):,}"
        )

    df = df.loc[
        df[TARGET_COLUMN].isin([0, 1])
    ].copy()

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype("int8")
    )

    df = df.sort_values(
        ["logged_at", "id"],
        kind="mergesort",
    ).reset_index(drop=True)

    print()
    print(f"Usable rows : {len(df):,}")
    print(
        f"Date range  : "
        f"{df['logged_at'].min()} "
        f"-> {df['logged_at'].max()}"
    )
    print(
        f"Base rate   : "
        f"{df[TARGET_COLUMN].mean():.4f}"
    )

    folds = make_folds(df)

    summaries = []
    predictions = []
    group_metrics = []
    detailed_results = []

    for fold_index, fold in enumerate(
        folds,
        start=1,
    ):
        result = train_fold(
            df=df,
            fold=fold,
            fold_index=fold_index,
        )

        summaries.append(
            result["summary"]
        )

        predictions.append(
            result["predictions"]
        )

        group_metrics.extend(
            result["group_metrics"]
        )

        detailed_results.append(
            {
                "summary": result["summary"],
                "top_feature_importance": (
                    result["feature_importance"]
                ),
            }
        )

    summary_df = pd.DataFrame(summaries)

    prediction_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    group_df = pd.DataFrame(
        group_metrics
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    prediction_df.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    group_df.to_csv(
        GROUP_PATH,
        index=False,
    )

    aggregate = {
        "mean_test_roc_auc": float(
            summary_df["test_roc_auc"].mean()
        ),
        "min_test_roc_auc": float(
            summary_df["test_roc_auc"].min()
        ),
        "mean_test_pr_auc": float(
            summary_df["test_pr_auc"].mean()
        ),
        "mean_top_1pct_lift": float(
            summary_df[
                "top_1pct_lift"
            ].mean()
        ),
        "min_top_1pct_lift": float(
            summary_df[
                "top_1pct_lift"
            ].min()
        ),
        "mean_top_5pct_lift": float(
            summary_df[
                "top_5pct_lift"
            ].mean()
        ),
        "min_top_5pct_lift": float(
            summary_df[
                "top_5pct_lift"
            ].min()
        ),
        "mean_top_10pct_lift": float(
            summary_df[
                "top_10pct_lift"
            ].mean()
        ),
        "min_top_10pct_lift": float(
            summary_df[
                "top_10pct_lift"
            ].min()
        ),
    }

    report = {
        "target": TARGET_COLUMN,
        "usable_rows": int(len(df)),
        "date_range": {
            "start": (
                df["logged_at"]
                .min()
                .isoformat()
            ),
            "end": (
                df["logged_at"]
                .max()
                .isoformat()
            ),
        },
        "folds": detailed_results,
        "aggregate": aggregate,
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
    print("=" * 82)
    print("WALK-FORWARD SUMMARY")
    print("=" * 82)

    print(
        summary_df[
            [
                "fold",
                "test_start",
                "test_end",
                "test_base_rate",
                "test_roc_auc",
                "test_pr_auc",
                "top_1pct_event_rate",
                "top_1pct_lift",
                "top_5pct_event_rate",
                "top_5pct_lift",
                "top_10pct_event_rate",
                "top_10pct_lift",
            ]
        ].to_string(index=False)
    )

    print()
    print("Aggregate:")
    print(
        f"  Mean test ROC AUC  : "
        f"{aggregate['mean_test_roc_auc']:.4f}"
    )
    print(
        f"  Min test ROC AUC   : "
        f"{aggregate['min_test_roc_auc']:.4f}"
    )
    print(
        f"  Mean top 1% lift   : "
        f"{aggregate['mean_top_1pct_lift']:.2f}x"
    )
    print(
        f"  Min top 1% lift    : "
        f"{aggregate['min_top_1pct_lift']:.2f}x"
    )
    print(
        f"  Mean top 5% lift   : "
        f"{aggregate['mean_top_5pct_lift']:.2f}x"
    )
    print(
        f"  Min top 5% lift    : "
        f"{aggregate['min_top_5pct_lift']:.2f}x"
    )
    print(
        f"  Mean top 10% lift  : "
        f"{aggregate['mean_top_10pct_lift']:.2f}x"
    )
    print(
        f"  Min top 10% lift   : "
        f"{aggregate['min_top_10pct_lift']:.2f}x"
    )

    print()
    print("=" * 82)
    print("WALK-FORWARD COMPLETE")
    print("=" * 82)
    print(f"Summary     : {SUMMARY_PATH}")
    print(f"Groups      : {GROUP_PATH}")
    print(f"Metrics     : {METRICS_PATH}")
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(
        f"Elapsed     : "
        f"{time.time() - started:.1f}s"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
