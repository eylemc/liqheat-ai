#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss

FEATURE_PATH = Path("data/features/liq_topology_v2_ml_features.parquet")
OUTPUT_ROOT = Path("data/models/position_guardian_v1")
REPORT_ROOT = Path("data/reports/position_guardian_v1")

HORIZONS_MINUTES = (15, 30, 60)
SIDES = ("LONG", "SHORT")
ACTIONS = ("EXIT", "REDUCE", "HOLD")
ACTION_TO_CODE = {"EXIT": 0, "REDUCE": 1, "HOLD": 2}
CODE_TO_ACTION = {v: k for k, v in ACTION_TO_CODE.items()}

CATEGORICAL_CANDIDATES = ["symbol", "timeframe", "nearest_side"]
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend_utc"]
TOPOLOGY_FEATURES = [
    "current_price",
    "has_upper_level", "has_lower_level", "has_topology", "nearest_side_code",
    "upper_distance_pct", "lower_distance_pct", "distance_advantage", "signed_distance_edge",
    "log1p_upper_distance_pct", "log1p_lower_distance_pct", "log1p_distance_advantage",
    "upper_pool_volume", "lower_pool_volume", "nearest_pool_volume", "farther_pool_volume",
    "upper_total_volume", "lower_total_volume",
    "upper_active_levels", "lower_active_levels",
    "log1p_upper_pool_volume", "log1p_lower_pool_volume",
    "log1p_nearest_pool_volume", "log1p_farther_pool_volume",
    "log1p_upper_total_volume", "log1p_lower_total_volume",
    "log1p_upper_active_levels", "log1p_lower_active_levels",
    "pool_volume_ratio", "log1p_pool_volume_ratio",
    "distance_pressure_ratio", "log1p_distance_pressure_ratio",
    "topology_imbalance", "total_volume_imbalance_check",
    "active_level_difference", "active_level_total",
]

FORBIDDEN_FEATURE_SUBSTRINGS = (
    "forward_", "future_", "target_", "label_", "sweep_code_", "first_hit_",
    "post_hit_", "strong_contrarian_", "direction_1h", "direction_4h",
)

BASE_COLUMNS = ["id", "logged_at", "symbol", "timeframe", "current_price"]


@dataclass(frozen=True)
class LabelConfig:
    hold_take_bps: float
    exit_stop_bps: float
    reduce_drawdown_bps: float
    endpoint_hold_bps: float
    endpoint_exit_bps: float


@dataclass(frozen=True)
class SplitConfig:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and train Liqheat Position Guardian v1")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--sample-every-minutes", type=int, default=15)
    p.add_argument("--max-train-rows", type=int, default=500_000)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.06)
    p.add_argument("--embargo-hours", type=float, default=4.0)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--hold-take-bps", type=float, default=25.0)
    p.add_argument("--exit-stop-bps", type=float, default=15.0)
    p.add_argument("--reduce-drawdown-bps", type=float, default=8.0)
    p.add_argument("--endpoint-hold-bps", type=float, default=8.0)
    p.add_argument("--endpoint-exit-bps", type=float, default=-8.0)
    p.add_argument("--minimum-confidence", type=float, default=0.55)
    p.add_argument("--minimum-coverage", type=float, default=0.05)
    return p.parse_args()


def ensure_paths() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(FEATURE_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def schema_columns() -> list[str]:
    return pq.read_schema(FEATURE_PATH).names


def feature_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    available = set(columns)
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in available]
    numeric = [c for c in CALENDAR_FEATURES + TOPOLOGY_FEATURES if c in available]
    features = list(dict.fromkeys(categorical + numeric))
    forbidden = [c for c in features if any(x in c for x in FORBIDDEN_FEATURE_SUBSTRINGS)]
    if forbidden:
        raise RuntimeError(f"Forbidden future-derived features selected: {forbidden}")
    if not features:
        raise RuntimeError("No model features found")
    return features, categorical


def load_stream(timeframe: str, features: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(BASE_COLUMNS + features))
    df = pd.read_parquet(
        FEATURE_PATH,
        columns=columns,
        filters=[("timeframe", "==", timeframe)],
    )
    if df.empty:
        raise RuntimeError(f"No rows found for timeframe={timeframe}")
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True).astype("datetime64[ns, UTC]")
    df["symbol"] = df["symbol"].astype("string")
    df = (
        df.sort_values(["symbol", "logged_at", "id"], kind="mergesort")
        .drop_duplicates("id")
        .reset_index(drop=True)
    )
    return df


def first_touch_action(
    future_prices: np.ndarray,
    current_price: float,
    side: str,
    cfg: LabelConfig,
) -> tuple[int, dict[str, float | int | None]]:
    if len(future_prices) == 0 or not math.isfinite(current_price) or current_price <= 0:
        return ACTION_TO_CODE["REDUCE"], {
            "mfe_bps": None, "mae_bps": None, "endpoint_bps": None,
            "tp_index": None, "sl_index": None,
        }

    direction = 1.0 if side == "LONG" else -1.0
    directional = direction * (future_prices / current_price - 1.0) * 10_000.0
    directional = directional[np.isfinite(directional)]
    if len(directional) == 0:
        return ACTION_TO_CODE["REDUCE"], {
            "mfe_bps": None, "mae_bps": None, "endpoint_bps": None,
            "tp_index": None, "sl_index": None,
        }

    mfe = float(np.max(directional))
    mae = float(np.min(directional))
    endpoint = float(directional[-1])

    tp_hits = np.flatnonzero(directional >= cfg.hold_take_bps)
    sl_hits = np.flatnonzero(directional <= -cfg.exit_stop_bps)
    tp_index = int(tp_hits[0]) if len(tp_hits) else None
    sl_index = int(sl_hits[0]) if len(sl_hits) else None

    if sl_index is not None and (tp_index is None or sl_index < tp_index):
        action = "EXIT"
    elif tp_index is not None and (sl_index is None or tp_index < sl_index):
        action = "HOLD"
    elif endpoint <= cfg.endpoint_exit_bps or mae <= -cfg.exit_stop_bps:
        action = "EXIT"
    elif endpoint >= cfg.endpoint_hold_bps and mae > -cfg.reduce_drawdown_bps:
        action = "HOLD"
    else:
        action = "REDUCE"

    return ACTION_TO_CODE[action], {
        "mfe_bps": mfe,
        "mae_bps": mae,
        "endpoint_bps": endpoint,
        "tp_index": tp_index,
        "sl_index": sl_index,
    }


def build_labels(
    stream: pd.DataFrame,
    horizon_minutes: int,
    side: str,
    cfg: LabelConfig,
) -> pd.DataFrame:
    horizon_ns = int(pd.Timedelta(minutes=horizon_minutes).value)
    records: list[dict] = []

    for symbol, group in stream.groupby("symbol", sort=False, observed=True):
        group = group.sort_values("logged_at", kind="mergesort")
        times = group["logged_at"].astype("int64").to_numpy()
        prices = pd.to_numeric(group["current_price"], errors="coerce").to_numpy(float)
        ids = group["id"].to_numpy()

        for i in range(len(group)):
            end = int(np.searchsorted(times, times[i] + horizon_ns, side="right"))
            if end <= i + 1:
                continue
            action, diag = first_touch_action(prices[i + 1:end], float(prices[i]), side, cfg)
            records.append({
                "id": ids[i],
                "symbol": str(symbol),
                "logged_at": pd.Timestamp(times[i], unit="ns", tz="UTC"),
                "side": side,
                "horizon_minutes": horizon_minutes,
                "target_action": action,
                **diag,
            })

    return pd.DataFrame.from_records(records)


def non_overlapping_sample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    bucket = df["logged_at"].dt.floor(f"{minutes}min")
    return (
        df.assign(_bucket=bucket)
        .sort_values(["symbol", "logged_at"], kind="mergesort")
        .drop_duplicates(["symbol", "_bucket"])
        .drop(columns="_bucket")
        .reset_index(drop=True)
    )


def make_split(times: pd.Series, validation_fraction: float, test_fraction: float, embargo_hours: float) -> SplitConfig:
    q_test = times.quantile(1.0 - test_fraction)
    pretest = times[times < q_test]
    relative_val = validation_fraction / max(1e-9, 1.0 - test_fraction)
    q_val = pretest.quantile(1.0 - relative_val)
    embargo = pd.Timedelta(hours=embargo_hours)
    return SplitConfig(
        train_end=q_val - embargo,
        validation_start=q_val + embargo,
        validation_end=q_test - embargo,
        test_start=q_test + embargo,
        test_end=times.max(),
    )


def prepare_x(df: pd.DataFrame, features: list[str], categorical: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = df[features].copy()
    cats = []
    for col in features:
        if col in categorical:
            x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
            cats.append(col)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    return x, cats


def temporal_downsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    idx = np.linspace(0, len(df) - 1, max_rows, dtype=int)
    return df.iloc[idx].copy()


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    args: argparse.Namespace,
) -> CatBoostClassifier:
    x_train, cats = prepare_x(train, features, categorical)
    x_val, _ = prepare_x(validation, features, categorical)
    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(x_train, label=train["target_action"], cat_features=cats),
        eval_set=Pool(x_val, label=validation["target_action"], cat_features=cats),
        early_stopping_rounds=60,
        verbose=False,
    )
    return model


def aligned_probabilities(model: CatBoostClassifier, df: pd.DataFrame, features: list[str], categorical: list[str]) -> np.ndarray:
    x, cats = prepare_x(df, features, categorical)
    raw = model.predict_proba(Pool(x, cat_features=cats))
    out = np.zeros((len(df), 3), dtype=float)
    mapping = {int(c): i for i, c in enumerate(model.classes_)}
    for code in range(3):
        if code in mapping:
            out[:, code] = raw[:, mapping[code]]
    out = np.clip(out, 1e-12, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = np.argmax(p, axis=1)
    return {
        "rows": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def selective_policy_metrics(y: np.ndarray, p: np.ndarray, minimum_confidence: float) -> dict:
    pred = np.argmax(p, axis=1)
    confidence = np.max(p, axis=1)
    mask = confidence >= minimum_confidence
    if not mask.any():
        return {"coverage": 0.0, "rows": 0, "balanced_accuracy": None, "macro_f1": None}
    return {
        "coverage": float(mask.mean()),
        "rows": int(mask.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred[mask])),
        "macro_f1": float(f1_score(y[mask], pred[mask], average="macro", labels=[0, 1, 2], zero_division=0)),
    }


def choose_confidence(validation_y: np.ndarray, validation_p: np.ndarray, minimum_coverage: float) -> dict:
    candidates = []
    for threshold in np.arange(0.40, 0.91, 0.05):
        m = selective_policy_metrics(validation_y, validation_p, float(threshold))
        score = -1.0
        if m["rows"] and m["coverage"] >= minimum_coverage:
            score = float(m["balanced_accuracy"] or 0.0) * math.sqrt(float(m["coverage"]))
        candidates.append({"threshold": float(round(threshold, 2)), "score": score, **m})
    valid = [x for x in candidates if x["score"] >= 0]
    selected = max(valid, key=lambda x: x["score"]) if valid else candidates[0]
    return {"selected": selected, "grid": candidates}


def majority_baseline(train_y: pd.Series, test_y: np.ndarray) -> dict:
    majority = int(train_y.value_counts().idxmax())
    pred = np.full(len(test_y), majority, dtype=int)
    p = np.full((len(test_y), 3), 1e-12, dtype=float)
    p[:, majority] = 1.0 - 2e-12
    return {"majority_class": CODE_TO_ACTION[majority], **classification_metrics(test_y, p)}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_paths()
    started = time.time()

    columns = schema_columns()
    features, categorical = feature_columns(columns)
    print(f"Loading {args.timeframe} stream with {len(features)} model features...")
    stream = load_stream(args.timeframe, features)
    print(f"Full stream rows: {len(stream):,}")

    label_cfg = LabelConfig(
        hold_take_bps=args.hold_take_bps,
        exit_stop_bps=args.exit_stop_bps,
        reduce_drawdown_bps=args.reduce_drawdown_bps,
        endpoint_hold_bps=args.endpoint_hold_bps,
        endpoint_exit_bps=args.endpoint_exit_bps,
    )

    all_reports: list[dict] = []

    for side in SIDES:
        for horizon in HORIZONS_MINUTES:
            print(f"\n=== {side} / {horizon}m ===")
            labels = build_labels(stream, horizon, side, label_cfg)
            dataset = stream.merge(labels, on=["id", "symbol", "logged_at"], how="inner", validate="one_to_one")
            dataset = non_overlapping_sample(dataset, max(args.sample_every_minutes, horizon))
            dataset = dataset.sort_values("logged_at", kind="mergesort").reset_index(drop=True)

            split = make_split(dataset["logged_at"], args.validation_fraction, args.test_fraction, args.embargo_hours)
            train = dataset[dataset["logged_at"] <= split.train_end].copy()
            validation = dataset[(dataset["logged_at"] >= split.validation_start) & (dataset["logged_at"] <= split.validation_end)].copy()
            test = dataset[(dataset["logged_at"] >= split.test_start) & (dataset["logged_at"] <= split.test_end)].copy()

            train = temporal_downsample(train, args.max_train_rows)
            if min(len(train), len(validation), len(test)) == 0:
                raise RuntimeError(f"Empty split for {side}/{horizon}m")

            print(f"Train={len(train):,} Val={len(validation):,} Test={len(test):,}")
            model = train_model(train, validation, features, categorical, args)
            p_val = aligned_probabilities(model, validation, features, categorical)
            confidence_choice = choose_confidence(validation["target_action"].to_numpy(), p_val, args.minimum_coverage)

            # Refit on train+validation using the best iteration learned above, preserving the untouched test.
            refit_rows = pd.concat([train, validation], ignore_index=True).sort_values("logged_at")
            refit_rows = temporal_downsample(refit_rows, args.max_train_rows)
            refit_iterations = int(getattr(model, "best_iteration_", args.iterations) or args.iterations) + 1
            refit_args = argparse.Namespace(**vars(args))
            refit_args.iterations = max(50, refit_iterations)
            final_model = train_model(refit_rows, validation.tail(max(50, min(500, len(validation)))), features, categorical, refit_args)
            p_test = aligned_probabilities(final_model, test, features, categorical)

            test_metrics = classification_metrics(test["target_action"].to_numpy(), p_test)
            selected_conf = float(confidence_choice["selected"]["threshold"])
            selective = selective_policy_metrics(test["target_action"].to_numpy(), p_test, selected_conf)
            baseline = majority_baseline(train["target_action"], test["target_action"].to_numpy())

            pred = np.argmax(p_test, axis=1)
            confidence = np.max(p_test, axis=1)
            prediction_frame = test[["id", "logged_at", "symbol", "current_price", "target_action", "mfe_bps", "mae_bps", "endpoint_bps"]].copy()
            prediction_frame["predicted_action"] = pred
            prediction_frame["predicted_action_name"] = [CODE_TO_ACTION[int(x)] for x in pred]
            prediction_frame["target_action_name"] = [CODE_TO_ACTION[int(x)] for x in prediction_frame["target_action"]]
            prediction_frame["confidence"] = confidence
            prediction_frame["p_exit"] = p_test[:, 0]
            prediction_frame["p_reduce"] = p_test[:, 1]
            prediction_frame["p_hold"] = p_test[:, 2]

            slug = f"{side.lower()}_{horizon}m"
            model_dir = OUTPUT_ROOT / slug
            model_dir.mkdir(parents=True, exist_ok=True)
            final_model.save_model(str(model_dir / "model.cbm"))
            write_json(model_dir / "features.json", {"features": features, "categorical": categorical})
            prediction_frame.to_parquet(REPORT_ROOT / f"{slug}_test_predictions.parquet", index=False)

            report = {
                "side": side,
                "horizon_minutes": horizon,
                "label_config": label_cfg.__dict__,
                "split": split.__dict__,
                "counts": {
                    "dataset": int(len(dataset)), "train": int(len(train)),
                    "validation": int(len(validation)), "test": int(len(test)),
                    "train_classes": train["target_action"].value_counts().sort_index().to_dict(),
                    "validation_classes": validation["target_action"].value_counts().sort_index().to_dict(),
                    "test_classes": test["target_action"].value_counts().sort_index().to_dict(),
                },
                "test_metrics": test_metrics,
                "selective_policy": {"selected_confidence": selected_conf, "test": selective, "validation_grid": confidence_choice["grid"]},
                "majority_baseline": baseline,
                "increment_vs_majority_balanced_accuracy": float(test_metrics["balanced_accuracy"] - baseline["balanced_accuracy"]),
                "model_path": str(model_dir / "model.cbm"),
                "predictions_path": str(REPORT_ROOT / f"{slug}_test_predictions.parquet"),
            }
            write_json(REPORT_ROOT / f"{slug}_report.json", report)
            all_reports.append(report)
            print(json.dumps({"test": test_metrics, "selective": selective, "baseline": baseline["balanced_accuracy"]}, indent=2))

    summary = {
        "status": "research_only",
        "action_codes": CODE_TO_ACTION,
        "feature_count": len(features),
        "features": features,
        "reports": all_reports,
        "runtime_seconds": time.time() - started,
    }
    write_json(REPORT_ROOT / "summary.json", summary)
    print(f"\nDone: {REPORT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
