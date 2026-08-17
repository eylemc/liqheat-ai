#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_position_guardian_v5_position_state as v5  # noqa: E402
import build_position_guardian_v6_dual_head as v6  # noqa: E402

OUT = Path("data/reports/position_guardian_v6_walkforward_calibration")
SIDES = ("LONG", "SHORT")
HORIZONS = (15, 30, 60)
HEADS = {"exit": "is_exit", "hold": "is_hold"}
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


@dataclass
class CalibrationModel:
    method: str
    model: object | None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "raw" or self.model is None:
            return np.clip(p, 1e-6, 1 - 1e-6)
        if self.method == "platt":
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
            return np.clip(self.model.predict_proba(z.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 1e-6, 1 - 1e-6)
        raise ValueError(f"Unknown calibration method: {self.method}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward calibration and per-symbol threshold stability for Position Guardian V6."
    )
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--matrix-len", type=int, default=20)
    p.add_argument("--sample-every-minutes", type=int, default=15)
    p.add_argument("--entry-ages-minutes", default="15,30,60,120,240")
    p.add_argument("--feature-groups", default="position_plus_topology,position_plus_topology_compact_matrix")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--initial-train-fraction", type=float, default=0.50)
    p.add_argument("--calibration-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction-per-fold", type=float, default=0.10)
    p.add_argument("--embargo-hours", type=float, default=4)
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--max-train-rows", type=int, default=500000)
    p.add_argument("--hold-improvement-bps", type=float, default=8)
    p.add_argument("--exit-deterioration-bps", type=float, default=8)
    p.add_argument("--future-stop-bps", type=float, default=15)
    p.add_argument("--future-recovery-bps", type=float, default=12)
    p.add_argument("--min-precision", type=float, default=0.45)
    p.add_argument("--min-coverage", type=float, default=0.02)
    p.add_argument("--min-alerts", type=int, default=75)
    p.add_argument("--min-symbol-rows", type=int, default=300)
    p.add_argument("--min-symbol-positives", type=int, default=30)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def safe_metric(fn, *args, default=None, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return default


def fit_calibrator(y: np.ndarray, p: np.ndarray) -> tuple[CalibrationModel, list[dict]]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    candidates: list[tuple[CalibrationModel, float]] = []
    diagnostics: list[dict] = []

    raw = CalibrationModel("raw", None)
    raw_brier = safe_metric(brier_score_loss, y, raw.transform(p), default=np.inf)
    candidates.append((raw, raw_brier))
    diagnostics.append({"method": "raw", "brier": raw_brier})

    if len(np.unique(y)) == 2:
        z = np.log(p / (1 - p)).reshape(-1, 1)
        platt_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        platt_model.fit(z, y)
        platt = CalibrationModel("platt", platt_model)
        brier = safe_metric(brier_score_loss, y, platt.transform(p), default=np.inf)
        candidates.append((platt, brier))
        diagnostics.append({"method": "platt", "brier": brier})

        if len(y) >= 500 and int(y.sum()) >= 50 and int((1 - y).sum()) >= 50:
            iso_model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            iso_model.fit(p, y)
            iso = CalibrationModel("isotonic", iso_model)
            brier = safe_metric(brier_score_loss, y, iso.transform(p), default=np.inf)
            candidates.append((iso, brier))
            diagnostics.append({"method": "isotonic", "brier": brier})

    chosen = min(candidates, key=lambda x: x[1])[0]
    for row in diagnostics:
        row["selected"] = row["method"] == chosen.method
    return chosen, diagnostics


def choose_threshold(y: np.ndarray, p: np.ndarray, a: argparse.Namespace) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rows = []
    for threshold in np.arange(0.30, 0.951, 0.025):
        pred = p >= threshold
        alerts = int(pred.sum())
        coverage = float(pred.mean()) if len(pred) else 0.0
        precision = float(precision_score(y, pred, zero_division=0)) if alerts else 0.0
        recall = float(recall_score(y, pred, zero_division=0)) if alerts else 0.0
        passes = alerts >= a.min_alerts and coverage >= a.min_coverage and precision >= a.min_precision
        score = precision * np.sqrt(max(coverage, 1e-12)) + 0.10 * recall
        rows.append({
            "threshold": float(threshold),
            "alerts": alerts,
            "coverage": coverage,
            "precision": precision,
            "recall": recall,
            "passes_constraints": bool(passes),
            "score": float(score),
        })
    valid = [r for r in rows if r["passes_constraints"]]
    if valid:
        return {**max(valid, key=lambda r: (r["score"], r["precision"])), "fallback": False}
    eligible = [r for r in rows if r["alerts"] >= max(20, min(a.min_alerts, 50))]
    return {**max(eligible or rows, key=lambda r: (r["precision"], r["coverage"])), "fallback": True}


def evaluate_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    alerts = int(pred.sum())
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "threshold": float(threshold),
        "alerts": alerts,
        "coverage": float(pred.mean()) if len(pred) else 0.0,
        "precision": float(precision_score(y, pred, zero_division=0)) if alerts else None,
        "recall": float(recall_score(y, pred, zero_division=0)) if alerts else 0.0,
        "roc_auc": safe_metric(roc_auc_score, y, p),
        "pr_auc": safe_metric(average_precision_score, y, p),
        "brier": safe_metric(brier_score_loss, y, p),
    }


def make_walkforward_splits(df: pd.DataFrame, a: argparse.Namespace) -> list[dict]:
    unique_times = np.array(sorted(pd.to_datetime(df["logged_at"], utc=True).unique()))
    n = len(unique_times)
    train_n = max(1, int(n * a.initial_train_fraction))
    cal_n = max(1, int(n * a.calibration_fraction))
    test_n = max(1, int(n * a.test_fraction_per_fold))
    embargo = pd.Timedelta(hours=a.embargo_hours)
    folds = []

    for fold in range(a.folds):
        train_end_idx = train_n + fold * test_n - 1
        cal_start_idx = train_end_idx + 1
        cal_end_idx = cal_start_idx + cal_n - 1
        test_start_idx = cal_end_idx + 1
        test_end_idx = min(test_start_idx + test_n - 1, n - 1)
        if test_start_idx >= n or cal_end_idx >= n:
            break

        train_end = pd.Timestamp(unique_times[train_end_idx])
        cal_start = pd.Timestamp(unique_times[cal_start_idx])
        cal_end = pd.Timestamp(unique_times[cal_end_idx])
        test_start = pd.Timestamp(unique_times[test_start_idx])
        test_end = pd.Timestamp(unique_times[test_end_idx])

        train = df[df["logged_at"] <= train_end - embargo].copy()
        calibration = df[(df["logged_at"] >= cal_start) & (df["logged_at"] <= cal_end - embargo)].copy()
        test = df[(df["logged_at"] >= test_start) & (df["logged_at"] <= test_end)].copy()
        if len(train) > a.max_train_rows:
            train = train.sort_values("logged_at").tail(a.max_train_rows).copy()
        if train.empty or calibration.empty or test.empty:
            continue
        folds.append({
            "fold": fold + 1,
            "train": train,
            "calibration": calibration,
            "test": test,
            "train_end": train["logged_at"].max(),
            "calibration_start": calibration["logged_at"].min(),
            "calibration_end": calibration["logged_at"].max(),
            "test_start": test["logged_at"].min(),
            "test_end": test["logged_at"].max(),
        })
    return folds


def predict_binary(model, df: pd.DataFrame, features: list[str], cats: list[str]) -> np.ndarray:
    x, _, _ = v5.prepare_xy(df, features)
    cat_cols = [c for c in cats if c in x.columns]
    return model.predict_proba(Pool(x, cat_features=cat_cols))[:, 1]


def threshold_for_symbol(
    calibration: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    symbol: str,
    global_threshold: dict,
    a: argparse.Namespace,
) -> tuple[dict, str]:
    mask = calibration["symbol"].astype(str).to_numpy() == symbol
    ys = y[mask]
    ps = p[mask]
    if len(ys) < a.min_symbol_rows or int(ys.sum()) < a.min_symbol_positives or len(np.unique(ys)) < 2:
        return dict(global_threshold), "global_fallback"
    local = choose_threshold(ys, ps, a)
    return local, "symbol_specific"


def summarize_stability(thresholds: pd.DataFrame, metrics: pd.DataFrame) -> list[dict]:
    keys = ["side", "horizon_minutes", "feature_group", "head", "symbol"]
    rows = []
    for key, g in thresholds.groupby(keys, dropna=False):
        record = dict(zip(keys, key))
        vals = pd.to_numeric(g["threshold"], errors="coerce").dropna()
        mg = metrics.merge(g[["fold", *keys]], on=["fold", *keys], how="inner")
        precisions = pd.to_numeric(mg["test_precision"], errors="coerce").dropna()
        coverages = pd.to_numeric(mg["test_coverage"], errors="coerce").dropna()
        record.update({
            "folds": int(len(g)),
            "threshold_mean": float(vals.mean()) if len(vals) else None,
            "threshold_std": float(vals.std(ddof=0)) if len(vals) else None,
            "threshold_min": float(vals.min()) if len(vals) else None,
            "threshold_max": float(vals.max()) if len(vals) else None,
            "threshold_range": float(vals.max() - vals.min()) if len(vals) else None,
            "fallback_rate": float(g["fallback"].astype(bool).mean()),
            "symbol_specific_rate": float((g["threshold_source"] == "symbol_specific").mean()),
            "test_precision_mean": float(precisions.mean()) if len(precisions) else None,
            "test_precision_std": float(precisions.std(ddof=0)) if len(precisions) else None,
            "test_coverage_mean": float(coverages.mean()) if len(coverages) else None,
            "stable_threshold": bool(len(vals) >= 2 and vals.std(ddof=0) <= 0.05),
        })
        rows.append(record)
    return rows


def main() -> None:
    a = parse_args()
    entry_ages = sorted({int(v.strip()) for v in a.entry_ages_minutes.split(",") if v.strip()})
    selected_groups = [x.strip() for x in a.feature_groups.split(",") if x.strip()]
    unknown = [g for g in selected_groups if g not in v6.FEATURE_GROUPS]
    if unknown:
        raise SystemExit(f"Unknown feature groups: {unknown}")

    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading raw Binance OHLCV and computing Koinvizyon Matrix...")
    matrix, parity = v5.v4.load_matrix_all(a)
    print("Loading topology stream...")
    topology = v5.v4.load_topology(a)
    topology = v5.v4.add_dynamic_features(topology)
    base = v5.v4.join_matrix(topology, matrix)

    fold_metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    fold_descriptions: list[dict] = []

    for side in SIDES:
        contextual = v5.add_position_context_features(base, side)
        for horizon in HORIZONS:
            print(f"{side} {horizon}m position-state dataset...")
            dataset = v5.build_position_states(contextual, side, entry_ages, horizon, a)
            if dataset.empty:
                raise RuntimeError(f"No dataset for {side} {horizon}m")
            dataset["is_exit"] = (dataset["action_code"] == 0).astype("int8")
            dataset["is_hold"] = (dataset["action_code"] == 2).astype("int8")
            folds = make_walkforward_splits(dataset, a)
            if not folds:
                raise RuntimeError(f"No walk-forward folds for {side} {horizon}m")

            for fold_info in folds:
                fold = fold_info["fold"]
                train = fold_info["train"]
                calibration = fold_info["calibration"]
                test = fold_info["test"]
                fold_descriptions.append({
                    "side": side,
                    "horizon_minutes": horizon,
                    "fold": fold,
                    "train_rows": len(train),
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                    "train_end": fold_info["train_end"],
                    "calibration_start": fold_info["calibration_start"],
                    "calibration_end": fold_info["calibration_end"],
                    "test_start": fold_info["test_start"],
                    "test_end": fold_info["test_end"],
                })

                for group_name in selected_groups:
                    print(f"  fold {fold} {group_name}")
                    features = v6.FEATURE_GROUPS[group_name]
                    for head, target_col in HEADS.items():
                        model, used, cats = v6.fit_binary(train, calibration, features, target_col, a)
                        p_cal_raw = predict_binary(model, calibration, used, cats)
                        p_test_raw = predict_binary(model, test, used, cats)
                        y_cal = calibration[target_col].to_numpy(dtype=int)
                        y_test = test[target_col].to_numpy(dtype=int)

                        calibrator, calibration_candidates = fit_calibrator(y_cal, p_cal_raw)
                        p_cal = calibrator.transform(p_cal_raw)
                        p_test = calibrator.transform(p_test_raw)
                        global_threshold = choose_threshold(y_cal, p_cal, a)

                        for symbol in ("ALL", *SYMBOLS):
                            if symbol == "ALL":
                                cal_mask = np.ones(len(calibration), dtype=bool)
                                test_mask = np.ones(len(test), dtype=bool)
                                threshold = global_threshold
                                source = "global"
                            else:
                                cal_mask = calibration["symbol"].astype(str).to_numpy() == symbol
                                test_mask = test["symbol"].astype(str).to_numpy() == symbol
                                threshold, source = threshold_for_symbol(
                                    calibration, y_cal, p_cal, symbol, global_threshold, a
                                )
                            if not test_mask.any():
                                continue

                            cal_eval = evaluate_threshold(y_cal[cal_mask], p_cal[cal_mask], threshold["threshold"])
                            test_eval = evaluate_threshold(y_test[test_mask], p_test[test_mask], threshold["threshold"])
                            result_base = {
                                "side": side,
                                "horizon_minutes": horizon,
                                "feature_group": group_name,
                                "head": head,
                                "fold": fold,
                                "symbol": symbol,
                                "calibration_method": calibrator.method,
                            }
                            threshold_rows.append({
                                **result_base,
                                "threshold": threshold["threshold"],
                                "threshold_source": source,
                                "fallback": threshold.get("fallback", False),
                                "calibration_precision": cal_eval["precision"],
                                "calibration_coverage": cal_eval["coverage"],
                                "calibration_alerts": cal_eval["alerts"],
                            })
                            fold_metric_rows.append({
                                **result_base,
                                "train_rows": len(train),
                                "calibration_rows": int(cal_mask.sum()),
                                "test_rows": int(test_mask.sum()),
                                "raw_test_roc_auc": safe_metric(roc_auc_score, y_test[test_mask], p_test_raw[test_mask]),
                                "raw_test_pr_auc": safe_metric(average_precision_score, y_test[test_mask], p_test_raw[test_mask]),
                                "raw_test_brier": safe_metric(brier_score_loss, y_test[test_mask], p_test_raw[test_mask]),
                                "calibrated_test_roc_auc": test_eval["roc_auc"],
                                "calibrated_test_pr_auc": test_eval["pr_auc"],
                                "calibrated_test_brier": test_eval["brier"],
                                "threshold": threshold["threshold"],
                                "test_alerts": test_eval["alerts"],
                                "test_coverage": test_eval["coverage"],
                                "test_precision": test_eval["precision"],
                                "test_recall": test_eval["recall"],
                                "test_positive_rate": test_eval["positive_rate"],
                                "precision_lift_vs_base": (
                                    test_eval["precision"] / test_eval["positive_rate"]
                                    if test_eval["precision"] is not None and test_eval["positive_rate"]
                                    else None
                                ),
                                "calibration_candidates": json.dumps(json_safe(calibration_candidates)),
                            })

    thresholds_df = pd.DataFrame(threshold_rows)
    metrics_df = pd.DataFrame(fold_metric_rows)
    stability_rows = summarize_stability(thresholds_df, metrics_df)
    stability_df = pd.DataFrame(stability_rows)
    folds_df = pd.DataFrame(fold_descriptions).drop_duplicates()

    thresholds_df.to_csv(OUT / "thresholds_by_fold_symbol.csv", index=False)
    metrics_df.to_csv(OUT / "fold_metrics.csv", index=False)
    stability_df.to_csv(OUT / "threshold_stability.csv", index=False)
    folds_df.to_csv(OUT / "walkforward_folds.csv", index=False)

    summary = {
        "status": "research_only",
        "analysis": "walk_forward_calibration_and_per_symbol_threshold_stability",
        "matrix_parity": parity,
        "settings": vars(a),
        "folds": json_safe(fold_descriptions),
        "threshold_stability": json_safe(stability_rows),
        "headline": {
            "stable_threshold_groups": int(stability_df["stable_threshold"].sum()) if not stability_df.empty else 0,
            "total_threshold_groups": int(len(stability_df)),
            "mean_fallback_rate": float(stability_df["fallback_rate"].mean()) if not stability_df.empty else None,
            "mean_test_precision": float(pd.to_numeric(metrics_df["test_precision"], errors="coerce").mean()) if not metrics_df.empty else None,
            "mean_precision_lift_vs_base": float(pd.to_numeric(metrics_df["precision_lift_vs_base"], errors="coerce").mean()) if not metrics_df.empty else None,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"Done: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
