#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

FEATURE_PATH = Path("data/features/liq_topology_v2_ml_features.parquet")
MODEL_ROOT = Path("data/models/position_guardian_v2")
REPORT_ROOT = Path("data/reports/position_guardian_v2")

SIDES = ("LONG", "SHORT")
HORIZONS = (15, 30, 60)

CATEGORICAL = ["symbol", "timeframe", "nearest_side"]
CALENDAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend_utc"]
TOPOLOGY = [
    "current_price",
    "has_upper_level", "has_lower_level", "has_topology",
    "nearest_side_code",
    "upper_distance_pct", "lower_distance_pct", "distance_advantage",
    "signed_distance_edge",
    "log1p_upper_distance_pct", "log1p_lower_distance_pct",
    "log1p_distance_advantage",
    "upper_pool_volume", "lower_pool_volume",
    "nearest_pool_volume", "farther_pool_volume",
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
BASE_COLS = ["id", "logged_at", "symbol", "timeframe", "current_price"]
FORBIDDEN_PARTS = (
    "future_", "forward_", "target_", "label_",
    "sweep_code_", "first_hit_", "post_hit_",
    "strong_contrarian_", "direction_1h", "direction_4h",
)
RNG = np.random.default_rng(42)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Position Guardian V2 exit-risk research")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--sample-every-minutes", type=int, default=15)
    p.add_argument("--max-train-rows", type=int, default=500_000)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--embargo-hours", type=float, default=4.0)
    p.add_argument("--exit-stop-bps", type=float, default=15.0)
    p.add_argument("--recovery-bps", type=float, default=12.0)
    p.add_argument("--endpoint-exit-bps", type=float, default=-8.0)
    p.add_argument("--minimum-alert-coverage", type=float, default=0.05)
    p.add_argument("--minimum-alert-precision", type=float, default=0.55)
    p.add_argument("--bootstrap-reps", type=int, default=300)
    return p.parse_args()


def jdefault(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, pd.Timestamp): return x.isoformat()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def feature_columns() -> list[str]:
    schema = set(pq.read_schema(FEATURE_PATH).names)
    features = [c for c in CATEGORICAL + CALENDAR + TOPOLOGY if c in schema]
    bad = [c for c in features if any(part in c for part in FORBIDDEN_PARTS)]
    if bad:
        raise RuntimeError(f"Forbidden feature columns: {bad}")
    required = {"id", "logged_at", "symbol", "timeframe", "current_price"}
    missing = required - schema
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    return list(dict.fromkeys(features))


def prepare_x(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = df[features].copy()
    cats = []
    for c in features:
        if c in CATEGORICAL:
            x[c] = x[c].astype("string").fillna("<MISSING>").astype(str)
            cats.append(c)
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x, cats


def split_times(ts: pd.Series, val_fraction: float, test_fraction: float, embargo: pd.Timedelta) -> dict:
    test_start = ts.quantile(1.0 - test_fraction)
    pretest = ts[ts < test_start]
    val_start = pretest.quantile(1.0 - val_fraction / (1.0 - test_fraction))
    return {
        "train_end": val_start - embargo,
        "validation_start": val_start + embargo,
        "validation_end": test_start - embargo,
        "test_start": test_start + embargo,
        "test_end": ts.max(),
    }


def sample_evenly(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    idx = np.linspace(0, len(df) - 1, max_rows, dtype=int)
    return df.iloc[idx].copy()


def make_exit_labels(df: pd.DataFrame, side: str, horizon_minutes: int, exit_stop_bps: float, recovery_bps: float, endpoint_exit_bps: float) -> pd.DataFrame:
    out = []
    horizon_ns = int(pd.Timedelta(minutes=horizon_minutes).value)
    stop = exit_stop_bps / 10_000.0
    recovery = recovery_bps / 10_000.0
    endpoint_cut = endpoint_exit_bps / 10_000.0
    direction = 1.0 if side == "LONG" else -1.0

    for symbol, g in df.groupby("symbol", sort=False, observed=True):
        g = g.sort_values("logged_at").reset_index(drop=True)
        times = g["logged_at"].astype("int64").to_numpy()
        prices = pd.to_numeric(g["current_price"], errors="coerce").to_numpy(float)
        for i in range(len(g)):
            p0 = prices[i]
            if not np.isfinite(p0) or p0 <= 0:
                continue
            end = np.searchsorted(times, times[i] + horizon_ns, side="right")
            if end <= i + 1:
                continue
            future = prices[i + 1:end]
            future = future[np.isfinite(future) & (future > 0)]
            if len(future) == 0:
                continue
            path = direction * (future / p0 - 1.0)
            mfe = float(np.max(path))
            mae = float(np.min(path))
            endpoint = float(path[-1])
            adverse_hits = np.flatnonzero(path <= -stop)
            recovery_hits = np.flatnonzero(path >= recovery)
            adverse_first = len(adverse_hits) > 0 and (len(recovery_hits) == 0 or adverse_hits[0] < recovery_hits[0])
            exit_risk = bool(adverse_first or endpoint <= endpoint_cut or (mae <= -stop and mfe < recovery))
            row = g.iloc[i].to_dict()
            row.update({
                "side": side,
                "horizon_minutes": horizon_minutes,
                "exit_risk": int(exit_risk),
                "mfe_bps": mfe * 10_000.0,
                "mae_bps": mae * 10_000.0,
                "endpoint_bps": endpoint * 10_000.0,
                "adverse_first": adverse_first,
            })
            out.append(row)
    result = pd.DataFrame(out)
    if result.empty:
        raise RuntimeError(f"No labels generated for {side} {horizon_minutes}m")
    return result


def metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(int)
    result = {
        "rows": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision_exit_risk": float(precision_score(y, pred, zero_division=0)),
        "recall_exit_risk": float(recall_score(y, pred, zero_division=0)),
        "specificity_no_exit": float(recall_score(1-y, 1-pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0,1])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0,1]).tolist(),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["pr_auc"] = float(average_precision_score(y, p))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def choose_threshold(y: np.ndarray, p: np.ndarray, min_coverage: float, min_precision: float) -> tuple[dict, list[dict]]:
    rows = []
    for t in np.arange(0.35, 0.91, 0.025):
        pred = p >= t
        coverage = float(pred.mean())
        if pred.sum() == 0:
            precision = recall = 0.0
        else:
            precision = float(precision_score(y, pred, zero_division=0))
            recall = float(recall_score(y, pred, zero_division=0))
        score = precision * recall * math.sqrt(max(coverage, 1e-12)) if coverage >= min_coverage and precision >= min_precision else -1.0
        rows.append({
            "threshold": float(round(t, 4)),
            "coverage": coverage,
            "alerts": int(pred.sum()),
            "precision_exit_risk": precision,
            "recall_exit_risk": recall,
            "score": float(score),
        })
    valid = [r for r in rows if r["score"] >= 0]
    if valid:
        best = max(valid, key=lambda r: (r["score"], r["precision_exit_risk"], r["recall_exit_risk"]))
    else:
        best = max(rows, key=lambda r: (r["precision_exit_risk"] * r["recall_exit_risk"], r["coverage"]))
    return best, rows


def confidence_buckets(y: np.ndarray, p: np.ndarray) -> list[dict]:
    bins = [(0.0,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.000001)]
    out = []
    for lo, hi in bins:
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        out.append({
            "probability_min": lo,
            "probability_max": min(hi, 1.0),
            "rows": int(m.sum()),
            "observed_exit_risk_rate": float(y[m].mean()),
            "mean_predicted_probability": float(p[m].mean()),
        })
    return out


def block_bootstrap_ba(df: pd.DataFrame, reps: int, threshold: float) -> dict:
    x = df.copy()
    x["_block"] = x["symbol"].astype(str) + "|" + x["logged_at"].dt.floor("D").astype(str)
    blocks = [g for _, g in x.groupby("_block", sort=False)]
    vals = []
    for _ in range(reps):
        sample = pd.concat([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))], ignore_index=True)
        vals.append(balanced_accuracy_score(sample["exit_risk"], (sample["p_exit_risk"] >= threshold).astype(int)))
    return {"low": float(np.quantile(vals, 0.025)), "median": float(np.quantile(vals, 0.5)), "high": float(np.quantile(vals, 0.975))}


def main() -> int:
    a = parse_args()
    started = time.time()
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(FEATURE_PATH)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    features = feature_columns()
    cols = list(dict.fromkeys(BASE_COLS + features))
    print("Loading feature stream...")
    data = pd.read_parquet(FEATURE_PATH, columns=cols, filters=[("timeframe", "==", a.timeframe)])
    data["logged_at"] = pd.to_datetime(data["logged_at"], utc=True).astype("datetime64[ns, UTC]")
    data["symbol"] = data["symbol"].astype("string")
    data = data.sort_values(["symbol", "logged_at", "id"], kind="mergesort").drop_duplicates("id").reset_index(drop=True)
    if a.sample_every_minutes > 0:
        data["_bucket"] = data["logged_at"].dt.floor(f"{a.sample_every_minutes}min")
        data = data.sort_values(["symbol", "logged_at"]).drop_duplicates(["symbol", "_bucket"]).drop(columns="_bucket").reset_index(drop=True)

    summary = {"status": "research_only", "target": "EXIT_RISK vs NO_EXIT_RISK", "features": features, "reports": []}
    for side in SIDES:
        for horizon in HORIZONS:
            print(f"\nBuilding labels: {side} {horizon}m")
            labeled = make_exit_labels(data, side, horizon, a.exit_stop_bps, a.recovery_bps, a.endpoint_exit_bps)
            cuts = split_times(labeled["logged_at"], a.validation_fraction, a.test_fraction, pd.Timedelta(hours=a.embargo_hours))
            train = labeled[labeled["logged_at"] <= cuts["train_end"]].copy()
            val = labeled[(labeled["logged_at"] >= cuts["validation_start"]) & (labeled["logged_at"] <= cuts["validation_end"])].copy()
            test = labeled[(labeled["logged_at"] >= cuts["test_start"]) & (labeled["logged_at"] <= cuts["test_end"])].copy()
            train = sample_evenly(train, a.max_train_rows)
            if min(len(train), len(val), len(test)) == 0:
                raise RuntimeError(f"Empty split for {side} {horizon}m")
            xtr, cats = prepare_x(train, features)
            xv, _ = prepare_x(val, features)
            xt, _ = prepare_x(test, features)
            model = CatBoostClassifier(iterations=a.iterations, depth=8, learning_rate=0.06, loss_function="Logloss", eval_metric="AUC", random_seed=42, auto_class_weights="Balanced", verbose=False, allow_writing_files=False)
            model.fit(Pool(xtr, label=train["exit_risk"], cat_features=cats), eval_set=Pool(xv, label=val["exit_risk"], cat_features=cats), early_stopping_rounds=70, verbose=False)
            p_val = model.predict_proba(Pool(xv, cat_features=cats))[:, 1]
            p_test = model.predict_proba(Pool(xt, cat_features=cats))[:, 1]
            chosen, grid = choose_threshold(val["exit_risk"].to_numpy(), p_val, a.minimum_alert_coverage, a.minimum_alert_precision)
            threshold = chosen["threshold"]
            test_out = test[["id","logged_at","symbol","timeframe","current_price","exit_risk","mfe_bps","mae_bps","endpoint_bps","adverse_first"]].copy()
            test_out["p_exit_risk"] = p_test
            test_out["exit_risk_alert"] = (p_test >= threshold).astype("int8")
            test_out["risk_level"] = pd.cut(p_test, bins=[-np.inf, 0.4, 0.6, 0.8, np.inf], labels=["LOW", "MODERATE", "HIGH", "CRITICAL"]).astype(str)
            model_dir = MODEL_ROOT / f"{side.lower()}_{horizon}m"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.cbm"
            model.save_model(model_path)
            pred_path = REPORT_ROOT / f"{side.lower()}_{horizon}m_test_predictions.parquet"
            test_out.to_parquet(pred_path, index=False)
            report = {
                "side": side,
                "horizon_minutes": horizon,
                "label_definition": {"exit_stop_bps": a.exit_stop_bps, "recovery_bps": a.recovery_bps, "endpoint_exit_bps": a.endpoint_exit_bps},
                "split": cuts,
                "counts": {
                    "dataset": len(labeled), "train": len(train), "validation": len(val), "test": len(test),
                    "train_exit_risk_rate": float(train["exit_risk"].mean()),
                    "validation_exit_risk_rate": float(val["exit_risk"].mean()),
                    "test_exit_risk_rate": float(test["exit_risk"].mean()),
                },
                "default_test_metrics": metrics(test["exit_risk"].to_numpy(), p_test, 0.5),
                "selected_policy": {
                    "selection_source": "inner_validation_only",
                    "chosen": chosen,
                    "validation_grid": grid,
                    "test_metrics": metrics(test["exit_risk"].to_numpy(), p_test, threshold),
                    "test_alert_coverage": float((p_test >= threshold).mean()),
                    "bootstrap_balanced_accuracy_95pct": block_bootstrap_ba(test_out, a.bootstrap_reps, threshold),
                },
                "calibration_buckets": confidence_buckets(test["exit_risk"].to_numpy(), p_test),
                "model_path": str(model_path),
                "predictions_path": str(pred_path),
            }
            report_path = REPORT_ROOT / f"{side.lower()}_{horizon}m_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=jdefault))
            summary["reports"].append(report)
            print(f"{side} {horizon}m | BA={report['default_test_metrics']['balanced_accuracy']:.4f} ROC-AUC={report['default_test_metrics']['roc_auc']:.4f} PR-AUC={report['default_test_metrics']['pr_auc']:.4f} chosen_t={threshold:.3f}")

    summary["runtime_seconds"] = time.time() - started
    (REPORT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=jdefault))
    print(f"\nDone: {REPORT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
