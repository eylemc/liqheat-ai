#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_market_risk_radar_v1 as v1  # noqa: E402
import build_position_guardian_v4_koinvizyon_matrix as v4  # noqa: E402

OUT = Path("data/reports/market_risk_radar_v2_adaptive_walkforward")
MODEL_OUT = Path("data/models/market_risk_radar_v2_adaptive_walkforward")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

BASELINE_FEATURES = list(v1.BASELINE_FEATURES)
STATIC_TOPOLOGY_FEATURES = list(dict.fromkeys(BASELINE_FEATURES + list(v4.STATIC_FEATURES)))
DYNAMIC_TOPOLOGY_FEATURES = list(
    dict.fromkeys(STATIC_TOPOLOGY_FEATURES + list(v4.DYNAMIC_FEATURES))
)
FEATURE_GROUPS = {
    "price_calendar_baseline": BASELINE_FEATURES,
    "price_calendar_static_topology": STATIC_TOPOLOGY_FEATURES,
    "price_calendar_dynamic_topology": DYNAMIC_TOPOLOGY_FEATURES,
}


@dataclass
class Calibrator:
    method: str
    model: object | None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "raw" or self.model is None:
            return p
        if self.method == "platt":
            z = np.log(p / (1 - p)).reshape(-1, 1)
            return np.clip(self.model.predict_proba(z)[:, 1], 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 1e-6, 1 - 1e-6)
        raise ValueError(self.method)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Liqheat Market Risk Radar V2: adaptive per-symbol labels and walk-forward calibration."
    )
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--sample-every-minutes", type=int, default=15)
    p.add_argument("--horizons-minutes", default="15,30,60")
    p.add_argument("--rolling-label-days", type=int, default=30)
    p.add_argument("--min-label-history", type=int, default=300)
    p.add_argument("--high-quantile", type=float, default=0.75)
    p.add_argument("--extreme-quantile", type=float, default=0.90)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--initial-train-fraction", type=float, default=0.50)
    p.add_argument("--calibration-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction-per-fold", type=float, default=0.10)
    p.add_argument("--embargo-hours", type=float, default=4)
    p.add_argument("--iterations", type=int, default=450)
    p.add_argument("--max-train-rows", type=int, default=500000)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--gpu-device", default="0")
    return p.parse_args()


def safe_metric(fn, *args, default=None, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return default


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def load_snapshots(a: argparse.Namespace) -> pd.DataFrame:
    # Matrix deliberately excluded in V2.
    topology = v4.load_topology(a)
    topology = v4.add_dynamic_features(topology)
    return topology.sort_values(["symbol", "logged_at"]).reset_index(drop=True)


def add_causal_adaptive_labels(df: pd.DataFrame, horizon: int, a: argparse.Namespace) -> pd.DataFrame:
    pieces = []
    window = f"{a.rolling_label_days}D"
    delay = pd.Timedelta(minutes=horizon)

    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("logged_at").copy()
        history = g[["logged_at", "future_range_bps"]].dropna().copy()
        history["available_at"] = history["logged_at"] + delay
        history = history.sort_values("available_at")

        s = history.set_index("available_at")["future_range_bps"]
        q50 = s.rolling(window, closed="left", min_periods=a.min_label_history).quantile(0.50)
        q75 = s.rolling(window, closed="left", min_periods=a.min_label_history).quantile(a.high_quantile)
        q90 = s.rolling(window, closed="left", min_periods=a.min_label_history).quantile(a.extreme_quantile)
        count = s.rolling(window, closed="left", min_periods=1).count()

        thresholds = pd.DataFrame({
            "threshold_available_at": q50.index,
            "adaptive_q50_bps": q50.to_numpy(),
            "adaptive_q75_bps": q75.to_numpy(),
            "adaptive_q90_bps": q90.to_numpy(),
            "adaptive_history_count": count.to_numpy(),
        }).sort_values("threshold_available_at")

        left = g.sort_values("logged_at")
        joined = pd.merge_asof(
            left,
            thresholds,
            left_on="logged_at",
            right_on="threshold_available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        joined["is_high_risk"] = (
            joined["future_range_bps"] >= joined["adaptive_q75_bps"]
        ).astype("Int8")
        joined["is_extreme_risk"] = (
            joined["future_range_bps"] >= joined["adaptive_q90_bps"]
        ).astype("Int8")
        joined["realized_risk_band"] = np.select(
            [
                joined["future_range_bps"] >= joined["adaptive_q90_bps"],
                joined["future_range_bps"] >= joined["adaptive_q75_bps"],
                joined["future_range_bps"] >= joined["adaptive_q50_bps"],
            ],
            ["EXTREME RISK", "HIGH RISK", "MEDIUM RISK"],
            default="LOW RISK",
        )
        pieces.append(joined)

    out = pd.concat(pieces, ignore_index=True)
    out = out.dropna(subset=["adaptive_q50_bps", "adaptive_q75_bps", "adaptive_q90_bps"])
    out["is_high_risk"] = out["is_high_risk"].astype("int8")
    out["is_extreme_risk"] = out["is_extreme_risk"].astype("int8")
    return out.sort_values("logged_at").reset_index(drop=True)


def make_folds(df: pd.DataFrame, a: argparse.Namespace) -> list[dict]:
    times = np.array(sorted(pd.to_datetime(df["logged_at"], utc=True).unique()))
    n = len(times)
    train_n = max(1, int(n * a.initial_train_fraction))
    cal_n = max(1, int(n * a.calibration_fraction))
    test_n = max(1, int(n * a.test_fraction_per_fold))
    embargo = pd.Timedelta(hours=a.embargo_hours)
    folds = []

    for i in range(a.folds):
        train_end_i = train_n + i * test_n - 1
        cal_start_i = train_end_i + 1
        cal_end_i = cal_start_i + cal_n - 1
        test_start_i = cal_end_i + 1
        test_end_i = min(test_start_i + test_n - 1, n - 1)
        if test_start_i >= n or cal_end_i >= n:
            break

        train_end = pd.Timestamp(times[train_end_i])
        cal_start = pd.Timestamp(times[cal_start_i])
        cal_end = pd.Timestamp(times[cal_end_i])
        test_start = pd.Timestamp(times[test_start_i])
        test_end = pd.Timestamp(times[test_end_i])

        tr = df[df["logged_at"] <= train_end - embargo].copy()
        ca = df[(df["logged_at"] >= cal_start) & (df["logged_at"] <= cal_end - embargo)].copy()
        te = df[(df["logged_at"] >= test_start) & (df["logged_at"] <= test_end)].copy()
        if len(tr) > a.max_train_rows:
            tr = tr.sort_values("logged_at").tail(a.max_train_rows).copy()
        if tr.empty or ca.empty or te.empty:
            continue
        folds.append({
            "fold": i + 1,
            "train": tr,
            "calibration": ca,
            "test": te,
            "train_end": tr["logged_at"].max(),
            "calibration_start": ca["logged_at"].min(),
            "calibration_end": ca["logged_at"].max(),
            "test_start": te["logged_at"].min(),
            "test_end": te["logged_at"].max(),
        })
    return folds


def fit_head(train: pd.DataFrame, calibration: pd.DataFrame, features: list[str], target: str, a):
    xtr, cats, used = v1.prepare_xy(train, features)
    xca, _, _ = v1.prepare_xy(calibration, used)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=a.iterations,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=6,
        random_seed=a.random_seed,
        auto_class_weights="Balanced",
        task_type="GPU",
        devices=a.gpu_device,
        bootstrap_type="Bayesian",
        metric_period=5,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(xtr, train[target], cat_features=cats),
        eval_set=Pool(xca, calibration[target], cat_features=cats),
        early_stopping_rounds=75,
        verbose=False,
    )
    return model, used, cats


def predict(model, df, used, cats):
    x, _, _ = v1.prepare_xy(df, used)
    return model.predict_proba(Pool(x, cat_features=[c for c in cats if c in x.columns]))[:, 1]


def fit_calibrator(y: np.ndarray, p: np.ndarray) -> tuple[Calibrator, list[dict]]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    candidates: list[tuple[Calibrator, float]] = []
    rows = []

    raw = Calibrator("raw", None)
    candidates.append((raw, brier_score_loss(y, raw.transform(p))))

    if len(np.unique(y)) == 2:
        z = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(max_iter=1000, random_state=42).fit(z, y)
        platt = Calibrator("platt", lr)
        candidates.append((platt, brier_score_loss(y, platt.transform(p))))

        if len(y) >= 500 and y.sum() >= 50 and (1 - y).sum() >= 50:
            iso_model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            iso_model.fit(p, y)
            iso = Calibrator("isotonic", iso_model)
            candidates.append((iso, brier_score_loss(y, iso.transform(p))))

    best = min(candidates, key=lambda x: x[1])[0]
    for cal, score in candidates:
        rows.append({"method": cal.method, "calibration_brier": float(score), "selected": cal.method == best.method})
    return best, rows


def metrics(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict:
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "raw_roc_auc": safe_metric(roc_auc_score, y, raw),
        "raw_pr_auc": safe_metric(average_precision_score, y, raw),
        "raw_brier": safe_metric(brier_score_loss, y, raw),
        "calibrated_roc_auc": safe_metric(roc_auc_score, y, calibrated),
        "calibrated_pr_auc": safe_metric(average_precision_score, y, calibrated),
        "calibrated_brier": safe_metric(brier_score_loss, y, calibrated),
    }


def band_from_score(score: np.ndarray) -> np.ndarray:
    return np.select(
        [score >= 75, score >= 50, score >= 25],
        ["EXTREME RISK", "HIGH RISK", "MEDIUM RISK"],
        default="LOW RISK",
    )


def band_metrics(realized: np.ndarray, predicted: np.ndarray) -> dict:
    order = {"LOW RISK": 0, "MEDIUM RISK": 1, "HIGH RISK": 2, "EXTREME RISK": 3}
    y = np.array([order[str(x)] for x in realized])
    p = np.array([order[str(x)] for x in predicted])
    return {
        "exact_band_accuracy": float(np.mean(y == p)),
        "within_one_band_accuracy": float(np.mean(np.abs(y - p) <= 1)),
        "mean_absolute_band_error": float(np.mean(np.abs(y - p))),
    }


def main() -> None:
    a = parse_args()
    horizons = sorted({int(x.strip()) for x in a.horizons_minutes.split(",") if x.strip()})
    if not 0 < a.high_quantile < a.extreme_quantile < 1:
        raise SystemExit("Require 0 < high_quantile < extreme_quantile < 1")

    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading topology stream (Matrix excluded)...")
    snapshots = load_snapshots(a)
    print("Loading 1m OHLCV...")
    minute_market = v1.load_minute_market()

    fold_rows = []
    stability_rows = []
    reports = []

    for horizon in horizons:
        print(f"{horizon}m adaptive risk dataset...")
        outcomes = v1.add_forward_range(snapshots, minute_market, horizon)
        dataset = add_causal_adaptive_labels(outcomes, horizon, a)
        folds = make_folds(dataset, a)
        if not folds:
            raise RuntimeError(f"No folds for {horizon}m")

        for group_name, features in FEATURE_GROUPS.items():
            print(f"  {group_name}")
            for fold in folds:
                for head, target in (("high", "is_high_risk"), ("extreme", "is_extreme_risk")):
                    model, used, cats = fit_head(fold["train"], fold["calibration"], features, target, a)
                    p_cal_raw = predict(model, fold["calibration"], used, cats)
                    p_test_raw = predict(model, fold["test"], used, cats)
                    calibrator, candidates = fit_calibrator(fold["calibration"][target].to_numpy(), p_cal_raw)
                    p_test_cal = calibrator.transform(p_test_raw)
                    m = metrics(fold["test"][target].to_numpy(), p_test_raw, p_test_cal)

                    for symbol in ("ALL", *SYMBOLS):
                        if symbol == "ALL":
                            mask = np.ones(len(fold["test"]), dtype=bool)
                        else:
                            mask = fold["test"]["symbol"].astype(str).to_numpy() == symbol
                        if mask.sum() == 0:
                            continue
                        ys = fold["test"][target].to_numpy()[mask]
                        rs = p_test_raw[mask]
                        cs = p_test_cal[mask]
                        sm = metrics(ys, rs, cs)
                        fold_rows.append({
                            "horizon_minutes": horizon,
                            "feature_group": group_name,
                            "head": head,
                            "fold": fold["fold"],
                            "symbol": symbol,
                            "calibration_method": calibrator.method,
                            "train_rows": len(fold["train"]),
                            "calibration_rows": len(fold["calibration"]),
                            "test_rows": int(mask.sum()),
                            **sm,
                            "calibration_candidates": candidates,
                        })

                    model_dir = MODEL_OUT / f"{horizon}m" / group_name / f"fold_{fold['fold']}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    model.save_model(str(model_dir / f"{head}_head.cbm"))

                # Combined calibrated score for this fold/group.
                # Refit references are intentionally local to the fold.
                high_model, high_used, high_cats = fit_head(fold["train"], fold["calibration"], features, "is_high_risk", a)
                ext_model, ext_used, ext_cats = fit_head(fold["train"], fold["calibration"], features, "is_extreme_risk", a)
                high_cal_raw = predict(high_model, fold["calibration"], high_used, high_cats)
                ext_cal_raw = predict(ext_model, fold["calibration"], ext_used, ext_cats)
                high_cal, _ = fit_calibrator(fold["calibration"]["is_high_risk"].to_numpy(), high_cal_raw)
                ext_cal, _ = fit_calibrator(fold["calibration"]["is_extreme_risk"].to_numpy(), ext_cal_raw)
                p_high = high_cal.transform(predict(high_model, fold["test"], high_used, high_cats))
                p_ext = ext_cal.transform(predict(ext_model, fold["test"], ext_used, ext_cats))
                score = np.clip(100 * (0.40 * p_high + 0.60 * p_ext), 0, 100)
                predicted_band = band_from_score(score)
                bm = band_metrics(fold["test"]["realized_risk_band"].to_numpy(), predicted_band)
                stability_rows.append({
                    "horizon_minutes": horizon,
                    "feature_group": group_name,
                    "fold": fold["fold"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "score_mean": float(score.mean()),
                    "score_median": float(np.median(score)),
                    "score_p90": float(np.quantile(score, 0.90)),
                    **bm,
                })

        reports.append({
            "horizon_minutes": horizon,
            "dataset_rows": len(dataset),
            "adaptive_label_positive_rates": {
                "high": float(dataset["is_high_risk"].mean()),
                "extreme": float(dataset["is_extreme_risk"].mean()),
            },
            "adaptive_thresholds_by_symbol": dataset.groupby("symbol")[[
                "adaptive_q50_bps", "adaptive_q75_bps", "adaptive_q90_bps"
            ]].agg(["mean", "std", "min", "max"]).to_dict(),
        })

    fold_df = pd.DataFrame(fold_rows)
    stability_df = pd.DataFrame(stability_rows)
    fold_df.to_csv(OUT / "fold_metrics.csv", index=False)
    stability_df.to_csv(OUT / "risk_score_stability.csv", index=False)

    aggregate = []
    if not fold_df.empty:
        for keys, g in fold_df.groupby(["horizon_minutes", "feature_group", "head", "symbol"]):
            aggregate.append({
                "horizon_minutes": keys[0],
                "feature_group": keys[1],
                "head": keys[2],
                "symbol": keys[3],
                "folds": int(len(g)),
                "calibrated_pr_auc_mean": float(g["calibrated_pr_auc"].mean()),
                "calibrated_pr_auc_std": float(g["calibrated_pr_auc"].std(ddof=0)),
                "calibrated_roc_auc_mean": float(g["calibrated_roc_auc"].mean()),
                "calibrated_brier_mean": float(g["calibrated_brier"].mean()),
                "positive_rate_mean": float(g["positive_rate"].mean()),
                "pr_auc_lift_vs_base": float(g["calibrated_pr_auc"].mean() / max(g["positive_rate"].mean(), 1e-12)),
            })
    pd.DataFrame(aggregate).to_csv(OUT / "aggregate_metrics.csv", index=False)

    summary = {
        "status": "research_only",
        "product": "Liqheat Market Risk Radar V2",
        "matrix_included": False,
        "label_definition": "Per-symbol rolling quantiles computed only from outcomes fully available before each snapshot.",
        "calibration": "Walk-forward raw vs Platt vs isotonic, selected by calibration-block Brier score.",
        "settings": vars(a),
        "feature_groups": {k: len(v) for k, v in FEATURE_GROUPS.items()},
        "reports": reports,
        "aggregate_metrics": aggregate,
        "risk_score_stability": stability_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"Done: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
