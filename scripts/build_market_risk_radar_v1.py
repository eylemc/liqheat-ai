#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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

import build_position_guardian_v4_koinvizyon_matrix as v4  # noqa: E402

OUT = Path("data/reports/market_risk_radar_v1")
MODEL_OUT = Path("data/models/market_risk_radar_v1")
MARKET_ROOT = Path("data/market/binance-futures-um")

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
HORIZONS = (15, 30, 60)

BASELINE_FEATURES = [
    "symbol", "timeframe",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend_utc",
    "current_price", "price_return_15m", "price_return_30m", "price_return_60m",
]

STATIC_TOPOLOGY_FEATURES = list(dict.fromkeys(BASELINE_FEATURES + v4.STATIC_FEATURES))
DYNAMIC_TOPOLOGY_FEATURES = list(
    dict.fromkeys(STATIC_TOPOLOGY_FEATURES + v4.DYNAMIC_FEATURES)
)
COMPACT_MATRIX_FEATURES = [
    "matrix_trend",
    "matrix_flip",
    "matrix_bars_since_flip",
    "matrix_distance_to_vwma_pct",
    "matrix_channel_width_pct",
    "matrix_vwma_slope_1",
    "matrix_vwma_slope_3",
    "matrix_trend_persistence",
    "matrix_age_minutes",
]

FEATURE_GROUPS = {
    "price_calendar_baseline": BASELINE_FEATURES,
    "static_topology": STATIC_TOPOLOGY_FEATURES,
    "dynamic_topology": DYNAMIC_TOPOLOGY_FEATURES,
    "topology_compact_matrix": list(
        dict.fromkeys(DYNAMIC_TOPOLOGY_FEATURES + COMPACT_MATRIX_FEATURES)
    ),
}

CAT_FEATURES = {"symbol", "timeframe", "nearest_side"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Liqheat Market Risk Radar V1: forward volatility and extreme-move risk."
    )
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--matrix-len", type=int, default=20)
    p.add_argument("--sample-every-minutes", type=int, default=15)
    p.add_argument("--horizons-minutes", default="15,30,60")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--max-train-rows", type=int, default=500000)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--embargo-hours", type=float, default=4)
    p.add_argument("--high-quantile", type=float, default=0.75)
    p.add_argument("--extreme-quantile", type=float, default=0.90)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--gpu-device", default="0")
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


def load_minute_market() -> dict[str, pd.DataFrame]:
    market: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        path = MARKET_ROOT / symbol / "1m" / f"{symbol}-1m.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        if "is_complete" in df.columns:
            df = df[df["is_complete"].fillna(False)].copy()
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True).astype(
            "datetime64[ns, UTC]"
        )
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        market[symbol] = df.sort_values("open_time").reset_index(drop=True)
    return market


def add_forward_range(
    snapshots: pd.DataFrame,
    minute_market: dict[str, pd.DataFrame],
    horizon_minutes: int,
) -> pd.DataFrame:
    pieces = []
    horizon_ns = int(pd.Timedelta(minutes=horizon_minutes).value)

    for symbol, g in snapshots.groupby("symbol", sort=False):
        g = g.sort_values("logged_at").copy()
        m = minute_market[str(symbol)]
        mt = m["open_time"].astype("int64").to_numpy()
        mh = m["high"].to_numpy(float)
        ml = m["low"].to_numpy(float)
        mc = m["close"].to_numpy(float)

        st = g["logged_at"].astype("int64").to_numpy()
        spot = pd.to_numeric(g["current_price"], errors="coerce").to_numpy(float)
        n = len(g)
        future_range = np.full(n, np.nan)
        future_abs_return = np.full(n, np.nan)
        future_up_excursion = np.full(n, np.nan)
        future_down_excursion = np.full(n, np.nan)

        for i in range(n):
            if not np.isfinite(spot[i]) or spot[i] <= 0:
                continue
            left = np.searchsorted(mt, st[i], side="right")
            right = np.searchsorted(mt, st[i] + horizon_ns, side="right")
            if right <= left:
                continue
            highs = mh[left:right]
            lows = ml[left:right]
            closes = mc[left:right]
            if not (np.isfinite(highs).any() and np.isfinite(lows).any()):
                continue
            max_high = np.nanmax(highs)
            min_low = np.nanmin(lows)
            endpoint = closes[np.flatnonzero(np.isfinite(closes))[-1]] if np.isfinite(closes).any() else np.nan
            future_range[i] = (max_high - min_low) / spot[i] * 10000.0
            future_up_excursion[i] = (max_high / spot[i] - 1.0) * 10000.0
            future_down_excursion[i] = (spot[i] / min_low - 1.0) * 10000.0
            if np.isfinite(endpoint):
                future_abs_return[i] = abs(endpoint / spot[i] - 1.0) * 10000.0

        g["future_range_bps"] = future_range
        g["future_abs_return_bps"] = future_abs_return
        g["future_up_excursion_bps"] = future_up_excursion
        g["future_down_excursion_bps"] = future_down_excursion
        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)
    return out.dropna(subset=["future_range_bps"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, a: argparse.Namespace):
    times = pd.to_datetime(df["logged_at"], utc=True)
    test_boundary = times.quantile(1.0 - a.test_fraction)
    validation_boundary = times[times < test_boundary].quantile(
        1.0 - a.validation_fraction / (1.0 - a.test_fraction)
    )
    embargo = pd.Timedelta(hours=a.embargo_hours)

    train = df[times <= validation_boundary - embargo].copy()
    validation = df[
        (times >= validation_boundary + embargo)
        & (times <= test_boundary - embargo)
    ].copy()
    test = df[times >= test_boundary + embargo].copy()

    if len(train) > a.max_train_rows:
        idx = np.linspace(0, len(train) - 1, a.max_train_rows, dtype=int)
        train = train.iloc[idx].copy()

    split_info = {
        "train_end": train["logged_at"].max(),
        "validation_start": validation["logged_at"].min(),
        "validation_end": validation["logged_at"].max(),
        "test_start": test["logged_at"].min(),
        "test_end": test["logged_at"].max(),
    }
    return train, validation, test, split_info


def prepare_xy(df: pd.DataFrame, features: list[str]):
    used = [c for c in dict.fromkeys(features) if c in df.columns]
    x = df[used].copy()
    cats = []
    for col in used:
        if col in CAT_FEATURES:
            x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
            cats.append(col)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    return x, cats, used


def fit_head(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target: str,
    a: argparse.Namespace,
):
    xtr, cats, used = prepare_xy(train, features)
    xva, _, _ = prepare_xy(validation, used)
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
        eval_set=Pool(xva, validation[target], cat_features=cats),
        early_stopping_rounds=75,
        verbose=False,
    )
    return model, used, cats


def predict_probability(model, df: pd.DataFrame, features: list[str], cats: list[str]):
    x, _, _ = prepare_xy(df, features)
    cat_cols = [c for c in cats if c in x.columns]
    return model.predict_proba(Pool(x, cat_features=cat_cols))[:, 1]


def head_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= 0.5
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": safe_metric(roc_auc_score, y, p),
        "pr_auc": safe_metric(average_precision_score, y, p),
        "brier": safe_metric(brier_score_loss, y, p),
        "precision_at_0_5": safe_metric(precision_score, y, pred, zero_division=0),
        "recall_at_0_5": safe_metric(recall_score, y, pred, zero_division=0),
    }


def risk_score(p_high: np.ndarray, p_extreme: np.ndarray) -> np.ndarray:
    # Extreme is a subset of high; weighting it more heavily gives a monotonic
    # 0-100 severity score without pretending to predict direction.
    score = 100.0 * (0.40 * p_high + 0.60 * p_extreme)
    return np.clip(score, 0.0, 100.0)


def risk_band(score: np.ndarray) -> np.ndarray:
    return np.select(
        [score >= 75.0, score >= 50.0, score >= 25.0],
        ["EXTREME RISK", "HIGH RISK", "MEDIUM RISK"],
        default="LOW RISK",
    )


def realized_band(values: np.ndarray, q50: float, q75: float, q90: float) -> np.ndarray:
    return np.select(
        [values >= q90, values >= q75, values >= q50],
        ["EXTREME RISK", "HIGH RISK", "MEDIUM RISK"],
        default="LOW RISK",
    )


def band_metrics(realized: np.ndarray, predicted: np.ndarray) -> dict:
    order = {"LOW RISK": 0, "MEDIUM RISK": 1, "HIGH RISK": 2, "EXTREME RISK": 3}
    y = np.array([order[str(v)] for v in realized], dtype=int)
    p = np.array([order[str(v)] for v in predicted], dtype=int)
    return {
        "exact_band_accuracy": float(np.mean(y == p)),
        "within_one_band_accuracy": float(np.mean(np.abs(y - p) <= 1)),
        "mean_absolute_band_error": float(np.mean(np.abs(y - p))),
        "predicted_band_counts": {
            name: int(np.sum(predicted == name)) for name in order
        },
        "realized_band_counts": {
            name: int(np.sum(realized == name)) for name in order
        },
    }


def main() -> None:
    a = parse_args()
    horizons = sorted(
        {int(v.strip()) for v in a.horizons_minutes.split(",") if v.strip()}
    )
    if not 0 < a.high_quantile < a.extreme_quantile < 1:
        raise SystemExit("Require 0 < high_quantile < extreme_quantile < 1")

    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading Koinvizyon Matrix...")
    matrix, parity = v4.load_matrix_all(a)
    print("Loading topology stream...")
    topology = v4.load_topology(a)
    topology = v4.add_dynamic_features(topology)
    snapshots = v4.join_matrix(topology, matrix)
    print("Loading 1m OHLCV for oracle-free forward outcomes...")
    minute_market = load_minute_market()

    reports = []
    latest_scores = []

    for horizon in horizons:
        print(f"{horizon}m future-risk dataset...")
        dataset = add_forward_range(snapshots, minute_market, horizon)
        train, validation, test, split_info = chronological_split(dataset, a)
        if min(len(train), len(validation), len(test)) == 0:
            raise RuntimeError(f"Empty split for {horizon}m")

        q50 = float(train["future_range_bps"].quantile(0.50))
        q75 = float(train["future_range_bps"].quantile(a.high_quantile))
        q90 = float(train["future_range_bps"].quantile(a.extreme_quantile))

        for frame in (train, validation, test):
            frame["is_high_risk"] = (frame["future_range_bps"] >= q75).astype("int8")
            frame["is_extreme_risk"] = (frame["future_range_bps"] >= q90).astype("int8")

        group_reports = []
        best_name = None
        best_validation_score = -np.inf
        best_payload = None

        for group_name, features in FEATURE_GROUPS.items():
            print(f"  {group_name}")
            high_model, used, cats = fit_head(
                train, validation, features, "is_high_risk", a
            )
            extreme_model, _, _ = fit_head(
                train, validation, used, "is_extreme_risk", a
            )

            val_high = predict_probability(high_model, validation, used, cats)
            val_extreme = predict_probability(extreme_model, validation, used, cats)
            test_high = predict_probability(high_model, test, used, cats)
            test_extreme = predict_probability(extreme_model, test, used, cats)

            val_score = risk_score(val_high, val_extreme)
            test_score = risk_score(test_high, test_extreme)
            predicted_bands = risk_band(test_score)
            realized_bands = realized_band(
                test["future_range_bps"].to_numpy(float), q50, q75, q90
            )

            validation_rank_score = np.nanmean([
                safe_metric(average_precision_score, validation["is_high_risk"], val_high, default=np.nan),
                safe_metric(average_precision_score, validation["is_extreme_risk"], val_extreme, default=np.nan),
            ])

            model_dir = MODEL_OUT / f"{horizon}m" / group_name
            model_dir.mkdir(parents=True, exist_ok=True)
            high_path = model_dir / "high_risk_head.cbm"
            extreme_path = model_dir / "extreme_risk_head.cbm"
            high_model.save_model(high_path)
            extreme_model.save_model(extreme_path)

            report = {
                "feature_group": group_name,
                "feature_count": len(used),
                "validation_selection_score_mean_pr_auc": validation_rank_score,
                "high_head": {
                    "validation": head_metrics(validation["is_high_risk"].to_numpy(), val_high),
                    "test": head_metrics(test["is_high_risk"].to_numpy(), test_high),
                },
                "extreme_head": {
                    "validation": head_metrics(validation["is_extreme_risk"].to_numpy(), val_extreme),
                    "test": head_metrics(test["is_extreme_risk"].to_numpy(), test_extreme),
                },
                "risk_score_test": {
                    "mean": float(np.mean(test_score)),
                    "median": float(np.median(test_score)),
                    "p90": float(np.quantile(test_score, 0.90)),
                    **band_metrics(realized_bands, predicted_bands),
                },
                "model_paths": {
                    "high": str(high_path),
                    "extreme": str(extreme_path),
                },
            }
            group_reports.append(report)

            if validation_rank_score > best_validation_score:
                best_validation_score = validation_rank_score
                best_name = group_name
                best_payload = (high_model, extreme_model, used, cats)

        if best_payload is None:
            raise RuntimeError(f"No best model for {horizon}m")

        best_high, best_extreme, best_features, best_cats = best_payload
        latest = snapshots.sort_values("logged_at").groupby("symbol", as_index=False).tail(1)
        latest_high = predict_probability(best_high, latest, best_features, best_cats)
        latest_extreme = predict_probability(best_extreme, latest, best_features, best_cats)
        latest_risk_score = risk_score(latest_high, latest_extreme)
        latest_risk_band = risk_band(latest_risk_score)

        for i, (_, row) in enumerate(latest.reset_index(drop=True).iterrows()):
            latest_scores.append({
                "symbol": str(row["symbol"]),
                "as_of": row["logged_at"],
                "horizon_minutes": horizon,
                "risk_score": float(latest_risk_score[i]),
                "risk_band": str(latest_risk_band[i]),
                "probability_high_risk": float(latest_high[i]),
                "probability_extreme_risk": float(latest_extreme[i]),
                "selected_feature_group": best_name,
                "research_only": True,
            })

        horizon_report = {
            "horizon_minutes": horizon,
            "split": split_info,
            "counts": {
                "dataset": len(dataset),
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "train_derived_range_thresholds_bps": {
                "medium_q50": q50,
                "high_q75": q75,
                "extreme_q90": q90,
            },
            "selected_feature_group_by_validation_mean_pr_auc": best_name,
            "feature_groups": group_reports,
        }
        reports.append(horizon_report)
        (OUT / f"risk_{horizon}m_report.json").write_text(
            json.dumps(json_safe(horizon_report), indent=2), encoding="utf-8"
        )

    summary = {
        "status": "research_only",
        "product": "Liqheat Market Risk Radar V1",
        "question": "How risky is it to initiate or carry a trade right now, independent of direction?",
        "risk_bands": {
            "0-24.999": "LOW RISK",
            "25-49.999": "MEDIUM RISK",
            "50-74.999": "HIGH RISK",
            "75-100": "EXTREME RISK",
        },
        "score_definition": "100 * (0.40 * P(range>=train_q75) + 0.60 * P(range>=train_q90))",
        "outcome_definition": "Future 1-minute OHLC high-low range relative to snapshot price; no future columns are model features.",
        "matrix_parity": parity,
        "feature_groups": {k: len(v) for k, v in FEATURE_GROUPS.items()},
        "settings": vars(a),
        "reports": reports,
        "latest_scores": latest_scores,
    }
    (OUT / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    (OUT / "latest_scores.json").write_text(
        json.dumps(json_safe(latest_scores), indent=2), encoding="utf-8"
    )
    pd.DataFrame(latest_scores).to_csv(OUT / "latest_scores.csv", index=False)

    print(f"Done: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
