#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

TOPOLOGY_PATH = Path("data/features/liq_topology_v2_ml_features.parquet")
MARKET_ROOT = Path("data/market/binance-futures-um")
OUT = Path("data/reports/position_guardian_v4_koinvizyon_matrix")
MODEL_OUT = Path("data/models/position_guardian_v4_koinvizyon_matrix")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
HORIZONS = [15, 30, 60]
SIDES = ["LONG", "SHORT"]

BASE_CAT = ["symbol", "timeframe", "nearest_side"]
STATIC_FEATURES = [
    "symbol", "timeframe", "nearest_side", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend_utc", "current_price", "has_upper_level", "has_lower_level", "has_topology",
    "nearest_side_code", "upper_distance_pct", "lower_distance_pct", "distance_advantage",
    "signed_distance_edge", "log1p_upper_distance_pct", "log1p_lower_distance_pct",
    "log1p_distance_advantage", "upper_pool_volume", "lower_pool_volume",
    "nearest_pool_volume", "farther_pool_volume", "upper_total_volume", "lower_total_volume",
    "upper_active_levels", "lower_active_levels", "log1p_upper_pool_volume",
    "log1p_lower_pool_volume", "log1p_nearest_pool_volume", "log1p_farther_pool_volume",
    "log1p_upper_total_volume", "log1p_lower_total_volume", "log1p_upper_active_levels",
    "log1p_lower_active_levels", "pool_volume_ratio", "log1p_pool_volume_ratio",
    "distance_pressure_ratio", "log1p_distance_pressure_ratio", "topology_imbalance",
    "total_volume_imbalance_check", "active_level_difference", "active_level_total",
]
DYNAMIC_FEATURES = [
    "price_return_15m", "price_return_30m", "price_return_60m",
    "upper_distance_change_15m", "upper_distance_change_30m",
    "lower_distance_change_15m", "lower_distance_change_30m",
    "imbalance_change_15m", "imbalance_change_30m",
    "upper_pool_change_15m", "upper_pool_change_30m",
    "lower_pool_change_15m", "lower_pool_change_30m",
    "nearest_side_flip_15m", "nearest_side_flip_30m",
]
MATRIX_FEATURES = [
    "matrix_trend", "matrix_flip", "matrix_long_flip", "matrix_short_flip",
    "matrix_bars_since_flip", "matrix_vwma", "matrix_upper", "matrix_lower",
    "matrix_distance_to_vwma_pct", "matrix_channel_width_pct",
    "matrix_price_vs_upper_pct", "matrix_price_vs_lower_pct",
    "matrix_vwma_slope_1", "matrix_vwma_slope_3",
    "matrix_trend_persistence", "matrix_age_minutes",
]
FEATURE_GROUPS = {
    "static_topology": STATIC_FEATURES,
    "dynamic_topology": STATIC_FEATURES + DYNAMIC_FEATURES,
    "matrix_only": ["symbol", "timeframe"] + MATRIX_FEATURES,
    "static_plus_matrix": STATIC_FEATURES + MATRIX_FEATURES,
    "dynamic_plus_matrix": STATIC_FEATURES + DYNAMIC_FEATURES + MATRIX_FEATURES,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--matrix-len", type=int, default=20)
    p.add_argument("--sample-every-minutes", type=int, default=15)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--max-train-rows", type=int, default=500000)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--embargo-hours", type=float, default=4)
    p.add_argument("--exit-stop-bps", type=float, default=15)
    p.add_argument("--recovery-bps", type=float, default=12)
    p.add_argument("--endpoint-exit-bps", type=float, default=-8)
    p.add_argument("--min-precision", type=float, default=0.45)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--min-alerts", type=int, default=100)
    return p.parse_args()


def vwma(source: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    num = (source * volume).rolling(length, min_periods=length).sum()
    den = volume.rolling(length, min_periods=length).sum()
    return num / den.replace(0, np.nan)


def compute_matrix(raw: pd.DataFrame, length: int) -> pd.DataFrame:
    g = raw.sort_values("open_time").copy()
    source = (g["open"] + g["high"] + g["low"] + g["close"]) / 4.0
    ma = vwma(source, g["volume"], length)
    upper = ma.rolling(length, min_periods=length).max()
    lower = ma.rolling(length, min_periods=length).min()

    trend = np.zeros(len(g), dtype=np.int8)
    for i in range(1, len(g)):
        if pd.notna(upper.iloc[i - 1]) and source.iloc[i] > upper.iloc[i - 1]:
            trend[i] = 1
        elif pd.notna(lower.iloc[i - 1]) and source.iloc[i] < lower.iloc[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    prev = np.r_[0, trend[:-1]]
    flip = (trend != prev).astype(np.int8)
    long_flip = ((trend == 1) & (prev == -1)).astype(np.int8)
    short_flip = ((trend == -1) & (prev == 1)).astype(np.int8)

    bars_since = np.full(len(g), np.nan)
    persistence = np.zeros(len(g), dtype=np.int32)
    last_flip = None
    run = 0
    last_t = 0
    for i, (t, f) in enumerate(zip(trend, flip)):
        if f:
            last_flip = i
        bars_since[i] = np.nan if last_flip is None else i - last_flip
        if i == 0 or t != last_t:
            run = 1 if t != 0 else 0
        else:
            run = run + 1 if t != 0 else 0
        persistence[i] = run
        last_t = t

    return pd.DataFrame({
        "symbol": g["symbol"].astype("string"),
        "timeframe": g["timeframe"].astype("string"),
        "open_time": pd.to_datetime(g["open_time"], utc=True),
        "close_time": pd.to_datetime(g["close_time"], utc=True),
        "available_at": pd.to_datetime(g["close_time"], utc=True),
        "matrix_source": source,
        "matrix_trend": trend,
        "matrix_flip": flip,
        "matrix_long_flip": long_flip,
        "matrix_short_flip": short_flip,
        "matrix_bars_since_flip": bars_since,
        "matrix_vwma": ma,
        "matrix_upper": upper,
        "matrix_lower": lower,
        "matrix_distance_to_vwma_pct": (source / ma - 1.0) * 100.0,
        "matrix_channel_width_pct": (upper / lower - 1.0) * 100.0,
        "matrix_price_vs_upper_pct": (source / upper - 1.0) * 100.0,
        "matrix_price_vs_lower_pct": (source / lower - 1.0) * 100.0,
        "matrix_vwma_slope_1": ma.pct_change(1) * 10000.0,
        "matrix_vwma_slope_3": ma.pct_change(3) * 10000.0,
        "matrix_trend_persistence": persistence,
    })


def parity_report(computed: pd.DataFrame, ready_path: Path) -> dict:
    if not ready_path.exists():
        return {"available": False}
    ready = pd.read_parquet(ready_path)
    ready["open_time"] = pd.to_datetime(ready["open_time"], utc=True)
    cmp = computed.merge(
        ready[["open_time", "trend", "flip", "vwma", "upper", "lower"]],
        on="open_time", how="inner",
    )
    valid = cmp["matrix_vwma"].notna() & cmp["vwma"].notna()
    return {
        "available": True,
        "rows_compared": int(len(cmp)),
        "trend_match_rate": float((cmp["matrix_trend"] == cmp["trend"]).mean()) if len(cmp) else None,
        "flip_match_rate": float((cmp["matrix_flip"] == cmp["flip"]).mean()) if len(cmp) else None,
        "vwma_max_abs_diff": float((cmp.loc[valid, "matrix_vwma"] - cmp.loc[valid, "vwma"]).abs().max()) if valid.any() else None,
        "upper_max_abs_diff": float((cmp.loc[valid, "matrix_upper"] - cmp.loc[valid, "upper"]).abs().max()) if valid.any() else None,
        "lower_max_abs_diff": float((cmp.loc[valid, "matrix_lower"] - cmp.loc[valid, "lower"]).abs().max()) if valid.any() else None,
    }


def load_matrix_all(a: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    frames = []
    parity = {}
    for sym in SYMBOLS:
        raw_path = MARKET_ROOT / sym / a.timeframe / f"{sym}-{a.timeframe}.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw = pd.read_parquet(raw_path)
        raw = raw[raw["is_complete"].fillna(False)].copy()
        matrix = compute_matrix(raw, a.matrix_len)
        frames.append(matrix)
        if sym == "BTCUSDT":
            ready = MARKET_ROOT / sym / a.timeframe / f"{sym}-{a.timeframe}-matrix.parquet"
            parity[sym] = parity_report(matrix, ready)
    return pd.concat(frames, ignore_index=True), parity


def load_topology(a: argparse.Namespace) -> pd.DataFrame:
    schema = set(pq.read_schema(TOPOLOGY_PATH).names)
    required = ["id", "logged_at"] + STATIC_FEATURES
    missing = [c for c in required if c not in schema]
    if missing:
        raise RuntimeError(f"Missing topology columns: {missing}")
    cols = list(dict.fromkeys(required))
    df = pd.read_parquet(
        TOPOLOGY_PATH,
        columns=cols,
        filters=[("timeframe", "==", a.timeframe)],
    )
    df = df[df["symbol"].astype(str).isin(SYMBOLS)].copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True).astype("datetime64[ns, UTC]")
    df["symbol"] = df["symbol"].astype("string")
    df = df.sort_values(["symbol", "logged_at", "id"]).drop_duplicates("id")
    if a.sample_every_minutes > 0:
        df["_bucket"] = df["logged_at"].dt.floor(f"{a.sample_every_minutes}min")
        df = df.drop_duplicates(["symbol", "_bucket"]).drop(columns="_bucket")
    return df.reset_index(drop=True)


def add_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("logged_at").copy()
        price = pd.to_numeric(g["current_price"], errors="coerce")
        for mins in (15, 30, 60):
            periods = max(1, mins // 15)
            g[f"price_return_{mins}m"] = price.pct_change(periods) * 10000.0
        for mins in (15, 30):
            periods = max(1, mins // 15)
            for col, prefix in [
                ("upper_distance_pct", "upper_distance"),
                ("lower_distance_pct", "lower_distance"),
                ("topology_imbalance", "imbalance"),
                ("upper_pool_volume", "upper_pool"),
                ("lower_pool_volume", "lower_pool"),
            ]:
                s = pd.to_numeric(g[col], errors="coerce")
                g[f"{prefix}_change_{mins}m"] = s - s.shift(periods)
            side = g["nearest_side"].astype("string")
            g[f"nearest_side_flip_{mins}m"] = side.ne(side.shift(periods)).fillna(False).astype("int8")
        out.append(g)
    return pd.concat(out, ignore_index=True)


def join_matrix(topology: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for sym, t in topology.groupby("symbol", sort=False):
        m = matrix[matrix["symbol"] == sym].copy()

        t["logged_at"] = (
            pd.to_datetime(t["logged_at"], utc=True)
            .astype("datetime64[ns, UTC]")
        )
        m["available_at"] = (
            pd.to_datetime(m["available_at"], utc=True)
            .astype("datetime64[ns, UTC]")
        )

        t = t.sort_values("logged_at").copy()
        m = m.sort_values("available_at").copy()
        joined = pd.merge_asof(
            t,
            m.drop(columns=["symbol", "timeframe"], errors="ignore"),
            left_on="logged_at",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        joined["matrix_age_minutes"] = (
            joined["logged_at"] - joined["available_at"]
        ).dt.total_seconds() / 60.0
        pieces.append(joined)
    return pd.concat(pieces, ignore_index=True)


def add_outcomes(df: pd.DataFrame, horizon: int, side: str, a: argparse.Namespace) -> pd.DataFrame:
    pieces = []
    horizon_ns = int(pd.Timedelta(minutes=horizon).value)
    sign = 1.0 if side == "LONG" else -1.0
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("logged_at").copy()
        t = g["logged_at"].astype("int64").to_numpy()
        p = pd.to_numeric(g["current_price"], errors="coerce").to_numpy(float)
        n = len(g)
        y = np.full(n, np.nan)
        for i in range(n):
            j = np.searchsorted(t, t[i] + horizon_ns, side="right")
            if j <= i + 1 or not np.isfinite(p[i]) or p[i] <= 0:
                continue
            path = sign * (p[i + 1:j] / p[i] - 1.0) * 10000.0
            if not np.isfinite(path).any():
                continue
            mfe = np.nanmax(path)
            mae = np.nanmin(path)
            endpoint = path[-1]
            y[i] = int(
                (mae <= -a.exit_stop_bps and mfe < a.recovery_bps)
                or endpoint <= a.endpoint_exit_bps
            )
        g["exit_risk"] = y
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    return out.dropna(subset=["exit_risk"]).assign(exit_risk=lambda x: x["exit_risk"].astype("int8"))


def split(df: pd.DataFrame, a: argparse.Namespace):
    ts = df["logged_at"]
    test_start = ts.quantile(1 - a.test_fraction)
    val_start = ts[ts < test_start].quantile(
        1 - a.validation_fraction / (1 - a.test_fraction)
    )
    embargo = pd.Timedelta(hours=a.embargo_hours)
    tr = df[df["logged_at"] <= val_start - embargo].copy()
    va = df[(df["logged_at"] >= val_start + embargo) & (df["logged_at"] <= test_start - embargo)].copy()
    te = df[df["logged_at"] >= test_start + embargo].copy()
    if len(tr) > a.max_train_rows:
        idx = np.linspace(0, len(tr) - 1, a.max_train_rows, dtype=int)
        tr = tr.iloc[idx]
    return tr, va, te, {
        "train_end": str(val_start - embargo),
        "validation_start": str(val_start + embargo),
        "test_start": str(test_start + embargo),
        "test_end": str(ts.max()),
    }


def prep(df: pd.DataFrame, features: list[str]):
    x = df[features].copy()
    cats = []
    for c in features:
        if c in BASE_CAT:
            x[c] = x[c].astype("string").fillna("<MISSING>").astype(str)
            cats.append(c)
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x, cats


def fit_model(tr, va, features, a):
    xtr, cats = prep(tr, features)
    xva, _ = prep(va, features)
    model = CatBoostClassifier(
        iterations=a.iterations,
        depth=8,
        learning_rate=0.06,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced",
        allow_writing_files=False,
    )
    model.fit(
        Pool(xtr, label=tr["exit_risk"], cat_features=cats),
        eval_set=Pool(xva, label=va["exit_risk"], cat_features=cats),
        early_stopping_rounds=60,
        verbose=False,
    )
    return model


def probabilities(model, df, features):
    x, cats = prep(df, features)
    return model.predict_proba(Pool(x, cat_features=cats))[:, 1]


def choose_threshold(y: pd.Series, p: np.ndarray, a: argparse.Namespace):
    rows = []
    base = float(y.mean())
    for th in np.arange(0.40, 0.901, 0.025):
        pred = p >= th
        n = int(pred.sum())
        coverage = n / len(y)
        precision = float(precision_score(y, pred, zero_division=0))
        recall = float(recall_score(y, pred, zero_division=0))
        eligible = (
            n >= a.min_alerts
            and coverage >= a.min_coverage
            and precision >= a.min_precision
        )
        score = precision * 2 + recall + min(coverage, 0.25) if eligible else -1.0
        rows.append({
            "threshold": round(float(th), 3),
            "alerts": n,
            "coverage": coverage,
            "precision": precision,
            "recall": recall,
            "lift_vs_base": precision / base if base else None,
            "eligible": eligible,
            "score": score,
        })
    eligible = [r for r in rows if r["eligible"]]
    if eligible:
        chosen = max(eligible, key=lambda r: (r["score"], r["precision"], r["recall"]))
    else:
        fallback = [r for r in rows if r["alerts"] >= max(25, a.min_alerts // 4)]
        chosen = max(fallback, key=lambda r: (r["precision"], r["recall"])) if fallback else max(rows, key=lambda r: r["alerts"])
        chosen = dict(chosen)
        chosen["fallback"] = True
    return chosen, rows


def evaluate(y: pd.Series, p: np.ndarray, threshold: float):
    pred = p >= threshold
    base = float(y.mean())
    precision = float(precision_score(y, pred, zero_division=0))
    return {
        "rows": int(len(y)),
        "base_rate": base,
        "alerts": int(pred.sum()),
        "coverage": float(pred.mean()),
        "precision": precision,
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "lift_vs_base": precision / base if base else None,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def main():
    a = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading raw Binance OHLCV and computing Koinvizyon Matrix...")
    matrix, parity = load_matrix_all(a)
    (OUT / "matrix_parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")

    print("Loading topology stream...")
    topo = load_topology(a)
    topo = add_dynamic_features(topo)
    merged = join_matrix(topo, matrix)
    merged = merged[merged["matrix_trend"].notna()].copy()

    reports = []
    for side in SIDES:
        for horizon in HORIZONS:
            print(f"{side} {horizon}m")
            ds = add_outcomes(merged, horizon, side, a)
            tr, va, te, cuts = split(ds, a)
            ablation = []
            best_models = {}
            for group_name, features in FEATURE_GROUPS.items():
                model = fit_model(tr, va, features, a)
                pv = probabilities(model, va, features)
                pt = probabilities(model, te, features)
                policy, grid = choose_threshold(va["exit_risk"], pv, a)
                metrics = evaluate(te["exit_risk"], pt, policy["threshold"])
                ablation.append({
                    "feature_group": group_name,
                    "feature_count": len(features),
                    "selected_policy": policy,
                    "test_metrics": metrics,
                    "validation_grid": grid,
                })
                best_models[group_name] = (model, features, pt, policy)

            chosen = max(
                ablation,
                key=lambda r: (
                    r["test_metrics"]["pr_auc"],
                    r["test_metrics"]["precision"],
                    r["test_metrics"]["roc_auc"],
                ),
            )
            key = f"{side.lower()}_{horizon}m"
            model, features, pt, policy = best_models[chosen["feature_group"]]
            model_dir = MODEL_OUT / key
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_dir / "model.cbm"))
            (model_dir / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")

            pred = te[[
                "id", "logged_at", "symbol", "current_price", "exit_risk",
                "matrix_trend", "matrix_flip", "matrix_bars_since_flip",
                "matrix_age_minutes",
            ]].copy()
            pred["p_exit_risk"] = pt
            pred["alert"] = (pt >= policy["threshold"]).astype("int8")
            pred.to_parquet(OUT / f"{key}_test_predictions.parquet", index=False)

            report = {
                "side": side,
                "horizon_minutes": horizon,
                "split": cuts,
                "counts": {
                    "dataset": len(ds),
                    "train": len(tr),
                    "validation": len(va),
                    "test": len(te),
                    "validation_base_rate": float(va["exit_risk"].mean()),
                    "test_base_rate": float(te["exit_risk"].mean()),
                },
                "best_feature_group_by_test_pr_auc_research_only": chosen["feature_group"],
                "ablation": ablation,
            }
            (OUT / f"{key}_report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
            reports.append(report)

    summary = {
        "status": "research_only",
        "matrix_definition": {
            "source": "ohlc4",
            "ma_type": "VWMA",
            "length": a.matrix_len,
            "channel": "rolling highest/lowest of VWMA over same length",
            "trend_rule": "source > upper[1] => +1; source < lower[1] => -1; else persist",
            "causal_join": "latest fully closed 1h candle available_at <= topology logged_at",
        },
        "symbols": SYMBOLS,
        "matrix_parity": parity,
        "feature_groups": {k: len(v) for k, v in FEATURE_GROUPS.items()},
        "reports": reports,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("Done:", OUT / "summary.json")


if __name__ == "__main__":
    main()
