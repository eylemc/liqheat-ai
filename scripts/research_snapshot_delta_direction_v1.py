#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "monitoring" / "radar_v2_performance.sqlite"
DEFAULT_OUT = PROJECT_ROOT / "data" / "reports" / "snapshot_delta_direction_v1"


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def parse_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        obj = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def extract_snapshot_features(payload_json: Any) -> dict[str, float]:
    item = parse_payload(payload_json)
    topology = item.get("topology") or {}

    upper_volume = safe_float(topology.get("upper_pool_volume"))
    lower_volume = safe_float(topology.get("lower_pool_volume"))
    upper_distance = safe_float(topology.get("upper_distance_pct"))
    lower_distance = safe_float(topology.get("lower_distance_pct"))
    topology_imbalance = safe_float(topology.get("topology_imbalance"))
    liquidity_pressure_score = safe_float(item.get("liquidity_pressure_score"))

    total_volume = upper_volume + lower_volume
    volume_balance = (
        (upper_volume - lower_volume) / total_volume
        if np.isfinite(total_volume) and total_volume > 0
        else np.nan
    )

    # Direction-independent event probability is not itself directional, but
    # it is retained for later conditioning / interaction analysis.
    event_pressure = (
        liquidity_pressure_score / 100.0
        if np.isfinite(liquidity_pressure_score)
        else np.nan
    )

    # Distance-weighted liquidity attraction. Positive means more/closer
    # liquidity above current price; negative means more/closer below.
    eps = 1e-6
    upper_pull = (
        upper_volume / max(upper_distance, eps)
        if np.isfinite(upper_volume) and np.isfinite(upper_distance) and upper_distance >= 0
        else np.nan
    )
    lower_pull = (
        lower_volume / max(lower_distance, eps)
        if np.isfinite(lower_volume) and np.isfinite(lower_distance) and lower_distance >= 0
        else np.nan
    )
    pull_total = upper_pull + lower_pull
    distance_weighted_balance = (
        (upper_pull - lower_pull) / pull_total
        if np.isfinite(pull_total) and pull_total > 0
        else np.nan
    )

    return {
        "upper_volume": upper_volume,
        "lower_volume": lower_volume,
        "upper_distance": upper_distance,
        "lower_distance": lower_distance,
        "topology_imbalance": topology_imbalance,
        "volume_balance": volume_balance,
        "distance_weighted_balance": distance_weighted_balance,
        "event_pressure": event_pressure,
    }


def linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return np.nan
    y = values[mask]
    x = np.arange(len(values), dtype=float)[mask]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def add_rolling_features(group: pd.DataFrame, window: int) -> pd.DataFrame:
    g = group.sort_values("generated_at").copy()

    for col in ["volume_balance", "distance_weighted_balance", "topology_imbalance"]:
        g[f"{col}_delta"] = g[col] - g[col].shift(window - 1)
        g[f"{col}_slope"] = (
            g[col]
            .rolling(window, min_periods=window)
            .apply(lambda x: linear_slope(x.to_numpy()), raw=False)
        )

    # Four deliberately simple hypotheses. We report all of them instead of
    # tuning weights on this tiny live sample.
    g["score_current_volume"] = g["volume_balance"]
    g["score_volume_flow"] = g["volume_balance_delta"]
    g["score_current_pull"] = g["distance_weighted_balance"]
    g["score_pull_flow"] = g["distance_weighted_balance_delta"]

    # Diagnostic consensus, not a trained model. Components are bounded near
    # [-1, 1], so equal weighting is intentionally transparent.
    g["score_consensus"] = g[
        [
            "score_current_volume",
            "score_volume_flow",
            "score_current_pull",
            "score_pull_flow",
        ]
    ].mean(axis=1, skipna=False)

    return g


def attach_future_price(
    frame: pd.DataFrame,
    horizon_minutes: int,
    tolerance_minutes: float,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []

    for _, group in frame.groupby("symbol", sort=False):
        g = group.sort_values("generated_at").copy()
        lookup = g[["generated_at", "current_price"]].rename(
            columns={"generated_at": "future_time", "current_price": "future_price"}
        )
        g["target_time"] = g["generated_at"] + pd.Timedelta(minutes=horizon_minutes)
        matched = pd.merge_asof(
            g.sort_values("target_time"),
            lookup.sort_values("future_time"),
            left_on="target_time",
            right_on="future_time",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
        )
        pieces.append(matched)

    out = pd.concat(pieces, ignore_index=True)
    out[f"future_return_bps_{horizon_minutes}m"] = (
        (out["future_price"] / out["current_price"] - 1.0) * 10000.0
    )
    return out


def evaluate_score(
    frame: pd.DataFrame,
    score_col: str,
    return_col: str,
    neutral_bps: float,
) -> dict[str, Any]:
    sample = frame[[score_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sample.empty:
        return {
            "score": score_col,
            "rows": 0,
            "directional_rows": 0,
            "coverage": 0.0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "mean_abs_future_return_bps": np.nan,
            "score_return_corr": np.nan,
        }

    sample = sample[sample[score_col] != 0].copy()
    sample["actual"] = np.where(
        sample[return_col] > neutral_bps,
        1,
        np.where(sample[return_col] < -neutral_bps, -1, 0),
    )
    sample["pred"] = np.sign(sample[score_col]).astype(int)
    directional = sample[sample["actual"] != 0].copy()

    if directional.empty:
        accuracy = np.nan
        balanced = np.nan
    else:
        accuracy = float((directional["pred"] == directional["actual"]).mean())
        recalls = []
        for label in (-1, 1):
            subset = directional[directional["actual"] == label]
            if not subset.empty:
                recalls.append(float((subset["pred"] == label).mean()))
        balanced = float(np.mean(recalls)) if recalls else np.nan

    corr = sample[[score_col, return_col]].corr().iloc[0, 1] if len(sample) >= 3 else np.nan

    return {
        "score": score_col,
        "rows": int(len(sample)),
        "directional_rows": int(len(directional)),
        "coverage": float(len(directional) / len(sample)) if len(sample) else 0.0,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "mean_abs_future_return_bps": float(sample[return_col].abs().mean()),
        "score_return_corr": float(corr) if np.isfinite(corr) else np.nan,
    }


def confidence_table(
    frame: pd.DataFrame,
    score_col: str,
    return_col: str,
    neutral_bps: float,
) -> pd.DataFrame:
    sample = frame[[score_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    sample = sample[sample[score_col] != 0]
    if sample.empty:
        return pd.DataFrame()

    sample["strength"] = sample[score_col].abs()
    sample["actual"] = np.where(
        sample[return_col] > neutral_bps,
        1,
        np.where(sample[return_col] < -neutral_bps, -1, 0),
    )
    sample["pred"] = np.sign(sample[score_col]).astype(int)
    sample = sample[sample["actual"] != 0]
    if sample.empty:
        return pd.DataFrame()

    quantiles = sample["strength"].quantile([0.50, 0.75, 0.90]).to_dict()
    rows = []
    for name, threshold in [
        ("all", 0.0),
        ("top50", quantiles.get(0.50, 0.0)),
        ("top25", quantiles.get(0.75, 0.0)),
        ("top10", quantiles.get(0.90, 0.0)),
    ]:
        subset = sample[sample["strength"] >= threshold]
        rows.append(
            {
                "score": score_col,
                "strength_band": name,
                "min_abs_score": float(threshold),
                "rows": int(len(subset)),
                "accuracy": float((subset["pred"] == subset["actual"]).mean()) if len(subset) else np.nan,
                "avg_abs_return_bps": float(subset[return_col].abs().mean()) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research direction from the evolution of the last N liquidation snapshots."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60])
    parser.add_argument("--neutral-bps", type=float, default=10.0)
    parser.add_argument("--tolerance-minutes", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.window < 3:
        raise SystemExit("--window must be >= 3")

    connection = sqlite3.connect(args.db)
    raw = pd.read_sql_query(
        """
        SELECT generated_at, symbol, current_price, raw_radar_score, payload_json
        FROM radar_final_observations
        ORDER BY symbol, generated_at
        """,
        connection,
    )
    connection.close()

    if raw.empty:
        raise SystemExit("radar_final_observations is empty")

    raw["generated_at"] = pd.to_datetime(raw["generated_at"], utc=True, errors="coerce")
    raw["current_price"] = pd.to_numeric(raw["current_price"], errors="coerce")
    raw = raw.dropna(subset=["generated_at", "symbol", "current_price"])

    extracted = pd.DataFrame(raw["payload_json"].map(extract_snapshot_features).tolist(), index=raw.index)
    frame = pd.concat([raw.drop(columns=["payload_json"]), extracted], axis=1)

    frame = (
        frame.groupby("symbol", group_keys=False, sort=False)
        .apply(lambda g: add_rolling_features(g, args.window), include_groups=True)
        .reset_index(drop=True)
    )

    score_columns = [
        "score_current_volume",
        "score_volume_flow",
        "score_current_pull",
        "score_pull_flow",
        "score_consensus",
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    confidence_frames: list[pd.DataFrame] = []

    enriched = frame.copy()
    for horizon in args.horizons:
        enriched = attach_future_price(enriched, horizon, args.tolerance_minutes)
        return_col = f"future_return_bps_{horizon}m"
        for score_col in score_columns:
            result = evaluate_score(enriched, score_col, return_col, args.neutral_bps)
            result["horizon_minutes"] = horizon
            metrics.append(result)

            conf = confidence_table(enriched, score_col, return_col, args.neutral_bps)
            if not conf.empty:
                conf["horizon_minutes"] = horizon
                confidence_frames.append(conf)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.out / "metrics.csv", index=False)

    confidence_df = pd.concat(confidence_frames, ignore_index=True) if confidence_frames else pd.DataFrame()
    confidence_df.to_csv(args.out / "confidence_ladder.csv", index=False)

    selected_cols = [
        "generated_at",
        "symbol",
        "current_price",
        "raw_radar_score",
        "volume_balance",
        "volume_balance_delta",
        "distance_weighted_balance",
        "distance_weighted_balance_delta",
        "score_consensus",
    ] + [f"future_return_bps_{h}m" for h in args.horizons]
    enriched[selected_cols].to_csv(args.out / "snapshot_features.csv", index=False)

    summary = {
        "status": "research_only",
        "source_db": str(args.db),
        "window_snapshots": args.window,
        "horizons_minutes": args.horizons,
        "neutral_bps": args.neutral_bps,
        "tolerance_minutes": args.tolerance_minutes,
        "rows": int(len(enriched)),
        "symbols": sorted(enriched["symbol"].dropna().unique().tolist()),
        "metrics": json.loads(metrics_df.to_json(orient="records")),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    printable = metrics_df.copy()
    for col in ["coverage", "accuracy", "balanced_accuracy", "score_return_corr"]:
        if col in printable:
            printable[col] = printable[col].map(lambda x: round(x, 4) if pd.notna(x) else x)
    if "mean_abs_future_return_bps" in printable:
        printable["mean_abs_future_return_bps"] = printable["mean_abs_future_return_bps"].round(2)

    print("\nSNAPSHOT DELTA DIRECTION V1")
    print(f"rows={len(enriched)} window={args.window} neutral_bps={args.neutral_bps}")
    print(printable.sort_values(["horizon_minutes", "balanced_accuracy"], ascending=[True, False]).to_string(index=False))

    if not confidence_df.empty:
        print("\nCONFIDENCE LADDER")
        show = confidence_df.copy()
        show["accuracy"] = show["accuracy"].round(4)
        show["min_abs_score"] = show["min_abs_score"].round(4)
        show["avg_abs_return_bps"] = show["avg_abs_return_bps"].round(2)
        print(show.sort_values(["horizon_minutes", "score", "min_abs_score"]).to_string(index=False))

    print(f"\nreports: {args.out}")


if __name__ == "__main__":
    main()
