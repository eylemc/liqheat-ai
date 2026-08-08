#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_snapshot_delta_direction_v1 import (
    DEFAULT_DB,
    add_rolling_features,
    attach_future_price,
    extract_snapshot_features,
    parse_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "reports" / "snapshot_delta_direction_v2"


def extract_market_heat(payload_json: Any) -> float:
    item = parse_payload(payload_json)
    risk = item.get("ai_market_risk") or {}
    try:
        value = float(risk.get("risk_score"))
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def add_v2_scores(group: pd.DataFrame, window: int) -> pd.DataFrame:
    g = add_rolling_features(group, window)

    # V1 showed that static liquidity concentration may act in the same
    # direction, while recent liquidity creation may be contrarian.
    g["score_contrarian_volume_flow"] = -g["score_volume_flow"]
    g["score_contrarian_pull_flow"] = -g["score_pull_flow"]

    # Transparent, untrained combinations. No weights are optimized here.
    g["score_static"] = g[
        ["score_current_volume", "score_current_pull"]
    ].mean(axis=1, skipna=False)

    g["score_contrarian_flow"] = g[
        ["score_contrarian_volume_flow", "score_contrarian_pull_flow"]
    ].mean(axis=1, skipna=False)

    g["score_static_plus_contrarian_flow"] = g[
        [
            "score_current_volume",
            "score_current_pull",
            "score_contrarian_volume_flow",
            "score_contrarian_pull_flow",
        ]
    ].mean(axis=1, skipna=False)

    return g


def direction_labels(sample: pd.DataFrame, score_col: str, return_col: str, neutral_bps: float) -> pd.DataFrame:
    s = sample[[score_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    s = s[s[score_col] != 0]
    s["actual"] = np.where(
        s[return_col] > neutral_bps,
        1,
        np.where(s[return_col] < -neutral_bps, -1, 0),
    )
    s["pred"] = np.sign(s[score_col]).astype(int)
    return s[s["actual"] != 0].copy()


def metric_row(
    frame: pd.DataFrame,
    score_col: str,
    return_col: str,
    neutral_bps: float,
    horizon: int,
    segment: str,
    symbol: str = "ALL",
) -> dict[str, Any]:
    directional = direction_labels(frame, score_col, return_col, neutral_bps)
    if directional.empty:
        return {
            "horizon_minutes": horizon,
            "segment": segment,
            "symbol": symbol,
            "score": score_col,
            "rows": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "avg_abs_return_bps": np.nan,
        }

    recalls = []
    for label in (-1, 1):
        part = directional[directional["actual"] == label]
        if not part.empty:
            recalls.append(float((part["pred"] == label).mean()))

    return {
        "horizon_minutes": horizon,
        "segment": segment,
        "symbol": symbol,
        "score": score_col,
        "rows": int(len(directional)),
        "accuracy": float((directional["pred"] == directional["actual"]).mean()),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else np.nan,
        "avg_abs_return_bps": float(directional[return_col].abs().mean()),
    }


def top_strength_rows(
    frame: pd.DataFrame,
    score_col: str,
    return_col: str,
    neutral_bps: float,
    horizon: int,
) -> list[dict[str, Any]]:
    base = frame[[score_col, return_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    base = base[base[score_col] != 0]
    if base.empty:
        return []
    base["strength"] = base[score_col].abs()
    q = base["strength"].quantile([0.50, 0.75, 0.90]).to_dict()
    rows = []
    for name, threshold in [
        ("all", 0.0),
        ("top50", q.get(0.50, 0.0)),
        ("top25", q.get(0.75, 0.0)),
        ("top10", q.get(0.90, 0.0)),
    ]:
        subset = frame[frame[score_col].abs() >= threshold]
        row = metric_row(subset, score_col, return_col, neutral_bps, horizon, f"strength_{name}")
        row["min_abs_score"] = float(threshold)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot Delta Direction V2 research")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60])
    parser.add_argument("--neutral-bps", type=float, default=10.0)
    parser.add_argument("--tolerance-minutes", type=float, default=3.0)
    parser.add_argument("--heat-threshold", type=float, default=80.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    raw = pd.read_sql_query(
        """
        SELECT generated_at, symbol, current_price, payload_json
        FROM radar_final_observations
        ORDER BY symbol, generated_at
        """,
        con,
    )
    con.close()

    if raw.empty:
        raise SystemExit("radar_final_observations is empty")

    raw["generated_at"] = pd.to_datetime(raw["generated_at"], utc=True, errors="coerce")
    raw["current_price"] = pd.to_numeric(raw["current_price"], errors="coerce")
    raw["market_heat_score"] = raw["payload_json"].map(extract_market_heat)
    extracted = pd.DataFrame(raw["payload_json"].map(extract_snapshot_features).tolist(), index=raw.index)
    frame = pd.concat([raw.drop(columns=["payload_json"]), extracted], axis=1)
    frame = frame.dropna(subset=["generated_at", "symbol", "current_price"])

    parts = [
        add_v2_scores(group, args.window)
        for _, group in frame.groupby("symbol", sort=False)
    ]
    enriched = pd.concat(parts, ignore_index=True)

    score_cols = [
        "score_current_volume",
        "score_current_pull",
        "score_contrarian_volume_flow",
        "score_contrarian_pull_flow",
        "score_static",
        "score_contrarian_flow",
        "score_static_plus_contrarian_flow",
    ]

    rows: list[dict[str, Any]] = []

    for horizon in args.horizons:
        enriched = attach_future_price(enriched, horizon, args.tolerance_minutes)
        ret = f"future_return_bps_{horizon}m"

        for score in score_cols:
            rows.append(metric_row(enriched, score, ret, args.neutral_bps, horizon, "all"))
            rows.extend(top_strength_rows(enriched, score, ret, args.neutral_bps, horizon))

            heat = enriched[enriched["market_heat_score"] >= args.heat_threshold]
            rows.append(
                metric_row(
                    heat,
                    score,
                    ret,
                    args.neutral_bps,
                    horizon,
                    f"market_heat_ge_{args.heat_threshold:g}",
                )
            )

            for symbol, sg in enriched.groupby("symbol", sort=False):
                rows.append(
                    metric_row(
                        sg,
                        score,
                        ret,
                        args.neutral_bps,
                        horizon,
                        "by_symbol",
                        str(symbol),
                    )
                )

    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "metrics.csv", index=False)

    summary = {
        "status": "research_only",
        "window_snapshots": args.window,
        "horizons_minutes": args.horizons,
        "neutral_bps": args.neutral_bps,
        "heat_threshold": args.heat_threshold,
        "rows": int(len(enriched)),
        "market_heat_available_rows": int(enriched["market_heat_score"].notna().sum()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    show = out.copy()
    for col in ["accuracy", "balanced_accuracy"]:
        show[col] = show[col].round(4)
    show["avg_abs_return_bps"] = show["avg_abs_return_bps"].round(2)

    print("\nSNAPSHOT DELTA DIRECTION V2")
    print(json.dumps(summary, indent=2))

    print("\nGLOBAL + STRENGTH + MARKET HEAT")
    main_segments = show[show["symbol"] == "ALL"].copy()
    print(
        main_segments.sort_values(
            ["horizon_minutes", "segment", "balanced_accuracy"],
            ascending=[True, True, False],
        ).to_string(index=False)
    )

    print("\nBY SYMBOL — STATIC+CONTRARIAN FLOW")
    by_symbol = show[
        (show["segment"] == "by_symbol")
        & (show["score"] == "score_static_plus_contrarian_flow")
    ]
    print(by_symbol.sort_values(["horizon_minutes", "balanced_accuracy"], ascending=[True, False]).to_string(index=False))

    print(f"\nreports: {args.out}")


if __name__ == "__main__":
    main()
