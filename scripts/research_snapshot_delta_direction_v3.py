#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB = PROJECT_ROOT / "data" / "monitoring" / "radar_v2_performance.sqlite"
OUT = PROJECT_ROOT / "data" / "reports" / "snapshot_delta_direction_v3"
SYMBOLS = {"BTCUSDT", "XAUUSDT"}
WINDOW = 10
HORIZON_MINUTES = 60
FIRST_HIT_BPS = 25.0
RANDOM_BASELINE = 0.50


def sf(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def parse_payload(v: Any) -> dict[str, Any]:
    if isinstance(v, dict):
        return v
    try:
        x = json.loads(v or "{}")
    except Exception:
        return {}
    return x if isinstance(x, dict) else {}


def extract(v: Any) -> dict[str, float]:
    item = parse_payload(v)
    topo = item.get("topology") or {}
    ai = item.get("ai_market_risk") or {}

    uv = sf(topo.get("upper_pool_volume"))
    lv = sf(topo.get("lower_pool_volume"))
    ud = sf(topo.get("upper_distance_pct"))
    ld = sf(topo.get("lower_distance_pct"))
    heat = sf(ai.get("risk_score"))

    total = uv + lv
    volume_balance = (uv - lv) / total if np.isfinite(total) and total > 0 else np.nan

    eps = 1e-6
    up_pull = uv / max(ud, eps) if np.isfinite(uv) and np.isfinite(ud) and ud >= 0 else np.nan
    dn_pull = lv / max(ld, eps) if np.isfinite(lv) and np.isfinite(ld) and ld >= 0 else np.nan
    pull_total = up_pull + dn_pull
    pull_balance = (up_pull - dn_pull) / pull_total if np.isfinite(pull_total) and pull_total > 0 else np.nan

    return {
        "market_heat": heat,
        "volume_balance": volume_balance,
        "pull_balance": pull_balance,
    }


def add_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("generated_at").copy()
    g["volume_flow"] = g["volume_balance"] - g["volume_balance"].shift(WINDOW - 1)
    g["pull_flow"] = g["pull_balance"] - g["pull_balance"].shift(WINDOW - 1)

    g["score_contrarian_volume_flow"] = -g["volume_flow"]
    g["score_contrarian_pull_flow"] = -g["pull_flow"]
    g["score_contrarian_flow"] = g[[
        "score_contrarian_volume_flow",
        "score_contrarian_pull_flow",
    ]].mean(axis=1, skipna=False)

    static = g[["volume_balance", "pull_balance"]].mean(axis=1, skipna=False)
    g["score_static_plus_contrarian_flow"] = (static + g["score_contrarian_flow"]) / 2.0
    return g


def attach_targets(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("generated_at").reset_index(drop=True).copy()
    terminal = np.full(len(g), np.nan)
    first_hit = np.full(len(g), np.nan)

    times = g["generated_at"].to_numpy()
    prices = g["current_price"].to_numpy(float)

    for i in range(len(g)):
        t0 = g.loc[i, "generated_at"]
        p0 = prices[i]
        if not np.isfinite(p0) or p0 <= 0:
            continue

        end = t0 + pd.Timedelta(minutes=HORIZON_MINUTES)
        j = i + 1
        last_price = np.nan
        hit = 0
        while j < len(g) and g.loc[j, "generated_at"] <= end:
            p = prices[j]
            if np.isfinite(p):
                last_price = p
                if hit == 0:
                    bps = (p / p0 - 1.0) * 10000.0
                    if bps >= FIRST_HIT_BPS:
                        hit = 1
                    elif bps <= -FIRST_HIT_BPS:
                        hit = -1
            j += 1

        if np.isfinite(last_price):
            r = (last_price / p0 - 1.0) * 10000.0
            terminal[i] = 1 if r > 0 else (-1 if r < 0 else 0)
        if hit != 0:
            first_hit[i] = hit

    g["target_terminal_60m"] = terminal
    g["target_first_hit_25bps_60m"] = first_hit
    return g


def ba(actual: pd.Series, pred: pd.Series) -> float:
    vals = []
    for label in (-1, 1):
        m = actual == label
        if m.any():
            vals.append(float((pred[m] == label).mean()))
    return float(np.mean(vals)) if vals else np.nan


def evaluate(frame: pd.DataFrame, symbol: str, target: str, score: str, filt: str, mask: pd.Series) -> dict[str, Any]:
    x = frame[mask & frame[target].isin([-1, 1]) & frame[score].notna() & (frame[score] != 0)].copy()
    if x.empty:
        return {
            "symbol": symbol, "target": target, "signal": score, "filter": filt,
            "rows": 0, "coverage": 0.0, "accuracy": np.nan,
            "balanced_accuracy": np.nan, "lift_vs_random": np.nan,
            "avg_abs_return_bps": np.nan,
        }
    pred = np.sign(x[score]).astype(int)
    actual = x[target].astype(int)
    acc = float((pred == actual).mean())
    return {
        "symbol": symbol,
        "target": target,
        "signal": score,
        "filter": filt,
        "rows": int(len(x)),
        "coverage": float(len(x) / max(1, int((mask & frame[target].isin([-1, 1])).sum()))),
        "accuracy": acc,
        "balanced_accuracy": ba(actual, pred),
        "lift_vs_random": acc - RANDOM_BASELINE,
        "avg_abs_return_bps": float(x["future_return_bps_60m"].abs().mean()) if "future_return_bps_60m" in x else np.nan,
    }


def attach_terminal_return(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("generated_at").reset_index(drop=True).copy()
    out = np.full(len(g), np.nan)
    for i in range(len(g)):
        t0 = g.loc[i, "generated_at"]
        p0 = g.loc[i, "current_price"]
        target = t0 + pd.Timedelta(minutes=HORIZON_MINUTES)
        future = g[(g["generated_at"] >= target - pd.Timedelta(minutes=3)) &
                   (g["generated_at"] <= target + pd.Timedelta(minutes=3))]
        if future.empty or not np.isfinite(p0) or p0 <= 0:
            continue
        row = future.iloc[(future["generated_at"] - target).abs().argmin()]
        out[i] = (float(row["current_price"]) / float(p0) - 1.0) * 10000.0
    g["future_return_bps_60m"] = out
    return g


def main() -> None:
    con = sqlite3.connect(DB)
    raw = pd.read_sql_query(
        """
        SELECT generated_at, symbol, current_price, payload_json
        FROM radar_final_observations
        WHERE symbol IN ('BTCUSDT','XAUUSDT')
        ORDER BY symbol, generated_at
        """, con)
    con.close()

    raw["generated_at"] = pd.to_datetime(raw["generated_at"], utc=True, errors="coerce")
    raw["current_price"] = pd.to_numeric(raw["current_price"], errors="coerce")
    raw = raw.dropna(subset=["generated_at", "symbol", "current_price"])

    ext = pd.DataFrame(raw["payload_json"].map(extract).tolist(), index=raw.index)
    frame = pd.concat([raw.drop(columns=["payload_json"]), ext], axis=1)

    parts = []
    for symbol, g in frame.groupby("symbol", sort=False):
        z = add_features(g)
        z = attach_terminal_return(z)
        z = attach_targets(z)
        parts.append(z)
    frame = pd.concat(parts, ignore_index=True)

    signals = [
        "score_contrarian_pull_flow",
        "score_contrarian_volume_flow",
        "score_contrarian_flow",
        "score_static_plus_contrarian_flow",
    ]
    targets = ["target_terminal_60m", "target_first_hit_25bps_60m"]

    rows = []
    for symbol in sorted(SYMBOLS):
        s = frame[frame["symbol"] == symbol].copy()
        for score in signals:
            valid_strength = s[score].abs().dropna()
            if valid_strength.empty:
                continue
            q75 = float(valid_strength.quantile(0.75))
            q90 = float(valid_strength.quantile(0.90))
            filters = {
                "heat_ge_80": s["market_heat"] >= 80,
                "heat_ge_90": s["market_heat"] >= 90,
                "heat_ge_80_top25": (s["market_heat"] >= 80) & (s[score].abs() >= q75),
                "heat_ge_80_top10": (s["market_heat"] >= 80) & (s[score].abs() >= q90),
                "heat_ge_90_top10": (s["market_heat"] >= 90) & (s[score].abs() >= q90),
            }
            for target in targets:
                for fname, mask in filters.items():
                    rows.append(evaluate(s, symbol, target, score, fname, mask))

    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "snapshot_delta_direction_v3.csv", index=False)

    best = result[result["rows"] >= 50].sort_values(["accuracy", "rows"], ascending=[False, False]).head(25)
    promising = result[(result["rows"] >= 100) & (result["accuracy"] >= 0.60)].sort_values(["accuracy", "rows"], ascending=[False, False])

    payload = {
        "status": "research_only",
        "symbols": sorted(SYMBOLS),
        "window_snapshots": WINDOW,
        "horizon_minutes": HORIZON_MINUTES,
        "first_hit_bps": FIRST_HIT_BPS,
        "random_baseline": RANDOM_BASELINE,
        "rows": int(len(frame)),
        "results": json.loads(result.to_json(orient="records")),
        "best_configurations": json.loads(best.to_json(orient="records")),
        "promising_configurations": json.loads(promising.to_json(orient="records")),
    }
    (OUT / "snapshot_delta_direction_v3.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# Snapshot Delta Direction V3",
        "",
        "status: research_only",
        "",
        "## Best configurations (rows >= 50)",
        "",
        best.to_markdown(index=False) if not best.empty else "No qualifying configurations.",
        "",
        "## Promising configurations (rows >= 100, accuracy >= 60%)",
        "",
        promising.to_markdown(index=False) if not promising.empty else "NO CONFIGURATION PASSED THRESHOLD",
    ]
    (OUT / "snapshot_delta_direction_v3.md").write_text("\n".join(md))

    print("\nSNAPSHOT DELTA DIRECTION V3")
    print(f"status=research_only rows={len(frame)} window={WINDOW} horizon={HORIZON_MINUTES}m first_hit={FIRST_HIT_BPS}bps")
    print("\nBEST CONFIGURATIONS (rows >= 50)")
    if best.empty:
        print("none")
    else:
        show = best.copy()
        for c in ["coverage", "accuracy", "balanced_accuracy", "lift_vs_random"]:
            show[c] = show[c].round(4)
        show["avg_abs_return_bps"] = show["avg_abs_return_bps"].round(2)
        print(show.to_string(index=False))

    print("\nPROMISING CONFIGURATIONS (rows >= 100, accuracy >= 60%)")
    if promising.empty:
        print("NO CONFIGURATION PASSED THRESHOLD")
    else:
        show = promising.copy()
        for c in ["coverage", "accuracy", "balanced_accuracy", "lift_vs_random"]:
            show[c] = show[c].round(4)
        show["avg_abs_return_bps"] = show["avg_abs_return_bps"].round(2)
        print(show.to_string(index=False))

    print(f"\nreports: {OUT}")


if __name__ == "__main__":
    main()
