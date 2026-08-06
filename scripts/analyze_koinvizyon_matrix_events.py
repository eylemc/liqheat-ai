#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TOPOLOGY_PATH = Path("data/features/liq_topology_v2_ml_features.parquet")
MARKET_ROOT = Path("data/market/binance-futures-um")
OUT = Path("data/reports/koinvizyon_matrix_event_study")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--matrix-len", type=int, default=20)
    p.add_argument("--horizons-hours", default="1,2,4")
    p.add_argument("--stop-bps", type=float, default=50.0)
    p.add_argument("--take-bps", type=float, default=50.0)
    p.add_argument("--topology-tolerance-minutes", type=int, default=90)
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

    out = pd.DataFrame({
        "symbol": g["symbol"].astype("string"),
        "timeframe": g["timeframe"].astype("string"),
        "open_time": pd.to_datetime(g["open_time"], utc=True).astype("datetime64[ns, UTC]"),
        "close_time": pd.to_datetime(g["close_time"], utc=True).astype("datetime64[ns, UTC]"),
        "available_at": pd.to_datetime(g["close_time"], utc=True).astype("datetime64[ns, UTC]"),
        "open": pd.to_numeric(g["open"], errors="coerce"),
        "high": pd.to_numeric(g["high"], errors="coerce"),
        "low": pd.to_numeric(g["low"], errors="coerce"),
        "close": pd.to_numeric(g["close"], errors="coerce"),
        "volume": pd.to_numeric(g["volume"], errors="coerce"),
        "matrix_source": source,
        "matrix_vwma": ma,
        "matrix_upper": upper,
        "matrix_lower": lower,
        "matrix_trend": trend,
        "matrix_flip": flip,
        "matrix_long_flip": long_flip,
        "matrix_short_flip": short_flip,
        "matrix_distance_to_vwma_pct": (source / ma - 1.0) * 100.0,
        "matrix_channel_width_pct": (upper / lower - 1.0) * 100.0,
    })
    out["flip_type"] = np.select(
        [out["matrix_long_flip"].eq(1), out["matrix_short_flip"].eq(1)],
        ["SHORT_TO_LONG", "LONG_TO_SHORT"],
        default="NONE",
    )
    return out


def load_matrix(a: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for sym in SYMBOLS:
        path = MARKET_ROOT / sym / a.timeframe / f"{sym}-{a.timeframe}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = pd.read_parquet(path)
        raw = raw[raw["is_complete"].fillna(False)].copy()
        frames.append(compute_matrix(raw, a.matrix_len))
    return pd.concat(frames, ignore_index=True)


def add_forward_outcomes(matrix: pd.DataFrame, horizons: list[int], stop_bps: float, take_bps: float) -> pd.DataFrame:
    pieces = []
    for sym, g in matrix.groupby("symbol", sort=False):
        g = g.sort_values("available_at").copy().reset_index(drop=True)
        times = g["available_at"].astype("int64").to_numpy()
        close = g["close"].to_numpy(float)
        high = g["high"].to_numpy(float)
        low = g["low"].to_numpy(float)
        n = len(g)

        for h in horizons:
            end_ns = int(pd.Timedelta(hours=h).value)
            ret = np.full(n, np.nan)
            mfe_long = np.full(n, np.nan)
            mae_long = np.full(n, np.nan)
            stop_long = np.full(n, np.nan)
            take_long = np.full(n, np.nan)
            stop_short = np.full(n, np.nan)
            take_short = np.full(n, np.nan)

            for i in range(n):
                j = np.searchsorted(times, times[i] + end_ns, side="right")
                if j <= i + 1 or not np.isfinite(close[i]) or close[i] <= 0:
                    continue
                end_price = close[j - 1]
                path_high = (high[i + 1:j] / close[i] - 1.0) * 10000.0
                path_low = (low[i + 1:j] / close[i] - 1.0) * 10000.0
                ret[i] = (end_price / close[i] - 1.0) * 10000.0
                mfe_long[i] = np.nanmax(path_high)
                mae_long[i] = np.nanmin(path_low)
                stop_long[i] = float(mae_long[i] <= -stop_bps)
                take_long[i] = float(mfe_long[i] >= take_bps)
                stop_short[i] = float(mfe_long[i] >= stop_bps)
                take_short[i] = float(mae_long[i] <= -take_bps)

            g[f"ret_{h}h_bps"] = ret
            g[f"mfe_long_{h}h_bps"] = mfe_long
            g[f"mae_long_{h}h_bps"] = mae_long
            g[f"long_stop_hit_{h}h"] = stop_long
            g[f"long_take_hit_{h}h"] = take_long
            g[f"short_stop_hit_{h}h"] = stop_short
            g[f"short_take_hit_{h}h"] = take_short

        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def load_topology(a: argparse.Namespace) -> pd.DataFrame:
    schema = set(pq.read_schema(TOPOLOGY_PATH).names)
    wanted = [
        "logged_at", "symbol", "timeframe", "nearest_side",
        "topology_imbalance", "signed_distance_edge",
        "upper_pool_volume", "lower_pool_volume",
        "upper_distance_pct", "lower_distance_pct",
    ]
    missing = [c for c in wanted if c not in schema]
    if missing:
        raise RuntimeError(f"Missing topology columns: {missing}")

    df = pd.read_parquet(
        TOPOLOGY_PATH,
        columns=wanted,
        filters=[("timeframe", "==", a.timeframe)],
    )
    df = df[df["symbol"].astype(str).isin(SYMBOLS)].copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True).astype("datetime64[ns, UTC]")
    df["symbol"] = df["symbol"].astype("string")
    df = df.sort_values(["logged_at", "symbol"]).reset_index(drop=True)
    return df


def join_topology(matrix: pd.DataFrame, topology: pd.DataFrame, tolerance_minutes: int) -> pd.DataFrame:
    out = []
    tol = pd.Timedelta(minutes=tolerance_minutes)
    for sym, m in matrix.groupby("symbol", sort=False):
        t = topology[topology["symbol"] == sym].copy()
        m = m.copy()
        m["available_at"] = pd.to_datetime(m["available_at"], utc=True).astype("datetime64[ns, UTC]")
        t["logged_at"] = pd.to_datetime(t["logged_at"], utc=True).astype("datetime64[ns, UTC]")
        m = m.sort_values("available_at")
        t = t.sort_values("logged_at")
        joined = pd.merge_asof(
            m,
            t.drop(columns=["symbol", "timeframe"], errors="ignore"),
            left_on="available_at",
            right_on="logged_at",
            direction="backward",
            tolerance=tol,
            allow_exact_matches=True,
        )
        joined["topology_age_minutes"] = (
            joined["available_at"] - joined["logged_at"]
        ).dt.total_seconds() / 60.0
        out.append(joined)
    return pd.concat(out, ignore_index=True)


def topology_state(df: pd.DataFrame) -> pd.Series:
    imbal = pd.to_numeric(df["topology_imbalance"], errors="coerce")
    edge = pd.to_numeric(df["signed_distance_edge"], errors="coerce")
    upper = pd.to_numeric(df["upper_pool_volume"], errors="coerce")
    lower = pd.to_numeric(df["lower_pool_volume"], errors="coerce")

    score = np.zeros(len(df), dtype=float)
    score += np.where(imbal > 0, 1, np.where(imbal < 0, -1, 0))
    score += np.where(edge > 0, 1, np.where(edge < 0, -1, 0))
    score += np.where(upper > lower, 1, np.where(upper < lower, -1, 0))

    return pd.Series(
        np.where(score >= 2, "BULL", np.where(score <= -2, "BEAR", "NEUTRAL")),
        index=df.index,
        dtype="string",
    )


def summarize_group(df: pd.DataFrame, horizons: list[int], flip_mode: str | None = None) -> dict:
    out = {"count": int(len(df))}
    if flip_mode:
        out["event"] = flip_mode
    for h in horizons:
        r = pd.to_numeric(df[f"ret_{h}h_bps"], errors="coerce")
        mfe = pd.to_numeric(df[f"mfe_long_{h}h_bps"], errors="coerce")
        mae = pd.to_numeric(df[f"mae_long_{h}h_bps"], errors="coerce")
        out[f"{h}h"] = {
            "valid": int(r.notna().sum()),
            "mean_return_bps": float(r.mean()) if r.notna().any() else None,
            "median_return_bps": float(r.median()) if r.notna().any() else None,
            "positive_return_rate": float((r > 0).mean()) if r.notna().any() else None,
            "mfe_long_median_bps": float(mfe.median()) if mfe.notna().any() else None,
            "mae_long_median_bps": float(mae.median()) if mae.notna().any() else None,
            "long_stop_hit_rate": float(pd.to_numeric(df[f"long_stop_hit_{h}h"], errors="coerce").mean()),
            "long_take_hit_rate": float(pd.to_numeric(df[f"long_take_hit_{h}h"], errors="coerce").mean()),
            "short_stop_hit_rate": float(pd.to_numeric(df[f"short_stop_hit_{h}h"], errors="coerce").mean()),
            "short_take_hit_rate": float(pd.to_numeric(df[f"short_take_hit_{h}h"], errors="coerce").mean()),
        }
    return out


def main() -> None:
    a = parse_args()
    horizons = [int(x.strip()) for x in a.horizons_hours.split(",") if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    print("Computing Koinvizyon Matrix...")
    matrix = load_matrix(a)
    matrix = add_forward_outcomes(matrix, horizons, a.stop_bps, a.take_bps)

    flips = matrix[matrix["flip_type"].ne("NONE")].copy()
    flips.to_csv(OUT / "flip_events.csv", index=False)

    flip_summary = {
        "status": "research_only",
        "matrix_len": a.matrix_len,
        "timeframe": a.timeframe,
        "stop_bps": a.stop_bps,
        "take_bps": a.take_bps,
        "events": {
            event: summarize_group(g, horizons, event)
            for event, g in flips.groupby("flip_type", sort=True)
        },
        "by_symbol": {
            sym: {
                event: summarize_group(g2, horizons, event)
                for event, g2 in g.groupby("flip_type", sort=True)
            }
            for sym, g in flips.groupby("symbol", sort=True)
        },
    }
    (OUT / "flip_summary.json").write_text(json.dumps(flip_summary, indent=2, default=str), encoding="utf-8")

    print("Joining topology for regime study...")
    topology = load_topology(a)
    aligned = join_topology(matrix, topology, a.topology_tolerance_minutes)
    aligned["topology_state"] = topology_state(aligned)
    aligned["matrix_state"] = aligned["matrix_trend"].map({1: "BULL", -1: "BEAR", 0: "NEUTRAL"}).astype("string")
    aligned["alignment_state"] = aligned["matrix_state"] + "_MATRIX__" + aligned["topology_state"] + "_TOPOLOGY"

    regime_rows = []
    regime_summary = {
        "status": "research_only",
        "topology_tolerance_minutes": a.topology_tolerance_minutes,
        "overall": {},
        "by_symbol": {},
    }
    for state, g in aligned.groupby("alignment_state", dropna=False, sort=True):
        regime_summary["overall"][str(state)] = summarize_group(g, horizons)
        regime_rows.append({"scope": "ALL", "symbol": "ALL", "alignment_state": str(state), **summarize_group(g, horizons)})

    for sym, gsym in aligned.groupby("symbol", sort=True):
        regime_summary["by_symbol"][sym] = {}
        for state, g in gsym.groupby("alignment_state", dropna=False, sort=True):
            regime_summary["by_symbol"][sym][str(state)] = summarize_group(g, horizons)
            regime_rows.append({"scope": "SYMBOL", "symbol": sym, "alignment_state": str(state), **summarize_group(g, horizons)})

    (OUT / "regime_summary.json").write_text(json.dumps(regime_summary, indent=2, default=str), encoding="utf-8")
    pd.json_normalize(regime_rows, sep=".").to_csv(OUT / "regime_summary.csv", index=False)

    print("Done:", OUT)


if __name__ == "__main__":
    main()
