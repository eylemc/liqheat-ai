from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import textara_api as api

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_START = "2026-03-30T00:00:00+00:00"
DEFAULT_TIMEFRAME = "24h"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Backfill historical LiqHeat liquidation pressure using the exact current "
            "production squeeze model over historical Supabase liq_logging payloads."
        )
    )
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=None, help="Optional UTC ISO timestamp; default = now")
    p.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument(
        "--output",
        default="data/research/liquidation_pressure/historical_pressure_full.parquet",
    )
    p.add_argument(
        "--features-output",
        default="data/research/liquidation_pressure/historical_pressure_features.parquet",
    )
    return p.parse_args()


def fetch_page(
    symbol: str,
    timeframe: str,
    start_iso: str,
    end_iso: str | None,
    offset: int,
    page_size: int,
) -> list[dict[str, Any]]:
    q = (
        api.supabase
        .table("liq_logging")
        .select(
            "id,logged_at,symbol,timeframe,current_price,"
            "liquidation_count,price_min,price_max,payload"
        )
        .eq("symbol", symbol)
        .eq("timeframe", timeframe)
        .gte("logged_at", start_iso)
        .order("logged_at", desc=False)
        .range(offset, offset + page_size - 1)
    )
    if end_iso:
        q = q.lte("logged_at", end_iso)
    r = q.execute()
    return list(r.data or [])


def reconstruct_batch(rows: list[dict[str, Any]]) -> pd.DataFrame:
    topology_rows: list[dict[str, Any]] = []
    source_by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        feature = api.topology_feature_from_live_row(row)
        if feature is None:
            continue
        topology_rows.append(feature)
        source_by_id[str(feature["id"])] = row

    if not topology_rows:
        return pd.DataFrame()

    frame = api.add_ml_features(topology_rows)
    X = frame[api.FEATURE_COLUMNS].copy()
    probabilities = api.model.predict_proba(X)
    classes = [int(v) for v in api.model.classes_]
    idx = {v: i for i, v in enumerate(classes)}

    long_p = probabilities[:, idx[-1]].astype(float)
    none_p = probabilities[:, idx[0]].astype(float)
    short_p = probabilities[:, idx[1]].astype(float)
    event_p = long_p + short_p
    raw_prediction = np.where(short_p >= long_p, "SHORT_SQUEEZE", "LONG_SQUEEZE")
    direction_p = np.where(short_p >= long_p, short_p, long_p)
    direction_conf = np.divide(
        direction_p,
        event_p,
        out=np.zeros_like(direction_p),
        where=event_p > 0,
    )

    out = pd.DataFrame(
        {
            "snapshot_id": frame["id"].astype(str),
            "logged_at": pd.to_datetime(frame["logged_at"], utc=True, errors="coerce"),
            "symbol": frame["symbol"].astype(str),
            "timeframe": frame["timeframe"].astype(str),
            "current_price": pd.to_numeric(frame["current_price"], errors="coerce"),
            "liquidation_pressure": event_p,
            "liquidation_pressure_score": event_p * 100.0,
            "raw_prediction": raw_prediction,
            "direction_confidence": direction_conf,
            "long_squeeze_probability": long_p,
            "no_event_probability": none_p,
            "short_squeeze_probability": short_p,
        }
    )
    out["signed_pressure"] = np.where(
        out["raw_prediction"].eq("SHORT_SQUEEZE"),
        out["liquidation_pressure_score"],
        -out["liquidation_pressure_score"],
    )
    out["liquidation_count"] = [
        source_by_id.get(sid, {}).get("liquidation_count")
        for sid in out["snapshot_id"]
    ]
    return out


def append_frame(writer: pq.ParquetWriter | None, frame: pd.DataFrame, path: Path) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def time_slope(values: pd.Series, times: pd.Series) -> float:
    if len(values) < 2:
        return np.nan
    x = (times - times.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 60.0
    y = values.to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.nanmax(x[mask]) <= 0:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def add_window_features(group: pd.DataFrame, minutes: int) -> pd.DataFrame:
    g = group.sort_values("logged_at").copy()
    t = pd.to_datetime(g["logged_at"], utc=True)
    s = pd.to_numeric(g["signed_pressure"], errors="coerce")
    idx = pd.DatetimeIndex(t)
    ser = pd.Series(s.to_numpy(), index=idx)
    win = f"{minutes}min"

    g[f"mean_{minutes}m"] = ser.rolling(win, min_periods=1).mean().to_numpy()
    g[f"abs_mean_{minutes}m"] = ser.abs().rolling(win, min_periods=1).mean().to_numpy()
    g[f"peak_abs_{minutes}m"] = ser.abs().rolling(win, min_periods=1).max().to_numpy()
    g[f"sample_count_{minutes}m"] = ser.rolling(win, min_periods=1).count().to_numpy()

    sign = np.sign(ser)
    pos = (sign > 0).astype(float).rolling(win, min_periods=1).mean()
    neg = (sign < 0).astype(float).rolling(win, min_periods=1).mean()
    g[f"persistence_{minutes}m"] = np.maximum(pos.to_numpy(), neg.to_numpy()) * 100.0
    g[f"dominant_direction_{minutes}m"] = np.where(
        pos.to_numpy() > neg.to_numpy(), "UP",
        np.where(neg.to_numpy() > pos.to_numpy(), "DOWN", "TIE"),
    )

    flips = (sign != sign.shift(1)).astype(float)
    flips.iloc[0] = 0.0
    g[f"flips_{minutes}m"] = flips.rolling(win, min_periods=1).sum().to_numpy()

    slopes = []
    left = 0
    times = t.reset_index(drop=True)
    vals = s.reset_index(drop=True)
    for right in range(len(g)):
        cutoff = times.iloc[right] - pd.Timedelta(minutes=minutes)
        while left < right and times.iloc[left] < cutoff:
            left += 1
        slopes.append(time_slope(vals.iloc[left:right + 1], times.iloc[left:right + 1]))
    g[f"slope_{minutes}m_per_min"] = slopes
    return g


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for symbol, group in raw.groupby("symbol", sort=False):
        g = group.sort_values("logged_at").copy()
        for minutes in (30, 60, 90, 120):
            g = add_window_features(g, minutes)
        g["acceleration_30_vs_120"] = g["mean_30m"] - g["mean_120m"]
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    features_output = Path(args.features_output)
    tmp = output.with_suffix(output.suffix + ".part")
    tmp.unlink(missing_ok=True)

    print("=" * 96)
    print("FULL HISTORICAL LIQUIDATION PRESSURE BACKFILL")
    print("=" * 96)
    print("Symbols   :", [s.upper() for s in args.symbols])
    print("Timeframe :", args.timeframe)
    print("Start     :", args.start)
    print("End       :", args.end or "NOW")
    print("Page size :", args.page_size)
    print("Method    : current production squeeze model + historical liq_logging payload")
    print()

    writer: pq.ParquetWriter | None = None
    total_written = 0
    started = time.time()

    try:
        for symbol in [s.upper() for s in args.symbols]:
            offset = 0
            fetched_total = 0
            reconstructed_total = 0
            print(f"--- {symbol} ---")
            while True:
                rows = fetch_page(
                    symbol,
                    args.timeframe,
                    args.start,
                    args.end,
                    offset,
                    args.page_size,
                )
                if not rows:
                    break
                fetched_total += len(rows)
                recon = reconstruct_batch(rows)
                if not recon.empty:
                    writer = append_frame(writer, recon, tmp)
                    reconstructed_total += len(recon)
                    total_written += len(recon)
                offset += len(rows)
                if fetched_total % 10000 < args.page_size:
                    elapsed = max(0.001, time.time() - started)
                    print(
                        f"{symbol}: fetched={fetched_total:,} reconstructed={reconstructed_total:,} "
                        f"total={total_written:,} rate={total_written/elapsed:,.1f} rows/s"
                    )
                if len(rows) < args.page_size:
                    break
            print(f"{symbol}: DONE fetched={fetched_total:,} reconstructed={reconstructed_total:,}")
    finally:
        if writer is not None:
            writer.close()

    if not tmp.exists():
        raise RuntimeError("No rows were reconstructed; output was not created.")
    tmp.replace(output)

    print("\nBuilding rolling research features...")
    raw = pd.read_parquet(output)
    raw = raw.sort_values(["symbol", "logged_at"]).drop_duplicates("snapshot_id", keep="last")
    features = build_features(raw)
    features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_output, index=False, compression="zstd")

    print("\n=== SUMMARY ===")
    summary = raw.groupby("symbol").agg(
        n=("snapshot_id", "size"),
        first=("logged_at", "min"),
        last=("logged_at", "max"),
        pressure_mean=("liquidation_pressure_score", "mean"),
        signed_mean=("signed_pressure", "mean"),
        direction_conf_mean=("direction_confidence", "mean"),
    )
    print(summary.to_string())
    print(f"\nRaw saved     : {output}")
    print(f"Features saved: {features_output}")
    print(f"Elapsed       : {time.time() - started:,.1f}s")
    print("\nIMPORTANT: scores are reconstructed with the CURRENT production squeeze model.")
    print("Strict Bias V2 validation must audit model-training chronology and use walk-forward/holdout splits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
