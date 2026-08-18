from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import textara_api as api


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_TIMEFRAME = "24h"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reconstruct historical LiqHeat liquidation pressure using the exact "
            "current production squeeze model and historical Supabase liq_logging payloads."
        )
    )
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--limit-per-symbol", type=int, default=1000)
    p.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    p.add_argument(
        "--output",
        default="data/research/liquidation_pressure/historical_reconstruction_probe.parquet",
    )
    return p.parse_args()


def fetch_rows(symbol: str, timeframe: str, since_iso: str, limit: int) -> list[dict[str, Any]]:
    response = (
        api.supabase
        .table("liq_logging")
        .select(
            "id,logged_at,symbol,timeframe,current_price,"
            "liquidation_count,price_min,price_max,payload"
        )
        .eq("symbol", symbol)
        .eq("timeframe", timeframe)
        .gte("logged_at", since_iso)
        .order("logged_at", desc=False)
        .limit(limit)
        .execute()
    )
    return list(response.data or [])


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

    out = pd.DataFrame({
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
    })

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


def main() -> int:
    args = parse_args()
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    since_iso = since.isoformat()

    print("=" * 92)
    print("HISTORICAL LIQUIDATION PRESSURE RECONSTRUCTION PROBE")
    print("=" * 92)
    print(f"Timeframe : {args.timeframe}")
    print(f"Since     : {since_iso}")
    print(f"Symbols   : {args.symbols}")
    print("Method    : exact current production squeeze model + historical liq_logging payload")
    print()

    pieces: list[pd.DataFrame] = []
    for symbol in args.symbols:
        rows = fetch_rows(symbol.upper(), args.timeframe, since_iso, args.limit_per_symbol)
        recon = reconstruct_batch(rows)
        print(f"{symbol.upper():8s} fetched={len(rows):5d} reconstructed={len(recon):5d}")
        if not recon.empty:
            pieces.append(recon)

    if not pieces:
        raise RuntimeError("No historical rows could be reconstructed.")

    out = pd.concat(pieces, ignore_index=True).sort_values(["symbol", "logged_at"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False, compression="zstd")

    print("\n=== RECONSTRUCTION SUMMARY ===")
    summary = (
        out.groupby("symbol")
        .agg(
            n=("snapshot_id", "size"),
            first=("logged_at", "min"),
            last=("logged_at", "max"),
            pressure_mean=("liquidation_pressure_score", "mean"),
            pressure_p50=("liquidation_pressure_score", "median"),
            pressure_max=("liquidation_pressure_score", "max"),
            signed_mean=("signed_pressure", "mean"),
            direction_conf_mean=("direction_confidence", "mean"),
        )
    )
    print(summary.round(4).to_string())

    print("\n=== SAMPLE ===")
    cols = [
        "logged_at", "symbol", "current_price", "liquidation_pressure_score",
        "raw_prediction", "direction_confidence", "signed_pressure",
    ]
    print(out[cols].tail(20).round(4).to_string(index=False))

    print(f"\nSaved: {output}")
    print("\nIMPORTANT: these are reconstructed scores from the CURRENT production model.")
    print("For strict historical validation, model-training chronology must be audited to avoid look-ahead leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
