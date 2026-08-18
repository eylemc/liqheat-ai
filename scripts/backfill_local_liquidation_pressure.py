from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catboost import CatBoostClassifier

DEFAULT_INPUT = Path("data/features/liq_topology_v2_ml_features.parquet")
DEFAULT_OUTPUT = Path("data/research/liquidation_pressure/local_historical_pressure.parquet")
DEFAULT_FEATURES_OUTPUT = Path("data/research/liquidation_pressure/local_historical_pressure_features.parquet")
MODEL_DIR = PROJECT_ROOT / "models" / "squeeze_v1"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_TIMEFRAME = "24h"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast local historical liquidation-pressure backfill from prebuilt topology ML features.")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    p.add_argument("--batch-size", type=int, default=100_000)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--features-output", default=str(DEFAULT_FEATURES_OUTPUT))
    return p.parse_args()


def rolling_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("logged_at").copy()
    t = pd.to_datetime(g["logged_at"], utc=True)
    ser = pd.Series(pd.to_numeric(g["signed_pressure"], errors="coerce").to_numpy(), index=pd.DatetimeIndex(t))

    for minutes in (30, 60, 90, 120):
        win = f"{minutes}min"
        roll = ser.rolling(win, min_periods=1)
        g[f"mean_{minutes}m"] = roll.mean().to_numpy()
        g[f"abs_mean_{minutes}m"] = ser.abs().rolling(win, min_periods=1).mean().to_numpy()
        g[f"peak_abs_{minutes}m"] = ser.abs().rolling(win, min_periods=1).max().to_numpy()
        g[f"sample_count_{minutes}m"] = roll.count().to_numpy()

        sign = np.sign(ser)
        pos = (sign > 0).astype(float).rolling(win, min_periods=1).mean()
        neg = (sign < 0).astype(float).rolling(win, min_periods=1).mean()
        g[f"persistence_{minutes}m"] = np.maximum(pos.to_numpy(), neg.to_numpy()) * 100.0
        g[f"dominant_direction_{minutes}m"] = np.where(
            pos.to_numpy() > neg.to_numpy(), "UP",
            np.where(neg.to_numpy() > pos.to_numpy(), "DOWN", "TIE"),
        )

        # Time-window flip count. Historical stream is approximately minute sampled.
        flips = (sign != sign.shift(1)).astype(float)
        if len(flips):
            flips.iloc[0] = 0.0
        g[f"flips_{minutes}m"] = flips.rolling(win, min_periods=1).sum().to_numpy()

        # Fast trend proxy: current pressure minus pressure at/near window start, per minute.
        prior = ser.reindex(ser.index - pd.Timedelta(minutes=minutes), method="nearest", tolerance=pd.Timedelta(minutes=2))
        g[f"slope_{minutes}m_per_min"] = (ser.to_numpy() - prior.to_numpy()) / float(minutes)

    g["acceleration_30_vs_120"] = g["mean_30m"] - g["mean_120m"]
    return g


def main() -> int:
    args = parse_args()
    started = time.time()
    input_path = Path(args.input)
    output_path = Path(args.output)
    features_path = Path(args.features_output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with open(MODEL_DIR / "features.json", encoding="utf-8") as f:
        feature_columns = json.load(f)

    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "model.cbm"))

    wanted_symbols = {s.upper() for s in args.symbols}
    columns = list(dict.fromkeys([
        "id", "logged_at", "symbol", "timeframe", "current_price", "liquidation_count",
        *feature_columns,
    ]))

    print("=" * 96)
    print("FAST LOCAL HISTORICAL LIQUIDATION PRESSURE BACKFILL")
    print("=" * 96)
    print("Input     :", input_path)
    print("Symbols   :", sorted(wanted_symbols))
    print("Timeframe :", args.timeframe)
    print("Batch size:", f"{args.batch_size:,}")
    print("Model     :", MODEL_DIR / "model.cbm")
    print()

    print("Loading local topology ML features...")
    df = pd.read_parquet(input_path, columns=columns)
    df["symbol"] = df["symbol"].astype(str)
    df["timeframe"] = df["timeframe"].astype(str)
    df = df[df["symbol"].isin(wanted_symbols) & df["timeframe"].eq(args.timeframe)].copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, errors="coerce")
    df = df.sort_values(["symbol", "logged_at", "id"], kind="mergesort").reset_index(drop=True)

    print(f"Selected  : {len(df):,} rows")
    print("Range     :", df["logged_at"].min(), "->", df["logged_at"].max())
    print()

    classes = [int(v) for v in model.classes_]
    idx = {v: i for i, v in enumerate(classes)}
    chunks = []

    for start in range(0, len(df), args.batch_size):
        end = min(start + args.batch_size, len(df))
        batch = df.iloc[start:end]
        probabilities = model.predict_proba(batch[feature_columns])
        long_p = probabilities[:, idx[-1]].astype(float)
        none_p = probabilities[:, idx[0]].astype(float)
        short_p = probabilities[:, idx[1]].astype(float)
        event_p = long_p + short_p
        short_side = short_p >= long_p
        direction_p = np.where(short_side, short_p, long_p)
        direction_conf = np.divide(direction_p, event_p, out=np.zeros_like(direction_p), where=event_p > 0)

        out = pd.DataFrame({
            "snapshot_id": batch["id"].astype(str).to_numpy(),
            "logged_at": batch["logged_at"].to_numpy(),
            "symbol": batch["symbol"].to_numpy(),
            "timeframe": batch["timeframe"].to_numpy(),
            "current_price": pd.to_numeric(batch["current_price"], errors="coerce").to_numpy(),
            "liquidation_pressure": event_p,
            "liquidation_pressure_score": event_p * 100.0,
            "raw_prediction": np.where(short_side, "SHORT_SQUEEZE", "LONG_SQUEEZE"),
            "direction_confidence": direction_conf,
            "long_squeeze_probability": long_p,
            "no_event_probability": none_p,
            "short_squeeze_probability": short_p,
            "signed_pressure": np.where(short_side, event_p * 100.0, -event_p * 100.0),
            "liquidation_count": pd.to_numeric(batch["liquidation_count"], errors="coerce").to_numpy(),
        })
        chunks.append(out)
        elapsed = max(time.time() - started, 0.001)
        print(f"Inference {end:,}/{len(df):,} ({end/len(df)*100:5.1f}%) rate={end/elapsed:,.0f} rows/s")

    raw = pd.concat(chunks, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output_path, index=False, compression="zstd")

    print("\nBuilding 30/60/90/120m rolling features...")
    feature_parts = []
    for symbol, group in raw.groupby("symbol", sort=False):
        print("Features  :", symbol, f"({len(group):,} rows)")
        feature_parts.append(rolling_features(group))
    features = pd.concat(feature_parts, ignore_index=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_path, index=False, compression="zstd")

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
    print("\nRaw saved     :", output_path)
    print("Features saved:", features_path)
    print(f"Elapsed       : {time.time() - started:,.1f}s")
    print("\nNOTE: these scores use the CURRENT production squeeze model over historical prebuilt features.")
    print("Strict Bias V2 validation still requires chronology/leakage audit and walk-forward/holdout evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
