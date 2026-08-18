import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from scripts import matrix_true_backtest as mtb


# ============================================================
# CONFIG
# ============================================================

ONE_MINUTE_PATH = (
    Path("data/market/binance-futures-um")
    / "BTCUSDT"
    / "1m"
    / "BTCUSDT-1m.parquet"
)

HISTORICAL_PATH = Path(
    "data/research/btc_1m_historical_2022-10_to_2025-08.parquet"
)

OUTPUT = Path(
    "data/research/btc_matrix_1h_regime_test.csv"
)

# Pre-specified regime parameters.
# We will inspect continuous relationships too, so these are
# descriptive buckets, NOT optimized trading rules.
ER_LENGTH = 12
CROSS_WINDOW = 12

# Forward horizons in 1m bars
HORIZONS = {
    "4h": 240,
    "12h": 720,
    "24h": 1440,
}


# ============================================================
# LOAD DATA
# ============================================================

current = pd.read_parquet(ONE_MINUTE_PATH)

frames = []

if HISTORICAL_PATH.exists():
    historical = pd.read_parquet(HISTORICAL_PATH)
    frames.append(historical)

frames.append(current)

one_minute = pd.concat(
    frames,
    ignore_index=True,
)

one_minute["open_time"] = pd.to_datetime(
    one_minute["open_time"],
    utc=True,
)

one_minute = (
    one_minute
    .sort_values("open_time")
    .drop_duplicates(
        subset=["open_time"],
        keep="last",
    )
    .reset_index(drop=True)
)

print("\n=== 1M DATA ===")
print("Rows :", f"{len(one_minute):,}")
print(
    "Range:",
    one_minute["open_time"].min(),
    "->",
    one_minute["open_time"].max(),
)


# ============================================================
# BUILD 1H MATRIX
# ============================================================

candles = mtb.aggregate_candles(
    one_minute,
    "1h",
)

matrix = mtb.add_matrix(
    candles.copy(),
    "1h",
)

matrix = (
    matrix
    .sort_values("close_time")
    .reset_index(drop=True)
)

# Matrix flips
flips = matrix[
    matrix["flip"].fillna(0).ne(0)
].copy()

flips["side"] = np.where(
    flips["long_flip"].fillna(0).ne(0),
    "BUY",
    "SELL",
)

print("\n=== 1H MATRIX ===")
print("Candles :", len(matrix))
print("Flips   :", len(flips))
print(
    "Range   :",
    flips["close_time"].min(),
    "->",
    flips["close_time"].max(),
)


# ============================================================
# EFFICIENCY RATIO
# ============================================================

close = matrix["close"].astype(float)

directional_move = (
    close.diff(ER_LENGTH).abs()
)

path_distance = (
    close.diff()
    .abs()
    .rolling(ER_LENGTH)
    .sum()
)

matrix["er"] = (
    directional_move
    / path_distance.replace(0, np.nan)
)


# ============================================================
# VWMA CROSS COUNT
# ============================================================

dist = (
    matrix["close"].astype(float)
    - matrix["vwma"].astype(float)
)

sign = np.sign(dist)

previous_sign = sign.shift(1)

cross = (
    (sign != previous_sign)
    & sign.notna()
    & previous_sign.notna()
).astype(int)

matrix["vwma_cross"] = cross

matrix["vwma_cross_count"] = (
    matrix["vwma_cross"]
    .rolling(CROSS_WINDOW)
    .sum()
)


# ============================================================
# ADD OTHER DESCRIPTIVE FEATURES
# ============================================================

matrix["abs_vwma_dist"] = (
    matrix["distance_to_vwma_pct"]
    .astype(float)
    .abs()
)

matrix["channel_width"] = (
    matrix["channel_width_pct"]
    .astype(float)
)

# Net move / ATR-like realized movement proxy
matrix["realized_path_pct"] = (
    close.pct_change()
    .abs()
    .rolling(CROSS_WINDOW)
    .sum()
    * 100
)


# ============================================================
# MERGE FEATURES INTO FLIPS
# ============================================================

feature_cols = [
    "close_time",
    "er",
    "vwma_cross_count",
    "abs_vwma_dist",
    "channel_width",
    "realized_path_pct",
]

flips = flips.merge(
    matrix[feature_cols],
    on="close_time",
    how="left",
    suffixes=("", "_feature"),
)


# ============================================================
# FORWARD MFE / MAE / RETURN
# ============================================================

minute_times = (
    pd.to_datetime(
        one_minute["open_time"],
        utc=True
    )
    .dt.as_unit("ns")
    .astype("int64")
    .to_numpy()
)

minute_high = (
    one_minute["high"]
    .astype(float)
    .to_numpy()
)

minute_low = (
    one_minute["low"]
    .astype(float)
    .to_numpy()
)

minute_close = (
    one_minute["close"]
    .astype(float)
    .to_numpy()
)


def forward_stats(signal_time, entry, side, bars):
    """
    Forward excursion measured strictly AFTER the Matrix
    candle has closed.
    """

    signal_time = pd.Timestamp(signal_time)

    # Matrix close_time is xx:59:59.999.
    # First tradable minute is next minute.
    start_time = signal_time + pd.Timedelta(milliseconds=1)

    start_key = pd.Timestamp(
        start_time
    ).value

    start = np.searchsorted(
        minute_times,
        start_key,
        side="left",
    )

    end = min(
        start + bars,
        len(one_minute),
    )

    if start >= len(one_minute) or end <= start:
        return np.nan, np.nan, np.nan

    highs = minute_high[start:end]
    lows = minute_low[start:end]

    final_close = minute_close[end - 1]

    if side == "BUY":
        mfe = (
            highs.max() / entry - 1
        ) * 100

        mae = (
            lows.min() / entry - 1
        ) * 100

        ret = (
            final_close / entry - 1
        ) * 100

    else:
        mfe = (
            entry / lows.min() - 1
        ) * 100

        mae = -(
            highs.max() / entry - 1
        ) * 100

        ret = (
            entry / final_close - 1
        ) * 100

    return mfe, mae, ret


rows = []

for _, row in flips.iterrows():

    out = {
        "time": row["close_time"],
        "side": row["side"],
        "entry": float(row["close"]),
        "er": row["er"],
        "vwma_cross_count": row["vwma_cross_count"],
        "abs_vwma_dist": row["abs_vwma_dist"],
        "channel_width": row["channel_width"],
        "realized_path_pct": row["realized_path_pct"],
    }

    for name, bars in HORIZONS.items():

        mfe, mae, ret = forward_stats(
            row["close_time"],
            float(row["close"]),
            row["side"],
            bars,
        )

        out[f"mfe_{name}"] = mfe
        out[f"mae_{name}"] = mae
        out[f"ret_{name}"] = ret

    rows.append(out)

df = pd.DataFrame(rows)

df = df.dropna(
    subset=[
        "er",
        "vwma_cross_count",
        "mfe_24h",
    ]
).reset_index(drop=True)


# ============================================================
# BASIC DISTRIBUTION
# ============================================================

print("\n=== REGIME FEATURE DISTRIBUTION ===")

print(
    df[
        [
            "er",
            "vwma_cross_count",
            "abs_vwma_dist",
            "channel_width",
        ]
    ]
    .describe()
    .round(4)
)


# ============================================================
# ER QUARTILES
# ============================================================

df["er_quartile"] = pd.qcut(
    df["er"],
    4,
    labels=[
        "Q1 CHOP",
        "Q2",
        "Q3",
        "Q4 TREND",
    ],
    duplicates="drop",
)


def summarize(group_col):

    result = (
        df.groupby(
            group_col,
            observed=True,
        )
        .agg(
            n=("time", "size"),

            avg_er=("er", "mean"),
            avg_cross=(
                "vwma_cross_count",
                "mean",
            ),

            mfe_4h=("mfe_4h", "mean"),
            mfe_12h=("mfe_12h", "mean"),
            mfe_24h=("mfe_24h", "mean"),

            mae_24h=("mae_24h", "mean"),
            ret_24h=("ret_24h", "mean"),

            hit_050=(
                "mfe_24h",
                lambda x: (
                    x >= 0.50
                ).mean() * 100,
            ),

            hit_100=(
                "mfe_24h",
                lambda x: (
                    x >= 1.00
                ).mean() * 100,
            ),

            hit_150=(
                "mfe_24h",
                lambda x: (
                    x >= 1.50
                ).mean() * 100,
            ),
        )
        .reset_index()
    )

    return result


print("\n=== BY ER QUARTILE ===")

print(
    summarize(
        "er_quartile"
    )
    .round(3)
    .to_string(index=False)
)


# ============================================================
# VWMA CROSS BUCKETS
# ============================================================

df["cross_bucket"] = pd.cut(
    df["vwma_cross_count"],
    bins=[
        -0.1,
        1,
        3,
        5,
        np.inf,
    ],
    labels=[
        "0-1 CLEAN",
        "2-3",
        "4-5",
        "6+ CHOP",
    ],
)

print("\n=== BY VWMA CROSS COUNT ===")

print(
    summarize(
        "cross_bucket"
    )
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 2D ER + CROSS TABLE
# ============================================================

er_median = df["er"].median()

df["er_state"] = np.where(
    df["er"] >= er_median,
    "HIGH_ER",
    "LOW_ER",
)

df["cross_state"] = np.where(
    df["vwma_cross_count"] <= 3,
    "LOW_CROSS",
    "HIGH_CROSS",
)

df["regime_2x2"] = (
    df["er_state"]
    + " + "
    + df["cross_state"]
)

print("\n=== ER x VWMA CROSS REGIME ===")

print(
    summarize(
        "regime_2x2"
    )
    .sort_values(
        "mfe_24h",
        ascending=False,
    )
    .round(3)
    .to_string(index=False)
)


# ============================================================
# CONTINUOUS CORRELATIONS
# ============================================================

print("\n=== CORRELATIONS ===")

corr_cols = [
    "er",
    "vwma_cross_count",
    "abs_vwma_dist",
    "channel_width",
    "mfe_4h",
    "mfe_12h",
    "mfe_24h",
    "mae_24h",
    "ret_24h",
]

print(
    df[corr_cols]
    .corr()
    .round(3)
    .to_string()
)


# ============================================================
# SIMPLE PRE-SPECIFIED REGIME SCREENS
# ============================================================

screens = {
    "ALL": np.ones(
        len(df),
        dtype=bool,
    ),

    "ER>=0.20":
        df["er"] >= 0.20,

    "ER>=0.30":
        df["er"] >= 0.30,

    "ER>=0.40":
        df["er"] >= 0.40,

    "CROSS<=3":
        df["vwma_cross_count"] <= 3,

    "CROSS<=2":
        df["vwma_cross_count"] <= 2,

    "ER>=0.20 + CROSS<=3":
        (
            (df["er"] >= 0.20)
            & (df["vwma_cross_count"] <= 3)
        ),

    "ER>=0.30 + CROSS<=3":
        (
            (df["er"] >= 0.30)
            & (df["vwma_cross_count"] <= 3)
        ),

    "ER>=0.30 + CROSS<=2":
        (
            (df["er"] >= 0.30)
            & (df["vwma_cross_count"] <= 2)
        ),
}

screen_rows = []

for name, mask in screens.items():

    x = df.loc[mask]

    if len(x) == 0:
        continue

    screen_rows.append({
        "screen": name,
        "n": len(x),
        "coverage_pct":
            len(x) / len(df) * 100,

        "avg_er":
            x["er"].mean(),

        "avg_cross":
            x["vwma_cross_count"].mean(),

        "mfe_4h":
            x["mfe_4h"].mean(),

        "mfe_12h":
            x["mfe_12h"].mean(),

        "mfe_24h":
            x["mfe_24h"].mean(),

        "mae_24h":
            x["mae_24h"].mean(),

        "ret_24h":
            x["ret_24h"].mean(),

        "hit_050":
            (x["mfe_24h"] >= 0.50)
            .mean() * 100,

        "hit_100":
            (x["mfe_24h"] >= 1.00)
            .mean() * 100,

        "hit_150":
            (x["mfe_24h"] >= 1.50)
            .mean() * 100,
    })

screen_df = pd.DataFrame(
    screen_rows
)

print("\n=== PRE-SPECIFIED REGIME SCREENS ===")

print(
    screen_df
    .sort_values(
        "mfe_24h",
        ascending=False,
    )
    .round(3)
    .to_string(index=False)
)


# ============================================================
# SIDE SPLIT
# ============================================================

print("\n=== BUY vs SELL ===")

side_summary = (
    df.groupby("side")
    .agg(
        n=("time", "size"),
        er=("er", "mean"),
        cross=("vwma_cross_count", "mean"),
        mfe_4h=("mfe_4h", "mean"),
        mfe_24h=("mfe_24h", "mean"),
        mae_24h=("mae_24h", "mean"),
        ret_24h=("ret_24h", "mean"),
    )
    .reset_index()
)

print(
    side_summary
    .round(3)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT,
    index=False,
)

print("\nSaved:", OUTPUT)

print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)
print(
    "This is a regime-anatomy test, NOT a trading-rule optimization."
)
print(
    "Primary question: do high-ER / low-cross Matrix flips produce"
)
print(
    "larger forward directional excursion than low-ER / high-cross flips?"
)
print(
    "Do not select a final threshold from this table yet."
)
