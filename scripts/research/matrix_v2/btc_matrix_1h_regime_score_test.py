import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from scripts import matrix_true_backtest as mtb


# ============================================================
# CONFIG
# ============================================================

HIST_FILE = Path(
    "data/research/btc_1m_historical_2022-10_to_2025-08.parquet"
)

RECENT_ROOT = Path(mtb.DATA_ROOT)

ER_PERIOD = 12

# Channel-width percentile:
# trailing history only, no future information.
PCTL_WINDOW = 24 * 90       # 90 days of 1H candles
PCTL_MIN = 24 * 14          # minimum 14 days

HORIZONS = {
    "4h": 240,
    "12h": 720,
    "24h": 1440,
}


# ============================================================
# HELPERS
# ============================================================

def find_recent_btc_1m():
    candidates = list(
        RECENT_ROOT.rglob("*BTCUSDT*")
    )

    files = [
        p for p in candidates
        if p.is_file()
        and p.suffix.lower() in {
            ".parquet", ".csv", ".feather"
        }
    ]

    if not files:
        # fallback: inspect all parquet files
        files = list(RECENT_ROOT.rglob("*.parquet"))

    if not files:
        raise RuntimeError(
            f"No recent BTC 1m file found under {RECENT_ROOT}"
        )

    print("\nRecent-data candidates:")
    for p in files[:20]:
        print(" ", p)

    return files


def read_any(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".feather":
        return pd.read_feather(path)

    raise RuntimeError(path)


def normalize_1m(df):
    df = df.copy()

    # detect timestamp column
    time_candidates = [
        "open_time",
        "timestamp",
        "time",
        "datetime",
        "date",
    ]

    tc = None

    for c in time_candidates:
        if c in df.columns:
            tc = c
            break

    if tc is None:
        raise RuntimeError(
            f"Timestamp column not found. Columns={df.columns.tolist()}"
        )

    if tc != "open_time":
        df = df.rename(columns={tc: "open_time"})

    # timestamp normalization
    if pd.api.types.is_numeric_dtype(df["open_time"]):
        vals = df["open_time"].dropna()

        if len(vals):
            med = float(vals.abs().median())

            if med > 1e17:
                unit = "ns"
            elif med > 1e14:
                unit = "us"
            elif med > 1e11:
                unit = "ms"
            else:
                unit = "s"

            df["open_time"] = pd.to_datetime(
                df["open_time"],
                unit=unit,
                utc=True,
            )
    else:
        df["open_time"] = pd.to_datetime(
            df["open_time"],
            utc=True,
        )

    needed = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    extra = [
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    missing = [x for x in needed if x not in df.columns]

    if missing:
        raise RuntimeError(
            f"Missing OHLCV columns: {missing}"
        )

    # aggregate_candles() expects these Binance fields.
    # Preserve them when present; if a source lacks one,
    # zero-fill because Matrix itself does not depend on them.
    for c in extra:
        if c not in df.columns:
            df[c] = 0.0

    keep = needed + extra

    if "symbol" not in df.columns:
        df["symbol"] = "BTCUSDT"

    if "timeframe" not in df.columns:
        df["timeframe"] = "1m"

    for c in keep:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = (
        df[
            ["symbol", "timeframe", "open_time"] + keep
        ]
        .dropna(subset=["open_time"] + needed)
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

if not HIST_FILE.exists():
    raise RuntimeError(
        f"Historical file missing: {HIST_FILE}"
    )

hist = normalize_1m(
    pd.read_parquet(HIST_FILE)
)

recent_files = find_recent_btc_1m()

recent_frames = []

for p in recent_files:
    try:
        x = normalize_1m(
            read_any(p)
        )

        # only keep plausible recent BTC data
        if len(x) > 1000:
            recent_frames.append(x)

    except Exception:
        continue

if not recent_frames:
    raise RuntimeError(
        "Could not load recent BTC 1m data."
    )

recent = pd.concat(
    recent_frames,
    ignore_index=True,
)

recent = (
    recent
    .sort_values("open_time")
    .drop_duplicates(
        subset=["open_time"],
        keep="last",
    )
)

minute = pd.concat(
    [hist, recent],
    ignore_index=True,
)

minute = (
    minute
    .sort_values("open_time")
    .drop_duplicates(
        subset=["open_time"],
        keep="last",
    )
    .reset_index(drop=True)
)

print("\n=== 1M DATA ===")
print("Rows :", f"{len(minute):,}")
print(
    "Range:",
    minute["open_time"].min(),
    "->",
    minute["open_time"].max(),
)


# ============================================================
# 1H CANDLES + MATRIX
# Use the exact existing Matrix research aggregation pipeline.
# ============================================================

hourly = mtb.aggregate_candles(
    minute.copy(),
    "1h",
)

matrix = mtb.add_matrix(
    hourly.copy(),
    "1h",
)

print("\nMatrix columns:")
print(matrix.columns.tolist())


# ============================================================
# REAL MATRIX FLIPS
# ============================================================

matrix["flip"] = pd.to_numeric(
    matrix["flip"],
    errors="coerce"
).fillna(0)

matrix["long_flip"] = pd.to_numeric(
    matrix["long_flip"],
    errors="coerce"
).fillna(0)

matrix["short_flip"] = pd.to_numeric(
    matrix["short_flip"],
    errors="coerce"
).fillna(0)

flips = matrix[
    matrix["flip"].ne(0)
].copy()

flips["matrix_side"] = np.where(
    flips["long_flip"].ne(0),
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
# REGIME FEATURES
# ============================================================

close = matrix["close"].astype(float)


# ------------------------------------------------------------
# Efficiency Ratio
# ------------------------------------------------------------

change = close.diff(
    ER_PERIOD
).abs()

path = (
    close.diff()
    .abs()
    .rolling(
        ER_PERIOD,
        min_periods=ER_PERIOD,
    )
    .sum()
)

matrix["er"] = (
    change / path.replace(0, np.nan)
)


# ------------------------------------------------------------
# Detect Matrix VWMA/channel columns
# ------------------------------------------------------------

def find_col(tokens):
    for c in matrix.columns:
        lc = c.lower()

        if all(
            t.lower() in lc
            for t in tokens
        ):
            return c

    return None


vwma_col = find_col(["vwma"])

upper_col = (
    find_col(["upper"])
    or find_col(["high", "channel"])
)

lower_col = (
    find_col(["lower"])
    or find_col(["low", "channel"])
)

print("\nDetected:")
print("VWMA  :", vwma_col)
print("Upper :", upper_col)
print("Lower :", lower_col)


# If Matrix exposes distance/channel directly, use them.
dist_col = find_col(["vwma", "dist"])
channel_col = find_col(["channel"])

if dist_col is not None:
    raw_dist = pd.to_numeric(
        matrix[dist_col],
        errors="coerce",
    )

    # Previous script showed percentages around e.g. 0.3495.
    # Convert to decimal if values look percentage-scaled.
    if raw_dist.abs().median() > 0.05:
        raw_dist = raw_dist / 100.0

    matrix["abs_vwma_dist"] = raw_dist.abs()

elif vwma_col is not None:
    vwma = pd.to_numeric(
        matrix[vwma_col],
        errors="coerce",
    )

    matrix["abs_vwma_dist"] = (
        (close - vwma).abs()
        / close
    )

else:
    raise RuntimeError(
        "Cannot calculate VWMA distance."
    )


if (
    upper_col is not None
    and lower_col is not None
):
    upper = pd.to_numeric(
        matrix[upper_col],
        errors="coerce",
    )

    lower = pd.to_numeric(
        matrix[lower_col],
        errors="coerce",
    )

    matrix["channel_width"] = (
        (upper - lower).abs()
        / close
    )

elif channel_col is not None:
    raw_channel = pd.to_numeric(
        matrix[channel_col],
        errors="coerce",
    )

    if raw_channel.abs().median() > 0.05:
        raw_channel = raw_channel / 100.0

    matrix["channel_width"] = raw_channel.abs()

else:
    raise RuntimeError(
        "Cannot calculate Matrix channel width."
    )


# ------------------------------------------------------------
# Normalized VWMA displacement
# ------------------------------------------------------------

matrix["norm_disp"] = (
    matrix["abs_vwma_dist"]
    / matrix["channel_width"].replace(
        0,
        np.nan,
    )
)


# ------------------------------------------------------------
# Trailing channel-width percentile
#
# IMPORTANT:
# percentile uses only PRIOR observations.
# ------------------------------------------------------------

cw = matrix["channel_width"]

matrix["channel_pctile"] = (
    cw.shift(1)
    .rolling(
        PCTL_WINDOW,
        min_periods=PCTL_MIN,
    )
    .apply(
        lambda x: (
            100.0
            * np.mean(
                x <= cw.loc[x.index[-1] + 1]
            )
        )
        if False else np.nan,
        raw=False,
    )
)

# Efficient causal percentile calculation.
# Rank current value against trailing history ONLY.
vals = cw.to_numpy(dtype=float)

pct = np.full(
    len(vals),
    np.nan,
)

for i in range(len(vals)):
    if i < PCTL_MIN:
        continue

    lo = max(
        0,
        i - PCTL_WINDOW,
    )

    history = vals[lo:i]
    history = history[
        np.isfinite(history)
    ]

    if len(history) < PCTL_MIN:
        continue

    if not np.isfinite(vals[i]):
        continue

    pct[i] = (
        np.mean(
            history <= vals[i]
        )
        * 100.0
    )

matrix["channel_pctile"] = pct


# ============================================================
# MERGE FEATURES INTO FLIPS
# ============================================================

features = matrix[
    [
        "close_time",
        "er",
        "abs_vwma_dist",
        "channel_width",
        "channel_pctile",
        "norm_disp",
    ]
].copy()

flips = flips.merge(
    features,
    on="close_time",
    how="left",
    suffixes=("", "_feature"),
)


# ============================================================
# FORWARD DIRECTIONAL STATS
# ============================================================

minute_ns = (
    pd.to_datetime(
        minute["open_time"],
        utc=True
    )
    .dt.as_unit("ns")
    .astype("int64")
    .to_numpy()
)

highs = minute["high"].to_numpy(
    dtype=float
)

lows = minute["low"].to_numpy(
    dtype=float
)

closes = minute["close"].to_numpy(
    dtype=float
)


def forward_stats(
    start_time,
    side,
    entry,
    minutes,
):
    t = pd.Timestamp(start_time)

    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    start_ns = t.value

    start = np.searchsorted(
        minute_ns,
        start_ns,
        side="right",
    )

    end_ns = (
        t
        + pd.Timedelta(
            minutes=minutes
        )
    ).value

    end = np.searchsorted(
        minute_ns,
        end_ns,
        side="right",
    )

    if start >= len(minute_ns):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    end = min(
        end,
        len(minute_ns),
    )

    if end <= start:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    hh = highs[start:end]
    ll = lows[start:end]

    final = closes[end - 1]

    if side == "BUY":
        mfe = (
            hh.max() / entry - 1
        ) * 100

        mae = (
            ll.min() / entry - 1
        ) * 100

        ret = (
            final / entry - 1
        ) * 100

    else:
        mfe = (
            1 - ll.min() / entry
        ) * 100

        mae = (
            1 - hh.max() / entry
        ) * 100

        ret = (
            1 - final / entry
        ) * 100

    return (
        mfe,
        mae,
        ret,
    )


rows = []

for _, row in flips.iterrows():

    out = {
        "time": row["close_time"],
        "side": row["matrix_side"],
        "entry": float(row["close"]),
        "er": row["er"],
        "channel_width": row["channel_width"],
        "channel_pctile": row["channel_pctile"],
        "abs_vwma_dist": row["abs_vwma_dist"],
        "norm_disp": row["norm_disp"],
    }

    for name, mins in HORIZONS.items():
        mfe, mae, ret = forward_stats(
            row["close_time"],
            row["matrix_side"],
            float(row["close"]),
            mins,
        )

        out[f"mfe_{name}"] = mfe
        out[f"mae_{name}"] = mae
        out[f"ret_{name}"] = ret

    rows.append(out)


df = pd.DataFrame(rows)

df = df.dropna(
    subset=[
        "er",
        "channel_pctile",
        "norm_disp",
        "mfe_24h",
    ]
).reset_index(drop=True)


# ============================================================
# RAW FEATURE ANATOMY
# ============================================================

print("\n=== USABLE SAMPLE ===")
print("N:", len(df))
print(
    df["time"].min(),
    "->",
    df["time"].max(),
)


def quartile_report(
    feature,
    title,
):
    x = df.copy()

    try:
        x["bucket"] = pd.qcut(
            x[feature],
            4,
            labels=[
                "Q1",
                "Q2",
                "Q3",
                "Q4",
            ],
            duplicates="drop",
        )
    except ValueError:
        print(
            f"\nCannot qcut {feature}"
        )
        return

    out = (
        x.groupby(
            "bucket",
            observed=True,
        )
        .agg(
            n=("entry", "size"),
            avg_feature=(feature, "mean"),
            mfe_4h=("mfe_4h", "mean"),
            mfe_12h=("mfe_12h", "mean"),
            mfe_24h=("mfe_24h", "mean"),
            mae_24h=("mae_24h", "mean"),
            ret_24h=("ret_24h", "mean"),
            hit_050=(
                "mfe_24h",
                lambda z: 100 * (z >= 0.50).mean(),
            ),
            hit_100=(
                "mfe_24h",
                lambda z: 100 * (z >= 1.00).mean(),
            ),
            hit_150=(
                "mfe_24h",
                lambda z: 100 * (z >= 1.50).mean(),
            ),
        )
        .reset_index()
    )

    print(
        f"\n=== {title} QUARTILES ==="
    )

    print(
        out.round(4).to_string(
            index=False
        )
    )


quartile_report(
    "er",
    "ER",
)

quartile_report(
    "channel_pctile",
    "CHANNEL WIDTH PERCENTILE",
)

quartile_report(
    "norm_disp",
    "NORMALIZED VWMA DISPLACEMENT",
)


# ============================================================
# COMPOSITE REGIME SCORE
#
# Causal ranking across PRIOR MATRIX FLIPS.
# No future observations enter a score.
# ============================================================

SCORE_WINDOW = 500
SCORE_MIN = 100

def prior_percentile(series, window=500, min_history=100):

    x = pd.to_numeric(
        series,
        errors="coerce"
    ).to_numpy(dtype=float)

    out = np.full(
        len(x),
        np.nan,
        dtype=float
    )

    for i in range(len(x)):

        if not np.isfinite(x[i]):
            continue

        lo = max(
            0,
            i - window
        )

        hist = x[lo:i]
        hist = hist[np.isfinite(hist)]

        if len(hist) < min_history:
            continue

        out[i] = (
            np.mean(hist <= x[i])
            * 100.0
        )

    return out


print("\n=== PRE-SCORE DIAGNOSTIC ===")
print("Rows:", len(df))

for c in [
    "er",
    "channel_width",
    "channel_pctile",
    "abs_vwma_dist",
    "norm_disp",
]:
    z = pd.to_numeric(
        df[c],
        errors="coerce"
    )

    print(
        f"{c:20s} "
        f"nonnull={z.notna().sum():5d} "
        f"finite={np.isfinite(z.to_numpy(dtype=float)).sum():5d}"
    )


df["er_rank"] = prior_percentile(
    df["er"],
    SCORE_WINDOW,
    SCORE_MIN
)

# channel_pctile is already causal versus prior 1H bars.
# We nevertheless rank it versus prior Matrix flips so that
# all three score components share the same 0-100 scale.
df["channel_rank"] = prior_percentile(
    df["channel_pctile"],
    SCORE_WINDOW,
    SCORE_MIN
)

df["norm_disp_rank"] = prior_percentile(
    df["norm_disp"],
    SCORE_WINDOW,
    SCORE_MIN
)


print("\n=== RANK DIAGNOSTIC ===")

for c in [
    "er_rank",
    "channel_rank",
    "norm_disp_rank",
]:
    z = pd.to_numeric(
        df[c],
        errors="coerce"
    )

    print(
        f"{c:20s} "
        f"nonnull={z.notna().sum():5d} "
        f"min={z.min()} "
        f"median={z.median()} "
        f"max={z.max()}"
    )


# Require all three components.
valid_score = (
    df[
        [
            "er_rank",
            "channel_rank",
            "norm_disp_rank",
        ]
    ]
    .notna()
    .all(axis=1)
)

df["regime_score"] = np.nan

df.loc[
    valid_score,
    "regime_score"
] = (
    df.loc[
        valid_score,
        [
            "er_rank",
            "channel_rank",
            "norm_disp_rank",
        ]
    ]
    .mean(axis=1)
)

score = (
    df[
        df["regime_score"].notna()
    ]
    .copy()
    .reset_index(drop=True)
)

print("\n=== REGIME SCORE DISTRIBUTION ===")

print("Usable scored flips:", len(score))

print(
    score[
        [
            "regime_score",
            "er",
            "channel_pctile",
            "norm_disp",
        ]
    ]
    .describe()
    .round(4)
    .to_string()
)

if len(score) < 100:
    raise RuntimeError(
        "Too few scored flips. Inspect PRE-SCORE and RANK diagnostics."
    )

# ============================================================
# SCORE QUARTILES
# ============================================================

score["regime_quartile"] = pd.qcut(
    score["regime_score"],
    4,
    labels=[
        "Q1 RANGE",
        "Q2",
        "Q3",
        "Q4 TREND",
    ],
    duplicates="drop",
)

summary = (
    score.groupby(
        "regime_quartile",
        observed=True,
    )
    .agg(
        n=("entry", "size"),
        avg_score=("regime_score", "mean"),
        avg_er=("er", "mean"),
        avg_channel_pct=(
            "channel_pctile",
            "mean",
        ),
        avg_norm_disp=(
            "norm_disp",
            "mean",
        ),
        mfe_4h=("mfe_4h", "mean"),
        mfe_12h=("mfe_12h", "mean"),
        mfe_24h=("mfe_24h", "mean"),
        mae_24h=("mae_24h", "mean"),
        ret_24h=("ret_24h", "mean"),
        hit_050=(
            "mfe_24h",
            lambda z: 100 * (z >= 0.50).mean(),
        ),
        hit_100=(
            "mfe_24h",
            lambda z: 100 * (z >= 1.00).mean(),
        ),
        hit_150=(
            "mfe_24h",
            lambda z: 100 * (z >= 1.50).mean(),
        ),
    )
    .reset_index()
)

print("\n=== MATRIX REGIME SCORE QUARTILES ===")

print(
    summary.round(4).to_string(
        index=False
    )
)


# ============================================================
# Q4 vs Q1
# ============================================================

q1 = score[
    score["regime_quartile"]
    == "Q1 RANGE"
]

q4 = score[
    score["regime_quartile"]
    == "Q4 TREND"
]

print("\n=== Q4 TREND vs Q1 RANGE ===")

for h in [
    "4h",
    "12h",
    "24h",
]:
    a = q1[f"mfe_{h}"].mean()
    b = q4[f"mfe_{h}"].mean()

    print(
        f"MFE {h.upper():>3} "
        f"Q1={a:+.4f}% "
        f"Q4={b:+.4f}% "
        f"lift={b/a:.2f}x"
        if a != 0
        else ""
    )

for target in [
    0.50,
    1.00,
    1.50,
]:
    a = 100 * (
        q1["mfe_24h"] >= target
    ).mean()

    b = 100 * (
        q4["mfe_24h"] >= target
    ).mean()

    print(
        f"24H hit >= {target:.2f}% "
        f"Q1={a:.1f}% "
        f"Q4={b:.1f}% "
        f"delta={b-a:+.1f}pp"
    )


# ============================================================
# CORRELATIONS
# ============================================================

cols = [
    "er",
    "channel_pctile",
    "norm_disp",
    "regime_score",
    "mfe_4h",
    "mfe_12h",
    "mfe_24h",
    "mae_24h",
    "ret_24h",
]

print("\n=== CORRELATIONS ===")

print(
    score[cols]
    .corr()
    .round(3)
    .to_string()
)


# ============================================================
# BUY / SELL BY REGIME
# ============================================================

bs = (
    score.groupby(
        [
            "side",
            "regime_quartile",
        ],
        observed=True,
    )
    .agg(
        n=("entry", "size"),
        mfe_4h=("mfe_4h", "mean"),
        mfe_24h=("mfe_24h", "mean"),
        ret_24h=("ret_24h", "mean"),
        hit_100=(
            "mfe_24h",
            lambda z: 100 * (z >= 1.0).mean(),
        ),
    )
    .reset_index()
)

print("\n=== BUY / SELL BY REGIME ===")

print(
    bs.round(4).to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

Path(
    "data/research"
).mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    "data/research/btc_matrix_1h_regime_score.csv",
    index=False,
)

summary.to_csv(
    "data/research/btc_matrix_1h_regime_score_quartiles.csv",
    index=False,
)

print("\nSaved:")
print(
    " data/research/btc_matrix_1h_regime_score.csv"
)
print(
    " data/research/btc_matrix_1h_regime_score_quartiles.csv"
)

print(
    "\n"
    + "=" * 90
)

print(
    "INTERPRETATION RULE\n"
    "This is regime anatomy only.\n"
    "No TP/SL optimization and no final threshold selection.\n"
    "Primary test: Q1 RANGE -> Q4 TREND should show "
    "monotonic improvement, especially MFE 4H/12H and "
    "24H target-hit rates.\n"
    "If monotonicity is weak, composite regime hypothesis "
    "is not supported."
)

print(
    "=" * 90
)
