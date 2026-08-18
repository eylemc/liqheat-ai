import pandas as pd
import numpy as np
from pathlib import Path

SIGNALS = Path(
    "data/research/btc_matrix_atr_walkforward_signals.csv"
)
OHLCV = Path(
    "data/market/binance-futures-um/BTCUSDT/1m/BTCUSDT-1m.parquet"
)

ATR_CUT = 60

# ============================================================
# LOAD
# ============================================================

sig = pd.read_csv(SIGNALS)
sig["time"] = pd.to_datetime(sig["time"], utc=True)
sig = sig.sort_values("time").reset_index(drop=True)

one = pd.read_parquet(OHLCV)
one["open_time"] = pd.to_datetime(one["open_time"], utc=True)

one = (
    one.sort_values("open_time")
       .drop_duplicates("open_time", keep="last")
       .reset_index(drop=True)
)

print("\n=== SIGNALS ===")
print("N:", len(sig))
print(sig.time.min(), "->", sig.time.max())

# ============================================================
# RECREATE SAME WALK-FORWARD FOLDS
# ============================================================

n = len(sig)

edges = [
    0,
    int(n * .25),
    int(n * .50),
    int(n * .75),
    n,
]

sig["fold"] = 0

for fold in range(1, 4):
    sig.loc[
        sig.index[edges[fold]:edges[fold+1]],
        "fold"
    ] = fold

# Only actual OOS folds
sig = sig[sig["fold"] > 0].copy()

# ============================================================
# BUILD 1H
# ============================================================

x = one.set_index("open_time")

h1 = (
    x.resample("1h", label="left", closed="left")
     .agg({
         "open":"first",
         "high":"max",
         "low":"min",
         "close":"last",
         "volume":"sum",
     })
     .dropna()
     .reset_index()
)

h1["close_time"] = (
    h1["open_time"]
    + pd.Timedelta(hours=1)
    - pd.Timedelta(milliseconds=1)
)

# ============================================================
# RETURNS / REALIZED VOL
# ============================================================

h1["logret"] = np.log(
    h1["close"] / h1["close"].shift(1)
)

for hours in [6, 12, 24, 48]:

    h1[f"rv_{hours}h"] = (
        h1["logret"]
        .rolling(hours)
        .std()
        * np.sqrt(hours)
        * 100
    )

# ============================================================
# ADX / DMI (WILDER)
# ============================================================

period = 14

high = h1["high"].astype(float)
low = h1["low"].astype(float)
close = h1["close"].astype(float)

prev_close = close.shift(1)

tr = pd.concat(
    [
        high-low,
        (high-prev_close).abs(),
        (low-prev_close).abs(),
    ],
    axis=1,
).max(axis=1)

up = high.diff()
down = -low.diff()

plus_dm = pd.Series(
    np.where(
        (up > down) & (up > 0),
        up,
        0.0
    ),
    index=h1.index
)

minus_dm = pd.Series(
    np.where(
        (down > up) & (down > 0),
        down,
        0.0
    ),
    index=h1.index
)

atr = tr.ewm(
    alpha=1/period,
    adjust=False
).mean()

plus_sm = plus_dm.ewm(
    alpha=1/period,
    adjust=False
).mean()

minus_sm = minus_dm.ewm(
    alpha=1/period,
    adjust=False
).mean()

plus_di = 100 * plus_sm / atr
minus_di = 100 * minus_sm / atr

dx = (
    100
    * (plus_di-minus_di).abs()
    / (plus_di+minus_di)
)

adx = dx.ewm(
    alpha=1/period,
    adjust=False
).mean()

h1["adx"] = adx
h1["plus_di"] = plus_di
h1["minus_di"] = minus_di

# Direction-neutral DMI strength
h1["di_abs_spread"] = (
    plus_di - minus_di
).abs()

# ============================================================
# BOLLINGER WIDTH
# ============================================================

bb_mid = close.rolling(20).mean()
bb_std = close.rolling(20).std()

bb_upper = bb_mid + 2*bb_std
bb_lower = bb_mid - 2*bb_std

h1["bb_width_pct"] = (
    (bb_upper-bb_lower)
    / bb_mid
    * 100
)

# BB width expansion
h1["bb_width_change_6h"] = (
    h1["bb_width_pct"]
    / h1["bb_width_pct"].shift(6)
    - 1
) * 100

# ============================================================
# VOLUME Z-SCORE
# ============================================================

vol_mean = (
    h1["volume"]
    .rolling(72)
    .mean()
)

vol_std = (
    h1["volume"]
    .rolling(72)
    .std()
)

h1["volume_z72"] = (
    (h1["volume"] - vol_mean)
    / vol_std
)

# ============================================================
# EFFICIENCY RATIO / CHOP PROXY
#
# ER close to 1 = directional
# ER close to 0 = noisy/choppy
# ============================================================

for hours in [6, 12, 24]:

    net = (
        close - close.shift(hours)
    ).abs()

    path = (
        close.diff().abs()
        .rolling(hours)
        .sum()
    )

    h1[f"er_{hours}h"] = (
        net / path
    )

# ============================================================
# TREND SLOPE
# ============================================================

for hours in [6, 12, 24]:

    h1[f"ret_{hours}h"] = (
        close / close.shift(hours) - 1
    ) * 100

# ============================================================
# MERGE FEATURES AT SIGNAL TIME
# ============================================================

feature_cols = [
    "close_time",
    "adx",
    "plus_di",
    "minus_di",
    "di_abs_spread",
    "bb_width_pct",
    "bb_width_change_6h",
    "volume_z72",
    "rv_6h",
    "rv_12h",
    "rv_24h",
    "rv_48h",
    "er_6h",
    "er_12h",
    "er_24h",
    "ret_6h",
    "ret_12h",
    "ret_24h",
]

sig = pd.merge_asof(
    sig.sort_values("time"),
    h1[feature_cols]
        .sort_values("close_time"),
    left_on="time",
    right_on="close_time",
    direction="backward",
)

# ============================================================
# DIRECTION-ADJUSTED FEATURES
# ============================================================

direction = np.where(
    sig["side"].str.upper() == "BUY",
    1.0,
    -1.0
)

sig["signed_ret_6h"] = (
    sig["ret_6h"] * direction
)

sig["signed_ret_12h"] = (
    sig["ret_12h"] * direction
)

sig["signed_ret_24h"] = (
    sig["ret_24h"] * direction
)

sig["dmi_directional"] = np.where(
    direction > 0,
    sig["plus_di"] - sig["minus_di"],
    sig["minus_di"] - sig["plus_di"],
)

# ============================================================
# ATR >= 60 ONLY
# ============================================================

a = sig[
    sig["atr_percentile"] >= ATR_CUT
].copy()

a["regime"] = np.where(
    a["fold"] == 2,
    "FOLD2_BAD",
    "FOLD1+3_GOOD",
)

print("\n=== ATR >= 60 SAMPLE ===")
print(
    a.groupby("fold")
     .agg(
         n=("ret","size"),
         avg_ret=("ret","mean"),
         win_pct=("ret",lambda x:(x>0).mean()*100),
         avg_atr=("atr_percentile","mean"),
         avg_disp=("disp_1h_15m_5m","mean"),
     )
     .round(3)
     .to_string()
)

# ============================================================
# FEATURE COMPARISON
# ============================================================

features = [
    "adx",
    "di_abs_spread",
    "dmi_directional",
    "bb_width_pct",
    "bb_width_change_6h",
    "volume_z72",
    "rv_6h",
    "rv_12h",
    "rv_24h",
    "rv_48h",
    "er_6h",
    "er_12h",
    "er_24h",
    "signed_ret_6h",
    "signed_ret_12h",
    "signed_ret_24h",
    "disp_1h_15m_5m",
    "atr_percentile",
]

rows = []

good = a[
    a["regime"] == "FOLD1+3_GOOD"
]

bad = a[
    a["regime"] == "FOLD2_BAD"
]

for col in features:

    g = good[col].dropna()
    b = bad[col].dropna()

    if len(g) == 0 or len(b) == 0:
        continue

    pooled = pd.concat([g,b]).std()

    diff = g.mean() - b.mean()

    effect = (
        diff / pooled
        if pooled > 0
        else np.nan
    )

    rows.append({
        "feature":col,
        "good_n":len(g),
        "bad_n":len(b),
        "good_mean":g.mean(),
        "bad_mean":b.mean(),
        "difference":diff,
        "effect_size":effect,
        "good_median":g.median(),
        "bad_median":b.median(),
    })

cmp = pd.DataFrame(rows)

cmp["abs_effect"] = (
    cmp["effect_size"].abs()
)

cmp = cmp.sort_values(
    "abs_effect",
    ascending=False
)

print("\n=== FOLD 1+3 vs FOLD 2 FEATURE DIFFERENCES ===")

print(
    cmp[
        [
            "feature",
            "good_n",
            "bad_n",
            "good_mean",
            "bad_mean",
            "difference",
            "effect_size",
            "good_median",
            "bad_median",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# ============================================================
# WITHIN ATR>=60: FEATURE vs RETURN
# ============================================================

corr_rows = []

for col in features:

    z = a[[col,"ret"]].dropna()

    if len(z) < 5:
        continue

    corr_rows.append({
        "feature":col,
        "n":len(z),
        "corr_return":
            z[col].corr(z["ret"]),
    })

corr = (
    pd.DataFrame(corr_rows)
    .assign(
        abs_corr=lambda x:
        x.corr_return.abs()
    )
    .sort_values(
        "abs_corr",
        ascending=False
    )
)

print("\n=== ATR>=60 FEATURE / RETURN CORRELATION ===")

print(
    corr[
        ["feature","n","corr_return"]
    ]
    .round(4)
    .to_string(index=False)
)

# ============================================================
# WINNER / LOSER ANATOMY
# ============================================================

a["winner"] = a["ret"] > 0

wl = []

for col in features:

    w = a.loc[
        a["winner"], col
    ].dropna()

    l = a.loc[
        ~a["winner"], col
    ].dropna()

    if len(w) == 0 or len(l) == 0:
        continue

    pooled = pd.concat([w,l]).std()

    diff = w.mean() - l.mean()

    wl.append({
        "feature":col,
        "win_mean":w.mean(),
        "loss_mean":l.mean(),
        "difference":diff,
        "effect_size":
            diff/pooled
            if pooled > 0
            else np.nan,
    })

wl = pd.DataFrame(wl)

wl["abs_effect"] = (
    wl["effect_size"].abs()
)

wl = wl.sort_values(
    "abs_effect",
    ascending=False
)

print("\n=== ATR>=60 WINNER vs LOSER ===")

print(
    wl[
        [
            "feature",
            "win_mean",
            "loss_mean",
            "difference",
            "effect_size",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# ============================================================
# INDIVIDUAL ATR>=60 SIGNALS
# ============================================================

show = [
    "time",
    "fold",
    "side",
    "ret",
    "result",
    "disp_1h_15m_5m",
    "atr_percentile",
    "adx",
    "dmi_directional",
    "bb_width_pct",
    "bb_width_change_6h",
    "volume_z72",
    "rv_24h",
    "er_12h",
    "signed_ret_12h",
]

print("\n=== INDIVIDUAL ATR>=60 SIGNALS ===")

print(
    a[show]
    .round(3)
    .to_string(index=False)
)

# ============================================================
# SAVE
# ============================================================

out = Path(
    "data/research/"
    "btc_matrix_atr60_regime_anatomy.csv"
)

a.to_csv(out,index=False)

print("\nSaved:",out)
