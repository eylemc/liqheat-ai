import pandas as pd
import numpy as np
from pathlib import Path

SIGNALS = Path("data/research/btc_matrix_1h_flip_displacement.csv")
OHLCV = Path("data/market/binance-futures-um/BTCUSDT/1m/BTCUSDT-1m.parquet")

DISP = 0.40
TP = 1.25
SL = 1.00
HORIZON_MIN = 1440

ATR_PERIOD = 14
ATR_LOOKBACK_HOURS = 24 * 90

ATR_THRESHOLDS = [None, 40, 50, 60]

# ============================================================
# LOAD DATA
# ============================================================

one = pd.read_parquet(OHLCV)
one["open_time"] = pd.to_datetime(one["open_time"], utc=True)

one = (
    one.sort_values("open_time")
       .drop_duplicates("open_time", keep="last")
       .reset_index(drop=True)
)

print("\n=== DATA ===")
print(f"1m candles : {len(one):,}")
print(f"Range      : {one.open_time.min()} -> {one.open_time.max()}")

# ============================================================
# 1H OHLCV
# ============================================================

x = one.set_index("open_time")

h1 = (
    x.resample("1h", label="left", closed="left")
     .agg({
         "open": "first",
         "high": "max",
         "low": "min",
         "close": "last",
         "volume": "sum",
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
# WILDER ATR(14)
# ============================================================

high = h1["high"].astype(float)
low = h1["low"].astype(float)
close = h1["close"].astype(float)

prev_close = close.shift(1)

tr = pd.concat(
    [
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ],
    axis=1,
).max(axis=1)

h1["atr"] = tr.ewm(
    alpha=1 / ATR_PERIOD,
    adjust=False,
    min_periods=ATR_PERIOD,
).mean()

h1["atr_pct_price"] = (
    h1["atr"] / h1["close"] * 100
)

# ============================================================
# TRAILING 90-DAY ATR PERCENTILE
#
# Current value is compared ONLY against preceding values.
# No future information.
# ============================================================

atr = h1["atr_pct_price"].to_numpy(dtype=float)

pct = np.full(len(atr), np.nan)

for i in range(len(atr)):

    start = max(0, i - ATR_LOOKBACK_HOURS)

    hist = atr[start:i]

    hist = hist[np.isfinite(hist)]

    if len(hist) < 24 * 30:
        continue

    current = atr[i]

    if not np.isfinite(current):
        continue

    pct[i] = (
        np.mean(hist <= current) * 100
    )

h1["atr_percentile"] = pct

# ============================================================
# SIGNALS
# ============================================================

sig = pd.read_csv(SIGNALS)

sig["time"] = pd.to_datetime(
    sig["time"],
    utc=True,
)

sig["side"] = sig["side"].str.upper()

sig = (
    sig.sort_values("time")
       .dropna(
           subset=[
               "entry",
               "side",
               "disp_1h_15m_5m",
           ]
       )
       .reset_index(drop=True)
)

sig = pd.merge_asof(
    sig.sort_values("time"),
    h1[
        [
            "close_time",
            "atr_pct_price",
            "atr_percentile",
        ]
    ].sort_values("close_time"),
    left_on="time",
    right_on="close_time",
    direction="backward",
)

# Fixed Matrix setup
sig = sig[
    sig["disp_1h_15m_5m"] >= DISP
].copy().reset_index(drop=True)

print("\n=== FIXED MATRIX SETUP ===")
print(f"DISP >= {DISP}")
print(f"Signals = {len(sig)}")
print(f"{sig.time.min()} -> {sig.time.max()}")

# ============================================================
# EXACT 1M FIRST TOUCH OUTCOME
# ============================================================

one_idx = one.set_index("open_time", drop=False)

def replay(row):

    entry = float(row["entry"])
    side = row["side"]

    start = (
        row["time"]
        + pd.Timedelta(milliseconds=1)
    )

    end = (
        start
        + pd.Timedelta(minutes=HORIZON_MIN)
    )

    path = one_idx.loc[
        (one_idx.index >= start)
        & (one_idx.index < end)
    ]

    if path.empty:
        return None

    if side == "BUY":
        tp_price = entry * (1 + TP / 100)
        sl_price = entry * (1 - SL / 100)
    else:
        tp_price = entry * (1 - TP / 100)
        sl_price = entry * (1 + SL / 100)

    for minute_no, (_, c) in enumerate(
        path.iterrows(),
        start=1,
    ):

        hi = float(c["high"])
        lo = float(c["low"])

        if side == "BUY":
            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price
            sl_hit = hi >= sl_price

        # Conservative assumption if both touched
        # within same 1m candle.
        if tp_hit and sl_hit:
            return "SL", -SL, minute_no

        if tp_hit:
            return "TP", TP, minute_no

        if sl_hit:
            return "SL", -SL, minute_no

    last = float(path.iloc[-1]["close"])

    if side == "BUY":
        ret = (last / entry - 1) * 100
    else:
        ret = (entry / last - 1) * 100

    return "TIMEOUT", ret, len(path)

results = []

for _, row in sig.iterrows():

    r = replay(row)

    if r is None:
        results.append(
            (None, np.nan, np.nan)
        )
    else:
        results.append(r)

sig["result"] = [x[0] for x in results]
sig["ret"] = [x[1] for x in results]
sig["minutes"] = [x[2] for x in results]

sig = sig.dropna(subset=["ret"]).copy()

# ============================================================
# WALK-FORWARD
#
# Chronological four equal test blocks.
# Fold 1 is intentionally skipped as test because it has no
# substantial prior sample inside this Matrix subset.
#
# This gives us later unseen periods:
#   Fold 1 -> prior 25%
#   Fold 2 -> prior 50%
#   Fold 3 -> prior 75%
#
# Thresholds are NOT optimized on folds.
# ============================================================

n = len(sig)

edges = [
    0,
    int(n * .25),
    int(n * .50),
    int(n * .75),
    n,
]

print("\n=== WALK-FORWARD SPLITS ===")

for i in range(1, 4):

    train_end = edges[i]
    test_start = edges[i]
    test_end = edges[i + 1]

    tr = sig.iloc[:train_end]
    te = sig.iloc[test_start:test_end]

    print(
        f"Fold {i}: "
        f"train={len(tr)} "
        f"test={len(te)} | "
        f"test {te.time.min()} -> {te.time.max()}"
    )

# ============================================================
# METRICS
# ============================================================

def metrics(g):

    if len(g) == 0:
        return None

    wins = g.loc[g["ret"] > 0, "ret"]
    losses = g.loc[g["ret"] < 0, "ret"]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    return {
        "n": len(g),
        "tp": int((g["result"] == "TP").sum()),
        "sl": int((g["result"] == "SL").sum()),
        "timeout": int((g["result"] == "TIMEOUT").sum()),
        "wr": (g["ret"] > 0).mean() * 100,
        "ev": g["ret"].mean(),
        "pf": pf,
        "total": (
            np.prod(1 + g["ret"].to_numpy() / 100)
            - 1
        ) * 100,
        "avg_disp": g["disp_1h_15m_5m"].mean(),
        "avg_atr_pctile": g["atr_percentile"].mean(),
    }

# ============================================================
# EACH WALK-FORWARD FOLD
# ============================================================

rows = []

for fold in range(1, 4):

    test_start = edges[fold]
    test_end = edges[fold + 1]

    te = sig.iloc[
        test_start:test_end
    ].copy()

    for threshold in ATR_THRESHOLDS:

        if threshold is None:
            g = te.copy()
            name = "BASE"
        else:
            g = te[
                te["atr_percentile"]
                >= threshold
            ].copy()

            name = f"ATR>={threshold}"

        m = metrics(g)

        if m is None:
            continue

        rows.append({
            "fold": fold,
            "filter": name,
            **m,
        })

wf = pd.DataFrame(rows)

print("\n=== WALK-FORWARD RESULTS ===")

print(
    wf.round({
        "wr": 1,
        "ev": 3,
        "pf": 2,
        "total": 2,
        "avg_disp": 3,
        "avg_atr_pctile": 1,
    }).to_string(index=False)
)

# ============================================================
# AGGREGATED OOS
# ============================================================

oos_start = edges[1]

oos = sig.iloc[oos_start:].copy()

agg_rows = []

for threshold in ATR_THRESHOLDS:

    if threshold is None:
        g = oos.copy()
        name = "BASE"
    else:
        g = oos[
            oos["atr_percentile"]
            >= threshold
        ].copy()

        name = f"ATR>={threshold}"

    m = metrics(g)

    if m is None:
        continue

    agg_rows.append({
        "filter": name,
        **m,
    })

agg = pd.DataFrame(agg_rows)

print("\n=== AGGREGATED WALK-FORWARD OOS ===")

print(
    agg.round({
        "wr": 1,
        "ev": 3,
        "pf": 2,
        "total": 2,
        "avg_disp": 3,
        "avg_atr_pctile": 1,
    })
    .sort_values("ev", ascending=False)
    .to_string(index=False)
)

# ============================================================
# FOLD-BY-FOLD EV LIFT
# ============================================================

print("\n=== EV LIFT vs BASE BY FOLD ===")

for fold in sorted(wf["fold"].unique()):

    f = wf[wf["fold"] == fold]

    base_row = f[
        f["filter"] == "BASE"
    ]

    if base_row.empty:
        continue

    base_ev = float(base_row.iloc[0]["ev"])

    print(f"\nFold {fold} BASE EV={base_ev:+.3f}%")

    for threshold in [40, 50, 60]:

        r = f[
            f["filter"]
            == f"ATR>={threshold}"
        ]

        if r.empty:
            continue

        ev = float(r.iloc[0]["ev"])
        nn = int(r.iloc[0]["n"])

        print(
            f" ATR>={threshold}: "
            f"N={nn:2d} "
            f"EV={ev:+.3f}% "
            f"LIFT={ev-base_ev:+.3f}pp"
        )

# ============================================================
# SAVE
# ============================================================

out_path = Path(
    "data/research/"
    "btc_matrix_atr_walkforward_signals.csv"
)

sig.to_csv(out_path, index=False)

print("\nSaved:", out_path)

print("\nFixed research specification:")
print("1H Matrix flip")
print(f"Displacement >= {DISP}")
print("ATR(14) / price")
print("ATR percentile = trailing 90 days")
print("ATR thresholds = 40, 50, 60 (pre-specified)")
print(f"TP={TP}% SL={SL}% horizon={HORIZON_MIN}m")
print("Exact first-touch replay from 1m candles")
