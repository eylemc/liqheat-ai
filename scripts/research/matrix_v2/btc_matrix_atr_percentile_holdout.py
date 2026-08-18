import pandas as pd
import numpy as np
from pathlib import Path

SIGNALS = Path("data/research/btc_matrix_1h_flip_displacement.csv")
OHLCV = Path("data/market/binance-futures-um/BTCUSDT/1m/BTCUSDT-1m.parquet")

DISP_THRESHOLD = 0.40
TP = 1.25
SL = 1.00
HORIZON = 1440

ATR_PERIOD = 14
LOOKBACK_HOURS = 24 * 90   # 90 days

# ============================================================
# LOAD 1M OHLCV
# ============================================================

one = pd.read_parquet(OHLCV)
one["open_time"] = pd.to_datetime(one["open_time"], utc=True)

one = (
    one.sort_values("open_time")
       .drop_duplicates("open_time", keep="last")
       .reset_index(drop=True)
)

one_idx = one.set_index("open_time", drop=False)

print("\n=== 1M OHLCV ===")
print(f"Candles : {len(one):,}")
print(f"Range   : {one.open_time.min()} -> {one.open_time.max()}")

# ============================================================
# BUILD CLOSED 1H CANDLES
# ============================================================

x = one.set_index("open_time")

h1 = (
    x.resample(
        "1h",
        label="left",
        closed="left"
    )
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
# ATR(14) - WILDER
# ============================================================

high = h1["high"].astype(float)
low = h1["low"].astype(float)
close = h1["close"].astype(float)

prev_close = close.shift(1)

tr = pd.concat([
    high - low,
    (high - prev_close).abs(),
    (low - prev_close).abs()
], axis=1).max(axis=1)

h1["atr"] = tr.ewm(
    alpha=1/ATR_PERIOD,
    adjust=False,
    min_periods=ATR_PERIOD
).mean()

h1["atr_pct_price"] = (
    h1["atr"] / h1["close"] * 100
)

# ============================================================
# ROLLING ATR PERCENTILE
# ============================================================

def rolling_percentile_last(window):
    if len(window) < 100:
        return np.nan

    current = window[-1]

    hist = window[:-1]

    if len(hist) == 0:
        return np.nan

    return (
        np.sum(hist <= current)
        / len(hist)
        * 100
    )

h1["atr_percentile"] = (
    h1["atr_pct_price"]
    .rolling(
        LOOKBACK_HOURS,
        min_periods=24*30
    )
    .apply(
        rolling_percentile_last,
        raw=True
    )
)

# ATR slope / expansion
h1["atr_1h_ago"] = h1["atr_pct_price"].shift(1)
h1["atr_3h_ago"] = h1["atr_pct_price"].shift(3)
h1["atr_6h_ago"] = h1["atr_pct_price"].shift(6)

h1["atr_rising_1h"] = (
    h1["atr_pct_price"] >
    h1["atr_1h_ago"]
)

h1["atr_rising_3h"] = (
    h1["atr_pct_price"] >
    h1["atr_3h_ago"]
)

h1["atr_rising_6h"] = (
    h1["atr_pct_price"] >
    h1["atr_6h_ago"]
)

# ============================================================
# LOAD MATRIX SIGNALS
# ============================================================

sig = pd.read_csv(SIGNALS)
sig["time"] = pd.to_datetime(sig["time"], utc=True)

sig = (
    sig.sort_values("time")
       .dropna(
           subset=[
               "disp_1h_15m_5m",
               "entry",
               "side",
           ]
       )
       .reset_index(drop=True)
)

split = int(len(sig) * 0.70)

train = sig.iloc[:split].copy()
test = sig.iloc[split:].copy().reset_index(drop=True)

test["side"] = test["side"].str.upper()

print("\n=== HOLDOUT ===")
print(f"Total : {len(sig)}")
print(f"Train : {len(train)}")
print(f"Test  : {len(test)}")
print(f"{test.time.min()} -> {test.time.max()}")

# ============================================================
# MERGE ATR STATE AT SIGNAL TIME
# ============================================================

cols = [
    "close_time",
    "atr_pct_price",
    "atr_percentile",
    "atr_rising_1h",
    "atr_rising_3h",
    "atr_rising_6h",
]

test = pd.merge_asof(
    test.sort_values("time"),
    h1[cols].sort_values("close_time"),
    left_on="time",
    right_on="close_time",
    direction="backward"
)

base = (
    test["disp_1h_15m_5m"]
    >= DISP_THRESHOLD
)

print("\n=== ATR DISTRIBUTION / BASE ===")

print(
    test.loc[
        base,
        ["atr_pct_price","atr_percentile"]
    ].describe().round(2).to_string()
)

# ============================================================
# EXACT FIRST TOUCH
# ============================================================

def replay(row):

    entry = float(row["entry"])
    side = row["side"]

    start = (
        row["time"]
        + pd.Timedelta(milliseconds=1)
    )

    end = (
        start
        + pd.Timedelta(minutes=HORIZON)
    )

    path = one_idx.loc[
        (one_idx.index >= start)
        & (one_idx.index < end)
    ]

    if path.empty:
        return None

    if side == "BUY":
        tp_price = entry * (1 + TP/100)
        sl_price = entry * (1 - SL/100)
    else:
        tp_price = entry * (1 - TP/100)
        sl_price = entry * (1 + SL/100)

    for minute_no, (_, c) in enumerate(
        path.iterrows(),
        start=1
    ):

        hi = float(c["high"])
        lo = float(c["low"])

        if side == "BUY":
            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price
            sl_hit = hi >= sl_price

        if tp_hit and sl_hit:
            return {
                "result":"SL",
                "ret":-SL,
                "minutes":minute_no
            }

        if tp_hit:
            return {
                "result":"TP",
                "ret":TP,
                "minutes":minute_no
            }

        if sl_hit:
            return {
                "result":"SL",
                "ret":-SL,
                "minutes":minute_no
            }

    last = float(path.iloc[-1]["close"])

    if side == "BUY":
        ret = (last/entry - 1)*100
    else:
        ret = (entry/last - 1)*100

    return {
        "result":"TIMEOUT",
        "ret":ret,
        "minutes":len(path)
    }

# ============================================================
# FILTERS
# ============================================================

filters = {
    "DISP>=0.40":
        base,

    "DISP + ATRpct>=25":
        base
        & (test["atr_percentile"] >= 25),

    "DISP + ATRpct>=40":
        base
        & (test["atr_percentile"] >= 40),

    "DISP + ATRpct>=50":
        base
        & (test["atr_percentile"] >= 50),

    "DISP + ATRpct>=60":
        base
        & (test["atr_percentile"] >= 60),

    "DISP + ATRpct>=75":
        base
        & (test["atr_percentile"] >= 75),

    "DISP + ATR rising1":
        base
        & test["atr_rising_1h"].fillna(False),

    "DISP + ATR rising3":
        base
        & test["atr_rising_3h"].fillna(False),

    "DISP + ATR rising6":
        base
        & test["atr_rising_6h"].fillna(False),

    "DISP + ATRpct>=50 + rising3":
        base
        & (test["atr_percentile"] >= 50)
        & test["atr_rising_3h"].fillna(False),

    "DISP + ATRpct>=60 + rising3":
        base
        & (test["atr_percentile"] >= 60)
        & test["atr_rising_3h"].fillna(False),
}

# ============================================================
# RUN
# ============================================================

rows = []

for name, mask in filters.items():

    g = test[mask].copy()

    outcomes = []

    for _, row in g.iterrows():

        r = replay(row)

        if r is not None:
            outcomes.append(r)

    if not outcomes:
        continue

    o = pd.DataFrame(outcomes)

    wins = o.loc[o.ret > 0, "ret"]
    losses = o.loc[o.ret < 0, "ret"]

    gp = wins.sum()
    gl = abs(losses.sum())

    pf = (
        gp/gl
        if gl > 0
        else np.inf
    )

    tp_times = o.loc[
        o.result == "TP",
        "minutes"
    ]

    rows.append({
        "filter":name,
        "n":len(o),
        "tp":int((o.result=="TP").sum()),
        "sl":int((o.result=="SL").sum()),
        "timeout":int((o.result=="TIMEOUT").sum()),
        "win_pct":(o.ret>0).mean()*100,
        "ev_pct":o.ret.mean(),
        "pf":pf,
        "avg_win":wins.mean() if len(wins) else np.nan,
        "avg_loss":losses.mean() if len(losses) else np.nan,
        "median_tp_min":(
            tp_times.median()
            if len(tp_times)
            else np.nan
        ),
        "avg_atr_pct":g["atr_pct_price"].mean(),
        "avg_atr_percentile":g["atr_percentile"].mean(),
        "avg_disp":g["disp_1h_15m_5m"].mean(),
    })

out = pd.DataFrame(rows)

print("\n=== MATRIX + ATR PERCENTILE HOLDOUT ===")

print(
    out.sort_values(
        ["ev_pct","pf"],
        ascending=False
    )
    .round({
        "win_pct":1,
        "ev_pct":3,
        "pf":2,
        "avg_win":3,
        "avg_loss":3,
        "median_tp_min":1,
        "avg_atr_pct":3,
        "avg_atr_percentile":1,
        "avg_disp":3,
    })
    .to_string(index=False)
)

print("\nFixed:")
print(f"DISP >= {DISP_THRESHOLD}")
print(f"TP={TP:.2f}% SL={SL:.2f}% horizon={HORIZON}m")
print("ATR percentile uses rolling 90-day history only")
