import sys
sys.path.insert(0, "/home/eylem/liqheat-ai")

import pandas as pd
import numpy as np
from pathlib import Path
from scripts import matrix_true_backtest as mtb

OHLCV = Path(
    "data/market/binance-futures-um/BTCUSDT/1m/BTCUSDT-1m.parquet"
)

START = pd.Timestamp("2025-08-06", tz="UTC")

ATR_PERIOD = 14
ATR_LOOKBACK_BARS = 6 * 90       # 4H bars in ~90 days
ATR_THRESHOLDS = [None, 40, 50, 60]

# Keep this intentionally small / pre-specified.
CONFIGS = [
    # horizon_hours, TP%, SL%
    (24, 1.50, 1.00),
    (24, 2.00, 1.00),

    (48, 2.00, 1.00),
    (48, 2.00, 1.50),
    (48, 3.00, 1.50),

    (72, 2.00, 1.00),
    (72, 3.00, 1.50),
    (72, 4.00, 1.50),
]

# ============================================================
# LOAD 1M
# ============================================================

one = pd.read_parquet(OHLCV).copy()
one["open_time"] = pd.to_datetime(one["open_time"], utc=True)

one = (
    one[one["open_time"] >= START]
    .sort_values("open_time")
    .drop_duplicates("open_time", keep="last")
    .reset_index(drop=True)
)

print("\n=== DATA ===")
print(f"1m candles : {len(one):,}")
print(f"Range      : {one.open_time.min()} -> {one.open_time.max()}")

one_idx = one.set_index("open_time", drop=False).sort_index()

# ============================================================
# REAL 4H MATRIX
# ============================================================

h4 = mtb.aggregate_candles(one.copy(), "4h")
m4 = mtb.add_matrix(h4.copy(), "4h").copy()

m4["close_time"] = pd.to_datetime(m4["close_time"], utc=True)
m4["available_at"] = pd.to_datetime(m4["available_at"], utc=True)

flips = m4[
    m4["long_flip"].fillna(False) |
    m4["short_flip"].fillna(False)
].copy()

flips["side"] = np.where(
    flips["long_flip"].fillna(False),
    "BUY",
    "SELL"
)

flips["signal_time"] = flips["available_at"]

flips = (
    flips.sort_values("signal_time")
    .reset_index(drop=True)
)

print("\n=== 4H MATRIX ===")
print("Signals:", len(flips))
print(
    flips.signal_time.min(),
    "->",
    flips.signal_time.max()
)

# ============================================================
# ATR(14) ON CLOSED 4H BARS
# ============================================================

high = m4["high"].astype(float)
low = m4["low"].astype(float)
close = m4["close"].astype(float)

prev_close = close.shift(1)

tr = pd.concat(
    [
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ],
    axis=1
).max(axis=1)

m4["atr"] = tr.ewm(
    alpha=1/ATR_PERIOD,
    adjust=False,
    min_periods=ATR_PERIOD
).mean()

m4["atr_pct_price"] = (
    m4["atr"] / m4["close"] * 100
)

# Trailing percentile using past values ONLY.
atr = m4["atr_pct_price"].to_numpy(dtype=float)

pct = np.full(len(atr), np.nan)

for i in range(len(atr)):

    start = max(0, i - ATR_LOOKBACK_BARS)

    hist = atr[start:i]
    hist = hist[np.isfinite(hist)]

    if len(hist) < 6 * 30:   # minimum ~30 days
        continue

    current = atr[i]

    if not np.isfinite(current):
        continue

    pct[i] = np.mean(hist <= current) * 100

m4["atr_percentile"] = pct

# ============================================================
# MERGE ATR TO FLIPS
# ============================================================

features = m4[
    [
        "available_at",
        "atr_pct_price",
        "atr_percentile",
    ]
].copy()

flips = pd.merge_asof(
    flips.sort_values("signal_time"),
    features.sort_values("available_at"),
    left_on="signal_time",
    right_on="available_at",
    direction="backward"
)

print("\n=== ATR AT SIGNALS ===")
print(
    flips[
        ["atr_pct_price", "atr_percentile"]
    ].describe().round(2).to_string()
)

# ============================================================
# ENTRY + FIRST TOUCH
# ============================================================

def first_entry(signal_time):

    p = one_idx.loc[
        one_idx.index >= signal_time
    ]

    if p.empty:
        return None

    r = p.iloc[0]

    return (
        r["open_time"],
        float(r["open"])
    )


def replay(row, horizon_hours, tp, sl):

    ent = first_entry(row["signal_time"])

    if ent is None:
        return None

    entry_time, entry = ent

    horizon_min = horizon_hours * 60

    end = (
        entry_time
        + pd.Timedelta(minutes=horizon_min)
    )

    path = one_idx.loc[
        (one_idx.index >= entry_time)
        & (one_idx.index < end)
    ]

    # Avoid seriously truncated final trades.
    if len(path) < horizon_min // 2:
        return None

    side = row["side"]

    if side == "BUY":

        tp_price = entry * (1 + tp/100)
        sl_price = entry * (1 - sl/100)

    else:

        # Use percentage return symmetry.
        tp_price = entry / (1 + tp/100)
        sl_price = entry / (1 - sl/100)

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

        # Conservative same-1m ambiguity.
        if tp_hit and sl_hit:
            return {
                "result": "SL",
                "ret": -sl,
                "minutes": minute_no,
            }

        if tp_hit:
            return {
                "result": "TP",
                "ret": tp,
                "minutes": minute_no,
            }

        if sl_hit:
            return {
                "result": "SL",
                "ret": -sl,
                "minutes": minute_no,
            }

    last = float(path.iloc[-1]["close"])

    if side == "BUY":
        ret = (last / entry - 1) * 100
    else:
        ret = (entry / last - 1) * 100

    return {
        "result": "TIMEOUT",
        "ret": ret,
        "minutes": len(path),
    }

# ============================================================
# PRECOMPUTE OUTCOMES
# ============================================================

for h, tp, sl in CONFIGS:

    key = f"h{h}_tp{tp}_sl{sl}"

    results = []

    for _, row in flips.iterrows():

        r = replay(
            row,
            h,
            tp,
            sl
        )

        results.append(r)

    flips[f"{key}_result"] = [
        x["result"] if x else None
        for x in results
    ]

    flips[f"{key}_ret"] = [
        x["ret"] if x else np.nan
        for x in results
    ]

    flips[f"{key}_minutes"] = [
        x["minutes"] if x else np.nan
        for x in results
    ]

# ============================================================
# WALK FORWARD SPLITS
#
# 78 signals is not huge.
# Use 3 OOS blocks after initial 25%.
# ============================================================

n = len(flips)

edges = [
    0,
    int(n * .25),
    int(n * .50),
    int(n * .75),
    n,
]

print("\n=== WALK-FORWARD SPLITS ===")

for fold in range(1, 4):

    train_end = edges[fold]
    test_start = edges[fold]
    test_end = edges[fold+1]

    tr = flips.iloc[:train_end]
    te = flips.iloc[test_start:test_end]

    print(
        f"Fold {fold}: "
        f"train={len(tr)} "
        f"test={len(te)} | "
        f"{te.signal_time.min()} -> "
        f"{te.signal_time.max()}"
    )

# ============================================================
# METRICS
# ============================================================

def metrics(g, key):

    ret_col = f"{key}_ret"
    res_col = f"{key}_result"
    min_col = f"{key}_minutes"

    x = g.dropna(
        subset=[ret_col]
    ).copy()

    if x.empty:
        return None

    wins = x.loc[
        x[ret_col] > 0,
        ret_col
    ]

    losses = x.loc[
        x[ret_col] < 0,
        ret_col
    ]

    gp = wins.sum()
    gl = abs(losses.sum())

    pf = (
        gp/gl
        if gl > 0
        else np.inf
    )

    tp_times = x.loc[
        x[res_col] == "TP",
        min_col
    ]

    return {
        "n": len(x),
        "tp": int(
            (x[res_col] == "TP").sum()
        ),
        "sl": int(
            (x[res_col] == "SL").sum()
        ),
        "timeout": int(
            (x[res_col] == "TIMEOUT").sum()
        ),
        "wr": (x[ret_col] > 0).mean()*100,
        "ev": x[ret_col].mean(),
        "pf": pf,
        "total": (
            np.prod(
                1 + x[ret_col].to_numpy()/100
            ) - 1
        ) * 100,
        "median_tp_min": (
            tp_times.median()
            if len(tp_times)
            else np.nan
        ),
        "avg_atr_pctile":
            x["atr_percentile"].mean(),
    }

# ============================================================
# PER-FOLD RESULTS
# ============================================================

rows = []

for fold in range(1, 4):

    te = flips.iloc[
        edges[fold]:edges[fold+1]
    ].copy()

    for h, tp, sl in CONFIGS:

        key = f"h{h}_tp{tp}_sl{sl}"

        for atr_th in ATR_THRESHOLDS:

            if atr_th is None:

                g = te.copy()
                filter_name = "BASE"

            else:

                g = te[
                    te["atr_percentile"]
                    >= atr_th
                ].copy()

                filter_name = (
                    f"ATR>={atr_th}"
                )

            m = metrics(g, key)

            if m is None:
                continue

            rows.append({
                "fold": fold,
                "H": h,
                "TP": tp,
                "SL": sl,
                "filter": filter_name,
                **m,
            })

wf = pd.DataFrame(rows)

print("\n=== WALK-FORWARD RESULTS ===")

print(
    wf.sort_values(
        ["H","TP","SL","fold","filter"]
    )
    .round({
        "wr":1,
        "ev":3,
        "pf":2,
        "total":2,
        "median_tp_min":1,
        "avg_atr_pctile":1,
    })
    .to_string(index=False)
)

# ============================================================
# AGGREGATED OOS
# ============================================================

oos = flips.iloc[
    edges[1]:
].copy()

agg_rows = []

for h, tp, sl in CONFIGS:

    key = f"h{h}_tp{tp}_sl{sl}"

    for atr_th in ATR_THRESHOLDS:

        if atr_th is None:

            g = oos.copy()
            filter_name = "BASE"

        else:

            g = oos[
                oos["atr_percentile"]
                >= atr_th
            ].copy()

            filter_name = (
                f"ATR>={atr_th}"
            )

        m = metrics(g, key)

        if m is None:
            continue

        agg_rows.append({
            "H": h,
            "TP": tp,
            "SL": sl,
            "filter": filter_name,
            **m,
        })

agg = pd.DataFrame(agg_rows)

print("\n=== AGGREGATED OOS ===")

print(
    agg.sort_values(
        ["H","TP","SL","ev"],
        ascending=[
            True, True, True, False
        ]
    )
    .round({
        "wr":1,
        "ev":3,
        "pf":2,
        "total":2,
        "median_tp_min":1,
        "avg_atr_pctile":1,
    })
    .to_string(index=False)
)

# ============================================================
# ATR LIFT vs BASE
# ============================================================

print("\n=== ATR EV LIFT vs BASE BY FOLD ===")

for h, tp, sl in CONFIGS:

    print(
        f"\n--- H={h}h "
        f"TP={tp:.2f}% "
        f"SL={sl:.2f}% ---"
    )

    for fold in range(1,4):

        f = wf[
            (wf.fold == fold)
            & (wf.H == h)
            & (wf.TP == tp)
            & (wf.SL == sl)
        ]

        b = f[
            f["filter"] == "BASE"
        ]

        if b.empty:
            continue

        base_ev = float(
            b.iloc[0]["ev"]
        )

        print(
            f"Fold {fold} "
            f"BASE={base_ev:+.3f}%"
        )

        for atr_th in [40,50,60]:

            r = f[
                f["filter"]
                == f"ATR>={atr_th}"
            ]

            if r.empty:
                continue

            ev = float(
                r.iloc[0]["ev"]
            )

            nn = int(
                r.iloc[0]["n"]
            )

            print(
                f" ATR>={atr_th}: "
                f"N={nn:2d} "
                f"EV={ev:+.3f}% "
                f"LIFT={ev-base_ev:+.3f}pp"
            )

# ============================================================
# SAVE
# ============================================================

Path("data/research").mkdir(
    parents=True,
    exist_ok=True
)

flips.to_csv(
    "data/research/"
    "btc_matrix_4h_atr_walkforward_signals.csv",
    index=False
)

wf.to_csv(
    "data/research/"
    "btc_matrix_4h_atr_walkforward_folds.csv",
    index=False
)

agg.to_csv(
    "data/research/"
    "btc_matrix_4h_atr_walkforward_agg.csv",
    index=False
)

print("\nSaved:")
print(
    " data/research/"
    "btc_matrix_4h_atr_walkforward_signals.csv"
)
print(
    " data/research/"
    "btc_matrix_4h_atr_walkforward_folds.csv"
)
print(
    " data/research/"
    "btc_matrix_4h_atr_walkforward_agg.csv"
)
