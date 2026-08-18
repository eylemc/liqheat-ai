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
ATR_LOOKBACK_BARS = 6 * 90

CONFIGS = [
    (72, 3.00, 1.50),
    (72, 4.00, 1.50),
]

FILTERS = {
    "BASE": None,
    "ATR>=50": ("ge", 50),
    "ATR>=60": ("ge", 60),
    "ATR40-70": ("band", 40, 70),
    "ATR40-80": ("band", 40, 80),
    "ATR50-70": ("band", 50, 70),
    "ATR50-80": ("band", 50, 80),
    "ATR50-90": ("band", 50, 90),
    "ATR60-80": ("band", 60, 80),
}

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
# BUILD REAL 4H MATRIX
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
# ATR(14) / PRICE ON CLOSED 4H BARS
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
    alpha=1 / ATR_PERIOD,
    adjust=False,
    min_periods=ATR_PERIOD
).mean()

m4["atr_pct_price"] = (
    m4["atr"] / m4["close"] * 100
)

# trailing percentile, history only
atr = m4["atr_pct_price"].to_numpy(dtype=float)
pct = np.full(len(atr), np.nan)

for i in range(len(atr)):

    start = max(0, i - ATR_LOOKBACK_BARS)

    hist = atr[start:i]
    hist = hist[np.isfinite(hist)]

    if len(hist) < 6 * 30:
        continue

    current = atr[i]

    if not np.isfinite(current):
        continue

    pct[i] = np.mean(hist <= current) * 100

m4["atr_percentile"] = pct

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

print("\n=== ATR DISTRIBUTION AT 4H FLIPS ===")
print(
    flips[
        ["atr_pct_price", "atr_percentile"]
    ]
    .describe()
    .round(2)
    .to_string()
)

# ============================================================
# ENTRY / FIRST TOUCH
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


def replay(row, horizon_h, tp, sl):

    ent = first_entry(row["signal_time"])

    if ent is None:
        return None

    entry_time, entry = ent

    horizon_min = horizon_h * 60

    end = (
        entry_time
        + pd.Timedelta(minutes=horizon_min)
    )

    path = one_idx.loc[
        (one_idx.index >= entry_time)
        & (one_idx.index < end)
    ]

    if len(path) < horizon_min // 2:
        return None

    side = row["side"]

    if side == "BUY":
        tp_price = entry * (1 + tp / 100)
        sl_price = entry * (1 - sl / 100)
    else:
        tp_price = entry / (1 + tp / 100)
        sl_price = entry / (1 - sl / 100)

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

        # Conservative same-minute ambiguity
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

    rs = []

    for _, row in flips.iterrows():
        rs.append(
            replay(row, h, tp, sl)
        )

    flips[f"{key}_result"] = [
        x["result"] if x else None
        for x in rs
    ]

    flips[f"{key}_ret"] = [
        x["ret"] if x else np.nan
        for x in rs
    ]

    flips[f"{key}_minutes"] = [
        x["minutes"] if x else np.nan
        for x in rs
    ]

# ============================================================
# SAME WALK-FORWARD SPLIT
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

    te = flips.iloc[
        edges[fold]:edges[fold+1]
    ]

    print(
        f"Fold {fold}: "
        f"N={len(te)} | "
        f"{te.signal_time.min()} -> "
        f"{te.signal_time.max()}"
    )

# ============================================================
# FILTER
# ============================================================

def apply_filter(df, spec):

    if spec is None:
        return df.copy()

    if spec[0] == "ge":

        _, lo = spec

        return df[
            df["atr_percentile"] >= lo
        ].copy()

    if spec[0] == "band":

        _, lo, hi = spec

        return df[
            (df["atr_percentile"] >= lo)
            & (df["atr_percentile"] < hi)
        ].copy()

    raise ValueError(spec)

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
        gp / gl
        if gl > 0
        else np.inf
    )

    tp_times = x.loc[
        x[res_col] == "TP",
        min_col
    ]

    compounded = (
        np.prod(
            1 + x[ret_col].to_numpy()/100
        ) - 1
    ) * 100

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
        "wr": (
            x[ret_col] > 0
        ).mean() * 100,
        "ev": x[ret_col].mean(),
        "pf": pf,
        "total": compounded,
        "median_tp_min": (
            tp_times.median()
            if len(tp_times)
            else np.nan
        ),
        "avg_atr": (
            x["atr_percentile"].mean()
        ),
    }

# ============================================================
# PER-FOLD
# ============================================================

rows = []

for h, tp, sl in CONFIGS:

    key = f"h{h}_tp{tp}_sl{sl}"

    for fold in range(1, 4):

        te = flips.iloc[
            edges[fold]:edges[fold+1]
        ].copy()

        for name, spec in FILTERS.items():

            g = apply_filter(
                te,
                spec
            )

            m = metrics(
                g,
                key
            )

            if m is None:
                continue

            rows.append({
                "setup": (
                    f"TP{tp:.0f}/SL{sl:.1f}"
                ),
                "fold": fold,
                "filter": name,
                **m,
            })

wf = pd.DataFrame(rows)

print("\n=== PER-FOLD RESULTS ===")

print(
    wf.sort_values(
        ["setup", "fold", "filter"]
    )
    .round({
        "wr": 1,
        "ev": 3,
        "pf": 2,
        "total": 2,
        "median_tp_min": 1,
        "avg_atr": 1,
    })
    .to_string(index=False)
)

# ============================================================
# EV LIFT MATRIX
# ============================================================

print("\n=== EV LIFT vs BASE ===")

for setup in wf["setup"].unique():

    print(f"\n--- {setup} / 72H ---")

    z = wf[
        wf["setup"] == setup
    ]

    for fold in range(1,4):

        f = z[
            z["fold"] == fold
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
            f"\nFold {fold} "
            f"BASE EV={base_ev:+.3f}%"
        )

        for name in FILTERS:

            if name == "BASE":
                continue

            r = f[
                f["filter"] == name
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
                f"{name:>9} "
                f"N={nn:2d} "
                f"EV={ev:+.3f}% "
                f"LIFT={ev-base_ev:+.3f}pp"
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

    for name, spec in FILTERS.items():

        g = apply_filter(
            oos,
            spec
        )

        m = metrics(
            g,
            key
        )

        if m is None:
            continue

        agg_rows.append({
            "setup": (
                f"TP{tp:.0f}/SL{sl:.1f}"
            ),
            "filter": name,
            **m,
        })

agg = pd.DataFrame(
    agg_rows
)

print("\n=== AGGREGATED OOS ===")

print(
    agg.sort_values(
        ["setup","ev"],
        ascending=[True,False]
    )
    .round({
        "wr":1,
        "ev":3,
        "pf":2,
        "total":2,
        "median_tp_min":1,
        "avg_atr":1,
    })
    .to_string(index=False)
)

# ============================================================
# CONSISTENCY SCORE
# ============================================================

consistency = []

for setup in wf["setup"].unique():

    z = wf[
        wf["setup"] == setup
    ]

    for name in FILTERS:

        if name == "BASE":
            continue

        lifts = []

        valid = True

        for fold in range(1,4):

            f = z[
                z["fold"] == fold
            ]

            b = f[
                f["filter"] == "BASE"
            ]

            r = f[
                f["filter"] == name
            ]

            if b.empty or r.empty:
                valid = False
                break

            lifts.append(
                float(r.iloc[0]["ev"])
                - float(b.iloc[0]["ev"])
            )

        if not valid:
            continue

        consistency.append({
            "setup": setup,
            "filter": name,
            "fold1_lift": lifts[0],
            "fold2_lift": lifts[1],
            "fold3_lift": lifts[2],
            "positive_folds":
                sum(x > 0 for x in lifts),
            "avg_lift":
                np.mean(lifts),
            "min_lift":
                np.min(lifts),
        })

c = pd.DataFrame(consistency)

print("\n=== CONSISTENCY RANKING ===")

print(
    c.sort_values(
        [
            "positive_folds",
            "avg_lift",
            "min_lift",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    )
    .round({
        "fold1_lift":3,
        "fold2_lift":3,
        "fold3_lift":3,
        "avg_lift":3,
        "min_lift":3,
    })
    .to_string(index=False)
)

# ============================================================
# SAVE
# ============================================================

Path(
    "data/research"
).mkdir(
    parents=True,
    exist_ok=True
)

wf.to_csv(
    "data/research/"
    "btc_matrix_4h_atr_band_walkforward.csv",
    index=False
)

agg.to_csv(
    "data/research/"
    "btc_matrix_4h_atr_band_agg.csv",
    index=False
)

c.to_csv(
    "data/research/"
    "btc_matrix_4h_atr_band_consistency.csv",
    index=False
)

print("\nSaved:")
print(
    " data/research/"
    "btc_matrix_4h_atr_band_walkforward.csv"
)
print(
    " data/research/"
    "btc_matrix_4h_atr_band_agg.csv"
)
print(
    " data/research/"
    "btc_matrix_4h_atr_band_consistency.csv"
)
