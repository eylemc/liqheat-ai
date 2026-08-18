import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, "/home/eylem/liqheat-ai")
from scripts import matrix_true_backtest as mtb

SYMBOL = "BTCUSDT"
DATA = Path("data/market/binance-futures-um/BTCUSDT/1m/BTCUSDT-1m.parquet")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

TP_LIST = [1.00, 1.50, 2.00, 2.50, 3.00]
SL_LIST = [0.75, 1.00, 1.50, 2.00]
HORIZONS = [1440, 2880, 4320]   # 24h, 48h, 72h

# Same broad historical sample we've been using
START = pd.Timestamp("2025-08-06", tz="UTC")


# ------------------------------------------------------------
# LOAD 1M
# ------------------------------------------------------------

one = pd.read_parquet(DATA).copy()

one["open_time"] = pd.to_datetime(one["open_time"], utc=True)

if "close_time" in one.columns:
    one["close_time"] = pd.to_datetime(one["close_time"], utc=True)

one = (
    one[one["open_time"] >= START]
    .sort_values("open_time")
    .drop_duplicates("open_time")
    .reset_index(drop=True)
)

print("\n=== 1M DATA ===")
print("Candles :", f"{len(one):,}")
print("Range   :", one.open_time.min(), "->", one.open_time.max())


# ------------------------------------------------------------
# BUILD THE REAL 4H MATRIX
# ------------------------------------------------------------

h4 = mtb.aggregate_candles(one, "4h")
m4 = mtb.add_matrix(h4.copy(), "4h")

print("\nMatrix columns:")
print(list(m4.columns))

flips = m4[
    m4["long_flip"].fillna(False) |
    m4["short_flip"].fillna(False)
].copy()

flips["side"] = np.where(
    flips["long_flip"].fillna(False),
    "BUY",
    "SELL"
)

# Signal is actionable only after completed 4H candle
if "available_at" in flips.columns:
    flips["signal_time"] = pd.to_datetime(flips["available_at"], utc=True)
else:
    flips["signal_time"] = pd.to_datetime(flips["close_time"], utc=True)

flips = flips.sort_values("signal_time").reset_index(drop=True)

print("\n=== BTC 4H MATRIX FLIPS ===")
print("Signals:", len(flips))
print(
    flips[
        ["signal_time", "side", "close",
         "distance_to_vwma_pct", "channel_width_pct"]
    ].tail(30).to_string(index=False)
)


# ------------------------------------------------------------
# 1M PATH HELPERS
# ------------------------------------------------------------

one_idx = one.set_index("open_time").sort_index()


def get_entry(signal_time):
    """
    First 1m candle at/after Matrix becomes available.
    Entry = that candle's open.
    """
    p = one_idx.loc[one_idx.index >= signal_time]

    if p.empty:
        return None

    t = p.index[0]
    return t, float(p.iloc[0]["open"])


def path_after(entry_time, horizon_min):
    end = entry_time + pd.Timedelta(minutes=horizon_min)

    return one_idx.loc[
        (one_idx.index >= entry_time) &
        (one_idx.index < end)
    ].copy()


def directional_returns(side, entry, path):
    if side == "BUY":
        fav = (path["high"].astype(float) / entry - 1.0) * 100
        adv = (path["low"].astype(float) / entry - 1.0) * 100
        close_ret = (path["close"].astype(float) / entry - 1.0) * 100
    else:
        fav = (entry / path["low"].astype(float) - 1.0) * 100
        adv = (entry / path["high"].astype(float) - 1.0) * 100
        close_ret = (entry / path["close"].astype(float) - 1.0) * 100

    return fav, adv, close_ret


# ------------------------------------------------------------
# MFE / MAE ANATOMY
# ------------------------------------------------------------

anatomy = []

for _, r in flips.iterrows():

    ent = get_entry(r.signal_time)

    if ent is None:
        continue

    entry_time, entry = ent

    rec = {
        "time": r.signal_time,
        "entry_time": entry_time,
        "side": r.side,
        "entry": entry,
        "disp_4h": abs(float(r["distance_to_vwma_pct"])),
        "channel_4h": float(r["channel_width_pct"]),
    }

    for mins, name in [
        (60, "1h"),
        (240, "4h"),
        (720, "12h"),
        (1440, "24h"),
        (2880, "48h"),
        (4320, "72h"),
    ]:
        p = path_after(entry_time, mins)

        if p.empty:
            rec[f"mfe_{name}"] = np.nan
            rec[f"mae_{name}"] = np.nan
            rec[f"ret_{name}"] = np.nan
            continue

        fav, adv, close_ret = directional_returns(
            r.side, entry, p
        )

        rec[f"mfe_{name}"] = float(fav.max())
        rec[f"mae_{name}"] = float(adv.min())
        rec[f"ret_{name}"] = float(close_ret.iloc[-1])

    anatomy.append(rec)

an = pd.DataFrame(anatomy)

print("\n=== 4H RAW SIGNAL ANATOMY ===")
cols = [
    "time", "side", "entry", "disp_4h",
    "mfe_1h", "mae_1h",
    "mfe_4h", "mae_4h",
    "mfe_12h", "mae_12h",
    "mfe_24h", "mae_24h", "ret_24h",
    "mfe_48h", "mae_48h", "ret_48h",
    "mfe_72h", "mae_72h", "ret_72h",
]
print(an[cols].tail(30).round(4).to_string(index=False))


print("\n=== MFE / MAE SUMMARY ===")

for h in ["1h", "4h", "12h", "24h", "48h", "72h"]:
    d = an[[f"mfe_{h}", f"mae_{h}"]].dropna()

    if len(d) == 0:
        continue

    print(
        f"{h:>3} | "
        f"N={len(d):3d} "
        f"MFE avg={d[f'mfe_{h}'].mean():+.3f}% "
        f"med={d[f'mfe_{h}'].median():+.3f}% | "
        f"MAE avg={d[f'mae_{h}'].mean():+.3f}% "
        f"med={d[f'mae_{h}'].median():+.3f}%"
    )


# ------------------------------------------------------------
# NATURAL FLIP -> FLIP
# ------------------------------------------------------------

natural = []

for i in range(len(flips) - 1):

    a = flips.iloc[i]
    b = flips.iloc[i + 1]

    ent = get_entry(a.signal_time)

    if ent is None:
        continue

    entry_time, entry = ent

    # exit at first 1m open at/after next signal availability
    ex = get_entry(b.signal_time)

    if ex is None:
        continue

    exit_time, exit_price = ex

    if a.side == "BUY":
        ret = (exit_price / entry - 1) * 100
    else:
        ret = (entry / exit_price - 1) * 100

    p = one_idx.loc[
        (one_idx.index >= entry_time) &
        (one_idx.index < exit_time)
    ]

    if len(p):
        fav, adv, _ = directional_returns(a.side, entry, p)
        mfe = float(fav.max())
        mae = float(adv.min())
    else:
        mfe = np.nan
        mae = np.nan

    natural.append({
        "time": a.signal_time,
        "side": a.side,
        "entry": entry,
        "exit_time": exit_time,
        "exit": exit_price,
        "hours": (exit_time-entry_time).total_seconds()/3600,
        "ret": ret,
        "mfe": mfe,
        "mae": mae,
    })

nat = pd.DataFrame(natural)

print("\n=== NATURAL 4H MATRIX FLIP -> FLIP ===")

if len(nat):
    wins = nat[nat.ret > 0]
    losses = nat[nat.ret <= 0]

    print("Trades       :", len(nat))
    print("Winners      :", len(wins))
    print("Losers       :", len(losses))
    print("Win rate     :", f"{(nat.ret > 0).mean()*100:.1f}%")
    print("Average ret  :", f"{nat.ret.mean():+.3f}%")
    print("Median ret   :", f"{nat.ret.median():+.3f}%")
    print("Average MFE  :", f"{nat.mfe.mean():+.3f}%")
    print("Average MAE  :", f"{nat.mae.mean():+.3f}%")
    print("Avg hold     :", f"{nat.hours.mean():.1f}h")

    gross_win = wins.ret.sum()
    gross_loss = abs(losses.ret.sum())

    pf = gross_win / gross_loss if gross_loss else np.inf

    print("Profit factor:", f"{pf:.2f}")

    print("\nLast natural trades:")
    print(nat.tail(25).round(4).to_string(index=False))


# ------------------------------------------------------------
# EXACT FIRST TOUCH TP / SL
# Conservative handling if TP and SL occur inside same 1m bar:
# count as SL.
# ------------------------------------------------------------

def first_touch(side, entry, path, tp, sl):

    for ts, bar in path.iterrows():

        hi = float(bar.high)
        lo = float(bar.low)

        if side == "BUY":
            tp_price = entry * (1 + tp/100)
            sl_price = entry * (1 - sl/100)

            hit_tp = hi >= tp_price
            hit_sl = lo <= sl_price

        else:
            tp_price = entry / (1 + tp/100)
            sl_price = entry / (1 - sl/100)

            hit_tp = lo <= tp_price
            hit_sl = hi >= sl_price

        if hit_tp and hit_sl:
            return "SL", -sl, ts

        if hit_tp:
            return "TP", tp, ts

        if hit_sl:
            return "SL", -sl, ts

    # timeout at final close
    if path.empty:
        return "NONE", np.nan, None

    last = float(path.iloc[-1].close)

    if side == "BUY":
        ret = (last / entry - 1) * 100
    else:
        ret = (entry / last - 1) * 100

    return "TIMEOUT", ret, path.index[-1]


grid = []

for horizon in HORIZONS:

    for tp in TP_LIST:

        for sl in SL_LIST:

            trades = []

            for _, r in flips.iterrows():

                ent = get_entry(r.signal_time)

                if ent is None:
                    continue

                entry_time, entry = ent
                p = path_after(entry_time, horizon)

                if len(p) < max(1, horizon // 2):
                    # Don't use badly truncated end-of-dataset trades
                    continue

                result, ret, exit_time = first_touch(
                    r.side,
                    entry,
                    p,
                    tp,
                    sl
                )

                trades.append({
                    "result": result,
                    "ret": ret
                })

            d = pd.DataFrame(trades)

            if d.empty:
                continue

            n = len(d)
            tp_n = int((d.result == "TP").sum())
            sl_n = int((d.result == "SL").sum())
            to_n = int((d.result == "TIMEOUT").sum())

            ev = d.ret.mean()
            wr = (d.ret > 0).mean() * 100

            gw = d.loc[d.ret > 0, "ret"].sum()
            gl = abs(d.loc[d.ret < 0, "ret"].sum())

            pf = gw / gl if gl else np.inf

            grid.append({
                "horizon_h": horizon // 60,
                "tp": tp,
                "sl": sl,
                "n": n,
                "TP": tp_n,
                "SL": sl_n,
                "timeout": to_n,
                "WR": wr,
                "EV": ev,
                "PF": pf,
                "TOTAL": d.ret.sum(),
            })

grid = pd.DataFrame(grid)

print("\n=== 4H MATRIX TP/SL FIRST-TOUCH GRID ===")
print(
    grid.sort_values(
        ["horizon_h", "EV"],
        ascending=[True, False]
    ).round(3).to_string(index=False)
)


# ------------------------------------------------------------
# SIMPLE CHRONOLOGICAL HOLDOUT
# No parameter selection. Same TP/SL grid shown independently
# in first 70% and final 30%.
# ------------------------------------------------------------

cut = int(len(flips) * 0.70)

train_flips = flips.iloc[:cut]
test_flips = flips.iloc[cut:]

print("\n=== CHRONOLOGICAL SPLIT ===")
print("Train signals:", len(train_flips))
print(
    train_flips.signal_time.min(),
    "->",
    train_flips.signal_time.max()
)
print("Test signals :", len(test_flips))
print(
    test_flips.signal_time.min(),
    "->",
    test_flips.signal_time.max()
)


def evaluate_subset(subset, tp, sl, horizon):

    rows = []

    for _, r in subset.iterrows():

        ent = get_entry(r.signal_time)

        if ent is None:
            continue

        entry_time, entry = ent
        p = path_after(entry_time, horizon)

        if len(p) < max(1, horizon // 2):
            continue

        result, ret, exit_time = first_touch(
            r.side, entry, p, tp, sl
        )

        rows.append((result, ret))

    if not rows:
        return None

    d = pd.DataFrame(rows, columns=["result", "ret"])

    gw = d.loc[d.ret > 0, "ret"].sum()
    gl = abs(d.loc[d.ret < 0, "ret"].sum())

    return {
        "n": len(d),
        "wr": (d.ret > 0).mean()*100,
        "ev": d.ret.mean(),
        "pf": gw/gl if gl else np.inf
    }


holdout_rows = []

for horizon in HORIZONS:
    for tp in TP_LIST:
        for sl in SL_LIST:

            tr = evaluate_subset(
                train_flips, tp, sl, horizon
            )
            te = evaluate_subset(
                test_flips, tp, sl, horizon
            )

            if tr is None or te is None:
                continue

            holdout_rows.append({
                "H": horizon//60,
                "TP": tp,
                "SL": sl,
                "train_n": tr["n"],
                "train_WR": tr["wr"],
                "train_EV": tr["ev"],
                "train_PF": tr["pf"],
                "test_n": te["n"],
                "test_WR": te["wr"],
                "test_EV": te["ev"],
                "test_PF": te["pf"],
            })

ho = pd.DataFrame(holdout_rows)

print("\n=== TRAIN vs HOLDOUT — ALL PRE-SPECIFIED CELLS ===")
print(
    ho.sort_values(
        ["test_EV"],
        ascending=False
    ).round(3).to_string(index=False)
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

Path("data/research").mkdir(parents=True, exist_ok=True)

an.to_csv(
    "data/research/btc_matrix_4h_raw_anatomy.csv",
    index=False
)

grid.to_csv(
    "data/research/btc_matrix_4h_raw_grid.csv",
    index=False
)

ho.to_csv(
    "data/research/btc_matrix_4h_raw_holdout.csv",
    index=False
)

print("\nSaved:")
print(" data/research/btc_matrix_4h_raw_anatomy.csv")
print(" data/research/btc_matrix_4h_raw_grid.csv")
print(" data/research/btc_matrix_4h_raw_holdout.csv")
