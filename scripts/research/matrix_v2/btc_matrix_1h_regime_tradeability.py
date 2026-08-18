import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# LOCKED SPECIFICATION
# ============================================================

THRESHOLD = 65.5833

CELLS = [
    (0.75, 0.50, 12),
    (0.75, 0.50, 24),

    (1.00, 0.75, 12),
    (1.00, 0.75, 24),

    (1.25, 1.00, 12),
    (1.25, 1.00, 24),
]

SCORE_FILE = Path(
    "data/research/btc_matrix_1h_regime_score.csv"
)

HIST_FILE = Path(
    "data/research/btc_1m_historical_2022-10_to_2025-08.parquet"
)

RECENT_FILE = Path(
    "data/market/binance-futures-um/"
    "BTCUSDT/1m/BTCUSDT-1m.parquet"
)

OUT_SUMMARY = Path(
    "data/research/"
    "btc_matrix_1h_regime_tradeability_summary.csv"
)

OUT_BLOCKS = Path(
    "data/research/"
    "btc_matrix_1h_regime_tradeability_blocks.csv"
)

OUT_TRADES = Path(
    "data/research/"
    "btc_matrix_1h_regime_tradeability_trades.csv"
)


# ============================================================
# LOAD SCORE SAMPLE
# ============================================================

score = pd.read_csv(SCORE_FILE)

time_col = None

for c in [
    "time",
    "signal_time",
    "close_time",
]:
    if c in score.columns:
        time_col = c
        break

if time_col is None:
    raise RuntimeError(
        "Cannot find signal timestamp column."
    )

score["signal_time"] = pd.to_datetime(
    score[time_col],
    utc=True,
)

score = (
    score
    .dropna(
        subset=[
            "signal_time",
            "regime_score",
            "side",
        ]
    )
    .sort_values("signal_time")
    .reset_index(drop=True)
)

score["side"] = (
    score["side"]
    .astype(str)
    .str.upper()
)

score = score[
    score["side"].isin(
        ["BUY", "SELL"]
    )
].copy()

score["high_regime"] = (
    score["regime_score"]
    >= THRESHOLD
)

score["period"] = (
    score["signal_time"].dt.year.astype(str)
    + "-"
    + np.where(
        score["signal_time"].dt.month <= 6,
        "H1",
        "H2",
    )
)

print("\n=== SCORE SAMPLE ===")
print("N:", len(score))
print(
    score.signal_time.min(),
    "->",
    score.signal_time.max()
)

print(
    "HIGH_REGIME:",
    int(score.high_regime.sum()),
    "/",
    len(score),
    f"({score.high_regime.mean()*100:.1f}%)"
)


# ============================================================
# LOAD 1M DATA
# ============================================================

hist = pd.read_parquet(HIST_FILE)
recent = pd.read_parquet(RECENT_FILE)

print("\nHistorical rows:", len(hist))
print("Recent rows    :", len(recent))


def detect_time_column(df):

    for c in [
        "open_time",
        "time",
        "timestamp",
        "datetime",
    ]:
        if c in df.columns:
            return c

    raise RuntimeError(
        "Cannot detect 1m timestamp column."
    )


def normalize_minute(df):

    df = df.copy()

    tc = detect_time_column(df)

    df["open_time"] = pd.to_datetime(
        df[tc],
        utc=True,
    )

    needed = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
    ]

    for c in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    return (
        df[needed]
        .dropna()
        .sort_values("open_time")
        .drop_duplicates(
            "open_time",
            keep="last",
        )
    )


hist = normalize_minute(hist)
recent = normalize_minute(recent)

minute = pd.concat(
    [hist, recent],
    ignore_index=True,
)

minute = (
    minute
    .sort_values("open_time")
    .drop_duplicates(
        "open_time",
        keep="last",
    )
    .reset_index(drop=True)
)

print("\n=== COMBINED 1M ===")
print("Rows :", len(minute))
print(
    "Range:",
    minute.open_time.min(),
    "->",
    minute.open_time.max()
)


# ============================================================
# INTEGER TIMESTAMP SEARCH
# Avoid tz-aware / tz-naive numpy problems.
# ============================================================

# Force nanoseconds explicitly.
# Pandas 3 may preserve ms/us resolution and astype("int64")
# would then be incompatible with Timestamp.value (always ns).
minute_ns = (
    minute["open_time"]
    .astype("datetime64[ns, UTC]")
    .astype("int64")
    .to_numpy()
)

print("\n=== TIMESTAMP SEARCH DIAGNOSTIC ===")
print("minute dtype :", minute["open_time"].dtype)
print("first minute :", minute["open_time"].iloc[0])
print("first ns     :", minute_ns[0])
print("last minute  :", minute["open_time"].iloc[-1])
print("last ns      :", minute_ns[-1])

_test_signal = pd.Timestamp(
    score["signal_time"].iloc[0]
)

_test_ns = int(_test_signal.value)

_test_idx = np.searchsorted(
    minute_ns,
    _test_ns,
    side="right",
)

print("test signal  :", _test_signal)
print("test ns      :", _test_ns)
print("search index :", _test_idx)

if _test_idx < len(minute):
    print(
        "matched minute:",
        minute["open_time"].iloc[_test_idx]
    )
else:
    print("ERROR: signal maps beyond minute dataset")

if not (
    minute_ns[0]
    < _test_ns
    < minute_ns[-1]
):
    raise RuntimeError(
        "Timestamp units/ranges still incompatible."
    )

high_arr = minute["high"].to_numpy(
    dtype=float
)

low_arr = minute["low"].to_numpy(
    dtype=float
)

close_arr = minute["close"].to_numpy(
    dtype=float
)


# ============================================================
# EXACT FIRST TOUCH
# Entry = first available 1m OPEN strictly after Matrix signal.
#
# If TP and SL both touch inside same 1m candle, intrabar order
# is unknowable from OHLC. Mark AMBIG rather than assuming.
# ============================================================

def replay_trade(
    signal_time,
    side,
    tp_pct,
    sl_pct,
    horizon_h,
):

    signal_ns = int(
        pd.Timestamp(signal_time).value
    )

    # first minute strictly after signal timestamp
    start = np.searchsorted(
        minute_ns,
        signal_ns,
        side="right",
    )

    if start >= len(minute):
        return None

    entry = float(
        minute.iloc[start]["open"]
    )

    entry_time = minute.iloc[start][
        "open_time"
    ]

    end_time = (
        pd.Timestamp(signal_time)
        + pd.Timedelta(
            hours=horizon_h
        )
    )

    end_ns = int(end_time.value)

    end = np.searchsorted(
        minute_ns,
        end_ns,
        side="right",
    )

    end = min(
        end,
        len(minute),
    )

    if end <= start:
        return None

    if side == "BUY":

        tp_price = entry * (
            1.0 + tp_pct / 100.0
        )

        sl_price = entry * (
            1.0 - sl_pct / 100.0
        )

    else:

        tp_price = entry * (
            1.0 - tp_pct / 100.0
        )

        sl_price = entry * (
            1.0 + sl_pct / 100.0
        )

    result = "TIMEOUT"
    ret = np.nan
    exit_time = minute.iloc[
        end - 1
    ]["open_time"]

    minutes_to_exit = (
        exit_time - entry_time
    ).total_seconds() / 60.0

    for i in range(start, end):

        hi = high_arr[i]
        lo = low_arr[i]

        if side == "BUY":

            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price

        else:

            tp_hit = lo <= tp_price
            sl_hit = hi >= sl_price

        if tp_hit and sl_hit:

            result = "AMBIG"
            ret = np.nan

            exit_time = minute.iloc[
                i
            ]["open_time"]

            minutes_to_exit = (
                exit_time - entry_time
            ).total_seconds() / 60.0

            break

        if tp_hit:

            result = "TP"
            ret = tp_pct

            exit_time = minute.iloc[
                i
            ]["open_time"]

            minutes_to_exit = (
                exit_time - entry_time
            ).total_seconds() / 60.0

            break

        if sl_hit:

            result = "SL"
            ret = -sl_pct

            exit_time = minute.iloc[
                i
            ]["open_time"]

            minutes_to_exit = (
                exit_time - entry_time
            ).total_seconds() / 60.0

            break

    if result == "TIMEOUT":

        final_close = float(
            close_arr[end - 1]
        )

        if side == "BUY":

            ret = (
                final_close / entry
                - 1.0
            ) * 100.0

        else:

            ret = (
                entry / final_close
                - 1.0
            ) * 100.0

    return {
        "entry_time": entry_time,
        "entry": entry,
        "exit_time": exit_time,
        "result": result,
        "ret": ret,
        "minutes": minutes_to_exit,
    }


# ============================================================
# REPLAY ALL LOCKED CELLS
# ============================================================

trade_rows = []

print("\n=== RUNNING EXACT 1M FIRST-TOUCH REPLAY ===")

for tp, sl, horizon in CELLS:

    print(
        f"TP={tp:.2f}% "
        f"SL={sl:.2f}% "
        f"H={horizon}h"
    )

    for _, row in score.iterrows():

        r = replay_trade(
            row["signal_time"],
            row["side"],
            tp,
            sl,
            horizon,
        )

        if r is None:
            continue

        trade_rows.append({
            "signal_time":
                row["signal_time"],

            "side":
                row["side"],

            "regime_score":
                row["regime_score"],

            "high_regime":
                row["high_regime"],

            "period":
                row["period"],

            "tp_pct": tp,
            "sl_pct": sl,
            "horizon_h": horizon,

            **r,
        })

trades = pd.DataFrame(
    trade_rows
)

print("\n=== REPLAY DIAGNOSTIC ===")
print("trade rows:", len(trade_rows))

if len(trades) == 0:
    raise RuntimeError(
        "Replay produced ZERO trades. "
        "Timestamp lookup or signal/minute alignment is broken."
    )

print(
    "Expected maximum:",
    len(score) * len(CELLS)
)

print(
    "Replay coverage:",
    f"{len(trades) / (len(score) * len(CELLS)) * 100:.2f}%"
)

print(
    "Trade columns:",
    list(trades.columns)
)


# ============================================================
# METRICS
# Exclude AMBIG from EV/PF because 1m ordering is unknowable.
# Report ambiguity separately.
# ============================================================

def metrics(x):

    if len(x) == 0:
        return None

    n_total = len(x)

    ambig = int(
        (x["result"] == "AMBIG").sum()
    )

    valid = x[
        x["result"] != "AMBIG"
    ].copy()

    if len(valid) == 0:
        return None

    tp_n = int(
        (valid["result"] == "TP").sum()
    )

    sl_n = int(
        (valid["result"] == "SL").sum()
    )

    timeout_n = int(
        (valid["result"] == "TIMEOUT").sum()
    )

    wins = valid[
        valid["ret"] > 0
    ]["ret"]

    losses = valid[
        valid["ret"] < 0
    ]["ret"]

    gross_win = wins.sum()
    gross_loss = -losses.sum()

    pf = (
        gross_win / gross_loss
        if gross_loss > 0
        else np.inf
    )

    return {
        "n": n_total,
        "valid_n": len(valid),
        "ambig": ambig,
        "tp": tp_n,
        "sl": sl_n,
        "timeout": timeout_n,

        "win_pct":
            (valid["ret"] > 0)
            .mean() * 100.0,

        "ev":
            valid["ret"].mean(),

        "median":
            valid["ret"].median(),

        "pf": pf,

        "total":
            valid["ret"].sum(),

        "avg_win":
            wins.mean()
            if len(wins)
            else np.nan,

        "avg_loss":
            losses.mean()
            if len(losses)
            else np.nan,
    }


# ============================================================
# OVERALL ALL vs HIGH
# ============================================================

summary_rows = []

for tp, sl, horizon in CELLS:

    cell = trades[
        (trades["tp_pct"] == tp)
        & (trades["sl_pct"] == sl)
        & (trades["horizon_h"] == horizon)
    ]

    for group_name, group in [
        ("ALL", cell),
        (
            "HIGH_REGIME",
            cell[cell["high_regime"]],
        ),
    ]:

        m = metrics(group)

        if m is None:
            continue

        summary_rows.append({
            "tp": tp,
            "sl": sl,
            "horizon_h": horizon,
            "group": group_name,
            **m,
        })

summary = pd.DataFrame(
    summary_rows
)

print("\n=== OVERALL TRADEABILITY ===")

print(
    summary.round(3)
    .to_string(index=False)
)


# ============================================================
# LIFT TABLE
# ============================================================

lift_rows = []

for tp, sl, horizon in CELLS:

    z = summary[
        (summary["tp"] == tp)
        & (summary["sl"] == sl)
        & (summary["horizon_h"] == horizon)
    ]

    a = z[
        z["group"] == "ALL"
    ]

    h = z[
        z["group"] == "HIGH_REGIME"
    ]

    if len(a) != 1 or len(h) != 1:
        continue

    a = a.iloc[0]
    h = h.iloc[0]

    lift_rows.append({
        "tp": tp,
        "sl": sl,
        "horizon_h": horizon,

        "all_n": a["valid_n"],
        "high_n": h["valid_n"],

        "all_ev": a["ev"],
        "high_ev": h["ev"],
        "ev_lift_pp":
            h["ev"] - a["ev"],

        "all_pf": a["pf"],
        "high_pf": h["pf"],
        "pf_lift":
            h["pf"] - a["pf"],

        "all_win_pct":
            a["win_pct"],

        "high_win_pct":
            h["win_pct"],

        "win_delta":
            h["win_pct"]
            - a["win_pct"],
    })

lift = pd.DataFrame(
    lift_rows
)

print("\n=== HIGH REGIME LIFT vs ALL ===")

print(
    lift.round(3)
    .to_string(index=False)
)


# ============================================================
# SIX-MONTH BLOCK ROBUSTNESS
# ============================================================

block_rows = []

for tp, sl, horizon in CELLS:

    cell = trades[
        (trades["tp_pct"] == tp)
        & (trades["sl_pct"] == sl)
        & (trades["horizon_h"] == horizon)
    ]

    for period in sorted(
        cell["period"].unique()
    ):

        b = cell[
            cell["period"] == period
        ]

        h = b[
            b["high_regime"]
        ]

        am = metrics(b)
        hm = metrics(h)

        if am is None or hm is None:
            continue

        block_rows.append({
            "tp": tp,
            "sl": sl,
            "horizon_h": horizon,
            "period": period,

            "all_n":
                am["valid_n"],

            "high_n":
                hm["valid_n"],

            "all_ev":
                am["ev"],

            "high_ev":
                hm["ev"],

            "ev_lift_pp":
                hm["ev"]
                - am["ev"],

            "all_pf":
                am["pf"],

            "high_pf":
                hm["pf"],

            "pf_lift":
                hm["pf"]
                - am["pf"],

            "all_win":
                am["win_pct"],

            "high_win":
                hm["win_pct"],

            "win_delta":
                hm["win_pct"]
                - am["win_pct"],
        })

blocks = pd.DataFrame(
    block_rows
)

print("\n=== SIX-MONTH TRADEABILITY BLOCKS ===")

print(
    blocks.round(3)
    .to_string(index=False)
)


# ============================================================
# CONSISTENCY
# Only periods with >=10 valid high-regime trades.
# ============================================================

print("\n=== CONSISTENCY BY LOCKED CELL ===")

for tp, sl, horizon in CELLS:

    z = blocks[
        (blocks["tp"] == tp)
        & (blocks["sl"] == sl)
        & (blocks["horizon_h"] == horizon)
        & (blocks["high_n"] >= 10)
    ]

    if len(z) == 0:
        continue

    print(
        f"\nTP={tp:.2f} "
        f"SL={sl:.2f} "
        f"H={horizon}h"
    )

    print(
        "Periods:",
        len(z)
    )

    print(
        "EV lift > 0:",
        f"{(z.ev_lift_pp > 0).sum()}/{len(z)}"
    )

    print(
        "PF lift > 0:",
        f"{(z.pf_lift > 0).sum()}/{len(z)}"
    )

    print(
        "Win-rate lift > 0:",
        f"{(z.win_delta > 0).sum()}/{len(z)}"
    )

    print(
        "Median EV lift:",
        f"{z.ev_lift_pp.median():+.3f}pp"
    )

    print(
        "Mean EV lift:",
        f"{z.ev_lift_pp.mean():+.3f}pp"
    )


# ============================================================
# SAVE
# ============================================================

OUT_SUMMARY.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

blocks.to_csv(
    OUT_BLOCKS,
    index=False,
)

trades.to_csv(
    OUT_TRADES,
    index=False,
)

lift.to_csv(
    "data/research/"
    "btc_matrix_1h_regime_tradeability_lift.csv",
    index=False,
)

print("\nSaved:")
print(" ", OUT_SUMMARY)
print(" ", OUT_BLOCKS)
print(" ", OUT_TRADES)
print(
    " data/research/"
    "btc_matrix_1h_regime_tradeability_lift.csv"
)

print("\n" + "="*90)
print("LOCKED RESEARCH INTERPRETATION")
print("="*90)
print(f"Regime threshold = {THRESHOLD}")
print("Threshold was NOT re-selected.")
print("TP/SL/horizon cells were pre-specified before this replay.")
print("1m exact first-touch used.")
print("Same-minute TP+SL collisions are AMBIG and excluded from EV/PF.")
print()
print(
    "Primary question:"
    " does HIGH_REGIME improve EV/PF over ALL"
    " across multiple locked cells and time blocks?"
)
print(
    "Do NOT select a new regime threshold from these results."
)
