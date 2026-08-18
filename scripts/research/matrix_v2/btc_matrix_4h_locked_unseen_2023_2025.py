import sys
sys.path.insert(0, "/home/eylem/liqheat-ai")

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

from scripts import matrix_true_backtest as mtb


# ============================================================
# LOCKED RESEARCH SPECIFICATION
# ============================================================

SYMBOL = "BTCUSDT"

# Warmup is ONLY for ATR percentile history.
DOWNLOAD_START = pd.Timestamp("2022-10-01 00:00:00", tz="UTC")

VALIDATION_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-08-05 23:59:59.999", tz="UTC")

HORIZON_HOURS = 72
TP = 4.00
SL = 1.50

ATR_PERIOD = 14
ATR_LOOKBACK_BARS = 6 * 90       # 90 days of 4h bars
ATR_MIN_HISTORY = 6 * 30         # require at least 30 days
ATR_LOW = 50
ATR_HIGH = 70

HISTORICAL_FILE = Path(
    "data/research/"
    "btc_1m_historical_2022-10_to_2025-08.parquet"
)

CURRENT_FILE = Path(
    "data/market/binance-futures-um/"
    "BTCUSDT/1m/BTCUSDT-1m.parquet"
)

OUT_DIR = Path("data/research")
OUT_DIR.mkdir(parents=True, exist_ok=True)


print("\n" + "="*90)
print("LOCKED UNSEEN VALIDATION")
print("="*90)
print("Signal       : BTC 4H Matrix flip")
print("ATR regime   : trailing-90d percentile [50,70)")
print("TP           : 4.00%")
print("SL           : 1.50%")
print("Max hold     : 72h")
print("Validation   :", VALIDATION_START, "->", VALIDATION_END)
print("Warmup starts:", DOWNLOAD_START)
print("NO PARAMETER RE-SELECTION")


# ============================================================
# HISTORICAL DOWNLOAD
#
# Uses the SAME request_klines() and payload_to_frame()
# implementation as matrix_true_backtest.py.
# Does NOT overwrite canonical market parquet.
# ============================================================

def download_historical():

    if HISTORICAL_FILE.exists():

        existing = pd.read_parquet(HISTORICAL_FILE)

        existing["open_time"] = pd.to_datetime(
            existing["open_time"],
            utc=True
        )

        existing = (
            existing
            .sort_values("open_time")
            .drop_duplicates("open_time", keep="last")
            .reset_index(drop=True)
        )

        print(
            "\nHistorical cache found:",
            f"{len(existing):,}",
            "rows"
        )

    else:
        existing = pd.DataFrame()

    if existing.empty:

        cursor = DOWNLOAD_START

    else:

        cursor = max(
            DOWNLOAD_START,
            existing["open_time"].max()
            + pd.Timedelta(minutes=1)
        )

    # We only need historical download through validation end.
    end_time = VALIDATION_END

    if cursor > end_time:

        print("Historical cache already complete.")
        return existing

    print("\n=== HISTORICAL DOWNLOAD ===")
    print("From:", cursor)
    print("To  :", end_time)

    session = requests.Session()

    parts = []

    cursor_ms = mtb.timestamp_ms(cursor)
    end_ms = mtb.timestamp_ms(end_time)

    request_count = 0
    row_count = 0

    while cursor_ms <= end_ms:

        payload = mtb.request_klines(
            session,
            SYMBOL,
            cursor_ms,
            end_ms
        )

        if not payload:
            break

        part = mtb.payload_to_frame(
            payload,
            SYMBOL
        )

        part["open_time"] = pd.to_datetime(
            part["open_time"],
            utc=True
        )

        part = part[
            part["open_time"] <= end_time
        ]

        if part.empty:
            break

        parts.append(part)

        request_count += 1
        row_count += len(part)

        latest = part["open_time"].max()

        cursor_ms = (
            mtb.timestamp_ms(latest)
            + 60_000
        )

        if (
            request_count % 25 == 0
            or len(payload) < mtb.REQUEST_LIMIT
        ):
            print(
                f"requests={request_count:,} "
                f"downloaded={row_count:,} "
                f"latest={latest}",
                flush=True
            )

        if len(payload) < mtb.REQUEST_LIMIT:
            break

        time.sleep(0.08)

    frames = []

    if not existing.empty:
        frames.append(existing)

    frames.extend(parts)

    if not frames:
        raise RuntimeError(
            "No historical BTC data available"
        )

    historical = pd.concat(
        frames,
        ignore_index=True
    )

    historical = (
        historical
        .sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
    )

    historical = historical[
        historical["open_time"].between(
            DOWNLOAD_START,
            VALIDATION_END
        )
    ].reset_index(drop=True)

    historical.to_parquet(
        HISTORICAL_FILE,
        index=False
    )

    print(
        "\nHistorical saved:",
        HISTORICAL_FILE
    )

    return historical


historical = download_historical()


# ============================================================
# GAP VALIDATION
# ============================================================

historical["open_time"] = pd.to_datetime(
    historical["open_time"],
    utc=True
)

historical = (
    historical
    .sort_values("open_time")
    .drop_duplicates("open_time", keep="last")
    .reset_index(drop=True)
)

delta_min = (
    historical["open_time"]
    .diff()
    .dt.total_seconds()
    .div(60)
)

gap_count = int(
    (delta_min > 1.0).sum()
)

duplicate_count = int(
    historical["open_time"].duplicated().sum()
)

print("\n=== HISTORICAL DATA VALIDATION ===")
print("Rows       :", f"{len(historical):,}")
print("Range      :", historical.open_time.min(), "->", historical.open_time.max())
print("Duplicates :", duplicate_count)
print("Time gaps  :", gap_count)

if gap_count:
    print("\nLargest gaps:")
    tmp = pd.DataFrame({
        "time": historical["open_time"],
        "gap_min": delta_min
    })

    print(
        tmp[tmp.gap_min > 1]
        .sort_values("gap_min", ascending=False)
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# USE HISTORICAL + CURRENT ONLY FOR COMPLETE PATH ACCESS
#
# Historical validation outcomes themselves are bounded strictly
# by VALIDATION_END, so later discovery-period candles cannot
# enter any trade result.
# ============================================================

current = pd.read_parquet(CURRENT_FILE)

current["open_time"] = pd.to_datetime(
    current["open_time"],
    utc=True
)

one = pd.concat(
    [
        historical,
        current
    ],
    ignore_index=True
)

one = (
    one
    .sort_values("open_time")
    .drop_duplicates("open_time", keep="last")
    .reset_index(drop=True)
)

print("\n=== COMBINED RESEARCH DATA ===")
print("Rows :", f"{len(one):,}")
print("Range:", one.open_time.min(), "->", one.open_time.max())


# ============================================================
# REAL 4H MATRIX USING EXISTING PRODUCTION RESEARCH CODE
# ============================================================

h4 = mtb.aggregate_candles(
    one.copy(),
    "4h"
)

m4 = mtb.add_matrix(
    h4.copy(),
    "4h"
)

m4["close_time"] = pd.to_datetime(
    m4["close_time"],
    utc=True
)

m4["available_at"] = pd.to_datetime(
    m4["available_at"],
    utc=True
)


# ============================================================
# ATR(14) / PRICE
# ============================================================

high = m4["high"].astype(float)
low = m4["low"].astype(float)
close = m4["close"].astype(float)

prev_close = close.shift(1)

tr = pd.concat(
    [
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ],
    axis=1
).max(axis=1)

m4["atr"] = tr.ewm(
    alpha=1 / ATR_PERIOD,
    adjust=False,
    min_periods=ATR_PERIOD
).mean()

m4["atr_pct_price"] = (
    m4["atr"]
    / m4["close"]
    * 100
)


# ============================================================
# TRAILING 90-DAY ATR PERCENTILE
#
# STRICTLY PAST DATA ONLY.
# Current ATR value is NOT included in historical distribution.
# ============================================================

atr = m4["atr_pct_price"].to_numpy(
    dtype=float
)

atr_pctile = np.full(
    len(atr),
    np.nan
)

for i in range(len(atr)):

    start = max(
        0,
        i - ATR_LOOKBACK_BARS
    )

    history = atr[start:i]

    history = history[
        np.isfinite(history)
    ]

    if len(history) < ATR_MIN_HISTORY:
        continue

    current_atr = atr[i]

    if not np.isfinite(current_atr):
        continue

    atr_pctile[i] = (
        np.mean(
            history <= current_atr
        )
        * 100
    )

m4["atr_percentile"] = atr_pctile


# ============================================================
# MATRIX FLIPS
# ============================================================

flips = m4[
    m4["long_flip"].fillna(False)
    | m4["short_flip"].fillna(False)
].copy()

flips["side"] = np.where(
    flips["long_flip"].fillna(False),
    "BUY",
    "SELL"
)

flips["signal_time"] = (
    flips["available_at"]
)

flips = (
    flips
    .sort_values("signal_time")
    .reset_index(drop=True)
)


# ============================================================
# STRICT UNSEEN VALIDATION WINDOW
#
# Require the COMPLETE 72h outcome to remain before
# VALIDATION_END.
# ============================================================

latest_valid_signal = (
    VALIDATION_END
    - pd.Timedelta(hours=HORIZON_HOURS)
)

validation = flips[
    (flips["signal_time"] >= VALIDATION_START)
    & (flips["signal_time"] <= latest_valid_signal)
].copy()

validation = validation.dropna(
    subset=["atr_percentile"]
).reset_index(drop=True)

print("\n=== STRICT UNSEEN 4H MATRIX SAMPLE ===")
print("All flips :", len(validation))
print("Range     :", validation.signal_time.min(), "->", validation.signal_time.max())

print("\nATR percentile distribution:")
print(
    validation["atr_percentile"]
    .describe()
    .round(2)
    .to_string()
)


# ============================================================
# EXACT 1M FIRST-TOUCH
# ============================================================

one_idx = (
    one
    .set_index("open_time", drop=False)
    .sort_index()
)


def first_entry(signal_time):

    path = one_idx.loc[
        one_idx.index >= signal_time
    ]

    if path.empty:
        return None

    row = path.iloc[0]

    return (
        row["open_time"],
        float(row["open"])
    )


def replay(row):

    ent = first_entry(
        row["signal_time"]
    )

    if ent is None:
        return None

    entry_time, entry = ent

    end_time = (
        entry_time
        + pd.Timedelta(hours=HORIZON_HOURS)
    )

    # Strict validation boundary protection.
    if end_time > VALIDATION_END:
        return None

    path = one_idx.loc[
        (one_idx.index >= entry_time)
        & (one_idx.index < end_time)
    ]

    required = (
        HORIZON_HOURS
        * 60
    )

    if len(path) < required // 2:
        return None

    side = row["side"]

    if side == "BUY":

        tp_price = (
            entry
            * (1 + TP / 100)
        )

        sl_price = (
            entry
            * (1 - SL / 100)
        )

    else:

        # Same percentage-return convention used
        # in our locked discovery tests.
        tp_price = (
            entry
            / (1 + TP / 100)
        )

        sl_price = (
            entry
            / (1 - SL / 100)
        )

    for minute_no, (_, candle) in enumerate(
        path.iterrows(),
        start=1
    ):

        hi = float(candle["high"])
        lo = float(candle["low"])

        if side == "BUY":

            hit_tp = (
                hi >= tp_price
            )

            hit_sl = (
                lo <= sl_price
            )

        else:

            hit_tp = (
                lo <= tp_price
            )

            hit_sl = (
                hi >= sl_price
            )

        # Conservative assumption for same-minute ambiguity.
        if hit_tp and hit_sl:

            return {
                "result": "SL",
                "ret": -SL,
                "minutes": minute_no,
                "entry": entry,
                "entry_time": entry_time
            }

        if hit_tp:

            return {
                "result": "TP",
                "ret": TP,
                "minutes": minute_no,
                "entry": entry,
                "entry_time": entry_time
            }

        if hit_sl:

            return {
                "result": "SL",
                "ret": -SL,
                "minutes": minute_no,
                "entry": entry,
                "entry_time": entry_time
            }

    last = float(
        path.iloc[-1]["close"]
    )

    if side == "BUY":

        ret = (
            last / entry - 1
        ) * 100

    else:

        ret = (
            entry / last - 1
        ) * 100

    return {
        "result": "TIMEOUT",
        "ret": ret,
        "minutes": len(path),
        "entry": entry,
        "entry_time": entry_time
    }


outcomes = []

for _, row in validation.iterrows():

    result = replay(row)

    if result is None:
        outcomes.append({
            "result": None,
            "ret": np.nan,
            "minutes": np.nan,
            "entry": np.nan,
            "entry_time": pd.NaT
        })

    else:
        outcomes.append(result)

outcome_df = pd.DataFrame(
    outcomes
)

for column in outcome_df.columns:

    validation[column] = (
        outcome_df[column].values
    )

validation = validation.dropna(
    subset=["ret"]
).copy()


# ============================================================
# LOCKED FILTER
# ============================================================

selected = validation[
    (validation["atr_percentile"] >= ATR_LOW)
    & (validation["atr_percentile"] < ATR_HIGH)
].copy()


# ============================================================
# METRICS
# ============================================================

def metrics(df):

    if len(df) == 0:
        return None

    r = df["ret"].to_numpy(
        dtype=float
    )

    winners = r[r > 0]
    losers = r[r < 0]

    gp = winners.sum()
    gl = abs(losers.sum())

    pf = (
        gp / gl
        if gl > 0
        else np.inf
    )

    compounded = (
        np.prod(
            1 + r / 100
        ) - 1
    ) * 100

    return {
        "n": len(df),
        "tp": int(
            (df["result"] == "TP").sum()
        ),
        "sl": int(
            (df["result"] == "SL").sum()
        ),
        "timeout": int(
            (df["result"] == "TIMEOUT").sum()
        ),
        "wr": np.mean(r > 0) * 100,
        "ev": np.mean(r),
        "median": np.median(r),
        "pf": pf,
        "compounded": compounded,
        "min": np.min(r),
        "max": np.max(r),
    }


base_m = metrics(
    validation
)

locked_m = metrics(
    selected
)


def print_metrics(title, m):

    print("\n" + title)

    if m is None:
        print("NO TRADES")
        return

    print("N          :", m["n"])
    print("TP         :", m["tp"])
    print("SL         :", m["sl"])
    print("TIMEOUT    :", m["timeout"])
    print("Win rate   :", f'{m["wr"]:.2f}%')
    print("EV / trade :", f'{m["ev"]:+.4f}%')
    print("Median     :", f'{m["median"]:+.4f}%')
    print("PF         :", f'{m["pf"]:.3f}')
    print("Compounded :", f'{m["compounded"]:+.2f}%')
    print("Min        :", f'{m["min"]:+.4f}%')
    print("Max        :", f'{m["max"]:+.4f}%')


print_metrics(
    "=== UNSEEN BASELINE / ALL 4H MATRIX FLIPS ===",
    base_m
)

print_metrics(
    "=== LOCKED ATR50-70 RULE / UNSEEN ===",
    locked_m
)

if (
    base_m is not None
    and locked_m is not None
):

    print(
        "\nEV LIFT vs BASE:",
        f'{locked_m["ev"] - base_m["ev"]:+.4f} pp'
    )


# ============================================================
# YEAR-BY-YEAR ROBUSTNESS
# ============================================================

validation["year"] = (
    validation["signal_time"].dt.year
)

selected["year"] = (
    selected["signal_time"].dt.year
)

print("\n=== YEAR-BY-YEAR LOCKED RULE ===")

year_rows = []

for year in sorted(
    validation["year"].unique()
):

    base_y = validation[
        validation["year"] == year
    ]

    selected_y = selected[
        selected["year"] == year
    ]

    bm = metrics(base_y)
    sm = metrics(selected_y)

    row = {
        "year": year,
        "base_n": bm["n"] if bm else 0,
        "base_ev": bm["ev"] if bm else np.nan,
        "base_pf": bm["pf"] if bm else np.nan,
        "locked_n": sm["n"] if sm else 0,
        "locked_ev": sm["ev"] if sm else np.nan,
        "locked_pf": sm["pf"] if sm else np.nan,
        "lift": (
            sm["ev"] - bm["ev"]
            if sm and bm
            else np.nan
        )
    }

    year_rows.append(row)

year_df = pd.DataFrame(
    year_rows
)

print(
    year_df
    .round({
        "base_ev": 3,
        "base_pf": 2,
        "locked_ev": 3,
        "locked_pf": 2,
        "lift": 3,
    })
    .to_string(index=False)
)


# ============================================================
# HALF-YEAR BLOCK ROBUSTNESS
# ============================================================

validation["half"] = np.where(
    validation["signal_time"].dt.month <= 6,
    "H1",
    "H2"
)

validation["period"] = (
    validation["signal_time"]
    .dt.year
    .astype(str)
    + "-"
    + validation["half"]
)

selected["half"] = np.where(
    selected["signal_time"].dt.month <= 6,
    "H1",
    "H2"
)

selected["period"] = (
    selected["signal_time"]
    .dt.year
    .astype(str)
    + "-"
    + selected["half"]
)

period_rows = []

for period in sorted(
    validation["period"].unique()
):

    b = validation[
        validation["period"] == period
    ]

    s = selected[
        selected["period"] == period
    ]

    bm = metrics(b)
    sm = metrics(s)

    period_rows.append({
        "period": period,
        "base_n": bm["n"] if bm else 0,
        "base_ev": bm["ev"] if bm else np.nan,
        "locked_n": sm["n"] if sm else 0,
        "locked_ev": sm["ev"] if sm else np.nan,
        "locked_pf": sm["pf"] if sm else np.nan,
        "lift": (
            sm["ev"] - bm["ev"]
            if sm and bm
            else np.nan
        )
    })

period_df = pd.DataFrame(
    period_rows
)

print("\n=== HALF-YEAR ROBUSTNESS ===")

print(
    period_df
    .round({
        "base_ev": 3,
        "locked_ev": 3,
        "locked_pf": 2,
        "lift": 3,
    })
    .to_string(index=False)
)


# ============================================================
# INDIVIDUAL LOCKED TRADES
# ============================================================

print("\n=== ALL LOCKED UNSEEN TRADES ===")

show = [
    "signal_time",
    "side",
    "atr_percentile",
    "entry",
    "result",
    "ret",
    "minutes",
]

print(
    selected[show]
    .round({
        "atr_percentile": 2,
        "entry": 2,
        "ret": 4,
        "minutes": 0,
    })
    .to_string(index=False)
)


# ============================================================
# SIMPLE BOOTSTRAP — VALIDATION ONLY
#
# This is now meaningful because rule was frozen before seeing
# this historical period.
# ============================================================

if len(selected) >= 2:

    RNG = np.random.default_rng(20260818)

    r = selected["ret"].to_numpy(
        dtype=float
    )

    N_BOOT = 100_000

    indices = RNG.integers(
        0,
        len(r),
        size=(
            N_BOOT,
            len(r)
        )
    )

    samples = r[indices]

    boot_ev = samples.mean(
        axis=1
    )

    ci95 = np.quantile(
        boot_ev,
        [0.025, 0.975]
    )

    ci90 = np.quantile(
        boot_ev,
        [0.05, 0.95]
    )

    print("\n=== UNSEEN BOOTSTRAP ===")
    print("Iterations :", f"{N_BOOT:,}")
    print(
        "EV 95% CI  :",
        f"[{ci95[0]:+.4f}%, {ci95[1]:+.4f}%]"
    )
    print(
        "EV 90% CI  :",
        f"[{ci90[0]:+.4f}%, {ci90[1]:+.4f}%]"
    )
    print(
        "P(EV > 0)  :",
        f"{np.mean(boot_ev > 0)*100:.2f}%"
    )

    if base_m:

        print(
            "P(EV > BASE):",
            f"{np.mean(boot_ev > base_m['ev'])*100:.2f}%"
        )


# ============================================================
# SAVE
# ============================================================

validation.to_csv(
    OUT_DIR /
    "btc_matrix_4h_unseen_2023_2025_all_signals.csv",
    index=False
)

selected.to_csv(
    OUT_DIR /
    "btc_matrix_4h_unseen_2023_2025_locked_atr50_70.csv",
    index=False
)

year_df.to_csv(
    OUT_DIR /
    "btc_matrix_4h_unseen_2023_2025_yearly.csv",
    index=False
)

period_df.to_csv(
    OUT_DIR /
    "btc_matrix_4h_unseen_2023_2025_halfyear.csv",
    index=False
)

print("\nSaved:")
print(
    " data/research/"
    "btc_matrix_4h_unseen_2023_2025_all_signals.csv"
)
print(
    " data/research/"
    "btc_matrix_4h_unseen_2023_2025_locked_atr50_70.csv"
)
print(
    " data/research/"
    "btc_matrix_4h_unseen_2023_2025_yearly.csv"
)
print(
    " data/research/"
    "btc_matrix_4h_unseen_2023_2025_halfyear.csv"
)

print("\n" + "="*90)
print("IMPORTANT")
print("="*90)
print("Do NOT change ATR band, TP, SL or horizon after seeing this result.")
print("This is the frozen-rule unseen validation.")
