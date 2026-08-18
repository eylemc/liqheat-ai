import pandas as pd
import numpy as np
from pathlib import Path

TRADES_FILE = Path(
    "data/research/"
    "btc_matrix_1h_regime_tradeability_trades.csv"
)

OUT_OVERALL = Path(
    "data/research/"
    "btc_matrix_1h_regime_three_state_overall.csv"
)

OUT_SIDES = Path(
    "data/research/"
    "btc_matrix_1h_regime_three_state_sides.csv"
)

OUT_BLOCKS = Path(
    "data/research/"
    "btc_matrix_1h_regime_three_state_blocks.csv"
)

# ============================================================
# FROZEN REGIME CUTS
# Learned earlier from TRAIN only.
# NO re-selection here.
# ============================================================

BLOCK_CUT = 36.0070
VALID_CUT = 65.5833

# Only the two already pre-specified cells we want to inspect.
CELLS = [
    (1.25, 1.00, 12),
    (1.25, 1.00, 24),
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(TRADES_FILE)

df["signal_time"] = pd.to_datetime(
    df["signal_time"],
    utc=True
)

df["side"] = (
    df["side"]
    .astype(str)
    .str.upper()
)

df["regime_score"] = pd.to_numeric(
    df["regime_score"],
    errors="coerce"
)

df["ret"] = pd.to_numeric(
    df["ret"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "signal_time",
        "side",
        "regime_score",
    ]
).copy()


# ============================================================
# THREE-STATE CLASSIFICATION
# ============================================================

def state_from_score(x):

    if x < BLOCK_CUT:
        return "BLOCK"

    if x < VALID_CUT:
        return "CAUTION"

    return "VALID"


df["regime_state"] = (
    df["regime_score"]
    .apply(state_from_score)
)

df["period"] = (
    df["signal_time"].dt.year.astype(str)
    + "-"
    + np.where(
        df["signal_time"].dt.month <= 6,
        "H1",
        "H2"
    )
)

print("\n=== FROZEN THREE-STATE SPEC ===")
print(f"BLOCK   : score < {BLOCK_CUT}")
print(
    f"CAUTION : {BLOCK_CUT} <= score < {VALID_CUT}"
)
print(f"VALID   : score >= {VALID_CUT}")
print("No threshold re-selection.")


# ============================================================
# FILTER TO LOCKED CELLS
# ============================================================

parts = []

for tp, sl, horizon in CELLS:

    z = df[
        (df["tp_pct"] == tp)
        & (df["sl_pct"] == sl)
        & (df["horizon_h"] == horizon)
    ].copy()

    if len(z):
        parts.append(z)

work = pd.concat(
    parts,
    ignore_index=True
)

print("\n=== INPUT SAMPLE ===")
print("Rows:", len(work))
print(
    work["signal_time"].min(),
    "->",
    work["signal_time"].max()
)


# ============================================================
# METRICS
# ============================================================

def metrics(x):

    if len(x) == 0:
        return None

    total_n = len(x)

    ambig = int(
        (x["result"] == "AMBIG").sum()
    )

    valid = x[
        x["result"] != "AMBIG"
    ].copy()

    valid = valid.dropna(
        subset=["ret"]
    )

    if len(valid) == 0:
        return None

    wins = valid.loc[
        valid["ret"] > 0,
        "ret"
    ]

    losses = valid.loc[
        valid["ret"] < 0,
        "ret"
    ]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    compounded = (
        np.prod(
            1 + valid["ret"].to_numpy(dtype=float) / 100
        ) - 1
    ) * 100

    return {
        "n": total_n,
        "valid_n": len(valid),
        "ambig": ambig,

        "tp": int(
            (valid["result"] == "TP").sum()
        ),

        "sl": int(
            (valid["result"] == "SL").sum()
        ),

        "timeout": int(
            (valid["result"] == "TIMEOUT").sum()
        ),

        "win_pct":
            (valid["ret"] > 0).mean() * 100,

        "ev":
            valid["ret"].mean(),

        "median":
            valid["ret"].median(),

        "pf": pf,

        "compounded":
            compounded,

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
# OVERALL THREE-STATE
# ============================================================

overall_rows = []

for tp, sl, horizon in CELLS:

    cell = work[
        (work["tp_pct"] == tp)
        & (work["sl_pct"] == sl)
        & (work["horizon_h"] == horizon)
    ]

    for state in [
        "BLOCK",
        "CAUTION",
        "VALID",
    ]:

        g = cell[
            cell["regime_state"] == state
        ]

        m = metrics(g)

        if m is None:
            continue

        overall_rows.append({
            "tp": tp,
            "sl": sl,
            "horizon_h": horizon,
            "state": state,
            "avg_score":
                g["regime_score"].mean(),
            **m,
        })

overall = pd.DataFrame(
    overall_rows
)

print("\n=== OVERALL BLOCK / CAUTION / VALID ===")

print(
    overall.round({
        "avg_score":2,
        "win_pct":1,
        "ev":3,
        "median":3,
        "pf":2,
        "compounded":2,
        "avg_win":3,
        "avg_loss":3,
    })
    .to_string(index=False)
)


# ============================================================
# MONOTONICITY CHECK
# ============================================================

print("\n=== MONOTONICITY CHECK ===")

for tp, sl, horizon in CELLS:

    z = overall[
        (overall["tp"] == tp)
        & (overall["sl"] == sl)
        & (overall["horizon_h"] == horizon)
    ].set_index("state")

    print(
        f"\nTP={tp:.2f}% "
        f"SL={sl:.2f}% "
        f"H={horizon}h"
    )

    if all(
        s in z.index
        for s in [
            "BLOCK",
            "CAUTION",
            "VALID",
        ]
    ):

        evs = [
            z.loc["BLOCK", "ev"],
            z.loc["CAUTION", "ev"],
            z.loc["VALID", "ev"],
        ]

        pfs = [
            z.loc["BLOCK", "pf"],
            z.loc["CAUTION", "pf"],
            z.loc["VALID", "pf"],
        ]

        wrs = [
            z.loc["BLOCK", "win_pct"],
            z.loc["CAUTION", "win_pct"],
            z.loc["VALID", "win_pct"],
        ]

        print(
            "EV      :",
            " -> ".join(
                f"{x:+.3f}%"
                for x in evs
            )
        )

        print(
            "PF      :",
            " -> ".join(
                f"{x:.2f}"
                for x in pfs
            )
        )

        print(
            "Win rate:",
            " -> ".join(
                f"{x:.1f}%"
                for x in wrs
            )
        )

        print(
            "EV monotonic:",
            bool(
                evs[0]
                <= evs[1]
                <= evs[2]
            )
        )

        print(
            "PF monotonic:",
            bool(
                pfs[0]
                <= pfs[1]
                <= pfs[2]
            )
        )


# ============================================================
# BUY / SELL
# ============================================================

side_rows = []

for tp, sl, horizon in CELLS:

    cell = work[
        (work["tp_pct"] == tp)
        & (work["sl_pct"] == sl)
        & (work["horizon_h"] == horizon)
    ]

    for side in [
        "BUY",
        "SELL",
    ]:

        for state in [
            "BLOCK",
            "CAUTION",
            "VALID",
        ]:

            g = cell[
                (cell["side"] == side)
                & (cell["regime_state"] == state)
            ]

            m = metrics(g)

            if m is None:
                continue

            side_rows.append({
                "tp": tp,
                "sl": sl,
                "horizon_h": horizon,
                "side": side,
                "state": state,
                **m,
            })

sides = pd.DataFrame(
    side_rows
)

print("\n=== BUY / SELL BY REGIME STATE ===")

print(
    sides.round({
        "win_pct":1,
        "ev":3,
        "median":3,
        "pf":2,
        "compounded":2,
    })
    .to_string(index=False)
)


# ============================================================
# SIX-MONTH BLOCKS
# ============================================================

block_rows = []

for tp, sl, horizon in CELLS:

    cell = work[
        (work["tp_pct"] == tp)
        & (work["sl_pct"] == sl)
        & (work["horizon_h"] == horizon)
    ]

    for period in sorted(
        cell["period"].unique()
    ):

        b = cell[
            cell["period"] == period
        ]

        for state in [
            "BLOCK",
            "CAUTION",
            "VALID",
        ]:

            g = b[
                b["regime_state"] == state
            ]

            m = metrics(g)

            if m is None:
                continue

            block_rows.append({
                "tp": tp,
                "sl": sl,
                "horizon_h": horizon,
                "period": period,
                "state": state,
                **m,
            })

blocks = pd.DataFrame(
    block_rows
)

print("\n=== SIX-MONTH BLOCKS / THREE STATES ===")

print(
    blocks.round({
        "win_pct":1,
        "ev":3,
        "pf":2,
        "compounded":2,
    })
    .to_string(index=False)
)


# ============================================================
# PERIOD CONSISTENCY: VALID vs BLOCK
# ============================================================

print("\n=== PERIOD CONSISTENCY: VALID vs BLOCK ===")

for tp, sl, horizon in CELLS:

    z = blocks[
        (blocks["tp"] == tp)
        & (blocks["sl"] == sl)
        & (blocks["horizon_h"] == horizon)
    ]

    periods = []

    for period in sorted(
        z["period"].unique()
    ):

        p = z[
            z["period"] == period
        ].set_index("state")

        if not (
            "BLOCK" in p.index
            and "VALID" in p.index
        ):
            continue

        # Avoid over-reading tiny cells.
        if (
            p.loc["BLOCK", "valid_n"] < 10
            or p.loc["VALID", "valid_n"] < 10
        ):
            continue

        periods.append({
            "period": period,

            "block_ev":
                p.loc["BLOCK", "ev"],

            "valid_ev":
                p.loc["VALID", "ev"],

            "ev_delta":
                p.loc["VALID", "ev"]
                - p.loc["BLOCK", "ev"],

            "block_pf":
                p.loc["BLOCK", "pf"],

            "valid_pf":
                p.loc["VALID", "pf"],

            "pf_delta":
                p.loc["VALID", "pf"]
                - p.loc["BLOCK", "pf"],
        })

    c = pd.DataFrame(periods)

    print(
        f"\nTP={tp:.2f}% "
        f"SL={sl:.2f}% "
        f"H={horizon}h"
    )

    if c.empty:
        print("No periods with >=10 trades in both states.")
        continue

    print(
        c.round(3)
        .to_string(index=False)
    )

    print(
        "VALID EV > BLOCK:",
        f"{(c.ev_delta > 0).sum()}/{len(c)}"
    )

    print(
        "VALID PF > BLOCK:",
        f"{(c.pf_delta > 0).sum()}/{len(c)}"
    )

    print(
        "Median EV advantage:",
        f"{c.ev_delta.median():+.3f}pp"
    )


# ============================================================
# SANITY CHECK
# ============================================================

print("\n=== SANITY CHECK ===")

for tp, sl, horizon in CELLS:

    cell = work[
        (work["tp_pct"] == tp)
        & (work["sl_pct"] == sl)
        & (work["horizon_h"] == horizon)
    ]

    counts = (
        cell["regime_state"]
        .value_counts()
    )

    total_states = int(
        counts.sum()
    )

    print(
        f"TP={tp:.2f} "
        f"SL={sl:.2f} "
        f"H={horizon}h | "
        f"ALL={len(cell)} "
        f"BLOCK={counts.get('BLOCK',0)} "
        f"CAUTION={counts.get('CAUTION',0)} "
        f"VALID={counts.get('VALID',0)} "
        f"SUM={total_states} "
        f"OK={total_states == len(cell)}"
    )


# ============================================================
# SAVE
# ============================================================

OUT_OVERALL.parent.mkdir(
    parents=True,
    exist_ok=True
)

overall.to_csv(
    OUT_OVERALL,
    index=False
)

sides.to_csv(
    OUT_SIDES,
    index=False
)

blocks.to_csv(
    OUT_BLOCKS,
    index=False
)

print("\nSaved:")
print(" ", OUT_OVERALL)
print(" ", OUT_SIDES)
print(" ", OUT_BLOCKS)

print("\n" + "="*90)
print("INTERPRETATION")
print("="*90)
print(
    "Frozen regime cuts only. "
    "No threshold tuning."
)
print(
    "Primary question: "
    "does trade quality improve "
    "BLOCK -> CAUTION -> VALID?"
)
print(
    "Secondary question: "
    "is VALID better than BLOCK "
    "across most six-month periods?"
)
