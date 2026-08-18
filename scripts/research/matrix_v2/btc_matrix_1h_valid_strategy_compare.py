import pandas as pd
import numpy as np
from pathlib import Path

TRADES_FILE = Path(
    "data/research/btc_matrix_1h_regime_tradeability_trades.csv"
)

OUT = Path(
    "data/research/btc_matrix_1h_valid_strategy_compare.csv"
)

OUT_BLOCKS = Path(
    "data/research/btc_matrix_1h_valid_strategy_compare_blocks.csv"
)

VALID_CUT = 65.5833

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
    utc=True,
)

df["entry_time"] = pd.to_datetime(
    df["entry_time"],
    utc=True,
)

df["exit_time"] = pd.to_datetime(
    df["exit_time"],
    utc=True,
)

df["ret"] = pd.to_numeric(
    df["ret"],
    errors="coerce",
)

df["regime_score"] = pd.to_numeric(
    df["regime_score"],
    errors="coerce",
)

df["side"] = (
    df["side"]
    .astype(str)
    .str.upper()
)

df["valid_gate"] = (
    df["regime_score"]
    >= VALID_CUT
)

df["period"] = (
    df["signal_time"].dt.year.astype(str)
    + "-"
    + np.where(
        df["signal_time"].dt.month <= 6,
        "H1",
        "H2",
    )
)

# ============================================================
# HELPERS
# ============================================================

def max_drawdown_from_returns(returns):

    r = np.asarray(
        returns,
        dtype=float,
    )

    if len(r) == 0:
        return np.nan

    equity = np.cumprod(
        1.0 + r / 100.0
    )

    peak = np.maximum.accumulate(
        equity
    )

    dd = (
        equity / peak - 1.0
    ) * 100.0

    return float(dd.min())


def max_consecutive_losses(returns):

    r = np.asarray(
        returns,
        dtype=float,
    )

    best = 0
    current = 0

    for x in r:

        if x < 0:
            current += 1
            best = max(
                best,
                current,
            )
        else:
            current = 0

    return best


def profit_factor(returns):

    r = np.asarray(
        returns,
        dtype=float,
    )

    gp = r[r > 0].sum()
    gl = abs(
        r[r < 0].sum()
    )

    if gl == 0:
        return np.inf

    return gp / gl


def compounded_return(returns):

    r = np.asarray(
        returns,
        dtype=float,
    )

    return (
        np.prod(
            1.0 + r / 100.0
        ) - 1.0
    ) * 100.0


def metrics(x):

    if len(x) == 0:
        return None

    z = (
        x[
            x["result"] != "AMBIG"
        ]
        .dropna(
            subset=["ret"]
        )
        .sort_values("signal_time")
        .copy()
    )

    if len(z) == 0:
        return None

    r = z["ret"].to_numpy(
        dtype=float,
    )

    holding_hours = (
        (
            z["exit_time"]
            - z["entry_time"]
        )
        .dt.total_seconds()
        .div(3600)
    )

    return {
        "n": len(z),

        "tp": int(
            (z["result"] == "TP").sum()
        ),

        "sl": int(
            (z["result"] == "SL").sum()
        ),

        "timeout": int(
            (z["result"] == "TIMEOUT").sum()
        ),

        "win_pct":
            np.mean(r > 0) * 100,

        "ev":
            r.mean(),

        "median":
            np.median(r),

        "pf":
            profit_factor(r),

        "compounded":
            compounded_return(r),

        "max_dd":
            max_drawdown_from_returns(r),

        "max_consec_loss":
            max_consecutive_losses(r),

        "avg_holding_h":
            holding_hours.mean(),

        "median_holding_h":
            holding_hours.median(),

        "avg_win":
            r[r > 0].mean()
            if np.any(r > 0)
            else np.nan,

        "avg_loss":
            r[r < 0].mean()
            if np.any(r < 0)
            else np.nan,
    }


# ============================================================
# OVERALL STRATEGY COMPARISON
# ============================================================

rows = []

for tp, sl, horizon in CELLS:

    cell = df[
        (df["tp_pct"] == tp)
        & (df["sl_pct"] == sl)
        & (df["horizon_h"] == horizon)
    ].copy()

    groups = {
        "ALL": cell,
        "VALID_ONLY": cell[
            cell["valid_gate"]
        ],
    }

    for name, g in groups.items():

        m = metrics(g)

        if m is None:
            continue

        rows.append({
            "tp": tp,
            "sl": sl,
            "horizon_h": horizon,
            "strategy": name,
            **m,
        })

summary = pd.DataFrame(
    rows
)

print("\n=== ALL vs VALID-ONLY STRATEGY ===")

print(
    summary.round({
        "win_pct":1,
        "ev":3,
        "median":3,
        "pf":2,
        "compounded":2,
        "max_dd":2,
        "avg_holding_h":2,
        "median_holding_h":2,
        "avg_win":3,
        "avg_loss":3,
    })
    .to_string(index=False)
)


# ============================================================
# DIRECT LIFT
# ============================================================

lift_rows = []

for tp, sl, horizon in CELLS:

    z = summary[
        (summary["tp"] == tp)
        & (summary["sl"] == sl)
        & (summary["horizon_h"] == horizon)
    ].set_index("strategy")

    if not (
        "ALL" in z.index
        and "VALID_ONLY" in z.index
    ):
        continue

    a = z.loc["ALL"]
    v = z.loc["VALID_ONLY"]

    lift_rows.append({
        "tp": tp,
        "sl": sl,
        "horizon_h": horizon,

        "trade_reduction_pct":
            (
                1
                - v["n"] / a["n"]
            ) * 100,

        "ev_lift_pp":
            v["ev"] - a["ev"],

        "pf_lift":
            v["pf"] - a["pf"],

        "win_lift_pp":
            v["win_pct"]
            - a["win_pct"],

        "max_dd_change_pp":
            v["max_dd"]
            - a["max_dd"],

        "max_consec_loss_change":
            v["max_consec_loss"]
            - a["max_consec_loss"],

        "compounded_change_pp":
            v["compounded"]
            - a["compounded"],
    })

lift = pd.DataFrame(
    lift_rows
)

print("\n=== VALID GATE IMPACT ===")

print(
    lift.round({
        "trade_reduction_pct":1,
        "ev_lift_pp":3,
        "pf_lift":2,
        "win_lift_pp":1,
        "max_dd_change_pp":2,
        "compounded_change_pp":2,
    })
    .to_string(index=False)
)


# ============================================================
# SIX-MONTH BLOCKS
# ============================================================

block_rows = []

for tp, sl, horizon in CELLS:

    cell = df[
        (df["tp_pct"] == tp)
        & (df["sl_pct"] == sl)
        & (df["horizon_h"] == horizon)
    ].copy()

    for period in sorted(
        cell["period"].unique()
    ):

        b = cell[
            cell["period"] == period
        ]

        for name, g in [
            ("ALL", b),
            (
                "VALID_ONLY",
                b[b["valid_gate"]],
            ),
        ]:

            m = metrics(g)

            if m is None:
                continue

            block_rows.append({
                "tp": tp,
                "sl": sl,
                "horizon_h": horizon,
                "period": period,
                "strategy": name,
                **m,
            })

blocks = pd.DataFrame(
    block_rows
)

print("\n=== SIX-MONTH STRATEGY BLOCKS ===")

print(
    blocks.round({
        "win_pct":1,
        "ev":3,
        "pf":2,
        "compounded":2,
        "max_dd":2,
    })
    .to_string(index=False)
)


# ============================================================
# BLOCK CONSISTENCY
# ============================================================

print("\n=== BLOCK CONSISTENCY ===")

for tp, sl, horizon in CELLS:

    z = blocks[
        (blocks["tp"] == tp)
        & (blocks["sl"] == sl)
        & (blocks["horizon_h"] == horizon)
    ]

    comps = []

    for period in sorted(
        z["period"].unique()
    ):

        p = z[
            z["period"] == period
        ].set_index("strategy")

        if not (
            "ALL" in p.index
            and "VALID_ONLY" in p.index
        ):
            continue

        if p.loc[
            "VALID_ONLY",
            "n"
        ] < 10:
            continue

        comps.append({
            "period": period,

            "ev_delta":
                p.loc[
                    "VALID_ONLY",
                    "ev"
                ]
                - p.loc[
                    "ALL",
                    "ev"
                ],

            "pf_delta":
                p.loc[
                    "VALID_ONLY",
                    "pf"
                ]
                - p.loc[
                    "ALL",
                    "pf"
                ],

            "dd_delta":
                p.loc[
                    "VALID_ONLY",
                    "max_dd"
                ]
                - p.loc[
                    "ALL",
                    "max_dd"
                ],
        })

    c = pd.DataFrame(
        comps
    )

    print(
        f"\nTP={tp:.2f}% "
        f"SL={sl:.2f}% "
        f"H={horizon}h"
    )

    if c.empty:
        print(
            "No sufficiently populated blocks."
        )
        continue

    print(
        c.round(3)
        .to_string(index=False)
    )

    print(
        "EV improved:",
        f"{(c.ev_delta > 0).sum()}/{len(c)}"
    )

    print(
        "PF improved:",
        f"{(c.pf_delta > 0).sum()}/{len(c)}"
    )

    print(
        "Drawdown improved:",
        f"{(c.dd_delta > 0).sum()}/{len(c)}"
    )

    print(
        "Median EV lift:",
        f"{c.ev_delta.median():+.3f}pp"
    )


# ============================================================
# BUY / SELL STRATEGY COMPARISON
# ============================================================

print("\n=== BUY / SELL VALID IMPACT ===")

side_rows = []

for tp, sl, horizon in CELLS:

    cell = df[
        (df["tp_pct"] == tp)
        & (df["sl_pct"] == sl)
        & (df["horizon_h"] == horizon)
    ]

    for side in [
        "BUY",
        "SELL",
    ]:

        base = cell[
            cell["side"] == side
        ]

        valid = base[
            base["valid_gate"]
        ]

        bm = metrics(base)
        vm = metrics(valid)

        if bm is None or vm is None:
            continue

        side_rows.append({
            "tp": tp,
            "sl": sl,
            "horizon_h": horizon,
            "side": side,

            "all_n":
                bm["n"],

            "valid_n":
                vm["n"],

            "all_ev":
                bm["ev"],

            "valid_ev":
                vm["ev"],

            "ev_lift":
                vm["ev"]
                - bm["ev"],

            "all_pf":
                bm["pf"],

            "valid_pf":
                vm["pf"],

            "pf_lift":
                vm["pf"]
                - bm["pf"],

            "all_dd":
                bm["max_dd"],

            "valid_dd":
                vm["max_dd"],
        })

sides = pd.DataFrame(
    side_rows
)

print(
    sides.round(3)
    .to_string(index=False)
)


# ============================================================
# SANITY
# ============================================================

print("\n=== SANITY ===")

for tp, sl, horizon in CELLS:

    cell = df[
        (df["tp_pct"] == tp)
        & (df["sl_pct"] == sl)
        & (df["horizon_h"] == horizon)
    ]

    valid = cell[
        cell["valid_gate"]
    ]

    print(
        f"H={horizon}h "
        f"ALL={len(cell)} "
        f"VALID={len(valid)} "
        f"coverage={len(valid)/len(cell)*100:.1f}%"
    )


# ============================================================
# SAVE
# ============================================================

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    OUT,
    index=False,
)

blocks.to_csv(
    OUT_BLOCKS,
    index=False,
)

lift.to_csv(
    "data/research/"
    "btc_matrix_1h_valid_strategy_compare_lift.csv",
    index=False,
)

sides.to_csv(
    "data/research/"
    "btc_matrix_1h_valid_strategy_compare_sides.csv",
    index=False,
)

print("\nSaved:")
print(" ", OUT)
print(" ", OUT_BLOCKS)
print(
    " data/research/"
    "btc_matrix_1h_valid_strategy_compare_lift.csv"
)
print(
    " data/research/"
    "btc_matrix_1h_valid_strategy_compare_sides.csv"
)

print("\n" + "="*90)
print("INTERPRETATION")
print("="*90)
print(
    "Frozen VALID gate = regime_score >= 65.5833"
)
print(
    "Strategy specification remains TP1.25 / SL1.00 "
    "at 12h and 24h."
)
print(
    "Primary question: does VALID-only improve "
    "EV/PF and reduce drawdown despite lower trade count?"
)
