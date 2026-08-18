import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path(
    "data/research/"
    "btc_matrix_1h_regime_score.csv"
)

OUT = Path(
    "data/research/"
    "btc_matrix_1h_regime_score_rolling_blocks.csv"
)

THRESHOLD = 65.5833

df = pd.read_csv(SRC)

df["time"] = pd.to_datetime(
    df["time"],
    utc=True,
)

df = (
    df
    .sort_values("time")
    .dropna(
        subset=[
            "regime_score",
            "mfe_4h",
            "mfe_12h",
            "mfe_24h",
        ]
    )
    .reset_index(drop=True)
)

print("\n=== SAMPLE ===")
print("N:", len(df))
print(
    df.time.min(),
    "->",
    df.time.max()
)

print("\nFixed threshold:", THRESHOLD)


# ============================================================
# SIX-MONTH BLOCK
# ============================================================

df["year"] = df["time"].dt.year
df["half"] = np.where(
    df["time"].dt.month <= 6,
    "H1",
    "H2",
)

df["period"] = (
    df["year"].astype(str)
    + "-"
    + df["half"]
)


def metrics(x):

    if len(x) == 0:
        return None

    return {
        "n": len(x),

        "mfe_4h":
            x["mfe_4h"].mean(),

        "mfe_12h":
            x["mfe_12h"].mean(),

        "mfe_24h":
            x["mfe_24h"].mean(),

        "mae_24h":
            x["mae_24h"].mean(),

        "ret_24h":
            x["ret_24h"].mean(),

        "hit_050":
            (x["mfe_24h"] >= 0.50)
            .mean() * 100,

        "hit_100":
            (x["mfe_24h"] >= 1.00)
            .mean() * 100,

        "hit_150":
            (x["mfe_24h"] >= 1.50)
            .mean() * 100,
    }


rows = []

for period in sorted(
    df["period"].unique()
):

    block = df[
        df["period"] == period
    ].copy()

    high = block[
        block["regime_score"]
        >= THRESHOLD
    ].copy()

    a = metrics(block)
    h = metrics(high)

    if a is None:
        continue

    row = {
        "period": period,

        "all_n": a["n"],

        "high_n":
            h["n"]
            if h else 0,

        "coverage_pct":
            (
                h["n"] / a["n"] * 100
                if h else 0
            ),

        "all_mfe4":
            a["mfe_4h"],

        "high_mfe4":
            (
                h["mfe_4h"]
                if h else np.nan
            ),

        "mfe4_lift_x":
            (
                h["mfe_4h"]
                / a["mfe_4h"]
                if (
                    h
                    and a["mfe_4h"] != 0
                )
                else np.nan
            ),

        "all_mfe12":
            a["mfe_12h"],

        "high_mfe12":
            (
                h["mfe_12h"]
                if h else np.nan
            ),

        "mfe12_lift_x":
            (
                h["mfe_12h"]
                / a["mfe_12h"]
                if (
                    h
                    and a["mfe_12h"] != 0
                )
                else np.nan
            ),

        "all_mfe24":
            a["mfe_24h"],

        "high_mfe24":
            (
                h["mfe_24h"]
                if h else np.nan
            ),

        "mfe24_lift_x":
            (
                h["mfe_24h"]
                / a["mfe_24h"]
                if (
                    h
                    and a["mfe_24h"] != 0
                )
                else np.nan
            ),

        "all_hit100":
            a["hit_100"],

        "high_hit100":
            (
                h["hit_100"]
                if h else np.nan
            ),

        "hit100_delta":
            (
                h["hit_100"]
                - a["hit_100"]
                if h else np.nan
            ),

        "all_hit150":
            a["hit_150"],

        "high_hit150":
            (
                h["hit_150"]
                if h else np.nan
            ),

        "hit150_delta":
            (
                h["hit_150"]
                - a["hit_150"]
                if h else np.nan
            ),

        "all_ret24":
            a["ret_24h"],

        "high_ret24":
            (
                h["ret_24h"]
                if h else np.nan
            ),

        "ret24_delta":
            (
                h["ret_24h"]
                - a["ret_24h"]
                if h else np.nan
            ),
    }

    rows.append(row)

res = pd.DataFrame(rows)


print("\n=== SIX-MONTH BLOCKS ===")

print(
    res.round({
        "coverage_pct":1,
        "all_mfe4":3,
        "high_mfe4":3,
        "mfe4_lift_x":2,
        "all_mfe12":3,
        "high_mfe12":3,
        "mfe12_lift_x":2,
        "all_mfe24":3,
        "high_mfe24":3,
        "mfe24_lift_x":2,
        "all_hit100":1,
        "high_hit100":1,
        "hit100_delta":1,
        "all_hit150":1,
        "high_hit150":1,
        "hit150_delta":1,
        "all_ret24":3,
        "high_ret24":3,
        "ret24_delta":3,
    })
    .to_string(index=False)
)


# ============================================================
# CONSISTENCY SUMMARY
# ============================================================

valid = res[
    res["high_n"] >= 10
].copy()

print("\n=== CONSISTENCY SUMMARY ===")
print(
    "Periods with >=10 high-regime signals:",
    len(valid)
)

if len(valid):

    print(
        "MFE4 lift > 1.00x:",
        f"{(valid.mfe4_lift_x > 1).sum()}/{len(valid)}"
    )

    print(
        "MFE4 lift > 1.20x:",
        f"{(valid.mfe4_lift_x > 1.20).sum()}/{len(valid)}"
    )

    print(
        "MFE4 lift > 1.30x:",
        f"{(valid.mfe4_lift_x > 1.30).sum()}/{len(valid)}"
    )

    print(
        "MFE12 lift > 1.00x:",
        f"{(valid.mfe12_lift_x > 1).sum()}/{len(valid)}"
    )

    print(
        "Hit1.0 delta > 0:",
        f"{(valid.hit100_delta > 0).sum()}/{len(valid)}"
    )

    print(
        "Hit1.0 delta >= +10pp:",
        f"{(valid.hit100_delta >= 10).sum()}/{len(valid)}"
    )

    print(
        "\nMedian MFE4 lift:",
        f"{valid.mfe4_lift_x.median():.2f}x"
    )

    print(
        "Mean MFE4 lift:",
        f"{valid.mfe4_lift_x.mean():.2f}x"
    )

    print(
        "Median hit1.0 delta:",
        f"{valid.hit100_delta.median():+.1f}pp"
    )


# ============================================================
# BUY / SELL PER BLOCK
# ============================================================

side_rows = []

for period in sorted(
    df["period"].unique()
):

    block = df[
        df["period"] == period
    ]

    high = block[
        block["regime_score"]
        >= THRESHOLD
    ]

    for side in [
        "BUY",
        "SELL",
    ]:

        a = block[
            block["side"] == side
        ]

        h = high[
            high["side"] == side
        ]

        if (
            len(a) == 0
            or len(h) == 0
        ):
            continue

        am = metrics(a)
        hm = metrics(h)

        side_rows.append({
            "period": period,
            "side": side,

            "all_n": len(a),
            "high_n": len(h),

            "mfe4_lift_x":
                hm["mfe_4h"]
                / am["mfe_4h"],

            "mfe12_lift_x":
                hm["mfe_12h"]
                / am["mfe_12h"],

            "hit100_delta":
                hm["hit_100"]
                - am["hit_100"],

            "ret24_delta":
                hm["ret_24h"]
                - am["ret_24h"],
        })

side = pd.DataFrame(
    side_rows
)

print("\n=== BUY / SELL BLOCK ROBUSTNESS ===")

print(
    side.round({
        "mfe4_lift_x":2,
        "mfe12_lift_x":2,
        "hit100_delta":1,
        "ret24_delta":3,
    })
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

res.to_csv(
    OUT,
    index=False
)

side.to_csv(
    "data/research/"
    "btc_matrix_1h_regime_score_rolling_blocks_sides.csv",
    index=False
)

print("\nSaved:")
print(
    " data/research/"
    "btc_matrix_1h_regime_score_rolling_blocks.csv"
)
print(
    " data/research/"
    "btc_matrix_1h_regime_score_rolling_blocks_sides.csv"
)

print("\n" + "="*90)
print("INTERPRETATION")
print("="*90)
print(
    "Threshold is frozen at 65.5833."
)
print(
    "Primary question: does HIGH_REGIME produce "
    "positive MFE4H lift across most six-month blocks?"
)
print(
    "Secondary: does 1.0% target-hit lift stay positive "
    "across most blocks?"
)
