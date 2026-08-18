import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path(
    "data/research/"
    "btc_matrix_1h_regime_score.csv"
)

OUT = Path(
    "data/research/"
    "btc_matrix_1h_regime_score_holdout.csv"
)

# ============================================================
# LOAD
# ============================================================

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

# ============================================================
# CHRONOLOGICAL SPLIT
#
# 50% TRAIN
# 25% VALIDATION
# 25% TEST
# ============================================================

n = len(df)

i1 = int(n * 0.50)
i2 = int(n * 0.75)

train = df.iloc[:i1].copy()
val   = df.iloc[i1:i2].copy()
test  = df.iloc[i2:].copy()

print("\n=== SPLIT ===")

for name, x in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test),
]:
    print(
        f"{name:10s} "
        f"N={len(x):4d} | "
        f"{x.time.min()} -> {x.time.max()}"
    )

# ============================================================
# LEARN THRESHOLD ON TRAIN ONLY
#
# High regime = top quartile of TRAIN score distribution.
# ============================================================

threshold = float(
    train["regime_score"].quantile(0.75)
)

print("\n=== TRAIN-LEARNED THRESHOLD ===")
print(
    f"75th percentile regime_score = "
    f"{threshold:.4f}"
)

# ============================================================
# METRICS
# ============================================================

def metrics(x):

    if len(x) == 0:
        return None

    return {
        "n": len(x),

        "avg_score":
            x["regime_score"].mean(),

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
            (
                x["mfe_24h"]
                >= 0.50
            ).mean() * 100,

        "hit_100":
            (
                x["mfe_24h"]
                >= 1.00
            ).mean() * 100,

        "hit_150":
            (
                x["mfe_24h"]
                >= 1.50
            ).mean() * 100,
    }

# ============================================================
# ALL vs HIGH-REGIME
# ============================================================

rows = []

for split_name, x in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test),
]:

    all_m = metrics(x)

    high = x[
        x["regime_score"]
        >= threshold
    ].copy()

    high_m = metrics(high)

    rows.append({
        "split": split_name,
        "group": "ALL",
        **all_m,
    })

    rows.append({
        "split": split_name,
        "group": "HIGH_REGIME",
        **high_m,
    })

result = pd.DataFrame(rows)

print("\n=== ALL vs HIGH REGIME ===")

print(
    result.round({
        "avg_score":3,
        "mfe_4h":3,
        "mfe_12h":3,
        "mfe_24h":3,
        "mae_24h":3,
        "ret_24h":3,
        "hit_050":1,
        "hit_100":1,
        "hit_150":1,
    })
    .to_string(index=False)
)

# ============================================================
# LIFT TABLE
# ============================================================

print("\n=== HIGH REGIME LIFT vs ALL ===")

lift_rows = []

for split_name in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    a = result[
        (result["split"] == split_name)
        & (result["group"] == "ALL")
    ].iloc[0]

    h = result[
        (result["split"] == split_name)
        & (result["group"] == "HIGH_REGIME")
    ].iloc[0]

    lift_rows.append({
        "split": split_name,
        "all_n": int(a["n"]),
        "high_n": int(h["n"]),
        "coverage_pct":
            h["n"] / a["n"] * 100,

        "mfe4_lift_x":
            h["mfe_4h"]
            / a["mfe_4h"],

        "mfe12_lift_x":
            h["mfe_12h"]
            / a["mfe_12h"],

        "mfe24_lift_x":
            h["mfe_24h"]
            / a["mfe_24h"],

        "hit050_delta":
            h["hit_050"]
            - a["hit_050"],

        "hit100_delta":
            h["hit_100"]
            - a["hit_100"],

        "hit150_delta":
            h["hit_150"]
            - a["hit_150"],

        "ret24_delta":
            h["ret_24h"]
            - a["ret_24h"],
    })

lift = pd.DataFrame(lift_rows)

print(
    lift.round({
        "coverage_pct":1,
        "mfe4_lift_x":2,
        "mfe12_lift_x":2,
        "mfe24_lift_x":2,
        "hit050_delta":1,
        "hit100_delta":1,
        "hit150_delta":1,
        "ret24_delta":3,
    })
    .to_string(index=False)
)

# ============================================================
# TEST QUARTILES USING TRAIN-LEARNED CUTS
#
# This is extra robustness:
# learn all quartile boundaries on TRAIN only,
# apply unchanged to VAL and TEST.
# ============================================================

q25 = float(
    train["regime_score"]
    .quantile(0.25)
)

q50 = float(
    train["regime_score"]
    .quantile(0.50)
)

q75 = threshold

print("\n=== TRAIN-LEARNED QUARTILE CUTS ===")
print(f"Q25={q25:.4f}")
print(f"Q50={q50:.4f}")
print(f"Q75={q75:.4f}")

def assign_bucket(x):

    if x < q25:
        return "Q1 RANGE"

    if x < q50:
        return "Q2"

    if x < q75:
        return "Q3"

    return "Q4 TREND"


quartile_rows = []

for split_name, x in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test),
]:

    z = x.copy()

    z["bucket"] = (
        z["regime_score"]
        .apply(assign_bucket)
    )

    for bucket in [
        "Q1 RANGE",
        "Q2",
        "Q3",
        "Q4 TREND",
    ]:

        g = z[
            z["bucket"] == bucket
        ]

        if len(g) == 0:
            continue

        m = metrics(g)

        quartile_rows.append({
            "split": split_name,
            "bucket": bucket,
            **m,
        })

quartiles = pd.DataFrame(
    quartile_rows
)

print(
    "\n=== TRAIN-DEFINED QUARTILES "
    "APPLIED TO ALL SPLITS ==="
)

print(
    quartiles.round({
        "avg_score":3,
        "mfe_4h":3,
        "mfe_12h":3,
        "mfe_24h":3,
        "mae_24h":3,
        "ret_24h":3,
        "hit_050":1,
        "hit_100":1,
        "hit_150":1,
    })
    .to_string(index=False)
)

# ============================================================
# BUY / SELL HIGH-REGIME CHECK
# ============================================================

print("\n=== HIGH REGIME BUY / SELL ===")

side_rows = []

for split_name, x in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test),
]:

    high = x[
        x["regime_score"]
        >= threshold
    ]

    for side in [
        "BUY",
        "SELL",
    ]:

        g = high[
            high["side"] == side
        ]

        if len(g) == 0:
            continue

        m = metrics(g)

        side_rows.append({
            "split": split_name,
            "side": side,
            **m,
        })

side_df = pd.DataFrame(
    side_rows
)

print(
    side_df.round({
        "mfe_4h":3,
        "mfe_12h":3,
        "mfe_24h":3,
        "ret_24h":3,
        "hit_100":1,
    })
    .to_string(index=False)
)

# ============================================================
# SIMPLE BOOTSTRAP ON VALIDATION + TEST
#
# Compare HIGH_REGIME MFE4H against ALL MFE4H.
# This is descriptive robustness, not a final p-value.
# ============================================================

RNG = np.random.default_rng(42)
N_BOOT = 50000

print("\n=== BOOTSTRAP MFE4H LIFT ===")

for split_name, x in [
    ("VALIDATION", val),
    ("TEST", test),
]:

    high = x[
        x["regime_score"]
        >= threshold
    ]

    if (
        len(x) < 5
        or len(high) < 5
    ):
        continue

    all_mfe = (
        x["mfe_4h"]
        .to_numpy(dtype=float)
    )

    high_mfe = (
        high["mfe_4h"]
        .to_numpy(dtype=float)
    )

    diffs = np.empty(
        N_BOOT,
        dtype=float
    )

    for i in range(N_BOOT):

        a = RNG.choice(
            all_mfe,
            size=len(all_mfe),
            replace=True
        )

        h = RNG.choice(
            high_mfe,
            size=len(high_mfe),
            replace=True
        )

        diffs[i] = (
            h.mean()
            - a.mean()
        )

    ci = np.quantile(
        diffs,
        [0.025, 0.975]
    )

    print(
        f"{split_name:10s} "
        f"diff={high_mfe.mean()-all_mfe.mean():+.4f}% "
        f"95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}] "
        f"P(diff>0)={(diffs>0).mean()*100:.1f}%"
    )

# ============================================================
# SAVE
# ============================================================

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUT,
    index=False,
)

lift.to_csv(
    "data/research/"
    "btc_matrix_1h_regime_score_holdout_lift.csv",
    index=False,
)

quartiles.to_csv(
    "data/research/"
    "btc_matrix_1h_regime_score_holdout_quartiles.csv",
    index=False,
)

side_df.to_csv(
    "data/research/"
    "btc_matrix_1h_regime_score_holdout_sides.csv",
    index=False,
)

print("\nSaved:")
print(
    " data/research/"
    "btc_matrix_1h_regime_score_holdout.csv"
)
print(
    " data/research/"
    "btc_matrix_1h_regime_score_holdout_lift.csv"
)
print(
    " data/research/"
    "btc_matrix_1h_regime_score_holdout_quartiles.csv"
)
print(
    " data/research/"
    "btc_matrix_1h_regime_score_holdout_sides.csv"
)

print("\n" + "="*90)
print("PRIMARY VALIDATION CRITERIA")
print("="*90)
print(
    "1) TRAIN learns Q75 threshold only."
)
print(
    "2) VALIDATION and TEST must use that exact same threshold."
)
print(
    "3) Prefer MFE4H lift > 1.30x in both VAL and TEST."
)
print(
    "4) Prefer 24H >=1.0% hit-rate lift > +10pp in both."
)
print(
    "5) BUY and SELL should both improve if this is truly a regime gate."
)
