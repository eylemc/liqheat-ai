import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path(
    "data/research/"
    "btc_matrix_4h_atr_band_walkforward.csv"
)

SIG = Path(
    "data/research/"
    "btc_matrix_4h_atr_walkforward_signals.csv"
)

RNG = np.random.default_rng(42)

N_BOOT = 100000
N_PERM = 100000

# Locked specification
TARGET_FILTER = "ATR50-70"
SETUP = "TP4/SL1.5"

# ============================================================
# LOAD PER-FOLD RESULTS
# ============================================================

wf = pd.read_csv(SRC)

print("\n=== LOCKED RESEARCH SPEC ===")
print("4H Matrix flip")
print("ATR percentile: [50,70)")
print("TP = 4.00%")
print("SL = 1.50%")
print("Horizon = 72h")
print("No threshold re-selection")

# ============================================================
# LOAD SIGNAL-LEVEL DATA
# ============================================================

sig = pd.read_csv(SIG)

time_col = (
    "signal_time"
    if "signal_time" in sig.columns
    else "time"
)

sig[time_col] = pd.to_datetime(
    sig[time_col],
    utc=True
)

sig = sig.sort_values(time_col).reset_index(drop=True)

# Same OOS definition as prior walk-forward:
# first 25% is initial training; remaining 75% = aggregated OOS.
n_total = len(sig)
oos_start = int(n_total * 0.25)

oos = sig.iloc[oos_start:].copy()

print("\n=== SAMPLE ===")
print("Total signals :", len(sig))
print("OOS signals   :", len(oos))
print(
    "OOS range     :",
    oos[time_col].min(),
    "->",
    oos[time_col].max()
)

# ============================================================
# FIND THE LOCKED OUTCOME COLUMNS
# ============================================================

ret_candidates = [
    c for c in sig.columns
    if (
        "h72" in c.lower()
        and "tp4" in c.lower()
        and "sl1.5" in c.lower()
        and c.lower().endswith("_ret")
    )
]

result_candidates = [
    c for c in sig.columns
    if (
        "h72" in c.lower()
        and "tp4" in c.lower()
        and "sl1.5" in c.lower()
        and c.lower().endswith("_result")
    )
]

if not ret_candidates:
    print("\nAvailable columns:")
    print(sig.columns.tolist())
    raise RuntimeError(
        "Could not identify 72h TP4 SL1.5 return column"
    )

RET_COL = ret_candidates[0]
RES_COL = (
    result_candidates[0]
    if result_candidates
    else None
)

print("\nReturn column :", RET_COL)
print("Result column :", RES_COL)

oos = oos.dropna(
    subset=[
        RET_COL,
        "atr_percentile"
    ]
).copy()

selected = oos[
    (oos["atr_percentile"] >= 50)
    & (oos["atr_percentile"] < 70)
].copy()

base = oos.copy()

if len(selected) == 0:
    raise RuntimeError("No ATR50-70 signals found")

r = selected[RET_COL].to_numpy(dtype=float)
base_r = base[RET_COL].to_numpy(dtype=float)

# ============================================================
# METRIC HELPERS
# ============================================================

def profit_factor(x):
    x = np.asarray(x, dtype=float)

    gp = x[x > 0].sum()
    gl = abs(x[x < 0].sum())

    if gl == 0:
        return np.inf

    return gp / gl


def compounded(x):
    x = np.asarray(x, dtype=float)

    return (
        np.prod(1 + x/100.0) - 1
    ) * 100


def summary(x):
    x = np.asarray(x, dtype=float)

    return {
        "n": len(x),
        "win_pct": np.mean(x > 0) * 100,
        "mean_ev": np.mean(x),
        "median": np.median(x),
        "pf": profit_factor(x),
        "compounded": compounded(x),
        "min": np.min(x),
        "max": np.max(x),
    }

s_sel = summary(r)
s_base = summary(base_r)

print("\n=== OBSERVED LOCKED RESULT ===")

for k,v in s_sel.items():
    if k == "n":
        print(f"{k:12s}: {v}")
    else:
        print(f"{k:12s}: {v:+.4f}")

print("\n=== BASE OOS ===")

for k,v in s_base.items():
    if k == "n":
        print(f"{k:12s}: {v}")
    else:
        print(f"{k:12s}: {v:+.4f}")

print(
    "\nObserved EV lift:",
    f"{s_sel['mean_ev'] - s_base['mean_ev']:+.4f} pp"
)

# ============================================================
# BOOTSTRAP SELECTED TRADES
#
# Question:
# If these ATR50-70 trades are representative of their
# underlying regime, how uncertain is their mean EV?
# ============================================================

boot_idx = RNG.integers(
    0,
    len(r),
    size=(N_BOOT, len(r))
)

boot_samples = r[boot_idx]

boot_ev = boot_samples.mean(axis=1)

boot_comp = (
    np.prod(
        1 + boot_samples/100.0,
        axis=1
    ) - 1
) * 100

ev_ci = np.quantile(
    boot_ev,
    [0.025, 0.05, 0.50, 0.95, 0.975]
)

comp_ci = np.quantile(
    boot_comp,
    [0.025, 0.50, 0.975]
)

p_ev_positive = np.mean(
    boot_ev > 0
)

p_ev_above_base = np.mean(
    boot_ev > s_base["mean_ev"]
)

print("\n=== BOOTSTRAP / SELECTED ATR50-70 ===")
print(f"Iterations        : {N_BOOT:,}")
print(
    "EV 95% CI        : "
    f"[{ev_ci[0]:+.4f}%, {ev_ci[4]:+.4f}%]"
)
print(
    "EV 90% CI        : "
    f"[{ev_ci[1]:+.4f}%, {ev_ci[3]:+.4f}%]"
)
print(
    "EV bootstrap med : "
    f"{ev_ci[2]:+.4f}%"
)
print(
    "P(EV > 0)        : "
    f"{p_ev_positive*100:.2f}%"
)
print(
    "P(EV > BASE EV)  : "
    f"{p_ev_above_base*100:.2f}%"
)

print(
    "Compounded 95%CI : "
    f"[{comp_ci[0]:+.2f}%, {comp_ci[2]:+.2f}%]"
)

# ============================================================
# PERMUTATION / RANDOM SUBSET TEST
#
# Null:
# ATR50-70 has no special relation to outcome.
#
# Draw N selected trades randomly from all OOS Matrix signals
# and compare mean EV to observed ATR50-70 EV.
# ============================================================

k = len(r)
N = len(base_r)

if k > N:
    raise RuntimeError("Selected sample larger than base")

perm_ev = np.empty(N_PERM)
perm_pf = np.empty(N_PERM)

for i in range(N_PERM):

    idx = RNG.choice(
        N,
        size=k,
        replace=False
    )

    sample = base_r[idx]

    perm_ev[i] = sample.mean()
    perm_pf[i] = profit_factor(sample)

obs_ev = r.mean()
obs_pf = profit_factor(r)

# one-sided empirical p-values
p_perm_ev = (
    np.sum(perm_ev >= obs_ev) + 1
) / (N_PERM + 1)

finite_pf = np.isfinite(perm_pf)

if np.isfinite(obs_pf) and finite_pf.any():
    p_perm_pf = (
        np.sum(
            perm_pf[finite_pf] >= obs_pf
        ) + 1
    ) / (
        finite_pf.sum() + 1
    )
else:
    p_perm_pf = np.nan

print("\n=== RANDOM SUBSET / PERMUTATION TEST ===")
print(f"Iterations          : {N_PERM:,}")
print(f"OOS pool            : {N}")
print(f"Selected N          : {k}")
print(
    f"Observed EV         : {obs_ev:+.4f}%"
)
print(
    f"Random subset EV avg: {perm_ev.mean():+.4f}%"
)
print(
    "Random EV 95% range: "
    f"[{np.quantile(perm_ev,.025):+.4f}%, "
    f"{np.quantile(perm_ev,.975):+.4f}%]"
)
print(
    f"Empirical p(EV)     : {p_perm_ev:.6f}"
)
print(
    f"Observed PF         : {obs_pf:.3f}"
)
print(
    f"Empirical p(PF)     : {p_perm_pf:.6f}"
)

# ============================================================
# LABEL-PERMUTATION TEST FOR ATR BAND
#
# Keep returns fixed; randomly shuffle ATR percentiles across
# OOS signals, then select whichever observations land in
# [50,70). This tests the ATR/outcome association more directly.
# ============================================================

atr_values = (
    oos["atr_percentile"]
    .to_numpy(dtype=float)
)

ret_values = (
    oos[RET_COL]
    .to_numpy(dtype=float)
)

label_ev = []

for _ in range(N_PERM):

    shuffled = RNG.permutation(
        atr_values
    )

    mask = (
        (shuffled >= 50)
        & (shuffled < 70)
    )

    if mask.sum() == 0:
        continue

    label_ev.append(
        ret_values[mask].mean()
    )

label_ev = np.asarray(
    label_ev,
    dtype=float
)

p_label = (
    np.sum(label_ev >= obs_ev) + 1
) / (len(label_ev) + 1)

print("\n=== ATR LABEL PERMUTATION ===")
print(
    "Valid iterations    :",
    f"{len(label_ev):,}"
)
print(
    f"Null EV mean        : {label_ev.mean():+.4f}%"
)
print(
    "Null EV 95% range   : "
    f"[{np.quantile(label_ev,.025):+.4f}%, "
    f"{np.quantile(label_ev,.975):+.4f}%]"
)
print(
    f"Empirical p-value   : {p_label:.6f}"
)

# ============================================================
# JACKKNIFE
#
# With N~9, one trade can matter a lot.
# Remove one selected trade at a time and recalc EV.
# ============================================================

jack = []

for i in range(len(r)):

    z = np.delete(r, i)

    jack.append({
        "removed": i,
        "ev": z.mean(),
        "pf": profit_factor(z),
    })

jack = pd.DataFrame(jack)

print("\n=== JACKKNIFE ROBUSTNESS ===")
print(
    "Leave-one-out EV min:",
    f"{jack.ev.min():+.4f}%"
)
print(
    "Leave-one-out EV max:",
    f"{jack.ev.max():+.4f}%"
)
print(
    "Leave-one-out EV avg:",
    f"{jack.ev.mean():+.4f}%"
)
print(
    "All leave-one-out EV positive:",
    bool((jack.ev > 0).all())
)

# ============================================================
# INDIVIDUAL SELECTED TRADES
# ============================================================

show = [
    time_col,
    "side",
    "atr_percentile",
    RET_COL,
]

if RES_COL:
    show.append(RES_COL)

print("\n=== LOCKED ATR50-70 OOS TRADES ===")

print(
    selected[show]
    .round({
        "atr_percentile":2,
        RET_COL:4,
    })
    .to_string(index=False)
)

# ============================================================
# SAVE SUMMARY
# ============================================================

out = {
    "selected_n": len(r),
    "selected_ev": obs_ev,
    "selected_pf": obs_pf,
    "selected_compounded": compounded(r),

    "base_n": len(base_r),
    "base_ev": s_base["mean_ev"],
    "base_pf": s_base["pf"],

    "bootstrap_ev_ci_95_low": ev_ci[0],
    "bootstrap_ev_ci_95_high": ev_ci[4],
    "bootstrap_prob_ev_positive": p_ev_positive,
    "bootstrap_prob_ev_above_base": p_ev_above_base,

    "random_subset_p_ev": p_perm_ev,
    "random_subset_p_pf": p_perm_pf,

    "atr_label_permutation_p_ev": p_label,

    "jackknife_ev_min": jack.ev.min(),
    "jackknife_ev_max": jack.ev.max(),
}

pd.DataFrame(
    [out]
).to_csv(
    "data/research/"
    "btc_matrix_4h_atr50_70_significance.csv",
    index=False
)

print(
    "\nSaved: data/research/"
    "btc_matrix_4h_atr50_70_significance.csv"
)
