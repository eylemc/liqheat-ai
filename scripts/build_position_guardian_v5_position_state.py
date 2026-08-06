#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, f1_score

# Reuse the validated causal topology + Koinvizyon Matrix pipeline.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import build_position_guardian_v4_koinvizyon_matrix as v4  # noqa: E402

OUT = Path("data/reports/position_guardian_v5_position_state")
MODEL_OUT = Path("data/models/position_guardian_v5_position_state")

ACTION_NAMES = {0: "EXIT_BETTER", 1: "NO_CLEAR_EDGE", 2: "HOLD_BETTER"}
SIDES = ("LONG", "SHORT")
HORIZONS = (15, 30, 60)

POSITION_FEATURES = [
    "position_age_minutes",
    "entry_price",
    "unrealized_pnl_bps",
    "max_favorable_since_entry_bps",
    "max_adverse_since_entry_bps",
    "profit_giveback_bps",
    "profit_retention_ratio",
    "current_vs_entry_abs_bps",
]
TOPOLOGY_FEATURES = list(v4.STATIC_FEATURES) + list(v4.DYNAMIC_FEATURES)
MATRIX_FEATURES = list(v4.MATRIX_FEATURES) + [
    "matrix_supports_position",
    "matrix_adverse_to_position",
    "matrix_recent_adverse_flip",
    "matrix_topology_agreement",
    "matrix_topology_conflict",
]
FEATURE_GROUPS = {
    "position_only": ["symbol", "timeframe", "side"] + POSITION_FEATURES,
    "position_plus_topology": ["symbol", "timeframe", "side"] + POSITION_FEATURES + TOPOLOGY_FEATURES,
    "position_plus_matrix": ["symbol", "timeframe", "side"] + POSITION_FEATURES + MATRIX_FEATURES,
    "full_position_matrix_topology": (
        ["symbol", "timeframe", "side"]
        + POSITION_FEATURES
        + TOPOLOGY_FEATURES
        + MATRIX_FEATURES
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Position Guardian V5: model HOLD vs EXIT for an already-open position."
    )
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--matrix-len", type=int, default=20)
    p.add_argument("--sample-every-minutes", type=int, default=15)
    p.add_argument("--entry-ages-minutes", default="15,30,60,120,240")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--max-train-rows", type=int, default=500000)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--embargo-hours", type=float, default=4)
    p.add_argument("--hold-improvement-bps", type=float, default=8)
    p.add_argument("--exit-deterioration-bps", type=float, default=8)
    p.add_argument("--future-stop-bps", type=float, default=15)
    p.add_argument("--future-recovery-bps", type=float, default=12)
    p.add_argument("--min-confidence", type=float, default=0.55)
    p.add_argument("--min-margin", type=float, default=0.10)
    p.add_argument("--min-selective-coverage", type=float, default=0.05)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def add_position_context_features(df: pd.DataFrame, side: str) -> pd.DataFrame:
    out = df.copy()
    sign = 1 if side == "LONG" else -1
    out["side"] = side

    trend = pd.to_numeric(out["matrix_trend"], errors="coerce").fillna(0)
    out["matrix_supports_position"] = (trend == sign).astype("int8")
    out["matrix_adverse_to_position"] = (trend == -sign).astype("int8")
    adverse_flip_col = "matrix_short_flip" if side == "LONG" else "matrix_long_flip"
    out["matrix_recent_adverse_flip"] = (
        pd.to_numeric(out[adverse_flip_col], errors="coerce").fillna(0) > 0
    ).astype("int8")

    imbalance = pd.to_numeric(out["topology_imbalance"], errors="coerce").fillna(0)
    topology_sign = np.sign(imbalance)
    out["matrix_topology_agreement"] = (
        (trend != 0) & (topology_sign != 0) & (trend == topology_sign)
    ).astype("int8")
    out["matrix_topology_conflict"] = (
        (trend != 0) & (topology_sign != 0) & (trend == -topology_sign)
    ).astype("int8")
    return out


def build_position_states(
    base: pd.DataFrame,
    side: str,
    entry_ages: list[int],
    horizon_minutes: int,
    a: argparse.Namespace,
) -> pd.DataFrame:
    sign = 1.0 if side == "LONG" else -1.0
    rows = []

    for _, g in base.groupby("symbol", sort=False):
        g = g.sort_values("logged_at").reset_index(drop=True).copy()
        times_ns = g["logged_at"].astype("int64").to_numpy()
        prices = pd.to_numeric(g["current_price"], errors="coerce").to_numpy(float)

        for age in entry_ages:
            age_ns = int(pd.Timedelta(minutes=age).value)
            horizon_ns = int(pd.Timedelta(minutes=horizon_minutes).value)

            for i in range(len(g)):
                entry_target = times_ns[i] - age_ns
                entry_idx = np.searchsorted(times_ns, entry_target, side="right") - 1
                if entry_idx < 0 or entry_idx >= i:
                    continue

                future_end = np.searchsorted(
                    times_ns, times_ns[i] + horizon_ns, side="right"
                )
                if future_end <= i + 1:
                    continue

                entry_price = prices[entry_idx]
                current_price = prices[i]
                if (
                    not np.isfinite(entry_price)
                    or not np.isfinite(current_price)
                    or entry_price <= 0
                    or current_price <= 0
                ):
                    continue

                past_path = sign * (
                    prices[entry_idx : i + 1] / entry_price - 1.0
                ) * 10000.0
                future_path = sign * (
                    prices[i + 1 : future_end] / current_price - 1.0
                ) * 10000.0
                if not np.isfinite(past_path).any() or not np.isfinite(future_path).any():
                    continue

                current_pnl = sign * (current_price / entry_price - 1.0) * 10000.0
                max_fav = float(np.nanmax(past_path))
                max_adv = float(np.nanmin(past_path))
                giveback = max(0.0, max_fav - current_pnl)
                retention = (
                    current_pnl / max_fav
                    if max_fav > 1e-9
                    else (1.0 if current_pnl >= 0 else 0.0)
                )

                future_endpoint = float(future_path[-1])
                future_mfe = float(np.nanmax(future_path))
                future_mae = float(np.nanmin(future_path))

                exit_better = (
                    future_endpoint <= -a.exit_deterioration_bps
                    or (
                        future_mae <= -a.future_stop_bps
                        and future_mfe < a.future_recovery_bps
                    )
                )
                hold_better = (
                    future_endpoint >= a.hold_improvement_bps
                    or (
                        future_mfe >= a.future_recovery_bps
                        and future_mae > -a.future_stop_bps
                    )
                )

                if exit_better and not hold_better:
                    action = 0
                elif hold_better and not exit_better:
                    action = 2
                else:
                    action = 1

                row = g.iloc[i].to_dict()
                row.update(
                    {
                        "side": side,
                        "position_age_minutes": age,
                        "entry_price": entry_price,
                        "unrealized_pnl_bps": current_pnl,
                        "max_favorable_since_entry_bps": max_fav,
                        "max_adverse_since_entry_bps": max_adv,
                        "profit_giveback_bps": giveback,
                        "profit_retention_ratio": retention,
                        "current_vs_entry_abs_bps": abs(current_pnl),
                        "future_incremental_return_bps": future_endpoint,
                        "future_mfe_bps": future_mfe,
                        "future_mae_bps": future_mae,
                        "action_code": action,
                        "action_name": ACTION_NAMES[action],
                        "decision_horizon_minutes": horizon_minutes,
                    }
                )
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["logged_at", "symbol", "position_age_minutes"])


def temporal_split(df: pd.DataFrame, a: argparse.Namespace):
    ts = df["logged_at"]
    test_start = ts.quantile(1 - a.test_fraction)
    before_test = ts[ts < test_start]
    val_start = before_test.quantile(
        1 - a.validation_fraction / (1 - a.test_fraction)
    )
    embargo = pd.Timedelta(hours=a.embargo_hours)

    train = df[df["logged_at"] <= val_start - embargo].copy()
    validation = df[
        (df["logged_at"] >= val_start)
        & (df["logged_at"] <= test_start - embargo)
    ].copy()
    test = df[df["logged_at"] >= test_start].copy()

    if len(train) > a.max_train_rows:
        train = train.sort_values("logged_at").tail(a.max_train_rows).copy()
    return train, validation, test, val_start, test_start


def prepare_xy(df: pd.DataFrame, features: list[str]):
    use = [c for c in features if c in df.columns]
    x = df[use].copy()
    categorical = [
        c for c in ("symbol", "timeframe", "nearest_side", "side") if c in use
    ]
    for c in categorical:
        x[c] = x[c].astype("string").fillna("UNKNOWN")
    for c in use:
        if c not in categorical:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x, categorical, use


def fit_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    a: argparse.Namespace,
):
    xtr, cats, used = prepare_xy(train, features)
    xva, _, _ = prepare_xy(validation, used)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=a.iterations,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=5,
        random_seed=a.random_seed,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    model.fit(
        Pool(xtr, train["action_code"], cat_features=cats),
        eval_set=Pool(xva, validation["action_code"], cat_features=cats),
        early_stopping_rounds=75,
        verbose=False,
    )
    return model, used, cats


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = np.argmax(proba, axis=1)
    return {
        "rows": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "class_counts": {
            ACTION_NAMES[k]: int((y == k).sum()) for k in (0, 1, 2)
        },
    }


def choose_selective_policy(
    validation: pd.DataFrame, proba: np.ndarray, a: argparse.Namespace
) -> dict:
    y = validation["action_code"].to_numpy()
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    sorted_p = np.sort(proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]

    candidates = []
    for conf in np.arange(a.min_confidence, 0.91, 0.05):
        for mar in np.arange(a.min_margin, 0.41, 0.05):
            mask = (confidence >= conf) & (margin >= mar)
            coverage = float(mask.mean())
            if mask.sum() < 30 or coverage < a.min_selective_coverage:
                continue
            ba = balanced_accuracy_score(y[mask], pred[mask])
            mf1 = f1_score(y[mask], pred[mask], average="macro", zero_division=0)
            score = 0.6 * ba + 0.4 * mf1 + 0.05 * np.sqrt(coverage)
            candidates.append(
                {
                    "confidence": float(conf),
                    "margin": float(mar),
                    "rows": int(mask.sum()),
                    "coverage": coverage,
                    "balanced_accuracy": float(ba),
                    "macro_f1": float(mf1),
                    "score": float(score),
                }
            )
    if not candidates:
        return {
            "confidence": a.min_confidence,
            "margin": a.min_margin,
            "fallback": True,
        }
    best = max(candidates, key=lambda r: r["score"])
    best["fallback"] = False
    return best


def selective_metrics(
    test: pd.DataFrame, proba: np.ndarray, policy: dict
) -> dict:
    y = test["action_code"].to_numpy()
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    sorted_p = np.sort(proba, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    mask = (
        (confidence >= policy["confidence"])
        & (margin >= policy["margin"])
    )
    if not mask.any():
        return {"rows": 0, "coverage": 0.0}
    result = metrics(y[mask], proba[mask])
    result["coverage"] = float(mask.mean())
    result["precision_exit_better"] = float(
        ((pred[mask] == 0) & (y[mask] == 0)).sum()
        / max(1, (pred[mask] == 0).sum())
    )
    result["precision_hold_better"] = float(
        ((pred[mask] == 2) & (y[mask] == 2)).sum()
        / max(1, (pred[mask] == 2).sum())
    )
    return result


def main() -> None:
    a = parse_args()
    entry_ages = sorted(
        {int(v.strip()) for v in a.entry_ages_minutes.split(",") if v.strip()}
    )
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading raw Binance OHLCV and computing Koinvizyon Matrix...")
    matrix, parity = v4.load_matrix_all(a)
    print("Loading topology stream...")
    topology = v4.load_topology(a)
    topology = v4.add_dynamic_features(topology)
    base = v4.join_matrix(topology, matrix)

    reports = []
    for side in SIDES:
        contextual = add_position_context_features(base, side)
        for horizon in HORIZONS:
            print(f"{side} {horizon}m position-state dataset...")
            dataset = build_position_states(
                contextual, side, entry_ages, horizon, a
            )
            if dataset.empty:
                raise RuntimeError(f"No rows for {side} {horizon}m")
            train, validation, test, val_start, test_start = temporal_split(dataset, a)

            report = {
                "side": side,
                "horizon_minutes": horizon,
                "entry_ages_minutes": entry_ages,
                "label_definition": {
                    "HOLD_BETTER": (
                        f"future endpoint >= +{a.hold_improvement_bps} bps or "
                        f"MFE >= {a.future_recovery_bps} bps without "
                        f"MAE <= -{a.future_stop_bps} bps"
                    ),
                    "EXIT_BETTER": (
                        f"future endpoint <= -{a.exit_deterioration_bps} bps or "
                        f"MAE <= -{a.future_stop_bps} bps without recovery"
                    ),
                    "NO_CLEAR_EDGE": "both/neither condition",
                },
                "split": {
                    "train_end": train["logged_at"].max(),
                    "validation_start": val_start,
                    "test_start": test_start,
                    "test_end": test["logged_at"].max(),
                },
                "counts": {
                    "dataset": len(dataset),
                    "train": len(train),
                    "validation": len(validation),
                    "test": len(test),
                    "test_classes": test["action_name"].value_counts().to_dict(),
                },
                "ablation": [],
            }

            for group_name, features in FEATURE_GROUPS.items():
                print(f"  {group_name}")
                model, used, cats = fit_model(train, validation, features, a)
                xva, _, _ = prepare_xy(validation, used)
                xte, _, _ = prepare_xy(test, used)
                pva = model.predict_proba(
                    Pool(xva, cat_features=[c for c in cats if c in xva.columns])
                )
                pte = model.predict_proba(
                    Pool(xte, cat_features=[c for c in cats if c in xte.columns])
                )
                policy = choose_selective_policy(validation, pva, a)
                group_report = {
                    "feature_group": group_name,
                    "feature_count": len(used),
                    "validation_metrics": metrics(
                        validation["action_code"].to_numpy(), pva
                    ),
                    "test_metrics": metrics(test["action_code"].to_numpy(), pte),
                    "selected_policy": policy,
                    "selective_test_metrics": selective_metrics(test, pte, policy),
                }
                model_dir = MODEL_OUT / f"{side.lower()}_{horizon}m"
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(model_dir / f"{group_name}.cbm")
                group_report["model_path"] = str(
                    model_dir / f"{group_name}.cbm"
                )
                report["ablation"].append(group_report)

            report["best_feature_group_by_test_macro_f1_research_only"] = max(
                report["ablation"],
                key=lambda r: r["test_metrics"]["macro_f1"],
            )["feature_group"]

            report_path = OUT / f"{side.lower()}_{horizon}m_report.json"
            report_path.write_text(
                json.dumps(json_safe(report), indent=2), encoding="utf-8"
            )
            reports.append(report)

    summary = {
        "status": "research_only",
        "question": "For an already-open position, is HOLD or EXIT better from now?",
        "action_codes": ACTION_NAMES,
        "matrix_parity": parity,
        "entry_ages_minutes": entry_ages,
        "feature_groups": {k: len(v) for k, v in FEATURE_GROUPS.items()},
        "reports": reports,
    }
    (OUT / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    print(f"Done: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
