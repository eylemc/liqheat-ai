#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_position_guardian_v5_position_state as v5  # noqa: E402

OUT = Path("data/reports/position_guardian_v6_dual_head")
MODEL_OUT = Path("data/models/position_guardian_v6_dual_head")

SIDES = ("LONG", "SHORT")
HORIZONS = (15, 30, 60)

COMPACT_MATRIX_FEATURES = [
    "matrix_supports_position",
    "matrix_adverse_to_position",
    "matrix_bars_since_flip",
]

BASE_FEATURES = list(
    dict.fromkeys(
        ["symbol", "timeframe", "side"]
        + list(v5.POSITION_FEATURES)
        + list(v5.TOPOLOGY_FEATURES)
    )
)

FEATURE_GROUPS = {
    "position_plus_topology": BASE_FEATURES,
    "position_plus_topology_compact_matrix": list(
        dict.fromkeys(BASE_FEATURES + COMPACT_MATRIX_FEATURES)
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Position Guardian V6: dual binary EXIT and HOLD heads."
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
    p.add_argument("--min-precision", type=float, default=0.45)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--min-alerts", type=int, default=100)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def safe_metric(fn, *args, default=None, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return default


def fit_binary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target_col: str,
    a: argparse.Namespace,
):
    features = list(dict.fromkeys(features))
    xtr, cats, used = v5.prepare_xy(train, features)
    xva, _, _ = v5.prepare_xy(validation, used)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=a.iterations,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=5,
        random_seed=a.random_seed,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
        task_type="GPU",
        devices="0",
        bootstrap_type="Bayesian",
    )
    model.fit(
        Pool(xtr, train[target_col], cat_features=cats),
        eval_set=Pool(xva, validation[target_col], cat_features=cats),
        early_stopping_rounds=75,
        verbose=False,
    )
    return model, used, cats


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (p >= threshold).astype(np.int8)
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "accuracy": safe_metric(accuracy_score, y, pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y, pred),
        "precision": safe_metric(precision_score, y, pred, zero_division=0),
        "recall": safe_metric(recall_score, y, pred, zero_division=0),
        "f1": safe_metric(f1_score, y, pred, zero_division=0),
        "roc_auc": safe_metric(roc_auc_score, y, p),
        "pr_auc": safe_metric(average_precision_score, y, p),
        "brier": safe_metric(brier_score_loss, y, p),
        "log_loss": safe_metric(log_loss, y, p, labels=[0, 1]),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray, a: argparse.Namespace) -> tuple[dict, list[dict]]:
    rows = []
    for threshold in np.arange(0.35, 0.951, 0.025):
        pred = p >= threshold
        alerts = int(pred.sum())
        coverage = float(pred.mean())
        precision = float(precision_score(y, pred, zero_division=0)) if alerts else 0.0
        recall = float(recall_score(y, pred, zero_division=0)) if alerts else 0.0
        f1 = float(f1_score(y, pred, zero_division=0)) if alerts else 0.0
        passes = (
            alerts >= a.min_alerts
            and coverage >= a.min_coverage
            and precision >= a.min_precision
        )
        score = precision * np.sqrt(max(coverage, 1e-12)) + 0.15 * recall
        rows.append(
            {
                "threshold": float(threshold),
                "alerts": alerts,
                "coverage": coverage,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "passes_constraints": bool(passes),
                "score": float(score),
            }
        )

    valid = [r for r in rows if r["passes_constraints"]]
    if valid:
        best = max(valid, key=lambda r: (r["score"], r["precision"], r["coverage"]))
        best = {**best, "fallback": False}
    else:
        eligible = [
            r
            for r in rows
            if r["alerts"] >= max(30, min(a.min_alerts, 100))
            and r["coverage"] >= min(a.min_coverage, 0.01)
        ]
        best = max(eligible or rows, key=lambda r: (r["precision"], r["f1"], r["coverage"]))
        best = {**best, "fallback": True}
    return best, rows


def combined_decisions(
    y_action: np.ndarray,
    exit_p: np.ndarray,
    hold_p: np.ndarray,
    exit_threshold: float,
    hold_threshold: float,
) -> dict:
    exit_on = exit_p >= exit_threshold
    hold_on = hold_p >= hold_threshold

    decision = np.full(len(y_action), 1, dtype=np.int8)
    decision[exit_on & ~hold_on] = 0
    decision[hold_on & ~exit_on] = 2
    conflict = exit_on & hold_on

    actionable = decision != 1
    true_actionable = np.isin(y_action, [0, 2])
    exit_mask = decision == 0
    hold_mask = decision == 2

    return {
        "rows": int(len(y_action)),
        "decision_counts": {
            "EXIT": int(exit_mask.sum()),
            "NO_CLEAR_EDGE": int((decision == 1).sum()),
            "HOLD": int(hold_mask.sum()),
            "CONFLICT": int(conflict.sum()),
        },
        "actionable_coverage": float(actionable.mean()),
        "conflict_rate": float(conflict.mean()),
        "actionable_accuracy": (
            float(np.mean(decision[actionable] == y_action[actionable]))
            if actionable.any()
            else None
        ),
        "precision_exit": float(np.mean(y_action[exit_mask] == 0)) if exit_mask.any() else None,
        "precision_hold": float(np.mean(y_action[hold_mask] == 2)) if hold_mask.any() else None,
        "recall_exit": float(np.sum(exit_mask & (y_action == 0)) / max(1, np.sum(y_action == 0))),
        "recall_hold": float(np.sum(hold_mask & (y_action == 2)) / max(1, np.sum(y_action == 2))),
        "capture_of_true_actionable": float(
            np.sum(actionable & true_actionable) / max(1, true_actionable.sum())
        ),
        "three_class_balanced_accuracy": safe_metric(
            balanced_accuracy_score, y_action, decision
        ),
        "three_class_macro_f1": safe_metric(
            f1_score, y_action, decision, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_action, decision, labels=[0, 1, 2]).tolist(),
    }


def main() -> None:
    a = parse_args()
    entry_ages = sorted(
        {int(v.strip()) for v in a.entry_ages_minutes.split(",") if v.strip()}
    )
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading raw Binance OHLCV and computing Koinvizyon Matrix...")
    matrix, parity = v5.v4.load_matrix_all(a)
    print("Loading topology stream...")
    topology = v5.v4.load_topology(a)
    topology = v5.v4.add_dynamic_features(topology)
    base = v5.v4.join_matrix(topology, matrix)

    reports = []

    for side in SIDES:
        contextual = v5.add_position_context_features(base, side)

        for horizon in HORIZONS:
            print(f"{side} {horizon}m dual-head dataset...")
            dataset = v5.build_position_states(contextual, side, entry_ages, horizon, a)
            if dataset.empty:
                raise RuntimeError(f"No rows for {side} {horizon}m")

            dataset["exit_target"] = (dataset["action_code"] == 0).astype("int8")
            dataset["hold_target"] = (dataset["action_code"] == 2).astype("int8")

            train, validation, test, val_start, test_start = v5.temporal_split(dataset, a)

            report = {
                "side": side,
                "horizon_minutes": horizon,
                "entry_ages_minutes": entry_ages,
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
                    "validation_exit_rate": validation["exit_target"].mean(),
                    "validation_hold_rate": validation["hold_target"].mean(),
                    "test_exit_rate": test["exit_target"].mean(),
                    "test_hold_rate": test["hold_target"].mean(),
                },
                "feature_groups": [],
            }

            for group_name, features in FEATURE_GROUPS.items():
                print(f"  {group_name}")

                exit_model, used, cats = fit_binary(
                    train, validation, features, "exit_target", a
                )
                hold_model, hold_used, hold_cats = fit_binary(
                    train, validation, features, "hold_target", a
                )

                xva, _, _ = v5.prepare_xy(validation, used)
                xte, _, _ = v5.prepare_xy(test, used)
                hxva, _, _ = v5.prepare_xy(validation, hold_used)
                hxte, _, _ = v5.prepare_xy(test, hold_used)

                exit_va = exit_model.predict_proba(
                    Pool(xva, cat_features=[c for c in cats if c in xva.columns])
                )[:, 1]
                exit_te = exit_model.predict_proba(
                    Pool(xte, cat_features=[c for c in cats if c in xte.columns])
                )[:, 1]
                hold_va = hold_model.predict_proba(
                    Pool(hxva, cat_features=[c for c in hold_cats if c in hxva.columns])
                )[:, 1]
                hold_te = hold_model.predict_proba(
                    Pool(hxte, cat_features=[c for c in hold_cats if c in hxte.columns])
                )[:, 1]

                exit_policy, exit_grid = choose_threshold(
                    validation["exit_target"].to_numpy(), exit_va, a
                )
                hold_policy, hold_grid = choose_threshold(
                    validation["hold_target"].to_numpy(), hold_va, a
                )

                group_report = {
                    "feature_group": group_name,
                    "feature_count": len(used),
                    "exit_head": {
                        "validation_default": binary_metrics(
                            validation["exit_target"].to_numpy(), exit_va
                        ),
                        "test_default": binary_metrics(
                            test["exit_target"].to_numpy(), exit_te
                        ),
                        "selected_policy": exit_policy,
                        "test_selected": binary_metrics(
                            test["exit_target"].to_numpy(),
                            exit_te,
                            exit_policy["threshold"],
                        ),
                        "validation_grid": exit_grid,
                    },
                    "hold_head": {
                        "validation_default": binary_metrics(
                            validation["hold_target"].to_numpy(), hold_va
                        ),
                        "test_default": binary_metrics(
                            test["hold_target"].to_numpy(), hold_te
                        ),
                        "selected_policy": hold_policy,
                        "test_selected": binary_metrics(
                            test["hold_target"].to_numpy(),
                            hold_te,
                            hold_policy["threshold"],
                        ),
                        "validation_grid": hold_grid,
                    },
                    "combined_test_policy": combined_decisions(
                        test["action_code"].to_numpy(),
                        exit_te,
                        hold_te,
                        exit_policy["threshold"],
                        hold_policy["threshold"],
                    ),
                }

                model_dir = MODEL_OUT / f"{side.lower()}_{horizon}m" / group_name
                model_dir.mkdir(parents=True, exist_ok=True)
                exit_path = model_dir / "exit_head.cbm"
                hold_path = model_dir / "hold_head.cbm"
                exit_model.save_model(exit_path)
                hold_model.save_model(hold_path)
                group_report["model_paths"] = {
                    "exit_head": str(exit_path),
                    "hold_head": str(hold_path),
                }

                report["feature_groups"].append(group_report)

            report["best_group_by_combined_macro_f1_research_only"] = max(
                report["feature_groups"],
                key=lambda r: (
                    r["combined_test_policy"]["three_class_macro_f1"]
                    if r["combined_test_policy"]["three_class_macro_f1"] is not None
                    else -1
                ),
            )["feature_group"]

            report_path = OUT / f"{side.lower()}_{horizon}m_report.json"
            report_path.write_text(
                json.dumps(v5.json_safe(report), indent=2), encoding="utf-8"
            )
            reports.append(report)

    summary = {
        "status": "research_only",
        "architecture": "binary_dual_head",
        "heads": {
            "exit": "EXIT_BETTER vs NOT_EXIT",
            "hold": "HOLD_BETTER vs NOT_HOLD",
        },
        "decision_rule": {
            "EXIT": "exit head above threshold and hold head below threshold",
            "HOLD": "hold head above threshold and exit head below threshold",
            "CONFLICT": "both heads above threshold",
            "NO_CLEAR_EDGE": "neither head above threshold",
        },
        "matrix_parity": parity,
        "entry_ages_minutes": entry_ages,
        "feature_groups": {k: len(v) for k, v in FEATURE_GROUPS.items()},
        "reports": reports,
    }
    (OUT / "summary.json").write_text(
        json.dumps(v5.json_safe(summary), indent=2), encoding="utf-8"
    )
    print(f"Done: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
