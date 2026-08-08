from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "topology_direction_v1"
MODEL_PATH = MODEL_DIR / "model.cbm"
FEATURES_PATH = MODEL_DIR / "features.json"


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce")
    num = pd.to_numeric(num, errors="coerce")
    return num / den.replace(0.0, np.nan)


def _ensure_live_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill direction-model features that can be derived causally from a live topology row."""
    if frame.empty:
        return frame.copy()

    out = frame.copy()

    numeric_candidates = [
        "upper_distance_pct",
        "lower_distance_pct",
        "upper_pool_volume",
        "lower_pool_volume",
        "nearest_pool_volume",
        "farther_pool_volume",
        "topology_imbalance",
        "upper_active_levels",
        "lower_active_levels",
        "active_level_difference",
        "active_level_total",
        "distance_advantage",
        "signed_distance_edge",
        "pool_volume_ratio",
        "distance_pressure_ratio",
    ]
    for col in numeric_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    upper_distance = out.get("upper_distance_pct")
    lower_distance = out.get("lower_distance_pct")
    upper_volume = out.get("upper_pool_volume")
    lower_volume = out.get("lower_pool_volume")

    if "distance_advantage" not in out.columns and upper_distance is not None and lower_distance is not None:
        out["distance_advantage"] = lower_distance - upper_distance

    if "signed_distance_edge" not in out.columns and upper_distance is not None and lower_distance is not None:
        total = upper_distance + lower_distance
        out["signed_distance_edge"] = _safe_div(lower_distance - upper_distance, total)

    if "pool_volume_ratio" not in out.columns and upper_volume is not None and lower_volume is not None:
        out["pool_volume_ratio"] = _safe_div(upper_volume, lower_volume)

    if "distance_pressure_ratio" not in out.columns and all(
        x is not None for x in (upper_distance, lower_distance, upper_volume, lower_volume)
    ):
        eps = 1e-9
        upper_pressure = upper_volume / upper_distance.clip(lower=eps)
        lower_pressure = lower_volume / lower_distance.clip(lower=eps)
        out["distance_pressure_ratio"] = _safe_div(upper_pressure, lower_pressure)

    if "active_level_difference" not in out.columns:
        if "upper_active_levels" in out.columns and "lower_active_levels" in out.columns:
            out["active_level_difference"] = (
                pd.to_numeric(out["upper_active_levels"], errors="coerce")
                - pd.to_numeric(out["lower_active_levels"], errors="coerce")
            )

    if "active_level_total" not in out.columns:
        if "upper_active_levels" in out.columns and "lower_active_levels" in out.columns:
            out["active_level_total"] = (
                pd.to_numeric(out["upper_active_levels"], errors="coerce")
                + pd.to_numeric(out["lower_active_levels"], errors="coerce")
            )

    if "logged_at" in out.columns:
        ts = pd.to_datetime(out["logged_at"], utc=True, errors="coerce")
        hour = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
        dow = ts.dt.dayofweek.astype(float)
        if "hour_sin" not in out.columns:
            out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        if "hour_cos" not in out.columns:
            out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        if "dow_sin" not in out.columns:
            out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        if "dow_cos" not in out.columns:
            out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    return out


class TopologyDirectionEngine:
    def __init__(self) -> None:
        self._lock = Lock()
        self._loaded = False
        self._error: str | None = None
        self.model: CatBoostClassifier | None = None
        self.features: list[str] = []
        self.categorical_features: list[str] = []
        self.positive_class = "UPPER_FIRST"
        self.negative_class = "LOWER_FIRST"

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(f"Direction model not found: {MODEL_PATH}")
                if not FEATURES_PATH.exists():
                    raise FileNotFoundError(f"Direction feature manifest not found: {FEATURES_PATH}")

                manifest = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
                features = list(manifest["features"])
                categorical = list(manifest.get("categorical_features", []))

                model = CatBoostClassifier()
                model.load_model(str(MODEL_PATH))

                self.model = model
                self.features = features
                self.categorical_features = categorical
                self.positive_class = str(manifest.get("positive_class", "UPPER_FIRST"))
                self.negative_class = str(manifest.get("negative_class", "LOWER_FIRST"))
                self._error = None
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
            finally:
                self._loaded = True

    @property
    def available(self) -> bool:
        self._load()
        return self.model is not None and bool(self.features) and not self._error

    @property
    def error(self) -> str | None:
        self._load()
        return self._error

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = _ensure_live_features(frame)
        missing = [c for c in self.features if c not in enriched.columns]
        if missing:
            raise KeyError(f"Missing live direction features: {missing}")

        x = enriched[self.features].copy()
        for col in self.features:
            if col in self.categorical_features:
                x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
            else:
                x[col] = pd.to_numeric(x[col], errors="coerce")
        return x

    def score_latest(self, frame: pd.DataFrame, symbol: str | None = None) -> dict[str, Any]:
        requested = str(symbol).upper() if symbol is not None else None

        if not self.available:
            return {
                "available": False,
                "reason": "MODEL_UNAVAILABLE",
                "symbol": requested,
                "horizon_minutes": 60,
                "error": self.error,
            }

        if frame.empty:
            return {
                "available": False,
                "reason": "NO_LIVE_DATA",
                "symbol": requested,
                "horizon_minutes": 60,
            }

        g = frame.copy()
        if requested is not None:
            if "symbol" not in g.columns:
                return {
                    "available": False,
                    "reason": "MISSING_SYMBOL_COLUMN",
                    "symbol": requested,
                    "horizon_minutes": 60,
                }
            g = g[g["symbol"].astype(str).str.upper() == requested]

        if g.empty:
            return {
                "available": False,
                "reason": "NO_SYMBOL_DATA",
                "symbol": requested,
                "horizon_minutes": 60,
            }

        if "logged_at" in g.columns:
            g = g.sort_values("logged_at")
        row = g.tail(1)

        try:
            x = self._prepare(row)
            cat_names = [c for c in self.categorical_features if c in x.columns]
            assert self.model is not None
            prob_up = float(
                self.model.predict_proba(
                    Pool(x, cat_features=cat_names)
                )[0, 1]
            )
        except Exception as exc:
            return {
                "available": False,
                "reason": "INFERENCE_ERROR",
                "symbol": requested,
                "horizon_minutes": 60,
                "error": f"{type(exc).__name__}: {exc}",
            }

        prob_up = float(np.clip(prob_up, 0.0, 1.0))
        prob_down = 1.0 - prob_up
        is_upper = prob_up >= 0.5
        prediction = self.positive_class if is_upper else self.negative_class
        confidence = prob_up if is_upper else prob_down

        latest_at = None
        if "logged_at" in row.columns:
            ts = pd.to_datetime(row.iloc[0]["logged_at"], utc=True, errors="coerce")
            latest_at = ts.isoformat() if pd.notna(ts) else None

        live_symbol = requested
        if live_symbol is None and "symbol" in row.columns:
            live_symbol = str(row.iloc[0]["symbol"]).upper()

        return {
            "available": True,
            "symbol": live_symbol,
            "horizon_minutes": 60,
            "prediction": prediction,
            "confidence": round(confidence, 6),
            "confidence_pct": round(confidence * 100.0, 1),
            "probability_upper_first": round(prob_up, 6),
            "probability_lower_first": round(prob_down, 6),
            "model_version": "topology_direction_v1",
            "research_preview": True,
            "as_of": latest_at,
        }


topology_direction_engine = TopologyDirectionEngine()
