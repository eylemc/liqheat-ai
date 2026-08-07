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
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "market_risk_radar_v2_live"

SUPPORTED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _apply_calibrator(p: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    method = str(payload.get("method", "raw"))
    if method == "raw":
        return p
    if method == "platt":
        coef = float(payload["coef"])
        intercept = float(payload["intercept"])
        z = np.log(p / (1.0 - p))
        logits = coef * z + intercept
        return np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-6, 1 - 1e-6)
    if method == "isotonic":
        x = np.asarray(payload["x_thresholds"], dtype=float)
        y = np.asarray(payload["y_thresholds"], dtype=float)
        return np.clip(np.interp(p, x, y, left=y[0], right=y[-1]), 1e-6, 1 - 1e-6)
    raise ValueError(f"Unknown calibrator method: {method}")


def _band(score: float) -> str:
    if score >= 75.0:
        return "EXTREME RISK"
    if score >= 50.0:
        return "HIGH RISK"
    if score >= 25.0:
        return "MEDIUM RISK"
    return "LOW RISK"


def add_dynamic_features_live(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the V2 15-minute dynamic feature construction on live topology history."""
    if frame.empty:
        return frame

    pieces = []
    for _, g in frame.groupby("symbol", sort=False):
        g = g.sort_values("logged_at").copy()
        price = pd.to_numeric(g["current_price"], errors="coerce")
        for mins in (15, 30, 60):
            periods = max(1, mins // 15)
            g[f"price_return_{mins}m"] = price.pct_change(periods) * 10000.0
        for mins in (15, 30):
            periods = max(1, mins // 15)
            for col, prefix in [
                ("upper_distance_pct", "upper_distance"),
                ("lower_distance_pct", "lower_distance"),
                ("topology_imbalance", "imbalance"),
                ("upper_pool_volume", "upper_pool"),
                ("lower_pool_volume", "lower_pool"),
            ]:
                s = pd.to_numeric(g[col], errors="coerce")
                g[f"{prefix}_change_{mins}m"] = s - s.shift(periods)
            side = g["nearest_side"].astype("string")
            g[f"nearest_side_flip_{mins}m"] = side.ne(side.shift(periods)).fillna(False).astype("int8")
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def sample_live_history(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 15-minute history compatible with research sampling while retaining
    the freshest row of the current bucket for genuinely live scoring.
    """
    if frame.empty:
        return frame
    out = []
    for _, g in frame.groupby("symbol", sort=False):
        g = g.sort_values("logged_at").copy()
        g["_bucket"] = pd.to_datetime(g["logged_at"], utc=True).dt.floor("15min")
        buckets = list(g.groupby("_bucket", sort=True))
        selected = []
        for i, (_, b) in enumerate(buckets):
            selected.append(b.iloc[[-1]] if i == len(buckets) - 1 else b.iloc[[0]])
        if selected:
            out.append(pd.concat(selected, ignore_index=True))
    if not out:
        return frame.iloc[0:0].copy()
    return pd.concat(out, ignore_index=True).drop(columns=["_bucket"], errors="ignore")


class MarketRiskEngine:
    def __init__(self) -> None:
        self._lock = Lock()
        self._loaded = False
        self._error: str | None = None
        self.manifest: dict[str, Any] | None = None
        self.models: dict[str, CatBoostClassifier] = {}
        self.calibrators: dict[str, dict[str, Any]] = {}

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                manifest_path = MODEL_DIR / "manifest.json"
                if not manifest_path.exists():
                    raise FileNotFoundError(f"Live V2 manifest not found: {manifest_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                models: dict[str, CatBoostClassifier] = {}
                calibrators: dict[str, dict[str, Any]] = {}
                for head in ("high", "extreme"):
                    cfg = manifest["heads"][head]
                    model = CatBoostClassifier()
                    model.load_model(str(MODEL_DIR / cfg["model"]))
                    models[head] = model
                    calibrators[head] = json.loads((MODEL_DIR / cfg["calibrator"]).read_text(encoding="utf-8"))
                self.manifest = manifest
                self.models = models
                self.calibrators = calibrators
                self._error = None
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
            finally:
                self._loaded = True

    @property
    def available(self) -> bool:
        self._load()
        return self.manifest is not None and not self._error

    @property
    def error(self) -> str | None:
        self._load()
        return self._error

    def _prepare(self, frame: pd.DataFrame, features: list[str], cats: list[str]) -> pd.DataFrame:
        x = frame[features].copy()
        for col in features:
            if col in cats:
                x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
            else:
                x[col] = pd.to_numeric(x[col], errors="coerce")
        return x

    def score_latest(self, frame: pd.DataFrame, symbol: str) -> dict[str, Any]:
        requested = str(symbol).upper()
        if requested not in SUPPORTED_SYMBOLS:
            return {
                "available": False,
                "reason": "SYMBOL_NOT_TRAINED",
                "symbol": requested,
                "horizon_minutes": 15,
            }
        if not self.available:
            return {
                "available": False,
                "reason": "MODEL_UNAVAILABLE",
                "symbol": requested,
                "horizon_minutes": 15,
                "error": self.error,
            }
        if frame.empty:
            return {"available": False, "reason": "NO_LIVE_HISTORY", "symbol": requested, "horizon_minutes": 15}

        g = frame[frame["symbol"].astype(str).str.upper() == requested].sort_values("logged_at")
        if g.empty:
            return {"available": False, "reason": "NO_SYMBOL_HISTORY", "symbol": requested, "horizon_minutes": 15}
        row = g.tail(1)

        probabilities: dict[str, float] = {}
        details: dict[str, Any] = {}
        for head in ("high", "extreme"):
            cfg = self.manifest["heads"][head]
            features = list(cfg["features"])
            cats = list(cfg["categorical_features"])
            missing = [c for c in features if c not in row.columns]
            if missing:
                return {
                    "available": False,
                    "reason": "MISSING_FEATURES",
                    "symbol": requested,
                    "horizon_minutes": 15,
                    "missing_features": missing,
                }
            x = self._prepare(row, features, cats)
            raw = float(self.models[head].predict_proba(Pool(x, cat_features=[c for c in cats if c in x.columns]))[0, 1])
            calibrated = float(_apply_calibrator(np.array([raw]), self.calibrators[head])[0])
            probabilities[head] = calibrated
            details[head] = {
                "raw_probability": round(raw, 6),
                "calibrated_probability": round(calibrated, 6),
                "feature_group": cfg["feature_group"],
                "calibration_method": cfg["calibration_method"],
            }

        score = float(np.clip(100.0 * (0.40 * probabilities["high"] + 0.60 * probabilities["extreme"]), 0.0, 100.0))
        latest_at = pd.to_datetime(row.iloc[0]["logged_at"], utc=True, errors="coerce")
        return {
            "available": True,
            "symbol": requested,
            "horizon_minutes": 15,
            "risk_score": round(score, 2),
            "risk_band": _band(score),
            "p_high": round(probabilities["high"], 6),
            "p_extreme": round(probabilities["extreme"], 6),
            "as_of": latest_at.isoformat() if pd.notna(latest_at) else None,
            "matrix_used_by_risk_model": False,
            "details": details,
        }


market_risk_engine = MarketRiskEngine()
