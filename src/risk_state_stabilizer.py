from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = PROJECT_ROOT / "data" / "runtime" / "risk_state_stabilizer_v1.json"

BANDS = ["LOW RISK", "MEDIUM RISK", "HIGH RISK", "EXTREME RISK"]
BAND_INDEX = {name: i for i, name in enumerate(BANDS)}


class RiskStateStabilizer:
    """Stateful display stabilizer for the live 15m Market Risk Radar.

    The ML model remains untouched. This layer only converts the raw percentile
    stream into a temporally stable product state using EMA smoothing,
    hysteresis and confirmation counts.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.20,
        upward_confirmations: int = 2,
        downward_confirmations: int = 3,
    ) -> None:
        self.alpha = float(alpha)
        self.upward_confirmations = int(upward_confirmations)
        self.downward_confirmations = int(downward_confirmations)
        self._lock = Lock()
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _initial_band(score: float) -> str:
        if score >= 90.0:
            return "EXTREME RISK"
        if score >= 75.0:
            return "HIGH RISK"
        if score >= 50.0:
            return "MEDIUM RISK"
        return "LOW RISK"

    @staticmethod
    def _adjacent_candidate(current: str, score: float) -> str:
        # Hysteresis thresholds. Only one band step is allowed per confirmed
        # transition, so HIGH can never collapse directly to LOW.
        if current == "LOW RISK":
            return "MEDIUM RISK" if score >= 55.0 else current
        if current == "MEDIUM RISK":
            if score >= 78.0:
                return "HIGH RISK"
            if score <= 45.0:
                return "LOW RISK"
            return current
        if current == "HIGH RISK":
            if score >= 92.0:
                return "EXTREME RISK"
            if score <= 68.0:
                return "MEDIUM RISK"
            return current
        if current == "EXTREME RISK":
            return "HIGH RISK" if score <= 84.0 else current
        return RiskStateStabilizer._initial_band(score)

    def _load(self) -> None:
        try:
            if STATE_PATH.exists():
                payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self._state = payload
        except Exception:
            self._state = {}

    def _save(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, STATE_PATH)

    def update(self, risk: dict[str, Any]) -> dict[str, Any]:
        if not risk.get("available"):
            return risk

        symbol = str(risk.get("symbol") or "UNKNOWN").upper()
        raw_score = float(risk.get("risk_score", 0.0))
        as_of = risk.get("as_of")

        with self._lock:
            previous = self._state.get(symbol)

            # Repeated API refreshes of the same underlying topology snapshot
            # must not count as independent confirmations.
            if previous and as_of and previous.get("last_as_of") == as_of:
                result = dict(risk)
                result.update({
                    "raw_risk_score": round(raw_score, 2),
                    "risk_score": int(round(float(previous["smoothed_score"]))),
                    "risk_band": str(previous["band"]),
                    "stabilized": True,
                    "stabilizer": self._diagnostics(previous),
                })
                return result

            if previous is None:
                smoothed = raw_score
                band = self._initial_band(smoothed)
                state = {
                    "smoothed_score": smoothed,
                    "band": band,
                    "pending_band": None,
                    "pending_count": 0,
                    "last_as_of": as_of,
                }
            else:
                old_smoothed = float(previous.get("smoothed_score", raw_score))
                smoothed = self.alpha * raw_score + (1.0 - self.alpha) * old_smoothed
                band = str(previous.get("band") or self._initial_band(old_smoothed))
                candidate = self._adjacent_candidate(band, smoothed)

                pending_band = previous.get("pending_band")
                pending_count = int(previous.get("pending_count", 0))

                if candidate == band:
                    pending_band = None
                    pending_count = 0
                else:
                    if pending_band == candidate:
                        pending_count += 1
                    else:
                        pending_band = candidate
                        pending_count = 1

                    moving_up = BAND_INDEX.get(candidate, 0) > BAND_INDEX.get(band, 0)
                    needed = self.upward_confirmations if moving_up else self.downward_confirmations

                    if pending_count >= needed:
                        band = candidate
                        pending_band = None
                        pending_count = 0

                state = {
                    "smoothed_score": smoothed,
                    "band": band,
                    "pending_band": pending_band,
                    "pending_count": pending_count,
                    "last_as_of": as_of,
                }

            self._state[symbol] = state
            self._save()

            result = dict(risk)
            result.update({
                "raw_risk_score": round(raw_score, 2),
                "risk_score": int(round(float(state["smoothed_score"]))),
                "risk_band": str(state["band"]),
                "stabilized": True,
                "stabilizer": self._diagnostics(state),
            })
            return result

    def _diagnostics(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": "risk-state-stabilizer-v1",
            "ema_alpha": self.alpha,
            "smoothed_score": round(float(state.get("smoothed_score", 0.0)), 2),
            "display_band": state.get("band"),
            "pending_band": state.get("pending_band"),
            "pending_count": int(state.get("pending_count", 0)),
            "upward_confirmations": self.upward_confirmations,
            "downward_confirmations": self.downward_confirmations,
            "hysteresis": {
                "low_to_medium": 55,
                "medium_to_low": 45,
                "medium_to_high": 78,
                "high_to_medium": 68,
                "high_to_extreme": 92,
                "extreme_to_high": 84,
            },
        }


risk_state_stabilizer = RiskStateStabilizer()
