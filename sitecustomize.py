"""Runtime backend patch for LiqHeat Radar score stabilization.

Python imports ``sitecustomize`` automatically during normal interpreter startup.
This module wraps ``src.matrix_live.combine_matrix_topology`` before the FastAPI
module imports it, keeping the raw score for research while exposing a calmer,
stateful score to the API/UI.
"""

from __future__ import annotations

from threading import Lock
from typing import Any


_SCORE_RISE_ALPHA = 0.35
_SCORE_FALL_ALPHA = 0.18
_score_lock = Lock()
_score_state: dict[str, dict[str, Any]] = {}


def _classify_opportunity(
    *,
    score: float,
    agrees: bool,
    conflicts: bool,
    alignment: float,
) -> str:
    if agrees and score >= 90.0 and alignment >= 75.0:
        return "CRITICAL"
    if agrees and score >= 80.0 and alignment >= 60.0:
        return "HIGH"
    if agrees and score >= 68.0:
        return "WATCH"
    if conflicts:
        return "CONFLICT"
    return "NORMAL"


def _install_patch() -> None:
    try:
        from src import matrix_live
    except Exception:
        # Never prevent the application from starting because an optional
        # runtime patch could not be installed.
        return

    original = matrix_live.combine_matrix_topology

    if getattr(original, "_liqheat_score_stability", False):
        return

    def stabilized_combine_matrix_topology(
        liquidity_pressure: float,
        topology_direction: int,
        matrix: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result = original(
            liquidity_pressure=liquidity_pressure,
            topology_direction=topology_direction,
            matrix=matrix,
        )

        raw_score = float(result.get("radar_score", 0.0))
        symbol = str((matrix or {}).get("symbol") or "UNKNOWN").upper()
        gate = str(result.get("matrix_gate") or "UNAVAILABLE").upper()
        matrix_direction = int((matrix or {}).get("direction") or 0)
        alignment = float((matrix or {}).get("alignment_score") or 0.0)
        agrees = result.get("matrix_agreement") is True
        conflicts = result.get("matrix_agreement") is False
        regime_key = (gate, matrix_direction)

        with _score_lock:
            previous = _score_state.get(symbol)

            if previous is None or previous.get("regime_key") != regime_key:
                stabilized_score = raw_score
                reset_reason = "INITIAL" if previous is None else "REGIME_CHANGE"
            else:
                previous_score = float(previous["stabilized_score"])
                alpha = (
                    _SCORE_RISE_ALPHA
                    if raw_score > previous_score
                    else _SCORE_FALL_ALPHA
                )
                stabilized_score = previous_score + alpha * (
                    raw_score - previous_score
                )
                reset_reason = None

            stabilized_score = max(0.0, min(100.0, stabilized_score))
            stabilized_score = round(stabilized_score, 2)

            _score_state[symbol] = {
                "raw_score": raw_score,
                "stabilized_score": stabilized_score,
                "regime_key": regime_key,
            }

        result["raw_radar_score"] = round(raw_score, 2)
        result["radar_score"] = stabilized_score
        result["opportunity"] = _classify_opportunity(
            score=stabilized_score,
            agrees=agrees,
            conflicts=conflicts,
            alignment=alignment,
        )
        result["score_stability"] = {
            "mode": "asymmetric_ema_v1",
            "rise_alpha": _SCORE_RISE_ALPHA,
            "fall_alpha": _SCORE_FALL_ALPHA,
            "raw_score": round(raw_score, 2),
            "stabilized_score": stabilized_score,
            "regime_key": {
                "matrix_gate": gate,
                "matrix_direction": matrix_direction,
            },
            "reset_reason": reset_reason,
        }

        return result

    stabilized_combine_matrix_topology._liqheat_score_stability = True
    matrix_live.combine_matrix_topology = stabilized_combine_matrix_topology


_install_patch()
