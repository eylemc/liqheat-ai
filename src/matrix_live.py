from __future__ import annotations

import math
import time
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import requests


BINANCE_BASE_URL = "https://fapi.binance.com"
MATRIX_LENGTH = 20

MATRIX_TIMEFRAMES = [
    "1d",
    "4h",
    "1h",
    "15m",
    "1m",
]

# Daily ana yön; alt timeframe'ler confirmation sağlar.
TIMEFRAME_WEIGHTS = {
    "1d": 30.0,
    "4h": 25.0,
    "1h": 20.0,
    "15m": 15.0,
    "1m": 10.0,
}

CACHE_SECONDS = 45

_session = requests.Session()
_session.headers.update({
    "User-Agent": "LiqHeat-AI-Radar-Matrix-V2/1.0",
    "Accept": "application/json",
})

_cache_lock = Lock()
_matrix_cache: dict[str, dict[str, Any]] = {}


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _fetch_closed_klines(
    symbol: str,
    timeframe: str,
    limit: int = 120,
) -> pd.DataFrame:
    """
    Binance USD-M Futures'tan yalnız kapanmış mumları döndürür.

    Matrix VWMA(20) + channel(20) için yaklaşık 40 mum gerekir.
    120 mum güvenli bir warm-up sağlar.
    """
    response = _session.get(
        BINANCE_BASE_URL + "/fapi/v1/klines",
        params={
            "symbol": symbol,
            "interval": timeframe,
            "limit": limit,
        },
        timeout=15,
    )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, list):
        raise RuntimeError(
            f"Unexpected Binance response: {payload}"
        )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    frame = pd.DataFrame(
        payload,
        columns=columns,
    )

    if frame.empty:
        return frame

    frame["open_time"] = pd.to_datetime(
        frame["open_time"],
        unit="ms",
        utc=True,
    )

    frame["close_time"] = pd.to_datetime(
        frame["close_time"],
        unit="ms",
        utc=True,
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    now = pd.Timestamp.now(tz="UTC")

    frame = frame[
        frame["close_time"] < now
    ].copy()

    return (
        frame
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def _persistent_trend(
    source: pd.Series,
    previous_upper: pd.Series,
    previous_lower: pd.Series,
) -> pd.Series:
    source_values = source.to_numpy(
        dtype=np.float64
    )

    upper_values = previous_upper.to_numpy(
        dtype=np.float64
    )

    lower_values = previous_lower.to_numpy(
        dtype=np.float64
    )

    output = np.zeros(
        len(source_values),
        dtype=np.int8,
    )

    state = 0

    for index in range(len(source_values)):
        source_value = source_values[index]
        upper_value = upper_values[index]
        lower_value = lower_values[index]

        if (
            np.isfinite(source_value)
            and np.isfinite(upper_value)
            and source_value > upper_value
        ):
            state = 1

        elif (
            np.isfinite(source_value)
            and np.isfinite(lower_value)
            and source_value < lower_value
        ):
            state = -1

        output[index] = state

    return pd.Series(
        output,
        index=source.index,
        dtype="int8",
    )


def _matrix_state(
    candles: pd.DataFrame,
    timeframe: str,
) -> dict[str, Any]:
    frame = candles.copy()

    if len(frame) < 40:
        raise RuntimeError(
            f"Insufficient {timeframe} candles: {len(frame)}"
        )

    frame["source"] = (
        frame["open"]
        + frame["high"]
        + frame["low"]
        + frame["close"]
    ) / 4.0

    weighted_source = (
        frame["source"]
        * frame["volume"]
    )

    frame["vwma"] = (
        weighted_source
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .sum()
        /
        frame["volume"]
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .sum()
        .replace(0, np.nan)
    )

    frame["upper"] = (
        frame["vwma"]
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .max()
    )

    frame["lower"] = (
        frame["vwma"]
        .rolling(
            MATRIX_LENGTH,
            min_periods=MATRIX_LENGTH,
        )
        .min()
    )

    frame["trend"] = _persistent_trend(
        frame["source"],
        frame["upper"].shift(1),
        frame["lower"].shift(1),
    )

    previous_trend = (
        frame["trend"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )

    frame["flip"] = np.select(
        [
            (frame["trend"] == 1)
            & (previous_trend == -1),

            (frame["trend"] == -1)
            & (previous_trend == 1),
        ],
        [
            1,
            -1,
        ],
        default=0,
    ).astype("int8")

    last = frame.iloc[-1]

    flip_positions = np.flatnonzero(
        frame["flip"].to_numpy() != 0
    )

    bars_since_flip = (
        int(len(frame) - 1 - flip_positions[-1])
        if len(flip_positions)
        else None
    )

    trend = int(last["trend"])

    return {
        "timeframe": timeframe,
        "trend": trend,
        "trend_label": (
            "BUY"
            if trend == 1
            else "SELL"
            if trend == -1
            else "NEUTRAL"
        ),
        "flip": int(last["flip"]),
        "bars_since_flip": bars_since_flip,
        "open_time": last["open_time"].isoformat(),
        "close_time": last["close_time"].isoformat(),
        "close": _safe_number(last["close"]),
        "source": _safe_number(last["source"]),
        "vwma": _safe_number(last["vwma"]),
        "upper": _safe_number(last["upper"]),
        "lower": _safe_number(last["lower"]),
        "distance_to_vwma_pct": (
            _safe_number(
                (
                    last["source"]
                    / last["vwma"]
                    - 1.0
                )
                if pd.notna(last["vwma"])
                and float(last["vwma"]) != 0
                else None
            )
        ),
        "channel_width_pct": (
            _safe_number(
                (
                    (
                        last["upper"]
                        - last["lower"]
                    )
                    / last["vwma"]
                )
                if pd.notna(last["vwma"])
                and float(last["vwma"]) != 0
                else None
            )
        ),
    }


def _build_symbol_matrix(
    symbol: str,
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}

    errors: dict[str, str] = {}

    for timeframe in MATRIX_TIMEFRAMES:
        try:
            candles = _fetch_closed_klines(
                symbol,
                timeframe,
            )

            states[timeframe] = _matrix_state(
                candles,
                timeframe,
            )

        except Exception as exc:
            errors[timeframe] = (
                f"{type(exc).__name__}: {exc}"
            )

    daily_state = states.get(
        "1d",
        {},
    ).get(
        "trend",
        0,
    )

    valid_daily = daily_state in {
        -1,
        1,
    }

    aligned_weight = 0.0
    available_weight = 0.0

    aligned_timeframes: list[str] = []
    opposing_timeframes: list[str] = []

    for timeframe in MATRIX_TIMEFRAMES:
        state = states.get(
            timeframe,
            {},
        ).get(
            "trend",
            0,
        )

        if state not in {-1, 1}:
            continue

        weight = TIMEFRAME_WEIGHTS[
            timeframe
        ]

        available_weight += weight

        if valid_daily and state == daily_state:
            aligned_weight += weight
            aligned_timeframes.append(
                timeframe
            )

        elif valid_daily:
            opposing_timeframes.append(
                timeframe
            )

    alignment_score = (
        100.0
        * aligned_weight
        / available_weight
        if available_weight > 0
        and valid_daily
        else 0.0
    )

    upper_core_aligned = (
        states.get("1d", {}).get("trend")
        in {-1, 1}
        and states.get("4h", {}).get("trend")
        == daily_state
        and states.get("1h", {}).get("trend")
        == daily_state
    )

    full_alignment = (
        valid_daily
        and all(
            states.get(
                timeframe,
                {},
            ).get(
                "trend"
            ) == daily_state
            for timeframe
            in MATRIX_TIMEFRAMES
        )
    )

    if full_alignment:
        regime = (
            "FULL_LONG_ALIGNMENT"
            if daily_state == 1
            else "FULL_SHORT_ALIGNMENT"
        )

    elif upper_core_aligned:
        regime = (
            "CORE_LONG_ALIGNMENT"
            if daily_state == 1
            else "CORE_SHORT_ALIGNMENT"
        )

    elif valid_daily:
        regime = (
            "LONG_REGIME_MIXED"
            if daily_state == 1
            else "SHORT_REGIME_MIXED"
        )

    else:
        regime = "UNAVAILABLE"

    return {
        "symbol": symbol,
        "available": bool(states),
        "generated_at": (
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
        ),
        "direction": daily_state,
        "direction_label": (
            "BULLISH"
            if daily_state == 1
            else "BEARISH"
            if daily_state == -1
            else "NEUTRAL"
        ),
        "regime": regime,
        "alignment_score": round(
            alignment_score,
            2,
        ),
        "upper_core_aligned": (
            upper_core_aligned
        ),
        "full_alignment": full_alignment,
        "aligned_timeframes": (
            aligned_timeframes
        ),
        "opposing_timeframes": (
            opposing_timeframes
        ),
        "timeframes": states,
        "errors": errors,
    }


def get_live_matrix(
    symbol: str,
) -> dict[str, Any]:
    requested = symbol.upper()
    now = time.time()

    with _cache_lock:
        cached = _matrix_cache.get(
            requested
        )

        if (
            cached is not None
            and now
            - float(
                cached.get(
                    "_cached_at",
                    0.0,
                )
            )
            < CACHE_SECONDS
        ):
            return {
                key: value
                for key, value
                in cached.items()
                if key != "_cached_at"
            }

    result = _build_symbol_matrix(
        requested
    )

    stored = {
        **result,
        "_cached_at": now,
    }

    with _cache_lock:
        _matrix_cache[
            requested
        ] = stored

    return result


def combine_matrix_topology(
    liquidity_pressure: float,
    topology_direction: int,
    matrix: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    İlk V2 birleşimi rule-based ve yorumlanabilir tutulur.

    liquidity_pressure:
      mevcut squeeze modelinin event probability değeri.

    topology_direction:
      +1 = SHORT_SQUEEZE / bullish event
      -1 = LONG_SQUEEZE / bearish event
    """
    pressure_score = float(
        np.clip(
            liquidity_pressure * 100.0,
            0.0,
            100.0,
        )
    )

    matrix_available = bool(
        matrix
        and matrix.get("available")
    )

    if not matrix_available:
        return {
            "radar_score": round(
                pressure_score,
                2,
            ),
            "opportunity": "UNFILTERED",
            "matrix_agreement": None,
            "matrix_gate": "UNAVAILABLE",
            "explanation": (
                "Liquidity Pressure only; "
                "Matrix unavailable."
            ),
        }

    matrix_direction = int(
        matrix.get(
            "direction",
            0,
        )
    )

    alignment_score = float(
        matrix.get(
            "alignment_score",
            0.0,
        )
    )

    alignment_factor = (
        alignment_score / 100.0
    )

    direction_valid = (
        matrix_direction in {-1, 1}
        and topology_direction in {-1, 1}
    )

    agrees = (
        direction_valid
        and matrix_direction
        == topology_direction
    )

    conflicts = (
        direction_valid
        and matrix_direction
        != topology_direction
    )

    if agrees:
        combined_score = (
            pressure_score * 0.72
            + alignment_score * 0.28
        )
        opportunity = "WATCH"
        matrix_gate = "PASS"
        explanation = (
            "Topology agrees with Matrix direction."
        )
    elif conflicts:
        combined_score = pressure_score * 0.45
        opportunity = "CONFLICT"
        matrix_gate = "BLOCK"
        explanation = (
            "Topology conflicts with Matrix direction."
        )
    else:
        combined_score = pressure_score * (
            0.70 + 0.30 * alignment_factor
        )
        opportunity = "UNCONFIRMED"
        matrix_gate = "UNAVAILABLE"
        explanation = (
            "Matrix or topology direction unavailable."
        )

    return {
        "radar_score": round(
            float(np.clip(combined_score, 0.0, 100.0)),
            2,
        ),
        "opportunity": opportunity,
        "matrix_agreement": agrees if direction_valid else None,
        "matrix_gate": matrix_gate,
        "explanation": explanation,
    }
