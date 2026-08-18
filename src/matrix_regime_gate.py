from __future__ import annotations

import math
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import requests


BINANCE_BASE_URL = "https://fapi.binance.com"
MATRIX_LENGTH = 20
ER_PERIOD = 12
PCTL_WINDOW = 24 * 90
PCTL_MIN = 24 * 14
SCORE_WINDOW = 500
SCORE_MIN = 100
VALID_THRESHOLD = 65.5833
LIVE_BAR_LIMIT = 5000
CACHE_SECONDS = 60

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "btc_matrix_1h_regime_score.csv"
)

_session = requests.Session()
_session.headers.update({
    "User-Agent": "LiqHeat-Matrix-Regime-Gate/1.0",
    "Accept": "application/json",
})

_cache_lock = Lock()
_cache: dict[str, dict[str, Any]] = {}


def _safe_number(value: Any, digits: int | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def _iso(value: Any) -> str | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _fetch_closed_1h(symbol: str, limit: int = LIVE_BAR_LIMIT) -> pd.DataFrame:
    rows: list[list[Any]] = []
    end_ms: int | None = None

    while len(rows) < limit:
        batch_limit = min(1500, limit - len(rows))
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": "1h",
            "limit": batch_limit,
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        response = _session.get(
            BINANCE_BASE_URL + "/fapi/v1/klines",
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            break

        rows = payload + rows
        oldest_open = int(payload[0][0])
        end_ms = oldest_open - 1
        if len(payload) < batch_limit:
            break

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    frame = pd.DataFrame(rows[-limit:], columns=columns)
    if frame.empty:
        return frame

    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    now = pd.Timestamp.now(tz="UTC")
    return (
        frame[frame["close_time"] < now]
        .dropna(subset=["open", "high", "low", "close", "volume"])
        .drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def _persistent_trend(
    source: pd.Series,
    previous_upper: pd.Series,
    previous_lower: pd.Series,
) -> pd.Series:
    src = source.to_numpy(dtype=float)
    up = previous_upper.to_numpy(dtype=float)
    lo = previous_lower.to_numpy(dtype=float)
    out = np.zeros(len(src), dtype=np.int8)
    state = 0
    for i in range(len(src)):
        if np.isfinite(src[i]) and np.isfinite(up[i]) and src[i] > up[i]:
            state = 1
        elif np.isfinite(src[i]) and np.isfinite(lo[i]) and src[i] < lo[i]:
            state = -1
        out[i] = state
    return pd.Series(out, index=source.index, dtype="int8")


def _build_matrix_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    frame["source"] = (
        frame["open"] + frame["high"] + frame["low"] + frame["close"]
    ) / 4.0

    weighted = frame["source"] * frame["volume"]
    volume_sum = frame["volume"].rolling(MATRIX_LENGTH, min_periods=MATRIX_LENGTH).sum()
    frame["vwma"] = (
        weighted.rolling(MATRIX_LENGTH, min_periods=MATRIX_LENGTH).sum()
        / volume_sum.replace(0, np.nan)
    )
    frame["upper"] = frame["vwma"].rolling(MATRIX_LENGTH, min_periods=MATRIX_LENGTH).max()
    frame["lower"] = frame["vwma"].rolling(MATRIX_LENGTH, min_periods=MATRIX_LENGTH).min()
    frame["trend"] = _persistent_trend(
        frame["source"], frame["upper"].shift(1), frame["lower"].shift(1)
    )
    previous_trend = frame["trend"].shift(1).fillna(0).astype("int8")
    frame["long_flip"] = (frame["trend"] == 1) & (previous_trend == -1)
    frame["short_flip"] = (frame["trend"] == -1) & (previous_trend == 1)
    frame["flip"] = np.select(
        [frame["long_flip"], frame["short_flip"]], [1, -1], default=0
    ).astype("int8")

    frame["distance_to_vwma_pct"] = (
        (frame["source"] - frame["vwma"])
        / frame["vwma"].replace(0, np.nan)
    )

    close = frame["close"].astype(float)
    change = close.diff(ER_PERIOD).abs()
    path = close.diff().abs().rolling(ER_PERIOD, min_periods=ER_PERIOD).sum()
    frame["er"] = change / path.replace(0, np.nan)

    # Frozen research definition. Intentionally uses CLOSE as denominator,
    # while abs VWMA distance comes from canonical Matrix / VWMA distance.
    frame["abs_vwma_dist"] = frame["distance_to_vwma_pct"].abs()
    frame["channel_width"] = (
        (frame["upper"] - frame["lower"]).abs()
        / close.replace(0, np.nan)
    )
    frame["norm_disp"] = (
        frame["abs_vwma_dist"]
        / frame["channel_width"].replace(0, np.nan)
    )

    values = frame["channel_width"].to_numpy(dtype=float)
    pct = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        if i < PCTL_MIN or not np.isfinite(values[i]):
            continue
        lo = max(0, i - PCTL_WINDOW)
        history = values[lo:i]
        history = history[np.isfinite(history)]
        if len(history) < PCTL_MIN:
            continue
        pct[i] = float(np.mean(history <= values[i]) * 100.0)
    frame["channel_pctile"] = pct
    return frame


def _prior_percentile(history: list[float], current: float) -> float | None:
    if not math.isfinite(current):
        return None
    finite = np.asarray([x for x in history[-SCORE_WINDOW:] if math.isfinite(x)], dtype=float)
    if len(finite) < SCORE_MIN:
        return None
    return float(np.mean(finite <= current) * 100.0)


def _load_baseline() -> pd.DataFrame:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Frozen regime baseline not found: {BASELINE_PATH}"
        )
    baseline = pd.read_csv(BASELINE_PATH)
    if "time" not in baseline.columns:
        raise RuntimeError("Regime baseline is missing the time column")
    baseline["time"] = pd.to_datetime(baseline["time"], utc=True, errors="coerce")
    for column in [
        "er", "channel_pctile", "norm_disp", "er_rank",
        "channel_rank", "norm_disp_rank", "regime_score",
    ]:
        if column in baseline.columns:
            baseline[column] = pd.to_numeric(baseline[column], errors="coerce")
    return (
        baseline.dropna(subset=["time", "er", "channel_pctile", "norm_disp"])
        .sort_values("time")
        .reset_index(drop=True)
    )


def _event_from_row(row: pd.Series, source: str) -> dict[str, Any]:
    score = _safe_number(row.get("regime_score"), 4)
    side = str(row.get("side") or "").upper() or None
    valid = score is not None and score >= VALID_THRESHOLD
    return {
        "side": side,
        "close_time": _iso(row.get("time")),
        "score": score,
        "status": "VALID" if valid else "BLOCK",
        "valid": valid,
        "er": _safe_number(row.get("er"), 6),
        "channel_percentile": _safe_number(row.get("channel_pctile"), 4),
        "normalized_displacement": _safe_number(row.get("norm_disp"), 6),
        "er_rank": _safe_number(row.get("er_rank"), 2),
        "channel_rank": _safe_number(row.get("channel_rank"), 2),
        "norm_disp_rank": _safe_number(row.get("norm_disp_rank"), 2),
        "source": source,
    }


def _build_gate(symbol: str) -> dict[str, Any]:
    requested = str(symbol).upper()
    if requested != "BTCUSDT":
        return {
            "available": False,
            "reason": "SYMBOL_NOT_VALIDATED",
            "symbol": requested,
            "timeframe": "1h",
            "threshold": VALID_THRESHOLD,
            "research_status": "BTCUSDT_1H_ONLY",
        }

    baseline = _load_baseline()
    candles = _fetch_closed_1h(requested)
    if len(candles) < PCTL_WINDOW + 50:
        raise RuntimeError(f"Insufficient live 1H history: {len(candles)} bars")

    matrix = _build_matrix_features(candles)
    live_flips = matrix[matrix["flip"] != 0].copy()
    live_flips["side"] = np.where(live_flips["flip"] == 1, "BUY", "SELL")
    live_flips["time"] = live_flips["close_time"]
    live_flips = live_flips.dropna(subset=["er", "channel_pctile", "norm_disp"])

    baseline_end = baseline["time"].max()
    new_flips = live_flips[live_flips["time"] > baseline_end].copy()

    er_history = baseline["er"].astype(float).tolist()
    ch_history = baseline["channel_pctile"].astype(float).tolist()
    nd_history = baseline["norm_disp"].astype(float).tolist()

    scored_new: list[dict[str, Any]] = []
    for _, row in new_flips.sort_values("time").iterrows():
        er = float(row["er"])
        ch = float(row["channel_pctile"])
        nd = float(row["norm_disp"])
        er_rank = _prior_percentile(er_history, er)
        ch_rank = _prior_percentile(ch_history, ch)
        nd_rank = _prior_percentile(nd_history, nd)
        score = None
        if er_rank is not None and ch_rank is not None and nd_rank is not None:
            score = (er_rank + ch_rank + nd_rank) / 3.0
        scored_new.append({
            "time": row["time"],
            "side": row["side"],
            "er": er,
            "channel_pctile": ch,
            "norm_disp": nd,
            "er_rank": er_rank,
            "channel_rank": ch_rank,
            "norm_disp_rank": nd_rank,
            "regime_score": score,
        })
        er_history.append(er)
        ch_history.append(ch)
        nd_history.append(nd)

    if scored_new:
        latest_row = pd.Series(scored_new[-1])
        latest_event = _event_from_row(latest_row, "live")
    else:
        latest_event = _event_from_row(baseline.iloc[-1], "baseline")

    valid_candidates: list[dict[str, Any]] = []
    baseline_valid = baseline[
        pd.to_numeric(baseline.get("regime_score"), errors="coerce") >= VALID_THRESHOLD
    ]
    if not baseline_valid.empty:
        valid_candidates.append(_event_from_row(baseline_valid.iloc[-1], "baseline"))
    for row in scored_new:
        if row.get("regime_score") is not None and float(row["regime_score"]) >= VALID_THRESHOLD:
            valid_candidates.append(_event_from_row(pd.Series(row), "live"))
    last_valid = valid_candidates[-1] if valid_candidates else None

    last_bar = matrix.iloc[-1]
    flip_positions = np.flatnonzero(matrix["flip"].to_numpy() != 0)
    bars_since_flip = (
        int(len(matrix) - 1 - flip_positions[-1]) if len(flip_positions) else None
    )
    trend = int(last_bar["trend"])
    trend_label = "BUY" if trend == 1 else "SELL" if trend == -1 else "NEUTRAL"

    return {
        "available": True,
        "symbol": requested,
        "timeframe": "1h",
        "threshold": VALID_THRESHOLD,
        "status": latest_event["status"],
        "regime_score": latest_event["score"],
        "matrix_trend": trend_label,
        "bars_since_flip": bars_since_flip,
        "latest_flip": latest_event,
        "last_valid_signal": last_valid,
        "new_flips_since_baseline": len(scored_new),
        "baseline_last_time": _iso(baseline_end),
        "live_last_close": _iso(last_bar["close_time"]),
        "research_status": "FROZEN_VALIDATED_GATE",
        "research_spec": {
            "er_period": ER_PERIOD,
            "channel_percentile_window": PCTL_WINDOW,
            "channel_percentile_min": PCTL_MIN,
            "score_window_flips": SCORE_WINDOW,
            "score_min_flips": SCORE_MIN,
            "channel_denominator": "close",
            "distance_denominator": "vwma",
        },
    }


def get_matrix_regime_gate(symbol: str) -> dict[str, Any]:
    requested = str(symbol).upper()
    now = time.time()
    with _cache_lock:
        cached = _cache.get(requested)
        if cached and now - float(cached.get("_cached_at", 0.0)) < CACHE_SECONDS:
            return {k: v for k, v in cached.items() if k != "_cached_at"}

    try:
        result = _build_gate(requested)
    except Exception as exc:
        result = {
            "available": False,
            "reason": "REGIME_GATE_ERROR",
            "symbol": requested,
            "timeframe": "1h",
            "threshold": VALID_THRESHOLD,
            "error": f"{type(exc).__name__}: {exc}",
        }

    stored = dict(result)
    stored["_cached_at"] = now
    with _cache_lock:
        _cache[requested] = stored
    return result
