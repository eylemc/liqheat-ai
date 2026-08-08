from __future__ import annotations

import asyncio
import json
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from supabase import create_client

from src.build_liq_topology_v2 import build_feature
from src.liquidation_heatmap_live import build_compact_heatmap
from src.matrix_live import (
    combine_matrix_topology,
    get_live_matrix,
)
from src.market_risk_live import (
    add_dynamic_features_live,
    market_risk_engine,
    sample_live_history,
)
from src.risk_state_stabilizer import risk_state_stabilizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "squeeze_v1"
ENV_PATH = PROJECT_ROOT / ".env"

REFRESH_SECONDS = 75
TIMEFRAME = "24h"
RISK_TIMEFRAME = "1h"
RISK_HISTORY_LIMIT = 256

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XAUUSDT",
    "XAGUSDT",
]

CATEGORICAL_FEATURES = [
    "symbol",
    "nearest_side",
]

SIDE_MAP = {
    "LOWER": -1,
    "TIE": 0,
    "UPPER": 1,
}

LOG1P_COLUMNS = [
    "current_price",
    "upper_distance_pct",
    "lower_distance_pct",
    "distance_advantage",
    "upper_pool_volume",
    "lower_pool_volume",
    "nearest_pool_volume",
    "farther_pool_volume",
    "pool_volume_ratio",
    "distance_pressure_ratio",
    "upper_active_levels",
    "lower_active_levels",
    "upper_total_volume",
    "lower_total_volume",
]


load_dotenv(ENV_PATH)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL veya SUPABASE_SECRET_KEY .env içinde bulunamadı"
    )

with open(MODEL_DIR / "features.json", encoding="utf-8") as file:
    FEATURE_COLUMNS = json.load(file)

with open(MODEL_DIR / "thresholds.json", encoding="utf-8") as file:
    THRESHOLDS = json.load(file)

model = CatBoostClassifier()
model.load_model(str(MODEL_DIR / "model.cbm"))

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)

cache_lock = Lock()

radar_cache: dict[str, Any] = {
    "engine": "liqheat-radar-v2-matrix-topology",
    "status": "STARTING",
    "generated_at": None,
    "last_success_at": None,
    "refresh_seconds": REFRESH_SECONDS,
    "timeframe": TIMEFRAME,
    "symbol_count": 0,
    "radar": [],
    "error": None,
}


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def safe_log1p_value(value: Any) -> float:
    number = json_safe_number(value)

    if number is None or number < 0:
        return np.nan

    return float(np.log1p(number))


def fetch_latest_rows() -> list[dict[str, Any]]:
    """
    Her sembolün en son 24h snapshot'ını ayrı sorguyla çeker.

    Bu yaklaşım, sembollerin timestamp'leri birebir aynı olmadığı için
    tek global latest timestamp kullanmaktan daha güvenlidir.
    """
    rows: list[dict[str, Any]] = []

    for symbol in SYMBOLS:
        response = (
            supabase
            .table("liq_logging")
            .select(
                "id,logged_at,symbol,timeframe,current_price,"
                "liquidation_count,price_min,price_max,payload"
            )
            .eq("symbol", symbol)
            .eq("timeframe", TIMEFRAME)
            .order("logged_at", desc=True)
            .limit(1)
            .execute()
        )

        if not response.data:
            continue

        rows.append(response.data[0])

    return rows


def topology_feature_from_live_row(
    row: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Offline topology builder'ın birebir build_feature() fonksiyonunu kullanır.
    Böylece training-serving skew oluşmaz.
    """
    adapted_row = {
        "id": row.get("id"),
        "logged_at": row.get("logged_at"),
        "symbol": row.get("symbol"),
        "timeframe": row.get("timeframe"),
        "current_price": row.get("current_price"),
        "payload_json": json.dumps(
            row.get("payload") or {},
            separators=(",", ":"),
        ),
        "source_file": "supabase-live",
    }

    try:
        return build_feature(adapted_row)
    except Exception:
        return None


def add_ml_features(
    topology_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    scripts/build_topology_v2_ml_features.py ile aynı dönüşümleri
    canlı satırlara uygular.
    """
    frame = pd.DataFrame(topology_rows)

    if frame.empty:
        return frame

    frame["logged_at"] = pd.to_datetime(
        frame["logged_at"],
        utc=True,
        errors="coerce",
    )

    frame["has_upper_level"] = (
        frame["nearest_upper_price"]
        .notna()
        .astype("int8")
    )

    frame["has_lower_level"] = (
        frame["nearest_lower_price"]
        .notna()
        .astype("int8")
    )

    frame["has_topology"] = (
        frame["nearest_side"]
        .notna()
        .astype("int8")
    )

    frame["nearest_side_code"] = (
        frame["nearest_side"]
        .map(SIDE_MAP)
        .astype("Int8")
    )

    frame["active_level_difference"] = (
        pd.to_numeric(
            frame["upper_active_levels"],
            errors="coerce",
        )
        - pd.to_numeric(
            frame["lower_active_levels"],
            errors="coerce",
        )
    )

    frame["active_level_total"] = (
        pd.to_numeric(
            frame["upper_active_levels"],
            errors="coerce",
        )
        + pd.to_numeric(
            frame["lower_active_levels"],
            errors="coerce",
        )
    )

    upper_total = pd.to_numeric(
        frame["upper_total_volume"],
        errors="coerce",
    )

    lower_total = pd.to_numeric(
        frame["lower_total_volume"],
        errors="coerce",
    )

    volume_denominator = upper_total + lower_total

    frame["total_volume_imbalance_check"] = np.where(
        volume_denominator > 0,
        (upper_total - lower_total) / volume_denominator,
        0.0,
    )

    frame["signed_distance_edge"] = (
        pd.to_numeric(
            frame["lower_distance_pct"],
            errors="coerce",
        )
        - pd.to_numeric(
            frame["upper_distance_pct"],
            errors="coerce",
        )
    )

    for column in LOG1P_COLUMNS:
        frame[f"log1p_{column}"] = (
            frame[column]
            .map(safe_log1p_value)
            .astype(float)
        )

    frame["hour_utc"] = (
        frame["logged_at"]
        .dt.hour
        .astype("Int8")
    )

    frame["day_of_week_utc"] = (
        frame["logged_at"]
        .dt.dayofweek
        .astype("Int8")
    )

    frame["is_weekend_utc"] = (
        frame["day_of_week_utc"] >= 5
    ).astype("int8")

    frame["hour_sin"] = np.sin(
        2 * np.pi * frame["hour_utc"].astype(float) / 24
    )

    frame["hour_cos"] = np.cos(
        2 * np.pi * frame["hour_utc"].astype(float) / 24
    )

    frame["dow_sin"] = np.sin(
        2
        * np.pi
        * frame["day_of_week_utc"].astype(float)
        / 7
    )

    frame["dow_cos"] = np.cos(
        2
        * np.pi
        * frame["day_of_week_utc"].astype(float)
        / 7
    )

    for column in CATEGORICAL_FEATURES:
        frame[column] = (
            frame[column]
            .astype("string")
            .fillna("<MISSING>")
            .astype(str)
        )

    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    for column in FEATURE_COLUMNS:
        if column not in CATEGORICAL_FEATURES:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    return frame


def fetch_market_risk_history(symbol: str) -> list[dict[str, Any]]:
    """Fetch enough 1h-topology snapshots to reproduce 15m dynamic V2 features."""
    response = (
        supabase
        .table("liq_logging")
        .select(
            "id,logged_at,symbol,timeframe,current_price,"
            "liquidation_count,price_min,price_max,payload"
        )
        .eq("symbol", str(symbol).upper())
        .eq("timeframe", RISK_TIMEFRAME)
        .order("logged_at", desc=True)
        .limit(RISK_HISTORY_LIMIT)
        .execute()
    )
    rows = list(response.data or [])
    rows.reverse()
    return rows


def build_ai_market_risk(symbol: str) -> dict[str, Any]:
    requested = str(symbol).upper()
    if requested not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
        return {
            "available": False,
            "reason": "SYMBOL_NOT_TRAINED",
            "symbol": requested,
            "horizon_minutes": 15,
        }

    rows = fetch_market_risk_history(requested)
    topology_rows: list[dict[str, Any]] = []
    for row in rows:
        feature = topology_feature_from_live_row(row)
        if feature is not None:
            topology_rows.append(feature)

    if not topology_rows:
        return {
            "available": False,
            "reason": "NO_LIVE_HISTORY",
            "symbol": requested,
            "horizon_minutes": 15,
        }

    frame = add_ml_features(topology_rows)
    frame = sample_live_history(frame)
    frame = add_dynamic_features_live(frame)
    raw_risk = market_risk_engine.score_latest(frame, requested)
    return risk_state_stabilizer.update(raw_risk)


def classify_direction(
    predicted_event: str,
    direction_confidence: float,
) -> tuple[str, str, str]:
    """
    Yön tahmini ile yön güvenini birbirinden ayırır.

    direction_confidence:
      < 0.58  -> yön teyitsiz
      0.58-0.65 -> lean
      >= 0.65 -> güçlü yön
    """
    if direction_confidence < 0.58:
        return (
            "NEUTRAL",
            "UNCONFIRMED",
            "UNCONFIRMED",
        )

    if predicted_event == "SHORT_SQUEEZE":
        if direction_confidence < 0.65:
            return (
                "LEAN_BULLISH",
                "SHORT_SQUEEZE",
                "LEAN",
            )

        return (
            "BULLISH",
            "SHORT_SQUEEZE",
            "CONFIRMED",
        )

    if direction_confidence < 0.65:
        return (
            "LEAN_BEARISH",
            "LONG_SQUEEZE",
            "LEAN",
        )

    return (
        "BEARISH",
        "LONG_SQUEEZE",
        "CONFIRMED",
    )


def classify_status(
    score: float,
    direction_confidence: float,
) -> str:
    """
    WATCH yalnız event riskine dayanabilir.

    ALERT ve CRITICAL için hem yüksek event skoru
    hem de yeterli yön teyidi gerekir.
    """
    if (
        score >= float(THRESHOLDS["critical"])
        and direction_confidence >= 0.65
    ):
        return "CRITICAL"

    if (
        score >= float(THRESHOLDS["alert"])
        and direction_confidence >= 0.60
    ):
        return "ALERT"

    if score >= float(THRESHOLDS["watch"]):
        return "WATCH"

    return "NORMAL"


def build_live_radar() -> dict[str, Any]:
    live_rows = fetch_latest_rows()

    if not live_rows:
        raise RuntimeError(
            "Supabase liq_logging tablosundan canlı snapshot alınamadı"
        )

    topology_rows = []

    source_by_id: dict[str, dict[str, Any]] = {}

    for row in live_rows:
        feature = topology_feature_from_live_row(row)

        if feature is None:
            continue

        topology_rows.append(feature)
        source_by_id[str(feature["id"])] = row

    if not topology_rows:
        raise RuntimeError(
            "Canlı snapshotlardan topology feature üretilemedi"
        )

    feature_frame = add_ml_features(topology_rows)

    if feature_frame.empty:
        raise RuntimeError("Canlı ML feature frame boş")

    X = feature_frame[FEATURE_COLUMNS].copy()

    probabilities = model.predict_proba(X)
    classes = [int(value) for value in model.classes_]

    class_index = {
        class_value: index
        for index, class_value in enumerate(classes)
    }

    long_index = class_index[-1]
    none_index = class_index[0]
    short_index = class_index[1]

    results: list[dict[str, Any]] = []

    now = pd.Timestamp.now(tz="UTC")

    for row_index in range(len(feature_frame)):
        feature_row = feature_frame.iloc[row_index]

        long_probability = float(
            probabilities[row_index, long_index]
        )

        none_probability = float(
            probabilities[row_index, none_index]
        )

        short_probability = float(
            probabilities[row_index, short_index]
        )

        event_probability = (
            long_probability
            + short_probability
        )

        if short_probability >= long_probability:
            raw_prediction = "SHORT_SQUEEZE"
            direction_probability = short_probability
        else:
            raw_prediction = "LONG_SQUEEZE"
            direction_probability = long_probability

        direction_confidence = (
            direction_probability / event_probability
            if event_probability > 0
            else 0.0
        )

        (
            bias,
            displayed_prediction,
            direction_state,
        ) = classify_direction(
            raw_prediction,
            direction_confidence,
        )

        topology_direction = (
            1
            if raw_prediction == "SHORT_SQUEEZE"
            else -1
        )

        matrix_data = None
        matrix_error = None

        try:
            matrix_data = get_live_matrix(
                str(feature_row["symbol"])
            )
        except Exception as exc:
            matrix_error = (
                f"{type(exc).__name__}: {exc}"
            )

        try:
            ai_market_risk = build_ai_market_risk(
                str(feature_row["symbol"])
            )
        except Exception as exc:
            ai_market_risk = {
                "available": False,
                "reason": "LIVE_INFERENCE_ERROR",
                "symbol": str(feature_row["symbol"]),
                "horizon_minutes": 15,
                "error": f"{type(exc).__name__}: {exc}",
            }

        combined = combine_matrix_topology(
            liquidity_pressure=event_probability,
            topology_direction=topology_direction,
            matrix=matrix_data,
        )

        status = classify_status(
            event_probability,
            direction_confidence,
        )

        logged_at = pd.to_datetime(
            feature_row["logged_at"],
            utc=True,
            errors="coerce",
        )

        age_seconds = None

        if pd.notna(logged_at):
            age_seconds = max(
                0.0,
                float(
                    (now - logged_at)
                    .total_seconds()
                ),
            )

        feature_id = str(feature_row["id"])
        source_row = source_by_id.get(feature_id, {})
        compact_heatmap = build_compact_heatmap(
            source_row.get("payload") or {},
            json_safe_number(feature_row["current_price"]) or 0.0,
        )

        results.append({
            "symbol": str(feature_row["symbol"]),
            "timeframe": TIMEFRAME,
            "logged_at": (
                logged_at.isoformat()
                if pd.notna(logged_at)
                else None
            ),
            "age_seconds": (
                round(age_seconds, 1)
                if age_seconds is not None
                else None
            ),
            "current_price": json_safe_number(
                feature_row["current_price"]
            ),
            "liquidation_heatmap": compact_heatmap,
            # New primary product output: direction-independent 15m AI market risk.
            "ai_market_risk": ai_market_risk,

            # Backward-compatible raw probability.
            "score": round(event_probability, 6),

            # Existing model output is now explicitly named.
            "liquidity_pressure": round(
                event_probability,
                6,
            ),
            "liquidity_pressure_score": round(
                event_probability * 100,
                2,
            ),

            # Matrix + topology opportunity score.
            "radar_score": combined[
                "radar_score"
            ],
            "opportunity": combined[
                "opportunity"
            ],
            "matrix_gate": combined[
                "matrix_gate"
            ],
            "matrix_agreement": combined[
                "matrix_agreement"
            ],
            "radar_explanation": combined[
                "explanation"
            ],

            # Legacy topology status remains visible.
            "status": status,
            "prediction": displayed_prediction,
            "raw_prediction": raw_prediction,
            "bias": bias,
            "direction_state": direction_state,
            "direction_confidence": round(
                direction_confidence,
                6,
            ),
            "alert_eligible": (
                status in {"ALERT", "CRITICAL"}
            ),
            "matrix": (
                matrix_data
                if matrix_data is not None
                else {
                    "available": False,
                    "error": matrix_error,
                }
            ),
            "probabilities": {
                "long_squeeze": round(
                    long_probability,
                    6,
                ),
                "no_event": round(
                    none_probability,
                    6,
                ),
                "short_squeeze": round(
                    short_probability,
                    6,
                ),
            },
            "topology": {
                "nearest_side": str(
                    feature_row["nearest_side"]
                ),
                "upper_distance_pct": json_safe_number(
                    feature_row[
                        "upper_distance_pct"
                    ]
                ),
                "lower_distance_pct": json_safe_number(
                    feature_row[
                        "lower_distance_pct"
                    ]
                ),
                "upper_pool_volume": json_safe_number(
                    feature_row[
                        "upper_pool_volume"
                    ]
                ),
                "lower_pool_volume": json_safe_number(
                    feature_row[
                        "lower_pool_volume"
                    ]
                ),
                "topology_imbalance": json_safe_number(
                    feature_row[
                        "topology_imbalance"
                    ]
                ),
            },
            "snapshot_id": feature_id,
            "liquidation_count": json_safe_number(
                source_row.get(
                    "liquidation_count"
                )
            ),
        })

    # Matrix AI Radar prioritizes the safest trained 15m conditions.
    # Untrained markets remain visible after scored markets.
    results.sort(
        key=lambda item: (
            0 if (item.get("ai_market_risk") or {}).get("available") else 1,
            float((item.get("ai_market_risk") or {}).get("risk_score", 999.0)),
        )
    )

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    freshest_age = min(
        (
            item["age_seconds"]
            for item in results
            if item["age_seconds"] is not None
        ),
        default=None,
    )

    stalest_age = max(
        (
            item["age_seconds"]
            for item in results
            if item["age_seconds"] is not None
        ),
        default=None,
    )

    return {
        "engine": "matrix-ai-radar-v2-live",
        "status": "ONLINE",
        "source": {
            "topology": "supabase.liq_logging",
            "matrix": "binance-usd-m-futures-klines",
        },
        "scoring": {
            "liquidity_pressure": (
                "Existing squeeze event probability"
            ),
            "matrix": (
                "VWMA(20) OHLC4 multi-timeframe regime"
            ),
            "radar_score": (
                "Legacy Matrix + Topology opportunity score; retained for diagnostics"
            ),
            "ai_market_risk": (
                "Primary product score: calibrated 15m V2 risk, Matrix excluded from risk model"
            ),
        },
        "generated_at": utc_iso(),
        "last_success_at": utc_iso(),
        "refresh_seconds": REFRESH_SECONDS,
        "timeframe": TIMEFRAME,
        "symbol_count": len(results),
        "freshest_snapshot_age_seconds": freshest_age,
        "stalest_snapshot_age_seconds": stalest_age,
        "thresholds": THRESHOLDS,
        "radar": results,
        "error": None,
    }


def refresh_cache() -> None:
    global radar_cache

    try:
        payload = build_live_radar()

        with cache_lock:
            radar_cache = payload

        print(
            f"[{utc_iso()}] "
            f"Radar refreshed: "
            f"{len(payload['radar'])} symbols",
            flush=True,
        )

    except Exception as exc:
        with cache_lock:
            previous = dict(radar_cache)
            previous["status"] = "DEGRADED"
            previous["generated_at"] = utc_iso()
            previous["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            radar_cache = previous

        print(
            f"[{utc_iso()}] "
            f"Radar refresh failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


async def radar_worker() -> None:
    while True:
        await asyncio.to_thread(refresh_cache)
        await asyncio.sleep(REFRESH_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(refresh_cache)

    task = asyncio.create_task(
        radar_worker()
    )

    try:
        yield
    finally:
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="LiqHeat AI Radar",
    version="0.4.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "static"),
    name="static",
)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(
        PROJECT_ROOT / "static" / "index.html"
    )


@app.get("/health")
def health() -> dict[str, Any]:
    with cache_lock:
        cache = dict(radar_cache)

    return {
        "status": cache.get("status"),
        "engine": cache.get("engine"),
        "last_success_at": cache.get(
            "last_success_at"
        ),
        "symbol_count": cache.get(
            "symbol_count",
            0,
        ),
        "error": cache.get("error"),
    }


@app.get("/radar")
def radar() -> dict[str, Any]:
    with cache_lock:
        return dict(radar_cache)


@app.get("/radar/{symbol}")
def radar_symbol(
    symbol: str,
) -> dict[str, Any]:
    requested = symbol.upper()

    with cache_lock:
        items = list(
            radar_cache.get("radar", [])
        )

    for item in items:
        if item["symbol"] == requested:
            return item

    raise HTTPException(
        status_code=404,
        detail=f"Symbol not found: {requested}",
    )


@app.post("/radar/refresh")
async def force_refresh() -> dict[str, Any]:
    await asyncio.to_thread(refresh_cache)

    with cache_lock:
        return dict(radar_cache)
