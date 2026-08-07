from __future__ import annotations

import math
from typing import Any


def _num(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else 0.0
    except Exception:
        return 0.0


def build_compact_heatmap(
    payload: dict[str, Any] | None,
    current_price: float,
    *,
    max_levels: int = 28,
    max_distance_pct: float = 0.08,
) -> dict[str, Any]:
    """Build a compact UI-safe liquidation map from a Liqheat payload.

    The payload dataPoints format is expected to be:
      [*, price, long_volume, short_volume, ...]

    Output contains only aggregated price levels and normalized intensity.
    """
    price_now = _num(current_price)
    if price_now <= 0 or not isinstance(payload, dict):
        return {"available": False, "levels": []}

    levels: dict[float, dict[str, float]] = {}
    for point in payload.get("dataPoints", []) or []:
        if not isinstance(point, list) or len(point) < 4:
            continue
        price = _num(point[1])
        if price <= 0:
            continue
        distance = (price - price_now) / price_now
        if abs(distance) > max_distance_pct:
            continue
        long_volume = max(0.0, _num(point[2]))
        short_volume = max(0.0, _num(point[3]))
        if long_volume + short_volume <= 0:
            continue
        bucket = levels.setdefault(price, {"long": 0.0, "short": 0.0})
        bucket["long"] += long_volume
        bucket["short"] += short_volume

    if not levels:
        return {"available": False, "levels": []}

    rows = []
    for price, volumes in levels.items():
        total = volumes["long"] + volumes["short"]
        rows.append({
            "price": price,
            "distance_pct": (price - price_now) / price_now * 100.0,
            "long_volume": volumes["long"],
            "short_volume": volumes["short"],
            "total_volume": total,
            "side": "ABOVE" if price > price_now else "BELOW" if price < price_now else "AT_PRICE",
        })

    # Keep a mixture of proximity and size: strongest near-price pools dominate,
    # but very large pools a little farther away remain visible.
    for row in rows:
        proximity = 1.0 / (1.0 + abs(row["distance_pct"]) * 0.85)
        row["display_weight"] = row["total_volume"] * proximity

    selected = sorted(rows, key=lambda r: r["display_weight"], reverse=True)[:max_levels]
    selected.sort(key=lambda r: r["price"], reverse=True)

    max_total = max(r["total_volume"] for r in selected) or 1.0
    max_abs_distance = max(abs(r["distance_pct"]) for r in selected) or 1.0

    for row in selected:
        row["intensity"] = round(row["total_volume"] / max_total, 6)
        row["position"] = round(0.5 + 0.5 * (row["distance_pct"] / max_abs_distance), 6)
        row["price"] = round(row["price"], 8)
        row["distance_pct"] = round(row["distance_pct"], 4)
        row["long_volume"] = round(row["long_volume"], 4)
        row["short_volume"] = round(row["short_volume"], 4)
        row["total_volume"] = round(row["total_volume"], 4)
        row.pop("display_weight", None)

    return {
        "available": True,
        "timeframe": "24h",
        "current_price": price_now,
        "max_distance_pct": round(max_abs_distance, 4),
        "levels": selected,
    }
