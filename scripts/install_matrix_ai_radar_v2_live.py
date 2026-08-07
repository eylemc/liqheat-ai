#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "src" / "textara_api.py"
INDEX = ROOT / "static" / "index.html"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"Already patched: {label}")
        return text
    if old not in text:
        raise SystemExit(f"Patch target not found: {label}")
    print(f"Patched: {label}")
    return text.replace(old, new, 1)


def patch_api() -> None:
    text = API.read_text(encoding="utf-8")
    backup = API.with_suffix(".py.before-matrix-ai-radar-v2-live")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    text = replace_once(
        text,
        '''from src.matrix_live import (\n    combine_matrix_topology,\n    get_live_matrix,\n)\n''',
        '''from src.matrix_live import (\n    combine_matrix_topology,\n    get_live_matrix,\n)\nfrom src.market_risk_live import (\n    add_dynamic_features_live,\n    market_risk_engine,\n    sample_live_history,\n)\n''',
        "market risk imports",
    )

    text = replace_once(
        text,
        '''REFRESH_SECONDS = 75\nTIMEFRAME = "24h"\n''',
        '''REFRESH_SECONDS = 75\nTIMEFRAME = "24h"\nRISK_TIMEFRAME = "1h"\nRISK_HISTORY_LIMIT = 256\n''',
        "risk runtime constants",
    )

    helper = '''\n\ndef fetch_market_risk_history(symbol: str) -> list[dict[str, Any]]:\n    """Fetch enough 1h-topology snapshots to reproduce 15m dynamic V2 features."""\n    response = (\n        supabase\n        .table("liq_logging")\n        .select(\n            "id,logged_at,symbol,timeframe,current_price,"\n            "liquidation_count,price_min,price_max,payload"\n        )\n        .eq("symbol", str(symbol).upper())\n        .eq("timeframe", RISK_TIMEFRAME)\n        .order("logged_at", desc=True)\n        .limit(RISK_HISTORY_LIMIT)\n        .execute()\n    )\n    rows = list(response.data or [])\n    rows.reverse()\n    return rows\n\n\ndef build_ai_market_risk(symbol: str) -> dict[str, Any]:\n    requested = str(symbol).upper()\n    if requested not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:\n        return {\n            "available": False,\n            "reason": "SYMBOL_NOT_TRAINED",\n            "symbol": requested,\n            "horizon_minutes": 15,\n        }\n\n    rows = fetch_market_risk_history(requested)\n    topology_rows: list[dict[str, Any]] = []\n    for row in rows:\n        feature = topology_feature_from_live_row(row)\n        if feature is not None:\n            topology_rows.append(feature)\n\n    if not topology_rows:\n        return {\n            "available": False,\n            "reason": "NO_LIVE_HISTORY",\n            "symbol": requested,\n            "horizon_minutes": 15,\n        }\n\n    frame = add_ml_features(topology_rows)\n    frame = sample_live_history(frame)\n    frame = add_dynamic_features_live(frame)\n    return market_risk_engine.score_latest(frame, requested)\n'''

    text = replace_once(
        text,
        '''    return frame\n\n\ndef classify_direction(\n''',
        '''    return frame\n''' + helper + '''\n\ndef classify_direction(\n''',
        "live 15m risk history and inference helpers",
    )

    text = replace_once(
        text,
        '''        combined = combine_matrix_topology(\n            liquidity_pressure=event_probability,\n            topology_direction=topology_direction,\n            matrix=matrix_data,\n        )\n''',
        '''        try:\n            ai_market_risk = build_ai_market_risk(\n                str(feature_row["symbol"])\n            )\n        except Exception as exc:\n            ai_market_risk = {\n                "available": False,\n                "reason": "LIVE_INFERENCE_ERROR",\n                "symbol": str(feature_row["symbol"]),\n                "horizon_minutes": 15,\n                "error": f"{type(exc).__name__}: {exc}",\n            }\n\n        combined = combine_matrix_topology(\n            liquidity_pressure=event_probability,\n            topology_direction=topology_direction,\n            matrix=matrix_data,\n        )\n''',
        "invoke live 15m AI risk",
    )

    text = replace_once(
        text,
        '''            "current_price": json_safe_number(\n                feature_row["current_price"]\n            ),\n            # Backward-compatible raw probability.\n''',
        '''            "current_price": json_safe_number(\n                feature_row["current_price"]\n            ),\n            # New primary product output: direction-independent 15m AI market risk.\n            "ai_market_risk": ai_market_risk,\n\n            # Backward-compatible raw probability.\n''',
        "expose ai_market_risk in API",
    )

    text = replace_once(
        text,
        '''    results.sort(\n        key=lambda item: item["radar_score"],\n        reverse=True,\n    )\n''',
        '''    # Matrix AI Radar prioritizes the safest trained 15m conditions.\n    # Untrained markets remain visible after scored markets.\n    results.sort(\n        key=lambda item: (\n            0 if (item.get("ai_market_risk") or {}).get("available") else 1,\n            float((item.get("ai_market_risk") or {}).get("risk_score", 999.0)),\n        )\n    )\n''',
        "risk-first ranking",
    )

    text = replace_once(
        text,
        '''        "engine": "liqheat-radar-v2-matrix-topology",\n''',
        '''        "engine": "matrix-ai-radar-v2-live",\n''',
        "engine identity",
    )

    text = replace_once(
        text,
        '''            "radar_score": (\n                "Rule-based Matrix + Topology opportunity score"\n            ),\n''',
        '''            "radar_score": (\n                "Legacy Matrix + Topology opportunity score; retained for diagnostics"\n            ),\n            "ai_market_risk": (\n                "Primary product score: calibrated 15m V2 risk, Matrix excluded from risk model"\n            ),\n''',
        "API scoring metadata",
    )

    API.write_text(text, encoding="utf-8")


def patch_index() -> None:
    text = INDEX.read_text(encoding="utf-8")
    backup = INDEX.with_suffix(".html.before-matrix-ai-radar-v2-live")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    replacements = [
        ("<title>LiqHeat AI Radar</title>", "<title>Matrix AI Radar</title>", "page title"),
        ("<h1>LIQHEAT AI RADAR</h1>", "<h1>MATRIX AI RADAR</h1>", "brand title"),
        ("<p>Matrix + Topology Intelligence</p>", "<p>Matrix Direction + 15m AI Market Risk</p>", "brand subtitle"),
        ("<small>Live Matrix + topology stream</small>", "<small>Live Matrix direction + AI risk</small>", "markets summary"),
        ("<span class=\"summary-label\">Highest radar score</span>", "<span class=\"summary-label\">Lowest 15m AI risk</span>", "risk summary label"),
        ("<small id=\"highestSymbol\">Waiting for radar</small>", "<small id=\"highestSymbol\">Waiting for AI risk</small>", "risk summary copy"),
        ("<span class=\"summary-label\">Active opportunities</span>", "<span class=\"summary-label\">Low-risk markets</span>", "low risk summary label"),
        ("<small>Watch, high or critical</small>", "<small>15m AI Risk = LOW</small>", "low risk summary copy"),
        ("<span class=\"eyebrow\">LIVE MARKET RADAR</span>\n        <h2>Where are pressure and trend aligned?</h2>", "<span class=\"eyebrow\">MATRIX AI RADAR</span>\n        <h2>Matrix direction with near-term trade risk</h2>", "section heading"),
        ("<h2>Radar ranking</h2>", "<h2>Best current trade conditions</h2>", "table heading"),
        ('''              <th>Radar score</th>\n              <th>Opportunity</th>\n              <th>Matrix</th>\n              <th>Alignment</th>\n              <th>Gate</th>\n              <th>Liquidity pressure</th>''', '''              <th>Matrix</th>\n              <th>15m risk score</th>\n              <th>AI risk</th>\n              <th>Alignment</th>\n              <th>Liquidity pressure</th>''', "table columns"),
        ("<span>LiqHeat Radar V2 — Matrix + Topology Engine</span>", "<span>Matrix AI Radar — Matrix Direction + 15m AI Market Risk</span>", "footer"),
        ('''  <script src="/static/outcome_stability.js?v=6"></script>\n  <script src="/static/watch_pulse.js?v=1"></script>\n  <script src="/static/app.js?v=4"></script>''', '''  <script src="/static/matrix_ai_radar.js?v=1"></script>''', "dashboard javascript"),
    ]
    for old, new, label in replacements:
        text = replace_once(text, old, new, label)
    INDEX.write_text(text, encoding="utf-8")


def main() -> None:
    patch_api()
    patch_index()
    print("Matrix AI Radar V2 live integration installed.")
    print("Backups were created next to patched files with .before-matrix-ai-radar-v2-live suffixes.")


if __name__ == "__main__":
    main()
