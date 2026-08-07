#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "src" / "textara_api.py"
JS = ROOT / "static" / "matrix_ai_radar.js"
INDEX = ROOT / "static" / "index.html"


def patch_api() -> None:
    text = API.read_text(encoding="utf-8")
    original = text

    if "from src.liquidation_heatmap_live import build_compact_heatmap" not in text:
        marker = "from src.build_liq_topology_v2 import build_feature\n"
        if marker not in text:
            raise SystemExit("API import marker not found")
        text = text.replace(
            marker,
            marker + "from src.liquidation_heatmap_live import build_compact_heatmap\n",
            1,
        )

    if "compact_heatmap = build_compact_heatmap(" not in text:
        marker = '        source_row = source_by_id.get(feature_id, {})\n'
        if marker not in text:
            raise SystemExit("API source_row marker not found")
        addition = '''        compact_heatmap = build_compact_heatmap(\n            source_row.get("payload") or {},\n            json_safe_number(feature_row["current_price"]) or 0.0,\n        )\n'''
        text = text.replace(marker, marker + addition, 1)

    if '"liquidation_heatmap": compact_heatmap' not in text:
        marker = '''            "current_price": json_safe_number(\n                feature_row["current_price"]\n            ),\n'''
        if marker not in text:
            raise SystemExit("API current_price marker not found")
        text = text.replace(
            marker,
            marker + '            "liquidation_heatmap": compact_heatmap,\n',
            1,
        )

    if text != original:
        backup = API.with_suffix(".py.before-compact-heatmap")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        API.write_text(text, encoding="utf-8")
        print("Patched API compact liquidation heatmap payload")
    else:
        print("API compact heatmap already installed")


def patch_js() -> None:
    text = JS.read_text(encoding="utf-8")
    original = text

    if "function compactLiquidationHeatmap(item)" not in text:
        marker = "function matrixTimeframeStrip(item) {"
        pos = text.find(marker)
        if pos < 0:
            raise SystemExit("JS matrixTimeframeStrip marker not found")
        fn = r'''function compactLiquidationHeatmap(item) {
  const map = item.liquidation_heatmap;
  if (!map?.available || !Array.isArray(map.levels) || !map.levels.length) {
    return '<div class="liq-mini-empty">Heatmap unavailable</div>';
  }
  const levels = map.levels.map((level) => {
    const intensity = Math.max(0, Math.min(1, Number(level.intensity || 0)));
    const position = Math.max(0, Math.min(1, Number(level.position ?? 0.5)));
    const top = (1 - position) * 100;
    const width = 18 + intensity * 82;
    const opacity = 0.20 + intensity * 0.80;
    const cls = level.side === "ABOVE" ? "liq-above" : "liq-below";
    const title = `${formatNumber(level.price, 6)} · ${formatNumber(level.total_volume, 0)}`;
    return `<div class="liq-level ${cls}" style="top:${top.toFixed(2)}%;opacity:${opacity.toFixed(2)}" title="${title}">
      <span style="width:${width.toFixed(1)}%"></span>
    </div>`;
  }).join("");
  return `<div class="liq-mini-wrap">
    <div class="liq-mini-title"><span>LIQUIDATION HEATMAP</span><small>${String(map.timeframe || "24h").toUpperCase()}</small></div>
    <div class="liq-mini-map">
      <div class="liq-price-line"></div>
      <div class="liq-price-dot"></div>
      ${levels}
    </div>
    <div class="liq-mini-axis"><span>below price</span><strong>${formatNumber(item.current_price, 6)}</strong><span>above price</span></div>
  </div>`;
}

'''
        text = text[:pos] + fn + text[pos:]

    if ".liq-mini-wrap" not in text:
        style_marker = "    .scalp-long{color:#4ce5a6}.scalp-short{color:#ff6f7d}.scalp-na{color:var(--muted)}\n"
        if style_marker not in text:
            # tolerate the older naming variant
            style_marker = "    .scalp-long{color:#4ce5a6}.scalp-short{color:#ff6f7d}.scalp-wait{color:var(--muted)}\n"
        if style_marker not in text:
            raise SystemExit("JS style marker not found")
        styles = r'''    .liq-mini-wrap{margin-top:12px;padding:10px 10px 8px;border-radius:10px;background:rgba(4,8,14,.32);border:1px solid rgba(255,255,255,.055)}
    .liq-mini-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;color:var(--muted);font-size:9px;font-weight:800;letter-spacing:.10em}.liq-mini-title small{font-size:8px;opacity:.75}
    .liq-mini-map{height:78px;position:relative;overflow:hidden;border-radius:7px;background:linear-gradient(90deg,rgba(70,224,164,.025),rgba(255,255,255,.015) 49%,rgba(255,255,255,.015) 51%,rgba(255,103,120,.025));border:1px solid rgba(255,255,255,.04)}
    .liq-price-line{position:absolute;left:0;right:0;top:50%;height:1px;background:rgba(255,255,255,.55);z-index:3;box-shadow:0 0 8px rgba(255,255,255,.14)}
    .liq-price-dot{position:absolute;left:50%;top:50%;width:5px;height:5px;border-radius:50%;background:#fff;transform:translate(-50%,-50%);z-index:4;box-shadow:0 0 9px rgba(255,255,255,.55)}
    .liq-level{position:absolute;left:0;right:0;height:3px;display:flex;align-items:center;pointer-events:auto;transform:translateY(-50%)}
    .liq-level span{display:block;height:100%;border-radius:4px;box-shadow:0 0 7px currentColor}
    .liq-below{justify-content:flex-start;padding-right:51%;color:#4ce5a6}.liq-below span{margin-left:auto;background:linear-gradient(90deg,rgba(74,148,255,.20),#4ce5a6)}
    .liq-above{justify-content:flex-end;padding-left:51%;color:#ff7b7f}.liq-above span{margin-right:auto;background:linear-gradient(90deg,#ffca61,#ff6578)}
    .liq-mini-axis{display:flex;justify-content:space-between;align-items:center;margin-top:5px;font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}.liq-mini-axis strong{font-size:9px;color:var(--text);font-weight:750;letter-spacing:0}.liq-mini-empty{margin-top:10px;padding:12px;text-align:center;border-radius:8px;color:var(--muted);font-size:9px;background:rgba(255,255,255,.025)}
'''
        text = text.replace(style_marker, style_marker + styles, 1)

    if "${compactLiquidationHeatmap(item)}" not in text:
        # Put the heatmap after the metric stack and before the explanatory box.
        marker = '''      </div>\n      <div class="reason-box">${riskExplanation(item)}</div>'''
        if marker not in text:
            raise SystemExit("JS card insertion marker not found")
        text = text.replace(
            marker,
            '''      </div>\n      ${compactLiquidationHeatmap(item)}\n      <div class="reason-box">${riskExplanation(item)}</div>''',
            1,
        )

    if text != original:
        backup = JS.with_suffix(".js.before-compact-heatmap")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        JS.write_text(text, encoding="utf-8")
        print("Patched Matrix AI Radar card mini heatmap")
    else:
        print("Dashboard mini heatmap already installed")


def bump_cache() -> None:
    text = INDEX.read_text(encoding="utf-8")
    original = text
    text = re.sub(
        r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
        '/static/matrix_ai_radar.js?v=8',
        text,
    )
    if text != original:
        INDEX.write_text(text, encoding="utf-8")
        print("Bumped Matrix AI Radar JS asset to v=8")


def main() -> None:
    patch_api()
    patch_js()
    bump_cache()
    print("Compact liquidation heatmap installed. Restart API and hard-refresh browser.")


if __name__ == "__main__":
    main()
