#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "matrix_ai_radar.js"
INDEX = ROOT / "static" / "index.html"

NEW_FUNCTION = r'''function expandedLiquidationHeatmap(item) {
  const map = item?.liquidation_heatmap;
  if (!map?.available || !Array.isArray(map.levels) || !map.levels.length) {
    return '<div class="liq-modal-empty">Heatmap unavailable</div>';
  }

  // Keep labels readable: show all bars, but only label the strongest / well-separated levels.
  const rawLevels = map.levels.map((level, index) => {
    const intensity = Math.max(0, Math.min(1, Number(level.intensity || 0)));
    const position = Math.max(0, Math.min(1, Number(level.position ?? 0.5)));
    return { ...level, intensity, position, index };
  });

  const labelled = new Set();
  ["BELOW", "ABOVE"].forEach((side) => {
    const candidates = rawLevels
      .filter((level) => level.side === side)
      .sort((a, b) => b.intensity - a.intensity);

    const accepted = [];
    for (const level of candidates) {
      // Roughly 4.2% of modal height separation prevents label collisions.
      if (accepted.every((other) => Math.abs(other.position - level.position) >= 0.042)) {
        accepted.push(level);
        labelled.add(level.index);
      }
      if (accepted.length >= 10) break;
    }
  });

  const levels = rawLevels.map((level) => {
    const top = (1 - level.position) * 100;
    const width = 12 + level.intensity * 88;
    const opacity = 0.22 + level.intensity * 0.78;
    const cls = level.side === "ABOVE" ? "liq-modal-above" : "liq-modal-below";
    const price = formatNumber(level.price, 6);
    const volume = formatNumber(level.total_volume, 0);
    const strong = level.intensity >= 0.72 ? " liq-modal-strong" : "";
    const label = labelled.has(level.index)
      ? `<em class="liq-modal-price-tag">${price}</em>`
      : "";

    return `<div class="liq-modal-level ${cls}${strong}" style="top:${top.toFixed(2)}%;opacity:${opacity.toFixed(2)}" title="${price} · ${volume}">
      <span style="width:${width.toFixed(1)}%"></span>
      ${label}
    </div>`;
  }).join("");

  return `<div class="liq-modal-map">
    <div class="liq-modal-price-line"></div>
    <div class="liq-modal-current-badge">${formatNumber(item.current_price, 6)}</div>
    <div class="liq-modal-side-label liq-modal-side-label-below">PRICE LEVELS</div>
    <div class="liq-modal-side-label liq-modal-side-label-above">PRICE LEVELS</div>
    ${levels}
  </div>
  <div class="liq-modal-axis"><span>Below current price</span><strong>Current ${formatNumber(item.current_price, 6)}</strong><span>Above current price</span></div>`;
}'''

EXTRA_STYLES = r'''
    .liq-modal-price-tag{position:absolute!important;z-index:8!important;display:inline-flex!important;align-items:center!important;min-height:20px!important;padding:2px 6px!important;border-radius:5px!important;background:rgba(9,14,22,.90)!important;border:1px solid rgba(255,255,255,.13)!important;color:#f4f7fb!important;font-size:10px!important;font-weight:800!important;font-style:normal!important;line-height:1!important;opacity:1!important;box-shadow:0 3px 10px rgba(0,0,0,.35)!important;white-space:nowrap!important}
    .liq-modal-below .liq-modal-price-tag{right:51.1%!important;transform:translateX(-6px)!important}
    .liq-modal-above .liq-modal-price-tag{left:51.1%!important;transform:translateX(6px)!important}
    .liq-modal-current-badge{position:absolute;left:50%;top:50%;z-index:12;transform:translate(-50%,-50%);padding:5px 9px;border-radius:7px;background:#f1f5fb;color:#101722;font-size:11px;font-weight:900;box-shadow:0 4px 16px rgba(0,0,0,.45);white-space:nowrap}
    .liq-modal-side-label{position:absolute;top:10px;z-index:6;color:rgba(220,229,241,.42);font-size:8px;font-weight:800;letter-spacing:.12em;pointer-events:none}
    .liq-modal-side-label-below{right:52%}.liq-modal-side-label-above{left:52%}
    .liq-modal-strong span{filter:brightness(1.18);box-shadow:0 0 16px currentColor,0 0 28px currentColor}
    @media(max-width:700px){.liq-modal-price-tag{font-size:9px!important;padding:2px 4px!important}.liq-modal-side-label{display:none}}
'''


def patch_js() -> None:
    text = JS.read_text(encoding="utf-8")
    original = text

    pattern = re.compile(
        r'function expandedLiquidationHeatmap\(item\) \{.*?\n\}\n\nfunction ensureHeatmapModal\(\)',
        re.S,
    )
    match = pattern.search(text)
    if not match:
        raise SystemExit("expandedLiquidationHeatmap() not found. Install heatmap modal V1 first.")

    replacement = NEW_FUNCTION + "\n\nfunction ensureHeatmapModal()"
    text = pattern.sub(replacement, text, count=1)

    if ".liq-modal-price-tag{" not in text:
        marker = "  `;\n  document.head.appendChild(style);"
        pos = text.find(marker)
        if pos < 0:
            raise SystemExit("Risk style block end marker not found")
        # Inject into the template literal immediately before it closes.
        text = text[:pos] + EXTRA_STYLES + text[pos:]

    if text != original:
        backup = JS.with_suffix(".js.before-modal-price-labels-v1")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        JS.write_text(text, encoding="utf-8")
        print("Patched heatmap modal with readable price labels")
    else:
        print("Heatmap modal price labels already patched")


def bump_cache() -> None:
    text = INDEX.read_text(encoding="utf-8")
    original = text
    text = re.sub(
        r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
        '/static/matrix_ai_radar.js?v=10',
        text,
    )
    if text != original:
        INDEX.write_text(text, encoding="utf-8")
        print("Bumped Matrix AI Radar JS asset to v=10")
    else:
        print("Asset cache version already v=10")


def main() -> None:
    patch_js()
    bump_cache()
    print("Done. Hard-refresh browser; API restart is not required.")


if __name__ == "__main__":
    main()
