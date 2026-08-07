#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "matrix_ai_radar.js"
INDEX = ROOT / "static" / "index.html"


def patch_js() -> None:
    text = JS.read_text(encoding="utf-8")
    original = text

    # Make mini heatmap clickable and self-identifying.
    text = text.replace(
        '<div class="liq-mini-wrap">\n    <div class="liq-mini-title">',
        '<div class="liq-mini-wrap liq-clickable" data-heatmap-symbol="${item.symbol}" role="button" tabindex="0" aria-label="Expand ${item.symbol} liquidation heatmap">\n    <div class="liq-mini-title"><span class="liq-expand-hint">Click to expand</span>',
        1,
    )

    if "function expandedLiquidationHeatmap(item)" not in text:
        marker = "function matrixTimeframeStrip(item) {"
        pos = text.find(marker)
        if pos < 0:
            raise SystemExit("matrixTimeframeStrip marker not found")

        modal_code = r'''function expandedLiquidationHeatmap(item) {
  const map = item?.liquidation_heatmap;
  if (!map?.available || !Array.isArray(map.levels) || !map.levels.length) {
    return '<div class="liq-modal-empty">Heatmap unavailable</div>';
  }

  const levels = map.levels.map((level) => {
    const intensity = Math.max(0, Math.min(1, Number(level.intensity || 0)));
    const position = Math.max(0, Math.min(1, Number(level.position ?? 0.5)));
    const top = (1 - position) * 100;
    const width = 12 + intensity * 88;
    const opacity = 0.22 + intensity * 0.78;
    const cls = level.side === "ABOVE" ? "liq-modal-above" : "liq-modal-below";
    const price = formatNumber(level.price, 6);
    const volume = formatNumber(level.total_volume, 0);
    return `<div class="liq-modal-level ${cls}" style="top:${top.toFixed(2)}%;opacity:${opacity.toFixed(2)}" title="${price} · ${volume}">
      <span style="width:${width.toFixed(1)}%"></span>
      <em>${price}</em>
    </div>`;
  }).join("");

  return `<div class="liq-modal-map">
    <div class="liq-modal-price-line"></div>
    <div class="liq-modal-price-label">${formatNumber(item.current_price, 6)}</div>
    ${levels}
  </div>
  <div class="liq-modal-axis"><span>Below price</span><strong>Current price</strong><span>Above price</span></div>`;
}

function ensureHeatmapModal() {
  if (document.getElementById("liqHeatmapModal")) return;
  const modal = document.createElement("div");
  modal.id = "liqHeatmapModal";
  modal.className = "liq-modal-backdrop hidden";
  modal.innerHTML = `
    <div class="liq-modal-card" role="dialog" aria-modal="true" aria-labelledby="liqHeatmapModalTitle">
      <div class="liq-modal-head">
        <div>
          <span>LIQUIDATION HEATMAP</span>
          <h3 id="liqHeatmapModalTitle">—</h3>
        </div>
        <button id="liqHeatmapModalClose" class="liq-modal-close" aria-label="Close heatmap">×</button>
      </div>
      <div class="liq-modal-meta">
        <span id="liqHeatmapModalTf">24H</span>
        <span id="liqHeatmapModalPrice">—</span>
      </div>
      <div id="liqHeatmapModalBody"></div>
      <div class="liq-modal-legend">
        <span class="legend-below">Below price liquidity</span>
        <span class="legend-above">Above price liquidity</span>
      </div>
    </div>`;
  document.body.appendChild(modal);

  const close = () => closeHeatmapModal();
  modal.addEventListener("click", (event) => {
    if (event.target === modal) close();
  });
  document.getElementById("liqHeatmapModalClose")?.addEventListener("click", close);
}

function openHeatmapModal(symbol) {
  ensureHeatmapModal();
  const item = (state.payload?.radar || []).find((row) => row.symbol === symbol);
  if (!item) return;
  const map = item.liquidation_heatmap || {};
  const modal = document.getElementById("liqHeatmapModal");
  document.getElementById("liqHeatmapModalTitle").textContent = `${item.symbol} · ${String(map.timeframe || "24h").toUpperCase()}`;
  document.getElementById("liqHeatmapModalTf").textContent = String(map.timeframe || "24h").toUpperCase();
  document.getElementById("liqHeatmapModalPrice").textContent = `Current ${formatNumber(item.current_price, 6)}`;
  document.getElementById("liqHeatmapModalBody").innerHTML = expandedLiquidationHeatmap(item);
  modal.classList.remove("hidden");
  document.body.classList.add("liq-modal-open");
}

function closeHeatmapModal() {
  const modal = document.getElementById("liqHeatmapModal");
  if (!modal) return;
  modal.classList.add("hidden");
  document.body.classList.remove("liq-modal-open");
}

function installHeatmapExpandHandlers() {
  ensureHeatmapModal();
  document.addEventListener("click", (event) => {
    const target = event.target.closest?.("[data-heatmap-symbol]");
    if (!target) return;
    openHeatmapModal(target.dataset.heatmapSymbol);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeHeatmapModal();
    if ((event.key === "Enter" || event.key === " ") && event.target?.matches?.("[data-heatmap-symbol]")) {
      event.preventDefault();
      openHeatmapModal(event.target.dataset.heatmapSymbol);
    }
  });
}

'''
        text = text[:pos] + modal_code + text[pos:]

    if ".liq-modal-backdrop" not in text:
        style_marker = "    .liq-mini-axis{display:flex;justify-content:space-between;align-items:center;margin-top:5px;font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}.liq-mini-axis strong{font-size:9px;color:var(--text);font-weight:750;letter-spacing:0}.liq-mini-empty{margin-top:10px;padding:12px;text-align:center;border-radius:8px;color:var(--muted);font-size:9px;background:rgba(255,255,255,.025)}\n"
        if style_marker not in text:
            raise SystemExit("mini heatmap style marker not found")
        styles = r'''    .liq-clickable{cursor:zoom-in;transition:border-color .18s ease,transform .18s ease}.liq-clickable:hover{border-color:rgba(84,200,255,.28);transform:translateY(-1px)}
    .liq-mini-title{position:relative}.liq-expand-hint{position:absolute;right:28px;top:0;font-size:7px!important;font-weight:650!important;letter-spacing:.04em!important;opacity:0;transition:opacity .18s ease;text-transform:none!important}.liq-clickable:hover .liq-expand-hint{opacity:.75}
    body.liq-modal-open{overflow:hidden}.liq-modal-backdrop{position:fixed;inset:0;z-index:9999;background:rgba(1,4,9,.78);backdrop-filter:blur(8px);display:flex;align-items:center;justify-content:center;padding:24px}.liq-modal-backdrop.hidden{display:none}
    .liq-modal-card{width:min(900px,94vw);max-height:90vh;overflow:auto;border-radius:16px;background:linear-gradient(180deg,rgba(17,23,34,.99),rgba(9,14,22,.99));border:1px solid rgba(255,255,255,.12);box-shadow:0 28px 90px rgba(0,0,0,.58);padding:18px}
    .liq-modal-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:10px}.liq-modal-head span{font-size:9px;color:#54c8ff;font-weight:800;letter-spacing:.13em}.liq-modal-head h3{margin:5px 0 0;font-size:21px;letter-spacing:-.02em}.liq-modal-close{width:34px;height:34px;border-radius:9px;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.035);color:var(--text);font-size:22px;line-height:1;cursor:pointer}.liq-modal-close:hover{background:rgba(255,255,255,.08)}
    .liq-modal-meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}.liq-modal-meta span:last-child{color:var(--text);font-weight:750;letter-spacing:0;text-transform:none}
    .liq-modal-map{height:min(520px,58vh);min-height:340px;position:relative;overflow:hidden;border-radius:12px;background:linear-gradient(90deg,rgba(76,229,166,.03),rgba(255,255,255,.012) 49%,rgba(255,255,255,.012) 51%,rgba(255,101,120,.03));border:1px solid rgba(255,255,255,.07)}
    .liq-modal-price-line{position:absolute;left:0;right:0;top:50%;height:1px;background:rgba(255,255,255,.75);z-index:5;box-shadow:0 0 12px rgba(255,255,255,.2)}.liq-modal-price-label{position:absolute;left:50%;top:50%;z-index:7;transform:translate(-50%,-50%);padding:4px 7px;border-radius:6px;background:#eef4ff;color:#111827;font-size:10px;font-weight:850;box-shadow:0 4px 14px rgba(0,0,0,.35)}
    .liq-modal-level{position:absolute;left:0;right:0;height:7px;display:flex;align-items:center;transform:translateY(-50%)}.liq-modal-level span{display:block;height:100%;border-radius:5px;box-shadow:0 0 13px currentColor}.liq-modal-level em{position:absolute;font-style:normal;font-size:9px;color:rgba(230,236,247,.72);opacity:.84}
    .liq-modal-below{justify-content:flex-start;padding-right:50.8%;color:#4ce5a6}.liq-modal-below span{margin-left:auto;background:linear-gradient(90deg,rgba(77,107,255,.28),#4ce5a6)}.liq-modal-below em{right:51.5%}
    .liq-modal-above{justify-content:flex-end;padding-left:50.8%;color:#ff697a}.liq-modal-above span{margin-right:auto;background:linear-gradient(90deg,#ffcb61,#ff5f78)}.liq-modal-above em{left:51.5%}
    .liq-modal-axis{display:flex;justify-content:space-between;align-items:center;margin:8px 2px 0;font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}.liq-modal-axis strong{color:var(--text);font-size:9px}.liq-modal-legend{display:flex;justify-content:space-between;gap:12px;margin-top:14px;font-size:9px;color:var(--muted)}.legend-below:before,.legend-above:before{content:"";display:inline-block;width:22px;height:4px;border-radius:3px;margin-right:7px;vertical-align:middle}.legend-below:before{background:linear-gradient(90deg,#4d6bff,#4ce5a6)}.legend-above:before{background:linear-gradient(90deg,#ffcb61,#ff5f78)}.liq-modal-empty{padding:80px 20px;text-align:center;color:var(--muted)}
    @media(max-width:700px){.liq-modal-backdrop{padding:10px}.liq-modal-card{width:100%;padding:13px}.liq-modal-map{height:62vh;min-height:300px}.liq-modal-level em{display:none}}
'''
        text = text.replace(style_marker, style_marker + styles, 1)

    if "installHeatmapExpandHandlers();" not in text:
        marker = "installRiskStyles();\n"
        if marker not in text:
            raise SystemExit("startup marker not found")
        text = text.replace(marker, marker + "installHeatmapExpandHandlers();\n", 1)

    if text != original:
        backup = JS.with_suffix(".js.before-heatmap-modal-v1")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        JS.write_text(text, encoding="utf-8")
        print("Patched expandable liquidation heatmap modal V1")
    else:
        print("Heatmap modal V1 already installed")


def bump_cache() -> None:
    text = INDEX.read_text(encoding="utf-8")
    original = text
    text = re.sub(
        r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
        '/static/matrix_ai_radar.js?v=9',
        text,
    )
    if text != original:
        INDEX.write_text(text, encoding="utf-8")
        print("Bumped Matrix AI Radar JS asset to v=9")
    else:
        print("Asset cache version already v=9")


def main() -> None:
    patch_js()
    bump_cache()
    print("Heatmap expand modal V1 installed. Hard-refresh browser; API restart is not required for this frontend-only change.")


if __name__ == "__main__":
    main()
