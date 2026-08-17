const state = { payload: null, seconds: 15, timer: null };
const $ = (id) => document.getElementById(id);

function formatNumber(value, maximumFractionDigits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString("en-US", { maximumFractionDigits });
}
function formatAge(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${Math.round(seconds)} sec`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}
function formatTime(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function humanize(value, fallback = "—") {
  const normalized = String(value || "").trim();
  return normalized ? normalized.replaceAll("_", " ") : fallback;
}
function matrixData(item) {
  return item.matrix && item.matrix.available !== false ? item.matrix : null;
}
function matrixAlignment(item) {
  const v = matrixData(item)?.alignment_score;
  return v === null || v === undefined ? null : Number(v);
}
function scalpIdea(item) {
  const label = matrixData(item)?.timeframes?.["1m"]?.trend_label;
  if (label === "BUY") return "LONG";
  if (label === "SELL") return "SHORT";
  return "N/A";
}
function riskData(item) {
  return item.ai_market_risk?.available ? item.ai_market_risk : null;
}
function heatScore(item) {
  const risk = riskData(item);
  return risk ? Number(risk.risk_score) : null;
}
function rawRiskBand(item) {
  return riskData(item)?.risk_band || "N/A";
}
function heatBand(item) {
  const band = rawRiskBand(item);
  if (band === "LOW RISK") return "LOW HEAT";
  if (band === "MEDIUM RISK") return "MEDIUM HEAT";
  if (band === "HIGH RISK") return "HIGH HEAT";
  if (band === "EXTREME RISK") return "EXTREME HEAT";
  return "N/A";
}
function heatOrder(item) {
  const score = heatScore(item);
  return score === null ? 999 : score;
}
function heatClass(item) {
  return rawRiskBand(item).toLowerCase().replaceAll(" ", "-").replace("/", "-");
}
function liquidationPressureDirection(item) {
  const raw = String(item.raw_prediction || item.prediction || "").toUpperCase();
  if (raw === "SHORT_SQUEEZE") return "UP";
  if (raw === "LONG_SQUEEZE") return "DOWN";
  return "N/A";
}
function liquidationPressureHtml(item) {
  const value = formatNumber(liquidityPressure(item), 0);
  const direction = liquidationPressureDirection(item);
  if (direction === "UP") return `<span class="liq-pressure liq-pressure-up"><b>↑</b><strong>${value}</strong></span>`;
  if (direction === "DOWN") return `<span class="liq-pressure liq-pressure-down"><b>↓</b><strong>${value}</strong></span>`;
  return `<span class="liq-pressure liq-pressure-na"><strong>${value}</strong></span>`;
}

function liquidityPressure(item) {
  if (item.liquidity_pressure_score !== undefined) return Number(item.liquidity_pressure_score);
  if (item.liquidity_pressure !== undefined) return Number(item.liquidity_pressure) * 100;
  return Number(item.score || 0) * 100;
}

function compactLiquidationHeatmap(item) {
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
  return `<div class="liq-mini-wrap liq-clickable" data-heatmap-symbol="${item.symbol}" role="button" tabindex="0" aria-label="Expand ${item.symbol} liquidation heatmap">
    <div class="liq-mini-title"><span class="liq-expand-hint">Click to expand</span><span>LIQUIDATION HEATMAP</span><small>${String(map.timeframe || "24h").toUpperCase()}</small></div>
    <div class="liq-mini-map">
      <div class="liq-price-line"></div>
      <div class="liq-price-dot"></div>
      ${levels}
    </div>
    <div class="liq-mini-axis"><span>below price</span><strong>${formatNumber(item.current_price, 6)}</strong><span>above price</span></div>
  </div>`;
}

function expandedLiquidationHeatmap(item) {
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
      // Keep a protected zone around current price so the current-price badge
      // can never collide with a liquidation price label.
      .filter((level) => Math.abs(level.position - 0.5) >= 0.065)
      .sort((a, b) => b.intensity - a.intensity);

    const accepted = [];
    for (const level of candidates) {
      // Roughly 4.2% of modal height separation prevents label collisions.
      if (accepted.every((other) => Math.abs(other.position - level.position) >= 0.055)) {
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
    ${levels}
  </div>
  <div class="liq-modal-axis"><span>Below current price</span><strong>Current ${formatNumber(item.current_price, 6)}</strong><span>Above current price</span></div>`;
}


function resolveHeatmapModalLabelCollisions() {
  const modal = document.getElementById("liqHeatmapModal");
  if (!modal || modal.classList.contains("hidden")) return;

  // Never show the duplicate PRICE LEVELS helper labels.
  modal.querySelectorAll(".liq-modal-side-label").forEach((el) => {
    el.style.setProperty("display", "none", "important");
  });

  const map = modal.querySelector(".liq-modal-map");
  const current = modal.querySelector(".liq-modal-current-badge, .liq-modal-price-label");
  if (!map) return;

  const mapRect = map.getBoundingClientRect();
  const currentRect = current?.getBoundingClientRect() || null;
  const centerY = mapRect.top + mapRect.height / 2;
  const centerGuardPx = 34;
  const minGapPx = 25;

  const allLabels = Array.from(
    modal.querySelectorAll(".liq-modal-price-tag, .liq-modal-level em")
  );

  // Reset first, because the modal can be reopened after a resize/refresh.
  allLabels.forEach((el) => {
    el.style.removeProperty("display");
    el.style.removeProperty("visibility");
  });

  const intersects = (a, b, pad = 0) => !(
    a.right + pad < b.left ||
    a.left - pad > b.right ||
    a.bottom + pad < b.top ||
    a.top - pad > b.bottom
  );

  // First remove anything in the current-price guard zone or touching current badge.
  allLabels.forEach((el) => {
    const r = el.getBoundingClientRect();
    const y = r.top + r.height / 2;
    if (Math.abs(y - centerY) < centerGuardPx || (currentRect && intersects(r, currentRect, 8))) {
      el.style.setProperty("display", "none", "important");
    }
  });

  // Then resolve collisions independently on each half. Stronger labels occur first in DOM
  // often enough, but sorting by visual Y makes the result deterministic.
  ["below", "above"].forEach((side) => {
    const selector = side === "below"
      ? ".liq-modal-below .liq-modal-price-tag, .liq-modal-below em"
      : ".liq-modal-above .liq-modal-price-tag, .liq-modal-above em";
    const labels = Array.from(modal.querySelectorAll(selector))
      .filter((el) => getComputedStyle(el).display !== "none")
      .sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);

    let lastRect = null;
    for (const el of labels) {
      const r = el.getBoundingClientRect();
      if (lastRect && r.top < lastRect.bottom + minGapPx) {
        el.style.setProperty("display", "none", "important");
        continue;
      }
      lastRect = r;
    }
  });
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
  requestAnimationFrame(resolveHeatmapModalLabelCollisions);
  modal.classList.remove("hidden");
  document.body.classList.add("liq-modal-open");
}

window.addEventListener("resize", () => requestAnimationFrame(resolveHeatmapModalLabelCollisions));

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

function matrixTimeframeStrip(item) {
  const matrix = matrixData(item);
  if (!matrix?.timeframes) return '<span class="matrix-unavailable">Matrix unavailable</span>';
  return ["1d", "4h", "1h", "15m", "1m"].map((timeframe) => {
    const label = matrix.timeframes[timeframe]?.trend_label || "—";
    const cls = label === "BUY" ? "matrix-buy" : label === "SELL" ? "matrix-sell" : "matrix-neutral";
    return `<span class="matrix-tf ${cls}"><small>${timeframe.toUpperCase()}</small><strong>${label}</strong></span>`;
  }).join("");
}

function heatExplanation(item) {
  const risk = riskData(item);
  if (!risk) {
    if (item.ai_market_risk?.reason === "SYMBOL_NOT_TRAINED") {
      return "Near-Term Market Heat is not trained for this market yet. Matrix scalp idea remains available.";
    }
    return "Near-Term Market Heat model is temporarily unavailable. Matrix scalp idea remains available.";
  }
  const high = Math.round(Number(risk.p_high || 0) * 100);
  const extreme = Math.round(Number(risk.p_extreme || 0) * 100);
  if (risk.risk_band === "LOW RISK") {
    return `Near-term market activity is comparatively calm. High-movement probability ${high}%, extreme-movement probability ${extreme}%.`;
  }
  if (risk.risk_band === "MEDIUM RISK") {
    return `Near-term market activity is elevated. High-movement probability ${high}%, extreme-movement probability ${extreme}%.`;
  }
  if (risk.risk_band === "HIGH RISK") {
    return `Near-term market movement potential is high. High-movement probability ${high}%, extreme-movement probability ${extreme}%.`;
  }
  return `Extreme near-term movement potential is elevated. High-movement probability ${high}%, extreme-movement probability ${extreme}%.`;
}

function installHeatStyles() {
  if (document.getElementById("matrixAiHeatStyles")) return;
  const style = document.createElement("style");
  style.id = "matrixAiHeatStyles";
  style.textContent = `
    .risk-band-pill{display:inline-flex;align-items:center;padding:6px 9px;border-radius:999px;font-size:10px;font-weight:800;letter-spacing:.08em;white-space:nowrap}
    .risk-low-risk{color:#4ce5a6;background:rgba(76,229,166,.10);border:1px solid rgba(76,229,166,.18)}
    .risk-medium-risk{color:#ffca61;background:rgba(255,202,97,.10);border:1px solid rgba(255,202,97,.18)}
    .risk-high-risk{color:#ff9b68;background:rgba(255,155,104,.11);border:1px solid rgba(255,155,104,.20)}
    .risk-extreme-risk{color:#ff6f7d;background:rgba(255,111,125,.12);border:1px solid rgba(255,111,125,.22)}
    .risk-n-a{color:var(--muted);background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)}
    .ai-risk-hero{margin:16px 0;padding:14px;border-radius:12px;border:1px solid rgba(255,255,255,.075);background:rgba(255,255,255,.03)}
    .ai-risk-heading{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:8px}
    .ai-risk-heading span:first-child{font-size:10px;color:var(--muted);font-weight:750;letter-spacing:.12em;text-transform:uppercase}
    .ai-risk-score{display:flex;align-items:baseline;gap:6px}.ai-risk-score strong{font-size:38px;letter-spacing:-.04em}.ai-risk-score span{color:var(--muted)}
    .ai-risk-sub{font-size:10px;color:var(--muted);margin-top:5px;text-transform:uppercase;letter-spacing:.09em}
    .matrix-direction-hero{margin-top:12px;display:flex;justify-content:space-between;align-items:center;padding:11px 12px;border-radius:10px;background:rgba(255,255,255,.025)}
    .matrix-direction-hero span{color:var(--muted);font-size:11px}.matrix-direction-hero strong{font-size:18px;letter-spacing:.04em}
    .scalp-long{color:#4ce5a6}.scalp-short{color:#ff6f7d}.scalp-na{color:var(--muted)}
    .liq-pressure{display:inline-flex;align-items:center;justify-content:flex-end;gap:7px;font-variant-numeric:tabular-nums}
    .liq-pressure b{font-size:22px;line-height:.75;font-weight:950;text-shadow:0 0 12px currentColor}
    .liq-pressure strong{font-size:12px;font-weight:900;color:inherit}
    .liq-pressure-up{color:#4ce5a6}
    .liq-pressure-down{color:#ff6f7d}
    .liq-pressure-na{color:var(--muted)}
    .liq-mini-wrap{margin-top:12px;padding:10px 10px 8px;border-radius:10px;background:rgba(4,8,14,.32);border:1px solid rgba(255,255,255,.055)}
    .liq-mini-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;color:var(--muted);font-size:9px;font-weight:800;letter-spacing:.10em}.liq-mini-title small{font-size:8px;opacity:.75}
    .liq-mini-map{height:78px;position:relative;overflow:hidden;border-radius:7px;background:linear-gradient(90deg,rgba(70,224,164,.025),rgba(255,255,255,.015) 49%,rgba(255,255,255,.015) 51%,rgba(255,103,120,.025));border:1px solid rgba(255,255,255,.04)}
    .liq-price-line{position:absolute;left:0;right:0;top:50%;height:1px;background:rgba(255,255,255,.55);z-index:3;box-shadow:0 0 8px rgba(255,255,255,.14)}
    .liq-price-dot{position:absolute;left:50%;top:50%;width:5px;height:5px;border-radius:50%;background:#fff;transform:translate(-50%,-50%);z-index:4;box-shadow:0 0 9px rgba(255,255,255,.55)}
    .liq-level{position:absolute;left:0;right:0;height:3px;display:flex;align-items:center;pointer-events:auto;transform:translateY(-50%)}
    .liq-level span{display:block;height:100%;border-radius:4px;box-shadow:0 0 7px currentColor}
    .liq-below{justify-content:flex-start;padding-right:51%;color:#4ce5a6}.liq-below span{margin-left:auto;background:linear-gradient(90deg,rgba(74,148,255,.20),#4ce5a6)}
    .liq-above{justify-content:flex-end;padding-left:51%;color:#ff7b7f}.liq-above span{margin-right:auto;background:linear-gradient(90deg,#ffca61,#ff6578)}
    .liq-mini-axis{display:flex;justify-content:space-between;align-items:center;margin-top:5px;font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}.liq-mini-axis strong{font-size:9px;color:var(--text);font-weight:750;letter-spacing:0}.liq-mini-empty{margin-top:10px;padding:12px;text-align:center;border-radius:8px;color:var(--muted);font-size:9px;background:rgba(255,255,255,.025)}
    .liq-clickable{cursor:zoom-in;transition:border-color .18s ease,transform .18s ease}.liq-clickable:hover{border-color:rgba(84,200,255,.28);transform:translateY(-1px)}
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

    .liq-modal-price-tag{position:absolute!important;z-index:8!important;display:inline-flex!important;align-items:center!important;min-height:20px!important;padding:2px 6px!important;border-radius:5px!important;background:rgba(9,14,22,.90)!important;border:1px solid rgba(255,255,255,.13)!important;color:#f4f7fb!important;font-size:10px!important;font-weight:800!important;font-style:normal!important;line-height:1!important;opacity:1!important;box-shadow:0 3px 10px rgba(0,0,0,.35)!important;white-space:nowrap!important}
    .liq-modal-below .liq-modal-price-tag{right:52%!important;transform:translateX(-8px)!important}
    .liq-modal-above .liq-modal-price-tag{left:52%!important;transform:translateX(8px)!important}
    .liq-modal-current-badge{position:absolute;left:50%;top:50%;z-index:12;transform:translate(-50%,-50%);padding:5px 9px;border-radius:7px;background:#f1f5fb;color:#101722;font-size:11px;font-weight:900;box-shadow:0 4px 16px rgba(0,0,0,.45);white-space:nowrap}
    .liq-modal-side-label{position:absolute;top:10px;z-index:6;color:rgba(220,229,241,.42);font-size:8px;font-weight:800;letter-spacing:.12em;pointer-events:none}
    .liq-modal-side-label-below{right:52%}.liq-modal-side-label-above{left:52%}
    .liq-modal-strong span{filter:brightness(1.18);box-shadow:0 0 16px currentColor,0 0 28px currentColor}
    @media(max-width:700px){.liq-modal-price-tag{font-size:9px!important;padding:2px 4px!important}.liq-modal-side-label{display:none}}

    .liq-modal-side-label{display:none!important}
  `;
  document.head.appendChild(style);
}

function cardGlow(item) {
  const band = rawRiskBand(item);
  if (band === "EXTREME RISK") return "rgba(255,111,125,.23)";
  if (band === "HIGH RISK") return "rgba(255,155,104,.19)";
  if (band === "MEDIUM RISK") return "rgba(255,202,97,.14)";
  if (band === "LOW RISK") return "rgba(76,229,166,.15)";
  return scalpIdea(item) === "SHORT" ? "rgba(255,111,125,.10)" : "rgba(84,200,255,.09)";
}

function renderCard(item) {
  const matrix = matrixData(item);
  const alignment = matrixAlignment(item);
  const score = heatScore(item);
  const band = heatBand(item);
  const idea = scalpIdea(item);
  const ideaCls = idea === "LONG" ? "scalp-long" : idea === "SHORT" ? "scalp-short" : "scalp-na";
  return `
    <article class="radar-card" style="--glow:${cardGlow(item)}">
      <div class="card-head"><span class="rank">#${item.rank}</span></div>
      <h3 class="symbol">${item.symbol}</h3>
      <div class="price">${formatNumber(item.current_price, 6)}</div>
      <div class="ai-risk-hero">
        <div class="matrix-direction-hero" style="margin-top:0;margin-bottom:12px"><span>Scalp idea · 1M Matrix</span><strong class="${ideaCls}">${idea}</strong></div>
        <div class="ai-risk-heading"><span>NEAR-TERM MARKET HEAT</span><span class="risk-band-pill risk-${heatClass(item)}">${band}</span></div>
        <div class="ai-risk-score"><strong>${score === null ? "—" : Math.round(score)}</strong><span>${score === null ? "" : "/ 100"}</span></div>
        <div class="ai-risk-sub">15m movement percentile vs historical calibration</div>
      </div>
      <div class="matrix-strip">${matrixTimeframeStrip(item)}</div>
      <div class="metric-stack">
        <div class="metric-row"><span>Matrix alignment</span><strong>${alignment === null ? "—" : `${formatNumber(alignment, 1)}%`}</strong></div>
        <div class="metric-row"><span>Liquidation Pressure</span>${liquidationPressureHtml(item)}</div>
        <div class="metric-row"><span>Heat horizon</span><strong>${riskData(item) ? "NEXT 15 MIN" : "—"}</strong></div>
      </div>
      ${compactLiquidationHeatmap(item)}
      <div class="reason-box">${heatExplanation(item)}</div>
      <div class="card-footer"><span>${matrix ? humanize(matrix.regime) : "Matrix unavailable"}</span><span>${formatAge(item.age_seconds)}</span></div>
    </article>`;
}

function renderTableRow(item) {
  const alignment = matrixAlignment(item);
  const score = heatScore(item);
  const idea = scalpIdea(item);
  const ideaCls = idea === "LONG" ? "scalp-long" : idea === "SHORT" ? "scalp-short" : "scalp-na";
  return `<tr>
    <td>#${item.rank}</td>
    <td><strong>${item.symbol}</strong></td>
    <td class="${ideaCls}"><strong>${idea}</strong></td>
    <td>${score === null ? "—" : Math.round(score)}</td>
    <td><span class="risk-band-pill risk-${heatClass(item)}">${heatBand(item)}</span></td>
    <td>${alignment === null ? "—" : `${formatNumber(alignment, 1)}%`}</td>
    <td>${liquidationPressureHtml(item)}</td>
    <td>${formatNumber(item.current_price, 6)}</td>
    <td>${formatAge(item.age_seconds)}</td>
  </tr>`;
}

function normalizeHeadings() {
  const subtitle = document.querySelector('.brand-row p');
  if (subtitle) subtitle.textContent = '1M Matrix Scalp Idea + Near-Term Market Heat';
  const heading = document.querySelector('.section-heading h2');
  if (heading) heading.textContent = 'Matrix scalp idea with near-term Market Heat';
  document.querySelectorAll('th').forEach((th) => {
    const t = th.textContent.trim().toLowerCase();
    if (t === 'matrix direction') th.textContent = 'Scalp idea';
    if (t === '15m risk score') th.textContent = 'Market Heat Score';
    if (t === 'ai risk') th.textContent = 'Heat Level';
  });
}

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function render(payload) {
  state.payload = payload;
  const radar = [...(payload.radar || [])].sort((a, b) => heatOrder(a) - heatOrder(b));
  radar.forEach((item, index) => { item.rank = index + 1; });
  const scored = radar.filter((item) => heatScore(item) !== null);
  const lowest = scored[0];
  const lowHeatCount = scored.filter((item) => rawRiskBand(item) === "LOW RISK").length;

  setText("engineStatus", payload.status || "UNKNOWN");
  const engineDot = $("engineDot");
  if (engineDot) engineDot.className = `status-dot ${(payload.status || "offline").toLowerCase()}`;
  setText("lastRefresh", formatTime(payload.generated_at));
  setText("symbolCount", payload.symbol_count ?? radar.length);
  setText("freshestAge", formatAge(payload.freshest_snapshot_age_seconds));
  setText("highestRisk", lowest ? formatNumber(heatScore(lowest), 2) : "—");
  setText("highestSymbol", lowest ? `${lowest.symbol} currently has the lowest Market Heat` : "Waiting for Market Heat");
  setText("activeWatches", lowHeatCount);
  if ($("radarCards")) $("radarCards").innerHTML = radar.length ? radar.map(renderCard).join("") : '<div class="loading-card">No live radar data available.</div>';
  if ($("radarTable")) $("radarTable").innerHTML = radar.map(renderTableRow).join("");
  if ($("rawJson")) $("rawJson").textContent = JSON.stringify(payload, null, 2);
}

async function loadRadar(force = false) {
  const button = $("refreshButton");
  try {
    if (button) {
      button.disabled = true;
      button.textContent = force ? "Refreshing…" : "Loading…";
    }
    const response = await fetch(force ? "/radar/refresh" : "/radar", force ? { method: "POST" } : {});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    setText("engineStatus", "OFFLINE");
    const engineDot = $("engineDot");
    if (engineDot) engineDot.className = "status-dot offline";
    if ($("radarCards")) $("radarCards").innerHTML = `<div class="loading-card">Radar unavailable: ${error.message}</div>`;
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Refresh now";
    }
    state.seconds = 15;
  }
}
function startCountdown() {
  clearInterval(state.timer);
  state.timer = setInterval(() => {
    state.seconds -= 1;
    setText("countdown", `Auto refresh in ${Math.max(state.seconds, 0)}s`);
    if (state.seconds <= 0) { state.seconds = 15; loadRadar(false); }
  }, 1000);
}

if ($("refreshButton")) $("refreshButton").addEventListener("click", () => loadRadar(true));
if ($("jsonToggle")) $("jsonToggle").addEventListener("click", () => $("jsonPanel")?.classList.toggle("hidden"));
if ($("jsonClose")) $("jsonClose").addEventListener("click", () => $("jsonPanel")?.classList.add("hidden"));
installHeatStyles();
installHeatmapExpandHandlers();
normalizeHeadings();
loadRadar(false);
startCountdown();
