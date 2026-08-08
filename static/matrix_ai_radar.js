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
function liquidityPressure(item) {
  if (item.liquidity_pressure_score !== undefined) return Number(item.liquidity_pressure_score);
  if (item.liquidity_pressure !== undefined) return Number(item.liquidity_pressure) * 100;
  return Number(item.score || 0) * 100;
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
      return "15m Market Heat is not trained for this market yet. Matrix scalp idea remains available.";
    }
    return "15m Market Heat model is temporarily unavailable. Matrix scalp idea remains available.";
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
      <div class="card-head"><span class="rank">#${item.rank}</span><span class="risk-band-pill risk-${heatClass(item)}">${band}</span></div>
      <h3 class="symbol">${item.symbol}</h3>
      <div class="price">${formatNumber(item.current_price, 6)}</div>
      <div class="ai-risk-hero">
        <div class="matrix-direction-hero" style="margin-top:0;margin-bottom:12px"><span>Scalp idea · 1M Matrix</span><strong class="${ideaCls}">${idea}</strong></div>
        <div class="ai-risk-heading"><span>15M MARKET HEAT</span><span class="risk-band-pill risk-${heatClass(item)}">${band}</span></div>
        <div class="ai-risk-score"><strong>${score === null ? "—" : formatNumber(score, 2)}</strong><span>${score === null ? "" : "/ 100"}</span></div>
        <div class="ai-risk-sub">Direction-independent near-term movement potential</div>
      </div>
      <div class="matrix-strip">${matrixTimeframeStrip(item)}</div>
      <div class="metric-stack">
        <div class="metric-row"><span>Matrix alignment</span><strong>${alignment === null ? "—" : `${formatNumber(alignment, 1)}%`}</strong></div>
        <div class="metric-row"><span>Liquidity pressure</span><strong>${formatNumber(liquidityPressure(item), 2)}</strong></div>
        <div class="metric-row"><span>Heat horizon</span><strong>${riskData(item) ? "NEXT 15 MIN" : "—"}</strong></div>
      </div>
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
    <td>${score === null ? "—" : formatNumber(score, 2)}</td>
    <td><span class="risk-band-pill risk-${heatClass(item)}">${heatBand(item)}</span></td>
    <td>${alignment === null ? "—" : `${formatNumber(alignment, 1)}%`}</td>
    <td>${formatNumber(liquidityPressure(item), 2)}</td>
    <td>${formatNumber(item.current_price, 6)}</td>
    <td>${formatAge(item.age_seconds)}</td>
  </tr>`;
}

function normalizeHeadings() {
  const subtitle = document.querySelector('.brand-row p');
  if (subtitle) subtitle.textContent = '1M Matrix Scalp Idea + 15m Market Heat';
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
normalizeHeadings();
loadRadar(false);
startCountdown();
