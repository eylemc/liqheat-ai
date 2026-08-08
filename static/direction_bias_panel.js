(() => {
  const REFRESH_MS = 15000;

  function ensureStyles() {
    if (document.getElementById("directionBiasStyles")) return;
    const style = document.createElement("style");
    style.id = "directionBiasStyles";
    style.textContent = `
      .direction-bias-block{margin:10px 0 2px;padding:11px 12px;border-radius:10px;background:rgba(84,200,255,.045);border:1px solid rgba(84,200,255,.12)}
      .direction-bias-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px}
      .direction-bias-head span{font-size:10px;color:var(--muted);font-weight:800;letter-spacing:.10em;text-transform:uppercase}
      .direction-preview-pill{padding:3px 6px;border-radius:999px;font-size:8px!important;letter-spacing:.07em!important;color:#9ab0c8!important;border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.035);white-space:nowrap}
      .direction-bias-main{display:flex;align-items:center;justify-content:space-between;gap:14px}
      .direction-bias-value{display:flex;align-items:center;gap:8px;font-size:16px;font-weight:900;letter-spacing:.025em}
      .direction-bias-value b{font-size:22px;line-height:1;text-shadow:0 0 12px currentColor}
      .direction-up{color:#4ce5a6}.direction-down{color:#ff6f7d}.direction-na{color:var(--muted)}
      .direction-confidence{text-align:right}.direction-confidence strong{display:block;font-size:17px;font-weight:900}.direction-confidence small{display:block;margin-top:2px;font-size:8px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
      .direction-strength{margin-top:7px;font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
    `;
    document.head.appendChild(style);
  }

  function strengthLabel(confidence) {
    if (!Number.isFinite(confidence)) return "Unavailable";
    if (confidence >= 0.80) return "Very strong model confidence";
    if (confidence >= 0.70) return "Strong model confidence";
    if (confidence >= 0.60) return "Moderate model confidence";
    return "Low model confidence";
  }

  function buildDirectionBlock(item) {
    const model = item?.direction_model;
    if (!model?.available) {
      return `
        <div class="direction-bias-block" data-direction-bias>
          <div class="direction-bias-head"><span>Direction Bias · 1H</span><span class="direction-preview-pill">Research Preview</span></div>
          <div class="direction-bias-main"><div class="direction-bias-value direction-na"><b>•</b><span>UNAVAILABLE</span></div></div>
        </div>`;
    }

    const prediction = String(model.prediction || "N/A").toUpperCase();
    const confidence = Number(model.confidence);
    const pct = Number.isFinite(Number(model.confidence_pct))
      ? Number(model.confidence_pct)
      : (Number.isFinite(confidence) ? confidence * 100 : null);
    const upper = prediction === "UPPER_FIRST";
    const lower = prediction === "LOWER_FIRST";
    const cls = upper ? "direction-up" : lower ? "direction-down" : "direction-na";
    const arrow = upper ? "↑" : lower ? "↓" : "•";

    return `
      <div class="direction-bias-block" data-direction-bias>
        <div class="direction-bias-head"><span>Direction Bias · 1H</span><span class="direction-preview-pill">Research Preview</span></div>
        <div class="direction-bias-main">
          <div class="direction-bias-value ${cls}"><b>${arrow}</b><span>${prediction}</span></div>
          <div class="direction-confidence"><strong>${pct === null ? "—" : `${pct.toFixed(1)}%`}</strong><small>Confidence</small></div>
        </div>
        <div class="direction-strength">${strengthLabel(confidence)}</div>
      </div>`;
  }

  function injectDirectionPanels(payload) {
    const bySymbol = new Map((payload?.radar || []).map((item) => [String(item.symbol), item]));
    document.querySelectorAll(".radar-card").forEach((card) => {
      const symbol = card.querySelector(".symbol")?.textContent?.trim();
      if (!symbol) return;
      const item = bySymbol.get(symbol);
      if (!item) return;

      card.querySelector("[data-direction-bias]")?.remove();
      const anchor = card.querySelector(".ai-risk-hero");
      if (!anchor) return;
      anchor.insertAdjacentHTML("afterend", buildDirectionBlock(item));
    });
  }

  let latestPayload = null;
  let fetching = false;

  async function refreshDirection() {
    if (fetching) return;
    fetching = true;
    try {
      const response = await fetch("/radar", { cache: "no-store" });
      if (!response.ok) return;
      latestPayload = await response.json();
      injectDirectionPanels(latestPayload);
    } catch (_) {
      // Keep the primary radar untouched if the research preview fails.
    } finally {
      fetching = false;
    }
  }

  ensureStyles();
  refreshDirection();
  setInterval(refreshDirection, REFRESH_MS);

  const observer = new MutationObserver(() => {
    if (latestPayload) injectDirectionPanels(latestPayload);
  });
  const cards = document.getElementById("radarCards");
  if (cards) observer.observe(cards, { childList: true, subtree: true });
})();
