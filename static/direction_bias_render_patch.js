(() => {
  const originalRenderCard = window.renderCard;
  if (typeof originalRenderCard !== "function") return;

  function directionBiasHtml(item) {
    const model = item?.direction_model;
    if (!model?.available) {
      return `
        <div class="direction-bias-inline direction-na">
          <div><span>Direction Bias · 1H</span><strong>UNAVAILABLE</strong></div>
          <div><span>Confidence</span><strong>—</strong></div>
        </div>`;
    }

    const prediction = String(model.prediction || "N/A").toUpperCase();
    const confidence = Number(model.confidence);
    const confidencePct = Number.isFinite(Number(model.confidence_pct))
      ? Number(model.confidence_pct)
      : (Number.isFinite(confidence) ? confidence * 100 : null);

    const upper = prediction === "UPPER_FIRST";
    const lower = prediction === "LOWER_FIRST";
    const cls = upper ? "direction-up" : lower ? "direction-down" : "direction-na";
    const arrow = upper ? "↑" : lower ? "↓" : "•";

    return `
      <div class="direction-bias-inline ${cls}">
        <div><span>Direction Bias · 1H</span><strong>${arrow} ${prediction}</strong></div>
        <div><span>Confidence</span><strong>${confidencePct === null ? "—" : `${confidencePct.toFixed(1)}%`}</strong></div>
      </div>`;
  }

  window.renderCard = function renderCardWithDirectionBias(item) {
    const html = originalRenderCard(item);
    const anchor = '<div class="matrix-strip">';
    if (!html.includes(anchor)) return html;
    return html.replace(anchor, `${directionBiasHtml(item)}${anchor}`);
  };

  if (!document.getElementById("directionBiasInlineStyles")) {
    const style = document.createElement("style");
    style.id = "directionBiasInlineStyles";
    style.textContent = `
      .direction-bias-inline{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:10px 0;padding:10px 12px;border-radius:10px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06)}
      .direction-bias-inline>div{display:flex;flex-direction:column;gap:4px}.direction-bias-inline>div:last-child{text-align:right}
      .direction-bias-inline span{font-size:8px;color:var(--muted);font-weight:800;letter-spacing:.09em;text-transform:uppercase}
      .direction-bias-inline strong{font-size:13px;font-weight:900;letter-spacing:.025em}
      .direction-bias-inline.direction-up strong{color:#4ce5a6}
      .direction-bias-inline.direction-down strong{color:#ff6f7d}
      .direction-bias-inline.direction-na strong{color:var(--muted)}
    `;
    document.head.appendChild(style);
  }
})();
