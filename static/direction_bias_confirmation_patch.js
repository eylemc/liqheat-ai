(() => {
  "use strict";

  try {
    const previousRenderCard = window.renderCard;
    if (typeof previousRenderCard !== "function") return;

    const CONFIDENCE_THRESHOLD = 0.70;

    function confirmationFor(item) {
      const model = item && item.direction_model;
      if (!model || model.available === false) {
        return { state: "NEUTRAL", cls: "lp-confirm-neutral", icon: "•" };
      }

      const prediction = String(model.prediction || "").toUpperCase();
      const confidence = Number(model.confidence);

      if (!Number.isFinite(confidence) || confidence < CONFIDENCE_THRESHOLD) {
        return { state: "NEUTRAL", cls: "lp-confirm-neutral", icon: "•" };
      }

      const rawPressure = String(item.raw_prediction || "").toUpperCase();
      let pressurePrediction = null;
      if (rawPressure === "SHORT_SQUEEZE") pressurePrediction = "UPPER_FIRST";
      if (rawPressure === "LONG_SQUEEZE") pressurePrediction = "LOWER_FIRST";

      if (!pressurePrediction || (prediction !== "UPPER_FIRST" && prediction !== "LOWER_FIRST")) {
        return { state: "NEUTRAL", cls: "lp-confirm-neutral", icon: "•" };
      }

      if (pressurePrediction === prediction) {
        return { state: "CONFIRMED", cls: "lp-confirm-ok", icon: "✓" };
      }

      return { state: "CONFLICT", cls: "lp-confirm-conflict", icon: "!" };
    }

    window.renderCard = function renderCardWithLpConfirmation(item) {
      const html = previousRenderCard(item);
      if (typeof html !== "string") return html;

      const marker = '<div class="matrix-strip">';
      if (!html.includes(marker)) return html;

      const c = confirmationFor(item);
      const block = `
        <div class="lp-confirm-row ${c.cls}">
          <span>LP Confirmation</span>
          <strong>${c.icon} ${c.state}</strong>
        </div>
      `;

      return html.replace(marker, block + marker);
    };

    const style = document.createElement("style");
    style.id = "directionBiasConfirmationStyles";
    style.textContent = `
      .lp-confirm-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:-2px 0 10px;padding:8px 11px;border-radius:9px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06)}
      .lp-confirm-row span{font-size:9px;font-weight:800;letter-spacing:.08em;color:var(--muted)}
      .lp-confirm-row strong{font-size:11px;font-weight:900;letter-spacing:.03em;white-space:nowrap}
      .lp-confirm-ok strong{color:#42e49d}
      .lp-confirm-conflict strong{color:#ffb84d}
      .lp-confirm-neutral strong{color:var(--muted)}
    `;
    document.head.appendChild(style);

    if (typeof state !== "undefined" && state.payload && typeof render === "function") {
      render(state.payload);
    }
  } catch (error) {
    console.error("Direction Bias LP confirmation patch disabled:", error);
  }
})();
