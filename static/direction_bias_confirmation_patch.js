(() => {
  "use strict";

  try {
    const previousRenderCard = window.renderCard;
    if (typeof previousRenderCard !== "function") return;

    function confirmationFor(item) {
      const temporal = item && item.lp_confirmation_v2;
      if (temporal && typeof temporal === "object") {
        const state = String(temporal.state || "NEUTRAL").toUpperCase();
        if (state === "CONFIRMED") return { state, cls: "lp-confirm-ok", icon: "✓", temporal };
        if (state === "CONFLICT") return { state, cls: "lp-confirm-conflict", icon: "!", temporal };
        return { state: "NEUTRAL", cls: "lp-confirm-neutral", icon: "•", temporal };
      }
      return { state: "NEUTRAL", cls: "lp-confirm-neutral", icon: "•", temporal: null };
    }

    function detailText(c) {
      const t = c.temporal;
      if (!t) return "2H history unavailable";
      if (t.reason === "INSUFFICIENT_2H_HISTORY") {
        return `Building 2H history (${Number(t.sample_count_120m || 0)}/60 min)`;
      }
      const persistence = Number(t.persistence_120m);
      const n = Number(t.sample_count_120m || 0);
      if (Number.isFinite(persistence) && n > 0) {
        return `2H persistence ${persistence.toFixed(0)}% · ${n} samples`;
      }
      return "2H temporal pressure check";
    }

    window.renderCard = function renderCardWithLpConfirmation(item) {
      const html = previousRenderCard(item);
      if (typeof html !== "string") return html;

      const marker = '<div class="matrix-strip">';
      if (!html.includes(marker)) return html;

      const c = confirmationFor(item);
      const block = `
        <div class="lp-confirm-row ${c.cls}">
          <div><span>LP Confirmation · 2H</span><small>${detailText(c)}</small></div>
          <strong>${c.icon} ${c.state}</strong>
        </div>
      `;

      return html.replace(marker, block + marker);
    };

    const style = document.createElement("style");
    style.id = "directionBiasConfirmationStyles";
    style.textContent = `
      .lp-confirm-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:-2px 0 10px;padding:8px 11px;border-radius:9px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06)}
      .lp-confirm-row>div{display:flex;flex-direction:column;gap:2px;min-width:0}
      .lp-confirm-row span{font-size:9px;font-weight:800;letter-spacing:.08em;color:var(--muted)}
      .lp-confirm-row small{font-size:9px;color:var(--muted);opacity:.8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
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
