(() => {
  "use strict";

  const style = document.createElement("style");
  style.id = "liqHeatmapReadabilityPatch";
  style.textContent = `
    /* Expanded heatmap readability patch. Presentation only; data/model untouched. */
    #liqHeatmapModal .liq-modal-card{
      width:min(1180px,94vw)!important;
      max-height:92vh!important;
      padding:22px 24px 20px!important;
    }

    #liqHeatmapModal .liq-modal-meta{
      margin:2px 0 12px!important;
      color:#9aa5b5!important;
      font-size:12px!important;
    }

    #liqHeatmapModal .liq-modal-map{
      position:relative!important;
      height:min(64vh,620px)!important;
      min-height:500px!important;
      margin:0!important;
      border:1px solid rgba(255,255,255,.09)!important;
      border-radius:14px!important;
      overflow:hidden!important;
      background:
        linear-gradient(to bottom,rgba(255,98,112,.025),transparent 44%,rgba(255,255,255,.018) 49%,rgba(255,255,255,.018) 51%,transparent 56%,rgba(76,229,166,.025)),
        repeating-linear-gradient(to bottom,transparent 0,transparent calc(10% - 1px),rgba(255,255,255,.035) calc(10% - 1px),rgba(255,255,255,.035) 10%),
        rgba(7,11,16,.72)!important;
    }

    #liqHeatmapModal .liq-modal-price-line{
      position:absolute!important;
      z-index:20!important;
      left:0!important;
      right:0!important;
      top:50%!important;
      height:1px!important;
      background:rgba(240,245,255,.72)!important;
      box-shadow:none!important;
    }

    #liqHeatmapModal .liq-modal-current-badge{
      position:absolute!important;
      z-index:30!important;
      left:50%!important;
      top:50%!important;
      transform:translate(-50%,-50%)!important;
      padding:7px 11px!important;
      border-radius:9px!important;
      background:#f5f7fb!important;
      color:#171b23!important;
      border:1px solid rgba(255,255,255,.75)!important;
      box-shadow:0 4px 18px rgba(0,0,0,.38)!important;
      font-style:normal!important;
      font-size:13px!important;
      font-weight:900!important;
      line-height:1!important;
      white-space:nowrap!important;
    }

    #liqHeatmapModal .liq-modal-level{
      position:absolute!important;
      z-index:8!important;
      left:10%!important;
      width:80%!important;
      height:7px!important;
      transform:translateY(-50%)!important;
      display:flex!important;
      align-items:center!important;
      gap:10px!important;
      pointer-events:auto!important;
      opacity:1!important;
      cursor:crosshair!important;
    }

    #liqHeatmapModal .liq-modal-level > span{
      display:block!important;
      flex:0 0 auto!important;
      height:100%!important;
      min-width:14px!important;
      border-radius:999px!important;
      filter:none!important;
    }

    #liqHeatmapModal .liq-modal-above > span{
      background:linear-gradient(90deg,#ff8a7c,#ff6374)!important;
      box-shadow:0 0 10px rgba(255,99,116,.30)!important;
    }

    #liqHeatmapModal .liq-modal-below > span{
      background:linear-gradient(90deg,#67efba,#35d99a)!important;
      box-shadow:0 0 10px rgba(76,229,166,.28)!important;
    }

    #liqHeatmapModal .liq-modal-strong > span{
      height:10px!important;
      box-shadow:0 0 14px currentColor!important;
    }

    /* Price labels sit immediately after the bar and stay fully opaque/readable. */
    #liqHeatmapModal .liq-modal-price-tag,
    #liqHeatmapModal .liq-modal-level em{
      position:relative!important;
      left:auto!important;
      top:auto!important;
      transform:none!important;
      width:auto!important;
      min-width:74px!important;
      padding:2px 6px!important;
      border-radius:5px!important;
      background:rgba(7,10,15,.96)!important;
      border:1px solid rgba(255,255,255,.13)!important;
      color:#f4f7fb!important;
      opacity:1!important;
      font-style:normal!important;
      font-size:12px!important;
      font-weight:900!important;
      text-align:left!important;
      line-height:1.25!important;
      white-space:nowrap!important;
      text-shadow:0 1px 2px rgba(0,0,0,.9)!important;
      box-shadow:0 2px 8px rgba(0,0,0,.28)!important;
      pointer-events:none!important;
    }

    #liqHeatmapModal .liq-modal-above .liq-modal-price-tag,
    #liqHeatmapModal .liq-modal-above em{color:#ff9ca7!important}
    #liqHeatmapModal .liq-modal-below .liq-modal-price-tag,
    #liqHeatmapModal .liq-modal-below em{color:#87f3c8!important}

    #liqHeatmapHoverTooltip{
      position:fixed;
      z-index:99999;
      display:none;
      min-width:150px;
      padding:8px 10px;
      border-radius:8px;
      border:1px solid rgba(255,255,255,.14);
      background:rgba(7,10,15,.97);
      box-shadow:0 8px 24px rgba(0,0,0,.42);
      pointer-events:none;
      color:#f5f7fb;
      font-size:12px;
      line-height:1.35;
      white-space:nowrap;
    }
    #liqHeatmapHoverTooltip strong{display:block;font-size:13px;font-weight:900;color:#fff}
    #liqHeatmapHoverTooltip span{display:block;margin-top:2px;color:#aab4c2;font-size:11px}
    #liqHeatmapHoverTooltip.below strong{color:#87f3c8}
    #liqHeatmapHoverTooltip.above strong{color:#ff9ca7}

    #liqHeatmapModal .liq-modal-axis{
      display:grid!important;
      grid-template-columns:1fr auto 1fr!important;
      align-items:center!important;
      gap:18px!important;
      margin-top:10px!important;
      color:#8791a1!important;
      font-size:11px!important;
      text-transform:uppercase!important;
      letter-spacing:.05em!important;
    }
    #liqHeatmapModal .liq-modal-axis span:last-child{text-align:right!important}
    #liqHeatmapModal .liq-modal-axis strong{
      color:#eef3fa!important;
      font-size:12px!important;
      letter-spacing:0!important;
    }

    #liqHeatmapModal .liq-modal-legend{
      display:flex!important;
      justify-content:space-between!important;
      margin-top:12px!important;
      font-size:11px!important;
      color:#8f99a8!important;
    }
    #liqHeatmapModal .legend-below::before,
    #liqHeatmapModal .legend-above::before{
      content:"";
      display:inline-block;
      width:22px;
      height:4px;
      margin-right:7px;
      border-radius:999px;
      vertical-align:middle;
    }
    #liqHeatmapModal .legend-below::before{background:#4ce5a6}
    #liqHeatmapModal .legend-above::before{background:#ff6374}

    @media(max-width:760px){
      #liqHeatmapModal .liq-modal-card{width:96vw!important;padding:16px!important}
      #liqHeatmapModal .liq-modal-map{min-height:430px!important;height:62vh!important}
      #liqHeatmapModal .liq-modal-level{left:5%!important;width:90%!important;gap:7px!important}
      #liqHeatmapModal .liq-modal-price-tag,
      #liqHeatmapModal .liq-modal-level em{min-width:64px!important;font-size:10px!important;padding:2px 4px!important}
    }
  `;
  document.head.appendChild(style);

  const ensureTooltip = () => {
    let tip = document.getElementById("liqHeatmapHoverTooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "liqHeatmapHoverTooltip";
      document.body.appendChild(tip);
    }
    return tip;
  };

  const hideTooltip = () => {
    const tip = document.getElementById("liqHeatmapHoverTooltip");
    if (tip) tip.style.display = "none";
  };

  document.addEventListener("pointerover", (event) => {
    const level = event.target.closest?.("#liqHeatmapModal .liq-modal-level");
    if (!level) return;

    const raw = level.dataset.hoverText || level.getAttribute("title") || "";
    if (!raw) return;
    level.dataset.hoverText = raw;
    level.removeAttribute("title");

    const [price, volume] = raw.split("·").map((part) => part.trim());
    const tip = ensureTooltip();
    tip.className = level.classList.contains("liq-modal-below") ? "below" : "above";
    tip.innerHTML = `<strong>Level ${price || "—"}</strong>${volume ? `<span>Liquidation ${volume}</span>` : ""}`;
    tip.style.display = "block";
  });

  document.addEventListener("pointermove", (event) => {
    const level = event.target.closest?.("#liqHeatmapModal .liq-modal-level");
    if (!level) return;
    const tip = ensureTooltip();
    const pad = 14;
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    const rect = tip.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.max(8, y)}px`;
  });

  document.addEventListener("pointerout", (event) => {
    const level = event.target.closest?.("#liqHeatmapModal .liq-modal-level");
    if (!level) return;
    const next = event.relatedTarget;
    if (next && level.contains(next)) return;
    hideTooltip();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideTooltip();
  });
})();
