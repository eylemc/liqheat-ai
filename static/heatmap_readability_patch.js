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

    /* Current-price line must be visually dominant but not overpower bars. */
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

    /* Every liquidation level uses the same horizontal reading direction.
       Price is encoded vertically; intensity is encoded by bar length. */
    #liqHeatmapModal .liq-modal-level{
      position:absolute!important;
      z-index:8!important;
      left:17%!important;
      width:66%!important;
      height:7px!important;
      transform:translateY(-50%)!important;
      display:flex!important;
      align-items:center!important;
      pointer-events:auto!important;
    }

    #liqHeatmapModal .liq-modal-level > span{
      display:block!important;
      height:100%!important;
      min-width:14px!important;
      border-radius:999px!important;
      filter:none!important;
    }

    #liqHeatmapModal .liq-modal-above > span{
      background:linear-gradient(90deg,rgba(255,121,94,.72),#ff6374)!important;
      box-shadow:0 0 10px rgba(255,99,116,.22)!important;
    }

    #liqHeatmapModal .liq-modal-below > span{
      background:linear-gradient(90deg,rgba(76,229,166,.68),#35d99a)!important;
      box-shadow:0 0 10px rgba(76,229,166,.20)!important;
    }

    #liqHeatmapModal .liq-modal-strong > span{
      height:10px!important;
      box-shadow:0 0 14px currentColor!important;
    }

    /* Put selected price labels in a dedicated left gutter instead of on top of bars. */
    #liqHeatmapModal .liq-modal-price-tag,
    #liqHeatmapModal .liq-modal-level em{
      position:absolute!important;
      left:-128px!important;
      top:50%!important;
      transform:translateY(-50%)!important;
      width:116px!important;
      padding:3px 7px!important;
      border-radius:6px!important;
      background:rgba(9,13,19,.94)!important;
      border:1px solid rgba(255,255,255,.08)!important;
      color:#d9e0ea!important;
      font-style:normal!important;
      font-size:11px!important;
      font-weight:750!important;
      text-align:right!important;
      line-height:1.25!important;
      white-space:nowrap!important;
      text-shadow:none!important;
    }

    #liqHeatmapModal .liq-modal-above .liq-modal-price-tag,
    #liqHeatmapModal .liq-modal-above em{color:#ff929e!important}
    #liqHeatmapModal .liq-modal-below .liq-modal-price-tag,
    #liqHeatmapModal .liq-modal-below em{color:#75eabc!important}

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
      #liqHeatmapModal .liq-modal-level{left:27%!important;width:69%!important}
      #liqHeatmapModal .liq-modal-price-tag,
      #liqHeatmapModal .liq-modal-level em{left:-105px!important;width:96px!important;font-size:10px!important}
    }
  `;
  document.head.appendChild(style);
})();
