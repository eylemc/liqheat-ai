(() => {
  function gateData(item) {
    const gate = item?.matrix_regime_gate;
    return gate?.available ? gate : null;
  }

  function gateSignal(item) {
    const gate = gateData(item);
    const side = String(gate?.latest_flip?.side || gate?.matrix_trend || "N/A").toUpperCase();
    return side === "BUY" || side === "SELL" ? side : "N/A";
  }

  function gateRisk(item) {
    const gate = gateData(item);
    if (!gate) return "RISK N/A";
    if (String(gate.risk_level || "").toUpperCase() === "LOW RISK") return "LOW RISK";
    return gate.status === "VALID" ? "LOW RISK" : "HIGH RISK";
  }

  function signalClass(signal) {
    return signal === "BUY" ? "signal-buy" : signal === "SELL" ? "signal-sell" : "signal-na";
  }

  function riskClass(risk) {
    if (risk === "LOW RISK") return "signal-risk-low";
    if (risk === "HIGH RISK") return "signal-risk-high";
    return "signal-risk-na";
  }

  function riskIcon(risk) {
    return risk === "LOW RISK" ? "✓" : risk === "HIGH RISK" ? "!" : "?";
  }

  function signalArrow(signal) {
    return signal === "BUY" ? "↑" : signal === "SELL" ? "↓" : "•";
  }

  function signalExplanation(item) {
    const signal = gateSignal(item);
    const risk = gateRisk(item);
    if (risk === "LOW RISK") {
      return `Current 1H Matrix signal is <b class="${signalClass(signal)}">${signal}</b>.<br>Market conditions confirm this signal.`;
    }
    if (risk === "HIGH RISK") {
      return `Current 1H Matrix signal is <b class="${signalClass(signal)}">${signal}</b>.<br>Market conditions do not confirm this signal.`;
    }
    return `Current 1H Matrix signal is <b class="${signalClass(signal)}">${signal}</b>.<br>Risk confirmation is not available.`;
  }

  // Frozen research rule: only apply LP confirmation when Direction Bias confidence >= 70%.
  // SHORT_SQUEEZE pressure maps to UPPER_FIRST; LONG_SQUEEZE maps to LOWER_FIRST.
  function directionConfirmation(item, prediction, confidence) {
    if (!Number.isFinite(confidence) || confidence < 0.70) {
      return { state: "NEUTRAL", cls: "confirmation-neutral", icon: "•" };
    }

    const rawPressureDirection = String(item?.raw_prediction || "").toUpperCase();
    const pressurePrediction = rawPressureDirection === "SHORT_SQUEEZE"
      ? "UPPER_FIRST"
      : rawPressureDirection === "LONG_SQUEEZE"
        ? "LOWER_FIRST"
        : null;

    if (!pressurePrediction || !["UPPER_FIRST", "LOWER_FIRST"].includes(prediction)) {
      return { state: "NEUTRAL", cls: "confirmation-neutral", icon: "•" };
    }

    if (pressurePrediction === prediction) {
      return { state: "CONFIRMED", cls: "confirmation-confirmed", icon: "✓" };
    }

    return { state: "CONFLICT", cls: "confirmation-conflict", icon: "!" };
  }

  function directionBiasHtmlV2(item) {
    const model = item?.direction_model;
    if (!model?.available) {
      return `
        <div class="direction-bias-inline direction-na">
          <div><span>Direction Bias · 1H</span><strong>UNAVAILABLE</strong></div>
          <div><span>Confidence</span><strong>—</strong></div>
          <div class="direction-confirmation confirmation-neutral"><span>LP Confirmation</span><strong>• NEUTRAL</strong></div>
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
    const confirmation = directionConfirmation(item, prediction, confidence);
    return `
      <div class="direction-bias-inline ${cls}">
        <div><span>Direction Bias · 1H</span><strong>${arrow} ${prediction}</strong></div>
        <div><span>Confidence</span><strong>${confidencePct === null ? "—" : `${confidencePct.toFixed(1)}%`}</strong></div>
        <div class="direction-confirmation ${confirmation.cls}"><span>LP Confirmation</span><strong>${confirmation.icon} ${confirmation.state}</strong></div>
      </div>`;
  }

  window.renderCard = function renderSignalFirstCard(item) {
    const matrix = matrixData(item);
    const alignment = matrixAlignment(item);
    const score = heatScore(item);
    const band = heatBand(item);
    const signal = gateSignal(item);
    const risk = gateRisk(item);
    const sCls = signalClass(signal);
    const rCls = riskClass(risk);

    return `
      <article class="radar-card radar-card-v2" style="--glow:${cardGlow(item)}">
        <div class="card-head"><span class="rank">#${item.rank}</span></div>
        <h3 class="symbol">${item.symbol}</h3>
        <div class="price">${formatNumber(item.current_price, 6)}</div>

        <div class="ai-risk-hero heat-compact-v2">
          <div class="ai-risk-heading"><span>NEAR-TERM MARKET HEAT</span><span class="risk-band-pill risk-${heatClass(item)}">${band}</span></div>
          <div class="ai-risk-score"><strong>${score === null ? "—" : Math.round(score)}</strong><span>${score === null ? "" : "/ 100"}</span></div>
          <div class="ai-risk-sub">Near-term movement percentile vs historical calibration</div>
        </div>

        <div class="matrix-signal-v2">
          <div class="matrix-signal-title">1H MATRIX SIGNAL</div>
          <div class="matrix-signal-main">
            <div class="matrix-signal-side ${sCls}"><span>${signalArrow(signal)}</span><strong>${signal}</strong></div>
            <div class="matrix-risk-pill ${rCls}"><span>${riskIcon(risk)}</span><strong>${risk}</strong></div>
          </div>
          <div class="matrix-signal-explain ${rCls}">
            <span class="matrix-explain-icon">${riskIcon(risk)}</span>
            <div>${signalExplanation(item)}</div>
          </div>
        </div>

        ${directionBiasHtmlV2(item)}
        <div class="matrix-strip">${matrixTimeframeStrip(item)}</div>
        <div class="metric-stack">
          <div class="metric-row"><span>Matrix alignment</span><strong>${alignment === null ? "—" : `${formatNumber(alignment, 1)}%`}</strong></div>
          <div class="metric-row"><span>Liquidation Pressure</span>${liquidationPressureHtml(item)}</div>
          <div class="metric-row"><span>Heat horizon</span><strong>${riskData(item) ? "NEAR TERM" : "—"}</strong></div>
        </div>
        ${compactLiquidationHeatmap(item)}
        <div class="reason-box">${heatExplanation(item)}</div>
        <div class="card-footer"><span>${matrix ? humanize(matrix.regime) : "Matrix unavailable"}</span><span>${formatAge(item.age_seconds)}</span></div>
      </article>`;
  };

  window.renderTableRow = function renderSignalFirstTableRow(item) {
    const alignment = matrixAlignment(item);
    const score = heatScore(item);
    const signal = gateSignal(item);
    const risk = gateRisk(item);
    return `<tr>
      <td>#${item.rank}</td>
      <td><strong>${item.symbol}</strong></td>
      <td class="${signalClass(signal)}"><strong>${signalArrow(signal)} ${signal}</strong></td>
      <td><span class="matrix-risk-pill matrix-risk-pill-table ${riskClass(risk)}"><span>${riskIcon(risk)}</span><strong>${risk}</strong></span></td>
      <td>${score === null ? "—" : Math.round(score)}</td>
      <td><span class="risk-band-pill risk-${heatClass(item)}">${heatBand(item)}</span></td>
      <td>${alignment === null ? "—" : `${formatNumber(alignment, 1)}%`}</td>
      <td>${liquidationPressureHtml(item)}</td>
      <td>${formatNumber(item.current_price, 6)}</td>
      <td>${formatAge(item.age_seconds)}</td>
    </tr>`;
  };

  const style = document.createElement("style");
  style.id = "radarSignalUiV2Styles";
  style.textContent = `
    .radar-card-v2 .heat-compact-v2{margin-bottom:14px}
    .matrix-signal-v2{margin:14px 0 10px;padding:14px;border-radius:12px;background:rgba(5,10,17,.32);border:1px solid rgba(255,255,255,.07)}
    .matrix-signal-title{font-size:9px;font-weight:850;letter-spacing:.13em;color:var(--muted);margin-bottom:10px}
    .matrix-signal-main{display:flex;align-items:center;justify-content:space-between;gap:12px;padding-bottom:11px;border-bottom:1px solid rgba(255,255,255,.07)}
    .matrix-signal-side{display:flex;align-items:center;gap:9px;font-weight:950}.matrix-signal-side span{font-size:31px;line-height:1}.matrix-signal-side strong{font-size:26px;letter-spacing:.02em}
    .signal-buy{color:#4ce5a6!important}.signal-sell{color:#ff6374!important}.signal-na{color:var(--muted)!important}
    .matrix-risk-pill{display:inline-flex;align-items:center;gap:7px;padding:7px 10px;border-radius:999px;border:1px solid currentColor;font-size:10px;letter-spacing:.05em;font-weight:900;white-space:nowrap}
    .matrix-risk-pill>span{display:inline-flex;align-items:center;justify-content:center;width:17px;height:17px;border:2px solid currentColor;border-radius:50%;font-size:11px;line-height:1}
    .signal-risk-low{color:#42e49d!important}.signal-risk-high{color:#ff5f70!important}.signal-risk-na{color:var(--muted)!important}
    .matrix-risk-pill.signal-risk-low{background:rgba(66,228,157,.07);border-color:rgba(66,228,157,.25)}
    .matrix-risk-pill.signal-risk-high{background:rgba(255,95,112,.07);border-color:rgba(255,95,112,.28)}
    .matrix-signal-explain{display:flex;align-items:center;gap:10px;margin-top:11px;padding:10px 11px;border-radius:9px;background:rgba(255,255,255,.025);font-size:10px;line-height:1.55;color:#c8d0de!important}
    .matrix-explain-icon{flex:0 0 auto;display:inline-flex;align-items:center;justify-content:center;width:27px;height:31px;border:2px solid currentColor;border-radius:9px 9px 12px 12px;font-size:14px;font-weight:950}
    .matrix-signal-explain.signal-risk-low .matrix-explain-icon{color:#42e49d}.matrix-signal-explain.signal-risk-high .matrix-explain-icon{color:#ff5f70}
    .matrix-risk-pill-table{padding:5px 8px;font-size:9px}.matrix-risk-pill-table>span{width:14px;height:14px;font-size:9px}
    .direction-bias-inline{display:grid;grid-template-columns:1.4fr .8fr 1fr;gap:10px;align-items:center}
    .direction-confirmation strong{display:inline-flex;align-items:center;gap:5px;font-size:11px!important;white-space:nowrap}
    .confirmation-confirmed strong{color:#42e49d!important}
    .confirmation-conflict strong{color:#ffb84d!important}
    .confirmation-neutral strong{color:var(--muted)!important}
    @media(max-width:700px){.matrix-signal-side span{font-size:27px}.matrix-signal-side strong{font-size:23px}.matrix-signal-v2{padding:12px}.direction-bias-inline{grid-template-columns:1fr 1fr}.direction-confirmation{grid-column:1/-1}}
  `;
  document.head.appendChild(style);

  document.querySelectorAll("th").forEach((th) => {
    const text = th.textContent.trim().toLowerCase();
    if (text === "scalp idea") th.textContent = "1H Signal";
  });

  const tableHead = document.querySelector(".table-wrap thead tr");
  if (tableHead) {
    tableHead.innerHTML = `
      <th>Rank</th><th>Market</th><th>1H Signal</th><th>Signal Risk</th>
      <th>Market Heat Score</th><th>Heat Level</th><th>Alignment</th>
      <th>Liquidation Pressure</th><th>Price</th><th>Data age</th>`;
  }

  const subtitle = document.querySelector('.brand-row p');
  if (subtitle) subtitle.textContent = '1H Matrix Signal + Near-Term Market Heat';
  const heading = document.querySelector('.section-heading h2');
  if (heading) heading.textContent = '1H Matrix signal with near-term Market Heat';

  if (typeof state !== "undefined" && state.payload) render(state.payload);
  else if (typeof loadRadar === "function") loadRadar(false);
})();
