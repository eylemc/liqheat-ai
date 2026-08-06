const state = {
  payload: null,
  seconds: 15,
  timer: null,
};

const $ = (id) => document.getElementById(id);

function formatNumber(value, maximumFractionDigits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }

  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits,
  });
}

function formatAge(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${Math.round(seconds)} sec`;

  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60);
  return `${minutes}m ${remaining}s`;
}

function formatTime(iso) {
  if (!iso) return "—";

  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function humanize(value, fallback = "—") {
  const normalized = String(value || "").trim();
  return normalized ? normalized.replaceAll("_", " ") : fallback;
}

function matrixData(item) {
  return item.matrix && item.matrix.available !== false ? item.matrix : null;
}

function matrixDirection(item) {
  return matrixData(item)?.direction_label || "UNAVAILABLE";
}

function matrixAlignment(item) {
  const value = matrixData(item)?.alignment_score;
  return value === null || value === undefined ? null : Number(value);
}

function opportunity(item) {
  return String(item.opportunity || item.status || "NORMAL").toUpperCase();
}

function expectedOutcome(item) {
  const prediction = String(
    item.raw_prediction || item.prediction || ""
  ).toUpperCase();

  // Important product mapping:
  // SHORT_SQUEEZE means upward price pressure.
  // LONG_SQUEEZE means downward price pressure.
  if (prediction === "SHORT_SQUEEZE") {
    return {
      label: "UPWARD",
      arrow: "↑",
      className: "outcome-upward",
    };
  }

  if (prediction === "LONG_SQUEEZE") {
    return {
      label: "DOWNWARD",
      arrow: "↓",
      className: "outcome-downward",
    };
  }

  return {
    label: "UNCONFIRMED",
    arrow: "•",
    className: "outcome-unconfirmed",
  };
}

function liquidityPressure(item) {
  if (item.liquidity_pressure_score !== undefined) {
    return Number(item.liquidity_pressure_score);
  }

  if (item.liquidity_pressure !== undefined) {
    return Number(item.liquidity_pressure) * 100;
  }

  return Number(item.score || 0) * 100;
}

function radarScore(item) {
  return Number(item.radar_score ?? liquidityPressure(item));
}

function matrixTimeframeStrip(item) {
  const matrix = matrixData(item);
  if (!matrix?.timeframes) {
    return '<span class="matrix-unavailable">Matrix unavailable</span>';
  }

  const order = ["1d", "4h", "1h", "15m", "5m"];

  return order.map((timeframe) => {
    const frame = matrix.timeframes[timeframe];
    const label = frame?.trend_label || "—";
    const directionClass = label === "BUY"
      ? "matrix-buy"
      : label === "SELL"
        ? "matrix-sell"
        : "matrix-neutral";

    return `
      <span class="matrix-tf ${directionClass}">
        <small>${timeframe.toUpperCase()}</small>
        <strong>${label}</strong>
      </span>
    `;
  }).join("");
}

function explain(item) {
  const matrix = matrixData(item);
  const pressure = liquidityPressure(item);

  if (!matrix) {
    return `Liquidity Pressure is ${formatNumber(pressure, 1)}. Matrix data is temporarily unavailable.`;
  }

  if (item.matrix_gate === "BLOCK") {
    return "Topology conflicts with the Daily Matrix regime. Directional opportunity is suppressed.";
  }

  if (item.matrix_gate === "PASS") {
    if (matrix.full_alignment) {
      return "Topology agrees with a fully aligned Matrix regime across 1D, 4H, 1H, 15M and 5M.";
    }

    if (matrix.upper_core_aligned) {
      return "Topology agrees with the core 1D, 4H and 1H Matrix regime. Lower timeframes are still developing.";
    }

    return "Topology agrees with the Daily Matrix direction, but timeframe alignment is partial.";
  }

  return item.radar_explanation || "Matrix and topology are not yet producing a confirmed opportunity.";
}

function cardGlow(item) {
  const value = opportunity(item);
  if (value === "CRITICAL") return "rgba(255,111,125,.27)";
  if (value === "HIGH") return "rgba(255,155,104,.23)";
  if (value === "WATCH") return "rgba(255,202,97,.18)";
  if (value === "CONFLICT") return "rgba(165,140,255,.18)";
  if (matrixDirection(item) === "BULLISH") return "rgba(76,229,166,.13)";
  if (matrixDirection(item) === "BEARISH") return "rgba(255,111,125,.13)";
  return "rgba(84,200,255,.10)";
}

function installOutcomeStyles() {
  if (document.getElementById("expectedOutcomeStyles")) return;

  const style = document.createElement("style");
  style.id = "expectedOutcomeStyles";
  style.textContent = `
    .expected-outcome {
      margin: 16px 0;
      padding: 13px 14px;
      border: 1px solid rgba(255,255,255,.075);
      border-radius: 12px;
      background: rgba(255,255,255,.035);
      box-shadow: inset 0 0 24px rgba(255,255,255,.018);
    }

    .expected-outcome-label {
      display: block;
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
    }

    .expected-outcome-value {
      display: flex;
      align-items: center;
      gap: 9px;
      font-size: 23px;
      font-weight: 850;
      letter-spacing: .035em;
      line-height: 1;
    }

    .expected-outcome-arrow {
      font-size: 29px;
      line-height: .8;
    }

    .outcome-upward .expected-outcome-value {
      color: var(--green);
      text-shadow: 0 0 22px rgba(76,229,166,.16);
    }

    .outcome-downward .expected-outcome-value {
      color: var(--red);
      text-shadow: 0 0 22px rgba(255,111,125,.16);
    }

    .outcome-unconfirmed .expected-outcome-value {
      color: var(--amber);
    }

    .table-outcome {
      font-weight: 800;
      letter-spacing: .035em;
      white-space: nowrap;
    }

    .table-outcome.outcome-upward { color: var(--green); }
    .table-outcome.outcome-downward { color: var(--red); }
    .table-outcome.outcome-unconfirmed { color: var(--amber); }
  `;

  document.head.appendChild(style);
}

function renderCard(item) {
  const score = radarScore(item);
  const pressure = liquidityPressure(item);
  const alignment = matrixAlignment(item);
  const matrix = matrixData(item);
  const opportunityValue = opportunity(item);
  const gate = String(item.matrix_gate || "UNAVAILABLE").toUpperCase();
  const outcome = expectedOutcome(item);

  return `
    <article class="radar-card" style="--glow:${cardGlow(item)}">
      <div class="card-head">
        <span class="rank">#${item.rank}</span>
        <span class="status-pill opportunity-${opportunityValue}">${opportunityValue}</span>
      </div>

      <h3 class="symbol">${item.symbol}</h3>
      <div class="price">${formatNumber(item.current_price, 6)}</div>

      <div class="risk-block">
        <span class="risk-label">Radar Score</span>
        <div class="risk-value">
          <strong>${formatNumber(score, 2)}</strong>
          <span>/ 100</span>
        </div>
        <div class="risk-bar">
          <div class="risk-fill" style="width:${Math.min(Math.max(score, 0), 100)}%"></div>
        </div>
      </div>

      <div class="expected-outcome ${outcome.className}">
        <span class="expected-outcome-label">Expected Outcome</span>
        <div class="expected-outcome-value">
          <span class="expected-outcome-arrow">${outcome.arrow}</span>
          <strong>${outcome.label}</strong>
        </div>
      </div>

      <div class="matrix-strip">${matrixTimeframeStrip(item)}</div>

      <div class="metric-stack">
        <div class="metric-row">
          <span>Matrix direction</span>
          <strong class="matrix-direction-${matrixDirection(item)}">${matrixDirection(item)}</strong>
        </div>
        <div class="metric-row">
          <span>Alignment</span>
          <strong>${alignment === null ? "—" : `${formatNumber(alignment, 1)}%`}</strong>
        </div>
        <div class="metric-row">
          <span>Liquidity pressure</span>
          <strong>${formatNumber(pressure, 2)}</strong>
        </div>
        <div class="metric-row">
          <span>Matrix gate</span>
          <strong class="gate-${gate}">${gate}</strong>
        </div>
      </div>

      ${item.matrix_agreement === false
        ? '<div class="conflict-warning">Topology conflicts with Matrix regime</div>'
        : ''}

      <div class="reason-box">${explain(item)}</div>

      <div class="card-footer">
        <span>${matrix ? humanize(matrix.regime) : "Matrix unavailable"}</span>
        <span>${formatAge(item.age_seconds)}</span>
      </div>
    </article>
  `;
}

function renderTableRow(item) {
  const alignment = matrixAlignment(item);
  const opportunityValue = opportunity(item);
  const gate = String(item.matrix_gate || "UNAVAILABLE").toUpperCase();
  const outcome = expectedOutcome(item);

  return `
    <tr>
      <td>#${item.rank}</td>
      <td><strong>${item.symbol}</strong></td>
      <td class="table-risk">${formatNumber(radarScore(item), 2)}</td>
      <td><span class="status-pill opportunity-${opportunityValue}">${opportunityValue}</span></td>
      <td class="table-outcome ${outcome.className}">${outcome.arrow} ${outcome.label}</td>
      <td class="matrix-direction-${matrixDirection(item)}">${matrixDirection(item)}</td>
      <td>${alignment === null ? "—" : `${formatNumber(alignment, 1)}%`}</td>
      <td><span class="gate-pill gate-${gate}">${gate}</span></td>
      <td>${formatNumber(liquidityPressure(item), 2)}</td>
      <td>${formatNumber(item.current_price, 6)}</td>
      <td>${formatAge(item.age_seconds)}</td>
    </tr>
  `;
}

function ensureOutcomeTableHeader() {
  const headerRow = document.querySelector(".table-section thead tr");
  if (!headerRow || headerRow.querySelector('[data-outcome-header="true"]')) return;

  const headers = headerRow.querySelectorAll("th");
  const matrixHeader = Array.from(headers).find(
    (header) => header.textContent.trim().toUpperCase() === "MATRIX"
  );

  if (!matrixHeader) return;

  const outcomeHeader = document.createElement("th");
  outcomeHeader.textContent = "Expected outcome";
  outcomeHeader.dataset.outcomeHeader = "true";
  headerRow.insertBefore(outcomeHeader, matrixHeader);
}

function render(payload) {
  state.payload = payload;

  const radar = [...(payload.radar || [])].sort(
    (a, b) => radarScore(b) - radarScore(a)
  );

  radar.forEach((item, index) => {
    item.rank = index + 1;
  });

  const highest = radar[0];
  const active = radar.filter((item) =>
    ["WATCH", "HIGH", "CRITICAL"].includes(opportunity(item))
  ).length;

  $("engineStatus").textContent = payload.status || "UNKNOWN";
  $("engineDot").className =
    `status-dot ${(payload.status || "offline").toLowerCase()}`;

  $("lastRefresh").textContent = formatTime(payload.generated_at);
  $("symbolCount").textContent = payload.symbol_count ?? radar.length;
  $("freshestAge").textContent = formatAge(payload.freshest_snapshot_age_seconds);
  $("highestRisk").textContent = highest ? formatNumber(radarScore(highest), 2) : "—";
  $("highestSymbol").textContent = highest
    ? `${highest.symbol} leads the radar`
    : "Waiting for radar";
  $("activeWatches").textContent = active;

  $("radarCards").innerHTML = radar.length
    ? radar.map(renderCard).join("")
    : '<div class="loading-card">No live radar data available.</div>';

  ensureOutcomeTableHeader();
  $("radarTable").innerHTML = radar.map(renderTableRow).join("");
  $("rawJson").textContent = JSON.stringify(payload, null, 2);
}

async function loadRadar(force = false) {
  const button = $("refreshButton");

  try {
    button.disabled = true;
    button.textContent = force ? "Refreshing…" : "Loading…";

    const endpoint = force ? "/radar/refresh" : "/radar";
    const options = force ? { method: "POST" } : {};
    const response = await fetch(endpoint, options);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    render(await response.json());
  } catch (error) {
    $("engineStatus").textContent = "OFFLINE";
    $("engineDot").className = "status-dot offline";
    $("radarCards").innerHTML =
      `<div class="loading-card">Radar unavailable: ${error.message}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "Refresh now";
    state.seconds = 15;
  }
}

function startCountdown() {
  clearInterval(state.timer);

  state.timer = setInterval(() => {
    state.seconds -= 1;
    $("countdown").textContent = `Auto refresh in ${Math.max(state.seconds, 0)}s`;

    if (state.seconds <= 0) {
      state.seconds = 15;
      loadRadar(false);
    }
  }, 1000);
}

$("refreshButton").addEventListener("click", () => loadRadar(true));
$("jsonToggle").addEventListener("click", () => $("jsonPanel").classList.toggle("hidden"));
$("jsonClose").addEventListener("click", () => $("jsonPanel").classList.add("hidden"));

installOutcomeStyles();
loadRadar(false);
startCountdown();
