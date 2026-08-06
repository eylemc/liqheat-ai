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

  const date = new Date(iso);

  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function humanBias(value) {
  return String(value || "NEUTRAL").replaceAll("_", " ");
}

function explain(item) {
  const topology = item.topology || {};
  const reasons = [];

  if (topology.nearest_side === "UPPER") {
    reasons.push("Upper liquidity is closer to price.");
  } else if (topology.nearest_side === "LOWER") {
    reasons.push("Lower liquidity is closer to price.");
  }

  const upper = Number(topology.upper_pool_volume || 0);
  const lower = Number(topology.lower_pool_volume || 0);

  if (upper > 0 && lower > 0) {
    const ratio = upper / lower;

    if (ratio >= 1.5) {
      reasons.push(`Upper pool is ${ratio.toFixed(1)}× larger.`);
    } else if (ratio <= 0.67) {
      reasons.push(`Lower pool is ${(1 / ratio).toFixed(1)}× larger.`);
    } else {
      reasons.push("Upper and lower pools are relatively balanced.");
    }
  }

  if (item.direction_state === "UNCONFIRMED") {
    reasons.push("Direction is not confirmed yet.");
  } else if (item.direction_state === "LEAN") {
    reasons.push("Directional evidence is emerging.");
  } else {
    reasons.push("Directional model agreement is strong.");
  }

  return reasons.slice(0, 3).join(" ");
}

function cardGlow(item) {
  if (item.status === "CRITICAL") return "rgba(255,111,125,.25)";
  if (item.status === "ALERT") return "rgba(255,155,104,.22)";
  if (item.status === "WATCH") return "rgba(255,202,97,.17)";
  if (String(item.bias).includes("BULLISH")) return "rgba(76,229,166,.13)";
  if (String(item.bias).includes("BEARISH")) return "rgba(255,111,125,.13)";
  return "rgba(84,200,255,.10)";
}

function renderCard(item) {
  const direction = Number(item.direction_confidence || 0) * 100;

  return `
    <article class="radar-card" style="--glow:${cardGlow(item)}">
      <div class="card-head">
        <span class="rank">#${item.rank}</span>
        <span class="status-pill status-${item.status}">${item.status}</span>
      </div>

      <h3 class="symbol">${item.symbol}</h3>
      <div class="price">${formatNumber(item.current_price, 6)}</div>

      <div class="risk-block">
        <span class="risk-label">Liquidity Pressure</span>
        <div class="risk-value">
          <strong>${formatNumber(item.radar_score, 2)}</strong>
          <span>/ 100</span>
        </div>
        <div class="risk-bar">
          <div class="risk-fill" style="width:${Math.min(item.radar_score, 100)}%"></div>
        </div>
      </div>

      <div class="metric-stack">
        <div class="metric-row">
          <span>Bias</span>
          <strong class="bias-${item.bias}">${humanBias(item.bias)}</strong>
        </div>
        <div class="metric-row">
          <span>Direction confidence</span>
          <strong>${formatNumber(direction, 1)}%</strong>
        </div>
        <div class="metric-row">
          <span>Expected event</span>
          <strong>${String(item.prediction || "UNCONFIRMED").replaceAll("_", " ")}</strong>
        </div>
      </div>

      <div class="reason-box">${explain(item)}</div>

      <div class="card-footer">
        <span>24h topology</span>
        <span>${formatAge(item.age_seconds)}</span>
      </div>
    </article>
  `;
}

function renderTableRow(item) {
  return `
    <tr>
      <td>#${item.rank}</td>
      <td><strong>${item.symbol}</strong></td>
      <td class="table-risk">${formatNumber(item.radar_score, 2)}</td>
      <td><span class="status-pill status-${item.status}">${item.status}</span></td>
      <td class="bias-${item.bias}">${humanBias(item.bias)}</td>
      <td>${formatNumber(Number(item.direction_confidence || 0) * 100, 1)}%</td>
      <td>${formatNumber(item.current_price, 6)}</td>
      <td>${formatAge(item.age_seconds)}</td>
    </tr>
  `;
}

function render(payload) {
  state.payload = payload;

  const radar = payload.radar || [];
  const highest = radar[0];
  const active = radar.filter((item) =>
    ["WATCH", "ALERT", "CRITICAL"].includes(item.status)
  ).length;

  $("engineStatus").textContent = payload.status || "UNKNOWN";
  $("engineDot").className =
    `status-dot ${(payload.status || "offline").toLowerCase()}`;

  $("lastRefresh").textContent = formatTime(payload.generated_at);
  $("symbolCount").textContent = payload.symbol_count ?? radar.length;
  $("freshestAge").textContent =
    formatAge(payload.freshest_snapshot_age_seconds);
  $("highestRisk").textContent =
    highest ? formatNumber(highest.radar_score, 2) : "—";
  $("highestSymbol").textContent =
    highest ? `${highest.symbol} leads the radar` : "Waiting for radar";
  $("activeWatches").textContent = active;

  $("radarCards").innerHTML = radar.length
    ? radar.map(renderCard).join("")
    : '<div class="loading-card">No live radar data available.</div>';

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

    const payload = await response.json();
    render(payload);
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
    $("countdown").textContent =
      `Auto refresh in ${Math.max(state.seconds, 0)}s`;

    if (state.seconds <= 0) {
      state.seconds = 15;
      loadRadar(false);
    }
  }, 1000);
}

$("refreshButton").addEventListener("click", () => loadRadar(true));

$("jsonToggle").addEventListener("click", () => {
  $("jsonPanel").classList.toggle("hidden");
});

$("jsonClose").addEventListener("click", () => {
  $("jsonPanel").classList.add("hidden");
});

loadRadar(false);
startCountdown();
