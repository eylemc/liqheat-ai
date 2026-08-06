(() => {
  "use strict";

  const STORAGE_KEY = "liqheat-radar-outcome-stability-v1";
  const ENTER_CONFIDENCE = 0.62;
  const HOLD_CONFIDENCE = 0.54;
  const REVERSE_CONFIDENCE = 0.66;
  const REQUIRED_CONFIRMATIONS = 2;

  const originalFetch = window.fetch.bind(window);

  function loadState() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") || {};
    } catch (_) {
      return {};
    }
  }

  function saveState(state) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (_) {
      // The radar must keep working even when storage is unavailable.
    }
  }

  function directionalEvidence(item) {
    const probabilities = item?.probabilities || {};
    const upward = Number(probabilities.short_squeeze || 0);
    const downward = Number(probabilities.long_squeeze || 0);
    const total = upward + downward;

    if (!Number.isFinite(total) || total <= 0) {
      return {
        candidate: "NEUTRAL",
        confidence: 0,
        upwardShare: 0.5,
      };
    }

    const upwardShare = upward / total;
    const downwardShare = downward / total;
    const confidence = Math.max(upwardShare, downwardShare);
    const candidate = upwardShare >= downwardShare ? "UPWARD" : "DOWNWARD";

    return {
      candidate,
      confidence,
      upwardShare,
    };
  }

  function stabilizeItem(item, state) {
    const symbol = String(item?.symbol || "UNKNOWN").toUpperCase();
    const evidence = directionalEvidence(item);
    const previous = state[symbol] || {
      outcome: "NEUTRAL",
      pending: null,
      confirmations: 0,
      changedAt: null,
    };

    let outcome = previous.outcome || "NEUTRAL";
    let pending = previous.pending || null;
    let confirmations = Number(previous.confirmations || 0);

    if (evidence.confidence < HOLD_CONFIDENCE) {
      outcome = "NEUTRAL";
      pending = null;
      confirmations = 0;
    } else if (outcome === "NEUTRAL") {
      if (evidence.confidence >= ENTER_CONFIDENCE) {
        if (pending === evidence.candidate) {
          confirmations += 1;
        } else {
          pending = evidence.candidate;
          confirmations = 1;
        }

        if (confirmations >= REQUIRED_CONFIRMATIONS) {
          outcome = evidence.candidate;
          pending = null;
          confirmations = 0;
        }
      } else {
        pending = null;
        confirmations = 0;
      }
    } else if (evidence.candidate === outcome) {
      pending = null;
      confirmations = 0;
    } else if (evidence.confidence >= REVERSE_CONFIDENCE) {
      if (pending === evidence.candidate) {
        confirmations += 1;
      } else {
        pending = evidence.candidate;
        confirmations = 1;
      }

      if (confirmations >= REQUIRED_CONFIRMATIONS) {
        outcome = evidence.candidate;
        pending = null;
        confirmations = 0;
      } else {
        outcome = "NEUTRAL";
      }
    } else {
      outcome = "NEUTRAL";
      pending = null;
      confirmations = 0;
    }

    const changed = outcome !== previous.outcome;

    state[symbol] = {
      outcome,
      pending,
      confirmations,
      changedAt: changed ? new Date().toISOString() : previous.changedAt,
      confidence: Number(evidence.confidence.toFixed(6)),
      upwardShare: Number(evidence.upwardShare.toFixed(6)),
    };

    const stabilized = {
      ...item,
      expected_outcome: outcome,
      outcome_stability: {
        mode: "neutral-plus-hysteresis",
        confidence: Number(evidence.confidence.toFixed(6)),
        upward_share: Number(evidence.upwardShare.toFixed(6)),
        pending_direction: pending,
        pending_confirmations: confirmations,
        required_confirmations: REQUIRED_CONFIRMATIONS,
        enter_confidence: ENTER_CONFIDENCE,
        reverse_confidence: REVERSE_CONFIDENCE,
      },
    };

    if (outcome === "UPWARD") {
      stabilized.raw_prediction = "SHORT_SQUEEZE";
      stabilized.prediction = "SHORT_SQUEEZE";
    } else if (outcome === "DOWNWARD") {
      stabilized.raw_prediction = "LONG_SQUEEZE";
      stabilized.prediction = "LONG_SQUEEZE";
    } else {
      stabilized.raw_prediction = "UNCONFIRMED";
      stabilized.prediction = "UNCONFIRMED";
    }

    return stabilized;
  }

  function stabilizePayload(payload) {
    if (!payload || !Array.isArray(payload.radar)) return payload;

    const state = loadState();
    const radar = payload.radar.map((item) => stabilizeItem(item, state));
    saveState(state);

    return {
      ...payload,
      outcome_stability: {
        mode: "neutral-plus-hysteresis",
        enter_confidence: ENTER_CONFIDENCE,
        hold_confidence: HOLD_CONFIDENCE,
        reverse_confidence: REVERSE_CONFIDENCE,
        required_confirmations: REQUIRED_CONFIRMATIONS,
      },
      radar,
    };
  }

  window.fetch = async (...args) => {
    const response = await originalFetch(...args);
    const requestUrl = String(args[0]?.url || args[0] || "");

    if (!requestUrl.includes("/radar")) {
      return response;
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      return response;
    }

    try {
      const payload = await response.clone().json();
      const stabilized = stabilizePayload(payload);
      const headers = new Headers(response.headers);
      headers.delete("content-length");

      return new Response(JSON.stringify(stabilized), {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    } catch (_) {
      return response;
    }
  };
})();

(() => {
  "use strict";

  const STYLE_ID = "matrixSignalLabelStyles";

  function installStyles() {
    if (document.getElementById(STYLE_ID)) return;

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .matrix-signals-label {
        display: block;
        margin: 2px 0 8px;
        color: var(--muted);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
      }
    `;
    document.head.appendChild(style);
  }

  function addLabels() {
    document.querySelectorAll(".radar-card .matrix-strip").forEach((strip) => {
      const previous = strip.previousElementSibling;
      if (previous?.classList.contains("matrix-signals-label")) return;

      const label = document.createElement("span");
      label.className = "matrix-signals-label";
      label.textContent = "Matrix Signals";
      strip.parentNode.insertBefore(label, strip);
    });
  }

  installStyles();
  addLabels();

  const observer = new MutationObserver(addLabels);
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
})();
