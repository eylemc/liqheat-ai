(() => {
  "use strict";

  const STORAGE_KEY = "liqheat-radar-outcome-stability-v3";

  // Directional outcomes are only meaningful when the combined Radar Score
  // reaches a minimum opportunity-quality threshold. Below this level the UI
  // must honestly display UNCONFIRMED, regardless of raw topology direction.
  const MIN_DIRECTIONAL_RADAR_SCORE = 60;

  // The backend refreshes roughly every 75 seconds while the UI polls every
  // 15 seconds. Confirmations therefore count only distinct backend snapshots,
  // never repeated reads of the same payload.
  const ENTER_CONFIDENCE = 0.64;
  const EXIT_CONFIDENCE = 0.56;
  const REVERSE_CONFIDENCE = 0.68;
  const ENTER_CONFIRMATIONS = 3;
  const EXIT_CONFIRMATIONS = 2;
  const REVERSE_CONFIRMATIONS = 3;
  const MIN_HOLD_MS = 5 * 60 * 1000;

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
      // Radar rendering must not depend on storage availability.
    }
  }

  function radarScore(item) {
    const direct = Number(item?.radar_score);
    if (Number.isFinite(direct)) return direct;

    const pressureScore = Number(item?.liquidity_pressure_score);
    if (Number.isFinite(pressureScore)) return pressureScore;

    const normalized = Number(item?.score);
    return Number.isFinite(normalized) ? normalized * 100 : 0;
  }

  function directionalEvidence(item) {
    const probabilities = item?.probabilities || {};
    const upward = Number(probabilities.short_squeeze || 0);
    const downward = Number(probabilities.long_squeeze || 0);
    const total = upward + downward;

    if (!Number.isFinite(total) || total <= 0) {
      return { candidate: "NEUTRAL", confidence: 0, upwardShare: 0.5 };
    }

    const upwardShare = upward / total;
    const downwardShare = downward / total;

    return {
      candidate: upwardShare >= downwardShare ? "UPWARD" : "DOWNWARD",
      confidence: Math.max(upwardShare, downwardShare),
      upwardShare,
    };
  }

  function normalizePrevious(previous = {}) {
    return {
      outcome: previous.outcome || "NEUTRAL",
      pending: previous.pending || null,
      pendingConfirmations: Number(previous.pendingConfirmations || 0),
      weakConfirmations: Number(previous.weakConfirmations || 0),
      changedAt: previous.changedAt || null,
      lastSnapshotKey: previous.lastSnapshotKey || null,
    };
  }

  function snapshotKey(payload, item) {
    return String(
      item?.logged_at || payload?.generated_at || payload?.last_success_at || ""
    );
  }

  function incrementPending(previous, candidate) {
    if (previous.pending === candidate) {
      return {
        pending: candidate,
        pendingConfirmations: previous.pendingConfirmations + 1,
      };
    }

    return { pending: candidate, pendingConfirmations: 1 };
  }

  function stabilizeItem(item, state, payload) {
    const symbol = String(item?.symbol || "UNKNOWN").toUpperCase();
    const score = radarScore(item);
    const evidence = directionalEvidence(item);
    const previous = normalizePrevious(state[symbol]);
    const currentSnapshotKey = snapshotKey(payload, item);

    // Hard quality gate: a low Radar Score can describe weak directional bias,
    // but it is not a tradeable directional call. Force UNCONFIRMED immediately
    // and clear any pending direction so stale state cannot leak through.
    if (score < MIN_DIRECTIONAL_RADAR_SCORE) {
      const nextState = {
        outcome: "NEUTRAL",
        pending: null,
        pendingConfirmations: 0,
        weakConfirmations: 0,
        changedAt:
          previous.outcome !== "NEUTRAL"
            ? new Date().toISOString()
            : previous.changedAt,
        lastSnapshotKey: currentSnapshotKey || previous.lastSnapshotKey,
        confidence: Number(evidence.confidence.toFixed(6)),
        upwardShare: Number(evidence.upwardShare.toFixed(6)),
        radarScore: Number(score.toFixed(6)),
        scoreGate: "BLOCK",
      };

      state[symbol] = nextState;
      return applyOutcome(item, "NEUTRAL", nextState);
    }

    // A 15-second browser poll may see the same 75-second backend snapshot
    // several times. Repeated reads must never advance the state machine.
    if (currentSnapshotKey && currentSnapshotKey === previous.lastSnapshotKey) {
      return applyOutcome(item, previous.outcome, {
        ...previous,
        confidence: evidence.confidence,
        upwardShare: evidence.upwardShare,
        radarScore: score,
        scoreGate: "PASS",
      });
    }

    let outcome = previous.outcome;
    let pending = previous.pending;
    let pendingConfirmations = previous.pendingConfirmations;
    let weakConfirmations = previous.weakConfirmations;
    const now = Date.now();
    const changedAtMs = previous.changedAt ? Date.parse(previous.changedAt) : 0;
    const holdElapsed = !changedAtMs || now - changedAtMs >= MIN_HOLD_MS;

    if (outcome === "NEUTRAL") {
      weakConfirmations = 0;

      if (evidence.confidence >= ENTER_CONFIDENCE) {
        const next = incrementPending(previous, evidence.candidate);
        pending = next.pending;
        pendingConfirmations = next.pendingConfirmations;

        if (pendingConfirmations >= ENTER_CONFIRMATIONS) {
          outcome = evidence.candidate;
          pending = null;
          pendingConfirmations = 0;
        }
      } else {
        pending = null;
        pendingConfirmations = 0;
      }
    } else if (evidence.candidate === outcome && evidence.confidence >= EXIT_CONFIDENCE) {
      // Supporting evidence resets all exit/reversal pressure.
      pending = null;
      pendingConfirmations = 0;
      weakConfirmations = 0;
    } else if (evidence.candidate !== outcome && evidence.confidence >= REVERSE_CONFIDENCE && holdElapsed) {
      // Keep displaying the established direction while the opposite case is
      // being confirmed. Do not flash NEUTRAL between every noisy observation.
      const next = incrementPending(previous, evidence.candidate);
      pending = next.pending;
      pendingConfirmations = next.pendingConfirmations;
      weakConfirmations = 0;

      if (pendingConfirmations >= REVERSE_CONFIRMATIONS) {
        outcome = evidence.candidate;
        pending = null;
        pendingConfirmations = 0;
      }
    } else if (evidence.confidence < EXIT_CONFIDENCE && holdElapsed) {
      // Only sustained weak evidence removes an established directional call.
      weakConfirmations += 1;
      pending = null;
      pendingConfirmations = 0;

      if (weakConfirmations >= EXIT_CONFIRMATIONS) {
        outcome = "NEUTRAL";
        weakConfirmations = 0;
      }
    } else {
      // Ambiguous or opposite-but-not-strong-enough evidence does not cause a
      // visual state change. Preserve the last confirmed outcome.
      pending = null;
      pendingConfirmations = 0;
      weakConfirmations = 0;
    }

    const changed = outcome !== previous.outcome;
    const nextState = {
      outcome,
      pending,
      pendingConfirmations,
      weakConfirmations,
      changedAt: changed ? new Date().toISOString() : previous.changedAt,
      lastSnapshotKey: currentSnapshotKey || previous.lastSnapshotKey,
      confidence: Number(evidence.confidence.toFixed(6)),
      upwardShare: Number(evidence.upwardShare.toFixed(6)),
      radarScore: Number(score.toFixed(6)),
      scoreGate: "PASS",
    };

    state[symbol] = nextState;
    return applyOutcome(item, outcome, nextState);
  }

  function applyOutcome(item, outcome, state) {
    const stabilized = {
      ...item,
      expected_outcome: outcome,
      outcome_stability: {
        mode: "score-gated-snapshot-state-machine-v3",
        confidence: Number(Number(state.confidence || 0).toFixed(6)),
        upward_share: Number(Number(state.upwardShare ?? 0.5).toFixed(6)),
        radar_score: Number(Number(state.radarScore || 0).toFixed(6)),
        score_gate: state.scoreGate || "BLOCK",
        minimum_directional_radar_score: MIN_DIRECTIONAL_RADAR_SCORE,
        pending_direction: state.pending || null,
        pending_confirmations: Number(state.pendingConfirmations || 0),
        weak_confirmations: Number(state.weakConfirmations || 0),
        enter_confirmations: ENTER_CONFIRMATIONS,
        exit_confirmations: EXIT_CONFIRMATIONS,
        reverse_confirmations: REVERSE_CONFIRMATIONS,
        enter_confidence: ENTER_CONFIDENCE,
        exit_confidence: EXIT_CONFIDENCE,
        reverse_confidence: REVERSE_CONFIDENCE,
        minimum_hold_seconds: MIN_HOLD_MS / 1000,
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
    const radar = payload.radar.map((item) => stabilizeItem(item, state, payload));
    saveState(state);

    return {
      ...payload,
      outcome_stability: {
        mode: "score-gated-snapshot-state-machine-v3",
        minimum_directional_radar_score: MIN_DIRECTIONAL_RADAR_SCORE,
        enter_confidence: ENTER_CONFIDENCE,
        exit_confidence: EXIT_CONFIDENCE,
        reverse_confidence: REVERSE_CONFIDENCE,
        enter_confirmations: ENTER_CONFIRMATIONS,
        exit_confirmations: EXIT_CONFIRMATIONS,
        reverse_confirmations: REVERSE_CONFIRMATIONS,
        minimum_hold_seconds: MIN_HOLD_MS / 1000,
      },
      radar,
    };
  }

  window.fetch = async (...args) => {
    const response = await originalFetch(...args);
    const requestUrl = String(args[0]?.url || args[0] || "");

    if (!requestUrl.includes("/radar")) return response;

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) return response;

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
