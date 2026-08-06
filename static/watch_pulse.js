(() => {
  "use strict";

  const STYLE_ID = "watchPulseStyles";

  function installStyles() {
    if (document.getElementById(STYLE_ID)) return;

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      @keyframes liqheat-watch-pulse {
        0%, 100% {
          border-color: rgba(255, 202, 97, .18);
          box-shadow:
            0 0 0 0 rgba(255, 202, 97, 0),
            0 18px 48px rgba(0, 0, 0, .26),
            inset 0 0 0 1px rgba(255, 255, 255, .015);
        }
        50% {
          border-color: rgba(255, 202, 97, .62);
          box-shadow:
            0 0 0 4px rgba(255, 202, 97, .08),
            0 0 34px rgba(255, 202, 97, .23),
            0 18px 48px rgba(0, 0, 0, .30),
            inset 0 0 26px rgba(255, 202, 97, .035);
        }
      }

      @keyframes liqheat-watch-badge-pulse {
        0%, 100% {
          filter: brightness(1);
          box-shadow: 0 0 0 rgba(255, 202, 97, 0);
        }
        50% {
          filter: brightness(1.22);
          box-shadow: 0 0 18px rgba(255, 202, 97, .34);
        }
      }

      .radar-card.watch-active {
        position: relative;
        animation: liqheat-watch-pulse 2.4s ease-in-out infinite;
        will-change: border-color, box-shadow;
      }

      .radar-card.watch-active::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        pointer-events: none;
        background: radial-gradient(
          circle at 82% 8%,
          rgba(255, 202, 97, .08),
          transparent 32%
        );
      }

      .radar-card.watch-active .opportunity-WATCH {
        animation: liqheat-watch-badge-pulse 2.4s ease-in-out infinite;
      }

      @media (prefers-reduced-motion: reduce) {
        .radar-card.watch-active,
        .radar-card.watch-active .opportunity-WATCH {
          animation: none;
        }

        .radar-card.watch-active {
          border-color: rgba(255, 202, 97, .58);
          box-shadow: 0 0 28px rgba(255, 202, 97, .18);
        }
      }
    `;

    document.head.appendChild(style);
  }

  function markWatchCards() {
    document.querySelectorAll(".radar-card").forEach((card) => {
      const pill = card.querySelector(".status-pill");
      const isWatch = pill?.textContent?.trim().toUpperCase() === "WATCH";
      card.classList.toggle("watch-active", Boolean(isWatch));
    });
  }

  function start() {
    installStyles();
    markWatchCards();

    const radarCards = document.getElementById("radarCards");
    if (!radarCards) return;

    const observer = new MutationObserver(markWatchCards);
    observer.observe(radarCards, {
      childList: true,
      subtree: true,
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
