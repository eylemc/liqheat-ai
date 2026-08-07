#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "matrix_ai_radar.js"
INDEX = ROOT / "static" / "index.html"

RESOLVER = r'''
function resolveHeatmapModalLabelCollisions() {
  const modal = document.getElementById("liqHeatmapModal");
  if (!modal || modal.classList.contains("hidden")) return;

  // Never show the duplicate PRICE LEVELS helper labels.
  modal.querySelectorAll(".liq-modal-side-label").forEach((el) => {
    el.style.setProperty("display", "none", "important");
  });

  const map = modal.querySelector(".liq-modal-map");
  const current = modal.querySelector(".liq-modal-current-badge, .liq-modal-price-label");
  if (!map) return;

  const mapRect = map.getBoundingClientRect();
  const currentRect = current?.getBoundingClientRect() || null;
  const centerY = mapRect.top + mapRect.height / 2;
  const centerGuardPx = 34;
  const minGapPx = 25;

  const allLabels = Array.from(
    modal.querySelectorAll(".liq-modal-price-tag, .liq-modal-level em")
  );

  // Reset first, because the modal can be reopened after a resize/refresh.
  allLabels.forEach((el) => {
    el.style.removeProperty("display");
    el.style.removeProperty("visibility");
  });

  const intersects = (a, b, pad = 0) => !(
    a.right + pad < b.left ||
    a.left - pad > b.right ||
    a.bottom + pad < b.top ||
    a.top - pad > b.bottom
  );

  // First remove anything in the current-price guard zone or touching current badge.
  allLabels.forEach((el) => {
    const r = el.getBoundingClientRect();
    const y = r.top + r.height / 2;
    if (Math.abs(y - centerY) < centerGuardPx || (currentRect && intersects(r, currentRect, 8))) {
      el.style.setProperty("display", "none", "important");
    }
  });

  // Then resolve collisions independently on each half. Stronger labels occur first in DOM
  // often enough, but sorting by visual Y makes the result deterministic.
  ["below", "above"].forEach((side) => {
    const selector = side === "below"
      ? ".liq-modal-below .liq-modal-price-tag, .liq-modal-below em"
      : ".liq-modal-above .liq-modal-price-tag, .liq-modal-above em";
    const labels = Array.from(modal.querySelectorAll(selector))
      .filter((el) => getComputedStyle(el).display !== "none")
      .sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);

    let lastRect = null;
    for (const el of labels) {
      const r = el.getBoundingClientRect();
      if (lastRect && r.top < lastRect.bottom + minGapPx) {
        el.style.setProperty("display", "none", "important");
        continue;
      }
      lastRect = r;
    }
  });
}
'''

EXTRA_CSS = r'''
    .liq-modal-side-label{display:none!important}
'''


def patch_js() -> None:
    text = JS.read_text(encoding="utf-8")
    original = text

    if "function resolveHeatmapModalLabelCollisions()" not in text:
        marker = "function ensureHeatmapModal()"
        pos = text.find(marker)
        if pos < 0:
            raise SystemExit("ensureHeatmapModal() marker not found")
        text = text[:pos] + RESOLVER + "\n" + text[pos:]

    # Call resolver every time modal content is rendered.
    call = "requestAnimationFrame(resolveHeatmapModalLabelCollisions);"
    if call not in text:
        pattern = re.compile(
            r'(document\.getElementById\("liqHeatmapModalBody"\)\.innerHTML\s*=\s*expandedLiquidationHeatmap\(item\);)'
        )
        if not pattern.search(text):
            raise SystemExit("modal body render line not found")
        text = pattern.sub(r'\1\n  ' + call, text, count=1)

    # Re-run after resize while the modal is open.
    resize_call = 'window.addEventListener("resize", () => requestAnimationFrame(resolveHeatmapModalLabelCollisions));'
    if resize_call not in text:
        marker = "function closeHeatmapModal()"
        pos = text.find(marker)
        if pos < 0:
            raise SystemExit("closeHeatmapModal() marker not found")
        text = text[:pos] + resize_call + "\n\n" + text[pos:]

    # Hard-hide duplicate helper labels even before resolver runs.
    if ".liq-modal-side-label{display:none!important}" not in text:
        marker = "  `;\n  document.head.appendChild(style);"
        pos = text.find(marker)
        if pos < 0:
            raise SystemExit("style template end marker not found")
        text = text[:pos] + EXTRA_CSS + text[pos:]

    if text != original:
        backup = JS.with_suffix(".js.before-heatmap-overlap-hard-v2")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        JS.write_text(text, encoding="utf-8")
        print("Patched hard heatmap modal collision resolver V2")
    else:
        print("Heatmap modal collision resolver V2 already installed")


def bump_cache() -> None:
    text = INDEX.read_text(encoding="utf-8")
    original = text
    text = re.sub(
        r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
        '/static/matrix_ai_radar.js?v=14',
        text,
    )
    if text != original:
        INDEX.write_text(text, encoding="utf-8")
        print("Bumped Matrix AI Radar JS asset to v=14")


def main() -> None:
    patch_js()
    bump_cache()
    print("Done. Hard-refresh browser. API restart is not required.")


if __name__ == "__main__":
    main()
