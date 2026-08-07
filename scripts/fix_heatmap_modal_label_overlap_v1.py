#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "matrix_ai_radar.js"
INDEX = ROOT / "static" / "index.html"


def patch_js() -> None:
    text = JS.read_text(encoding="utf-8")
    original = text

    # 1) Never label liquidation levels too close to the current-price line.
    old = '''    const candidates = rawLevels
      .filter((level) => level.side === side)
      .sort((a, b) => b.intensity - a.intensity);'''
    new = '''    const candidates = rawLevels
      .filter((level) => level.side === side)
      // Keep a protected zone around current price so the current-price badge
      // can never collide with a liquidation price label.
      .filter((level) => Math.abs(level.position - 0.5) >= 0.065)
      .sort((a, b) => b.intensity - a.intensity);'''
    if old in text:
        text = text.replace(old, new, 1)
        print("Patched current-price no-label zone")
    elif "Math.abs(level.position - 0.5) >= 0.065" in text:
        print("Current-price no-label zone already installed")
    else:
        raise SystemExit("Could not find heatmap label candidate block")

    # 2) Increase vertical separation between visible labels.
    text = text.replace(
        "Math.abs(other.position - level.position) >= 0.042",
        "Math.abs(other.position - level.position) >= 0.055",
    )

    # 3) Remove redundant PRICE LEVELS headings inside the graph.
    text = re.sub(
        r'\n\s*<div class="liq-modal-side-label liq-modal-side-label-below">PRICE LEVELS</div>\n\s*<div class="liq-modal-side-label liq-modal-side-label-above">PRICE LEVELS</div>',
        "",
        text,
        count=1,
    )

    # 4) Slightly move price tags away from the center line for readability.
    text = text.replace(
        '.liq-modal-below .liq-modal-price-tag{right:51.1%!important;transform:translateX(-6px)!important}',
        '.liq-modal-below .liq-modal-price-tag{right:52%!important;transform:translateX(-8px)!important}',
    )
    text = text.replace(
        '.liq-modal-above .liq-modal-price-tag{left:51.1%!important;transform:translateX(6px)!important}',
        '.liq-modal-above .liq-modal-price-tag{left:52%!important;transform:translateX(8px)!important}',
    )

    if text != original:
        backup = JS.with_suffix(".js.before-modal-overlap-fix-v1")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        JS.write_text(text, encoding="utf-8")
        print("Patched heatmap modal label overlap")
    else:
        print("No JS changes needed")


def bump_cache() -> None:
    text = INDEX.read_text(encoding="utf-8")
    original = text
    text = re.sub(
        r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
        '/static/matrix_ai_radar.js?v=13',
        text,
    )
    if text != original:
        INDEX.write_text(text, encoding="utf-8")
        print("Bumped Matrix AI Radar asset to v=13")
    else:
        print("Asset cache already v=13")


def main() -> None:
    patch_js()
    bump_cache()
    print("Done. Hard-refresh browser; API restart is not required.")


if __name__ == "__main__":
    main()
