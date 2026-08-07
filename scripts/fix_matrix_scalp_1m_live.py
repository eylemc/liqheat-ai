#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def patch_matrix_live():
    path = ROOT / "src" / "matrix_live.py"
    text = path.read_text(encoding="utf-8")
    original = text

    text = re.sub(
        r'MATRIX_TIMEFRAMES\s*=\s*\[\s*"1d",\s*"4h",\s*"1h",\s*"15m",\s*"5m",?\s*\]',
        'MATRIX_TIMEFRAMES = [\n    "1d",\n    "4h",\n    "1h",\n    "15m",\n    "1m",\n]',
        text,
        count=1,
        flags=re.S,
    )

    text = re.sub(
        r'TIMEFRAME_WEIGHTS\s*=\s*\{\s*"1d":\s*30\.0,\s*"4h":\s*25\.0,\s*"1h":\s*20\.0,\s*"15m":\s*15\.0,\s*"5m":\s*10\.0,?\s*\}',
        'TIMEFRAME_WEIGHTS = {\n    "1d": 30.0,\n    "4h": 25.0,\n    "1h": 20.0,\n    "15m": 15.0,\n    "1m": 10.0,\n}',
        text,
        count=1,
        flags=re.S,
    )

    if text == original:
        raise SystemExit("matrix_live.py patch target not found or already patched")

    backup = path.with_suffix(path.suffix + ".before-scalp-1m")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print("Patched Matrix timeframes: 1D, 4H, 1H, 15M, 1M")


def patch_dashboard_js():
    path = ROOT / "static" / "matrix_ai_radar.js"
    text = path.read_text(encoding="utf-8")
    original = text

    text = text.replace(
        '["1d", "4h", "1h", "15m", "5m"]',
        '["1d", "4h", "1h", "15m", "1m"]',
    )

    marker = '''function matrixDirection(item) {
  return matrixData(item)?.direction_label || "UNAVAILABLE";
}
'''
    scalp_fn = '''function matrixDirection(item) {
  return matrixData(item)?.direction_label || "UNAVAILABLE";
}
function scalpIdea(item) {
  const signal = String(matrixData(item)?.timeframes?.["1m"]?.trend_label || "").toUpperCase();
  if (signal === "BUY") return { label: "LONG", className: "scalp-long" };
  if (signal === "SELL") return { label: "SHORT", className: "scalp-short" };
  return { label: "WAIT", className: "scalp-wait" };
}
'''
    if "function scalpIdea(item)" not in text:
        if marker not in text:
            raise SystemExit("matrixDirection marker not found in dashboard JS")
        text = text.replace(marker, scalp_fn, 1)

    old_hero = '''<div class="matrix-direction-hero"><span>Matrix direction</span><strong class="matrix-direction-${matrixDirection(item)}">${matrixDirection(item)}</strong></div>'''
    new_hero = '''<div class="matrix-direction-hero"><span>SCALP IDEA · 1M MATRIX</span><strong class="${scalpIdea(item).className}">${scalpIdea(item).label}</strong></div>'''
    text = text.replace(old_hero, new_hero)

    old_table = '''<td class="matrix-direction-${matrixDirection(item)}"><strong>${matrixDirection(item)}</strong></td>'''
    new_table = '''<td class="${scalpIdea(item).className}"><strong>${scalpIdea(item).label}</strong></td>'''
    text = text.replace(old_table, new_table)

    style_marker = '''    .matrix-direction-hero span{color:var(--muted);font-size:11px}.matrix-direction-hero strong{font-size:16px;letter-spacing:.03em}
'''
    style_new = style_marker + '''    .scalp-long{color:#4ce5a6}.scalp-short{color:#ff6f7d}.scalp-wait{color:var(--muted)}
'''
    if ".scalp-long" not in text:
        text = text.replace(style_marker, style_new, 1)

    if text == original:
        raise SystemExit("matrix_ai_radar.js patch target not found or already patched")

    backup = path.with_suffix(path.suffix + ".before-scalp-1m")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print("Patched dashboard: SCALP IDEA LONG/SHORT from 1m Matrix")


def patch_index_cache_buster():
    path = ROOT / "static" / "index.html"
    text = path.read_text(encoding="utf-8")
    original = text

    # The live installer may have inserted matrix_ai_radar.js with any version.
    if "matrix_ai_radar.js" in text:
        text = re.sub(
            r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
            '/static/matrix_ai_radar.js?v=3',
            text,
        )
    else:
        # Fallback: replace app.js with the Matrix AI dashboard script.
        text = re.sub(
            r'<script src="/static/app\.js(?:\?v=\d+)?"></script>',
            '<script src="/static/matrix_ai_radar.js?v=3"></script>',
            text,
            count=1,
        )

    if text != original:
        backup = path.with_suffix(path.suffix + ".before-scalp-1m")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print("Bumped dashboard asset version to v=3")
    else:
        print("Index cache-buster already current")


def patch_table_header():
    path = ROOT / "static" / "index.html"
    text = path.read_text(encoding="utf-8")
    original = text
    text = text.replace("<th>Matrix direction</th>", "<th>Scalp idea</th>")
    text = text.replace("<th>Matrix</th>", "<th>Scalp idea</th>")
    if text != original:
        path.write_text(text, encoding="utf-8")
        print("Patched table header: Scalp idea")


def main():
    patch_matrix_live()
    patch_dashboard_js()
    patch_index_cache_buster()
    patch_table_header()
    print("Done. Restart src.textara_api and hard-refresh the browser.")


if __name__ == "__main__":
    main()
