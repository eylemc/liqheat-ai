#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def patch_matrix_live():
    path = ROOT / "src" / "matrix_live.py"
    text = path.read_text(encoding="utf-8")
    original = text

    # Normalize Matrix timeframe set even if formatting differs.
    text, n1 = re.subn(
        r'MATRIX_TIMEFRAMES\s*=\s*\[(.*?)\]',
        'MATRIX_TIMEFRAMES = [\n    "1d",\n    "4h",\n    "1h",\n    "15m",\n    "1m",\n]',
        text,
        count=1,
        flags=re.S,
    )

    text, n2 = re.subn(
        r'TIMEFRAME_WEIGHTS\s*=\s*\{(.*?)\}',
        'TIMEFRAME_WEIGHTS = {\n    "1d": 30.0,\n    "4h": 25.0,\n    "1h": 20.0,\n    "15m": 15.0,\n    "1m": 10.0,\n}',
        text,
        count=1,
        flags=re.S,
    )

    if n1 == 0 or n2 == 0:
        raise SystemExit("Could not locate MATRIX_TIMEFRAMES/TIMEFRAME_WEIGHTS in src/matrix_live.py")

    if text != original:
        backup = path.with_suffix(path.suffix + ".before-scalp-1m")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print("Patched Matrix timeframes: 1D, 4H, 1H, 15M, 1M")
    else:
        print("Matrix timeframes already correct: 1D, 4H, 1H, 15M, 1M")


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

    # Replace hero regardless of whitespace variations.
    text = re.sub(
        r'<div class="matrix-direction-hero"><span>Matrix direction</span><strong[^>]*>\$\{matrixDirection\(item\)\}</strong></div>',
        '<div class="matrix-direction-hero"><span>SCALP IDEA · 1M MATRIX</span><strong class="${scalpIdea(item).className}">${scalpIdea(item).label}</strong></div>',
        text,
        count=1,
    )

    text = re.sub(
        r'<td class="matrix-direction-\$\{matrixDirection\(item\)\}"><strong>\$\{matrixDirection\(item\)\}</strong></td>',
        '<td class="${scalpIdea(item).className}"><strong>${scalpIdea(item).label}</strong></td>',
        text,
        count=1,
    )

    style_marker = '    .matrix-direction-hero span{color:var(--muted);font-size:11px}.matrix-direction-hero strong{font-size:16px;letter-spacing:.03em}\n'
    if ".scalp-long" not in text:
        if style_marker in text:
            text = text.replace(
                style_marker,
                style_marker + '    .scalp-long{color:#4ce5a6}.scalp-short{color:#ff6f7d}.scalp-wait{color:var(--muted)}\n',
                1,
            )

    if text != original:
        backup = path.with_suffix(path.suffix + ".before-scalp-1m")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print("Patched dashboard: SCALP IDEA LONG/SHORT from 1m Matrix")
    else:
        print("Dashboard scalp idea already patched")


def patch_index_cache_buster():
    path = ROOT / "static" / "index.html"
    text = path.read_text(encoding="utf-8")
    original = text

    if "matrix_ai_radar.js" in text:
        text = re.sub(
            r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
            '/static/matrix_ai_radar.js?v=4',
            text,
        )
    else:
        text = re.sub(
            r'<script src="/static/app\.js(?:\?v=\d+)?"></script>',
            '<script src="/static/matrix_ai_radar.js?v=4"></script>',
            text,
            count=1,
        )

    text = text.replace("<th>Matrix direction</th>", "<th>Scalp idea</th>")
    text = text.replace("<th>Matrix</th>", "<th>Scalp idea</th>")

    if text != original:
        backup = path.with_suffix(path.suffix + ".before-scalp-1m")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print("Patched index and bumped dashboard asset to v=4")
    else:
        print("Index/cache-buster already current")


def verify():
    matrix = (ROOT / "src" / "matrix_live.py").read_text(encoding="utf-8")
    js = (ROOT / "static" / "matrix_ai_radar.js").read_text(encoding="utf-8")
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    checks = {
        "backend_1m": '"1m"' in matrix and '"5m"' not in re.search(r'MATRIX_TIMEFRAMES\s*=\s*\[(.*?)\]', matrix, re.S).group(1),
        "frontend_1m": '["1d", "4h", "1h", "15m", "1m"]' in js,
        "scalp_idea": "function scalpIdea(item)" in js and "SCALP IDEA · 1M MATRIX" in js,
        "cache_v4": "matrix_ai_radar.js?v=4" in index,
    }
    print("VERIFY:", checks)
    if not all(checks.values()):
        raise SystemExit("Verification failed")


def main():
    patch_matrix_live()
    patch_dashboard_js()
    patch_index_cache_buster()
    verify()
    print("Done. Restart src.textara_api and hard-refresh the browser.")


if __name__ == "__main__":
    main()
