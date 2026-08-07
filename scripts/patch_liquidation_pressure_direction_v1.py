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

    # Add helpers once.
    if "function liquidationPressureDirection(item)" not in text:
        marker = "function liquidityPressure(item) {"
        idx = text.find(marker)
        if idx < 0:
            raise SystemExit("liquidityPressure() helper not found")
        helper = '''function liquidationPressureDirection(item) {\n  const raw = String(item.raw_prediction || item.prediction || \"\").toUpperCase();\n  if (raw === \"SHORT_SQUEEZE\") return \"UP\";\n  if (raw === \"LONG_SQUEEZE\") return \"DOWN\";\n  return \"N/A\";\n}\nfunction liquidationPressureHtml(item) {\n  const value = formatNumber(liquidityPressure(item), 0);\n  const direction = liquidationPressureDirection(item);\n  if (direction === \"UP\") return `<span class=\"liq-pressure liq-pressure-up\"><b>↑</b><strong>${value}</strong></span>`;\n  if (direction === \"DOWN\") return `<span class=\"liq-pressure liq-pressure-down\"><b>↓</b><strong>${value}</strong></span>`;\n  return `<span class=\"liq-pressure liq-pressure-na\"><strong>${value}</strong></span>`;\n}\n\n'''
        text = text[:idx] + helper + text[idx:]

    # Card metric row.
    text = re.sub(
        r'<div class="metric-row"><span>Liquidity pressure</span><strong>\$\{formatNumber\(liquidityPressure\(item\),\s*2\)\}</strong></div>',
        '<div class="metric-row"><span>Liquidation Pressure</span>${liquidationPressureHtml(item)}</div>',
        text,
    )

    # Table cell.
    text = re.sub(
        r'<td>\$\{formatNumber\(liquidityPressure\(item\),\s*2\)\}</td>',
        '<td>${liquidationPressureHtml(item)}</td>',
        text,
    )

    # Any visible table/header wording left in JS.
    text = text.replace("Liquidity pressure", "Liquidation Pressure")
    text = text.replace("LIQUIDITY PRESSURE", "LIQUIDATION PRESSURE")

    # Styling in dynamic style block.
    if ".liq-pressure{" not in text:
        style_marker = "    .scalp-long{color:#4ce5a6}.scalp-short{color:#ff6f7d}.scalp-na{color:var(--muted)}"
        styles = style_marker + '''\n    .liq-pressure{display:inline-flex;align-items:center;justify-content:flex-end;gap:7px;font-variant-numeric:tabular-nums}\n    .liq-pressure b{font-size:22px;line-height:.75;font-weight:950;text-shadow:0 0 12px currentColor}\n    .liq-pressure strong{font-size:12px;font-weight:900;color:inherit}\n    .liq-pressure-up{color:#4ce5a6}\n    .liq-pressure-down{color:#ff6f7d}\n    .liq-pressure-na{color:var(--muted)}'''
        if style_marker not in text:
            raise SystemExit("dynamic style marker not found")
        text = text.replace(style_marker, styles, 1)

    if text == original:
        print("JS already patched or targets not found")
    else:
        backup = JS.with_suffix(".js.before-liquidation-pressure-direction-v1")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        JS.write_text(text, encoding="utf-8")
        print("Patched card/table Liquidation Pressure direction")


def patch_index() -> None:
    text = INDEX.read_text(encoding="utf-8")
    original = text
    text = text.replace("Liquidity pressure", "Liquidation Pressure")
    text = text.replace("LIQUIDITY PRESSURE", "LIQUIDATION PRESSURE")
    text = re.sub(
        r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
        '/static/matrix_ai_radar.js?v=14',
        text,
    )
    if text != original:
        INDEX.write_text(text, encoding="utf-8")
        print("Patched index label and bumped asset to v=14")
    else:
        print("Index already patched")


def verify() -> None:
    js = JS.read_text(encoding="utf-8")
    index = INDEX.read_text(encoding="utf-8")
    checks = {
        "direction_helper": "function liquidationPressureDirection(item)" in js,
        "up_arrow": "liq-pressure-up" in js and "↑" in js,
        "down_arrow": "liq-pressure-down" in js and "↓" in js,
        "renamed": "Liquidation Pressure" in js or "Liquidation Pressure" in index,
        "cache_v14": "matrix_ai_radar.js?v=14" in index,
    }
    print("VERIFY:", checks)
    if not all(checks.values()):
        raise SystemExit("Verification failed")


def main() -> None:
    patch_js()
    patch_index()
    verify()
    print("Done. Hard-refresh browser; API restart is not required.")


if __name__ == "__main__":
    main()
