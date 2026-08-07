#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "static" / "index.html"
JS = ROOT / "static" / "matrix_ai_radar.js"


def patch_index() -> None:
    text = INDEX.read_text(encoding="utf-8")
    original = text

    # Page title + visible product name. Tolerate all previous naming variants.
    text = re.sub(
        r"<title>[^<]*(?:Radar)[^<]*</title>",
        "<title>LiqHeat Radar</title>",
        text,
        count=1,
        flags=re.I,
    )
    text = re.sub(
        r"<h1>\s*(?:MATRIX|LIQHEAT(?: AI)?)\s+RADAR\s*</h1>",
        "<h1>LIQHEAT RADAR</h1>",
        text,
        count=1,
        flags=re.I,
    )

    # Remove the brand tagline under the title, regardless of the wording
    # introduced by previous installers.
    text = re.sub(
        r'(<div class="brand-row">.*?<h1>LIQHEAT RADAR</h1>)\s*<p>.*?</p>',
        r"\1",
        text,
        count=1,
        flags=re.S | re.I,
    )

    # Remove the entire four-card summary block.
    text, n = re.subn(
        r'\s*<section class="summary-grid">.*?</section>\s*',
        "\n\n",
        text,
        count=1,
        flags=re.S | re.I,
    )
    if n:
        print("Removed summary-grid block")
    else:
        print("Summary-grid already absent")

    # Cache-bust current dashboard JS if present.
    text = re.sub(
        r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
        '/static/matrix_ai_radar.js?v=12',
        text,
    )

    if text != original:
        backup = INDEX.with_suffix(".html.before-liqheat-header-v1")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        INDEX.write_text(text, encoding="utf-8")
        print("Patched index: LIQHEAT RADAR, no tagline, no summary cards")
    else:
        print("Index already simplified")


def patch_js() -> None:
    text = JS.read_text(encoding="utf-8")
    original = text

    # normalizeHeadings() must not recreate the removed tagline.
    text = re.sub(
        r"\s*const subtitle = document\.querySelector\('\.brand-row p'\);\s*\n\s*if \(subtitle\) subtitle\.textContent = [^;]+;",
        "",
        text,
        count=1,
    )

    # Summary elements no longer exist. Remove their render writes so the
    # dashboard does not throw on null.textContent.
    summary_ids = (
        "symbolCount",
        "freshestAge",
        "highestRisk",
        "highestSymbol",
        "activeWatches",
    )
    lines = text.splitlines()
    filtered: list[str] = []
    removed = 0
    for line in lines:
        if any(f'$("{element_id}")' in line for element_id in summary_ids):
            removed += 1
            continue
        filtered.append(line)
    text = "\n".join(filtered) + ("\n" if original.endswith("\n") else "")

    if text != original:
        backup = JS.with_suffix(".js.before-liqheat-header-v1")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        JS.write_text(text, encoding="utf-8")
        print(f"Patched dashboard JS; removed {removed} obsolete summary writes")
    else:
        print("Dashboard JS already compatible with removed summary cards")


def verify() -> None:
    index = INDEX.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    checks = {
        "brand_liqheat_radar": "<h1>LIQHEAT RADAR</h1>" in index,
        "tagline_removed": not re.search(r'<div class="brand-row">.*?<h1>LIQHEAT RADAR</h1>\s*<p>', index, re.S),
        "summary_removed": 'class="summary-grid"' not in index,
        "summary_js_removed": all(f'$("{x}")' not in js for x in ["symbolCount", "freshestAge", "highestRisk", "highestSymbol", "activeWatches"]),
    }
    print("VERIFY:", checks)
    if not all(checks.values()):
        raise SystemExit("Verification failed")


def main() -> None:
    patch_index()
    patch_js()
    verify()
    print("LiqHeat Radar header simplification installed. Hard-refresh browser; API restart is not required.")


if __name__ == "__main__":
    main()
