#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "static" / "index.html"
JS = ROOT / "static" / "matrix_ai_radar.js"


def main() -> None:
    html = INDEX.read_text(encoding="utf-8")
    original_html = html

    # Page title + visible brand. Tolerates MATRIX AI RADAR / LIQHEAT AI RADAR / prior variants.
    html = re.sub(r"<title>[^<]*Radar[^<]*</title>", "<title>LiqHeat Radar</title>", html, count=1, flags=re.I)
    html = re.sub(r"<h1>\s*(?:MATRIX|LIQHEAT)(?:\s+AI)?\s+RADAR\s*</h1>", "<h1>LIQHEAT RADAR</h1>", html, count=1, flags=re.I)

    # Remove any tagline directly under the brand h1, whatever previous patch wrote there.
    html = re.sub(
        r"(<h1>LIQHEAT RADAR</h1>)\s*<p>.*?</p>",
        r"\1",
        html,
        count=1,
        flags=re.I | re.S,
    )

    # Remove summary block if it still exists. Non-greedy up to the next section-heading.
    html = re.sub(
        r"\s*<section\s+class=\"summary-grid\">.*?</section>\s*(?=<section\s+class=\"section-heading\">)",
        "\n\n    ",
        html,
        count=1,
        flags=re.S,
    )

    # Force a fresh JS fetch without depending on the current version number.
    html = re.sub(
        r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
        '/static/matrix_ai_radar.js?v=12',
        html,
    )

    if html != original_html:
        INDEX.write_text(html, encoding="utf-8")
        print("Patched index: LIQHEAT RADAR, no tagline, no summary block")
    else:
        print("Index already has requested structure")

    # Prevent JS normalizeHeadings() from recreating the old tagline.
    if JS.exists():
        js = JS.read_text(encoding="utf-8")
        original_js = js
        js = re.sub(
            r"\s*const subtitle = document\.querySelector\('\.brand-row p'\);\s*if \(subtitle\) subtitle\.textContent = [^;]+;",
            "",
            js,
            count=1,
        )
        if js != original_js:
            JS.write_text(js, encoding="utf-8")
            print("Removed JS tagline recreation")

    check = INDEX.read_text(encoding="utf-8")
    verify = {
        "brand_liqheat_radar": "<h1>LIQHEAT RADAR</h1>" in check,
        "tagline_removed": "1M Matrix Scalp Idea + 15m AI Market Risk" not in check and "Matrix + Topology Intelligence" not in check,
        "summary_removed": 'class="summary-grid"' not in check,
    }
    print("VERIFY:", verify)
    if not all(verify.values()):
        raise SystemExit("Verification failed")


if __name__ == "__main__":
    main()
