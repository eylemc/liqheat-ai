#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
js = ROOT / "static" / "matrix_ai_radar.js"
index = ROOT / "static" / "index.html"

text = js.read_text(encoding="utf-8")
original = text

# Risk band already appears inside the risk panel; remove duplicate top-right pill.
text = text.replace(
    '<div class="card-head"><span class="rank">#${item.rank}</span><span class="risk-band-pill risk-${riskClass(item)}">${band}</span></div>',
    '<div class="card-head"><span class="rank">#${item.rank}</span></div>',
)

# Risk score is now an empirical percentile. Show it as a clean integer.
text = text.replace('formatNumber(score, 2)', 'Math.round(score)')
text = text.replace('formatNumber(riskScore(lowest), 2)', 'Math.round(riskScore(lowest))')

# Clarify what the score means.
text = text.replace(
    'Direction-independent near-term trade risk',
    '15m risk percentile vs historical calibration',
)

if text != original:
    js.write_text(text, encoding="utf-8")
    print("Patched UI: single risk badge + integer percentile score")
else:
    print("UI already patched or expected strings not found")

html = index.read_text(encoding="utf-8")
old_html = html
if "matrix_ai_radar.js" in html:
    html = re.sub(
        r'/static/matrix_ai_radar\.js(?:\?v=\d+)?',
        '/static/matrix_ai_radar.js?v=6',
        html,
    )
if html != old_html:
    index.write_text(html, encoding="utf-8")
    print("Bumped Matrix AI Radar asset to v=6")

print("Done.")
