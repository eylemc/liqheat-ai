#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "src" / "matrix_live.py"
INDEX = ROOT / "static" / "index.html"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        if new in text:
            print(f"Already patched: {label}")
            return text
        raise SystemExit(f"Patch target not found: {label}")
    print(f"Patched: {label}")
    return text.replace(old, new, 1)


text = MATRIX.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''MATRIX_TIMEFRAMES = [\n    "1d",\n    "4h",\n    "1h",\n    "15m",\n    "5m",\n]''',
    '''MATRIX_TIMEFRAMES = [\n    "1d",\n    "4h",\n    "1h",\n    "15m",\n    "1m",\n]''',
    "Matrix timeframe 5m -> 1m",
)
text = replace_once(
    text,
    '''    "15m": 15.0,\n    "5m": 10.0,''',
    '''    "15m": 15.0,\n    "1m": 10.0,''',
    "Matrix timeframe weight 5m -> 1m",
)
MATRIX.write_text(text, encoding="utf-8")

if INDEX.exists():
    html = INDEX.read_text(encoding="utf-8")
    replacements = [
        ("Matrix Direction + 15m AI Market Risk", "1M Matrix Scalp Idea + 15m AI Market Risk", "brand subtitle"),
        ("Matrix direction with near-term trade risk", "Matrix scalp idea with near-term trade risk", "section heading"),
        ("Matrix Direction", "Scalp Idea", "table column"),
        ("Matrix direction", "Scalp idea", "table column lowercase"),
    ]
    for old, new, label in replacements:
        if old in html:
            html = html.replace(old, new)
            print(f"Patched: {label}")
    INDEX.write_text(html, encoding="utf-8")

print("Matrix 1m Scalp Idea V1 installed.")
print("Scalp mapping: 1m BUY -> LONG, 1m SELL -> SHORT.")
