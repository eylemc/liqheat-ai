#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "textara_api.py"

MARKER = 'reason": "INSUFFICIENT_1H_HISTORY"'
NEEDLE = 'def build_ai_market_risk(symbol: str) -> dict[str, Any]:\n    requested = str(symbol).upper()\n'
INSERT = '''def build_ai_market_risk(symbol: str) -> dict[str, Any]:
    requested = str(symbol).upper()

    # XAU/XAG currently have no usable historical 1h topology stream.
    # Do not present stale/test-derived 15m Market Heat as a live signal.
    if requested in {"XAUUSDT", "XAGUSDT"}:
        return {
            "available": False,
            "reason": "INSUFFICIENT_1H_HISTORY",
            "symbol": requested,
            "horizon_minutes": 15,
        }
'''

text = TARGET.read_text(encoding="utf-8")

if MARKER in text:
    print("Already patched: XAU/XAG Market Heat forced to N/A until 1h history exists.")
    raise SystemExit(0)

if NEEDLE not in text:
    raise RuntimeError(
        "Could not find build_ai_market_risk() insertion point in src/textara_api.py"
    )

text = text.replace(NEEDLE, INSERT, 1)
TARGET.write_text(text, encoding="utf-8")

print("Patched: src/textara_api.py")
print("XAUUSDT/XAGUSDT Near-Term Market Heat will return N/A.")
print("Reason: INSUFFICIENT_1H_HISTORY")
print("BTC/ETH/SOL behavior is unchanged.")
