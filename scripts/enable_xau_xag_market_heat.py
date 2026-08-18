#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    ROOT / "scripts" / "build_market_risk_radar_v1.py": [
        (
            'SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")',
            'SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT", "XAGUSDT")',
        ),
    ],
    ROOT / "scripts" / "build_market_risk_radar_v2_adaptive_walkforward.py": [
        (
            'SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")',
            'SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT", "XAGUSDT")',
        ),
    ],
    ROOT / "scripts" / "build_position_guardian_v4_koinvizyon_matrix.py": [
        (
            'SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]',
            'SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT", "XAGUSDT"]',
        ),
    ],
    ROOT / "src" / "market_risk_live.py": [
        (
            'SUPPORTED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}',
            'SUPPORTED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT", "XAGUSDT"}',
        ),
    ],
    ROOT / "src" / "textara_api.py": [
        (
            'if requested not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:',
            'if requested not in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT", "XAGUSDT"}:',
        ),
    ],
}


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if new in text:
            print(f"Already patched: {path.relative_to(ROOT)}")
            continue
        if old not in text:
            raise RuntimeError(f"Expected source not found in {path}: {old}")
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"Patched: {path.relative_to(ROOT)}")


for file_path, replacements in FILES.items():
    patch_file(file_path, replacements)

print("\nXAUUSDT/XAGUSDT Market Heat source support enabled.")
print("IMPORTANT: retrain/export the V2 live model bundle before restarting Radar.")
print("Run:")
print("  python scripts/export_market_risk_radar_v2_live.py")
print("  python scripts/build_market_risk_v2_live_reference.py")
