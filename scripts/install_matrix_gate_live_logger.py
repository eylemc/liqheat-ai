#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

HOME = Path.home()
REPO = HOME / "liqheat-ai"
UNIT_DIR = HOME / ".config" / "systemd" / "user"
UNIT = UNIT_DIR / "liqheat-matrix-gate-logger.service"

UNIT_TEXT = f"""[Unit]
Description=LiqHeat Matrix Regime Gate Research Logger
After=network-online.target liqheat-radar-api.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={REPO}
Environment=PYTHONUNBUFFERED=1
ExecStart={REPO}/.venv/bin/python {REPO}/scripts/matrix_gate_live_logger.py --api http://127.0.0.1:8000/radar --db data/research/matrix_gate_live/matrix_gate_live.sqlite3 --interval 60
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""


def main() -> int:
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    UNIT.write_text(UNIT_TEXT, encoding="utf-8")
    print(f"Wrote: {UNIT}")
    print()
    print("Run:")
    print("  systemctl --user daemon-reload")
    print("  systemctl --user enable --now liqheat-matrix-gate-logger.service")
    print("  systemctl --user status liqheat-matrix-gate-logger.service --no-pager")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
