#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

SERVICE = Path.home() / ".config/systemd/user/liqheat-liquidation-pressure-logger.service"

UNIT = """[Unit]
Description=LiqHeat Liquidation Pressure History Logger
After=network-online.target liqheat-radar-api.service

[Service]
Type=simple
WorkingDirectory=/home/eylem/liqheat-ai
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/eylem/liqheat-ai/.venv/bin/python /home/eylem/liqheat-ai/scripts/liquidation_pressure_history_logger.py --interval 60
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""


def main() -> int:
    SERVICE.parent.mkdir(parents=True, exist_ok=True)
    SERVICE.write_text(UNIT, encoding="utf-8")
    print(f"Installed: {SERVICE}")
    print("Run:")
    print("  systemctl --user daemon-reload")
    print("  systemctl --user enable --now liqheat-liquidation-pressure-logger.service")
    print("DB:")
    print("  data/research/liquidation_pressure/liquidation_pressure.sqlite3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
