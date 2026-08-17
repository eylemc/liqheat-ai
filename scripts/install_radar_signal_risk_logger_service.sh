#!/usr/bin/env bash
set -euo pipefail

ROOT="${LIQHEAT_ROOT:-/home/eylem/liqheat-ai}"
SERVICE="liqheat-radar-signal-risk-logger.service"
UNIT="/etc/systemd/system/${SERVICE}"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Missing venv python: ${ROOT}/.venv/bin/python" >&2
  exit 1
fi

mkdir -p "${ROOT}/logs" "${ROOT}/data/research/radar_signal_risk"

sudo tee "${UNIT}" >/dev/null <<EOF
[Unit]
Description=LiqHeat Radar Matrix Signal + AI Risk Research Logger
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=eylem
Group=eylem
WorkingDirectory=${ROOT}
ExecStart=${ROOT}/.venv/bin/python ${ROOT}/scripts/radar_signal_risk_logger.py --url http://127.0.0.1:8000/radar --db ${ROOT}/data/research/radar_signal_risk/radar_signal_risk.sqlite3 --interval 5
Restart=always
RestartSec=5
StandardOutput=append:${ROOT}/logs/radar_signal_risk_logger.log
StandardError=append:${ROOT}/logs/radar_signal_risk_logger.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE}"

echo
echo "================ SIGNAL/RISK LOGGER STATUS ================"
sudo systemctl status "${SERVICE}" --no-pager

echo
echo "================ SIGNAL/RISK LOGGER LOG ==================="
sudo journalctl -u "${SERVICE}" -n 30 --no-pager || true

echo
echo "DB: ${ROOT}/data/research/radar_signal_risk/radar_signal_risk.sqlite3"
echo "LOG: ${ROOT}/logs/radar_signal_risk_logger.log"
