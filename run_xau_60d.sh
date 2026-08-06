#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

export LIQHEAT_SYMBOL="XAUUSDT"
export LIQHEAT_TIMEFRAME="24h"
export LIQHEAT_DAYS="60"
export LIQHEAT_FUTURE_HOURS="4"
export LIQHEAT_MAX_POOL_DISTANCE_PCT="0.03"
export LIQHEAT_V5_WORKERS="6"

mkdir -p logs reports data/processed

echo
echo "=========================================="
echo "XAU V5 — 60 Günlük Nearest-Pool Testi"
echo "=========================================="

python src/validate_nearest_pool_v5.py \
  2>&1 | tee logs/xau_validate_nearest_pool_v5.log

echo
echo "=========================================="
echo "XAU V6 — Distance × Volume Matrix"
echo "=========================================="

python src/analyze_edge_matrix_v6.py \
  2>&1 | tee logs/xau_analyze_edge_matrix_v6.log

echo
echo "=========================================="
echo "XAU 60 Günlük Araştırma Tamamlandı"
echo "=========================================="

python - <<'PY'
import json
from pathlib import Path

v5_path = Path("reports/xauusdt_60d_nearest_pool_v5.json")
v6_path = Path("reports/xauusdt_60d_edge_matrix_v6.json")

if not v5_path.exists():
    raise FileNotFoundError(f"V5 raporu bulunamadı: {v5_path}")

if not v6_path.exists():
    raise FileNotFoundError(f"V6 raporu bulunamadı: {v6_path}")

v5 = json.loads(v5_path.read_text())
v6 = json.loads(v6_path.read_text())

overall = v5["overall"]
independent = v5["non_overlapping_4h"]

print()
print("==========================================")
print("XAU FINAL SCOREBOARD")
print("==========================================")

print(
    "Overall:",
    f"rows={overall['rows']}",
    f"accuracy={overall['accuracy']:.4f}",
    f"balanced={overall['balanced_accuracy']:.4f}",
    f"MCC={overall['mcc']:.4f}",
)

print(
    "Independent 4h:",
    f"rows={independent['rows']}",
    f"accuracy={independent['accuracy']:.4f}",
    f"balanced={independent['balanced_accuracy']:.4f}",
    f"MCC={independent['mcc']:.4f}",
)

print()
print("Independent high-confidence rules:")

for row in v6["independent_4h_confidence_rules"]:
    accuracy = row.get("accuracy")

    if accuracy is None:
        continue

    print(
        f"{row['name']:<30}",
        f"rows={row['rows']:>4}",
        f"coverage={row['coverage']:.3f}",
        f"accuracy={accuracy:.4f}",
        f"CI95=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]",
    )

print()
print("Sessions:")

for session, metrics in v6["sessions"].items():
    print(
        f"{session:<18}",
        f"rows={metrics['rows']:>4}",
        f"accuracy={metrics['accuracy']:.4f}",
    )

print()
print("Signal directions:")

for direction, metrics in v6["directions"].items():
    print(
        f"{direction:<8}",
        f"rows={metrics['rows']:>4}",
        f"accuracy={metrics['accuracy']:.4f}",
    )
PY
