#!/usr/bin/env python3
from pathlib import Path

path = Path(__file__).resolve().parent / "analyze_position_guardian_v6_walkforward_calibration.py"
text = path.read_text(encoding="utf-8")

old = '''                            base = {
                                "side": side,
                                "horizon_minutes": horizon,
                                "feature_group": group_name,
                                "head": head,
                                "fold": fold,
                                "symbol": symbol,
                                "calibration_method": calibrator.method,
                            }
                            threshold_rows.append({
                                **base,
'''
new = '''                            result_base = {
                                "side": side,
                                "horizon_minutes": horizon,
                                "feature_group": group_name,
                                "head": head,
                                "fold": fold,
                                "symbol": symbol,
                                "calibration_method": calibrator.method,
                            }
                            threshold_rows.append({
                                **result_base,
'''

if old in text:
    text = text.replace(old, new, 1)
elif "result_base = {" not in text:
    raise SystemExit("Expected shadowing block not found")

old2 = '''                            fold_metric_rows.append({
                                **base,
'''
new2 = '''                            fold_metric_rows.append({
                                **result_base,
'''
if old2 in text:
    text = text.replace(old2, new2, 1)
elif "**result_base," not in text:
    raise SystemExit("Expected metrics shadowing block not found")

path.write_text(text, encoding="utf-8")
print(f"Patched: {path}")
