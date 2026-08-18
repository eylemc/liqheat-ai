#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "src" / "textara_api.py"
JS = ROOT / "static" / "matrix_ai_radar.js"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"Already patched: {label}")
        return text
    if old not in text:
        raise SystemExit(f"Patch target not found: {label}")
    print(f"Patched: {label}")
    return text.replace(old, new, 1)


def patch_api() -> None:
    text = API.read_text(encoding="utf-8")
    backup = API.with_suffix(".py.before-matrix-regime-gate")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    text = replace_once(
        text,
        '''from src.matrix_live import (\n    combine_matrix_topology,\n    get_live_matrix,\n)\n''',
        '''from src.matrix_live import (\n    combine_matrix_topology,\n    get_live_matrix,\n)\nfrom src.matrix_regime_gate import get_matrix_regime_gate\n''',
        "Matrix regime gate import",
    )

    text = replace_once(
        text,
        '''        try:\n            ai_market_risk = build_ai_market_risk(\n                str(feature_row["symbol"])\n            )\n''',
        '''        try:\n            matrix_regime_gate = get_matrix_regime_gate(\n                str(feature_row["symbol"])\n            )\n        except Exception as exc:\n            matrix_regime_gate = {\n                "available": False,\n                "reason": "REGIME_GATE_ERROR",\n                "symbol": str(feature_row["symbol"]),\n                "timeframe": "1h",\n                "error": f"{type(exc).__name__}: {exc}",\n            }\n\n        try:\n            ai_market_risk = build_ai_market_risk(\n                str(feature_row["symbol"])\n            )\n''',
        "Matrix regime gate live inference",
    )

    text = replace_once(
        text,
        '''            "matrix": (\n                matrix_data\n                if matrix_data is not None\n                else {\n                    "available": False,\n                    "error": matrix_error,\n                }\n            ),\n''',
        '''            "matrix": (\n                matrix_data\n                if matrix_data is not None\n                else {\n                    "available": False,\n                    "error": matrix_error,\n                }\n            ),\n            "matrix_regime_gate": matrix_regime_gate,\n''',
        "Expose Matrix regime gate in Radar payload",
    )

    text = replace_once(
        text,
        '''            "direction_model": (\n                "1h topology first-hit direction model; research preview"\n            ),\n''',
        '''            "direction_model": (\n                "1h topology first-hit direction model; research preview"\n            ),\n            "matrix_regime_gate": (\n                "Frozen BTCUSDT 1H Matrix flip quality gate; VALID >= 65.5833"\n            ),\n''',
        "Matrix regime scoring metadata",
    )

    API.write_text(text, encoding="utf-8")


def patch_js() -> None:
    text = JS.read_text(encoding="utf-8")
    backup = JS.with_suffix(".js.before-matrix-regime-gate")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    helpers = '''\nfunction regimeGateData(item) {\n  return item.matrix_regime_gate?.available ? item.matrix_regime_gate : null;\n}\nfunction regimeGateClass(status) {\n  return String(status || "").toUpperCase() === "VALID" ? "regime-valid" : "regime-block";\n}\nfunction regimeGateTime(iso) {\n  if (!iso) return "—";\n  return new Date(iso).toLocaleString([], { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });\n}\nfunction matrixRegimeGateHtml(item) {\n  const gate = regimeGateData(item);\n  if (!gate) return "";\n  const latest = gate.latest_flip || {};\n  const lastValid = gate.last_valid_signal || {};\n  const status = String(gate.status || latest.status || "BLOCK").toUpperCase();\n  const score = gate.regime_score;\n  const threshold = Number(gate.threshold || 65.5833);\n  const side = latest.side || "—";\n  const erRank = latest.er_rank;\n  const channelRank = latest.channel_rank;\n  const dispRank = latest.norm_disp_rank;\n  const lastValidText = lastValid.side\n    ? `${lastValid.side} · ${regimeGateTime(lastValid.close_time)} · ${formatNumber(lastValid.score, 1)}`\n    : "No validated signal in baseline";\n  return `<div class="matrix-regime-gate ${regimeGateClass(status)}">\n    <div class="regime-gate-head"><span>MATRIX REGIME GATE · 1H</span><strong>${status}</strong></div>\n    <div class="regime-gate-score"><b>${score === null || score === undefined ? "—" : formatNumber(score, 1)}</b><span>/ ${formatNumber(threshold, 1)}</span></div>\n    <div class="regime-gate-flip"><span>Latest Matrix flip</span><strong>${side}</strong><small>${regimeGateTime(latest.close_time)}</small></div>\n    <div class="regime-gate-ranks">\n      <span>ER <b>${formatNumber(erRank, 1)}</b></span>\n      <span>Channel <b>${formatNumber(channelRank, 1)}</b></span>\n      <span>Disp <b>${formatNumber(dispRank, 1)}</b></span>\n    </div>\n    <div class="regime-gate-last"><span>Last VALID</span><strong>${lastValidText}</strong></div>\n  </div>`;\n}\n'''

    text = replace_once(
        text,
        '''function riskData(item) {\n  return item.ai_market_risk?.available ? item.ai_market_risk : null;\n}\n''',
        '''function riskData(item) {\n  return item.ai_market_risk?.available ? item.ai_market_risk : null;\n}\n''' + helpers,
        "Matrix regime gate JS helpers",
    )

    text = replace_once(
        text,
        '''    .ai-risk-sub{font-size:10px;color:var(--muted);margin-top:5px;text-transform:uppercase;letter-spacing:.09em}\n''',
        '''    .ai-risk-sub{font-size:10px;color:var(--muted);margin-top:5px;text-transform:uppercase;letter-spacing:.09em}\n    .matrix-regime-gate{margin:12px 0 14px;padding:13px;border-radius:12px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08)}\n    .matrix-regime-gate.regime-valid{border-color:rgba(76,229,166,.25);box-shadow:inset 0 0 24px rgba(76,229,166,.035)}\n    .matrix-regime-gate.regime-block{border-color:rgba(255,111,125,.20);box-shadow:inset 0 0 24px rgba(255,111,125,.025)}\n    .regime-gate-head{display:flex;justify-content:space-between;align-items:center;gap:10px}.regime-gate-head span{font-size:9px;color:var(--muted);font-weight:800;letter-spacing:.11em}.regime-gate-head strong{font-size:10px;letter-spacing:.08em;padding:5px 8px;border-radius:999px}.regime-valid .regime-gate-head strong{color:#4ce5a6;background:rgba(76,229,166,.10)}.regime-block .regime-gate-head strong{color:#ff6f7d;background:rgba(255,111,125,.10)}\n    .regime-gate-score{display:flex;align-items:baseline;gap:5px;margin:8px 0}.regime-gate-score b{font-size:30px;letter-spacing:-.04em}.regime-gate-score span{font-size:11px;color:var(--muted)}\n    .regime-gate-flip{display:grid;grid-template-columns:1fr auto;gap:2px 10px;align-items:center;padding-top:7px;border-top:1px solid rgba(255,255,255,.055)}.regime-gate-flip span{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}.regime-gate-flip strong{font-size:13px}.regime-gate-flip small{grid-column:1 / -1;color:var(--muted);font-size:9px}\n    .regime-gate-ranks{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:9px}.regime-gate-ranks span{padding:6px;border-radius:7px;background:rgba(255,255,255,.03);font-size:8px;color:var(--muted);text-align:center}.regime-gate-ranks b{display:block;margin-top:2px;color:var(--text);font-size:11px}\n    .regime-gate-last{display:flex;justify-content:space-between;gap:10px;margin-top:8px;font-size:8px;color:var(--muted)}.regime-gate-last strong{color:var(--text);font-size:8px;text-align:right;font-weight:700}\n''',
        "Matrix regime gate styles",
    )

    text = replace_once(
        text,
        '''      <div class="matrix-strip">${matrixTimeframeStrip(item)}</div>\n''',
        '''      ${matrixRegimeGateHtml(item)}\n      <div class="matrix-strip">${matrixTimeframeStrip(item)}</div>\n''',
        "Matrix regime gate card panel",
    )

    JS.write_text(text, encoding="utf-8")


def main() -> None:
    patch_api()
    patch_js()
    print("Matrix Regime Gate live integration installed.")
    print("BTCUSDT 1H frozen threshold: 65.5833")
    print("Restart the Radar API service after running this installer.")


if __name__ == "__main__":
    main()
