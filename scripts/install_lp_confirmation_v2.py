from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "src/textara_api.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"Already patched: {label}")
        return text
    if old not in text:
        raise RuntimeError(f"Patch anchor not found: {label}")
    print(f"Patched: {label}")
    return text.replace(old, new, 1)


def main() -> int:
    text = API.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "from src.liquidation_heatmap_live import build_compact_heatmap\n",
        "from src.liquidation_heatmap_live import build_compact_heatmap\nfrom src.lp_confirmation_live import build_lp_confirmation\n",
        "LP confirmation import",
    )

    anchor = '''        try:\n            direction_model = topology_direction_engine.score_latest(\n                feature_frame.iloc[[row_index]],\n                str(feature_row["symbol"]),\n            )\n        except Exception as exc:\n            direction_model = {\n                "available": False,\n                "reason": "LIVE_INFERENCE_ERROR",\n                "symbol": str(feature_row["symbol"]),\n                "horizon_minutes": 60,\n                "error": f"{type(exc).__name__}: {exc}",\n            }\n\n'''

    replacement = anchor + '''        try:\n            lp_confirmation_v2 = build_lp_confirmation(\n                str(feature_row["symbol"]),\n                direction_model,\n            )\n        except Exception as exc:\n            lp_confirmation_v2 = {\n                "available": False,\n                "state": "NEUTRAL",\n                "reason": "LP_CONFIRMATION_ERROR",\n                "error": f"{type(exc).__name__}: {exc}",\n            }\n\n'''

    text = replace_once(
        text,
        anchor,
        replacement,
        "temporal LP confirmation inference",
    )

    text = replace_once(
        text,
        '            "direction_model": direction_model,\n',
        '            "direction_model": direction_model,\n            "lp_confirmation_v2": lp_confirmation_v2,\n',
        "expose temporal LP confirmation",
    )

    API.write_text(text, encoding="utf-8")
    print("Temporal LP Confirmation V2 integration installed.")
    print("Restart liqheat-radar-api.service after py_compile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
