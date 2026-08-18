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
        "from src.topology_direction_live import topology_direction_engine\n",
        "from src.topology_direction_live import topology_direction_engine\nfrom src.direction_bias_temporal import record_direction_bias, build_temporal_direction_bias\n",
        "temporal Direction Bias import",
    )

    anchor = '''        try:\n            direction_model = topology_direction_engine.score_latest(\n                feature_frame.iloc[[row_index]],\n                str(feature_row["symbol"]),\n            )\n        except Exception as exc:\n            direction_model = {\n                "available": False,\n                "reason": "LIVE_INFERENCE_ERROR",\n                "symbol": str(feature_row["symbol"]),\n                "horizon_minutes": 60,\n                "error": f"{type(exc).__name__}: {exc}",\n            }\n\n'''

    replacement = anchor + '''        try:\n            record_direction_bias(\n                str(feature_row["symbol"]),\n                str(feature_row["id"]),\n                direction_model,\n            )\n            direction_bias_temporal = build_temporal_direction_bias(\n                str(feature_row["symbol"]),\n                direction_model,\n            )\n        except Exception as exc:\n            direction_bias_temporal = {\n                "available": False,\n                "reason": "TEMPORAL_DIRECTION_ERROR",\n                "error": f"{type(exc).__name__}: {exc}",\n            }\n\n'''
    text = replace_once(text, anchor, replacement, "temporal Direction Bias inference")
    text = replace_once(
        text,
        '            "direction_model": direction_model,\n',
        '            "direction_model": direction_model,\n            "direction_bias_temporal": direction_bias_temporal,\n',
        "expose temporal Direction Bias",
    )
    API.write_text(text, encoding="utf-8")
    print("Temporal Direction Bias V1 integration installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
