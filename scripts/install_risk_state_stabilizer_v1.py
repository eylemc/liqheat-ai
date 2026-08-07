#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "src" / "textara_api.py"


def main() -> None:
    text = API.read_text(encoding="utf-8")
    original = text

    import_line = "from src.risk_state_stabilizer import risk_state_stabilizer\n"
    if import_line not in text:
        anchor = "from src.market_risk_live import (\n"
        idx = text.find(anchor)
        if idx < 0:
            raise SystemExit("Could not find market_risk_live import block in src/textara_api.py")
        end = text.find(")\n", idx)
        if end < 0:
            raise SystemExit("Could not locate end of market_risk_live import block")
        end += 2
        text = text[:end] + import_line + text[end:]
        print("Patched: risk stabilizer import")
    else:
        print("Already patched: risk stabilizer import")

    old = "    return market_risk_engine.score_latest(frame, requested)\n"
    new = (
        "    raw_risk = market_risk_engine.score_latest(frame, requested)\n"
        "    return risk_state_stabilizer.update(raw_risk)\n"
    )
    if new in text:
        print("Already patched: stabilized live risk return")
    elif old in text:
        text = text.replace(old, new, 1)
        print("Patched: stabilized live risk return")
    else:
        raise SystemExit("Could not find live risk return target in src/textara_api.py")

    if text != original:
        backup = API.with_suffix(".py.before-risk-state-stabilizer-v1")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        API.write_text(text, encoding="utf-8")

    verify = {
        "import": import_line in text,
        "update_call": "risk_state_stabilizer.update(raw_risk)" in text,
    }
    print("VERIFY:", verify)
    if not all(verify.values()):
        raise SystemExit("Risk State Stabilizer installation verification failed")
    print("Risk State Stabilizer V1 installed.")


if __name__ == "__main__":
    main()
