#!/usr/bin/env python3
"""Audit the Liqheat CatBoost research pipeline with the OpenAI Responses API.

The script collects selected source/report files, excludes large datasets and
model binaries, runs four independent reviews, and writes a consolidated
experiment roadmap. It never modifies training code automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
ALLOWED_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".csv", ".tsv",
}
BLOCKED_SUFFIXES = {
    ".parquet", ".feather", ".arrow", ".pkl", ".pickle", ".joblib",
    ".bin", ".pt", ".pth", ".onnx", ".npy", ".npz", ".db",
    ".sqlite", ".zip", ".gz", ".tar",
}
BLOCKED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}
SECRET_MARKERS = (
    "OPENAI_API_KEY=", "SUPABASE_SERVICE_ROLE_KEY=", "SUPABASE_SECRET_KEY=",
    "DATABASE_URL=", "PRIVATE_KEY=", "API_SECRET=",
)
MAX_FILE_BYTES = 250_000
MAX_FILE_CHARS = 45_000
MAX_TOTAL_CHARS = 500_000


@dataclass
class SourceFile:
    path: str
    sha256: str
    original_bytes: int
    included_chars: int
    truncated: bool
    content: str


AUDIT_ROLES = {
    "problem_and_label_audit": """
Act as a senior quant researcher. Audit the observation unit, labels and
forecast horizon before suggesting hyperparameter changes. Examine UPPER_FIRST,
LOWER_FIRST, NONE and INVALID logic, overlapping windows, repeated snapshots,
event independence, volatility-aware horizons, two-stage edge/direction models,
abstention and label noise. Mark every claim OBSERVED, INFERRED or HYPOTHESIS.
""",
    "leakage_and_validation_audit": """
Act as an adversarial ML auditor. Find temporal leakage, missing purge/embargo,
train/test overlap of the same liquidity event, future-derived features,
near-duplicate observations, symbol/timeframe dependence, misleading accuracy,
class imbalance, calibration problems and unstable confidence buckets. Every
finding needs evidence, a falsifiable test and a measurable success criterion.
""",
    "feature_and_model_audit": """
Act as a CatBoost and market-microstructure specialist. Audit preprocessing,
features, loss and model structure. Consider distance/volatility/age normalized
liquidity, cluster velocity, persistence, replenishment, post-sweep state,
regime and multi-timeframe alignment, redundant features, categorical handling,
calibration, global versus per-symbol models, and only then hyperparameter search.
""",
    "adversarial_reviewer": """
Challenge the entire premise. Consider whether weak performance reflects a weak
or non-stationary target rather than CatBoost. Question the real baseline,
effective sample size, regime-specific predictability, selective prediction,
when no-trade is correct, and robustness across market regimes.
""",
}

SYSTEM_INSTRUCTIONS = """
You are auditing the Liqheat liquidation-topology ML research pipeline. Never
invent implementation details or metrics absent from the supplied files. Say
unknown when evidence is missing. Classify claims as OBSERVED, INFERRED or
HYPOTHESIS. Give falsifiable tests and measurable success metrics. Do not promise
financial performance. Return valid JSON only, without markdown fences.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI audit for Liqheat ML")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/reports/openai_ml_audit"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--max-total-chars", type=int, default=MAX_TOTAL_CHARS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def redact_secrets(text: str) -> str:
    redacted = []
    for line in text.splitlines():
        upper = line.upper()
        if any(marker in upper for marker in SECRET_MARKERS):
            key = line.split("=", 1)[0].strip() if "=" in line else "SECRET"
            redacted.append(f"{key}=<REDACTED>")
        else:
            redacted.append(line)
    return "\n".join(redacted)


def iter_paths(root: Path, globs: list[str]) -> Iterable[Path]:
    if globs:
        seen: set[Path] = set()
        for pattern in globs:
            for path in root.glob(pattern):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    yield path
        return
    for path in root.rglob("*"):
        if path.is_file() and not any(part in BLOCKED_DIRS for part in path.parts):
            yield path


def load_sources(
    root: Path, globs: list[str], max_total_chars: int,
) -> tuple[list[SourceFile], list[dict[str, str]]]:
    sources: list[SourceFile] = []
    skipped: list[dict[str, str]] = []
    total = 0
    paths = sorted(iter_paths(root, globs), key=lambda p: str(p.relative_to(root)))

    for path in paths:
        rel = str(path.relative_to(root))
        suffix = path.suffix.lower()
        try:
            size = path.stat().st_size
        except OSError as exc:
            skipped.append({"path": rel, "reason": f"stat_error:{exc}"})
            continue
        if suffix in BLOCKED_SUFFIXES or suffix not in ALLOWED_SUFFIXES or size > MAX_FILE_BYTES:
            skipped.append({"path": rel, "reason": "suffix_or_size"})
            continue
        if total >= max_total_chars:
            skipped.append({"path": rel, "reason": "context_limit"})
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            skipped.append({"path": rel, "reason": f"read_error:{exc}"})
            continue
        text = redact_secrets(raw.decode("utf-8", errors="replace"))
        count = min(len(text), MAX_FILE_CHARS, max_total_chars - total)
        included = text[:count]
        sources.append(SourceFile(
            path=rel,
            sha256=hashlib.sha256(raw).hexdigest(),
            original_bytes=len(raw),
            included_chars=len(included),
            truncated=count < len(text),
            content=included,
        ))
        total += len(included)
    return sources, skipped


def source_bundle(sources: list[SourceFile]) -> str:
    chunks = []
    for src in sources:
        chunks.append(
            f"===== FILE: {src.path} =====\n"
            f"sha256: {src.sha256}\ntruncated: {src.truncated}\n"
            f"{src.content}\n===== END FILE ====="
        )
    return "\n\n".join(chunks)


def parse_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`").lstrip()
        if value.startswith("json"):
            value = value[4:].lstrip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed


def call_json(
    client: OpenAI, model: str, prompt: str, retries: int = 3,
) -> tuple[dict[str, Any], str, str]:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=prompt,
            )
            raw = response.output_text
            return parse_json(raw), raw, response.id
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"OpenAI request failed: {error}") from error


def audit_prompt(name: str, role: str, bundle: str) -> str:
    return f"""
AUDIT NAME: {name}
ROLE:\n{role}

PROJECT FILES:\n{bundle}

Return this JSON shape:
{{
  "audit_name": "{name}",
  "executive_summary": "string",
  "findings": [{{
    "id": "string",
    "classification": "OBSERVED|INFERRED|HYPOTHESIS",
    "severity": "critical|high|medium|low",
    "title": "string",
    "evidence": [{{"file": "relative/path", "detail": "string"}}],
    "why_it_matters": "string",
    "verification_test": "string",
    "recommended_change": "string",
    "success_metric": "string",
    "risk": "string"
  }}],
  "missing_evidence": ["string"],
  "top_3_next_actions": ["string"]
}}
Use at most 12 findings. Return JSON only.
"""


def synthesis_prompt(audits: dict[str, dict[str, Any]]) -> str:
    return f"""
Combine these audits into an ordered, falsifiable research roadmap:
{json.dumps(audits, ensure_ascii=False, indent=2)}

Return JSON with:
- executive_decision: current_pipeline_status, most_likely_bottleneck,
  catboost_should_be_kept, reason
- blocking_issues: ranked issue, evidence_class, reason
- experiments: priority, experiment_id, title, hypothesis, implementation,
  comparison, metrics, minimum_success_threshold, failure_interpretation,
  estimated_complexity, dependencies, leakage_checks
- recommended_evaluation_protocol: split_method, embargo_or_purge,
  primary_metrics, secondary_metrics, reporting_slices,
  selective_prediction_policy
- stop_conditions
- first_implementation_batch

Order label, leakage, event sampling and baseline experiments before feature,
model and hyperparameter work. Use at most 12 experiments. Return JSON only.
"""


def render_markdown(meta: dict[str, Any], audits: dict[str, Any], plan: dict[str, Any]) -> str:
    decision = plan.get("executive_decision", {})
    lines = [
        "# Liqheat OpenAI ML Audit", "",
        f"- Generated: {meta['generated_at']}",
        f"- Model: `{meta['model']}`",
        f"- Included files: {meta['included_file_count']}", "",
        "## Executive decision", "",
        f"**Status:** {decision.get('current_pipeline_status', 'unknown')}", "",
        f"**Most likely bottleneck:** {decision.get('most_likely_bottleneck', 'unknown')}", "",
        f"**Keep CatBoost:** {decision.get('catboost_should_be_kept', 'unknown')}", "",
        decision.get("reason", ""), "", "## Experiment roadmap", "",
    ]
    for exp in plan.get("experiments", []):
        lines += [
            f"### {exp.get('priority', '?')}. {exp.get('experiment_id', '')} — {exp.get('title', '')}", "",
            f"**Hypothesis:** {exp.get('hypothesis', '')}", "",
            f"**Comparison:** {exp.get('comparison', '')}", "",
            "**Implementation:**",
        ]
        lines += [f"- {step}" for step in exp.get("implementation", [])]
        lines += ["", f"**Success:** {exp.get('minimum_success_threshold', '')}", ""]
    lines += ["## Individual audits", ""]
    for name, audit in audits.items():
        lines += [f"### {name}", "", audit.get("executive_summary", ""), ""]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    out = args.output_dir.expanduser()
    if not out.is_absolute():
        out = (root / out).resolve()
    if not root.is_dir():
        print(f"ERROR: project root not found: {root}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    sources, skipped = load_sources(root, args.include, args.max_total_chars)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "model": args.model,
        "included_file_count": len(sources),
        "included_char_count": sum(item.included_chars for item in sources),
        "included_files": [
            {k: v for k, v in asdict(item).items() if k != "content"}
            for item in sources
        ],
        "skipped_files": skipped,
    }
    (out / "input_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Included {len(sources)} files, {meta['included_char_count']:,} chars")
    if not sources:
        print("ERROR: no suitable files found", file=sys.stderr)
        return 3
    if args.dry_run:
        for item in meta["included_files"]:
            print(f"- {item['path']} ({item['included_chars']:,} chars)")
        return 0
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set", file=sys.stderr)
        return 4

    client = OpenAI()
    bundle = source_bundle(sources)
    audits: dict[str, dict[str, Any]] = {}
    response_ids: dict[str, str] = {}
    for name, role in AUDIT_ROLES.items():
        print(f"Running {name}")
        parsed, raw, response_id = call_json(client, args.model, audit_prompt(name, role, bundle))
        audits[name] = parsed
        response_ids[name] = response_id
        (out / f"{name}.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / f"{name}.raw.txt").write_text(raw, encoding="utf-8")

    print("Synthesizing roadmap")
    plan, raw, response_id = call_json(client, args.model, synthesis_prompt(audits))
    response_ids["synthesis"] = response_id
    (out / "experiment_roadmap.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "experiment_roadmap.raw.txt").write_text(raw, encoding="utf-8")
    meta["response_ids"] = response_ids
    (out / "input_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "OPENAI_ML_AUDIT.md").write_text(render_markdown(meta, audits, plan), encoding="utf-8")
    print(f"Done: {out / 'OPENAI_ML_AUDIT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
