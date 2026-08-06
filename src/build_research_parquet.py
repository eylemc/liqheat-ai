from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd


RAW_ROOT = Path("data/raw-mirror")
PARQUET_ROOT = Path("data/research-parquet")
REPORTS = Path("reports")

DEFAULT_WORKERS = max(1, min(6, (os.cpu_count() or 4) - 1))
BATCH_ROWS = 5_000

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("logged_at", pa.timestamp("us", tz="UTC")),
        pa.field("symbol", pa.string()),
        pa.field("timeframe", pa.string()),
        pa.field("current_price", pa.float64()),
        pa.field("liquidation_count", pa.int64()),
        pa.field("price_min", pa.float64()),
        pa.field("price_max", pa.float64()),
        pa.field("rows", pa.int64()),
        pa.field("cols", pa.int64()),
        pa.field("payload_json", pa.large_string()),
        pa.field("raw_value", pa.large_string()),
        pa.field("source_file", pa.string()),
    ]
)


@dataclass(frozen=True)
class Job:
    source: Path
    destination: Path
    metadata: Path


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_timestamp(value: Any):
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        text = value.strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    return None


def json_text(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    try:
        return orjson.dumps(value).decode("utf-8")
    except Exception:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )


def normalize_row(value: Any, source_file: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "id": None,
            "logged_at": None,
            "symbol": None,
            "timeframe": None,
            "current_price": None,
            "liquidation_count": None,
            "price_min": None,
            "price_max": None,
            "rows": None,
            "cols": None,
            "payload_json": None,
            "raw_value": json_text(value),
            "source_file": source_file,
        }

    return {
        "id": str(value["id"]) if value.get("id") is not None else None,
        "logged_at": parse_timestamp(value.get("logged_at")),
        "symbol": (
            str(value["symbol"])
            if value.get("symbol") is not None
            else None
        ),
        "timeframe": (
            str(value["timeframe"])
            if value.get("timeframe") is not None
            else None
        ),
        "current_price": safe_float(value.get("current_price")),
        "liquidation_count": safe_int(value.get("liquidation_count")),
        "price_min": safe_float(value.get("price_min")),
        "price_max": safe_float(value.get("price_max")),
        "rows": safe_int(value.get("rows")),
        "cols": safe_int(value.get("cols")),
        "payload_json": json_text(value.get("payload")),
        "raw_value": json_text(value.get("_raw_value")),
        "source_file": source_file,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def flush_batch(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0

    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    writer.write_table(table, row_group_size=BATCH_ROWS)
    count = len(rows)
    rows.clear()
    return count


def convert_one(job: Job, force: bool = False) -> dict[str, Any]:
    started = time.time()

    job.destination.parent.mkdir(parents=True, exist_ok=True)
    job.metadata.parent.mkdir(parents=True, exist_ok=True)

    if (
        not force
        and job.destination.exists()
        and job.metadata.exists()
    ):
        try:
            existing = json.loads(
                job.metadata.read_text(encoding="utf-8")
            )

            if (
                existing.get("status") == "complete"
                and existing.get("rows") is not None
            ):
                return {
                    "status": "skipped",
                    "source": str(job.source),
                    "destination": str(job.destination),
                    "rows": int(existing["rows"]),
                    "bytes": job.destination.stat().st_size,
                }
        except Exception:
            pass

    temporary = job.destination.with_suffix(".parquet.part")
    temporary.unlink(missing_ok=True)

    row_count = 0
    malformed_lines = 0
    raw_value_rows = 0
    first_timestamp = None
    last_timestamp = None
    batch: list[dict[str, Any]] = []

    writer = pq.ParquetWriter(
        temporary,
        PARQUET_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )

    try:
        decompressor = zstd.ZstdDecompressor()

        with job.source.open("rb") as compressed:
            with decompressor.stream_reader(compressed) as reader:
                # Zstd stream'i buffered reader üzerinden satır satır oku.
                import io

                buffered = io.BufferedReader(reader, buffer_size=4 * 1024 * 1024)

                for raw_line in buffered:
                    raw_line = raw_line.strip()

                    if not raw_line:
                        continue

                    try:
                        value = orjson.loads(raw_line)
                    except orjson.JSONDecodeError:
                        malformed_lines += 1
                        value = raw_line.decode("utf-8", errors="replace")

                    normalized = normalize_row(
                        value,
                        str(job.source.relative_to(RAW_ROOT)),
                    )

                    if normalized["raw_value"] is not None:
                        raw_value_rows += 1

                    timestamp = normalized["logged_at"]

                    if timestamp is not None:
                        if first_timestamp is None:
                            first_timestamp = timestamp
                        last_timestamp = timestamp

                    batch.append(normalized)

                    if len(batch) >= BATCH_ROWS:
                        row_count += flush_batch(writer, batch)

        row_count += flush_batch(writer, batch)
        writer.close()
        writer = None

        temporary.replace(job.destination)

        result = {
            "status": "complete",
            "source": str(job.source),
            "destination": str(job.destination),
            "rows": row_count,
            "malformed_lines": malformed_lines,
            "raw_value_rows": raw_value_rows,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "parquet_bytes": job.destination.stat().st_size,
            "parquet_mb": round(
                job.destination.stat().st_size / 1024 / 1024,
                3,
            ),
            "sha256": sha256_file(job.destination),
            "elapsed_seconds": round(time.time() - started, 3),
        }

        temp_meta = job.metadata.with_suffix(".json.tmp")
        temp_meta.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        temp_meta.replace(job.metadata)

        return result

    except Exception as exc:
        if writer is not None:
            writer.close()

        temporary.unlink(missing_ok=True)

        return {
            "status": "error",
            "source": str(job.source),
            "destination": str(job.destination),
            "error": repr(exc),
            "elapsed_seconds": round(time.time() - started, 3),
        }


def discover_jobs() -> list[Job]:
    jobs: list[Job] = []

    for source in sorted(RAW_ROOT.rglob("*.jsonl.zst")):
        relative = source.relative_to(RAW_ROOT)

        destination = (
            PARQUET_ROOT
            / relative.parent
            / source.name.replace(".jsonl.zst", ".parquet")
        )

        metadata = destination.with_suffix(".parquet.meta.json")

        jobs.append(
            Job(
                source=source,
                destination=destination,
                metadata=metadata,
            )
        )

    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    jobs = discover_jobs()

    if not jobs:
        raise RuntimeError(
            f"Kaynak dosya bulunamadı: {RAW_ROOT}"
        )

    print("==========================================")
    print("LiqHeat Research Parquet Builder")
    print("==========================================")
    print("Source files:", f"{len(jobs):,}")
    print("Workers:", args.workers)
    print("Destination:", PARQUET_ROOT.resolve())
    print("Payload mode: JSON string, lossless")
    print()

    counters = {
        "complete": 0,
        "skipped": 0,
        "error": 0,
        "rows": 0,
        "bytes": 0,
        "malformed": 0,
        "raw_value_rows": 0,
    }

    results: list[dict[str, Any]] = []
    started = time.time()

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        future_map = {
            executor.submit(convert_one, job, args.force): job
            for job in jobs
        }

        for completed, future in enumerate(
            as_completed(future_map),
            start=1,
        ):
            job = future_map[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "error",
                    "source": str(job.source),
                    "error": repr(exc),
                }

            results.append(result)

            status = result.get("status", "error")
            counters[status] = counters.get(status, 0) + 1
            counters["rows"] += int(result.get("rows", 0) or 0)
            counters["bytes"] += int(
                result.get(
                    "parquet_bytes",
                    result.get("bytes", 0),
                )
                or 0
            )
            counters["malformed"] += int(
                result.get("malformed_lines", 0) or 0
            )
            counters["raw_value_rows"] += int(
                result.get("raw_value_rows", 0) or 0
            )

            elapsed = time.time() - started
            rate = completed / elapsed if elapsed else 0
            remaining = len(jobs) - completed
            eta = remaining / rate if rate else 0

            print(
                "\r"
                f"Files {completed:,}/{len(jobs):,} | "
                f"done={counters['complete']:,} "
                f"skip={counters['skipped']:,} "
                f"error={counters['error']:,} | "
                f"rows={counters['rows']:,} | "
                f"parquet={counters['bytes'] / 1024**3:.2f} GB | "
                f"ETA={eta / 60:.1f} min",
                end="",
                flush=True,
            )

    print()

    summary = {
        "status": (
            "complete"
            if counters["error"] == 0
            else "complete_with_errors"
        ),
        "source_files": len(jobs),
        "completed_files": counters["complete"],
        "skipped_files": counters["skipped"],
        "error_files": counters["error"],
        "rows": counters["rows"],
        "parquet_bytes": counters["bytes"],
        "parquet_gb": round(counters["bytes"] / 1024**3, 3),
        "malformed_lines": counters["malformed"],
        "raw_value_rows": counters["raw_value_rows"],
        "elapsed_seconds": round(time.time() - started, 3),
        "elapsed_minutes": round((time.time() - started) / 60, 3),
    }

    Path("reports/research_parquet_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    errors = [
        item
        for item in results
        if item.get("status") == "error"
    ]

    Path("reports/research_parquet_errors.json").write_text(
        json.dumps(
            errors,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n==========================================")
    print("PARQUET BUILD COMPLETE")
    print("==========================================")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
