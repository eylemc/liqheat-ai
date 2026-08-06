#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/liqheat-ai"
source .venv/bin/activate

mkdir -p src data/raw-mirror logs reports

python -m pip install --quiet \
  python-dotenv \
  supabase \
  zstandard

cat > src/full_rest_mirror.py <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import zstandard as zstd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL veya SUPABASE_SECRET_KEY .env içinde bulunamadı"
    )

TABLE = "liq_logging"

ROOT = Path(
    os.getenv(
        "LIQHEAT_MIRROR_ROOT",
        "data/raw-mirror",
    )
)

LOG_DIR = Path("logs")
REPORT_DIR = Path("reports")

ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date.fromisoformat(
    os.getenv(
        "LIQHEAT_MIRROR_START",
        "2026-03-30",
    )
)

# Bugün de dahil edilir. Son gün çalışırken eksik olabilir;
# script yarın tekrar çalıştırılırsa o gün yeniden doğrulanabilir.
END_DATE = date.fromisoformat(
    os.getenv(
        "LIQHEAT_MIRROR_END",
        date.today().isoformat(),
    )
)

PAGE_SIZE = int(
    os.getenv("LIQHEAT_MIRROR_PAGE_SIZE", "200")
)

WORKERS = int(
    os.getenv("LIQHEAT_MIRROR_WORKERS", "4")
)

MAX_RETRIES = int(
    os.getenv("LIQHEAT_MIRROR_MAX_RETRIES", "8")
)

ZSTD_LEVEL = int(
    os.getenv("LIQHEAT_MIRROR_ZSTD_LEVEL", "8")
)

# Mevcut veri dağılımımız:
# BTC / ETH / SOL: dört timeframe
# XAU / XAG: 24h
DEFAULT_STREAMS: list[tuple[str, str]] = [
    ("BTCUSDT", "1h"),
    ("BTCUSDT", "4h"),
    ("BTCUSDT", "24h"),
    ("BTCUSDT", "1w"),

    ("ETHUSDT", "1h"),
    ("ETHUSDT", "4h"),
    ("ETHUSDT", "24h"),
    ("ETHUSDT", "1w"),

    ("SOLUSDT", "1h"),
    ("SOLUSDT", "4h"),
    ("SOLUSDT", "24h"),
    ("SOLUSDT", "1w"),

    ("XAUUSDT", "24h"),
    ("XAGUSDT", "24h"),
]

# İstenirse:
# LIQHEAT_MIRROR_STREAMS="BTCUSDT:1h,XAUUSDT:24h"
streams_env = os.getenv("LIQHEAT_MIRROR_STREAMS", "").strip()

if streams_env:
    STREAMS: list[tuple[str, str]] = []

    for item in streams_env.split(","):
        symbol, timeframe = item.strip().split(":", 1)
        STREAMS.append((symbol, timeframe))
else:
    STREAMS = DEFAULT_STREAMS

SELECT_COLUMNS = ",".join(
    [
        "id",
        "logged_at",
        "symbol",
        "timeframe",
        "current_price",
        "liquidation_count",
        "price_min",
        "price_max",
        "rows",
        "cols",
        "payload",
    ]
)

thread_local = threading.local()
print_lock = threading.Lock()
manifest_lock = threading.Lock()

GLOBAL_MANIFEST = ROOT / "manifest.jsonl"
ERROR_LOG = LOG_DIR / "full_mirror_errors.jsonl"


def get_client():
    client = getattr(thread_local, "client", None)

    if client is None:
        client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
        )
        thread_local.client = client

    return client


def atomic_json_write(
    path: Path,
    value: dict[str, Any],
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def append_jsonl(
    path: Path,
    value: dict[str, Any],
    lock: threading.Lock,
) -> None:
    line = json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )

    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def day_paths(
    symbol: str,
    timeframe: str,
    day: date,
) -> tuple[Path, Path, Path]:
    directory = (
        ROOT
        / symbol
        / timeframe
        / f"{day.year:04d}"
        / f"{day.month:02d}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = day.isoformat()

    data_path = directory / f"{stem}.jsonl.zst"
    metadata_path = directory / f"{stem}.meta.json"
    temporary_path = directory / f"{stem}.jsonl.zst.part"

    return data_path, metadata_path, temporary_path


def fetch_page(
    symbol: str,
    timeframe: str,
    start_iso: str,
    end_iso: str,
    offset: int,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_client()

            response = (
                client.table(TABLE)
                .select(SELECT_COLUMNS)
                .eq("symbol", symbol)
                .eq("timeframe", timeframe)
                .gte("logged_at", start_iso)
                .lt("logged_at", end_iso)
                .order("logged_at", desc=False)
                .order("id", desc=False)
                .range(
                    offset,
                    offset + PAGE_SIZE - 1,
                )
                .execute()
            )

            return response.data or []

        except Exception as exc:
            last_error = exc

            wait_seconds = min(
                90.0,
                (2 ** attempt)
                + random.uniform(0.0, 2.0),
            )

            with print_lock:
                print(
                    "\nRETRY "
                    f"{symbol}/{timeframe}/{start_iso[:10]} "
                    f"offset={offset:,} "
                    f"attempt={attempt}/{MAX_RETRIES} "
                    f"wait={wait_seconds:.1f}s "
                    f"error={exc}",
                    flush=True,
                )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Sayfa alınamadı: {last_error}"
    )


def export_day(
    symbol: str,
    timeframe: str,
    day: date,
) -> dict[str, Any]:
    started_at = time.time()

    data_path, metadata_path, temporary_path = (
        day_paths(symbol, timeframe, day)
    )

    # Tamamlanmış ve meta dosyası bulunan parçayı atla.
    if data_path.exists() and metadata_path.exists():
        try:
            existing = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                existing.get("status") == "complete"
                and existing.get("sha256")
                and existing.get("rows") is not None
            ):
                return {
                    "status": "skipped",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "day": day.isoformat(),
                    "rows": int(existing["rows"]),
                    "bytes": int(
                        existing.get(
                            "compressed_bytes",
                            data_path.stat().st_size,
                        )
                    ),
                }
        except Exception:
            pass

    temporary_path.unlink(missing_ok=True)

    next_day = day + timedelta(days=1)

    start_iso = f"{day.isoformat()}T00:00:00+00:00"
    end_iso = f"{next_day.isoformat()}T00:00:00+00:00"

    row_count = 0
    offset = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    compressor = zstd.ZstdCompressor(
        level=ZSTD_LEVEL,
        threads=1,
        write_checksum=True,
    )

    try:
        with temporary_path.open("wb") as raw_handle:
            with compressor.stream_writer(
                raw_handle,
                closefd=False,
            ) as compressed_handle:

                while True:
                    batch = fetch_page(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_iso=start_iso,
                        end_iso=end_iso,
                        offset=offset,
                    )

                    if not batch:
                        break

                    for row in batch:
                        timestamp = row.get("logged_at")

                        if first_timestamp is None:
                            first_timestamp = timestamp

                        last_timestamp = timestamp

                        encoded = (
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                default=str,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")

                        compressed_handle.write(encoded)
                        row_count += 1

                    offset += len(batch)

                    if len(batch) < PAGE_SIZE:
                        break

        # Boş günler de geçerli parça olarak kaydedilir.
        temporary_path.replace(data_path)

        checksum = file_sha256(data_path)

        metadata = {
            "status": "complete",
            "table": TABLE,
            "symbol": symbol,
            "timeframe": timeframe,
            "day": day.isoformat(),
            "range_start": start_iso,
            "range_end": end_iso,
            "rows": row_count,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "compressed_bytes": data_path.stat().st_size,
            "compressed_mb": round(
                data_path.stat().st_size
                / 1024
                / 1024,
                3,
            ),
            "sha256": checksum,
            "file": str(data_path),
            "elapsed_seconds": round(
                time.time() - started_at,
                3,
            ),
        }

        atomic_json_write(
            metadata_path,
            metadata,
        )

        append_jsonl(
            GLOBAL_MANIFEST,
            metadata,
            manifest_lock,
        )

        return metadata

    except Exception as exc:
        temporary_path.unlink(missing_ok=True)

        error = {
            "status": "error",
            "symbol": symbol,
            "timeframe": timeframe,
            "day": day.isoformat(),
            "error": repr(exc),
            "elapsed_seconds": round(
                time.time() - started_at,
                3,
            ),
        }

        append_jsonl(
            ERROR_LOG,
            error,
            manifest_lock,
        )

        return error


def all_days(
    start: date,
    end: date,
) -> list[date]:
    values: list[date] = []
    current = start

    while current <= end:
        values.append(current)
        current += timedelta(days=1)

    return values


days = all_days(
    START_DATE,
    END_DATE,
)

jobs: list[tuple[str, str, date]] = [
    (symbol, timeframe, day)
    for symbol, timeframe in STREAMS
    for day in days
]

print("==========================================")
print("LiqHeat Full REST Mirror")
print("==========================================")
print("Table:", TABLE)
print("Date range:", START_DATE, "→", END_DATE)
print("Streams:", len(STREAMS))
print("Daily jobs:", len(jobs))
print("Workers:", WORKERS)
print("Page size:", PAGE_SIZE)
print("Compression:", f"zstd level {ZSTD_LEVEL}")
print("Destination:", ROOT.resolve())
print()
print("Streams:")
for symbol, timeframe in STREAMS:
    print(f"  - {symbol}/{timeframe}")
print()

counters = {
    "complete": 0,
    "skipped": 0,
    "error": 0,
    "rows": 0,
    "bytes": 0,
}

started = time.time()

with ThreadPoolExecutor(
    max_workers=WORKERS
) as executor:
    future_map = {
        executor.submit(
            export_day,
            symbol,
            timeframe,
            day,
        ): (symbol, timeframe, day)
        for symbol, timeframe, day in jobs
    }

    completed_jobs = 0

    for future in as_completed(future_map):
        completed_jobs += 1

        symbol, timeframe, day = future_map[future]

        try:
            result = future.result()
        except Exception as exc:
            result = {
                "status": "error",
                "symbol": symbol,
                "timeframe": timeframe,
                "day": day.isoformat(),
                "error": repr(exc),
            }

        status = result.get("status", "error")

        if status not in counters:
            status = "error"

        counters[status] += 1
        counters["rows"] += int(
            result.get("rows", 0) or 0
        )
        counters["bytes"] += int(
            result.get(
                "compressed_bytes",
                result.get("bytes", 0),
            )
            or 0
        )

        elapsed = time.time() - started
        jobs_per_second = (
            completed_jobs / elapsed
            if elapsed > 0
            else 0.0
        )
        remaining = len(jobs) - completed_jobs
        eta_seconds = (
            remaining / jobs_per_second
            if jobs_per_second > 0
            else 0.0
        )

        with print_lock:
            print(
                "\r"
                f"Jobs {completed_jobs:,}/{len(jobs):,} | "
                f"done={counters['complete']:,} "
                f"skip={counters['skipped']:,} "
                f"error={counters['error']:,} | "
                f"rows={counters['rows']:,} | "
                f"compressed="
                f"{counters['bytes'] / 1024 / 1024 / 1024:.2f} GB | "
                f"ETA={eta_seconds / 60:.1f} min",
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
    "table": TABLE,
    "start_date": START_DATE.isoformat(),
    "end_date": END_DATE.isoformat(),
    "streams": [
        {
            "symbol": symbol,
            "timeframe": timeframe,
        }
        for symbol, timeframe in STREAMS
    ],
    "jobs": len(jobs),
    "completed_jobs": counters["complete"],
    "skipped_jobs": counters["skipped"],
    "error_jobs": counters["error"],
    "rows": counters["rows"],
    "compressed_bytes": counters["bytes"],
    "compressed_gb": round(
        counters["bytes"]
        / 1024
        / 1024
        / 1024,
        3,
    ),
    "elapsed_seconds": round(
        time.time() - started,
        3,
    ),
    "elapsed_hours": round(
        (time.time() - started) / 3600,
        3,
    ),
    "root": str(ROOT.resolve()),
    "manifest": str(
        GLOBAL_MANIFEST.resolve()
    ),
    "error_log": str(ERROR_LOG.resolve()),
}

summary_path = (
    REPORT_DIR
    / "full_rest_mirror_summary.json"
)

atomic_json_write(
    summary_path,
    summary,
)

print("\n==========================================")
print("MIRROR RUN COMPLETE")
print("==========================================")
print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)
print("\nSummary:", summary_path)
PY

echo
echo "=========================================="
echo "LiqHeat full mirror başlıyor"
echo "=========================================="

python src/full_rest_mirror.py \
  2>&1 | tee -a logs/full_rest_mirror.log

echo
echo "=========================================="
echo "Mirror işlemi sona erdi"
echo "=========================================="

du -sh data/raw-mirror || true

if [[ -f reports/full_rest_mirror_summary.json ]]; then
    cat reports/full_rest_mirror_summary.json
fi
