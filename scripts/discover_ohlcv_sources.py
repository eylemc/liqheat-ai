#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
from dotenv import load_dotenv
from supabase import create_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ohlcv_discovery_report.json"
)

DATA_ROOT = PROJECT_ROOT / "data"
DUCKDB_ROOT = DATA_ROOT / "duckdb"

OHLCV_ALIASES = {
    "open": {
        "open",
        "o",
        "open_price",
        "price_open",
    },
    "high": {
        "high",
        "h",
        "high_price",
        "price_high",
    },
    "low": {
        "low",
        "l",
        "low_price",
        "price_low",
    },
    "close": {
        "close",
        "c",
        "close_price",
        "price_close",
    },
    "volume": {
        "volume",
        "v",
        "base_volume",
        "quote_volume",
        "vol",
    },
    "timestamp": {
        "timestamp",
        "time",
        "datetime",
        "date",
        "open_time",
        "close_time",
        "logged_at",
        "created_at",
        "ts",
    },
}

SUPABASE_TABLES = [
    "liq_logging",
    "liquidation_cache",
    "liq_insights",
    "liq_insight_snapshots",
    "ai_analysis_cache",
    "liq_logging_archive_runs",
]


def normalize_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def classify_columns(
    columns: list[str],
) -> dict[str, list[str]]:
    normalized = {
        column: normalize_name(column)
        for column in columns
    }

    matches: dict[str, list[str]] = {}

    for canonical, aliases in OHLCV_ALIASES.items():
        found = [
            original
            for original, normalized_name
            in normalized.items()
            if normalized_name in aliases
        ]

        if found:
            matches[canonical] = found

    return matches


def is_complete_ohlcv(
    matches: dict[str, list[str]],
) -> bool:
    return all(
        key in matches
        for key in (
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
    )


def recursive_key_paths(
    value: Any,
    prefix: str = "",
    depth: int = 0,
    maximum_depth: int = 6,
) -> list[str]:
    if depth > maximum_depth:
        return []

    paths: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            current = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            paths.append(current)

            paths.extend(
                recursive_key_paths(
                    child,
                    current,
                    depth + 1,
                    maximum_depth,
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(
            value[:5]
        ):
            current = (
                f"{prefix}[{index}]"
                if prefix
                else f"[{index}]"
            )

            paths.extend(
                recursive_key_paths(
                    child,
                    current,
                    depth + 1,
                    maximum_depth,
                )
            )

    return paths


def classify_paths(
    paths: list[str],
) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}

    for path in paths:
        leaf = re.split(
            r"[.\[\]]+",
            path,
        )[-1]

        normalized_leaf = normalize_name(leaf)

        for canonical, aliases in OHLCV_ALIASES.items():
            if normalized_leaf in aliases:
                matches.setdefault(
                    canonical,
                    [],
                ).append(path)

    return matches


def inspect_parquet_files() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    if not DATA_ROOT.exists():
        return findings

    parquet_files = sorted(
        DATA_ROOT.rglob("*.parquet")
    )

    print()
    print("=" * 100)
    print("LOCAL PARQUET SCHEMA SCAN")
    print("=" * 100)
    print(
        "Parquet files found:",
        f"{len(parquet_files):,}",
    )

    for index, path in enumerate(
        parquet_files,
        start=1,
    ):
        try:
            schema = pq.read_schema(path)
            columns = schema.names
            matches = classify_columns(columns)

            nested_paths: list[str] = []

            for field in schema:
                field_string = str(field.type).lower()

                if (
                    "struct" in field_string
                    or "list" in field_string
                    or "map" in field_string
                ):
                    nested_paths.append(
                        f"{field.name}: {field.type}"
                    )

            if matches or nested_paths:
                finding = {
                    "path": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "size_mb": round(
                        path.stat().st_size
                        / 1024**2,
                        3,
                    ),
                    "columns": columns,
                    "ohlcv_matches": matches,
                    "complete_ohlcv": (
                        is_complete_ohlcv(
                            matches
                        )
                    ),
                    "nested_fields": nested_paths,
                }

                findings.append(finding)

                print()
                print(
                    f"[{index}/{len(parquet_files)}]",
                    finding["path"],
                )
                print(
                    "OHLCV matches:",
                    matches or "none",
                )
                print(
                    "Complete OHLCV:",
                    finding["complete_ohlcv"],
                )

        except Exception as exc:
            findings.append({
                "path": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    return findings


def inspect_duckdb() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    database_files = sorted(
        DUCKDB_ROOT.glob("*.duckdb")
    ) if DUCKDB_ROOT.exists() else []

    print()
    print("=" * 100)
    print("DUCKDB SCAN")
    print("=" * 100)
    print(
        "DuckDB files found:",
        len(database_files),
    )

    for database_path in database_files:
        database_result: dict[str, Any] = {
            "path": str(
                database_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "objects": [],
        }

        try:
            connection = duckdb.connect(
                str(database_path),
                read_only=True,
            )

            objects = connection.execute(
                """
                SELECT
                    table_schema,
                    table_name,
                    table_type
                FROM information_schema.tables
                WHERE table_schema NOT IN (
                    'information_schema',
                    'pg_catalog'
                )
                ORDER BY
                    table_schema,
                    table_name
                """
            ).fetchall()

            for (
                schema_name,
                table_name,
                table_type,
            ) in objects:
                qualified = (
                    f'"{schema_name}".'
                    f'"{table_name}"'
                )

                try:
                    description = connection.execute(
                        f"DESCRIBE {qualified}"
                    ).fetchall()

                    columns = [
                        row[0]
                        for row in description
                    ]

                    matches = classify_columns(
                        columns
                    )

                    row_count = connection.execute(
                        f"SELECT COUNT(*) "
                        f"FROM {qualified}"
                    ).fetchone()[0]

                    object_result = {
                        "schema": schema_name,
                        "name": table_name,
                        "type": table_type,
                        "rows": int(row_count),
                        "columns": columns,
                        "ohlcv_matches": matches,
                        "complete_ohlcv": (
                            is_complete_ohlcv(
                                matches
                            )
                        ),
                    }

                    database_result[
                        "objects"
                    ].append(object_result)

                    if matches:
                        print()
                        print(
                            f"{database_path.name}:",
                            f"{schema_name}.{table_name}",
                        )
                        print(
                            "Rows:",
                            f"{row_count:,}",
                        )
                        print(
                            "OHLCV matches:",
                            matches,
                        )
                        print(
                            "Complete OHLCV:",
                            object_result[
                                "complete_ohlcv"
                            ],
                        )

                except Exception as exc:
                    database_result[
                        "objects"
                    ].append({
                        "schema": schema_name,
                        "name": table_name,
                        "type": table_type,
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    })

            connection.close()

        except Exception as exc:
            database_result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        findings.append(database_result)

    return findings


def inspect_supabase() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    url = os.getenv("SUPABASE_URL")
    key = os.getenv(
        "SUPABASE_SECRET_KEY"
    )

    print()
    print("=" * 100)
    print("SUPABASE SCAN")
    print("=" * 100)

    if not url or not key:
        print(
            "Supabase credentials not found."
        )

        return [{
            "error": (
                "SUPABASE_URL or "
                "SUPABASE_SECRET_KEY missing"
            )
        }]

    client = create_client(url, key)

    for table in SUPABASE_TABLES:
        table_result: dict[str, Any] = {
            "table": table,
        }

        try:
            response = (
                client
                .table(table)
                .select("*")
                .order(
                    "created_at",
                    desc=True,
                )
                .limit(5)
                .execute()
            )

        except Exception:
            try:
                response = (
                    client
                    .table(table)
                    .select("*")
                    .limit(5)
                    .execute()
                )

            except Exception as exc:
                table_result["error"] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                findings.append(
                    table_result
                )

                continue

        rows = response.data or []

        table_result["sample_rows"] = len(rows)

        if not rows:
            table_result["columns"] = []
            findings.append(table_result)
            continue

        columns = sorted({
            key_name
            for row in rows
            if isinstance(row, dict)
            for key_name in row.keys()
        })

        direct_matches = classify_columns(
            columns
        )

        all_paths: list[str] = []

        for row in rows:
            all_paths.extend(
                recursive_key_paths(row)
            )

        nested_matches = classify_paths(
            sorted(set(all_paths))
        )

        table_result.update({
            "columns": columns,
            "direct_ohlcv_matches": (
                direct_matches
            ),
            "nested_ohlcv_matches": (
                nested_matches
            ),
            "complete_direct_ohlcv": (
                is_complete_ohlcv(
                    direct_matches
                )
            ),
            "complete_nested_ohlcv": (
                is_complete_ohlcv(
                    nested_matches
                )
            ),
            "sample_key_paths": sorted(
                set(all_paths)
            )[:300],
        })

        findings.append(table_result)

        print()
        print("Table:", table)
        print(
            "Sample rows:",
            len(rows),
        )
        print(
            "Columns:",
            ", ".join(columns),
        )
        print(
            "Direct OHLCV:",
            direct_matches or "none",
        )
        print(
            "Nested OHLCV:",
            nested_matches or "none",
        )

    return findings


def summarize(
    parquet_results: list[dict[str, Any]],
    duckdb_results: list[dict[str, Any]],
    supabase_results: list[dict[str, Any]],
) -> dict[str, Any]:
    local_complete = [
        item
        for item in parquet_results
        if item.get("complete_ohlcv")
    ]

    duckdb_complete = []

    for database in duckdb_results:
        for item in database.get(
            "objects",
            [],
        ):
            if item.get("complete_ohlcv"):
                duckdb_complete.append({
                    "database": database.get(
                        "path"
                    ),
                    **item,
                })

    supabase_complete = [
        item
        for item in supabase_results
        if (
            item.get(
                "complete_direct_ohlcv"
            )
            or item.get(
                "complete_nested_ohlcv"
            )
        )
    ]

    usable_sources = (
        len(local_complete)
        + len(duckdb_complete)
        + len(supabase_complete)
    )

    recommendation = (
        "USE_EXISTING_OHLCV"
        if usable_sources
        else "DOWNLOAD_BINANCE_1M_OHLCV"
    )

    return {
        "usable_source_count": (
            usable_sources
        ),
        "local_complete_sources": (
            local_complete
        ),
        "duckdb_complete_sources": (
            duckdb_complete
        ),
        "supabase_complete_sources": (
            supabase_complete
        ),
        "recommendation": recommendation,
    }


def main() -> int:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_results = (
        inspect_parquet_files()
    )

    duckdb_results = inspect_duckdb()

    supabase_results = (
        inspect_supabase()
    )

    summary = summarize(
        parquet_results,
        duckdb_results,
        supabase_results,
    )

    report = {
        "summary": summary,
        "parquet": parquet_results,
        "duckdb": duckdb_results,
        "supabase": supabase_results,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("OHLCV DISCOVERY COMPLETE")
    print("=" * 100)
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print("Report:", REPORT_PATH)

    if (
        summary["recommendation"]
        == "DOWNLOAD_BINANCE_1M_OHLCV"
    ):
        print()
        print(
            "RESULT: Existing sources do not "
            "contain complete OHLCV."
        )
        print(
            "NEXT: Download BTCUSDT, ETHUSDT "
            "and SOLUSDT 1m candles."
        )
    else:
        print()
        print(
            "RESULT: At least one complete "
            "OHLCV source was found."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
