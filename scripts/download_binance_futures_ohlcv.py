#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

START_TIME_UTC = "2026-03-30T00:00:00Z"
REQUEST_LIMIT = 1000
ONE_MINUTE_MS = 60_000

RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

NUMERIC_FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def completed_minute_boundary() -> pd.Timestamp:
    """
    Son kapanmış 1m mumunun open-time üst sınırı.

    Örnek:
      şimdi 18:42:37 ise 18:42 mumu açık olduğundan
      indirilecek son open time 18:41:00'dır.
    """
    return utc_now().floor("min") - pd.Timedelta(minutes=1)


def timestamp_to_ms(value: pd.Timestamp) -> int:
    return int(value.timestamp() * 1000)


def ensure_utc(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)

    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")

    return parsed


def request_klines(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
    maximum_attempts: int = 8,
) -> list[list[Any]]:
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": REQUEST_LIMIT,
    }

    for attempt in range(1, maximum_attempts + 1):
        try:
            response = session.get(
                BASE_URL + KLINES_ENDPOINT,
                params=params,
                timeout=30,
            )

            if response.status_code == 429:
                retry_after = int(
                    response.headers.get(
                        "Retry-After",
                        max(2, attempt * 2),
                    )
                )

                print(
                    f"{symbol}: rate limited; "
                    f"sleeping {retry_after}s",
                    flush=True,
                )

                time.sleep(retry_after)
                continue

            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, list):
                raise RuntimeError(
                    f"Unexpected Binance response: {payload}"
                )

            return payload

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as exc:
            if attempt >= maximum_attempts:
                raise

            delay = min(30, 2 ** attempt)

            print(
                f"{symbol}: request failed "
                f"({type(exc).__name__}: {exc}); "
                f"retry in {delay}s",
                flush=True,
            )

            time.sleep(delay)

    raise RuntimeError("Unreachable retry state")


def payload_to_frame(
    rows: list[list[Any]],
    symbol: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(
        rows,
        columns=RAW_COLUMNS,
    )

    frame["symbol"] = symbol

    frame["open_time"] = pd.to_datetime(
        frame["open_time"],
        unit="ms",
        utc=True,
    )

    frame["close_time"] = pd.to_datetime(
        frame["close_time"],
        unit="ms",
        utc=True,
    )

    for column in NUMERIC_FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).astype("float64")

    frame["trade_count"] = pd.to_numeric(
        frame["trade_count"],
        errors="coerce",
    ).astype("Int64")

    frame = frame.drop(
        columns=["ignore"],
    )

    frame = frame.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    return frame


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_parquet(path)

    frame["open_time"] = pd.to_datetime(
        frame["open_time"],
        utc=True,
        errors="coerce",
    )

    frame["close_time"] = pd.to_datetime(
        frame["close_time"],
        utc=True,
        errors="coerce",
    )

    return frame


def validate_ohlcv(
    frame: pd.DataFrame,
    symbol: str,
) -> dict[str, Any]:
    ordered = (
        frame
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    duplicated = int(
        ordered["open_time"].duplicated().sum()
    )

    time_differences = (
        ordered["open_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    gaps = ordered.loc[
        time_differences > 1.0,
        ["open_time"],
    ].copy()

    if not gaps.empty:
        gaps["gap_minutes"] = (
            time_differences.loc[gaps.index]
            .astype(float)
        )

        gaps["previous_open_time"] = (
            ordered["open_time"]
            .shift(1)
            .loc[gaps.index]
        )

    invalid_high = int(
        (
            ordered["high"]
            < ordered[
                ["open", "close", "low"]
            ].max(axis=1)
        ).sum()
    )

    invalid_low = int(
        (
            ordered["low"]
            > ordered[
                ["open", "close", "high"]
            ].min(axis=1)
        ).sum()
    )

    negative_volume = int(
        (ordered["volume"] < 0).sum()
    )

    report: dict[str, Any] = {
        "symbol": symbol,
        "rows": int(len(ordered)),
        "minimum_open_time": (
            ordered["open_time"].min().isoformat()
            if len(ordered)
            else None
        ),
        "maximum_open_time": (
            ordered["open_time"].max().isoformat()
            if len(ordered)
            else None
        ),
        "duplicate_open_times": duplicated,
        "gap_count": int(len(gaps)),
        "missing_minutes_estimate": int(
            sum(
                max(
                    0,
                    math.floor(value) - 1,
                )
                for value in gaps.get(
                    "gap_minutes",
                    pd.Series(dtype=float),
                )
            )
        ),
        "invalid_high_rows": invalid_high,
        "invalid_low_rows": invalid_low,
        "negative_volume_rows": negative_volume,
        "first_gaps": (
            [
                {
                    "previous_open_time": (
                        row["previous_open_time"]
                        .isoformat()
                    ),
                    "next_open_time": (
                        row["open_time"]
                        .isoformat()
                    ),
                    "gap_minutes": float(
                        row["gap_minutes"]
                    ),
                }
                for _, row in gaps.head(20).iterrows()
            ]
            if len(gaps)
            else []
        ),
    }

    return report


def write_atomic_parquet(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    frame.to_parquet(
        temporary_path,
        index=False,
        compression="zstd",
    )

    temporary_path.replace(output_path)


def aggregate_candles(
    one_minute: pd.DataFrame,
    rule: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    UTC epoch tabanlı, sol-kapalı mum üretir.

    1h:
      00:00–00:59

    4h:
      00:00–03:59, 04:00–07:59 ...

    24h:
      00:00–23:59 UTC
    """
    ordered = (
        one_minute
        .sort_values("open_time")
        .set_index("open_time")
    )

    aggregated = ordered.resample(
        rule,
        origin="epoch",
        offset="0min",
        label="left",
        closed="left",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trade_count": "sum",
        "taker_buy_base_volume": "sum",
        "taker_buy_quote_volume": "sum",
    })

    candle_counts = (
        ordered["close"]
        .resample(
            rule,
            origin="epoch",
            offset="0min",
            label="left",
            closed="left",
        )
        .count()
        .rename("source_1m_count")
    )

    aggregated = aggregated.join(
        candle_counts
    )

    aggregated = aggregated.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    expected_counts = {
        "1h": 60,
        "4h": 240,
        "24h": 1440,
    }

    expected_count = expected_counts[
        timeframe
    ]

    aggregated[
        "is_complete"
    ] = (
        aggregated["source_1m_count"]
        == expected_count
    )

    aggregated["timeframe"] = timeframe

    aggregated["symbol"] = str(
        one_minute["symbol"].iloc[0]
    )

    duration = pd.Timedelta(rule)

    aggregated["close_time"] = (
        aggregated.index
        + duration
        - pd.Timedelta(milliseconds=1)
    )

    aggregated = (
        aggregated
        .reset_index()
        .rename(
            columns={
                "open_time": "open_time",
            }
        )
    )

    return aggregated


def download_symbol(
    session: requests.Session,
    symbol: str,
    output_root: Path,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> dict[str, Any]:
    symbol_root = output_root / symbol

    raw_path = (
        symbol_root
        / "1m"
        / f"{symbol}-1m.parquet"
    )

    existing = load_existing(raw_path)

    if existing.empty:
        next_start = requested_start
    else:
        existing = (
            existing
            .sort_values("open_time")
            .drop_duplicates(
                subset=["open_time"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        next_start = (
            existing["open_time"].max()
            + pd.Timedelta(minutes=1)
        )

        if next_start < requested_start:
            next_start = requested_start

    print()
    print("=" * 100)
    print(symbol)
    print("=" * 100)
    print("Existing rows:", f"{len(existing):,}")
    print("Download from:", next_start.isoformat())
    print("Download to  :", requested_end.isoformat())

    downloaded_parts: list[pd.DataFrame] = []

    cursor_ms = timestamp_to_ms(next_start)
    end_ms = timestamp_to_ms(requested_end)

    request_count = 0
    downloaded_rows = 0

    while cursor_ms <= end_ms:
        batch = request_klines(
            session=session,
            symbol=symbol,
            start_ms=cursor_ms,
            end_ms=end_ms,
        )

        request_count += 1

        if not batch:
            break

        batch_frame = payload_to_frame(
            batch,
            symbol,
        )

        if batch_frame.empty:
            break

        batch_frame = batch_frame[
            batch_frame["open_time"]
            <= requested_end
        ]

        if batch_frame.empty:
            break

        downloaded_parts.append(
            batch_frame
        )

        downloaded_rows += len(
            batch_frame
        )

        last_open_time = (
            batch_frame["open_time"].max()
        )

        cursor_ms = (
            timestamp_to_ms(last_open_time)
            + ONE_MINUTE_MS
        )

        if (
            request_count % 25 == 0
            or len(batch) < REQUEST_LIMIT
        ):
            print(
                f"{symbol}: requests={request_count:,} "
                f"downloaded={downloaded_rows:,} "
                f"latest={last_open_time.isoformat()}",
                flush=True,
            )

        if len(batch) < REQUEST_LIMIT:
            break

        # Conservative pacing.
        time.sleep(0.08)

    frames = []

    if not existing.empty:
        frames.append(existing)

    frames.extend(downloaded_parts)

    if not frames:
        raise RuntimeError(
            f"No OHLCV data available for {symbol}"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
    )

    combined = combined[
        combined["open_time"].between(
            requested_start,
            requested_end,
            inclusive="both",
        )
    ].reset_index(drop=True)

    write_atomic_parquet(
        combined,
        raw_path,
    )

    validation = validate_ohlcv(
        combined,
        symbol,
    )

    aggregate_reports: dict[str, Any] = {}

    for timeframe, rule in [
        ("1h", "1h"),
        ("4h", "4h"),
        ("24h", "24h"),
    ]:
        aggregate = aggregate_candles(
            combined,
            rule,
            timeframe,
        )

        aggregate_path = (
            symbol_root
            / timeframe
            / f"{symbol}-{timeframe}.parquet"
        )

        write_atomic_parquet(
            aggregate,
            aggregate_path,
        )

        aggregate_reports[
            timeframe
        ] = {
            "rows": int(len(aggregate)),
            "complete_rows": int(
                aggregate["is_complete"].sum()
            ),
            "incomplete_rows": int(
                (~aggregate["is_complete"]).sum()
            ),
            "minimum_open_time": (
                aggregate["open_time"]
                .min()
                .isoformat()
                if len(aggregate)
                else None
            ),
            "maximum_open_time": (
                aggregate["open_time"]
                .max()
                .isoformat()
                if len(aggregate)
                else None
            ),
            "output": str(
                aggregate_path
            ),
        }

    result = {
        "symbol": symbol,
        "requested_start": (
            requested_start.isoformat()
        ),
        "requested_end": (
            requested_end.isoformat()
        ),
        "existing_rows": int(
            len(existing)
        ),
        "downloaded_rows": int(
            downloaded_rows
        ),
        "final_rows": int(
            len(combined)
        ),
        "request_count": int(
            request_count
        ),
        "raw_output": str(raw_path),
        "raw_output_mb": round(
            raw_path.stat().st_size
            / 1024**2,
            3,
        ),
        "validation": validation,
        "aggregates": aggregate_reports,
    }

    print()
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download Binance USD-M Futures "
            "1m OHLCV and aggregate candles."
        )
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
    )

    parser.add_argument(
        "--start",
        default=START_TIME_UTC,
    )

    parser.add_argument(
        "--end",
        default=None,
        help=(
            "UTC end open-time. Default: "
            "last fully closed one-minute candle."
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            "data/market/"
            "binance-futures-um"
        ),
    )

    args = parser.parse_args()

    started = time.time()

    start = ensure_utc(args.start)

    end = (
        ensure_utc(args.end)
        if args.end
        else completed_minute_boundary()
    )

    if end < start:
        raise ValueError(
            f"End before start: {start} > {end}"
        )

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "LiqHeat-Matrix-Research/1.0"
        ),
        "Accept": "application/json",
    })

    print("=" * 100)
    print(
        "BINANCE USD-M FUTURES "
        "OHLCV DOWNLOAD"
    )
    print("=" * 100)
    print("Symbols:", args.symbols)
    print("Start  :", start.isoformat())
    print("End    :", end.isoformat())
    print("Source :", BASE_URL + KLINES_ENDPOINT)
    print()

    results = []

    for symbol in args.symbols:
        result = download_symbol(
            session=session,
            symbol=str(symbol).upper(),
            output_root=output_root,
            requested_start=start,
            requested_end=end,
        )

        results.append(result)

    report = {
        "status": "complete",
        "source": (
            BASE_URL + KLINES_ENDPOINT
        ),
        "market": "binance-usd-m-futures",
        "interval": "1m",
        "symbols": [
            str(symbol).upper()
            for symbol in args.symbols
        ],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "results": results,
        "elapsed_seconds": round(
            time.time() - started,
            3,
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    report_path = Path(
        "reports/market_data/"
        "binance_futures_ohlcv_report.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("OHLCV DOWNLOAD COMPLETE")
    print("=" * 100)
    print("Report:", report_path)
    print(
        "Elapsed:",
        f"{report['elapsed_seconds']:.1f}s",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
