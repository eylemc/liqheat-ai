from __future__ import annotations

from pathlib import Path
import itertools
import json
import re
import subprocess
import sys
import time


SOURCE_SCRIPT = Path(
    "scripts/research_topology_v2_squeeze_events.py"
)

GENERATED_DIR = Path(
    "scripts/generated_squeeze_grid"
)

OUTPUT_ROOT = Path(
    "data/research/topology_v2_squeeze_grid"
)

LOG_DIR = Path("logs")

PYTHON = Path(".venv/bin/python")

TIMEFRAMES = [
    "1h",
    "4h",
    "24h",
    "1w",
]

FUTURE_MINUTES = [
    15,
    30,
    60,
]

PRECURSOR_MINUTES = [
    5,
    15,
    30,
]

VOLATILITY_QUANTILES = [
    0.90,
    0.95,
    0.975,
]

EXPECTED_RESULT_FILES = [
    "detected_squeeze_events.parquet",
    "squeeze_event_dataset.parquet",
    "walk_forward_summary.csv",
    "walk_forward_predictions.parquet",
    "feature_importance.csv",
    "report.json",
]


def replace_required(
    text: str,
    pattern: str,
    replacement: str,
    description: str,
) -> str:
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )

    if count != 1:
        raise RuntimeError(
            f"Could not patch {description}; matches={count}"
        )

    return updated


def quantile_slug(
    value: float,
) -> str:
    return (
        f"{value:g}"
        .replace(".", "p")
    )


def experiment_name(
    timeframe: str,
    future_minutes: int,
    precursor_minutes: int,
    volatility_quantile: float,
) -> str:
    return (
        f"tf_{timeframe}"
        f"__future_{future_minutes}m"
        f"__precursor_{precursor_minutes}m"
        f"__q_{quantile_slug(volatility_quantile)}"
    )


def experiment_complete(
    output_dir: Path,
) -> bool:
    if not output_dir.exists():
        return False

    for filename in EXPECTED_RESULT_FILES:
        path = output_dir / filename

        if (
            not path.exists()
            or path.stat().st_size == 0
        ):
            return False

    report_path = output_dir / "report.json"

    try:
        report = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    required_report_keys = {
        "events",
        "dataset",
        "walk_forward",
        "aggregate",
    }

    return required_report_keys.issubset(
        report.keys()
    )


def create_variant(
    source_text: str,
    timeframe: str,
    future_minutes: int,
    precursor_minutes: int,
    volatility_quantile: float,
) -> tuple[
    str,
    Path,
    Path,
]:
    name = experiment_name(
        timeframe=timeframe,
        future_minutes=future_minutes,
        precursor_minutes=precursor_minutes,
        volatility_quantile=volatility_quantile,
    )

    script_path = (
        GENERATED_DIR
        / f"{name}.py"
    )

    output_dir = (
        OUTPUT_ROOT
        / name
    )

    text = source_text

    text = replace_required(
        text,
        (
            r'^OUTPUT_DIR = Path\(\n'
            r'\s*"data/research/topology_v2_squeeze_events"\n'
            r'\)$'
        ),
        (
            "OUTPUT_DIR = Path(\n"
            f'    "{output_dir.as_posix()}"\n'
            ")"
        ),
        "OUTPUT_DIR",
    )

    text = replace_required(
        text,
        r'^TOPOLOGY_TIMEFRAME = "[^"]+"$',
        (
            f'TOPOLOGY_TIMEFRAME = "{timeframe}"'
        ),
        "TOPOLOGY_TIMEFRAME",
    )

    text = replace_required(
        text,
        (
            r'^FUTURE_WINDOW = '
            r'pd\.Timedelta\(minutes=\d+\)$'
        ),
        (
            "FUTURE_WINDOW = "
            f"pd.Timedelta(minutes={future_minutes})"
        ),
        "FUTURE_WINDOW",
    )

    text = replace_required(
        text,
        (
            r'^PRECURSOR_OFFSET = '
            r'pd\.Timedelta\(minutes=\d+\)$'
        ),
        (
            "PRECURSOR_OFFSET = "
            f"pd.Timedelta(minutes={precursor_minutes})"
        ),
        "PRECURSOR_OFFSET",
    )

    text = replace_required(
        text,
        (
            r'^VOLATILITY_QUANTILE = '
            r'[0-9.]+$'
        ),
        (
            "VOLATILITY_QUANTILE = "
            f"{volatility_quantile}"
        ),
        "VOLATILITY_QUANTILE",
    )

    script_path.write_text(
        text,
        encoding="utf-8",
    )

    return (
        name,
        script_path,
        output_dir,
    )


def main() -> int:
    if not SOURCE_SCRIPT.exists():
        print(
            f"ERROR: source script missing: "
            f"{SOURCE_SCRIPT}",
            file=sys.stderr,
        )
        return 1

    if not PYTHON.exists():
        print(
            f"ERROR: virtualenv Python missing: "
            f"{PYTHON}",
            file=sys.stderr,
        )
        return 1

    GENERATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_text = (
        SOURCE_SCRIPT.read_text(
            encoding="utf-8"
        )
    )

    # Önce kaynak scriptin syntax kontrolü.
    compile_result = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(SOURCE_SCRIPT),
        ],
        capture_output=True,
        text=True,
    )

    if compile_result.returncode != 0:
        print(
            "ERROR: source squeeze script "
            "does not compile.",
            file=sys.stderr,
        )
        print(
            compile_result.stderr,
            file=sys.stderr,
        )
        return 1

    experiments = list(
        itertools.product(
            TIMEFRAMES,
            FUTURE_MINUTES,
            PRECURSOR_MINUTES,
            VOLATILITY_QUANTILES,
        )
    )

    total_experiments = len(experiments)

    print("=" * 90)
    print(
        "TOPOLOGY V2 — OVERNIGHT "
        "FULL SQUEEZE RESEARCH GRID"
    )
    print("=" * 90)
    print(
        f"Experiments        : "
        f"{total_experiments}"
    )
    print(
        f"Topology timeframes: "
        f"{TIMEFRAMES}"
    )
    print(
        f"Future windows     : "
        f"{FUTURE_MINUTES} minutes"
    )
    print(
        f"Precursor offsets  : "
        f"{PRECURSOR_MINUTES} minutes"
    )
    print(
        f"Volatility levels  : "
        f"{VOLATILITY_QUANTILES}"
    )
    print()
    print(
        "Every experiment uses every symbol "
        "and every available row in the selected "
        "topology timeframe."
    )
    print()

    overall_started = time.time()

    succeeded = 0
    skipped = 0
    failed = 0

    failed_experiments = []

    for index, parameters in enumerate(
        experiments,
        start=1,
    ):
        (
            timeframe,
            future_minutes,
            precursor_minutes,
            volatility_quantile,
        ) = parameters

        try:
            (
                name,
                script_path,
                output_dir,
            ) = create_variant(
                source_text=source_text,
                timeframe=timeframe,
                future_minutes=future_minutes,
                precursor_minutes=precursor_minutes,
                volatility_quantile=volatility_quantile,
            )
        except Exception as error:
            failed += 1
            failed_experiments.append(
                {
                    "experiment": (
                        experiment_name(
                            timeframe,
                            future_minutes,
                            precursor_minutes,
                            volatility_quantile,
                        )
                    ),
                    "error": (
                        f"variant creation failed: "
                        f"{error}"
                    ),
                }
            )

            print()
            print(
                f"[{index:03d}/{total_experiments:03d}] "
                f"VARIANT CREATION FAILED: {error}",
                flush=True,
            )
            continue

        if experiment_complete(
            output_dir
        ):
            skipped += 1

            print()
            print(
                f"[{index:03d}/{total_experiments:03d}] "
                f"SKIP completed: {name}",
                flush=True,
            )
            continue

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_path = (
            LOG_DIR
            / f"{name}.log"
        )

        print()
        print("-" * 90)
        print(
            f"[{index:03d}/{total_experiments:03d}] "
            f"{name}"
        )
        print(f"Script : {script_path}")
        print(f"Output : {output_dir}")
        print(f"Log    : {log_path}")
        print("-" * 90, flush=True)

        syntax_result = subprocess.run(
            [
                str(PYTHON),
                "-m",
                "py_compile",
                str(script_path),
            ],
            capture_output=True,
            text=True,
        )

        if syntax_result.returncode != 0:
            failed += 1

            error_text = (
                syntax_result.stderr.strip()
                or "generated script compile failed"
            )

            failed_experiments.append(
                {
                    "experiment": name,
                    "error": error_text,
                }
            )

            log_path.write_text(
                error_text + "\n",
                encoding="utf-8",
            )

            print(
                f"FAILED compile: {name}",
                flush=True,
            )
            continue

        experiment_started = time.time()

        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_file:
            process = subprocess.run(
                [
                    str(PYTHON),
                    str(script_path),
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        elapsed_seconds = (
            time.time()
            - experiment_started
        )

        if (
            process.returncode == 0
            and experiment_complete(
                output_dir
            )
        ):
            succeeded += 1
            status = "OK"
        else:
            failed += 1
            status = (
                f"FAILED({process.returncode})"
            )

            failed_experiments.append(
                {
                    "experiment": name,
                    "return_code": (
                        process.returncode
                    ),
                    "log": str(log_path),
                }
            )

        print(
            f"{status}: {name} "
            f"in {elapsed_seconds:.1f}s",
            flush=True,
        )

        completed_count = (
            succeeded
            + skipped
            + failed
        )

        average_seconds = (
            (
                time.time()
                - overall_started
            )
            / max(
                1,
                completed_count,
            )
        )

        remaining = (
            total_experiments
            - completed_count
        )

        estimated_remaining_minutes = (
            average_seconds
            * remaining
            / 60
        )

        print(
            f"Progress: "
            f"{completed_count}/"
            f"{total_experiments} | "
            f"estimated remaining "
            f"{estimated_remaining_minutes:.1f} min",
            flush=True,
        )

    total_elapsed = (
        time.time()
        - overall_started
    )

    failure_path = (
        OUTPUT_ROOT
        / "failed_experiments.json"
    )

    failure_path.write_text(
        json.dumps(
            failed_experiments,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("OVERNIGHT GRID FINISHED")
    print("=" * 90)
    print(f"Succeeded : {succeeded}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(
        f"Elapsed   : "
        f"{total_elapsed / 60:.1f} minutes"
    )
    print(
        f"Failures  : {failure_path}"
    )

    # Bazı deneyler hata verse bile özet scriptinin
    # tamamlananları işlemesi için exit code 0 dön.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
