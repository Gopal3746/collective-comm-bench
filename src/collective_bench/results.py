import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from collective_bench.benchmark import (
    CollectiveOperation,
    LatencySummary,
)


@dataclass(frozen=True)
class BenchmarkResult:
    operation: str
    message_size_bytes: int
    world_size: int
    backend: str
    warmup_iterations: int
    measured_iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


def create_benchmark_result(
    operation: CollectiveOperation,
    message_size_bytes: int,
    world_size: int,
    backend: str,
    warmup_iterations: int,
    measured_iterations: int,
    summary: LatencySummary,
) -> BenchmarkResult:
    return BenchmarkResult(
        operation=operation.value,
        message_size_bytes=message_size_bytes,
        world_size=world_size,
        backend=backend,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        mean_ms=summary.mean_ms,
        median_ms=summary.median_ms,
        min_ms=summary.min_ms,
        max_ms=summary.max_ms,
    )


def write_results_csv(
    results: Sequence[BenchmarkResult],
    output_path: Path,
) -> None:
    if not results:
        raise ValueError("At least one benchmark result is required.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        BenchmarkResult.__dataclass_fields__.keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))
