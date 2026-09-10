import csv

import pytest

from collective_bench.benchmark import (
    CollectiveOperation,
    LatencySummary,
)
from collective_bench.results import (
    create_benchmark_result,
    write_results_csv,
)


def test_create_benchmark_result() -> None:
    summary = LatencySummary(
        mean_ms=5.5,
        median_ms=5.4,
        min_ms=5.1,
        max_ms=6.0,
    )

    result = create_benchmark_result(
        operation=CollectiveOperation.ALL_REDUCE,
        message_size_bytes=1024,
        world_size=4,
        backend="gloo",
        warmup_iterations=5,
        measured_iterations=20,
        summary=summary,
    )

    assert result.operation == "all_reduce"
    assert result.message_size_bytes == 1024
    assert result.world_size == 4
    assert result.backend == "gloo"
    assert result.mean_ms == pytest.approx(5.5)


def test_write_results_csv(tmp_path) -> None:
    summary = LatencySummary(
        mean_ms=5.5,
        median_ms=5.4,
        min_ms=5.1,
        max_ms=6.0,
    )

    result = create_benchmark_result(
        operation=CollectiveOperation.BROADCAST,
        message_size_bytes=4096,
        world_size=4,
        backend="gloo",
        warmup_iterations=5,
        measured_iterations=20,
        summary=summary,
    )

    output_path = tmp_path / "benchmark.csv"

    write_results_csv(
        results=[result],
        output_path=output_path,
    )

    assert output_path.exists()

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(
            csv.DictReader(csv_file)
        )

    assert len(rows) == 1
    assert rows[0]["operation"] == "broadcast"
    assert rows[0]["message_size_bytes"] == "4096"
    assert rows[0]["world_size"] == "4"
    assert float(rows[0]["mean_ms"]) == pytest.approx(
        5.5
    )


def test_write_results_csv_rejects_empty_results(
    tmp_path,
) -> None:
    output_path = tmp_path / "benchmark.csv"

    with pytest.raises(ValueError):
        write_results_csv(
            results=[],
            output_path=output_path,
        )
