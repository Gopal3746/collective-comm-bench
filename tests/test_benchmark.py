import pytest
import torch

from collective_bench.benchmark import (
    BenchmarkConfig,
    CollectiveOperation,
    numel_for_bytes,
    prepare_all_gather_outputs,
    summarize_latencies,
)


def test_numel_for_one_kib_float32() -> None:
    assert numel_for_bytes(
        1024,
        dtype=torch.float32,
    ) == 256


def test_numel_rejects_zero_bytes() -> None:
    with pytest.raises(ValueError):
        numel_for_bytes(0)


def test_numel_rejects_unaligned_size() -> None:
    with pytest.raises(ValueError):
        numel_for_bytes(
            1025,
            dtype=torch.float32,
        )


def test_summarize_latencies() -> None:
    summary = summarize_latencies(
        [0.001, 0.002, 0.003],
    )

    assert summary.mean_ms == pytest.approx(2.0)
    assert summary.median_ms == pytest.approx(2.0)
    assert summary.min_ms == pytest.approx(1.0)
    assert summary.max_ms == pytest.approx(3.0)


def test_summarize_latencies_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        summarize_latencies([])


def test_collective_operation_values() -> None:
    assert CollectiveOperation.ALL_REDUCE.value == "all_reduce"
    assert CollectiveOperation.ALL_GATHER.value == "all_gather"
    assert CollectiveOperation.BROADCAST.value == "broadcast"


def test_prepare_all_gather_outputs() -> None:
    tensor = torch.zeros(
        256,
        dtype=torch.float32,
    )

    outputs = prepare_all_gather_outputs(
        operation=CollectiveOperation.ALL_GATHER,
        tensor=tensor,
        world_size=4,
    )

    assert outputs is not None
    assert len(outputs) == 4

    for output in outputs:
        assert output.shape == tensor.shape
        assert output.dtype == tensor.dtype


def test_non_all_gather_needs_no_output_buffers() -> None:
    tensor = torch.zeros(
        256,
        dtype=torch.float32,
    )

    outputs = prepare_all_gather_outputs(
        operation=CollectiveOperation.ALL_REDUCE,
        tensor=tensor,
        world_size=4,
    )

    assert outputs is None


def test_benchmark_config_defaults() -> None:
    config = BenchmarkConfig(
        message_size_bytes=1024
    )

    assert config.warmup_iterations == 5
    assert config.measured_iterations == 20
