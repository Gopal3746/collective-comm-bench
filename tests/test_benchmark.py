import pytest
import torch

from collective_bench.benchmark import (
    numel_for_bytes,
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
