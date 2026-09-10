import pytest

from collective_bench.benchmark import (
    CollectiveOperation,
)
from collective_bench.metrics import (
    bandwidth_gbps,
    normalized_bandwidth_gbps,
    normalized_transfer_bytes,
)


def test_all_reduce_transfer_bytes() -> None:
    transfer_bytes = normalized_transfer_bytes(
        operation=CollectiveOperation.ALL_REDUCE,
        message_size_bytes=1024,
        world_size=4,
    )

    assert transfer_bytes == pytest.approx(1536.0)


def test_all_gather_transfer_bytes() -> None:
    transfer_bytes = normalized_transfer_bytes(
        operation=CollectiveOperation.ALL_GATHER,
        message_size_bytes=1024,
        world_size=4,
    )

    assert transfer_bytes == pytest.approx(3072.0)


def test_broadcast_transfer_bytes() -> None:
    transfer_bytes = normalized_transfer_bytes(
        operation=CollectiveOperation.BROADCAST,
        message_size_bytes=1024,
        world_size=4,
    )

    assert transfer_bytes == pytest.approx(1024.0)


def test_bandwidth_gbps() -> None:
    result = bandwidth_gbps(
        transfer_bytes=1_000_000_000,
        latency_ms=1000.0,
    )

    assert result == pytest.approx(1.0)


def test_normalized_bandwidth() -> None:
    result = normalized_bandwidth_gbps(
        operation=CollectiveOperation.BROADCAST,
        message_size_bytes=1_000_000_000,
        world_size=4,
        latency_ms=1000.0,
    )

    assert result == pytest.approx(1.0)


def test_bandwidth_rejects_zero_latency() -> None:
    with pytest.raises(ValueError):
        bandwidth_gbps(
            transfer_bytes=1024,
            latency_ms=0.0,
        )


def test_transfer_bytes_rejects_invalid_world_size() -> None:
    with pytest.raises(ValueError):
        normalized_transfer_bytes(
            operation=CollectiveOperation.ALL_REDUCE,
            message_size_bytes=1024,
            world_size=1,
        )
