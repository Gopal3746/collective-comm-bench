from collective_bench.benchmark import CollectiveOperation


def normalized_transfer_bytes(
    operation: CollectiveOperation,
    message_size_bytes: int,
    world_size: int,
) -> float:
    if message_size_bytes <= 0:
        raise ValueError(
            "message_size_bytes must be greater than zero."
        )

    if world_size <= 1:
        raise ValueError(
            "world_size must be greater than one."
        )

    if operation == CollectiveOperation.ALL_REDUCE:
        factor = 2 * (world_size - 1) / world_size
        return message_size_bytes * factor

    if operation == CollectiveOperation.ALL_GATHER:
        return message_size_bytes * (world_size - 1)

    if operation == CollectiveOperation.BROADCAST:
        return float(message_size_bytes)

    raise ValueError(
        f"Unsupported collective operation: {operation}"
    )


def bandwidth_gbps(
    transfer_bytes: float,
    latency_ms: float,
) -> float:
    if transfer_bytes <= 0:
        raise ValueError(
            "transfer_bytes must be greater than zero."
        )

    if latency_ms <= 0:
        raise ValueError(
            "latency_ms must be greater than zero."
        )

    latency_seconds = latency_ms / 1000.0

    return transfer_bytes / latency_seconds / 1_000_000_000


def normalized_bandwidth_gbps(
    operation: CollectiveOperation,
    message_size_bytes: int,
    world_size: int,
    latency_ms: float,
) -> float:
    transfer_bytes = normalized_transfer_bytes(
        operation=operation,
        message_size_bytes=message_size_bytes,
        world_size=world_size,
    )

    return bandwidth_gbps(
        transfer_bytes=transfer_bytes,
        latency_ms=latency_ms,
    )
