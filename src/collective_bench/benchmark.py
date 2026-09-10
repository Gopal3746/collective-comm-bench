from dataclasses import dataclass
from enum import Enum
from statistics import mean, median
from time import perf_counter

import torch
import torch.distributed as dist


class CollectiveOperation(str, Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    BROADCAST = "broadcast"


@dataclass(frozen=True)
class BenchmarkConfig:
    message_size_bytes: int
    warmup_iterations: int = 5
    measured_iterations: int = 20


@dataclass(frozen=True)
class LatencySummary:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


def numel_for_bytes(
    size_bytes: int,
    dtype: torch.dtype = torch.float32,
) -> int:
    if size_bytes <= 0:
        raise ValueError("size_bytes must be greater than zero.")

    element_size = torch.empty((), dtype=dtype).element_size()

    if size_bytes % element_size != 0:
        raise ValueError(
            f"size_bytes must be divisible by the dtype element size "
            f"({element_size} bytes)."
        )

    return size_bytes // element_size


def summarize_latencies(
    latencies_seconds: list[float],
) -> LatencySummary:
    if not latencies_seconds:
        raise ValueError("At least one latency measurement is required.")

    latencies_ms = [
        latency * 1000.0
        for latency in latencies_seconds
    ]

    return LatencySummary(
        mean_ms=mean(latencies_ms),
        median_ms=median(latencies_ms),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
    )


def prepare_all_gather_outputs(
    operation: CollectiveOperation,
    tensor: torch.Tensor,
    world_size: int,
) -> list[torch.Tensor] | None:
    if operation != CollectiveOperation.ALL_GATHER:
        return None

    return [
        torch.empty_like(tensor)
        for _ in range(world_size)
    ]


def execute_collective(
    operation: CollectiveOperation,
    tensor: torch.Tensor,
    world_size: int,
    all_gather_outputs: list[torch.Tensor] | None,
) -> None:
    if operation == CollectiveOperation.ALL_REDUCE:
        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )
        return

    if operation == CollectiveOperation.ALL_GATHER:
        if all_gather_outputs is None:
            raise RuntimeError(
                "all_gather requires output tensors."
            )

        if len(all_gather_outputs) != world_size:
            raise RuntimeError(
                "all_gather output list must match world size."
            )

        dist.all_gather(
            all_gather_outputs,
            tensor,
        )
        return

    if operation == CollectiveOperation.BROADCAST:
        dist.broadcast(
            tensor,
            src=0,
        )
        return

    raise ValueError(
        f"Unsupported collective operation: {operation}"
    )


def validate_collective_result(
    operation: CollectiveOperation,
    tensor: torch.Tensor,
    world_size: int,
    all_gather_outputs: list[torch.Tensor] | None,
) -> None:
    if operation == CollectiveOperation.ALL_REDUCE:
        expected = float(
            sum(range(1, world_size + 1))
        )

        if not torch.all(tensor == expected):
            raise RuntimeError(
                "all_reduce produced an unexpected result."
            )

        return

    if operation == CollectiveOperation.ALL_GATHER:
        if all_gather_outputs is None:
            raise RuntimeError(
                "all_gather results are missing."
            )

        for source_rank, output in enumerate(
            all_gather_outputs
        ):
            expected = float(source_rank + 1)

            if not torch.all(output == expected):
                raise RuntimeError(
                    "all_gather produced an unexpected result."
                )

        return

    if operation == CollectiveOperation.BROADCAST:
        expected = 1.0

        if not torch.all(tensor == expected):
            raise RuntimeError(
                "broadcast produced an unexpected result."
            )

        return

    raise ValueError(
        f"Unsupported collective operation: {operation}"
    )


def benchmark_collective(
    rank: int,
    world_size: int,
    operation: CollectiveOperation,
    config: BenchmarkConfig,
) -> LatencySummary | None:
    if config.warmup_iterations < 0:
        raise ValueError(
            "warmup_iterations cannot be negative."
        )

    if config.measured_iterations <= 0:
        raise ValueError(
            "measured_iterations must be greater than zero."
        )

    numel = numel_for_bytes(
        config.message_size_bytes
    )

    tensor = torch.full(
        (numel,),
        fill_value=float(rank + 1),
        dtype=torch.float32,
    )

    all_gather_outputs = prepare_all_gather_outputs(
        operation=operation,
        tensor=tensor,
        world_size=world_size,
    )

    for _ in range(config.warmup_iterations):
        tensor.fill_(float(rank + 1))

        dist.barrier()

        execute_collective(
            operation=operation,
            tensor=tensor,
            world_size=world_size,
            all_gather_outputs=all_gather_outputs,
        )

    local_latencies: list[float] = []

    for _ in range(config.measured_iterations):
        tensor.fill_(float(rank + 1))

        dist.barrier()

        start = perf_counter()

        execute_collective(
            operation=operation,
            tensor=tensor,
            world_size=world_size,
            all_gather_outputs=all_gather_outputs,
        )

        elapsed = perf_counter() - start

        local_latencies.append(elapsed)

    validate_collective_result(
        operation=operation,
        tensor=tensor,
        world_size=world_size,
        all_gather_outputs=all_gather_outputs,
    )

    local_tensor = torch.tensor(
        local_latencies,
        dtype=torch.float64,
    )

    if rank == 0:
        gathered_latencies = [
            torch.empty_like(local_tensor)
            for _ in range(world_size)
        ]
    else:
        gathered_latencies = None

    dist.gather(
        local_tensor,
        gather_list=gathered_latencies,
        dst=0,
    )

    if rank != 0:
        return None

    if gathered_latencies is None:
        raise RuntimeError(
            "Rank 0 did not receive latency measurements."
        )

    measurements = torch.stack(
        gathered_latencies
    )

    slowest_rank_per_iteration = (
        measurements.max(dim=0).values.tolist()
    )

    return summarize_latencies(
        slowest_rank_per_iteration
    )
