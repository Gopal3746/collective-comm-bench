from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter

import torch
import torch.distributed as dist


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


def summarize_latencies(latencies_seconds: list[float]) -> LatencySummary:
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


def benchmark_all_reduce(
    rank: int,
    world_size: int,
    config: BenchmarkConfig,
) -> LatencySummary | None:
    numel = numel_for_bytes(config.message_size_bytes)

    tensor = torch.full(
        (numel,),
        fill_value=float(rank + 1),
        dtype=torch.float32,
    )

    for _ in range(config.warmup_iterations):
        tensor.fill_(float(rank + 1))

        dist.barrier()

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

    local_latencies: list[float] = []

    for _ in range(config.measured_iterations):
        tensor.fill_(float(rank + 1))

        dist.barrier()

        start = perf_counter()

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

        elapsed = perf_counter() - start

        local_latencies.append(elapsed)

    local_tensor = torch.tensor(
        local_latencies,
        dtype=torch.float64,
    )

    if rank == 0:
        gathered = [
            torch.empty_like(local_tensor)
            for _ in range(world_size)
        ]
    else:
        gathered = None

    dist.gather(
        local_tensor,
        gather_list=gathered,
        dst=0,
    )

    if rank != 0:
        return None

    if gathered is None:
        raise RuntimeError("Rank 0 did not receive latency measurements.")

    measurements = torch.stack(gathered)

    slowest_rank_per_iteration = measurements.max(dim=0).values.tolist()

    return summarize_latencies(slowest_rank_per_iteration)
