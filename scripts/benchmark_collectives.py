import os

import torch.distributed as dist
import torch.multiprocessing as mp

from collective_bench.benchmark import (
    BenchmarkConfig,
    CollectiveOperation,
    LatencySummary,
    benchmark_collective,
)

WORLD_SIZE = 4

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29501"

MESSAGE_SIZE_BYTES = 1024 * 1024

WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 20

OPERATIONS = (
    CollectiveOperation.ALL_REDUCE,
    CollectiveOperation.ALL_GATHER,
    CollectiveOperation.BROADCAST,
)


def print_summary(
    operation: CollectiveOperation,
    summary: LatencySummary,
) -> None:
    print()
    print(operation.value)
    print("-" * 40)
    print(f"Mean latency:   {summary.mean_ms:.3f} ms")
    print(f"Median latency: {summary.median_ms:.3f} ms")
    print(f"Min latency:    {summary.min_ms:.3f} ms")
    print(f"Max latency:    {summary.max_ms:.3f} ms")


def run_worker(
    rank: int,
    world_size: int,
) -> None:
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    try:
        config = BenchmarkConfig(
            message_size_bytes=MESSAGE_SIZE_BYTES,
            warmup_iterations=WARMUP_ITERATIONS,
            measured_iterations=MEASURED_ITERATIONS,
        )

        for operation in OPERATIONS:
            summary = benchmark_collective(
                rank=rank,
                world_size=world_size,
                operation=operation,
                config=config,
            )

            if rank == 0:
                if summary is None:
                    raise RuntimeError(
                        "Rank 0 did not receive benchmark results."
                    )

                print_summary(
                    operation=operation,
                    summary=summary,
                )

    finally:
        dist.destroy_process_group()


def main() -> None:
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    message_size_mib = (
        MESSAGE_SIZE_BYTES / (1024 * 1024)
    )

    print("Collective Latency Benchmark")
    print("-" * 40)
    print("Backend:             gloo")
    print(f"World size:          {WORLD_SIZE}")
    print(
        f"Message size:        "
        f"{message_size_mib:.2f} MiB"
    )
    print(
        f"Warmup iterations:   "
        f"{WARMUP_ITERATIONS}"
    )
    print(
        f"Measured iterations: "
        f"{MEASURED_ITERATIONS}"
    )

    mp.spawn(
        run_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
