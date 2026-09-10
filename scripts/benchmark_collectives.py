import os
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from collective_bench.benchmark import (
    BenchmarkConfig,
    CollectiveOperation,
    LatencySummary,
    benchmark_collective,
)
from collective_bench.results import (
    BenchmarkResult,
    create_benchmark_result,
    write_results_csv,
)

WORLD_SIZE = 4

BACKEND = "gloo"

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29501"

KIB = 1024
MIB = 1024 * KIB

MESSAGE_SIZES_BYTES = (
    1 * KIB,
    4 * KIB,
    16 * KIB,
    64 * KIB,
    256 * KIB,
    1 * MIB,
    4 * MIB,
    16 * MIB,
    64 * MIB,
)

WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 20

OPERATIONS = (
    CollectiveOperation.ALL_REDUCE,
    CollectiveOperation.ALL_GATHER,
    CollectiveOperation.BROADCAST,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmark_results.csv"
)


def format_message_size(size_bytes: int) -> str:
    if size_bytes >= MIB:
        return f"{size_bytes / MIB:.0f} MiB"

    return f"{size_bytes / KIB:.0f} KiB"


def print_summary(
    operation: CollectiveOperation,
    message_size_bytes: int,
    summary: LatencySummary,
) -> None:
    print(
        f"{operation.value:<12} "
        f"{format_message_size(message_size_bytes):>8} "
        f"mean={summary.mean_ms:>9.3f} ms "
        f"median={summary.median_ms:>9.3f} ms"
    )


def run_worker(
    rank: int,
    world_size: int,
) -> None:
    dist.init_process_group(
        backend=BACKEND,
        rank=rank,
        world_size=world_size,
    )

    results: list[BenchmarkResult] = []

    try:
        for message_size_bytes in MESSAGE_SIZES_BYTES:
            config = BenchmarkConfig(
                message_size_bytes=message_size_bytes,
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

                if rank != 0:
                    continue

                if summary is None:
                    raise RuntimeError(
                        "Rank 0 did not receive benchmark results."
                    )

                result = create_benchmark_result(
                    operation=operation,
                    message_size_bytes=message_size_bytes,
                    world_size=world_size,
                    backend=BACKEND,
                    warmup_iterations=WARMUP_ITERATIONS,
                    measured_iterations=MEASURED_ITERATIONS,
                    summary=summary,
                )

                results.append(result)

                print_summary(
                    operation=operation,
                    message_size_bytes=message_size_bytes,
                    summary=summary,
                )

                write_results_csv(
                    results=results,
                    output_path=RESULTS_PATH,
                )

    finally:
        dist.destroy_process_group()

    if rank == 0:
        print()
        print(
            f"Saved {len(results)} benchmark results to:"
        )
        print(RESULTS_PATH)


def main() -> None:
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    print("Distributed Collective Message-Size Sweep")
    print("-" * 60)
    print(f"Backend:             {BACKEND}")
    print(f"World size:          {WORLD_SIZE}")
    print(f"Operations:          {len(OPERATIONS)}")
    print(
        f"Message sizes:       {len(MESSAGE_SIZES_BYTES)}"
    )
    print(
        f"Benchmark points:    "
        f"{len(OPERATIONS) * len(MESSAGE_SIZES_BYTES)}"
    )
    print(
        f"Warmup iterations:   {WARMUP_ITERATIONS}"
    )
    print(
        f"Measured iterations: "
        f"{MEASURED_ITERATIONS}"
    )
    print()
    print(
        f"{'Operation':<12} "
        f"{'Size':>8} "
        f"{'Latency'}"
    )
    print("-" * 60)

    mp.spawn(
        run_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
