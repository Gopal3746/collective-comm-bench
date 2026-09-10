import os

import torch.distributed as dist
import torch.multiprocessing as mp

from collective_bench.benchmark import (
    BenchmarkConfig,
    benchmark_all_reduce,
)

WORLD_SIZE = 4

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29501"

MESSAGE_SIZE_BYTES = 1024 * 1024

WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 20


def run_worker(rank: int, world_size: int) -> None:
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

        summary = benchmark_all_reduce(
            rank=rank,
            world_size=world_size,
            config=config,
        )

        if rank == 0:
            if summary is None:
                raise RuntimeError("Rank 0 did not receive benchmark results.")

            print()
            print("Results")
            print("-" * 40)
            print(f"Mean latency:   {summary.mean_ms:.3f} ms")
            print(f"Median latency: {summary.median_ms:.3f} ms")
            print(f"Min latency:    {summary.min_ms:.3f} ms")
            print(f"Max latency:    {summary.max_ms:.3f} ms")

    finally:
        dist.destroy_process_group()


def main() -> None:
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    message_size_mib = MESSAGE_SIZE_BYTES / (1024 * 1024)

    print("All-Reduce Latency Benchmark")
    print("-" * 40)
    print("Backend:             gloo")
    print(f"World size:          {WORLD_SIZE}")
    print(f"Message size:        {message_size_mib:.2f} MiB")
    print(f"Warmup iterations:   {WARMUP_ITERATIONS}")
    print(f"Measured iterations: {MEASURED_ITERATIONS}")

    mp.spawn(
        run_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
