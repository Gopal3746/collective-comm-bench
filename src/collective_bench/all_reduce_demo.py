import torch
import torch.distributed as dist


def expected_rank_sum(world_size: int) -> float:
    """Return the expected sum of rank values 1..world_size."""
    return float(sum(range(1, world_size + 1)))


def run_worker(rank: int, world_size: int) -> None:
    """Run one distributed worker and perform an all-reduce."""

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    try:
        tensor = torch.tensor([float(rank + 1)])

        print(
            f"[rank {rank}] before all_reduce: "
            f"{tensor.item():.1f}",
            flush=True,
        )

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

        expected = expected_rank_sum(world_size)

        print(
            f"[rank {rank}] after all_reduce:  "
            f"{tensor.item():.1f}",
            flush=True,
        )

        if tensor.item() != expected:
            raise RuntimeError(
                f"Rank {rank} expected {expected}, "
                f"but received {tensor.item()}."
            )

    finally:
        dist.destroy_process_group()
