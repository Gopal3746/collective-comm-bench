import os

import torch.multiprocessing as mp

from collective_bench.all_reduce_demo import run_worker

WORLD_SIZE = 4
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"


def main() -> None:
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    print("Launching distributed all-reduce demo")
    print(f"World size: {WORLD_SIZE}")
    print("Backend: gloo")
    print()

    mp.spawn(
        run_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )

    print()
    print("All workers completed successfully.")


if __name__ == "__main__":
    main()
