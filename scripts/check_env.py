import platform
import sys

import torch
import torch.distributed as dist


def main() -> None:
    print("Distributed Collectives Benchmark")
    print("-" * 40)

    print(f"Python:              {sys.version.split()[0]}")
    print(f"PyTorch:             {torch.__version__}")
    print(f"Platform:            {platform.system()} {platform.machine()}")

    distributed_available = dist.is_available()
    gloo_available = distributed_available and dist.is_gloo_available()

    print(f"Distributed support: {distributed_available}")
    print(f"Gloo available:      {gloo_available}")

    if not distributed_available:
        raise RuntimeError(
            "torch.distributed is not available in this PyTorch installation."
        )

    if not gloo_available:
        raise RuntimeError(
            "The Gloo backend is not available in this PyTorch installation."
        )

    print()
    print("Environment ready for local Gloo benchmarks.")


if __name__ == "__main__":
    main()
