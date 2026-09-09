# Distributed Collectives Benchmark

A small benchmarking project for exploring distributed collective communication using PyTorch.

The project runs multiple local CPU processes with PyTorch Distributed and the Gloo backend to measure the behavior of collective operations across different message sizes.

The benchmark focuses on:

* `all_reduce`
* `all_gather`
* `broadcast`
* communication latency
* effective throughput
* message-size scaling

The experiments run locally on CPU and are intended to demonstrate distributed communication concepts rather than GPU cluster performance.

A later comparison discusses how these results differ from NCCL-based GPU communication.

## Development

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify that PyTorch Distributed and the Gloo backend are available:

```bash
python scripts/check_env.py
```

Run the test suite:

```bash
python -m pytest
```

Run lint checks:

```bash
ruff check .
```

## Multi-Process All-Reduce Demo

The project includes a minimal distributed example that launches four local Python processes using `torch.multiprocessing.spawn`.

Each process joins the same PyTorch Distributed process group using the Gloo backend.

Run the demo with:

```bash
python scripts/run_all_reduce.py
```

Each worker creates a tensor containing a rank-dependent value:

* rank 0: `1`
* rank 1: `2`
* rank 2: `3`
* rank 3: `4`

The workers then execute an `all_reduce` operation using `SUM`.

Conceptually:

```text
1 + 2 + 3 + 4 = 10
```

After the collective completes, every rank contains the resulting value:

```text
rank 0 -> 10
rank 1 -> 10
rank 2 -> 10
rank 3 -> 10
```

This demonstrates the core distributed communication flow used by the later benchmark harness:

```text
spawn processes
      |
      v
initialize process group
      |
      v
create local tensors
      |
      v
execute collective operation
      |
      v
receive collective result
      |
      v
destroy process group
```

The order of output from individual ranks may vary because the workers execute concurrently in separate processes.

## Current Scope

The project currently supports:

* local multi-process execution
* PyTorch Distributed initialization
* Gloo CPU communication
* four-worker execution
* `all_reduce` with `SUM`
* validation that every rank receives the expected result

Upcoming work will extend this foundation with:

* collective-operation timing
* warm-up iterations
* repeated benchmark iterations
* message-size sweeps
* `all_gather`
* `broadcast`
* throughput calculations
* benchmark result export
* latency and bandwidth visualizations
* comparison of Gloo CPU communication with NCCL GPU communication

## Project Structure

```text
distributed-collectives-benchmark/
├── README.md
├── pyproject.toml
├── results/
├── scripts/
│   ├── check_env.py
│   └── run_all_reduce.py
├── src/
│   └── collective_bench/
│       ├── __init__.py
│       └── all_reduce_demo.py
└── tests/
    ├── test_import.py
    └── test_all_reduce_demo.py
```

## Goal

The goal of this project is not to simulate GPU-cluster performance.

Instead, it provides a small and reproducible environment for understanding the mechanics and performance characteristics of distributed collective communication before comparing those concepts with GPU-oriented systems such as NCCL.
