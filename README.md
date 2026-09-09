# Distributed Collectives Benchmark

A small benchmarking project for exploring distributed collective
communication using PyTorch.

The project runs multiple local CPU processes with PyTorch Distributed and the
Gloo backend to measure the behavior of collective operations across different
message sizes.

The benchmark focuses on:

- `all_reduce`
- `all_gather`
- `broadcast`
- communication latency
- effective throughput
- message-size scaling

The experiments run locally on CPU and are intended to demonstrate distributed
communication concepts rather than GPU cluster performance.

A later comparison discusses how these results differ from NCCL-based
GPU communication.

## Development

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
