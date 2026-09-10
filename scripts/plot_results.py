import csv
from pathlib import Path

import matplotlib.pyplot as plt

from collective_bench.benchmark import (
    CollectiveOperation,
)
from collective_bench.metrics import (
    normalized_bandwidth_gbps,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmark_results.csv"
)

PLOTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "plots"
)


def load_results(
    results_path: Path,
) -> list[dict[str, str]]:
    if not results_path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {results_path}"
        )

    with results_path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def group_results(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = {}

    for row in rows:
        operation = row["operation"]

        grouped.setdefault(
            operation,
            [],
        )

        grouped[operation].append(
            {
                "message_size_bytes": float(
                    row["message_size_bytes"]
                ),
                "median_ms": float(
                    row["median_ms"]
                ),
                "world_size": float(
                    row["world_size"]
                ),
            }
        )

    for values in grouped.values():
        values.sort(
            key=lambda item: item[
                "message_size_bytes"
            ]
        )

    return grouped


def create_latency_plot(
    grouped: dict[str, list[dict[str, float]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for operation, values in grouped.items():
        sizes = [
            value["message_size_bytes"]
            for value in values
        ]

        latencies = [
            value["median_ms"]
            for value in values
        ]

        ax.plot(
            sizes,
            latencies,
            marker="o",
            label=operation,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel("Median Latency (ms)")

    ax.set_title(
        "Gloo Collective Latency vs Message Size"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def create_bandwidth_plot(
    grouped: dict[str, list[dict[str, float]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for operation_name, values in grouped.items():
        operation = CollectiveOperation(
            operation_name
        )

        sizes: list[float] = []
        bandwidths: list[float] = []

        for value in values:
            message_size_bytes = int(
                value["message_size_bytes"]
            )

            world_size = int(
                value["world_size"]
            )

            latency_ms = value["median_ms"]

            bandwidth = normalized_bandwidth_gbps(
                operation=operation,
                message_size_bytes=message_size_bytes,
                world_size=world_size,
                latency_ms=latency_ms,
            )

            sizes.append(
                message_size_bytes
            )

            bandwidths.append(
                bandwidth
            )

        ax.plot(
            sizes,
            bandwidths,
            marker="o",
            label=operation.value,
        )

    ax.set_xscale("log", base=2)

    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel(
        "Normalized Bandwidth (GB/s)"
    )

    ax.set_title(
        "Gloo Collective Bandwidth vs Message Size"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def main() -> None:
    rows = load_results(
        RESULTS_PATH
    )

    grouped = group_results(
        rows
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    latency_path = (
        PLOTS_DIR
        / "latency_vs_message_size.png"
    )

    bandwidth_path = (
        PLOTS_DIR
        / "bandwidth_vs_message_size.png"
    )

    create_latency_plot(
        grouped=grouped,
        output_path=latency_path,
    )

    create_bandwidth_plot(
        grouped=grouped,
        output_path=bandwidth_path,
    )

    print(
        f"Created latency plot: {latency_path}"
    )

    print(
        f"Created bandwidth plot: {bandwidth_path}"
    )


if __name__ == "__main__":
    main()
