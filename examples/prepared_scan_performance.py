#!/usr/bin/env python3
"""Plot synchronized CPU/GPU prepared-scan runtime and executable memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-json", type=Path, required=True)
    parser.add_argument("--gpu-json", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)

    cpu = json.loads(args.cpu_json.read_text(encoding="utf-8"))
    gpu = json.loads(args.gpu_json.read_text(encoding="utf-8"))
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (11.5, 7.5),
            "figure.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    _runtime_panel(axes[0, 0], cpu, "CPU warm execution")
    _runtime_panel(axes[0, 1], gpu, "GPU warm execution")
    _memory_panel(axes[1, 0], cpu, "CPU executable memory")
    _memory_panel(axes[1, 1], gpu, "GPU executable memory")
    for label, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
        ax.text(-0.13, 1.02, f"({label})", transform=ax.transAxes, fontweight="bold")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    return 0


def _runtime_panel(ax, payload: dict, title: str) -> None:
    sizes = np.asarray([entry["num_cases"] for entry in payload["results"]])
    for label in payload["warmup"]:
        seconds = [entry["modes"][label]["seconds_min"] for entry in payload["results"]]
        ax.loglog(sizes, seconds, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Scan size")
    ax.set_ylabel("Wall time [s]")
    ax.legend()


def _memory_panel(ax, payload: dict, title: str) -> None:
    labels = list(payload["warmup"])
    memory = [payload["warmup"][label]["temporary_size_bytes"] / 2**20 for label in labels]
    ax.bar(labels, memory, color=("#0072B2", "#009E73", "#D55E00")[: len(labels)])
    ax.set_title(title)
    ax.set_ylabel("Temporary memory [MiB]")
    ax.tick_params(axis="x", rotation=20)


if __name__ == "__main__":
    raise SystemExit(main())
