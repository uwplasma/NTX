#!/usr/bin/env python3
"""Generate publication-style NTX strong-scaling figures from benchmark JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.0, 7.8),
            "figure.dpi": 220,
            "font.size": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-json", type=Path, required=True)
    parser.add_argument("--gpu-json", type=Path, required=True)
    parser.add_argument("--figure-title", type=str, default=None)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "performance_strong_scaling",
    )
    args = parser.parse_args(argv)

    _configure_style()
    cpu = json.loads(args.cpu_json.read_text(encoding="utf-8"))
    gpu = json.loads(args.gpu_json.read_text(encoding="utf-8"))
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    _plot_backend(axes[:, 0], cpu, "CPU")
    _plot_backend(axes[:, 1], gpu, "GPU")
    if args.figure_title is not None:
        fig.suptitle(args.figure_title, fontsize=13)

    for label, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
        ax.text(
            -0.14,
            1.02,
            f"({label})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    json_path = output_prefix.with_suffix(".json")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    json_path.write_text(
        json.dumps(_summary_payload(cpu, gpu), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {json_path}")
    return 0


def _plot_backend(axes, payload: dict, label: str) -> None:
    speed_ax, runtime_ax = axes
    multiprocess = payload.get("multiprocess", [])
    device_parallel = payload.get("device_parallel", [])

    if multiprocess:
        workers = np.asarray([entry["workers"] for entry in multiprocess], dtype=float)
        speed = np.asarray([entry["speedup_vs_serial"] for entry in multiprocess], dtype=float)
        seconds = np.asarray([entry["seconds"] for entry in multiprocess], dtype=float)
        speed_ax.plot(
            workers,
            speed,
            marker="s",
            lw=2.1,
            ls="--",
            color="#0072B2",
            label="Multiprocess",
        )
        runtime_ax.plot(
            workers,
            seconds,
            marker="s",
            lw=2.1,
            ls="--",
            color="#0072B2",
            label="Multiprocess",
        )
    if device_parallel:
        counts = np.asarray(
            [entry["effective_device_count"] for entry in device_parallel],
            dtype=float,
        )
        speed = np.asarray([entry["speedup_vs_serial"] for entry in device_parallel], dtype=float)
        seconds = np.asarray([entry["seconds"] for entry in device_parallel], dtype=float)
        speed_ax.plot(
            counts,
            speed,
            marker="^",
            lw=2.0,
            ls=":",
            color="#D55E00",
            label="Single-process device parallel",
        )
        runtime_ax.plot(
            counts,
            seconds,
            marker="^",
            lw=2.0,
            ls=":",
            color="#D55E00",
            label="Single-process device parallel",
        )

    serial_seconds = float(payload["serial"]["seconds"])
    speed_ax.axhline(1.0, color="#111827", lw=1.2, ls="--")
    runtime_ax.axhline(serial_seconds, color="#111827", lw=1.2, ls="--", label="Serial")

    grid = payload.get("grid", {})
    grid_label = f"{grid.get('n_theta', '?')}x{grid.get('n_zeta', '?')}x{grid.get('n_xi', '?')}"
    speed_ax.set_title(f"{label} speedup, {payload['num_cases']} cases ({grid_label})")
    speed_ax.set_xlabel("Workers or healthy devices")
    speed_ax.set_ylabel("Speedup vs serial")
    speed_ax.set_ylim(bottom=0.0)
    speed_ax.legend(loc="best")
    runtime_ax.set_title(f"{label} wall time")
    runtime_ax.set_xlabel("Workers or healthy devices")
    runtime_ax.set_ylabel("Wall time [s]")
    runtime_ax.set_yscale("log")
    runtime_ax.legend(loc="best")
    runtime_ax.text(
        0.97,
        0.05,
        f"RSS {payload.get('max_rss_mb', 0.0):.0f} MB",
        transform=runtime_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d1d5db", "alpha": 0.92},
    )


def _summary_payload(cpu: dict, gpu: dict) -> dict[str, object]:
    return {
        "artifact": "strong_scaling_summary",
        "claim_scope": (
            "Summarizes fixed-workload strong scaling for serial, "
            "single-process device-parallel, and multiprocess NTX scans."
        ),
        "cpu": _backend_summary(cpu),
        "gpu": _backend_summary(gpu),
    }


def _backend_summary(payload: dict) -> dict[str, object]:
    return {
        "backend": payload.get("backend"),
        "surface": payload.get("surface"),
        "grid": payload.get("grid", {}),
        "num_cases": int(payload.get("num_cases", 0)),
        "local_device_count": int(payload.get("local_device_count", 0)),
        "healthy_parallel_device_count": int(payload.get("healthy_parallel_device_count", 0)),
        "max_rss_mb": float(payload.get("max_rss_mb", 0.0)),
        "serial_seconds": float(payload["serial"]["seconds"]),
        "best_multiprocess_speedup_vs_serial": _max_entry_value(
            payload.get("multiprocess", []),
            "speedup_vs_serial",
        ),
        "best_device_parallel_speedup_vs_serial": _max_entry_value(
            payload.get("device_parallel", []),
            "speedup_vs_serial",
        ),
        "max_abs_delta_serial_vs_multiprocess_d11": _max_entry_value(
            payload.get("multiprocess", []),
            "max_abs_delta_serial_d11",
        ),
        "max_abs_delta_serial_vs_device_parallel_d11": _max_entry_value(
            payload.get("device_parallel", []),
            "max_abs_delta_serial_d11",
        ),
    }


def _max_entry_value(entries: list[dict], key: str) -> float | None:
    values = [float(entry[key]) for entry in entries if key in entry]
    if not values:
        return None
    return max(values)


if __name__ == "__main__":
    raise SystemExit(main())
