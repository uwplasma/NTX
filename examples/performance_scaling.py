#!/usr/bin/env python3
"""Generate publication-style CPU/GPU scaling figures from NTX benchmark JSON."""

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
            "figure.figsize": (12.0, 8.0),
            "figure.dpi": 220,
            "font.size": 10.5,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
        default=ROOT / "docs" / "_static" / "performance_scaling",
    )
    args = parser.parse_args(argv)

    _configure_style()
    cpu = json.loads(args.cpu_json.read_text(encoding="utf-8"))
    gpu = json.loads(args.gpu_json.read_text(encoding="utf-8"))
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    _plot_runtime_panel(axes[0, 0], cpu, "CPU Scaling")
    _plot_speedup_panel(axes[1, 0], cpu, "CPU Speedup")
    _plot_runtime_panel(axes[0, 1], gpu, "GPU Scaling")
    _plot_speedup_panel(axes[1, 1], gpu, "GPU Speedup")
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


def _plot_runtime_panel(ax, payload: dict, title: str) -> None:
    results = payload["results"]
    cases = np.asarray([entry["num_cases"] for entry in results], dtype=float)
    serial = np.asarray([entry["serial_seconds"] for entry in results], dtype=float)
    multiprocess = np.asarray([entry["multiprocess_seconds"] for entry in results], dtype=float)

    ax.loglog(cases, serial, marker="o", lw=2.2, color="#111827", label="Serial batched")
    ax.loglog(
        cases,
        multiprocess,
        marker="s",
        lw=2.2,
        ls="--",
        color="#0072B2",
        label="Multiprocess",
    )
    if "device_parallel_seconds" in results[0]:
        device_parallel = np.asarray(
            [entry["device_parallel_seconds"] for entry in results],
            dtype=float,
        )
        ax.loglog(
            cases,
            device_parallel,
            marker="^",
            lw=1.8,
            ls=":",
            color="#D55E00",
            label="Single-process device parallel",
        )
    grid = payload.get("grid", {})
    grid_label = f"{grid.get('n_theta', '?')}x{grid.get('n_zeta', '?')}x{grid.get('n_xi', '?')}"
    ax.set_title(f"{title} ({grid_label})")
    ax.set_xlabel("Scan size")
    ax.set_ylabel("Wall time [s]")
    ax.legend(loc="upper left")
    _annotate_crossover(ax, cases, serial, multiprocess)
    if "max_rss_mb" in payload:
        ax.text(
            0.97,
            0.05,
            f"max RSS {payload['max_rss_mb']:.0f} MB",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": "#d1d5db",
                "alpha": 0.92,
            },
        )


def _plot_speedup_panel(ax, payload: dict, title: str) -> None:
    results = payload["results"]
    cases = np.asarray([entry["num_cases"] for entry in results], dtype=float)
    multiprocess = np.asarray(
        [entry["multiprocess_speedup_vs_serial"] for entry in results],
        dtype=float,
    )
    ax.semilogx(cases, multiprocess, marker="s", lw=2.2, ls="--", color="#0072B2")
    if "device_parallel_speedup_vs_serial" in results[0]:
        device_parallel = np.asarray(
            [entry["device_parallel_speedup_vs_serial"] for entry in results],
            dtype=float,
        )
        ax.semilogx(cases, device_parallel, marker="^", lw=1.8, ls=":", color="#D55E00")
    ax.axhline(1.0, color="#111827", lw=1.2, ls="--")
    ax.fill_between(cases, 0.0, 1.0, color="#d1d5db", alpha=0.25)
    ymax = max(1.05, float(np.nanmax(multiprocess)) * 1.15)
    ax.fill_between(cases, 1.0, ymax, color="#bfdbfe", alpha=0.18)
    ax.set_ylim(0.0, ymax)
    ax.set_title(title)
    ax.set_xlabel("Scan size")
    ax.set_ylabel("Speedup vs serial")
    ax.text(
        0.03,
        0.08,
        "serial preferred",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#374151",
    )
    ax.text(
        0.62,
        0.88,
        "throughput lane",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#1d4ed8",
    )


def _annotate_crossover(
    ax,
    cases: np.ndarray,
    serial: np.ndarray,
    multiprocess: np.ndarray,
) -> None:
    faster = np.nonzero(multiprocess < serial)[0]
    if faster.size == 0:
        ax.text(
            0.03,
            0.08,
            "No crossover in tested range",
            transform=ax.transAxes,
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
        )
        return
    if faster.size == 1 and int(faster[0]) == 0 and np.all(multiprocess[1:] >= serial[1:]):
        ax.text(
            0.03,
            0.08,
            "Smallest-size point is startup dominated",
            transform=ax.transAxes,
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
        )
        return
    index = int(faster[0])
    ax.axvline(cases[index], color="#0072B2", lw=1.2, ls=":")
    ax.text(
        0.03,
        0.08,
        f"Multiprocess wins from {int(cases[index])} cases",
        transform=ax.transAxes,
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )


def _summary_payload(cpu: dict, gpu: dict) -> dict[str, object]:
    return {
        "artifact": "performance_scaling_summary",
        "claim_scope": (
            "Summarizes serial, single-process device-parallel, and "
            "multiprocess throughput timing, resident-memory, and correctness "
            "metadata from committed scaling benchmark JSON files."
        ),
        "cpu": _backend_summary(cpu),
        "gpu": _backend_summary(gpu),
    }


def _backend_summary(payload: dict) -> dict[str, object]:
    results = payload["results"]
    return {
        "backend": payload.get("backend"),
        "surface": payload.get("surface"),
        "grid": payload.get("grid", {}),
        "sizes": [int(entry["num_cases"]) for entry in results],
        "local_device_count": int(payload.get("local_device_count", 0)),
        "healthy_parallel_device_count": int(payload.get("healthy_parallel_device_count", 0)),
        "max_rss_mb": float(payload.get("max_rss_mb", 0.0)),
        "multiprocess_crossover_cases": _first_crossover_case(
            results,
            "multiprocess_seconds",
        ),
        "device_parallel_crossover_cases": _first_crossover_case(
            results,
            "device_parallel_seconds",
        ),
        "best_multiprocess_speedup_vs_serial": _max_result_value(
            results,
            "multiprocess_speedup_vs_serial",
        ),
        "best_device_parallel_speedup_vs_serial": _max_result_value(
            results,
            "device_parallel_speedup_vs_serial",
        ),
        "max_abs_delta_serial_vs_multiprocess_d11": _max_result_value(
            results,
            "max_abs_delta_serial_vs_multiprocess_d11",
        ),
        "max_abs_delta_serial_vs_device_parallel_d11": _max_result_value(
            results,
            "max_abs_delta_serial_vs_device_parallel_d11",
        ),
    }


def _first_crossover_case(results: list[dict], timing_key: str) -> int | None:
    for entry in results:
        if timing_key in entry and float(entry[timing_key]) < float(entry["serial_seconds"]):
            return int(entry["num_cases"])
    return None


def _max_result_value(results: list[dict], key: str) -> float | None:
    values = [float(entry[key]) for entry in results if key in entry]
    if not values:
        return None
    return max(values)


if __name__ == "__main__":
    raise SystemExit(main())
