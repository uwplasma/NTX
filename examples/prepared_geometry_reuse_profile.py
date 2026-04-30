#!/usr/bin/env python3
"""Profile direct, prepared, and compiled monoenergetic scan paths."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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
            "figure.figsize": (12.0, 5.4),
            "figure.dpi": 220,
            "font.size": 10.0,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _max_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(raw) / (1024.0 * 1024.0)
    return float(raw) / 1024.0


def _block_tree(tree: Any) -> Any:
    import jax

    return jax.tree_util.tree_map(jax.block_until_ready, tree)


def _as_vector(result: Any) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    values = jnp.stack(
        [
            result.D11,
            result.D31,
            result.D13,
            result.D33,
            result.D33_spitzer,
        ]
    )
    return np.asarray(jax.device_get(values), dtype=float)


def _relative_mismatch(reference: np.ndarray, candidate: np.ndarray) -> float:
    scale = np.maximum(1.0, np.abs(reference))
    return float(np.max(np.abs(candidate - reference) / scale))


def _timed(label: str, func: Callable[[], Any]) -> tuple[str, Any, float]:
    start = time.perf_counter()
    value = _block_tree(func())
    return label, value, float(time.perf_counter() - start)


def _case_counts(preset: str) -> tuple[int, ...]:
    if preset == "smoke":
        return (3, 6)
    return (4, 16, 48)


def _grid_for_preset(preset: str):
    from ntx import GridSpec

    if preset == "smoke":
        return GridSpec(5, 5, 4)
    return GridSpec(9, 11, 6)


def _make_cases(count: int) -> tuple[Any, ...]:
    import jax.numpy as jnp

    from ntx import MonoenergeticCase

    nu_values = jnp.geomspace(1.0e-4, 1.0e-2, count)
    epsi_values = jnp.linspace(-4.0e-4, 4.0e-4, count)
    return tuple(
        MonoenergeticCase(nu_hat=float(nu), epsi_hat=float(epsi))
        for nu, epsi in zip(nu_values, epsi_values, strict=True)
    )


def _profile_case_count(surface: Any, grid: Any, count: int) -> dict[str, Any]:
    from ntx import (
        compile_prepared_solver,
        prepare_monoenergetic_system,
        solve_monoenergetic,
        solve_prepared,
    )

    cases = _make_cases(count)

    _, direct_results, direct_seconds = _timed(
        "direct",
        lambda: [solve_monoenergetic(surface, grid, case) for case in cases],
    )

    start = time.perf_counter()
    prepared = prepare_monoenergetic_system(surface, grid)
    _block_tree(prepared)
    prepare_seconds = float(time.perf_counter() - start)

    _, prepared_results, prepared_loop_seconds = _timed(
        "prepared",
        lambda: [solve_prepared(prepared, case) for case in cases],
    )
    prepared_total_seconds = prepare_seconds + prepared_loop_seconds

    compiled = compile_prepared_solver(prepared)
    _, compiled_first, compiled_first_seconds = _timed(
        "compiled_first",
        lambda: compiled(cases[0]),
    )
    _, compiled_results, compiled_steady_seconds = _timed(
        "compiled_steady",
        lambda: [compiled(case) for case in cases],
    )

    direct_vectors = [_as_vector(result) for result in direct_results]
    prepared_vectors = [_as_vector(result) for result in prepared_results]
    compiled_vectors = [_as_vector(result) for result in compiled_results]
    compiled_first_vector = _as_vector(compiled_first)

    max_prepared_mismatch = max(
        _relative_mismatch(reference, candidate)
        for reference, candidate in zip(direct_vectors, prepared_vectors, strict=True)
    )
    max_compiled_mismatch = max(
        _relative_mismatch(reference, candidate)
        for reference, candidate in zip(direct_vectors, compiled_vectors, strict=True)
    )
    first_compiled_mismatch = _relative_mismatch(direct_vectors[0], compiled_first_vector)

    return {
        "num_cases": count,
        "direct_seconds": direct_seconds,
        "prepare_seconds": prepare_seconds,
        "prepared_loop_seconds": prepared_loop_seconds,
        "prepared_total_seconds": prepared_total_seconds,
        "compiled_first_seconds": compiled_first_seconds,
        "compiled_steady_seconds": compiled_steady_seconds,
        "compiled_total_seconds": compiled_first_seconds + compiled_steady_seconds,
        "prepared_speedup_vs_direct": direct_seconds / prepared_total_seconds,
        "compiled_steady_speedup_vs_direct": direct_seconds / compiled_steady_seconds,
        "compiled_total_speedup_vs_direct": direct_seconds
        / (compiled_first_seconds + compiled_steady_seconds),
        "max_prepared_relative_mismatch": max_prepared_mismatch,
        "max_compiled_relative_mismatch": max_compiled_mismatch,
        "first_compiled_relative_mismatch": first_compiled_mismatch,
    }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "best_prepared_speedup_vs_direct": max(
            float(row["prepared_speedup_vs_direct"]) for row in results
        ),
        "best_compiled_steady_speedup_vs_direct": max(
            float(row["compiled_steady_speedup_vs_direct"]) for row in results
        ),
        "max_prepared_relative_mismatch": max(
            float(row["max_prepared_relative_mismatch"]) for row in results
        ),
        "max_compiled_relative_mismatch": max(
            float(row["max_compiled_relative_mismatch"]) for row in results
        ),
        "max_compiled_first_call_seconds": max(
            float(row["compiled_first_seconds"]) for row in results
        ),
        "max_rss_mb": _max_rss_mb(),
    }


def _warm_up_solver(surface: Any, grid: Any) -> float:
    from ntx import MonoenergeticCase, solve_monoenergetic

    case = MonoenergeticCase(nu_hat=3.0e-4, epsi_hat=1.0e-4)
    start = time.perf_counter()
    _block_tree(solve_monoenergetic(surface, grid, case))
    return float(time.perf_counter() - start)


def _plot(payload: dict[str, Any], output_prefix: Path) -> None:
    _configure_style()
    results = payload["results"]
    cases = np.asarray([row["num_cases"] for row in results], dtype=float)
    direct = np.asarray([row["direct_seconds"] for row in results], dtype=float)
    prepared = np.asarray([row["prepared_total_seconds"] for row in results], dtype=float)
    compiled = np.asarray([row["compiled_steady_seconds"] for row in results], dtype=float)
    prepared_speedup = np.asarray(
        [row["prepared_speedup_vs_direct"] for row in results],
        dtype=float,
    )
    compiled_speedup = np.asarray(
        [row["compiled_steady_speedup_vs_direct"] for row in results],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].loglog(cases, direct, marker="o", lw=2.2, color="#111827", label="Direct")
    axes[0].loglog(
        cases,
        prepared,
        marker="s",
        lw=2.2,
        color="#0072B2",
        label="Prepared total",
    )
    axes[0].loglog(
        cases,
        compiled,
        marker="^",
        lw=2.2,
        color="#D55E00",
        label="Compiled steady",
    )
    grid = payload["grid"]
    axes[0].set_title(
        f"Prepared geometry reuse ({grid['n_theta']}x{grid['n_zeta']}x{grid['n_xi']})"
    )
    axes[0].set_xlabel("Repeated monoenergetic cases")
    axes[0].set_ylabel("Wall time [s]")
    axes[0].legend(loc="upper left")

    axes[1].loglog(
        cases,
        prepared_speedup,
        marker="s",
        lw=2.2,
        color="#0072B2",
        label="Prepared total",
    )
    axes[1].loglog(
        cases,
        compiled_speedup,
        marker="^",
        lw=2.2,
        color="#D55E00",
        label="Compiled steady",
    )
    axes[1].axhline(1.0, color="#111827", lw=1.1, ls="--")
    axes[1].set_title("Speedup against direct repeated solves")
    axes[1].set_xlabel("Repeated monoenergetic cases")
    axes[1].set_ylabel("Speedup")
    axes[1].legend(loc="upper left")
    axes[1].text(
        0.03,
        0.08,
        (
            "Agreement max rel. mismatch: "
            f"{payload['summary_metrics']['max_compiled_relative_mismatch']:.1e}"
        ),
        transform=axes[1].transAxes,
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    for label, ax in zip(("a", "b"), axes, strict=True):
        ax.text(
            -0.12,
            1.02,
            f"({label})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "paper"), default="paper")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--case-counts", type=str, default=None)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "prepared_geometry_reuse_profile",
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument(
        "--perfetto",
        action="store_true",
        help="Also emit perfetto_trace.json.gz when --trace-dir is set.",
    )
    parser.add_argument(
        "--device-memory-profile",
        type=Path,
        default=None,
        help="Optional pprof-format device-memory snapshot written after profiling.",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.backend is not None:
        os.environ.setdefault("JAX_PLATFORM_NAME", args.backend)

    import jax

    from ntx import example_surface

    if args.backend is not None and jax.default_backend() != args.backend:
        raise SystemExit(
            f"requested --backend={args.backend} but JAX initialized {jax.default_backend()}"
        )
    trace_context = (
        jax.profiler.trace(
            str(args.trace_dir),
            create_perfetto_trace=bool(args.perfetto),
        )
        if args.trace_dir is not None
        else nullcontext()
    )
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

    with trace_context:
        surface = example_surface()
        grid = _grid_for_preset(args.preset)
        warmup_seconds = _warm_up_solver(surface, grid)
        if args.case_counts is None:
            counts = _case_counts(args.preset)
        else:
            counts = tuple(
                int(item.strip()) for item in args.case_counts.split(",") if item.strip()
            )
        results = [_profile_case_count(surface, grid, count) for count in counts]
    if args.device_memory_profile is not None:
        args.device_memory_profile.parent.mkdir(parents=True, exist_ok=True)
        jax.profiler.save_device_memory_profile(str(args.device_memory_profile))

    payload = {
        "artifact": "prepared_geometry_reuse_profile",
        "claim_scope": (
            "Profiles repeated monoenergetic solves on one prepared geometry. "
            "The artifact supports performance guidance and does not change "
            "physics validation claims."
        ),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "preset": args.preset,
        "grid": {
            "n_theta": grid.n_theta,
            "n_zeta": grid.n_zeta,
            "n_xi": grid.n_xi,
            "x64": grid.x64,
        },
        "warmup_seconds": warmup_seconds,
        "results": results,
        "summary_metrics": _summary(results),
        "notes": (
            "Direct solves rebuild geometry and derivative operators for every case; "
            "prepared solves hoist those arrays; compiled steady timings reuse the same "
            "JAX trace after one first-call compilation."
        ),
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(payload, args.output_prefix)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
