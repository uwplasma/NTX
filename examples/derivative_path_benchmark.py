#!/usr/bin/env python3
"""Benchmark direct reverse-mode and prepared custom-VJP derivative paths."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    MonoenergeticCase,
    example_surface,
    prepare_monoenergetic_system,
    solve_prepared_coefficient_vector,
    solve_prepared_coefficient_vector_vjp,
)
from ntx.config import enable_x64  # noqa: E402

GRID = GridSpec(7, 9, 6)
NU_HAT = 3.0e-4
SCAN_SIZES = (1, 2, 4, 8, 16, 32)
ER_MIN = 1.0e-6
ER_MAX = 3.0e-3
REPEATS = 3
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "derivative_path_benchmark"


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (11.6, 6.2),
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
            "savefig.pad_inches": 0.05,
        }
    )


def _time_callable(fn, argument) -> float:
    start = time.perf_counter()
    value = fn(argument)
    jax.block_until_ready(value)
    return time.perf_counter() - start


def main(output_prefix: Path = OUTPUT_PREFIX) -> None:
    enable_x64(True)
    _configure_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    surface = example_surface(dtype=GRID.jax_dtype)
    prepared = prepare_monoenergetic_system(surface, GRID)

    def direct_scalar(er_hat):
        return solve_prepared_coefficient_vector(
            prepared,
            MonoenergeticCase(nu_hat=NU_HAT, er_hat=er_hat),
        )[3]

    def custom_scalar(er_hat):
        return solve_prepared_coefficient_vector_vjp(
            prepared,
            MonoenergeticCase(nu_hat=NU_HAT, er_hat=er_hat),
        )[3]

    direct_grad = jax.jit(jax.vmap(jax.grad(direct_scalar)))
    custom_grad = jax.jit(jax.vmap(jax.grad(custom_scalar)))

    counts = []
    direct_times = []
    custom_times = []
    max_relative_mismatch = []

    for count in SCAN_SIZES:
        er_hat_scan = jnp.geomspace(ER_MIN, ER_MAX, count)

        _ = direct_grad(er_hat_scan)
        _ = custom_grad(er_hat_scan)

        direct_measurements = [_time_callable(direct_grad, er_hat_scan) for _ in range(REPEATS)]
        custom_measurements = [_time_callable(custom_grad, er_hat_scan) for _ in range(REPEATS)]

        direct_values = np.asarray(direct_grad(er_hat_scan))
        custom_values = np.asarray(custom_grad(er_hat_scan))
        mismatch = np.max(
            np.abs(direct_values - custom_values)
            / np.maximum(np.abs(direct_values), 1.0e-30)
        )

        counts.append(count)
        direct_times.append(min(direct_measurements))
        custom_times.append(min(custom_measurements))
        max_relative_mismatch.append(mismatch)

    counts_array = np.asarray(counts, dtype=float)
    direct_times_array = np.asarray(direct_times)
    custom_times_array = np.asarray(custom_times)
    mismatch_array = np.asarray(max_relative_mismatch)
    speedup_array = direct_times_array / np.maximum(custom_times_array, 1.0e-30)

    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    axes[0].loglog(
        counts_array,
        direct_times_array,
        color="#0072B2",
        lw=2.3,
        marker="o",
        ms=5,
        label="Direct reverse-mode",
    )
    axes[0].loglog(
        counts_array,
        custom_times_array,
        color="#D55E00",
        lw=2.3,
        marker="s",
        ms=5,
        label="Prepared custom VJP",
    )
    axes[0].set_xlabel("Scan size")
    axes[0].set_ylabel("Best-of-3 wall time [s]")
    axes[0].set_title(r"$\partial D_{33}/\partial \hat E_r$ on a prepared surface")
    axes[0].legend(loc="upper left")
    axes[0].text(
        0.03,
        0.04,
        (
            rf"$N_\theta={GRID.n_theta}$, $N_\zeta={GRID.n_zeta}$, $N_\xi={GRID.n_xi}$" "\n"
            rf"$\hat\nu={NU_HAT:.1e}$, $\hat E_r \in [{ER_MIN:.1e}, {ER_MAX:.1e}]$"
        ),
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )

    axes[1].semilogx(
        counts_array,
        speedup_array,
        color="#009E73",
        lw=2.3,
        marker="D",
        ms=5,
        label="Speedup",
    )
    axes[1].semilogx(
        counts_array,
        mismatch_array,
        color="#111827",
        lw=1.8,
        ls="--",
        marker="^",
        ms=4.5,
        label="Max relative mismatch",
    )
    axes[1].axhline(1.0, color="#9ca3af", lw=1.2, ls=":")
    axes[1].axhline(1.0e-8, color="#9ca3af", lw=1.2, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Scan size")
    axes[1].set_ylabel("Speedup / mismatch")
    axes[1].set_title("Derivative-path efficiency and agreement")
    axes[1].legend(loc="best")

    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    summary = {
        "grid": {
            "n_theta": GRID.n_theta,
            "n_zeta": GRID.n_zeta,
            "n_xi": GRID.n_xi,
        },
        "nu_hat": NU_HAT,
        "er_min": ER_MIN,
        "er_max": ER_MAX,
        "scan_sizes": counts,
        "direct_times_seconds": direct_times,
        "prepared_times_seconds": custom_times,
        "speedup_prepared_vs_direct": speedup_array.tolist(),
        "max_relative_mismatch": max_relative_mismatch,
        "figure_png": str(output_prefix.with_suffix(".png")),
        "figure_pdf": str(output_prefix.with_suffix(".pdf")),
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plt.close(fig)

    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
