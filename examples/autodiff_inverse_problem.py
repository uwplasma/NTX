#!/usr/bin/env python3
"""Autodiff inverse-problem example for NTX."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import GridSpec, example_inverse_problem  # noqa: E402


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 180,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def main() -> None:
    _configure_style()
    result = example_inverse_problem(grid=GridSpec(7, 9, 6))
    output_dir = ROOT / "docs" / "_static"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "autodiff_inverse_problem.png"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    iterations = np.arange(1, len(result.loss_history) + 1)
    axes[0].plot(iterations, np.asarray(result.amplitude_history), color="#1f77b4", lw=2)
    axes[0].axhline(float(result.target_amplitude), color="#d62728", ls="--", lw=1.5)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Inferred amplitude")
    axes[0].set_title("Inverse solve convergence")

    axes[1].semilogy(iterations, np.asarray(result.loss_history), color="#2ca02c", lw=2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Objective reduction")

    axes[2].loglog(
        np.asarray(result.nu_hat),
        np.asarray(result.target_response),
        lw=2,
        label="Target",
    )
    axes[2].loglog(
        np.asarray(result.nu_hat),
        np.asarray(result.initial_response),
        lw=1.5,
        ls=":",
        label="Initial guess",
    )
    axes[2].loglog(
        np.asarray(result.nu_hat),
        np.asarray(result.fitted_response),
        lw=2,
        ls="--",
        label="Recovered",
    )
    axes[2].set_xlabel(r"$\hat{\nu}$")
    axes[2].set_ylabel(r"$D_{11}$")
    axes[2].set_title("Transport response")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
