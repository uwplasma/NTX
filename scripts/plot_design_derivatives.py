#!/usr/bin/env python3
"""Render the design-gradient cost and accuracy figure used in the README.

Reads the record written by ``benchmarks/bench_design_derivatives.py`` so the
figure cannot drift from the measurement: if the numbers change, rerun the
benchmark and this redraws them.

    python scripts/plot_design_derivatives.py \\
        --record benchmarks/results/design_derivatives.json \\
        --output docs/_static/design_derivatives
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REVERSE = "#0072B2"
FINITE = "#D55E00"
GREY = "#666666"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "font.size": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "legend.frameon": False,
        }
    )


def render(record: dict, output: Path) -> None:
    rows = record["rows"]
    parameters = np.array([r["parameters"] for r in rows], float)
    reverse = np.array([r["reverse_seconds"] for r in rows]) * 1e3
    finite = np.array([r["finite_difference_seconds"] for r in rows]) * 1e3
    reverse_error = np.array([r["reverse_relative_error"] for r in rows])
    finite_error = np.array([r["finite_difference_relative_error"] for r in rows])

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    ax = axes[0]
    ax.loglog(parameters, finite, "o-", color=FINITE, ms=5, lw=1.6,
              label="finite differences")
    ax.loglog(parameters, reverse, "s-", color=REVERSE, ms=5, lw=1.6,
              label="NTX adjoint")
    reference = finite[0] * parameters / parameters[0]
    ax.loglog(parameters, reference, ":", color=GREY, lw=1.2, label="linear in $P$")
    ax.set_xlabel("design parameters $P$")
    ax.set_ylabel("wall time for the full gradient [ms]")
    ax.set_title("one adjoint solve, not $2P$ of them", fontsize=10, loc="left")
    ax.legend(loc="upper left")
    best = max(r["speedup"] for r in rows)
    ax.annotate(
        f"{best:.0f}$\\times$ at $P={int(parameters[np.argmax([r['speedup'] for r in rows])])}$",
        xy=(parameters[-1], reverse[-1]),
        xytext=(-6, 14),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=REVERSE,
    )

    ax = axes[1]
    ax.loglog(parameters, np.maximum(finite_error, 1e-18), "o-", color=FINITE, ms=5,
              lw=1.6, label="finite differences")
    ax.loglog(parameters, np.maximum(reverse_error, 1e-18), "s-", color=REVERSE, ms=5,
              lw=1.6, label="NTX adjoint")
    ax.set_xlabel("design parameters $P$")
    ax.set_ylabel("relative error vs forward mode")
    ax.set_title("and exact to rounding", fontsize=10, loc="left")
    ax.set_ylim(1e-16, 1e-7)
    ax.legend(loc="center left")

    grid = record["grid"]
    fig.text(
        0.5,
        -0.06,
        f"$D_{{11}}$ on a {grid['n_theta']}$\\times${grid['n_zeta']} surface grid, "
        f"$N_\\xi={grid['n_xi']}$, $\\hat\\nu={record['nu_hat']:g}$, float64 on "
        f"{record['provenance']['device_kind']}. "
        "Finite differences use central steps; forward mode arbitrates.",
        ha="center",
        fontsize=7.6,
        color=GREY,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {output.with_suffix('.png')} and {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(json.loads(Path(args.record).read_text(encoding="utf-8")), Path(args.output))


if __name__ == "__main__":
    main()
