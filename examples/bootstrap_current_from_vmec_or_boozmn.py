#!/usr/bin/env python3
"""NTX-only reduced bootstrap-current response from VMEC or Boozer geometry.

Edit the configuration block below, run the script, and inspect the figure and
JSON summary written next to ``OUTPUT_PREFIX``.
"""
# ruff: noqa: I001

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    MonoenergeticCase,
    load_boozmn_surface,
    solve_monoenergetic,
)
from ntx._checkout_paths import find_neopax_root, fixture_path  # noqa: E402
from ntx.vmec_jax_vmec import surface_from_vmec_jax_vmec_wout_file  # noqa: E402


NEOPAX_ROOT = find_neopax_root()
DEFAULT_WOUT = (
    NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
    if NEOPAX_ROOT is not None
    else fixture_path("sample_wout.nc")
)
DEFAULT_BOOZMN = (
    NEOPAX_ROOT / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
    if NEOPAX_ROOT is not None
    else fixture_path("sample_boozmn.nc")
)

# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
SURFACE_SOURCE = "auto"  # "auto", "vmec", or "boozmn"
WOUT_PATH = DEFAULT_WOUT
BOOZMN_PATH = DEFAULT_BOOZMN
RHO_GRID = np.linspace(0.15, 0.72, 10)
NU_HAT = 1.0e-5
ER_HAT = 0.0
GRID = GridSpec(n_theta=25, n_zeta=25, n_xi=40)
OUTPUT_PREFIX = ROOT / "docs" / "_static" / "bootstrap_current_from_vmec_or_boozmn"


def select_surface_loader():
    if SURFACE_SOURCE not in {"auto", "vmec", "boozmn"}:
        raise ValueError(f"unsupported SURFACE_SOURCE={SURFACE_SOURCE!r}")
    if SURFACE_SOURCE in {"auto", "boozmn"} and BOOZMN_PATH.exists():
        return (
            lambda rho_value: load_boozmn_surface(BOOZMN_PATH, rho=float(rho_value)).surface,
            "boozmn",
        )
    if WOUT_PATH.exists():
        return (
            lambda rho_value: surface_from_vmec_jax_vmec_wout_file(
                WOUT_PATH,
                s=float(rho_value**2),
            ),
            "vmec_jax",
        )
    raise FileNotFoundError("no usable VMEC or Boozer input file was found")


def radial_profiles(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    density = 3.2e19 * (1.0 - rho**8) + 0.45e19
    temperature = 3.0e3 * (1.0 - rho**2) + 0.8e3
    density_prime = -3.2e19 * 8.0 * rho**7
    temperature_prime = -6.0e3 * rho
    dlnn_drho = density_prime / density
    dlnT_drho = temperature_prime / temperature
    return density, temperature, dlnn_drho, dlnT_drho


def solve_profiles() -> dict[str, np.ndarray | str | float]:
    load_surface, mode = select_surface_loader()
    d11 = np.zeros_like(RHO_GRID)
    d13 = np.zeros_like(RHO_GRID)
    d33_hat = np.zeros_like(RHO_GRID)
    b0 = np.zeros_like(RHO_GRID)
    iota = np.zeros_like(RHO_GRID)

    for idx, rho_value in enumerate(RHO_GRID):
        surface = load_surface(rho_value)
        result = solve_monoenergetic(
            surface,
            GRID,
            MonoenergeticCase(nu_hat=NU_HAT, er_hat=ER_HAT),
        )
        d11[idx] = float(result.D11)
        d13[idx] = float(result.D13)
        d33_hat[idx] = float(result.D33 * NU_HAT)
        b0[idx] = float(surface.b0)
        iota[idx] = float(surface.iota)

    density, temperature, dlnn_drho, dlnT_drho = radial_profiles(RHO_GRID)
    current_response = density * (-dlnn_drho * d13 - 0.75 * dlnT_drho * d33_hat)
    current_response /= np.max(np.abs(current_response))

    return {
        "mode": mode,
        "rho": RHO_GRID,
        "density": density,
        "temperature": temperature,
        "d11": d11,
        "d13": d13,
        "d33_hat": d33_hat,
        "b0": b0,
        "iota": iota,
        "current_response": current_response,
    }


def plot_profiles(data: dict[str, np.ndarray | str | float]) -> None:
    rho = np.asarray(data["rho"])
    density = np.asarray(data["density"])
    temperature = np.asarray(data["temperature"])
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), constrained_layout=True)

    axes[0, 0].plot(rho, np.asarray(data["b0"]), color="#1f77b4", lw=2.3)
    axes[0, 0].set_ylabel(r"$B_{00}$ [T]")
    axes[0, 0].set_title("Magnetic Geometry")
    ax_iota = axes[0, 0].twinx()
    ax_iota.plot(rho, np.abs(np.asarray(data["iota"])), color="#ff7f0e", lw=2.0, ls="--")
    ax_iota.set_ylabel(r"$|\iota|$")

    axes[0, 1].plot(
        rho,
        density / density.max(),
        color="#2ca02c",
        lw=2.3,
        label=r"$n / n(0)$",
    )
    axes[0, 1].plot(
        rho,
        temperature / temperature.max(),
        color="#d62728",
        lw=2.3,
        label=r"$T / T(0)$",
    )
    axes[0, 1].set_title("Radial profile inputs")
    axes[0, 1].set_ylabel("Normalized amplitude")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        rho,
        np.asarray(data["d33_hat"]),
        color="#9467bd",
        lw=2.4,
        label=rf"$\hat{{\nu}} D_{{33}}$ at $\hat{{\nu}}={NU_HAT:.0e}$",
    )
    axes[1, 0].plot(
        rho,
        np.asarray(data["d11"]),
        color="#1f77b4",
        lw=2.0,
        ls="--",
        label=r"$D_{11}$",
    )
    axes[1, 0].set_title("Parallel-Flow Drive")
    axes[1, 0].set_xlabel(r"$\rho$")
    axes[1, 0].set_ylabel("Scaled coefficient")
    axes[1, 0].legend(frameon=False, loc="best")

    axes[1, 1].plot(
        rho,
        np.asarray(data["current_response"]),
        color="#111111",
        lw=2.6,
    )
    axes[1, 1].fill_between(
        rho,
        0.0,
        np.asarray(data["current_response"]),
        color="#111111",
        alpha=0.12,
    )
    axes[1, 1].set_title("Reduced Current Response")
    axes[1, 1].set_xlabel(r"$\rho$")
    axes[1, 1].set_ylabel("Normalized profile")

    for ax in axes.flat:
        ax.set_xlim(rho[0], rho[-1])
        ax.grid(alpha=0.22, lw=0.6)

    fig.suptitle(
        f"NTX reduced current response from {data['mode']} geometry",
        fontsize=14,
        y=1.02,
    )
    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(data: dict[str, np.ndarray | str | float]) -> None:
    summary = {
        "surface_source": data["mode"],
        "wout": WOUT_PATH.name if WOUT_PATH.exists() else None,
        "boozmn": BOOZMN_PATH.name if BOOZMN_PATH.exists() else None,
        "nu_hat": NU_HAT,
        "er_hat": ER_HAT,
        "grid": {
            "n_theta": GRID.n_theta,
            "n_zeta": GRID.n_zeta,
            "n_xi": GRID.n_xi,
        },
        "rho": np.asarray(data["rho"]).tolist(),
        "D11": np.asarray(data["d11"]).tolist(),
        "D13": np.asarray(data["d13"]).tolist(),
        "nuD33": np.asarray(data["d33_hat"]).tolist(),
        "bootstrap_current_response": np.asarray(data["current_response"]).tolist(),
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
    }
    OUTPUT_PREFIX.with_suffix(".json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    data = solve_profiles()
    plot_profiles(data)
    write_summary(data)
    print(f"reduced current-response figure: {OUTPUT_PREFIX.with_suffix('.png')}")
    print(f"reduced current-response summary: {OUTPUT_PREFIX.with_suffix('.json')}")


if __name__ == "__main__":
    main()
