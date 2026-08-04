"""Plotting helpers for NTX file-backed run outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._inputfiles import load_run_output


def plot_run_output(
    path: str | Path,
    *,
    output_prefix: str | Path | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> tuple[Path, ...]:
    """Plot an NTX `.nc`, `.npz`, or `.h5` run-output payload.

    The figure is intentionally compact: it shows the magnetic-field strength,
    radial-drift source, solved transport coefficients, and the core run
    diagnostics needed to triage a single monoenergetic calculation.
    """

    import matplotlib.pyplot as plt

    _configure_style(plt)
    output_path = Path(path).expanduser().resolve()
    data = load_run_output(output_path)
    prefix = (
        output_path.with_suffix("")
        if output_prefix is None
        else Path(output_prefix).expanduser().resolve()
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)

    fig = _run_output_figure(plt, data, output_path)
    written = []
    for fmt in formats:
        figure_path = prefix.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(figure_path)
        written.append(figure_path)
    plt.close(fig)
    return tuple(written)


def _configure_style(plt) -> None:
    """Apply the plot style, from a reset baseline.

    Starts from `default` so a caller's own rcParams cannot leak in and change
    what the figure looks like.
    """
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


def _run_output_figure(plt, data: dict[str, np.ndarray], output_path: Path):
    """Draw the field and geometry panels for a run output."""
    theta = np.asarray(data["theta_grid"])
    zeta = np.asarray(data["zeta_grid"])
    b = np.asarray(data["b"])
    drift = np.asarray(data["radial_drift_spatial"])

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    mesh_theta, mesh_zeta = np.meshgrid(theta, zeta, indexing="ij")

    im0 = axes[0, 0].pcolormesh(mesh_theta, mesh_zeta, b, shading="auto", cmap="viridis")
    axes[0, 0].set_xlabel(r"$\theta$")
    axes[0, 0].set_ylabel(r"$\zeta$")
    axes[0, 0].set_title("Magnetic-field strength")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.84, label=r"$B$")

    im1 = axes[0, 1].pcolormesh(mesh_theta, mesh_zeta, drift, shading="auto", cmap="coolwarm")
    axes[0, 1].set_xlabel(r"$\theta$")
    axes[0, 1].set_ylabel(r"$\zeta$")
    axes[0, 1].set_title("Radial-drift source")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.84, label=r"$\omega_d$")

    coeff_labels = [
        r"$D_{11}$",
        r"$D_{31}$",
        r"$D_{13}$",
        r"$D_{33}$",
        r"$D_{33}^{\mathrm{Sp}}$",
    ]
    coeff_values = [
        _scalar(data["D11"]),
        _scalar(data["D31"]),
        _scalar(data["D13"]),
        _scalar(data["D33"]),
        _scalar(data["D33_spitzer"]),
    ]
    axes[1, 0].bar(
        coeff_labels,
        coeff_values,
        color=["#0072B2", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"],
    )
    axes[1, 0].set_ylabel("Coefficient value")
    axes[1, 0].set_title("Solved transport coefficients")
    axes[1, 0].tick_params(axis="x", rotation=20)

    summary = (
        rf"$\hat{{\nu}}={_scalar(data['nu_hat']):.2e}$"
        "\n"
        rf"$\hat{{\epsilon}}={_scalar(data['epsi_hat_resolved']):.2e}$"
        "\n"
        rf"residual$={_scalar(data['residual_l2']):.2e}$"
        "\n"
        rf"Onsager$={_scalar(data['onsager_residual']):.2e}$"
        "\n"
        rf"$N_\theta={int(_scalar(data['n_theta']))},\;"
        rf"N_\zeta={int(_scalar(data['n_zeta']))},\;"
        rf"N_\xi={int(_scalar(data['n_xi']))}$"
        "\n"
        f"{output_path.name}"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        summary,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d1d5db", "alpha": 0.96},
    )
    axes[1, 1].set_title("Run summary")

    for label_text, ax in zip(("a", "b", "c", "d"), axes.ravel(), strict=True):
        ax.text(
            -0.14,
            1.02,
            f"({label_text})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
    return fig


def _scalar(value) -> float:
    """Coerce a zero-dimensional array to a float."""
    return float(np.asarray(value))


__all__ = ["plot_run_output"]
