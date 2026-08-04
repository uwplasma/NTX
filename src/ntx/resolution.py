"""Geometry sampling reports for Fourier-collocation solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec

RECOMMENDED_ANGULAR_OVERSAMPLING = 2.25


@dataclass(frozen=True)
class GeometryResolutionReport:
    """Nyquist audit for the retained geometry harmonics.

    NTX evaluates phases as ``m * theta + n * nfp * zeta`` on one field
    period. The toroidal field-period factor therefore cancels from the sampled
    mode count: ``n_zeta`` resolves the retained integer ``n`` harmonics.
    """

    m_min: int
    m_max: int
    n_min: int
    n_max: int
    theta_nyquist_floor: int
    zeta_nyquist_floor: int
    n_theta: int
    n_zeta: int
    theta_oversampling: float
    zeta_oversampling: float
    status: Literal["resolved", "undersampled"]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def resolved(self) -> bool:
        """Whether the grid samples the retained modes without aliasing."""
        return self.status == "resolved"

    def require_resolved(self) -> None:
        """Raise before geometry assembly when retained modes would alias."""
        if self.errors:
            raise ValueError("geometry grid is undersampled: " + "; ".join(self.errors))


def geometry_resolution_report(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    *,
    warning_oversampling: float = RECOMMENDED_ANGULAR_OVERSAMPLING,
) -> GeometryResolutionReport:
    """Return retained harmonic extrema, Nyquist floors, and sampling status.

    Call this host-side before passing a surface as a dynamic argument to
    ``jax.jit``. Concrete surfaces are checked automatically during solver
    preparation; traced integer mode arrays cannot be converted into a host
    report without changing the transform contract.
    """
    if warning_oversampling < 1.0:
        raise ValueError("warning_oversampling must be at least 1")
    m = np.asarray(surface.m, dtype=int)
    n = np.asarray(surface.n, dtype=int)
    if m.size == 0 or n.size == 0:
        raise ValueError("geometry must retain at least one Fourier harmonic")

    m_min, m_max = int(m.min()), int(m.max())
    n_min, n_max = int(n.min()), int(n.max())
    # Nyquist: representing harmonics up to max|m| needs 2*max|m|+1 samples.
    # Below that the grid aliases high modes onto low ones, which looks like
    # a converged answer rather than an error.
    theta_floor = 2 * int(np.max(np.abs(m))) + 1
    zeta_floor = 2 * int(np.max(np.abs(n))) + 1
    theta_ratio = grid.n_theta / theta_floor
    zeta_ratio = grid.n_zeta / zeta_floor
    errors = []
    warnings = []
    if grid.n_theta < theta_floor:
        errors.append(
            f"n_theta={grid.n_theta} is below the retained-mode Nyquist floor "
            f"{theta_floor} for max|m|={theta_floor // 2}"
        )
    elif theta_ratio < warning_oversampling:
        warnings.append(
            f"theta oversampling is only {theta_ratio:.3g}; angular convergence is required"
        )
    if grid.n_zeta < zeta_floor:
        errors.append(
            f"n_zeta={grid.n_zeta} is below the retained-mode Nyquist floor "
            f"{zeta_floor} for max|n|={zeta_floor // 2}"
        )
    elif zeta_ratio < warning_oversampling:
        warnings.append(
            f"zeta oversampling is only {zeta_ratio:.3g}; angular convergence is required"
        )
    return GeometryResolutionReport(
        m_min=m_min,
        m_max=m_max,
        n_min=n_min,
        n_max=n_max,
        theta_nyquist_floor=theta_floor,
        zeta_nyquist_floor=zeta_floor,
        n_theta=grid.n_theta,
        n_zeta=grid.n_zeta,
        theta_oversampling=theta_ratio,
        zeta_oversampling=zeta_ratio,
        status="undersampled" if errors else "resolved",
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


__all__ = [
    "GeometryResolutionReport",
    "RECOMMENDED_ANGULAR_OVERSAMPLING",
    "geometry_resolution_report",
]
