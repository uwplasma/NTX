"""Measured angular-oversampling audits for variable-coefficient solves."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .._solver import MonoenergeticCase, solve_prepared
from ..config import enable_x64
from ..geometry import BoozerSurface, VmecSurface
from ..grids import GridSpec
from ..resolution import (
    RECOMMENDED_ANGULAR_OVERSAMPLING,
    geometry_resolution_report,
)
from ..solver import prepare_monoenergetic_system

DEFAULT_OVERSAMPLING_RATIOS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
RECOMMENDED_OVERSAMPLING = RECOMMENDED_ANGULAR_OVERSAMPLING


@dataclass(frozen=True)
class AngularOversamplingPoint:
    """One angular grid, coefficient error, and compiled-cost measurement."""

    requested_ratio: float
    n_theta: int
    n_zeta: int
    theta_oversampling: float
    zeta_oversampling: float
    coefficients: tuple[float, float, float]
    relative_errors: tuple[float, float, float]
    max_relative_error: float
    schur_residual_l2: float
    prepare_seconds: float
    lowering_seconds: float
    compilation_seconds: float
    first_execution_seconds: float
    warm_execution_seconds: float
    temporary_size_bytes: int | None


@dataclass(frozen=True)
class AngularOversamplingAudit:
    """Oversampling ladder relative to its finest angular reference grid."""

    theta_nyquist_floor: int
    zeta_nyquist_floor: int
    n_xi: int
    coefficient_atol: float
    recommended_oversampling: float
    points: tuple[AngularOversamplingPoint, ...]

    @property
    def recommended_point(self) -> AngularOversamplingPoint:
        """Return the first measured point at or above the recommendation."""

        return next(
            point for point in self.points if point.requested_ratio >= self.recommended_oversampling
        )


@dataclass(frozen=True)
class _RawPoint:
    """One measured oversampling point: the grid used and what it cost."""
    requested_ratio: float
    n_theta: int
    n_zeta: int
    theta_oversampling: float
    zeta_oversampling: float
    coefficients: tuple[float, float, float]
    schur_residual_l2: float
    prepare_seconds: float
    lowering_seconds: float
    compilation_seconds: float
    first_execution_seconds: float
    warm_execution_seconds: float
    temporary_size_bytes: int | None


def _odd_ceiling(value: float) -> int:
    """Round up to the next odd integer.

    Odd angular resolutions keep the grid symmetric about theta = 0, which
    avoids a half-cell offset between the positive and negative branches.
    """
    integer = int(np.ceil(value))
    return integer if integer % 2 else integer + 1


def _memory_stat(memory: object, name: str) -> int | None:
    """Read one device memory field, tolerating its absence."""
    value = getattr(memory, name, None)
    return None if value is None else int(value)


def _block_result(result) -> None:
    """Wait for a result to be materialized.

    JAX is asynchronous, so a timing that does not block measures dispatch
    rather than the solve.
    """
    jax.block_until_ready(result.D11)


def _profile_grid(
    surface: BoozerSurface | VmecSurface,
    case: MonoenergeticCase,
    grid: GridSpec,
    *,
    repeats: int,
) -> _RawPoint:
    """Time one grid resolution end to end."""
    started = time.perf_counter()
    prepared = prepare_monoenergetic_system(
        surface,
        grid,
        require_resolved_geometry=True,
    )
    prepare_seconds = time.perf_counter() - started
    epsi_hat = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
    kernel = jax.jit(
        lambda nu, epsi: solve_prepared(
            prepared,
            MonoenergeticCase(nu_hat=nu, epsi_hat=epsi),
        )
    )
    nu = jnp.asarray(case.nu_hat, dtype=grid.jax_dtype)
    epsi = jnp.asarray(epsi_hat, dtype=grid.jax_dtype)

    started = time.perf_counter()
    lowered = kernel.lower(nu, epsi)
    lowering_seconds = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compilation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = executable(nu, epsi)
    _block_result(result)
    first_execution_seconds = time.perf_counter() - started
    warm_timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = executable(nu, epsi)
        _block_result(result)
        warm_timings.append(time.perf_counter() - started)
    memory = executable.memory_analysis()
    report = geometry_resolution_report(surface, grid)
    return _RawPoint(
        requested_ratio=0.0,
        n_theta=grid.n_theta,
        n_zeta=grid.n_zeta,
        theta_oversampling=report.theta_oversampling,
        zeta_oversampling=report.zeta_oversampling,
        coefficients=(float(result.D11), float(result.D31), float(result.D33)),
        schur_residual_l2=float(result.schur_residual_l2),
        prepare_seconds=prepare_seconds,
        lowering_seconds=lowering_seconds,
        compilation_seconds=compilation_seconds,
        first_execution_seconds=first_execution_seconds,
        warm_execution_seconds=min(warm_timings),
        temporary_size_bytes=_memory_stat(memory, "temp_size_in_bytes"),
    )


def audit_angular_oversampling(
    surface: BoozerSurface | VmecSurface,
    case: MonoenergeticCase,
    *,
    ratios: tuple[float, ...] = DEFAULT_OVERSAMPLING_RATIOS,
    n_xi: int = 16,
    coefficient_atol: float = 1.0e-12,
    recommended_oversampling: float = RECOMMENDED_OVERSAMPLING,
    repeats: int = 2,
) -> AngularOversamplingAudit:
    """Measure angular-grid error and compiled cost against the finest grid.

    The ratio multiplies the odd Nyquist floor independently in each angle.
    The final ratio is the numerical reference, not an analytical solution.
    Research promotion still requires successive-grid convergence and an
    independent-code comparison where one is available.
    """

    if len(ratios) < 2 or any(ratio < 1.0 for ratio in ratios):
        raise ValueError("ratios must contain at least two values no smaller than 1")
    if any(right <= left for left, right in zip(ratios, ratios[1:], strict=False)):
        raise ValueError("ratios must be strictly increasing")
    if not ratios[-1] > recommended_oversampling:
        raise ValueError("the reference ratio must exceed recommended_oversampling")
    if coefficient_atol <= 0.0:
        raise ValueError("coefficient_atol must be positive")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    enable_x64(True)
    m_max = int(np.max(np.abs(np.asarray(surface.m))))
    n_max = int(np.max(np.abs(np.asarray(surface.n))))
    theta_floor = 2 * m_max + 1
    zeta_floor = 2 * n_max + 1
    raw_points = []
    for ratio in ratios:
        grid = GridSpec(
            _odd_ceiling(ratio * theta_floor),
            _odd_ceiling(ratio * zeta_floor),
            n_xi,
        )
        raw = _profile_grid(surface, case, grid, repeats=repeats)
        raw_points.append(replace(raw, requested_ratio=ratio))

    reference = np.asarray(raw_points[-1].coefficients)
    points = []
    for raw in raw_points:
        values = np.asarray(raw.coefficients)
        relative = np.abs(values - reference) / np.maximum(np.abs(reference), coefficient_atol)
        relative_errors = (
            float(relative[0]),
            float(relative[1]),
            float(relative[2]),
        )
        points.append(
            AngularOversamplingPoint(
                requested_ratio=raw.requested_ratio,
                n_theta=raw.n_theta,
                n_zeta=raw.n_zeta,
                theta_oversampling=raw.theta_oversampling,
                zeta_oversampling=raw.zeta_oversampling,
                coefficients=raw.coefficients,
                relative_errors=relative_errors,
                max_relative_error=float(np.max(relative)),
                schur_residual_l2=raw.schur_residual_l2,
                prepare_seconds=raw.prepare_seconds,
                lowering_seconds=raw.lowering_seconds,
                compilation_seconds=raw.compilation_seconds,
                first_execution_seconds=raw.first_execution_seconds,
                warm_execution_seconds=raw.warm_execution_seconds,
                temporary_size_bytes=raw.temporary_size_bytes,
            )
        )
    return AngularOversamplingAudit(
        theta_nyquist_floor=theta_floor,
        zeta_nyquist_floor=zeta_floor,
        n_xi=n_xi,
        coefficient_atol=coefficient_atol,
        recommended_oversampling=recommended_oversampling,
        points=tuple(points),
    )


__all__ = [
    "AngularOversamplingAudit",
    "AngularOversamplingPoint",
    "DEFAULT_OVERSAMPLING_RATIOS",
    "RECOMMENDED_OVERSAMPLING",
    "audit_angular_oversampling",
]
