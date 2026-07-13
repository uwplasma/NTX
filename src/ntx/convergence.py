"""Host-side convergence ladders for research-grade monoenergetic solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .resolution import geometry_resolution_report
from .solver import MonoenergeticCase, TransportResult, solve_monoenergetic


@dataclass(frozen=True)
class ConvergenceStep:
    """One evaluated grid and its change from the previous step in that phase."""

    phase: Literal["angular", "legendre"]
    grid: GridSpec
    coefficients: tuple[float, float, float, float, float]
    absolute_change: tuple[float, float, float] | None
    relative_change: tuple[float, float, float] | None
    accepted: bool | None


@dataclass(frozen=True)
class AdaptiveConvergenceResult:
    """Complete angular/Legendre audit and explicit promotion status."""

    status: Literal["converged", "unresolved", "model-out-of-scope"]
    steps: tuple[ConvergenceStep, ...]
    result: TransportResult | None
    angular_converged: bool
    legendre_converged: bool
    message: str


def _snapshot(result: TransportResult) -> tuple[float, float, float, float, float]:
    return tuple(
        float(value)
        for value in (result.D11, result.D31, result.D13, result.D33, result.D33_spitzer)
    )


def _changes(
    previous: tuple[float, ...],
    current: tuple[float, ...],
    *,
    rtol: float,
    atol: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], bool]:
    indices = np.asarray([0, 1, 3])
    old = np.asarray(previous)[indices]
    new = np.asarray(current)[indices]
    absolute = np.abs(new - old)
    scale = np.maximum(np.abs(old), np.abs(new))
    tolerance = atol + rtol * scale
    denominator = np.maximum(np.maximum(scale, atol), np.finfo(float).tiny)
    relative = absolute / denominator
    return (
        tuple(float(value) for value in absolute),
        tuple(float(value) for value in relative),
        bool(np.all(absolute <= tolerance)),
    )


def solve_monoenergetic_converged(
    surface: BoozerSurface | VmecSurface,
    case: MonoenergeticCase,
    *,
    angular_resolutions: tuple[tuple[int, int], ...],
    legendre_orders: tuple[int, ...],
    rtol: float = 1.0e-2,
    atol: float | tuple[float, float, float] = 1.0e-10,
    required_successive: int = 2,
    dtype: str = "float64",
    x64: bool = True,
) -> AdaptiveConvergenceResult:
    """Refine angular and Legendre orders independently until both stabilize.

    Angular grids are evaluated first at ``legendre_orders[0]``. The selected
    angular grid is then held fixed while ``N_xi`` is refined. A research-grade
    result requires ``required_successive`` accepted changes in each phase;
    the last grid is never silently promoted when a ladder is exhausted.
    """
    if rtol < 0.0:
        raise ValueError("rtol must be non-negative")
    if required_successive < 1:
        raise ValueError("required_successive must be positive")
    if len(angular_resolutions) < required_successive + 1:
        raise ValueError("angular_resolutions is too short for the successive gate")
    if len(legendre_orders) < required_successive + 1:
        raise ValueError("legendre_orders is too short for the successive gate")
    atol_values = np.broadcast_to(np.asarray(atol, dtype=float), (3,))
    if np.any(atol_values < 0.0):
        raise ValueError("atol must be non-negative")

    steps: list[ConvergenceStep] = []
    previous = None
    accepted_run = 0
    angular_converged = False
    final_result = None
    selected_angular = angular_resolutions[-1]
    for n_theta, n_zeta in angular_resolutions:
        grid = GridSpec(n_theta, n_zeta, legendre_orders[0], dtype=dtype, x64=x64)
        report = geometry_resolution_report(surface, grid)
        if not report.resolved:
            return AdaptiveConvergenceResult(
                "model-out-of-scope",
                tuple(steps),
                None,
                False,
                False,
                "angular ladder contains an undersampled geometry grid: "
                + "; ".join(report.errors),
            )
        final_result = solve_monoenergetic(surface, grid, case)
        current = _snapshot(final_result)
        if not np.all(np.isfinite(current)):
            return AdaptiveConvergenceResult(
                "model-out-of-scope",
                tuple(steps),
                None,
                False,
                False,
                "non-finite coefficient encountered during angular refinement",
            )
        change = (
            None
            if previous is None
            else _changes(previous, current, rtol=rtol, atol=atol_values)
        )
        accepted = None if change is None else change[2]
        steps.append(
            ConvergenceStep(
                "angular",
                grid,
                current,
                None if change is None else change[0],
                None if change is None else change[1],
                accepted,
            )
        )
        accepted_run = accepted_run + 1 if accepted else 0
        previous = current
        selected_angular = (n_theta, n_zeta)
        if accepted_run >= required_successive:
            angular_converged = True
            break

    previous = _snapshot(final_result) if final_result is not None else None
    accepted_run = 0
    legendre_converged = False
    for n_xi in legendre_orders[1:]:
        grid = GridSpec(*selected_angular, n_xi, dtype=dtype, x64=x64)
        final_result = solve_monoenergetic(surface, grid, case)
        current = _snapshot(final_result)
        if not np.all(np.isfinite(current)):
            return AdaptiveConvergenceResult(
                "model-out-of-scope",
                tuple(steps),
                None,
                angular_converged,
                False,
                "non-finite coefficient encountered during Legendre refinement",
            )
        assert previous is not None
        absolute, relative, accepted = _changes(
            previous, current, rtol=rtol, atol=atol_values
        )
        steps.append(ConvergenceStep("legendre", grid, current, absolute, relative, accepted))
        accepted_run = accepted_run + 1 if accepted else 0
        previous = current
        if accepted_run >= required_successive:
            legendre_converged = True
            break

    converged = angular_converged and legendre_converged
    return AdaptiveConvergenceResult(
        "converged" if converged else "unresolved",
        tuple(steps),
        final_result,
        angular_converged,
        legendre_converged,
        (
            "angular and Legendre ladders passed consecutive refinement gates"
            if converged
            else "one or more ladders ended before consecutive refinement gates passed"
        ),
    )


__all__ = [
    "AdaptiveConvergenceResult",
    "ConvergenceStep",
    "solve_monoenergetic_converged",
]
