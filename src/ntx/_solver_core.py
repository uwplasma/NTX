"""Monoenergetic solve orchestration."""

from __future__ import annotations

from jax import Array

from ._solver_prepared import solve_prepared, solve_prepared_internal
from ._solver_types import (
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    TransportResult,
)
from .config import enable_x64
from .geometry import BoozerSurface, VmecSurface, geometry_on_grid
from .grids import GridSpec
from .operators import derivative_blocks


def prepare_monoenergetic_system(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
) -> PreparedMonoenergeticSystem:
    """Precompute geometry and derivative blocks for repeated solves."""

    enable_x64(grid.x64)
    geom = geometry_on_grid(surface, grid)
    d_theta, d_zeta = derivative_blocks(geom)
    return PreparedMonoenergeticSystem(
        surface=surface,
        grid=grid,
        geometry=geom,
        d_theta=d_theta,
        d_zeta=d_zeta,
    )


def solve_monoenergetic(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    case: MonoenergeticCase,
) -> TransportResult:
    """Solve one monoenergetic DKE case."""

    prepared = prepare_monoenergetic_system(surface, grid)
    return solve_prepared(prepared, case)


def solve_monoenergetic_internal(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    case: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Solve one monoenergetic case and return `(Dij, f, s)` low-order arrays."""

    prepared = prepare_monoenergetic_system(surface, grid)
    return solve_prepared_internal(prepared, case)


__all__ = [
    "prepare_monoenergetic_system",
    "solve_monoenergetic",
    "solve_monoenergetic_internal",
]
