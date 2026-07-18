"""Monoenergetic solve orchestration."""

from __future__ import annotations

from jax import Array, core

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
from .resolution import geometry_resolution_report


def prepare_monoenergetic_system(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    *,
    require_resolved_geometry: bool = False,
) -> PreparedMonoenergeticSystem:
    """Precompute geometry and derivatives, optionally enforcing Nyquist sampling."""

    enable_x64(grid.x64)
    if require_resolved_geometry and not isinstance(surface.m, core.Tracer):
        geometry_resolution_report(surface, grid).require_resolved()
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
    *,
    require_resolved_geometry: bool = False,
) -> TransportResult:
    """Solve one monoenergetic DKE case."""

    prepared = prepare_monoenergetic_system(
        surface, grid, require_resolved_geometry=require_resolved_geometry
    )
    return solve_prepared(prepared, case)


def solve_monoenergetic_internal(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    case: MonoenergeticCase,
    *,
    require_resolved_geometry: bool = False,
) -> tuple[Array, Array, Array]:
    """Solve one monoenergetic case and return `(Dij, f, s)` low-order arrays."""

    prepared = prepare_monoenergetic_system(
        surface, grid, require_resolved_geometry=require_resolved_geometry
    )
    return solve_prepared_internal(prepared, case)


__all__ = [
    "prepare_monoenergetic_system",
    "solve_monoenergetic",
    "solve_monoenergetic_internal",
]
