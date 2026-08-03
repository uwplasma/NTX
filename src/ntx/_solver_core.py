"""Monoenergetic solve orchestration."""

from __future__ import annotations

from jax import Array, core

from ._solver_prepared import solve_prepared, solve_prepared_internal
from ._solver_types import (
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    TransportResult,
)
from .config import enable_x64, geometry_precision_matches
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
    if not isinstance(surface.m, core.Tracer) and not geometry_precision_matches(
        surface, grid
    ):
        msg = (
            f"surface was built at a narrower precision than grid.dtype="
            f"{grid.dtype!r} requests. JAX fixes an array's dtype when it is "
            "created, so a surface constructed while x64 was off stays "
            "single-precision and is promoted silently here -- the run would "
            "finish, report float64, and be wrong in the eighth digit. Build "
            "the surface after importing ntx (which enables x64), or pass a "
            "GridSpec whose dtype matches the surface."
        )
        raise ValueError(msg)
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
    adjoint_window: int | None = None,
) -> TransportResult:
    """Solve one monoenergetic DKE case.

    ``adjoint_window`` bounds the memory of a reverse-mode derivative of this
    solve. ``None`` retains every Legendre row, which is exact. A finite window
    retains ``3 + adjoint_window`` rows instead, so the reverse pass stops
    growing with ``grid.n_xi``; :func:`ntx.advise_adjoint_window` estimates a
    starting value. A forward solve is unaffected either way.
    """

    prepared = prepare_monoenergetic_system(
        surface, grid, require_resolved_geometry=require_resolved_geometry
    )
    return solve_prepared(prepared, case, adjoint_window=adjoint_window)


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
