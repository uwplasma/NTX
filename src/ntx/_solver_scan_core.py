"""Single-device scan orchestration for the monoenergetic solver."""

from __future__ import annotations

from jax import Array

from ._solver_core import prepare_monoenergetic_system
from ._solver_prepared import solve_prepared
from ._solver_scan_execution import (
    _coefficients_dict,
    _resolved_scan_inputs,
    _scan_coefficients_batched,
    _scan_coefficients_serial,
)
from ._solver_types import MonoenergeticCase, TransportResult
from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec


def solve_scan(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    cases: tuple[MonoenergeticCase, ...],
) -> list[TransportResult]:
    """Solve a Python-level scan of monoenergetic cases."""

    prepared = prepare_monoenergetic_system(surface, grid)
    return [solve_prepared(prepared, case) for case in cases]


def solve_monoenergetic_scan(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    nu_hat: Array,
    *,
    epsi_hat: Array | None = None,
    er_hat: Array | None = None,
    scan_batch_size: int | None = None,
) -> dict[str, Array]:
    """Scan collisionality and radial electric field on one JAX device.

    ``scan_batch_size`` optionally splits the flattened scan into fixed-size
    batches. This preserves coefficient values while bounding peak memory on
    CPUs and memory-constrained accelerators.

    Reverse-mode differentiation of this scan works and is checked against a
    finite difference. It uses the taped path; the bounded reverse pass of
    :func:`ntx.solve_monoenergetic` is not available here yet, because a
    ``custom_vjp`` under the scan's batching raises. Differentiate
    :func:`ntx.solve_prepared` per point if you need the window.
    """

    prepared = prepare_monoenergetic_system(surface, grid)
    nu_values, epsi_values, output_shape = _resolved_scan_inputs(
        prepared,
        grid,
        nu_hat,
        epsi_hat,
        er_hat,
    )
    flat_nu = nu_values.ravel()
    flat_epsi = epsi_values.ravel()
    if scan_batch_size is None:
        coeffs = _scan_coefficients_serial(prepared, flat_nu, flat_epsi)
    else:
        coeffs = _scan_coefficients_batched(
            prepared,
            flat_nu,
            flat_epsi,
            batch_size=scan_batch_size,
        )
    return _coefficients_dict(coeffs.reshape((*output_shape, 5)))
