"""Prepared monoenergetic solve path and custom-VJP core."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from ._solver_adjoint import (
    _coefficient_mode_pullback,
    _parameter_gradient_from_adjoint,
    _prepared_implicit_vjp_primal,
)
from ._solver_factorization import (
    _residual_norm,
    _solve_factorized_adjoint,
    _solve_modes,
)
from ._solver_types import (
    CompiledPreparedSolver,
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    TransportResult,
    transport_result_from_arrays,
)
from .config import enable_x64
from .geometry import BoozerSurface, VmecSurface, geometry_on_grid
from .grids import GridSpec
from .operators import (
    OperatorContext,
    derivative_blocks,
    source_modes,
)
from .transport import coefficients_from_modes, onsager_error


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


def solve_prepared(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> TransportResult:
    """Solve one monoenergetic case using precomputed geometry and derivatives."""

    return transport_result_from_arrays(_solve_prepared_arrays(prepared, case))


def solve_prepared_internal(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Solve one prepared monoenergetic case and return `(Dij, f, s)` low-order arrays."""

    values = _solve_prepared_arrays(prepared, case)
    result = transport_result_from_arrays(values)
    dij = _monoenergetic_matrix(result.D11, result.D31, result.D13, result.D33)
    return dij, values[9], values[10]


def solve_prepared_coefficient_vector(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> Array:
    """Return the coefficient vector `[D11, D31, D13, D33, D33_spitzer]`."""

    return _solve_prepared_coefficient_vector_raw(
        prepared,
        case.nu_hat,
        case.resolved_epsi_hat(prepared.geometry.transport_psi_scale),
    )


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def solve_prepared_coefficient_vector_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> Array:
    """Coefficient-vector solve with an explicit custom-VJP contract point."""

    return solve_prepared_coefficient_vector(prepared, case)


def compile_prepared_solver(
    prepared: PreparedMonoenergeticSystem,
) -> CompiledPreparedSolver:
    """Return a jitted monoenergetic solver for repeated solves on one geometry."""

    compiled = jax.jit(
        lambda nu_hat, epsi_hat: _solve_prepared_arrays_from_values(
            prepared,
            nu_hat,
            epsi_hat,
        )
    )

    def solve(case: MonoenergeticCase) -> TransportResult:
        epsi_hat = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
        return transport_result_from_arrays(compiled(case.nu_hat, epsi_hat))

    return solve


def _solve_prepared_coefficient_vector_vjp_fwd(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[
    Array,
    tuple[Array, Array, Array | None, bool, bool, Array, Array, Array, Array, Array, Array],
]:
    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper = (
        _prepared_implicit_vjp_primal(
            prepared,
            case.nu_hat,
            resolved_epsi_hat,
        )
    )
    return coefficients, (
        jnp.asarray(case.nu_hat),
        resolved_epsi_hat,
        None if transport_scale is None else jnp.asarray(transport_scale),
        case.epsi_hat is not None,
        case.er_hat is not None,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
    )


def _solve_prepared_coefficient_vector_vjp_bwd(
    prepared: PreparedMonoenergeticSystem,
    residuals: tuple[
        Array,
        Array,
        Array | None,
        bool,
        bool,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ],
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase]:
    (
        nu_hat,
        resolved_epsi_hat,
        transport_scale,
        uses_epsi_hat,
        uses_er_hat,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
    ) = residuals
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        nu_hat,
        resolved_epsi_hat,
    )
    f1_bar_low, f3_bar_low, nu_bar_direct = _coefficient_mode_pullback(
        prepared.geometry,
        f1_full[:3],
        f3_full[:3],
        ctx.nu_hat,
        coefficient_bar,
    )
    g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
    g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
    lambda1 = _solve_factorized_adjoint(saved_lu, saved_piv, saved_lower, saved_upper, g1)
    lambda3 = _solve_factorized_adjoint(saved_lu, saved_piv, saved_lower, saved_upper, g3)
    nu_bar_implicit, epsi_bar = _parameter_gradient_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
    )
    nu_bar = nu_bar_direct + nu_bar_implicit
    if uses_epsi_hat:
        return (MonoenergeticCase(nu_hat=nu_bar, epsi_hat=epsi_bar, er_hat=None),)
    if uses_er_hat:
        assert transport_scale is not None
        er_bar = epsi_bar / transport_scale
        return (MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=er_bar),)
    return (MonoenergeticCase(nu_hat=nu_bar, epsi_hat=None, er_hat=None),)


solve_prepared_coefficient_vector_vjp.defvjp(
    _solve_prepared_coefficient_vector_vjp_fwd,
    _solve_prepared_coefficient_vector_vjp_bwd,
)


def _solve_prepared_coefficient_vector_raw(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> Array:
    values = _solve_prepared_arrays_from_values(prepared, nu_hat, epsi_hat)
    return jnp.stack(values[:5])


def _solve_prepared_arrays(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, ...]:
    return _solve_prepared_arrays_from_values(
        prepared,
        case.nu_hat,
        case.resolved_epsi_hat(prepared.geometry.transport_psi_scale),
    )


def _solve_prepared_arrays_from_values(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> tuple[Array, ...]:
    geom = prepared.geometry
    grid = prepared.grid
    ctx = _operator_context(prepared.surface, geom, grid, nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    f1_modes, f3_modes = _solve_modes(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        s1,
        s3,
    )
    d11, d31, d13, d33, d33_spitzer = coefficients_from_modes(
        geom, f1_modes, f3_modes, ctx.nu_hat
    )
    residual = _residual_norm(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        s1,
        f1_modes,
    )
    return (
        d11,
        d31,
        d13,
        d33,
        d33_spitzer,
        f1_modes,
        f3_modes,
        residual,
        onsager_error(d31, d13),
        _stack_internal_systems(f1_modes, f3_modes),
        _stack_internal_systems(s1[:3], s3[:3]),
    )


def _stack_internal_systems(primary: Array, parallel: Array) -> Array:
    return jnp.stack((primary, primary, parallel))


def _monoenergetic_matrix(d11: Array, d31: Array, d13: Array, d33: Array) -> Array:
    return jnp.asarray([[d11, d11, d13], [d11, d11, d13], [d31, d31, d33]])


def _operator_context(
    surface: BoozerSurface | VmecSurface,
    geom,
    grid: GridSpec,
    nu_hat,
    epsi_hat,
) -> OperatorContext:
    return OperatorContext(
        surface=surface,
        geometry=geom,
        nu_hat=jnp.asarray(nu_hat, dtype=grid.jax_dtype),
        epsi_hat=jnp.asarray(epsi_hat, dtype=grid.jax_dtype),
    )
