"""The drift-kinetic solve: types, context, preparation, factorization, adjoint.

One pass of the solver, from the caller's configuration through the prepared
operator and its factorization to the adjoint and the truncation window. The
scan driver that repeats this over a parameter sweep lives in _solver_scan.
"""

from __future__ import annotations

import jax.numpy as jnp
from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .operators import OperatorContext
import jax
from jax import Array
from solvax import (
    BlockTridiagFactors,
    block_thomas_factor_fn,
    block_thomas_solve,
    block_thomas_truncated_fn_with_residual,
)
from .operators import (
    OperatorContext,
    apply_nullspace_condition,
    block_parameters,
    operator_blocks,
    operator_blocks_from_parameters,
)
from collections.abc import Callable
from dataclasses import dataclass
from jax import Array, tree_util
from .geometry import BoozerSurface, GeometryOnGrid, VmecSurface
from .operators import OperatorContext, parameter_derivative_blocks, source_modes
from .transport import coefficients_from_modes
from functools import partial
from .operators import source_modes
from .transport import coefficients_from_modes, onsager_error
from jax import Array, core
from .config import enable_x64, geometry_precision_matches
from .geometry import BoozerSurface, VmecSurface, geometry_on_grid
from .operators import derivative_blocks
from .resolution import geometry_resolution_report
import solvax
from .operators import OperatorContext, block_parameters, source_modes

__all__ = [
    "_operator_context",
    "audit_prepared_residuals",
    "compile_prepared_solver",
    "prepare_monoenergetic_system",
    "solve_monoenergetic",
    "solve_monoenergetic_internal",
    "solve_prepared",
    "solve_prepared_coefficient_vector",
    "solve_prepared_coefficient_vector_vjp",
    "solve_prepared_internal",
]


# --- _solver_context: Shared solver context construction. ---

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


# --- _solver_factorization: Block-tridiagonal solve and factorized adjoint helpers. ---

def _solve_modes(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    s1: Array,
    s3: Array,
    adjoint_window: int | None = None,
) -> tuple[Array, Array]:
    """Return source solutions for modes 0, 1, and 2."""

    f1, f3, _ = _solve_modes_with_tail_residual(
        ctx, n_xi, d_theta, d_zeta, s1, s3, adjoint_window
    )
    return f1, f3


def _solve_modes_with_tail_residual(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    s1: Array,
    s3: Array,
    adjoint_window: int | None = None,
) -> tuple[Array, Array, Array]:
    """Return retained modes and the residual of the tail-eliminated system.

    The forward solve is the same either way: a full-tail elimination keeping
    only the three retained modes. What ``adjoint_window`` changes is how a
    derivative of it is taken.

    ``None`` (the default) differentiates the elimination directly. Reverse mode
    records the whole sweep, so its memory grows with ``n_xi``, but *both* AD
    modes work, which matters --- the derivative audits use forward mode.

    An integer selects the exact-window rule: the rows are regenerated on demand
    and only ``3 + adjoint_window`` of them are retained, so the reverse pass
    stops growing with the Legendre resolution. This path is a ``custom_vjp``
    and is therefore **reverse-mode only**; ``jax.jacfwd`` and ``jax.jvp``
    through it raise. Passing ``adjoint_window = n_xi + 1`` retains every row
    and is exact, matching the taped gradient to rounding while still costing
    less; a shorter window trades a quantified error for bounded memory, and
    :func:`advise_adjoint_window` estimates where that becomes worthwhile.
    """

    keep_lowest = 3
    n_blocks = n_xi + 1
    rhs_low = jnp.stack((s1[:keep_lowest], s3[:keep_lowest]), axis=-1)
    block_fn = _parameterized_block_fn(d_theta, d_zeta)
    if adjoint_window is None:
        modes, residual = block_thomas_truncated_fn_with_residual(
            _operator_block_fn(ctx, d_theta, d_zeta),
            n_blocks=n_blocks,
            rhs_low=rhs_low,
            keep_lowest=keep_lowest,
            residual_rhs_index=0,
        )
    else:
        modes, residual = block_thomas_truncated_fn_with_residual(
            block_fn,
            n_blocks=n_blocks,
            rhs_low=rhs_low,
            keep_lowest=keep_lowest,
            params=block_parameters(ctx),
            adjoint_window=int(adjoint_window),
            residual_rhs_index=0,
        )
    f1_modes = modes[..., 0]
    f3_modes = modes[..., 1]
    return f1_modes, f3_modes, residual


def _factorize_prepared_modes(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array, Array]:
    factors = block_thomas_factor_fn(_operator_block_fn(ctx, d_theta, d_zeta), n_blocks=n_xi + 1)
    return factors.delta_lu, factors.delta_piv, factors.lower, factors.upper


def _solve_factorized_modes(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    source: Array,
) -> Array:
    factors = BlockTridiagFactors(saved_lu, saved_piv, saved_lower, saved_upper)
    return block_thomas_solve(factors, source)


def _solve_factorized_adjoint(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    source_bar: Array,
) -> Array:
    factors = BlockTridiagFactors(saved_lu, saved_piv, saved_lower, saved_upper)
    return block_thomas_solve(factors, source_bar, transpose=True)


def _operator_block_fn(ctx: OperatorContext, d_theta: Array, d_zeta: Array):
    def block_fn(k):
        return _conditioned_operator_blocks(ctx, k, d_theta, d_zeta)

    return block_fn


def _parameterized_block_fn(d_theta: Array, d_zeta: Array):
    """Row generator taking the parameters explicitly, for the bounded adjoint."""

    def block_fn(params, k):
        lower, diagonal, upper = operator_blocks_from_parameters(
            params, k, d_theta, d_zeta
        )
        return _apply_nullspace_at_first_row(k, lower, diagonal, upper)

    return block_fn


def _apply_nullspace_at_first_row(k, lower, diagonal, upper):
    """Impose ``f^(0)(0,0) = 0`` on row zero, which is otherwise singular."""

    def fix_nullspace(args):
        diagonal_in, upper_in = args
        diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal_in, upper_in)
        assert upper_fixed is not None
        return diagonal_fixed, upper_fixed

    diagonal, upper = jax.lax.cond(
        jnp.asarray(k) == 0, fix_nullspace, lambda args: args, (diagonal, upper)
    )
    return lower, diagonal, upper


def _conditioned_operator_blocks(
    ctx: OperatorContext,
    k: int | Array,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array]:
    lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)
    return _apply_nullspace_at_first_row(k, lower, diagonal, upper)


def _full_mode_residual_norm(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    source: Array,
    modes: Array,
) -> Array:
    """Evaluate the original block equations for every retained full mode."""
    if modes.shape[0] != n_xi + 1 or source.shape[0] != n_xi + 1:
        raise ValueError("full residual requires n_xi + 1 source and solution modes")
    residuals = []
    for k in range(n_xi + 1):
        lower, diagonal, upper = _conditioned_operator_blocks(ctx, k, d_theta, d_zeta)
        value = diagonal @ modes[k] - source[k]
        if k > 0:
            value = value + lower @ modes[k - 1]
        if k < n_xi:
            value = value + upper @ modes[k + 1]
        residuals.append(value)
    residual = jnp.concatenate(residuals)
    return jnp.linalg.norm(residual) / jnp.sqrt(residual.size)


def _full_mode_relative_residual_norm(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    source: Array,
    modes: Array,
) -> Array:
    residual_rms = _full_mode_residual_norm(ctx, n_xi, d_theta, d_zeta, source, modes)
    source_rms = jnp.linalg.norm(source) / jnp.sqrt(source.size)
    tiny = jnp.finfo(residual_rms.dtype).tiny
    return residual_rms / jnp.maximum(source_rms, tiny)


def _full_mode_transpose_relative_residual_norm(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    source_bar: Array,
    adjoint_modes: Array,
) -> Array:
    """Evaluate ``||A.T @ lambda - g|| / ||g||`` from physics blocks."""

    if adjoint_modes.shape[0] != n_xi + 1 or source_bar.shape[0] != n_xi + 1:
        raise ValueError("transpose residual requires n_xi + 1 source and solution modes")
    applied = [jnp.zeros_like(adjoint_modes[0]) for _ in range(n_xi + 1)]
    for k in range(n_xi + 1):
        lower, diagonal, upper = _conditioned_operator_blocks(ctx, k, d_theta, d_zeta)
        applied[k] = applied[k] + diagonal.T @ adjoint_modes[k]
        if k > 0:
            applied[k - 1] = applied[k - 1] + lower.T @ adjoint_modes[k]
        if k < n_xi:
            applied[k + 1] = applied[k + 1] + upper.T @ adjoint_modes[k]
    residual = jnp.concatenate(
        [value - source for value, source in zip(applied, source_bar, strict=True)]
    )
    source_norm = jnp.linalg.norm(source_bar)
    tiny = jnp.finfo(residual.dtype).tiny
    return jnp.linalg.norm(residual) / jnp.maximum(source_norm, tiny)


# --- _solver_types: Core monoenergetic solver dataclasses and result helpers. ---

@dataclass(frozen=True)
class MonoenergeticCase:
    """Monoenergetic DKE parameters."""

    nu_hat: float | Array
    epsi_hat: float | Array | None = None
    er_hat: float | Array | None = None

    def resolved_epsi_hat(self, transport_psi_scale: float | Array | None) -> Array:
        if self.epsi_hat is not None and self.er_hat is not None:
            msg = "set only one of epsi_hat or er_hat"
            raise ValueError(msg)
        if self.epsi_hat is not None:
            return jnp.asarray(self.epsi_hat)
        if self.er_hat is not None:
            if transport_psi_scale is None:
                msg = "er_hat requires a surface with a transport normalization scale"
                raise ValueError(msg)
            return jnp.asarray(self.er_hat) / jnp.asarray(transport_psi_scale)
        if transport_psi_scale is not None:
            return jnp.zeros_like(jnp.asarray(transport_psi_scale))
        return jnp.asarray(0.0)


tree_util.register_dataclass(MonoenergeticCase)


@dataclass(frozen=True)
class TransportResult:
    """Monoenergetic coefficients, retained modes, and solver diagnostics.

    ``residual_l2`` is retained for API compatibility. It is the RMS residual
    of the tail-eliminated Schur system, not the residual of every original
    Legendre block row. Use ``schur_residual_l2`` for explicit new code and
    :func:`ntx.audit_prepared_residuals` when a full-system residual is needed.
    """

    D11: Array
    D31: Array
    D13: Array
    D33: Array
    D33_spitzer: Array
    f1_modes: Array
    f3_modes: Array
    residual_l2: Array
    onsager_residual: Array

    @property
    def schur_residual_l2(self) -> Array:
        """RMS algebraic residual of the complete tail-eliminated system."""

        return self.residual_l2

    def as_dict(self) -> dict[str, float]:
        return {
            "D11": float(self.D11),
            "D31": float(self.D31),
            "D13": float(self.D13),
            "D33": float(self.D33),
            "D33_spitzer": float(self.D33_spitzer),
            "residual_l2": float(self.residual_l2),
            "onsager_residual": float(self.onsager_residual),
        }


tree_util.register_dataclass(TransportResult)


@dataclass(frozen=True)
class ResidualAuditResult:
    """Opt-in comparison of the low-memory and full Legendre solves."""

    tail_eliminated_l2: Array
    full_system_l2: Array
    retained_mode_max_abs_error: Array
    n_modes: int

    @property
    def schur_residual_l2(self) -> Array:
        """RMS residual of the tail-eliminated Schur system."""

        return self.tail_eliminated_l2

    @property
    def full_system_residual_l2(self) -> Array:
        """RMS residual obtained by applying every original Legendre row."""

        return self.full_system_l2


tree_util.register_dataclass(
    ResidualAuditResult,
    data_fields=(
        "tail_eliminated_l2",
        "full_system_l2",
        "retained_mode_max_abs_error",
    ),
    meta_fields=("n_modes",),
)


@dataclass(frozen=True)
class PreparedMonoenergeticSystem:
    """Cached geometry and derivative operators for repeated solves."""

    surface: BoozerSurface | VmecSurface
    grid: GridSpec
    geometry: GeometryOnGrid
    d_theta: Array
    d_zeta: Array


tree_util.register_dataclass(PreparedMonoenergeticSystem)


CompiledPreparedSolver = Callable[[MonoenergeticCase], TransportResult]


def transport_result_from_arrays(values: tuple[Array, ...]) -> TransportResult:
    return TransportResult(
        D11=values[0],
        D31=values[1],
        D13=values[2],
        D33=values[3],
        D33_spitzer=values[4],
        f1_modes=values[5],
        f3_modes=values[6],
        residual_l2=values[7],
        onsager_residual=values[8],
    )


# --- _solver_adjoint: Prepared-solver adjoint and custom-VJP helper algebra. ---

def _prepared_implicit_vjp_primal(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    geom = prepared.geometry
    grid = prepared.grid
    ctx = _operator_context(prepared.surface, geom, grid, nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    saved_lu, saved_piv, saved_lower, saved_upper = _factorize_prepared_modes(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
    )
    f1_full = _solve_factorized_modes(saved_lu, saved_piv, saved_lower, saved_upper, s1)
    f3_full = _solve_factorized_modes(saved_lu, saved_piv, saved_lower, saved_upper, s3)

    def coefficient_fn(modes1, modes3, nu_value):
        return jnp.stack(coefficients_from_modes(geom, modes1, modes3, nu_value))

    coefficients = coefficient_fn(f1_full[:3], f3_full[:3], ctx.nu_hat)
    return coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper


def _coefficient_mode_pullback(
    geom,
    f1_low: Array,
    f3_low: Array,
    nu_hat: Array,
    coefficient_bar: Array,
) -> tuple[Array, Array, Array]:
    def coefficient_fn(modes1, modes3, nu_value):
        return jnp.stack(coefficients_from_modes(geom, modes1, modes3, nu_value))

    _, pullback = jax.vjp(coefficient_fn, f1_low, f3_low, nu_hat)
    f1_bar, f3_bar, nu_bar = pullback(coefficient_bar)
    return f1_bar, f3_bar, nu_bar


def _parameter_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
) -> tuple[Array, Array]:
    def zero_first_row(block: Array) -> Array:
        return block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))

    nu_bar = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    epsi_bar = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = zero_first_row(diagonal_nu)
            diagonal_epsi = zero_first_row(diagonal_epsi)
        nu_bar = nu_bar - (
            jnp.vdot(lambda1[k], diagonal_nu @ f1_full[k])
            + jnp.vdot(lambda3[k], diagonal_nu @ f3_full[k])
        )
        epsi_bar = epsi_bar - (
            jnp.vdot(lambda1[k], diagonal_epsi @ f1_full[k])
            + jnp.vdot(lambda3[k], diagonal_epsi @ f3_full[k])
        )
    return nu_bar, epsi_bar


# --- _solver_prepared: Prepared monoenergetic solve path and custom-VJP wrappers. ---

def solve_prepared(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    *,
    adjoint_window: int | None = None,
) -> TransportResult:
    """Solve one monoenergetic case using precomputed geometry and derivatives.

    ``adjoint_window`` bounds the reverse pass; see
    :func:`ntx.advise_adjoint_window`. It has no effect on a forward solve.
    """

    return transport_result_from_arrays(
        _solve_prepared_arrays(prepared, case, adjoint_window=adjoint_window)
    )


def solve_prepared_internal(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Solve one prepared monoenergetic case and return `(Dij, f, s)` low-order arrays."""

    values = _solve_prepared_arrays(prepared, case)
    result = transport_result_from_arrays(values)
    dij = _monoenergetic_matrix(result.D11, result.D31, result.D13, result.D33)
    return dij, values[9], values[10]


def audit_prepared_residuals(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> ResidualAuditResult:
    """Reconstruct all Legendre modes and independently audit every block row.

    This diagnostic stores full block factors and modes. Use it for validation,
    not in memory-constrained production scans.
    """
    grid = prepared.grid
    epsi_hat = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
    ctx = _operator_context(
        prepared.surface, prepared.geometry, grid, case.nu_hat, epsi_hat
    )
    source, parallel_source = source_modes(ctx, grid.n_xi)
    retained, _, tail_residual = _solve_modes_with_tail_residual(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        source,
        parallel_source,
    )
    factors = _factorize_prepared_modes(
        ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta
    )
    full_modes = _solve_factorized_modes(*factors, source)
    full_residual = _full_mode_residual_norm(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        source,
        full_modes,
    )
    return ResidualAuditResult(
        tail_eliminated_l2=tail_residual,
        full_system_l2=full_residual,
        retained_mode_max_abs_error=jnp.max(jnp.abs(retained - full_modes[:3])),
        n_modes=grid.n_xi + 1,
    )


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
    *,
    adjoint_window: int | None = None,
) -> Array:
    values = _solve_prepared_arrays_from_values(
        prepared, nu_hat, epsi_hat, adjoint_window=adjoint_window
    )
    return jnp.stack(values[:5])


def _solve_prepared_arrays(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    *,
    adjoint_window: int | None = None,
) -> tuple[Array, ...]:
    return _solve_prepared_arrays_from_values(
        prepared,
        case.nu_hat,
        case.resolved_epsi_hat(prepared.geometry.transport_psi_scale),
        adjoint_window=adjoint_window,
    )


def _solve_prepared_arrays_from_values(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
    *,
    adjoint_window: int | None = None,
) -> tuple[Array, ...]:
    geom = prepared.geometry
    grid = prepared.grid
    ctx = _operator_context(prepared.surface, geom, grid, nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    f1_modes, f3_modes, residual = _solve_modes_with_tail_residual(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
        s1,
        s3,
        adjoint_window,
    )
    d11, d31, d13, d33, d33_spitzer = coefficients_from_modes(
        geom, f1_modes, f3_modes, ctx.nu_hat
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


# --- _solver_core: Monoenergetic solve orchestration. ---

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


# --- _solver_window: Choosing the adjoint window: a cheap estimate, and a certified one. ---

KEEP_LOWEST = 3  # the transport coefficients read Legendre modes 0, 1 and 2


COEFFICIENTS = ("D11", "D31", "D13", "D33")


def advise_adjoint_window(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
):
    """Estimate where the Legendre chain becomes localized enough to truncate.

    Returns SOLVAX's ``LocalizationWindow``: the per-row transfer norms
    ``rho_k``, the first row where they fall below one, and a suggested window.

    The estimate is read from the operator, not from a gradient, and it is not
    a certificate: ``certified`` is always ``False``. The physics behind it is
    that pitch-angle scattering damps Legendre mode ``l`` like ``nu*l(l+1)``
    while the streaming coupling grows only like ``l``, so the chain contracts
    faster the higher one climbs, and the row where it starts contracting moves
    outward as collisions weaken. Use the value as a starting point and widen
    it until the gradient stops moving.
    """

    return solvax.localization_crossover_window(
        lambda k: _conditioned_operator_blocks(ctx, k, d_theta, d_zeta),
        n_xi + 1,
        keep_lowest=3,
    )


def certify_adjoint_window(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    *,
    rtol: float = 1.0e-6,
    coefficient: str = "D11",
):
    """Smallest adjoint window whose gradient error is *provably* within ``rtol``.

    :func:`ntx.advise_adjoint_window` reads the operator and returns a
    plausible window; this returns one with a proof attached. The difference
    matters when the gradient drives an optimizer, because a window chosen by
    eye is wrong by an amount nobody measured.

    A certificate is a statement about one differentiated quantity, so it needs
    that quantity's cotangent rather than the operator alone. The transport
    coefficients are linear functionals of the three retained Legendre modes,
    so the cotangent follows exactly from differentiating
    :func:`ntx.transport.coefficients_from_modes` -- no extra solve, and no
    approximation of the thing being certified.

    Args:
        prepared: geometry from :func:`ntx.prepare_monoenergetic_system`.
        case: the collisionality and electric field to certify at. Both matter:
            weaker collisionality pushes the crossover outward, so the window
            certified at one collisionality is not certified at another.
        rtol: target relative error of the parameter gradient.
        coefficient: which coefficient's gradient to certify --- ``"D11"``,
            ``"D31"``, ``"D13"`` or ``"D33"``.

    Returns:
        SOLVAX's ``LocalizationWindow`` with ``certified=True``. It converts to
        ``int``, so it passes straight to ``adjoint_window=``.
        ``certified_relative_error`` is the proven bound and ``status`` says
        whether a proper window was found or the exact one was returned.

    Note:
        Every step of the bound is a worst case, so the certified window is
        wider than the shortest that would have worked --- by a couple of rows
        on a well-localized chain, by considerably more when it barely
        contracts. Where the chain does not localize the exact window comes
        back, which is correct and saves nothing.
    """

    if coefficient not in COEFFICIENTS:
        msg = f"coefficient must be one of {COEFFICIENTS}; got {coefficient!r}"
        raise ValueError(msg)
    index = COEFFICIENTS.index(coefficient)

    grid = prepared.grid
    geom = prepared.geometry
    epsi_hat = case.resolved_epsi_hat(geom.transport_psi_scale)
    ctx = _operator_context(prepared.surface, geom, grid, case.nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    rhs_low = jnp.stack((s1[:KEEP_LOWEST], s3[:KEEP_LOWEST]), axis=-1)

    f1_modes, f3_modes, _ = _solve_modes_with_tail_residual(
        ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta, s1, s3, None
    )
    retained = jnp.stack((f1_modes, f3_modes), axis=-1)

    def selected(modes: Array) -> Array:
        values = coefficients_from_modes(
            geom, modes[..., 0], modes[..., 1], ctx.nu_hat
        )
        return jnp.asarray(values[index])

    cotangent = jax.grad(selected)(retained)

    return solvax.certified_adjoint_window(
        _parameterized_block_fn(prepared.d_theta, prepared.d_zeta),
        grid.n_xi + 1,
        KEEP_LOWEST,
        block_parameters(ctx),
        rhs_low,
        cotangent,
        rtol=rtol,
    )
