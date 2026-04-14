"""Dense JAX block-tridiagonal monoenergetic DKE solver."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, tree_util
from jax.scipy.linalg import lu_factor, lu_solve

from .config import enable_x64
from .geometry import BoozerSurface, GeometryOnGrid, VmecSurface, example_surface, geometry_on_grid
from .grids import GridSpec
from .operators import (
    OperatorContext,
    apply_nullspace_condition,
    derivative_blocks,
    operator_blocks,
    source_modes,
)
from .transport import coefficients_from_modes, onsager_error


@dataclass(frozen=True)
class MonoenergeticCase:
    """Monoenergetic DKE parameters."""

    nu_hat: float
    epsi_hat: float | None = None
    er_hat: float | None = None

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
    D11: Array
    D31: Array
    D13: Array
    D33: Array
    D33_spitzer: Array
    f1_modes: Array
    f3_modes: Array
    residual_l2: Array
    onsager_residual: Array

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
class PreparedMonoenergeticSystem:
    """Cached geometry and derivative operators for repeated solves."""

    surface: BoozerSurface | VmecSurface
    grid: GridSpec
    geometry: GeometryOnGrid
    d_theta: Array
    d_zeta: Array


tree_util.register_dataclass(PreparedMonoenergeticSystem)


CompiledPreparedSolver = Callable[[MonoenergeticCase], TransportResult]


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

    return _transport_result_from_arrays(_solve_prepared_arrays(prepared, case))


def solve_prepared_internal(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Solve one prepared monoenergetic case and return `(Dij, f, s)` low-order arrays."""

    values = _solve_prepared_arrays(prepared, case)
    result = _transport_result_from_arrays(values)
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
    """Coefficient-vector solve with an explicit custom-VJP contract point.

    The current backward pass still differentiates the raw coefficient kernel
    exactly. This keeps the public API stable while NTX transitions toward an
    implicit or adjoint derivative for the prepared dense solve.
    """

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
        return _transport_result_from_arrays(compiled(case.nu_hat, epsi_hat))

    return solve


def _solve_prepared_coefficient_vector_vjp_fwd(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, tuple[Array, Array, Array | None, bool, bool]]:
    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    coefficients = _solve_prepared_coefficient_vector_raw(
        prepared,
        case.nu_hat,
        resolved_epsi_hat,
    )
    return coefficients, (
        jnp.asarray(case.nu_hat),
        resolved_epsi_hat,
        None if transport_scale is None else jnp.asarray(transport_scale),
        case.epsi_hat is not None,
        case.er_hat is not None,
    )


def _solve_prepared_coefficient_vector_vjp_bwd(
    prepared: PreparedMonoenergeticSystem,
    residuals: tuple[Array, Array, Array | None, bool, bool],
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase]:
    nu_hat, resolved_epsi_hat, transport_scale, uses_epsi_hat, uses_er_hat = residuals
    _, pullback = jax.vjp(
        lambda trial_nu_hat, trial_epsi_hat: _solve_prepared_coefficient_vector_raw(
            prepared,
            trial_nu_hat,
            trial_epsi_hat,
        ),
        nu_hat,
        resolved_epsi_hat,
    )
    nu_bar, epsi_bar = pullback(coefficient_bar)
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


def _transport_result_from_arrays(values: tuple[Array, ...]) -> TransportResult:
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
    """Solve one monoenergetic case and return raw array outputs."""

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
    """Solve one monoenergetic case given resolved scalar inputs."""

    geom = prepared.geometry
    grid = prepared.grid
    ctx = _operator_context(
        prepared.surface,
        geom,
        grid,
        nu_hat,
        epsi_hat,
    )
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


def solve_scan(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    cases: tuple[MonoenergeticCase, ...],
) -> list[TransportResult]:
    """Solve a Python-level scan of monoenergetic cases."""

    prepared = prepare_monoenergetic_system(surface, grid)
    return [solve_prepared(prepared, case) for case in cases]


def _stack_internal_systems(primary: Array, parallel: Array) -> Array:
    return jnp.stack((primary, primary, parallel))


def _monoenergetic_matrix(d11: Array, d31: Array, d13: Array, d33: Array) -> Array:
    return jnp.asarray(
        [
            [d11, d11, d13],
            [d11, d11, d13],
            [d31, d31, d33],
        ]
    )


def solve_monoenergetic_scan(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    nu_hat: Array,
    *,
    epsi_hat: Array | None = None,
    er_hat: Array | None = None,
) -> dict[str, Array]:
    """Vectorized scan over collisionality and radial electric field."""

    prepared = prepare_monoenergetic_system(surface, grid)
    nu_values, epsi_values, output_shape = _resolved_scan_inputs(
        prepared,
        grid,
        nu_hat,
        epsi_hat,
        er_hat,
    )
    coeffs = _scan_coefficients_serial(prepared, nu_values.ravel(), epsi_values.ravel())
    return _coefficients_dict(coeffs.reshape((*output_shape, 5)))


def solve_monoenergetic_parallel_scan(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    nu_hat: Array,
    *,
    epsi_hat: Array | None = None,
    er_hat: Array | None = None,
    num_devices: int | None = None,
) -> dict[str, Array]:
    """Device-parallel scan over collisionality and radial electric field.

    This path shards the flattened scan across local devices at the host level
    and runs the stable single-device scan on each shard. It is intended for
    larger scans where throughput matters more than autodiff through the whole
    multi-device dispatch layer.
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
    devices = healthy_parallel_devices()
    available_devices = len(devices)
    device_count = available_devices if num_devices is None else min(num_devices, available_devices)
    if device_count < 1:
        msg = "no healthy local JAX devices are available for parallel execution"
        raise ValueError(msg)
    if available_devices < jax.local_device_count():
        warnings.warn(
            "some local JAX devices failed the NTX smoke solve and were excluded "
            "from parallel execution",
            RuntimeWarning,
            stacklevel=2,
        )
    if flat_nu.size == 0:
        coeffs = jnp.zeros((*output_shape, 5), dtype=grid.jax_dtype)
        return _coefficients_dict(coeffs)
    if device_count == 1:
        coeffs = _scan_coefficients_serial(prepared, flat_nu, flat_epsi)
        return _coefficients_dict(coeffs.reshape((*output_shape, 5)))

    shard_count = min(device_count, flat_nu.size)
    shard_size = math.ceil(flat_nu.size / shard_count)
    padded_size = shard_size * shard_count
    pad = padded_size - flat_nu.size
    if pad:
        flat_nu = jnp.pad(flat_nu, (0, pad), mode="edge")
        flat_epsi = jnp.pad(flat_epsi, (0, pad), mode="edge")
    nu_shards = np.asarray(flat_nu).reshape((shard_count, shard_size))
    epsi_shards = np.asarray(flat_epsi).reshape((shard_count, shard_size))

    def worker(args):
        device, nu_shard, epsi_shard = args
        with jax.default_device(device):
            values = solve_monoenergetic_scan(
                surface,
                grid,
                jnp.asarray(nu_shard, dtype=grid.jax_dtype),
                epsi_hat=jnp.asarray(epsi_shard, dtype=grid.jax_dtype),
            )
        return {key: np.asarray(jax.device_get(value)) for key, value in values.items()}

    with ThreadPoolExecutor(max_workers=shard_count) as pool:
        shard_results = list(
            pool.map(
                worker,
                zip(devices[:shard_count], nu_shards, epsi_shards, strict=True),
            )
        )

    output = {}
    for key in ("D11", "D31", "D13", "D33", "D33_spitzer"):
        joined = np.concatenate([result[key] for result in shard_results], axis=0)[: nu_values.size]
        output[key] = jnp.asarray(joined).reshape(output_shape)
    return output


def local_parallel_device_count() -> int:
    """Return the number of local devices available to JAX parallel scans."""

    return jax.local_device_count()


def healthy_parallel_device_count() -> int:
    """Return the number of local devices that pass a small NTX smoke solve."""

    return len(healthy_parallel_devices())


def healthy_parallel_devices() -> tuple:
    """Return local devices that pass a small NTX smoke solve."""

    return _healthy_parallel_devices_cached()


def _solve_modes(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    s1: Array,
    s3: Array,
) -> tuple[Array, Array]:
    """Return source solutions for modes 0, 1, and 2."""

    lower_terminal, delta, lower_next = _terminal_delta(ctx, n_xi, d_theta, d_zeta)
    x = lu_solve(lu_factor(delta), lower_next)

    n_fs = delta.shape[0]
    saved_delta_init = jnp.zeros((3, n_fs, n_fs), dtype=delta.dtype)
    saved_lower_init = jnp.zeros((3, n_fs, n_fs), dtype=delta.dtype)
    saved_upper_init = jnp.zeros((3, n_fs, n_fs), dtype=delta.dtype)
    if n_xi == 2:
        saved_delta_init = saved_delta_init.at[2].set(delta)
        saved_lower_init = saved_lower_init.at[2].set(lower_terminal)

    def scan_step(carry, k):
        x_prev, saved_delta, saved_lower, saved_upper = carry
        lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)

        def fix_nullspace(args):
            diagonal_in, upper_in = args
            diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal_in, upper_in)
            assert upper_fixed is not None
            return diagonal_fixed, upper_fixed

        diagonal, upper = jax.lax.cond(
            k == 0,
            fix_nullspace,
            lambda args: args,
            (diagonal, upper),
        )
        delta_k = diagonal - upper @ x_prev

        def save_needed(args):
            saved_delta_in, saved_lower_in, saved_upper_in = args
            return (
                saved_delta_in.at[k].set(delta_k),
                saved_lower_in.at[k].set(lower),
                saved_upper_in.at[k].set(upper),
            )

        saved_delta, saved_lower, saved_upper = jax.lax.cond(
            k <= 2,
            save_needed,
            lambda args: args,
            (saved_delta, saved_lower, saved_upper),
        )
        x_next = jax.lax.cond(
            k > 0,
            lambda _: lu_solve(lu_factor(delta_k), lower),
            lambda _: x_prev,
            operand=None,
        )
        return (x_next, saved_delta, saved_lower, saved_upper), None

    ks = jnp.arange(n_xi - 1, -1, -1)
    (_, saved_delta, saved_lower, saved_upper), _ = jax.lax.scan(
        scan_step,
        (x, saved_delta_init, saved_lower_init, saved_upper_init),
        ks,
    )

    sigma1 = {
        2: s1[2],
        1: s1[1],
        0: s1[0],
    }
    sigma3 = {
        2: s3[2],
        1: s3[1],
        0: s3[0],
    }

    lu2 = lu_factor(saved_delta[2])
    y1 = lu_solve(lu2, sigma1[2])
    sigma1[1] = s1[1] - saved_upper[1] @ y1

    lu1 = lu_factor(saved_delta[1])
    y13 = lu_solve(lu1, jnp.stack((sigma1[1], sigma3[1]), axis=-1))
    y1 = y13[:, 0]
    y3 = y13[:, 1]
    sigma1[0] = s1[0] - saved_upper[0] @ y1
    sigma3[0] = s3[0] - saved_upper[0] @ y3

    f1 = []
    f3 = []
    lu0 = lu_factor(saved_delta[0])
    f03 = lu_solve(lu0, jnp.stack((sigma1[0], sigma3[0]), axis=-1))
    f1_0 = f03[:, 0]
    f3_0 = f03[:, 1]
    f1.append(f1_0)
    f3.append(f3_0)
    rhs_13 = jnp.stack(
        (
            sigma1[1] - saved_lower[1] @ f1[0],
            sigma3[1] - saved_lower[1] @ f3[0],
        ),
        axis=-1,
    )
    f13 = lu_solve(lu1, rhs_13)
    f1.append(f13[:, 0])
    f3.append(f13[:, 1])

    rhs_23 = jnp.stack(
        (
            sigma1[2] - saved_lower[2] @ f1[1],
            sigma3[2] - saved_lower[2] @ f3[1],
        ),
        axis=-1,
    )
    f23 = lu_solve(lu2, rhs_23)
    f1.append(f23[:, 0])
    f3.append(f23[:, 1])
    return jnp.stack(f1), jnp.stack(f3)


def _terminal_delta(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array]:
    lower, diagonal, _ = operator_blocks(ctx, n_xi, d_theta, d_zeta)
    return lower, diagonal, lower


def _residual_norm(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    source: Array,
    modes: Array,
) -> Array:
    """Residual norm for the solved low Legendre modes."""

    residuals = []
    for k in range(3):
        lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)
        if k == 0:
            diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal, upper)
            assert upper_fixed is not None
            diagonal = diagonal_fixed
            upper = upper_fixed
        value = diagonal @ modes[k] - source[k]
        if k > 0:
            value = value + lower @ modes[k - 1]
        if k < 2:
            value = value + upper @ modes[k + 1]
        residuals.append(value)
    residual = jnp.concatenate(residuals)
    _ = n_xi
    return jnp.linalg.norm(residual) / jnp.sqrt(residual.size)

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


def _resolved_scan_inputs(
    prepared: PreparedMonoenergeticSystem,
    grid: GridSpec,
    nu_hat: Array,
    epsi_hat: Array | None,
    er_hat: Array | None,
) -> tuple[Array, Array, tuple[int, ...]]:
    geom = prepared.geometry
    if epsi_hat is not None and er_hat is not None:
        msg = "set only one of epsi_hat or er_hat"
        raise ValueError(msg)
    nu_values = jnp.asarray(nu_hat, dtype=grid.jax_dtype)
    if epsi_hat is None:
        if er_hat is None:
            epsi_values = jnp.zeros_like(nu_values)
        else:
            if geom.transport_psi_scale is None:
                msg = "er_hat scans require a surface with a transport normalization scale"
                raise ValueError(msg)
            epsi_values = jnp.asarray(er_hat, dtype=grid.jax_dtype) / geom.transport_psi_scale
    else:
        epsi_values = jnp.asarray(epsi_hat, dtype=grid.jax_dtype)
    nu_values, epsi_values = jnp.broadcast_arrays(nu_values, epsi_values)
    return nu_values, epsi_values, nu_values.shape


def _scan_coefficients_serial(
    prepared: PreparedMonoenergeticSystem,
    nu_values: Array,
    epsi_values: Array,
) -> Array:
    geom = prepared.geometry
    grid = prepared.grid

    def solve_one(nu_value, epsi_value):
        ctx = _operator_context(prepared.surface, geom, grid, nu_value, epsi_value)
        s1, s3 = source_modes(ctx, grid.n_xi)
        f1_modes, f3_modes = _solve_modes(
            ctx,
            grid.n_xi,
            prepared.d_theta,
            prepared.d_zeta,
            s1,
            s3,
        )
        return jnp.stack(coefficients_from_modes(geom, f1_modes, f3_modes, nu_value))

    return jax.jit(jax.vmap(solve_one))(nu_values, epsi_values)


def _coefficients_dict(coeffs: Array) -> dict[str, Array]:
    return {
        "D11": coeffs[..., 0],
        "D31": coeffs[..., 1],
        "D13": coeffs[..., 2],
        "D33": coeffs[..., 3],
        "D33_spitzer": coeffs[..., 4],
    }


@lru_cache(maxsize=1)
def _healthy_parallel_devices_cached() -> tuple:
    devices = tuple(jax.local_devices())
    healthy = []
    surface = example_surface()
    grid = GridSpec(5, 5, 4)
    nu = jnp.logspace(-4, -3, 3)
    er = jnp.linspace(0.0, 1e-3, 3)
    for device in devices:
        try:
            with jax.default_device(device):
                coeffs = solve_monoenergetic_scan(surface, grid, nu, er_hat=er)
                is_finite = all(bool(jnp.all(jnp.isfinite(value))) for value in coeffs.values())
            if is_finite:
                healthy.append(device)
        except Exception:
            continue
    return tuple(healthy)
