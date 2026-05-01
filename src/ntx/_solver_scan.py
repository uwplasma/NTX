"""Scan and device-parallel helpers for the monoenergetic solver."""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._solver_context import _operator_context
from ._solver_core import prepare_monoenergetic_system
from ._solver_factorization import _solve_modes
from ._solver_prepared import solve_prepared
from ._solver_types import MonoenergeticCase, PreparedMonoenergeticSystem, TransportResult
from .geometry import BoozerSurface, VmecSurface, example_surface
from .grids import GridSpec
from .transport import coefficients_from_modes


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
    """Vectorized scan over collisionality and radial electric field.

    ``scan_batch_size`` optionally splits the flattened scan into fixed-size
    batches. This preserves the coefficient values while lowering peak memory
    for large CPU or memory-constrained GPU scans.
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


def solve_monoenergetic_parallel_scan(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    nu_hat: Array,
    *,
    epsi_hat: Array | None = None,
    er_hat: Array | None = None,
    num_devices: int | None = None,
    scan_batch_size: int | None = None,
) -> dict[str, Array]:
    """Device-parallel scan over collisionality and radial electric field.

    ``scan_batch_size`` is applied inside each device shard. It is useful when
    the scan is wide enough to benefit from device sharding but each shard still
    needs bounded peak memory.
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
                scan_batch_size=scan_batch_size,
            )
        return {key: np.asarray(jax.device_get(value)) for key, value in values.items()}

    with ThreadPoolExecutor(max_workers=shard_count) as pool:
        shard_results = list(
            pool.map(worker, zip(devices[:shard_count], nu_shards, epsi_shards, strict=True))
        )

    output = {}
    for key in ("D11", "D31", "D13", "D33", "D33_spitzer"):
        joined = np.concatenate([result[key] for result in shard_results], axis=0)[: nu_values.size]
        output[key] = jnp.asarray(joined).reshape(output_shape)
    return output


def local_parallel_device_count() -> int:
    return jax.local_device_count()


def healthy_parallel_device_count() -> int:
    return len(healthy_parallel_devices())


def healthy_parallel_devices() -> tuple:
    return _healthy_parallel_devices_cached()


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
    return _scan_coefficients_function(prepared)(nu_values, epsi_values)


def _scan_coefficients_batched(
    prepared: PreparedMonoenergeticSystem,
    nu_values: Array,
    epsi_values: Array,
    *,
    batch_size: int,
) -> Array:
    if batch_size < 1:
        msg = "scan_batch_size must be a positive integer"
        raise ValueError(msg)
    case_count = int(nu_values.size)
    if case_count == 0:
        return jnp.zeros((0, 5), dtype=prepared.grid.jax_dtype)

    solve_batch = _scan_coefficients_function(prepared)
    chunks = []
    for start in range(0, case_count, batch_size):
        stop = min(start + batch_size, case_count)
        chunk_nu = nu_values[start:stop]
        chunk_epsi = epsi_values[start:stop]
        valid_count = stop - start
        if valid_count < batch_size:
            pad = batch_size - valid_count
            chunk_nu = jnp.pad(chunk_nu, (0, pad), mode="edge")
            chunk_epsi = jnp.pad(chunk_epsi, (0, pad), mode="edge")
        chunks.append(solve_batch(chunk_nu, chunk_epsi)[:valid_count])
    return jnp.concatenate(chunks, axis=0)


def _scan_coefficients_function(prepared: PreparedMonoenergeticSystem):
    geom = prepared.geometry
    grid = prepared.grid

    def solve_one(nu_value, epsi_value):
        ctx = _operator_context(prepared.surface, geom, grid, nu_value, epsi_value)
        from .operators import source_modes

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

    return jax.jit(jax.vmap(solve_one))


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
