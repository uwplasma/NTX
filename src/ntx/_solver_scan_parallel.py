"""Device health checks and local-device scan sharding."""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._solver_core import prepare_monoenergetic_system
from ._solver_scan_core import solve_monoenergetic_scan
from ._solver_scan_execution import (
    _coefficients_dict,
    _resolved_scan_inputs,
    _scan_coefficients_batched,
    _scan_coefficients_serial,
)
from .geometry import BoozerSurface, VmecSurface, example_surface
from .grids import GridSpec


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
    """Shard a scan over healthy local JAX devices.

    ``scan_batch_size`` is applied inside each device shard so device sharding
    and peak-memory control remain independent choices.
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
    """Return the number of local JAX devices visible to this process."""

    return jax.local_device_count()


def healthy_parallel_device_count() -> int:
    """Return the number of local devices that pass the NTX smoke solve."""

    return len(healthy_parallel_devices())


def healthy_parallel_devices() -> tuple:
    """Return cached local JAX devices that pass a finite NTX smoke solve."""

    return _healthy_parallel_devices_cached()


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
