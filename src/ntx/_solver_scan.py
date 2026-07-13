"""Scan and device-parallel helpers for the monoenergetic solver."""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._solver_core import prepare_monoenergetic_system
from ._solver_prepared import _solve_prepared_coefficient_vector_raw, solve_prepared
from ._solver_types import MonoenergeticCase, PreparedMonoenergeticSystem, TransportResult
from .geometry import BoozerSurface, VmecSurface, example_surface
from .grids import GridSpec

ScanExecutionMode = Literal["auto", "sequential", "vectorized"]
SUPPORTED_SCAN_BATCH_SIZES = (1, 8, 32, 128)


class _CompiledBatchFunction(Protocol):
    def __call__(self, nu_values: Array, epsi_values: Array) -> Array: ...

    def lower(self, nu_values: Array, epsi_values: Array) -> Any: ...


@dataclass(frozen=True)
class PreparedScanCompilationReport:
    """Ahead-of-time warmup timings and executable memory estimates."""

    lowering_seconds: float
    compilation_seconds: float
    first_execution_seconds: float
    warm_execution_seconds: float
    generated_code_size_bytes: int | None
    argument_size_bytes: int | None
    output_size_bytes: int | None
    temporary_size_bytes: int | None


class CompiledPreparedScanSolver:
    """Reusable fixed-shape scan solver for one prepared geometry.

    The final input chunk is padded to ``batch_size`` and trimmed after the
    solve. Consequently, repeated calls reuse one executable instead of
    recompiling for each scan length.
    """

    def __init__(
        self,
        prepared: PreparedMonoenergeticSystem,
        *,
        batch_size: int,
        execution_mode: Literal["sequential", "vectorized"],
        solve_batch: _CompiledBatchFunction,
    ) -> None:
        self.prepared = prepared
        self.batch_size = batch_size
        self.execution_mode = execution_mode
        self._solve_batch = solve_batch

    def __call__(
        self,
        nu_hat: Array,
        *,
        epsi_hat: Array | None = None,
        er_hat: Array | None = None,
    ) -> dict[str, Array]:
        """Solve a scan while preserving the broadcast input shape."""

        nu_values, epsi_values, output_shape = _resolved_scan_inputs(
            self.prepared,
            self.prepared.grid,
            nu_hat,
            epsi_hat,
            er_hat,
        )
        coeffs = _run_fixed_batch_scan(
            self.prepared,
            nu_values.ravel(),
            epsi_values.ravel(),
            batch_size=self.batch_size,
            solve_batch=self._solve_batch,
        )
        return _coefficients_dict(coeffs.reshape((*output_shape, 5)))

    def warmup(self) -> PreparedScanCompilationReport:
        """Lower, compile, and execute the fixed-shape scan once explicitly."""

        dtype = self.prepared.grid.jax_dtype
        nu = jnp.ones((self.batch_size,), dtype=dtype)
        epsi = jnp.zeros((self.batch_size,), dtype=dtype)

        started = time.perf_counter()
        lowered = self._solve_batch.lower(nu, epsi)
        lowering_seconds = time.perf_counter() - started

        started = time.perf_counter()
        executable = lowered.compile()
        compilation_seconds = time.perf_counter() - started

        started = time.perf_counter()
        first = executable(nu, epsi)
        first.block_until_ready()
        first_execution_seconds = time.perf_counter() - started

        started = time.perf_counter()
        warm = executable(nu, epsi)
        warm.block_until_ready()
        warm_execution_seconds = time.perf_counter() - started

        memory = executable.memory_analysis()
        return PreparedScanCompilationReport(
            lowering_seconds=lowering_seconds,
            compilation_seconds=compilation_seconds,
            first_execution_seconds=first_execution_seconds,
            warm_execution_seconds=warm_execution_seconds,
            generated_code_size_bytes=_memory_stat(memory, "generated_code_size_in_bytes"),
            argument_size_bytes=_memory_stat(memory, "argument_size_in_bytes"),
            output_size_bytes=_memory_stat(memory, "output_size_in_bytes"),
            temporary_size_bytes=_memory_stat(memory, "temp_size_in_bytes"),
        )


def compile_prepared_scan_solver(
    prepared: PreparedMonoenergeticSystem,
    *,
    batch_size: int | None = None,
    execution_mode: ScanExecutionMode = "auto",
) -> CompiledPreparedScanSolver:
    """Create a reusable fixed-shape scan solver for ``prepared``.

    CPU execution defaults to sequential ``lax.map`` to bound memory. Other
    accelerators default to vectorized execution. Explicit modes are useful for
    measured crossover studies.
    """

    mode = _resolve_scan_execution_mode(execution_mode)
    resolved_batch_size = (8 if mode == "sequential" else 32) if batch_size is None else batch_size
    if resolved_batch_size not in SUPPORTED_SCAN_BATCH_SIZES:
        supported = ", ".join(str(value) for value in SUPPORTED_SCAN_BATCH_SIZES)
        msg = f"batch_size must be one of the supported fixed buckets: {supported}"
        raise ValueError(msg)
    if mode == "sequential":
        solve_batch = jax.jit(
            lambda nu_values, epsi_values: _scan_coefficients_sequential_impl(
                prepared, nu_values, epsi_values
            )
        )
    else:
        solve_batch = jax.jit(
            lambda nu_values, epsi_values: _scan_coefficients_vectorized_impl(
                prepared, nu_values, epsi_values
            )
        )
    return CompiledPreparedScanSolver(
        prepared,
        batch_size=resolved_batch_size,
        execution_mode=mode,
        solve_batch=solve_batch,
    )


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
    mode = _resolve_scan_execution_mode("auto")
    if mode == "sequential":
        return _scan_coefficients_sequential(prepared, nu_values, epsi_values)
    return _scan_coefficients_vectorized(prepared, nu_values, epsi_values)


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

    mode = _resolve_scan_execution_mode("auto")
    kernel = (
        _scan_coefficients_sequential if mode == "sequential" else _scan_coefficients_vectorized
    )
    return _run_fixed_batch_scan(
        prepared,
        nu_values,
        epsi_values,
        batch_size=batch_size,
        solve_batch=lambda chunk_nu, chunk_epsi: kernel(prepared, chunk_nu, chunk_epsi),
    )


def _run_fixed_batch_scan(
    prepared: PreparedMonoenergeticSystem,
    nu_values: Array,
    epsi_values: Array,
    *,
    batch_size: int,
    solve_batch: Callable[[Array, Array], Array],
) -> Array:
    """Execute fixed-size chunks so every call has one compiled shape."""

    if batch_size < 1:
        msg = "scan_batch_size must be a positive integer"
        raise ValueError(msg)
    case_count = int(nu_values.size)
    if case_count == 0:
        return jnp.zeros((0, 5), dtype=prepared.grid.jax_dtype)
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


def _solve_scan_point(
    prepared: PreparedMonoenergeticSystem,
    nu_value: Array,
    epsi_value: Array,
) -> Array:
    return _solve_prepared_coefficient_vector_raw(prepared, nu_value, epsi_value)


@jax.jit
def _scan_coefficients_sequential(
    prepared: PreparedMonoenergeticSystem,
    nu_values: Array,
    epsi_values: Array,
) -> Array:
    return _scan_coefficients_sequential_impl(prepared, nu_values, epsi_values)


def _scan_coefficients_sequential_impl(
    prepared: PreparedMonoenergeticSystem,
    nu_values: Array,
    epsi_values: Array,
) -> Array:
    return jax.lax.map(
        lambda values: _solve_scan_point(prepared, values[0], values[1]),
        (nu_values, epsi_values),
    )


@jax.jit
def _scan_coefficients_vectorized(
    prepared: PreparedMonoenergeticSystem,
    nu_values: Array,
    epsi_values: Array,
) -> Array:
    return _scan_coefficients_vectorized_impl(prepared, nu_values, epsi_values)


def _scan_coefficients_vectorized_impl(
    prepared: PreparedMonoenergeticSystem,
    nu_values: Array,
    epsi_values: Array,
) -> Array:
    return jax.vmap(lambda nu_value, epsi_value: _solve_scan_point(prepared, nu_value, epsi_value))(
        nu_values, epsi_values
    )


def _resolve_scan_execution_mode(
    execution_mode: ScanExecutionMode,
) -> Literal["sequential", "vectorized"]:
    if execution_mode == "auto":
        return "sequential" if jax.default_backend() == "cpu" else "vectorized"
    if execution_mode not in ("sequential", "vectorized"):
        msg = "execution_mode must be 'auto', 'sequential', or 'vectorized'"
        raise ValueError(msg)
    return execution_mode


def _memory_stat(memory, name: str) -> int | None:
    if memory is None:
        return None
    value = getattr(memory, name, None)
    return None if value is None else int(value)


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
