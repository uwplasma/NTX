"""Non-differentiable multiprocess scan helpers for throughput-oriented runs."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np


def solve_monoenergetic_multiprocess_scan(
    surface,
    grid,
    nu_hat,
    *,
    epsi_hat=None,
    er_hat=None,
    backend: str = "gpu",
    workers: int | None = None,
    gpu_ids: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Run a scan in separate worker processes.

    This path is intended for the throughput lane. It is not designed to remain
    differentiable through the process boundary.
    """

    import jax.numpy as jnp

    from .solver import _resolved_scan_inputs, prepare_monoenergetic_system

    prepared = prepare_monoenergetic_system(surface, grid)
    nu_values, epsi_values, output_shape = _resolved_scan_inputs(
        prepared,
        grid,
        nu_hat,
        epsi_hat,
        er_hat,
    )
    flat_nu = np.asarray(nu_values).ravel()
    flat_epsi = np.asarray(epsi_values).ravel()
    if flat_nu.size == 0:
        zeros = jnp.zeros(output_shape, dtype=grid.jax_dtype)
        return {
            "D11": zeros,
            "D31": zeros,
            "D13": zeros,
            "D33": zeros,
            "D33_spitzer": zeros,
        }

    device_ids = _worker_ids(backend, workers=workers, gpu_ids=gpu_ids)
    shard_count = min(len(device_ids), flat_nu.size)
    shard_size = math.ceil(flat_nu.size / shard_count)
    padded_size = shard_count * shard_size
    if padded_size > flat_nu.size:
        pad = padded_size - flat_nu.size
        flat_nu = cast(np.ndarray, np.pad(flat_nu, (0, pad), mode="edge"))
        flat_epsi = cast(np.ndarray, np.pad(flat_epsi, (0, pad), mode="edge"))
    nu_shards = flat_nu.reshape((shard_count, shard_size))
    epsi_shards = flat_epsi.reshape((shard_count, shard_size))

    surface_payload = _surface_to_payload(surface)
    grid_payload = _grid_to_payload(grid)
    tasks = [
        {
            "backend": backend,
            "worker_id": device_ids[index],
            "surface": surface_payload,
            "grid": grid_payload,
            "nu": nu_shards[index],
            "epsi": epsi_shards[index],
        }
        for index in range(shard_count)
    ]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=shard_count, mp_context=ctx) as pool:
        shard_results = list(pool.map(_solve_scan_worker, tasks))

    output: dict[str, Any] = {}
    for key in ("D11", "D31", "D13", "D33", "D33_spitzer"):
        values = np.concatenate([shard[key] for shard in shard_results], axis=0)[: nu_values.size]
        output[key] = jnp.asarray(values).reshape(output_shape)
    return output


def _worker_ids(
    backend: str,
    *,
    workers: int | None,
    gpu_ids: tuple[int, ...] | None,
) -> tuple[int, ...]:
    import jax

    if backend == "gpu":
        if gpu_ids is None:
            count = jax.local_device_count()
            return tuple(range(count)) if workers is None else tuple(range(min(workers, count)))
        return gpu_ids if workers is None else gpu_ids[:workers]
    if backend == "cpu":
        cpu_workers = 1 if workers is None else max(1, workers)
        return tuple(range(cpu_workers))
    msg = f"unsupported backend {backend!r}"
    raise ValueError(msg)


def _surface_to_payload(surface) -> dict[str, Any]:
    from .geometry import BoozerSurface

    if isinstance(surface, BoozerSurface):
        return {
            "kind": "boozer",
            "m": np.asarray(surface.m),
            "n": np.asarray(surface.n),
            "b_cos": np.asarray(surface.b_cos),
            "nfp": surface.nfp,
            "iota": surface.iota,
            "psi_p": surface.psi_p,
            "b_theta": surface.b_theta,
            "b_zeta": surface.b_zeta,
            "chi_p": surface.chi_p,
            "b0": surface.b0,
            "b_sin": None if surface.b_sin is None else np.asarray(surface.b_sin),
            "stellarator_symmetric": surface.stellarator_symmetric,
        }
    return {
        "kind": "vmec",
        "path": str(surface.path),
        "requested_psi_n": surface.requested_psi_n,
        "psi_n": surface.psi_n,
        "nfp": surface.nfp,
        "ns": surface.ns,
        "mpol": surface.mpol,
        "ntor": surface.ntor,
        "total_mode_count": surface.total_mode_count,
        "loaded_mode_count": surface.loaded_mode_count,
        "iota": surface.iota,
        "m": np.asarray(surface.m),
        "n": np.asarray(surface.n),
        "b_cos": np.asarray(surface.b_cos),
        "jacobian_cos": np.asarray(surface.jacobian_cos),
        "b_sub_theta_cos": np.asarray(surface.b_sub_theta_cos),
        "b_sub_zeta_cos": np.asarray(surface.b_sub_zeta_cos),
        "b_sup_theta_cos": np.asarray(surface.b_sup_theta_cos),
        "b_sup_zeta_cos": np.asarray(surface.b_sup_zeta_cos),
        "b0": surface.b0,
        "psi_a_hat": surface.psi_a_hat,
        "phi_edge": surface.phi_edge,
        "r_n": surface.r_n,
        "r_hat": surface.r_hat,
        "dpsi_hat_dr_hat": surface.dpsi_hat_dr_hat,
        "dr_hat_dpsi_hat": surface.dr_hat_dpsi_hat,
        "aminor_p": surface.aminor_p,
        "psi_p": surface.psi_p,
        "transport_psi_scale": surface.transport_psi_scale,
        "stellarator_symmetric": surface.stellarator_symmetric,
    }


def _grid_to_payload(grid) -> dict[str, Any]:
    return asdict(grid)


def _surface_from_payload(payload: dict[str, Any]):
    import jax.numpy as jnp

    from .geometry import BoozerSurface, VmecSurface

    if payload["kind"] == "boozer":
        return BoozerSurface(
            m=jnp.asarray(payload["m"]),
            n=jnp.asarray(payload["n"]),
            b_cos=jnp.asarray(payload["b_cos"]),
            nfp=int(payload["nfp"]),
            iota=float(payload["iota"]),
            psi_p=float(payload["psi_p"]),
            b_theta=float(payload["b_theta"]),
            b_zeta=float(payload["b_zeta"]),
            chi_p=None if payload["chi_p"] is None else float(payload["chi_p"]),
            b0=None if payload["b0"] is None else float(payload["b0"]),
            b_sin=None if payload["b_sin"] is None else jnp.asarray(payload["b_sin"]),
            stellarator_symmetric=bool(payload["stellarator_symmetric"]),
        )
    return VmecSurface(
        path=Path(payload["path"]),
        requested_psi_n=float(payload["requested_psi_n"]),
        psi_n=float(payload["psi_n"]),
        nfp=int(payload["nfp"]),
        ns=int(payload["ns"]),
        mpol=int(payload["mpol"]),
        ntor=int(payload["ntor"]),
        total_mode_count=int(payload["total_mode_count"]),
        loaded_mode_count=int(payload["loaded_mode_count"]),
        iota=float(payload["iota"]),
        m=jnp.asarray(payload["m"]),
        n=jnp.asarray(payload["n"]),
        b_cos=jnp.asarray(payload["b_cos"]),
        jacobian_cos=jnp.asarray(payload["jacobian_cos"]),
        b_sub_theta_cos=jnp.asarray(payload["b_sub_theta_cos"]),
        b_sub_zeta_cos=jnp.asarray(payload["b_sub_zeta_cos"]),
        b_sup_theta_cos=jnp.asarray(payload["b_sup_theta_cos"]),
        b_sup_zeta_cos=jnp.asarray(payload["b_sup_zeta_cos"]),
        b0=float(payload["b0"]),
        psi_a_hat=float(payload["psi_a_hat"]),
        phi_edge=float(payload["phi_edge"]),
        r_n=float(payload["r_n"]),
        r_hat=float(payload["r_hat"]),
        dpsi_hat_dr_hat=float(payload["dpsi_hat_dr_hat"]),
        dr_hat_dpsi_hat=float(payload["dr_hat_dpsi_hat"]),
        aminor_p=None if payload["aminor_p"] is None else float(payload["aminor_p"]),
        psi_p=None if payload["psi_p"] is None else float(payload["psi_p"]),
        transport_psi_scale=float(payload["transport_psi_scale"]),
        stellarator_symmetric=bool(payload["stellarator_symmetric"]),
    )


def _solve_scan_worker(task: dict[str, Any]) -> dict[str, np.ndarray]:
    backend = task["backend"]
    if backend == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(task["worker_id"])
        os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
    else:
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax.numpy as jnp

    from .config import enable_x64
    from .grids import GridSpec
    from .solver import solve_monoenergetic_scan

    enable_x64(bool(task["grid"]["x64"]))
    grid = GridSpec(**task["grid"])
    surface = _surface_from_payload(task["surface"])
    coeffs = solve_monoenergetic_scan(
        surface,
        grid,
        jnp.asarray(task["nu"], dtype=grid.jax_dtype),
        epsi_hat=jnp.asarray(task["epsi"], dtype=grid.jax_dtype),
    )
    return {key: np.asarray(value) for key, value in coeffs.items()}
