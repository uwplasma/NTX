from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from ntx import GridSpec, example_surface, solve_monoenergetic_multiprocess_scan
from ntx.geometry import VmecSurface
from ntx.parallel import (
    _grid_to_payload,
    _solve_scan_worker,
    _surface_from_payload,
    _surface_to_payload,
    _worker_ids,
)


def test_multiprocess_zero_size_scan_returns_zeros():
    result = solve_monoenergetic_multiprocess_scan(
        example_surface(),
        GridSpec(5, 5, 4),
        jnp.asarray([]),
        backend="cpu",
        workers=2,
    )
    for value in result.values():
        assert value.shape == (0,)


def test_worker_ids_cover_cpu_gpu_and_error(monkeypatch):
    fake_jax = SimpleNamespace(local_device_count=lambda: 3)
    monkeypatch.setitem(__import__("sys").modules, "jax", fake_jax)
    assert _worker_ids("gpu", workers=None, gpu_ids=None) == (0, 1, 2)
    assert _worker_ids("gpu", workers=2, gpu_ids=None) == (0, 1)
    assert _worker_ids("gpu", workers=1, gpu_ids=(4, 7)) == (4,)
    assert _worker_ids("cpu", workers=None, gpu_ids=None) == (0,)
    assert _worker_ids("cpu", workers=3, gpu_ids=None) == (0, 1, 2)
    with pytest.raises(ValueError, match="unsupported backend"):
        _worker_ids("tpu", workers=1, gpu_ids=None)


def test_parallel_payload_roundtrip_for_boozer_and_vmec():
    boozer_surface = example_surface()
    boozer_payload = _surface_to_payload(boozer_surface)
    boozer_roundtrip = _surface_from_payload(boozer_payload)
    assert jnp.allclose(boozer_roundtrip.b_cos, boozer_surface.b_cos)
    assert boozer_roundtrip.nfp == boozer_surface.nfp

    vmec_surface = VmecSurface(
        path=__import__("pathlib").Path("wout_test.nc"),
        requested_psi_n=0.25,
        psi_n=0.25,
        nfp=2,
        ns=3,
        mpol=3,
        ntor=1,
        total_mode_count=2,
        loaded_mode_count=2,
        iota=0.6,
        m=jnp.asarray([0, 1], dtype=jnp.int32),
        n=jnp.asarray([0, 1], dtype=jnp.int32),
        b_cos=jnp.asarray([1.0, 0.1]),
        jacobian_cos=jnp.asarray([1.0, 0.0]),
        b_sub_theta_cos=jnp.asarray([0.2, 0.0]),
        b_sub_zeta_cos=jnp.asarray([1.1, 0.0]),
        b_sup_theta_cos=jnp.asarray([0.3, 0.0]),
        b_sup_zeta_cos=jnp.asarray([1.2, 0.0]),
        b0=1.0,
        psi_a_hat=1.0,
        phi_edge=1.0,
        r_n=0.5,
        r_hat=0.5,
        dpsi_hat_dr_hat=1.0,
        dr_hat_dpsi_hat=1.0,
        aminor_p=1.0,
        psi_p=1.0,
        transport_psi_scale=1.0,
    )
    vmec_payload = _surface_to_payload(vmec_surface)
    vmec_roundtrip = _surface_from_payload(vmec_payload)
    assert jnp.allclose(vmec_roundtrip.b_cos, vmec_surface.b_cos)
    assert vmec_roundtrip.path.name == "wout_test.nc"


def test_grid_payload_roundtrip_and_worker_cpu_path():
    grid = GridSpec(5, 5, 4)
    grid_payload = _grid_to_payload(grid)
    assert grid_payload["n_theta"] == 5
    task = {
        "backend": "cpu",
        "worker_id": 0,
        "surface": _surface_to_payload(example_surface()),
        "grid": grid_payload,
        "nu": np.asarray([1e-3, 2e-3]),
        "epsi": np.asarray([0.0, 1e-4]),
    }
    result = _solve_scan_worker(task)
    assert set(result) == {"D11", "D31", "D13", "D33", "D33_spitzer"}
    assert result["D11"].shape == (2,)
