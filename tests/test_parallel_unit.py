from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import ntx._solver_scan as solver_scan
import ntx.parallel as parallel_mod
from ntx import GridSpec, example_surface, solve_monoenergetic_multiprocess_scan
from ntx.geometry import VmecSurface
from ntx.parallel import (
    _grid_to_payload,
    _solve_scan_worker,
    _surface_from_payload,
    _surface_to_payload,
    _worker_ids,
)
from ntx.solver import MonoenergeticCase


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


def test_multiprocess_scan_nonempty_branch_uses_executor(monkeypatch):
    grid = GridSpec(5, 5, 4)
    fake_prepared = SimpleNamespace(grid=grid)
    monkeypatch.setattr(parallel_mod, "_worker_ids", lambda *args, **kwargs: (0, 1))

    def fake_prepare(*args, **kwargs):
        return fake_prepared

    def fake_resolved(*args, **kwargs):
        return (
            np.asarray([1.0e-3, 2.0e-3, 3.0e-3]),
            np.asarray([0.0, 1.0e-4, 2.0e-4]),
            (3,),
        )

    class DummyPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, tasks):
            return [
                {
                    "D11": np.asarray([1.0, 2.0]),
                    "D31": np.asarray([3.0, 4.0]),
                    "D13": np.asarray([5.0, 6.0]),
                    "D33": np.asarray([7.0, 8.0]),
                    "D33_spitzer": np.asarray([9.0, 10.0]),
                },
                {
                    "D11": np.asarray([11.0, 12.0]),
                    "D31": np.asarray([13.0, 14.0]),
                    "D13": np.asarray([15.0, 16.0]),
                    "D33": np.asarray([17.0, 18.0]),
                    "D33_spitzer": np.asarray([19.0, 20.0]),
                },
            ]

    solver_stub = SimpleNamespace(
        _resolved_scan_inputs=fake_resolved,
        prepare_monoenergetic_system=fake_prepare,
    )
    monkeypatch.setitem(__import__("sys").modules, "ntx.solver", solver_stub)
    monkeypatch.setattr(parallel_mod.mp, "get_context", lambda _: "ctx")
    monkeypatch.setattr(parallel_mod, "ProcessPoolExecutor", DummyPool)

    result = solve_monoenergetic_multiprocess_scan(
        example_surface(),
        grid,
        jnp.asarray([1.0e-3, 2.0e-3, 3.0e-3]),
        backend="cpu",
        workers=2,
    )
    assert jnp.allclose(result["D11"], jnp.asarray([1.0, 2.0, 11.0]))
    assert jnp.allclose(result["D33_spitzer"], jnp.asarray([9.0, 10.0, 19.0]))


def test_solve_scan_and_resolved_scan_inputs_branches():
    case = MonoenergeticCase(nu_hat=1.0e-3, er_hat=1.0e-4)
    result = solver_scan.solve_scan(example_surface(), GridSpec(5, 5, 4), (case,))
    assert len(result) == 1
    prepared = solver_scan.prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))
    with pytest.raises(ValueError, match="set only one of epsi_hat or er_hat"):
        solver_scan._resolved_scan_inputs(
            prepared,
            prepared.grid,
            jnp.asarray([1.0e-3]),
            jnp.asarray([0.0]),
            jnp.asarray([1.0e-4]),
        )
    prepared = SimpleNamespace(
        geometry=SimpleNamespace(transport_psi_scale=None),
        grid=GridSpec(5, 5, 4),
    )
    with pytest.raises(ValueError, match="transport normalization scale"):
        solver_scan._resolved_scan_inputs(
            prepared,
            prepared.grid,
            jnp.asarray([1.0e-3]),
            None,
            jnp.asarray([1.0e-4]),
        )


def test_parallel_scan_handles_zero_devices_empty_inputs_and_single_device(monkeypatch):
    grid = GridSpec(5, 5, 4)
    fake_prepared = SimpleNamespace(grid=grid)
    monkeypatch.setattr(
        solver_scan,
        "prepare_monoenergetic_system",
        lambda *args, **kwargs: fake_prepared,
    )

    def fake_resolved(*args, **kwargs):
        return jnp.asarray([]), jnp.asarray([]), (0,)

    monkeypatch.setattr(solver_scan, "_resolved_scan_inputs", fake_resolved)
    monkeypatch.setattr(solver_scan, "healthy_parallel_devices", lambda: ())
    with pytest.raises(ValueError, match="no healthy local JAX devices"):
        solver_scan.solve_monoenergetic_parallel_scan(example_surface(), grid, jnp.asarray([]))

    monkeypatch.setattr(solver_scan, "healthy_parallel_devices", lambda: (object(),))
    zeros = solver_scan.solve_monoenergetic_parallel_scan(example_surface(), grid, jnp.asarray([]))
    assert all(value.shape == (0,) for value in zeros.values())

    monkeypatch.setattr(
        solver_scan,
        "_resolved_scan_inputs",
        lambda *args, **kwargs: (
            jnp.asarray([1.0e-3, 2.0e-3]),
            jnp.asarray([0.0, 1.0e-4]),
            (2,),
        ),
    )
    monkeypatch.setattr(
        solver_scan,
        "_scan_coefficients_serial",
        lambda prepared, nu_values, epsi_values: jnp.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        ),
    )
    serial = solver_scan.solve_monoenergetic_parallel_scan(
        example_surface(),
        grid,
        jnp.asarray([1.0e-3, 2.0e-3]),
        num_devices=1,
    )
    assert jnp.allclose(serial["D33"], jnp.asarray([4.0, 9.0]))

    batched_calls = []
    monkeypatch.setattr(
        solver_scan,
        "_scan_coefficients_batched",
        lambda prepared, nu_values, epsi_values, *, batch_size: (
            batched_calls.append(batch_size)
            or jnp.asarray(
                [
                    [11.0, 12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0, 20.0],
                ]
            )
        ),
    )
    serial_batched = solver_scan.solve_monoenergetic_parallel_scan(
        example_surface(),
        grid,
        jnp.asarray([1.0e-3, 2.0e-3]),
        num_devices=1,
        scan_batch_size=1,
    )
    assert jnp.allclose(serial_batched["D33"], jnp.asarray([14.0, 19.0]))
    assert batched_calls == [1]


def test_parallel_scan_warns_and_shards_when_devices_are_filtered(monkeypatch):
    grid = GridSpec(5, 5, 4)
    fake_prepared = SimpleNamespace(grid=grid)
    monkeypatch.setattr(
        solver_scan,
        "prepare_monoenergetic_system",
        lambda *args, **kwargs: fake_prepared,
    )
    monkeypatch.setattr(
        solver_scan,
        "_resolved_scan_inputs",
        lambda *args, **kwargs: (
            jnp.asarray([1.0e-3, 2.0e-3, 3.0e-3]),
            jnp.asarray([0.0, 1.0e-4, 2.0e-4]),
            (3,),
        ),
    )
    devices = ("d0", "d1")
    monkeypatch.setattr(solver_scan, "healthy_parallel_devices", lambda: devices)
    monkeypatch.setattr(solver_scan.jax, "local_device_count", lambda: 3)

    class DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(solver_scan.jax, "default_device", lambda device: DummyContext())
    monkeypatch.setattr(solver_scan.jax, "device_get", lambda value: value)

    seen_batch_sizes = []

    def fake_scan(surface, grid, nu_hat, *, epsi_hat=None, er_hat=None, scan_batch_size=None):
        seen_batch_sizes.append(scan_batch_size)
        size = len(nu_hat)
        base = np.arange(size, dtype=float) + 1.0
        return {
            "D11": base,
            "D31": base + 10.0,
            "D13": base + 20.0,
            "D33": base + 30.0,
            "D33_spitzer": base + 40.0,
        }

    monkeypatch.setattr(solver_scan, "solve_monoenergetic_scan", fake_scan)
    with pytest.warns(RuntimeWarning, match="failed the NTX smoke solve"):
        result = solver_scan.solve_monoenergetic_parallel_scan(
            example_surface(),
            grid,
            jnp.asarray([1.0e-3, 2.0e-3, 3.0e-3]),
            scan_batch_size=2,
        )
    assert result["D11"].shape == (3,)
    assert jnp.allclose(result["D11"], jnp.asarray([1.0, 2.0, 1.0]))
    assert seen_batch_sizes == [2, 2]


def test_healthy_parallel_helpers_cover_count_and_exception_branch(monkeypatch):
    solver_scan._healthy_parallel_devices_cached.cache_clear()
    monkeypatch.setattr(solver_scan.jax, "local_devices", lambda: ("good", "bad"))

    class DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(solver_scan.jax, "default_device", lambda device: DummyContext())

    def fake_scan(surface, grid, nu, *, er_hat=None, epsi_hat=None):
        if nu is None:  # pragma: no cover - defensive
            raise RuntimeError("unexpected")
        if er_hat is None:
            raise RuntimeError("unexpected")
        if len(getattr(fake_scan, "calls", [])) == 0:
            fake_scan.calls = ["good"]
            return {key: jnp.asarray([1.0]) for key in ("D11", "D31", "D13", "D33", "D33_spitzer")}
        raise RuntimeError("device failed")

    fake_scan.calls = []
    monkeypatch.setattr(solver_scan, "solve_monoenergetic_scan", fake_scan)
    healthy = solver_scan.healthy_parallel_devices()
    assert healthy == ("good",)
    assert solver_scan.healthy_parallel_device_count() == 1
    solver_scan._healthy_parallel_devices_cached.cache_clear()


def test_solve_scan_worker_sets_gpu_environment(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)

    config_stub = SimpleNamespace(enable_x64=lambda flag: None)
    grids_stub = SimpleNamespace(GridSpec=GridSpec)

    def fake_scan(surface, grid, nu_hat, *, epsi_hat=None, er_hat=None):
        size = len(nu_hat)
        base = np.arange(size, dtype=float) + 1.0
        return {key: base for key in ("D11", "D31", "D13", "D33", "D33_spitzer")}

    solver_stub = SimpleNamespace(solve_monoenergetic_scan=fake_scan)
    monkeypatch.setitem(sys.modules, "ntx.config", config_stub)
    monkeypatch.setitem(sys.modules, "ntx.grids", grids_stub)
    monkeypatch.setitem(sys.modules, "ntx.solver", solver_stub)

    task = {
        "backend": "gpu",
        "worker_id": 7,
        "surface": _surface_to_payload(example_surface()),
        "grid": _grid_to_payload(GridSpec(5, 5, 4)),
        "nu": np.asarray([1e-3, 2e-3]),
        "epsi": np.asarray([0.0, 1e-4]),
    }
    result = _solve_scan_worker(task)
    assert set(result) == {"D11", "D31", "D13", "D33", "D33_spitzer"}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert os.environ["JAX_PLATFORM_NAME"] == "gpu"
