from __future__ import annotations

import builtins
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from rich.console import Console

from ntx import inputfiles, io, neopax
from ntx.booz import _packed_surface_grid, load_boozmn_surface
from ntx.geometry import BoozerSurface, VmecSurface, example_surface
from ntx.grids import GridSpec
from ntx.neopax import build_ntx_neopax_scan, build_ntx_neopax_scan_from_surfaces
from ntx.solver import MonoenergeticCase
from ntx.vmec import (
    _interp_mode_columns,
    _interpolated_value,
    _iota_grid_from_wout,
    _mode_major,
    _resolve_psi_n,
    _select_mode_set,
    load_vmec_surface,
)
from ntx.vmec_jax_vmec import _interp_mode_columns as _vmec_jax_interp_mode_columns
from ntx.vmec_jax_vmec import surface_from_vmec_jax_vmec_wout


def _import_blocker(name: str):
    original_import = builtins.__import__

    def blocker(module_name, globals=None, locals=None, fromlist=(), level=0):
        if module_name == name or module_name.startswith(f"{name}."):
            raise ModuleNotFoundError(name)
        return original_import(module_name, globals, locals, fromlist, level)

    return blocker


class _FakeBoozXform:
    def __init__(self, bmnc_b, *, s_in=(0.0, 0.5, 1.0)):
        self.xm_b = np.asarray([0, 1], dtype=np.int32)
        self.xn_b = np.asarray([0, 2], dtype=np.int32)
        self.bmnc_b = np.asarray(bmnc_b, dtype=np.float64)
        self.iota = np.asarray([0.4, 0.5, 0.6], dtype=np.float64)
        self.Boozer_I_all = np.asarray([0.2, 0.3, 0.4], dtype=np.float64)
        self.Boozer_G_all = np.asarray([1.0, 1.1, 1.2], dtype=np.float64)
        self.s_in = np.asarray(s_in, dtype=np.float64)
        self.phi = None
        self.nfp = np.asarray(2, dtype=np.int32)
        self.verbose = 0

    def read_boozmn(self, _path: str) -> None:
        return None


def _fake_booz_module(fake_class):
    module = ModuleType("booz_xform_jax")
    module.Booz_xform = fake_class
    return module


def _fake_vmec_module(wout):
    vmec_api = ModuleType("vmec_jax.api")
    vmec_api.read_wout = lambda path: wout
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.read_wout = vmec_api.read_wout
    vmec_pkg.api = vmec_api
    return vmec_pkg, vmec_api


def _base_wout(**overrides):
    values = dict(
        lasym=False,
        nfp=2,
        ns=3,
        mpol=3,
        ntor=1,
        phi=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        iotaf=np.asarray([0.6, 0.65, 0.7], dtype=np.float64),
        xm=np.asarray([0, 1], dtype=np.int32),
        xn=np.asarray([0, 2], dtype=np.int32),
        xm_nyq=np.asarray([0, 1], dtype=np.int32),
        xn_nyq=np.asarray([0, 2], dtype=np.int32),
        bmnc=np.asarray([[1.0, 0.1], [1.1, 0.1], [1.2, 0.1]], dtype=np.float64),
        gmnc=np.asarray([[1.0, 0.0], [1.1, 0.0], [1.2, 0.0]], dtype=np.float64),
        bsubumnc=np.asarray([[0.2, 0.0], [0.21, 0.0], [0.22, 0.0]], dtype=np.float64),
        bsubvmnc=np.asarray([[1.0, 0.0], [1.01, 0.0], [1.02, 0.0]], dtype=np.float64),
        bsupumnc=np.asarray([[0.3, 0.0], [0.31, 0.0], [0.32, 0.0]], dtype=np.float64),
        bsupvmnc=np.asarray([[1.1, 0.0], [1.11, 0.0], [1.12, 0.0]], dtype=np.float64),
        Aminor_p=np.asarray(2.0, dtype=np.float64),
        signgs=np.asarray(1, dtype=np.int32),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gridspec_and_geometry_validation_errors():
    with pytest.raises(ValueError, match="at least 3"):
        GridSpec(2, 5, 4)
    with pytest.raises(ValueError, match="at least 2"):
        GridSpec(5, 5, 1)
    with pytest.raises(ValueError, match="same length"):
        BoozerSurface(
            m=jnp.asarray([0, 1]),
            n=jnp.asarray([0]),
            b_cos=jnp.asarray([1.0, 0.1]),
            nfp=1,
            iota=0.5,
            psi_p=1.0,
            b_theta=0.1,
            b_zeta=1.0,
        )
    with pytest.raises(ValueError, match="same length as m"):
        VmecSurface(
            path=Path("dummy"),
            requested_psi_n=0.25,
            psi_n=0.25,
            nfp=1,
            ns=3,
            mpol=2,
            ntor=1,
            total_mode_count=1,
            loaded_mode_count=1,
            iota=0.4,
            m=jnp.asarray([0]),
            n=jnp.asarray([0]),
            b_cos=jnp.asarray([1.0]),
            jacobian_cos=jnp.asarray([1.0, 2.0]),
            b_sub_theta_cos=jnp.asarray([0.1]),
            b_sub_zeta_cos=jnp.asarray([1.0]),
            b_sup_theta_cos=jnp.asarray([0.2]),
            b_sup_zeta_cos=jnp.asarray([1.1]),
            b0=1.0,
            psi_a_hat=1.0,
            phi_edge=2.0,
            r_n=0.5,
            r_hat=1.0,
            dpsi_hat_dr_hat=1.0,
            dr_hat_dpsi_hat=1.0,
            aminor_p=2.0,
        )


def test_booz_loader_error_branches(monkeypatch, tmp_path):
    fixture = tmp_path / "surface.nc"
    fixture.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_boozmn_surface(fixture, s=0.25, rho=0.5)

    monkeypatch.setattr(builtins, "__import__", _import_blocker("booz_xform_jax"))
    with pytest.raises(ModuleNotFoundError, match="booz_xform_jax"):
        load_boozmn_surface(fixture, s=0.25)
    monkeypatch.undo()

    monkeypatch.setitem(
        sys.modules,
        "booz_xform_jax",
        _fake_booz_module(lambda: _FakeBoozXform([1.0, 2.0])),
    )
    with pytest.raises(ValueError, match="2D"):
        load_boozmn_surface(fixture, s=0.25)

    monkeypatch.setitem(
        sys.modules,
        "booz_xform_jax",
        _fake_booz_module(lambda: _FakeBoozXform([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])),
    )
    with pytest.raises(ValueError, match="is zero"):
        load_boozmn_surface(fixture, s=0.25)


def test_packed_surface_grid_error_branches(monkeypatch, tmp_path):
    fixture = tmp_path / "packed.nc"
    fixture.write_text("", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "netCDF4", raising=False)
    monkeypatch.setattr(builtins, "__import__", _import_blocker("netCDF4"))
    with pytest.raises(ValueError, match="netCDF4"):
        _packed_surface_grid(fixture, 2)
    monkeypatch.undo()


def test_packed_surface_grid_success_and_shape_branches(monkeypatch, tmp_path):
    fixture = tmp_path / "packed.nc"
    fixture.write_text("", encoding="utf-8")

    class _FakeDataset:
        def __init__(self, _path, mode="r"):
            assert mode == "r"
            self.variables = {
                "jlist": np.asarray([2, 4], dtype=np.int64),
                "buco_b": np.zeros((5,), dtype=np.float64),
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    module = ModuleType("netCDF4")
    module.Dataset = _FakeDataset
    monkeypatch.setitem(sys.modules, "netCDF4", module)
    grid = _packed_surface_grid(fixture, 2)
    assert np.allclose(grid, np.asarray([0.125, 0.625]))


def test_packed_surface_grid_ns_b_and_error_paths(monkeypatch, tmp_path):
    fixture = tmp_path / "packed.nc"
    fixture.write_text("", encoding="utf-8")

    def _dataset_factory(variables):
        class _FakeDataset:
            def __init__(self, _path, mode="r"):
                assert mode == "r"
                self.variables = variables

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _FakeDataset

    module = ModuleType("netCDF4")

    module.Dataset = _dataset_factory(
        {
            "jlist": np.asarray([2, 3], dtype=np.int64),
            "ns_b": np.asarray([4], dtype=np.int64),
        }
    )
    monkeypatch.setitem(sys.modules, "netCDF4", module)
    grid = _packed_surface_grid(fixture, 2)
    assert np.allclose(grid, np.asarray([1.0 / 6.0, 0.5]))

    module.Dataset = _dataset_factory({})
    with pytest.raises(ValueError, match="radial grid length does not match"):
        _packed_surface_grid(fixture, 2)

    module.Dataset = _dataset_factory({"jlist": np.asarray([1], dtype=np.int64)})
    with pytest.raises(ValueError, match="metadata does not match"):
        _packed_surface_grid(fixture, 2)

    module.Dataset = _dataset_factory({"jlist": np.asarray([1, 2], dtype=np.int64)})
    with pytest.raises(ValueError, match="unable to determine"):
        _packed_surface_grid(fixture, 2)

    module.Dataset = _dataset_factory(
        {"jlist": np.asarray([1, 1], dtype=np.int64), "ns_b": np.asarray([1], dtype=np.int64)}
    )
    with pytest.raises(ValueError, match="at least 2"):
        _packed_surface_grid(fixture, 2)


def test_booz_loader_surface_index_and_packed_grid_success(monkeypatch, tmp_path):
    fixture = tmp_path / "surface.nc"
    fixture.write_text("", encoding="utf-8")

    class _PackedFakeBoozXform(_FakeBoozXform):
        def __init__(self):
            super().__init__(
                [[1.0, 0.20], [1.5, 0.10]],
                s_in=(0.0, 0.5, 1.0),
            )
            self.phi = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
            self.iota = np.asarray([0.4, 0.6, 0.8], dtype=np.float64)
            self.Boozer_I_all = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
            self.Boozer_G_all = np.asarray([0.1, 0.1, 0.1], dtype=np.float64)

    monkeypatch.setitem(
        sys.modules,
        "booz_xform_jax",
        _fake_booz_module(_PackedFakeBoozXform),
    )

    called = {}

    def _fake_packed_grid(path, ns_b):
        called["path"] = path
        called["ns_b"] = ns_b
        return np.asarray([0.0, 0.5], dtype=np.float64)

    monkeypatch.setattr("ntx.booz._packed_surface_grid", _fake_packed_grid)

    payload = load_boozmn_surface(fixture, surface_index=1, min_bmn_to_load=0.75)
    assert called["path"] == fixture.resolve()
    assert called["ns_b"] == 2
    assert payload.surface_index == 1
    assert payload.s == pytest.approx(0.5)
    assert payload.rho == pytest.approx(np.sqrt(0.5))
    assert payload.mode_count == 1
    assert payload.surface.iota == pytest.approx(-0.6)
    assert payload.surface.b_theta == pytest.approx(-0.5)
    assert payload.surface.b_zeta == pytest.approx(0.1)
    assert payload.surface.n.tolist() == [0]


def test_booz_loader_surface_index_out_of_range(monkeypatch, tmp_path):
    fixture = tmp_path / "surface.nc"
    fixture.write_text("", encoding="utf-8")
    class _SurfaceIndexBoozXform(_FakeBoozXform):
        def __init__(self):
            super().__init__([[1.0, 0.2], [1.1, 0.1]], s_in=(0.0, 1.0))
            self.iota = np.asarray([0.4, 0.6], dtype=np.float64)
            self.Boozer_I_all = np.asarray([0.2, 0.3], dtype=np.float64)
            self.Boozer_G_all = np.asarray([1.0, 1.1], dtype=np.float64)

    monkeypatch.setitem(
        sys.modules,
        "booz_xform_jax",
        _fake_booz_module(_SurfaceIndexBoozXform),
    )

    with pytest.raises(IndexError):
        load_boozmn_surface(fixture, surface_index=5)


def test_vmec_helper_error_branches():
    with pytest.raises(ValueError, match="2D"):
        _mode_major(np.ones((3,)))
    with pytest.raises(ValueError, match="does not provide an iota profile"):
        _iota_grid_from_wout(SimpleNamespace())
    with pytest.raises(ValueError, match="between 0 and 1"):
        _resolve_psi_n(np.asarray([0.0, 0.5, 1.0]), 2.0, 0)
    with pytest.raises(ValueError, match="0, 1, or 2"):
        _resolve_psi_n(np.asarray([0.0, 0.5, 1.0]), 0.5, 7)
    assert _resolve_psi_n(np.asarray([0.0, 0.5, 1.0]), 0.6, 2) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="2D"):
        _interp_mode_columns(np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0]), 0.5)
    with pytest.raises(ValueError, match="1D"):
        _interpolated_value(np.ones((2, 2)), np.ones((2,)), 0.5, order=2)
    with pytest.raises(ValueError, match="same length"):
        _interpolated_value(np.asarray([0.0, 1.0]), np.asarray([1.0]), 0.5, order=2)
    with pytest.raises(ValueError, match="at least one node"):
        _interpolated_value(np.asarray([]), np.asarray([]), 0.5, order=2)
    assert _interpolated_value(
        np.asarray([0.0, 1.0]),
        np.asarray([2.0, 4.0]),
        0.25,
        order=2,
    ) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="smaller than the reduced mode table"):
        _select_mode_set(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 2], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            nfp=2,
            mpol=3,
            ntor=1,
            option=1,
            mode_convention="reduced",
        )
    with pytest.raises(ValueError, match="filtered_nyquist"):
        _select_mode_set(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 2], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 2], dtype=np.int32),
            nfp=2,
            mpol=3,
            ntor=1,
            option=1,
            mode_convention="invalid",
        )
    with pytest.raises(ValueError, match="must be 1 or 2"):
        _select_mode_set(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 2], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 2], dtype=np.int32),
            nfp=2,
            mpol=3,
            ntor=1,
            option=7,
            mode_convention="reduced",
        )


def test_vmec_loader_error_branches(monkeypatch, tmp_path):
    missing = tmp_path / "missing.nc"
    with pytest.raises(FileNotFoundError):
        load_vmec_surface(missing, psi_n=0.25)

    fixture = tmp_path / "wout.nc"
    fixture.write_text("", encoding="utf-8")
    monkeypatch.setattr(builtins, "__import__", _import_blocker("vmec_jax"))
    with pytest.raises(ModuleNotFoundError, match="vmec_jax"):
        load_vmec_surface(fixture, psi_n=0.25)
    monkeypatch.undo()

    for override, psi_n_value, match in (
        ({"lasym": True}, 0.25, "lasym=true"),
        ({"ns": 1}, 0.25, "at least two radial surfaces"),
        ({}, 0.0, "requires surface.psi_n > 0"),
        ({"Aminor_p": np.asarray(0.0)}, 0.25, "nonzero Aminor_p"),
    ):
        pkg, api = _fake_vmec_module(_base_wout(**override))
        monkeypatch.setitem(sys.modules, "vmec_jax", pkg)
        monkeypatch.setitem(sys.modules, "vmec_jax.api", api)
        expected = NotImplementedError if match == "lasym=true" else ValueError
        with pytest.raises(expected, match=match):
            load_vmec_surface(fixture, psi_n=psi_n_value)


def test_vmec_loader_covers_mode_and_transport_error_branches(monkeypatch, tmp_path):
    fixture = tmp_path / "wout.nc"
    fixture.write_text("", encoding="utf-8")
    pkg, api = _fake_vmec_module(_base_wout())
    monkeypatch.setitem(sys.modules, "vmec_jax", pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", api)

    original_interp = load_vmec_surface.__globals__["_interp_mode_columns"]
    original_select = load_vmec_surface.__globals__["_select_mode_set"]
    original_resolve = load_vmec_surface.__globals__["_resolve_psi_n"]

    try:
        def _bad_len_interp(x, values, xq):
            if getattr(_bad_len_interp, "calls", 0) == 0:
                _bad_len_interp.calls = 1
                return np.asarray([1.0], dtype=np.float64)
            return original_interp(x, values, xq)

        _bad_len_interp.calls = 0
        monkeypatch.setitem(load_vmec_surface.__globals__, "_interp_mode_columns", _bad_len_interp)
        with pytest.raises(ValueError, match="mode-number arrays do not match"):
            load_vmec_surface(fixture, psi_n=0.25)

        monkeypatch.setitem(load_vmec_surface.__globals__, "_interp_mode_columns", original_interp)

        def _bad_select(*args, **kwargs):
            return (
                np.asarray([1, 2], dtype=np.int32),
                np.asarray([0, 2], dtype=np.int32),
                np.asarray([0, 1], dtype=np.int32),
            )

        monkeypatch.setitem(load_vmec_surface.__globals__, "_select_mode_set", _bad_select)
        with pytest.raises(ValueError, match="first VMEC mode"):
            load_vmec_surface(fixture, psi_n=0.25)

        monkeypatch.setitem(load_vmec_surface.__globals__, "_select_mode_set", original_select)

        def _zero_b0_interp(x, values, xq):
            if values.shape[0] == 2:
                return np.asarray([0.0, 0.1], dtype=np.float64)
            return original_interp(x, values, xq)

        monkeypatch.setitem(load_vmec_surface.__globals__, "_interp_mode_columns", _zero_b0_interp)
        with pytest.raises(ValueError, match="zero magnetic-field strength"):
            load_vmec_surface(fixture, psi_n=0.25)

        monkeypatch.setitem(load_vmec_surface.__globals__, "_interp_mode_columns", original_interp)
        zero_phi_pkg, zero_phi_api = _fake_vmec_module(
            _base_wout(phi=np.asarray([0.0, 0.0, 0.0], dtype=np.float64))
        )
        monkeypatch.setitem(sys.modules, "vmec_jax", zero_phi_pkg)
        monkeypatch.setitem(sys.modules, "vmec_jax.api", zero_phi_api)
        monkeypatch.setitem(
            load_vmec_surface.__globals__,
            "_resolve_psi_n",
            lambda *args, **kwargs: 0.25,
        )
        with pytest.raises(ValueError, match="dpsi_hat/dr_hat = 0"):
            load_vmec_surface(fixture, psi_n=0.25)
    finally:
        monkeypatch.setitem(load_vmec_surface.__globals__, "_interp_mode_columns", original_interp)
        monkeypatch.setitem(load_vmec_surface.__globals__, "_select_mode_set", original_select)
        monkeypatch.setitem(load_vmec_surface.__globals__, "_resolve_psi_n", original_resolve)


def test_vmec_jax_vmec_error_branches():
    with pytest.raises(ValueError, match="s must be between 0 and 1"):
        surface_from_vmec_jax_vmec_wout(_base_wout(), s=2.0)
    with pytest.raises(ValueError, match="at least two radial surfaces"):
        surface_from_vmec_jax_vmec_wout(_base_wout(ns=1), s=0.25)
    with pytest.raises(ValueError, match="2D"):
        _vmec_jax_interp_mode_columns(np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0]), 0.5)
    with pytest.raises(ValueError, match="zero magnetic-field strength"):
        surface_from_vmec_jax_vmec_wout(
            _base_wout(bmnc=np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)),
            s=0.25,
        )


def test_neopax_shape_and_hdf5_branches(tmp_path):
    surface = example_surface()
    with pytest.raises(ValueError, match="same shape"):
        build_ntx_neopax_scan(
            lambda rho: surface,
            rho=jnp.asarray([0.25]),
            nu_v=jnp.asarray([1.0e-3]),
            Es=jnp.asarray([[0.0], [1.0e-3]]),
            Er=jnp.asarray([[0.0]]),
            drds=jnp.asarray([1.0]),
            grid=GridSpec(5, 5, 4),
        )
    with pytest.raises(ValueError, match="number of surfaces"):
        build_ntx_neopax_scan_from_surfaces(
            (surface,),
            rho=jnp.asarray([0.25, 0.5]),
            nu_v=jnp.asarray([1.0e-3]),
            Es=jnp.asarray([[0.0]]),
            Er=jnp.asarray([[0.0]]),
            drds=jnp.asarray([1.0, 1.1]),
            grid=GridSpec(5, 5, 4),
        )

    scan = build_ntx_neopax_scan_from_surfaces(
        (surface,),
        rho=jnp.asarray([0.25]),
        nu_v=jnp.asarray([1.0e-3]),
        Es=jnp.asarray([[0.0]]),
        Er=jnp.asarray([[0.0]]),
        drds=jnp.asarray([1.0]),
        grid=GridSpec(5, 5, 4),
        source_name="scan",
    )
    path = tmp_path / "scan.h5"
    neopax.write_neopax_scan_hdf5(scan, path)
    with h5py.File(path, "r") as handle:
        assert "source_name" in handle.attrs


def test_solver_inputfile_and_cli_error_branches(monkeypatch, tmp_path, capsys):
    with pytest.raises(ValueError, match="set only one"):
        MonoenergeticCase(1.0e-3, epsi_hat=0.0, er_hat=1.0e-3).resolved_epsi_hat(1.0)
    with pytest.raises(ValueError, match="transport normalization"):
        MonoenergeticCase(1.0e-3, er_hat=1.0e-3).resolved_epsi_hat(None)

    input_path = tmp_path / "run.toml"
    input_path.write_text(
        "[surface]\ntype='vmec'\n[grid]\nn_theta=5\nn_zeta=5\nn_xi=4\n[case]\nnu_hat=1e-3\n[output]\nnpz='result.npz'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="surface.path is required"):
        inputfiles.load_run_config(input_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python",
            "solve",
            "--example",
            "--nu-hat",
            "1e-2",
            "--n-theta",
            "5",
            "--n-zeta",
            "5",
            "--n-xi",
            "4",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("ntx.cli", run_name="__main__")
    assert excinfo.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["D11"] > 0.0


def test_io_and_case_table_branches(tmp_path):
    missing_b00 = tmp_path / "surface.ddkes2.data"
    missing_b00.write_text(
        "&datain\nnzperiod=1\npsip=1.0\nchip=-0.5\nbtheta=0.1\nbzeta=1.0\n/\nborbi(0,1)=0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing borbi"):
        io.load_dkes_surface(missing_b00)

    config = inputfiles.RunConfig(
        input_path=Path("run.toml"),
        surface=inputfiles.SurfaceSpec(type="vmec", path=Path("wout.nc"), psi_n=0.25),
        grid=GridSpec(5, 5, 4),
        case=MonoenergeticCase(1.0e-3, epsi_hat=0.0, er_hat=1.0e-3),
        output=inputfiles.OutputSpec(npz=Path("result.npz"), include_modes=True),
        verbose=False,
    )
    surface = VmecSurface(
        path=Path("wout.nc"),
        requested_psi_n=0.25,
        psi_n=0.25,
        nfp=1,
        ns=3,
        mpol=2,
        ntor=1,
        total_mode_count=1,
        loaded_mode_count=1,
        iota=0.4,
        m=jnp.asarray([0]),
        n=jnp.asarray([0]),
        b_cos=jnp.asarray([1.0]),
        jacobian_cos=jnp.asarray([1.0]),
        b_sub_theta_cos=jnp.asarray([0.1]),
        b_sub_zeta_cos=jnp.asarray([1.0]),
        b_sup_theta_cos=jnp.asarray([0.2]),
        b_sup_zeta_cos=jnp.asarray([1.1]),
        b0=1.0,
        psi_a_hat=1.0,
        phi_edge=2.0,
        r_n=0.5,
        r_hat=1.0,
        dpsi_hat_dr_hat=1.2,
        dr_hat_dpsi_hat=0.8333333333333,
        aminor_p=2.0,
    )
    table = inputfiles._case_table(config, surface)
    console = Console(record=True, width=120)
    console.print(table)
    rendered = console.export_text()
    assert "requires transport normalization" in rendered
