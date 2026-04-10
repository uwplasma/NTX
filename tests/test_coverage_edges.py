from __future__ import annotations

import builtins
import dataclasses
import importlib
import json
import runpy
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from rich.console import Console

import ntx
from ntx import (
    __version__,
    benchmarks,
    cli,
    geometry,
    inputfiles,
    io,
    neopax,
    solver,
)
from ntx import (
    operators as operators_mod,
)
from ntx import (
    sfincs_geometry as sfincs_geometry_mod,
)
from ntx import (
    vmec_jax_backend as vmec_jax_backend_mod,
)
from ntx import (
    vmec_reference as vmec_reference_mod,
)
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
from ntx.vmec_reference import _interp_mode_columns as _reference_interp_mode_columns
from ntx.vmec_reference import _mode_index, load_vmec_surface_reference


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
        self.xn_b = np.asarray([0, 5], dtype=np.int32)
        self.bmnc_b = np.asarray(bmnc_b, dtype=np.float64)
        self.iota = np.asarray([0.4, 0.5, 0.6], dtype=np.float64)
        self.Boozer_I_all = np.asarray([0.2, 0.3, 0.4], dtype=np.float64)
        self.Boozer_G_all = np.asarray([1.0, 1.1, 1.2], dtype=np.float64)
        self.s_in = np.asarray(s_in, dtype=np.float64)
        self.phi = None
        self.nfp = np.asarray(5, dtype=np.int32)
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
    vmec_pkg.api = vmec_api
    return vmec_pkg, vmec_api


def _base_wout(**overrides):
    values = dict(
        lasym=False,
        nfp=5,
        ns=3,
        mpol=3,
        ntor=2,
        phi=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        iotaf=np.asarray([0.6, 0.65, 0.7], dtype=np.float64),
        xm=np.asarray([0, 1], dtype=np.int32),
        xn=np.asarray([0, 5], dtype=np.int32),
        xm_nyq=np.asarray([0, 1], dtype=np.int32),
        xn_nyq=np.asarray([0, 5], dtype=np.int32),
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


def test_package_version_fallback(monkeypatch):
    import importlib.metadata as metadata

    monkeypatch.setattr(
        metadata,
        "version",
        lambda name: (_ for _ in ()).throw(metadata.PackageNotFoundError()),
    )
    reloaded = importlib.reload(ntx)
    assert reloaded.__version__ == "0.1.0"
    importlib.reload(ntx)
    assert __version__


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


def test_evaluate_fourier_series_with_sine_coefficients():
    value, dtheta, dzeta = geometry.evaluate_fourier_series(
        jnp.asarray([1]),
        jnp.asarray([1]),
        jnp.asarray([0.5]),
        jnp.asarray(0.1),
        jnp.asarray(0.2),
        nfp=2,
        sin_coeffs=jnp.asarray([0.25]),
    )
    assert jnp.isfinite(value)
    assert jnp.isfinite(dtheta)
    assert jnp.isfinite(dzeta)


def test_checkout_path_none_branches(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    repo = workspace / "NTX"
    repo.mkdir(parents=True)
    monkeypatch.setattr(ntx._checkout_paths, "repo_root", lambda: repo)
    for env_name in (
        "NTX_REFERENCE_PYTHON_ROOT",
        "NTX_REFERENCE_EXECUTABLE_ROOT",
        "NTX_REFERENCE_EXECUTABLE",
        "VMEC_JAX_ROOT",
    ):
        monkeypatch.delenv(env_name, raising=False)
    assert ntx._checkout_paths.find_reference_python_root() is None
    assert ntx._checkout_paths.find_reference_executable_root() is None
    assert ntx._checkout_paths.find_reference_executable() is None
    assert ntx._checkout_paths.find_vmec_jax_example_input() is None


def test_checkout_path_additional_env_branches(monkeypatch, tmp_path):
    reference_root = tmp_path / "reference-root"
    reference_root.mkdir()
    omnigenity_root = tmp_path / "omnigenity-root"
    omnigenity_root.mkdir()
    monkeypatch.setenv("NTX_REFERENCE_EXECUTABLE_ROOT", str(reference_root))
    monkeypatch.setenv("OMNIGENITY_OPTIMIZATION_ROOT", str(omnigenity_root))
    assert ntx._checkout_paths.find_reference_executable_root() == reference_root.resolve()
    assert ntx._checkout_paths.find_omnigenity_optimization_root() == omnigenity_root.resolve()


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

    monkeypatch.setitem(
        sys.modules,
        "booz_xform_jax",
        _fake_booz_module(lambda: _FakeBoozXform([[1.0, 1.1, 1.2], [0.1, 0.1, 0.1]])),
    )
    with pytest.raises(IndexError, match="outside"):
        load_boozmn_surface(fixture, surface_index=-1)
    payload = load_boozmn_surface(fixture, surface_index=0)
    assert payload.surface_index == 0


def test_packed_surface_grid_error_branches(monkeypatch, tmp_path):
    fixture = tmp_path / "packed.nc"
    fixture.write_text("", encoding="utf-8")

    monkeypatch.delitem(sys.modules, "netCDF4", raising=False)
    monkeypatch.setattr(builtins, "__import__", _import_blocker("netCDF4"))
    with pytest.raises(ValueError, match="netCDF4"):
        _packed_surface_grid(fixture, 2)
    monkeypatch.undo()

    class FakeHandle:
        def __init__(self, variables):
            self.variables = variables

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def dataset_from(variables):
        return lambda path, mode="r": FakeHandle(variables)

    fake_netcdf = ModuleType("netCDF4")
    fake_netcdf.Dataset = dataset_from({})
    monkeypatch.setitem(sys.modules, "netCDF4", fake_netcdf)
    with pytest.raises(ValueError, match="does not match"):
        _packed_surface_grid(fixture, 2)

    fake_netcdf.Dataset = dataset_from({"jlist": np.asarray([1, 2]), "buco_b": np.zeros((3,))})
    with pytest.raises(ValueError, match="does not match"):
        _packed_surface_grid(fixture, 3)

    fake_netcdf.Dataset = dataset_from({"jlist": np.asarray([1, 2])})
    with pytest.raises(ValueError, match="determine packed Boozer radial resolution"):
        _packed_surface_grid(fixture, 2)

    fake_netcdf.Dataset = dataset_from({"jlist": np.asarray([1]), "ns_b": np.asarray([1])})
    with pytest.raises(ValueError, match="at least 2"):
        _packed_surface_grid(fixture, 1)


def test_vmec_helper_error_branches():
    with pytest.raises(ValueError, match="2D"):
        _mode_major(np.ones((3,)))
    with pytest.raises(ValueError, match="does not provide an iota profile"):
        _iota_grid_from_wout(SimpleNamespace())
    assert _resolve_psi_n(np.asarray([0.0, 0.5, 1.0]), 0.25, 2) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        _resolve_psi_n(np.asarray([0.0, 0.5, 1.0]), 2.0, 0)
    with pytest.raises(ValueError, match="0, 1, or 2"):
        _resolve_psi_n(np.asarray([0.0, 0.5, 1.0]), 0.5, 7)
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
            np.asarray([0, 5], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            nfp=5,
            mpol=3,
            ntor=2,
            option=1,
            mode_convention="reduced",
        )
    with pytest.raises(ValueError, match="filtered_nyquist"):
        _select_mode_set(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 5], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 5], dtype=np.int32),
            nfp=5,
            mpol=3,
            ntor=2,
            option=1,
            mode_convention="bad",
        )
    with pytest.raises(ValueError, match="1 or 2"):
        _select_mode_set(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 5], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0, 5], dtype=np.int32),
            nfp=5,
            mpol=3,
            ntor=2,
            option=9,
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

    wout = _base_wout(lasym=True)
    pkg, api = _fake_vmec_module(wout)
    monkeypatch.setitem(sys.modules, "vmec_jax", pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", api)
    with pytest.raises(NotImplementedError, match="lasym=true"):
        load_vmec_surface(fixture, psi_n=0.25)

    for override, psi_n_value, match in (
        ({"ns": 1}, 0.25, "at least two radial surfaces"),
        ({}, 0.0, "requires surface.psi_n > 0"),
        ({"Aminor_p": np.asarray(0.0)}, 0.25, "nonzero Aminor_p"),
    ):
        pkg, api = _fake_vmec_module(_base_wout(**override))
        monkeypatch.setitem(sys.modules, "vmec_jax", pkg)
        monkeypatch.setitem(sys.modules, "vmec_jax.api", api)
        with pytest.raises(ValueError, match=match):
            load_vmec_surface(fixture, psi_n=psi_n_value)

    pkg, api = _fake_vmec_module(_base_wout(phi=np.asarray([0.0, 0.5, 0.0])))
    monkeypatch.setitem(sys.modules, "vmec_jax", pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", api)
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="dpsi_hat/dr_hat = 0"):
            load_vmec_surface(fixture, psi_n=0.25)

    monkeypatch.setattr(
        "ntx.vmec._select_mode_set",
        lambda *args, **kwargs: (np.asarray([0, 1]), np.asarray([0, 5]), np.asarray([0, 1])),
    )
    monkeypatch.setattr("ntx.vmec._interp_mode_columns", lambda *args, **kwargs: np.asarray([1.0]))
    pkg, api = _fake_vmec_module(_base_wout())
    monkeypatch.setitem(sys.modules, "vmec_jax", pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", api)
    with pytest.raises(ValueError, match="do not match"):
        load_vmec_surface(fixture, psi_n=0.25)


def test_vmec_loader_more_error_branches(monkeypatch, tmp_path):
    fixture = tmp_path / "wout.nc"
    fixture.write_text("", encoding="utf-8")
    pkg, api = _fake_vmec_module(_base_wout())
    monkeypatch.setitem(sys.modules, "vmec_jax", pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", api)

    monkeypatch.setattr(
        "ntx.vmec._select_mode_set",
        lambda *args, **kwargs: (np.asarray([1]), np.asarray([0]), np.asarray([0])),
    )
    with pytest.raises(ValueError, match="expected the first VMEC mode"):
        load_vmec_surface(fixture, psi_n=0.25)

    monkeypatch.setattr(
        "ntx.vmec._select_mode_set",
        lambda *args, **kwargs: (np.asarray([0]), np.asarray([0]), np.asarray([0])),
    )
    monkeypatch.setattr("ntx.vmec._interp_mode_columns", lambda *args, **kwargs: np.asarray([0.0]))
    with pytest.raises(ValueError, match="zero magnetic-field strength"):
        load_vmec_surface(fixture, psi_n=0.25)


def test_reference_and_vmec_jax_helper_errors(tmp_path):
    missing = tmp_path / "missing.nc"
    with pytest.raises(FileNotFoundError):
        load_vmec_surface_reference(missing, s=0.25)
    with pytest.raises(ValueError, match="s must be between 0 and 1"):
        surface_from_vmec_jax_vmec_wout(_base_wout(), s=2.0)
    with pytest.raises(ValueError, match="at least two radial surfaces"):
        surface_from_vmec_jax_vmec_wout(_base_wout(ns=1), s=0.25)
    with pytest.raises(ValueError, match="2D"):
        _reference_interp_mode_columns(np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0]), 0.5)
    with pytest.raises(ValueError, match="2D"):
        _vmec_jax_interp_mode_columns(np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0]), 0.5)
    with pytest.raises(ValueError, match="not found"):
        _mode_index(np.asarray([0, 1]), np.asarray([0, 1]), 3, 3)


def test_reference_and_scan_benchmark_error_branches(tmp_path):
    table = np.zeros(1, dtype=[("nu_hat", float), ("er_hat", float), ("D11", float)])
    table["nu_hat"] = 1.0e-5
    table["er_hat"] = 0.0
    with pytest.raises(ValueError, match="no monoenergetic row"):
        benchmarks.select_monoenergetic_row(table, nu_hat=1.0e-4, er_hat=0.0)
    assert benchmarks.relative_error(1.0, 0.0) > 0.0

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

    scan_with_attrs = dataclasses.replace(scan, a_b=1.5, psia=2.0)
    attr_path = tmp_path / "scan-attrs.h5"
    neopax.write_neopax_scan_hdf5(scan_with_attrs, attr_path)
    with h5py.File(attr_path, "r") as handle:
        assert handle.attrs["a_b"] == pytest.approx(1.5)
        assert handle.attrs["psia"] == pytest.approx(2.0)


def test_solver_and_inputfile_error_branches(monkeypatch, tmp_path):
    case = MonoenergeticCase(1.0e-3, epsi_hat=0.0, er_hat=1.0e-3)
    with pytest.raises(ValueError, match="set only one"):
        case.resolved_epsi_hat(1.0)
    with pytest.raises(ValueError, match="transport normalization"):
        MonoenergeticCase(1.0e-3, er_hat=1.0e-3).resolved_epsi_hat(None)

    original_prepare = solver.prepare_monoenergetic_system
    fake_prepared = SimpleNamespace(geometry=SimpleNamespace(transport_psi_scale=None))
    monkeypatch.setattr(solver, "prepare_monoenergetic_system", lambda surface, grid: fake_prepared)
    with pytest.raises(ValueError, match="set only one"):
        solver.solve_monoenergetic_scan(
            example_surface(),
            GridSpec(5, 5, 4),
            jnp.asarray([1e-3]),
            epsi_hat=0.0,
            er_hat=0.0,
        )
    with pytest.raises(ValueError, match="transport normalization"):
        solver.solve_monoenergetic_scan(
            example_surface(),
            GridSpec(5, 5, 4),
            jnp.asarray([1e-3]),
            er_hat=0.0,
        )
    monkeypatch.setattr(solver, "prepare_monoenergetic_system", original_prepare)

    input_path = tmp_path / "run.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                "type='vmec'",
                "",
                "[grid]",
                "n_theta=5",
                "n_zeta=5",
                "n_xi=4",
                "",
                "[case]",
                "nu_hat=1e-3",
                "",
                "[output]",
                "npz='result.npz'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="surface.path is required"):
        inputfiles.load_run_config(input_path)
    assert MonoenergeticCase(1.0e-3).resolved_epsi_hat(None) == pytest.approx(0.0)
    solved = solver.solve_scan(
        example_surface(),
        GridSpec(5, 5, 4),
        (MonoenergeticCase(1.0e-3),),
    )
    assert len(solved) == 1


def test_io_error_branches(tmp_path):
    missing_b00 = tmp_path / "surface.ddkes2.data"
    missing_b00.write_text(
        "\n".join(
            [
                "&datain",
                "nzperiod = 1",
                "psip = 1.0",
                "chip = -0.5",
                "btheta = 0.1",
                "bzeta = 1.0",
                "/",
                "borbi(0,1) = 0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing borbi"):
        io.load_dkes_surface(missing_b00)

    magnetic = tmp_path / "surface.dat"
    magnetic.write_text(
        "\n".join(
            [
                "Number of periods = 5",
                "psi_p = 1",
                "chi_p = -0.5",
                "iota = 0.5",
                "B00 = 1",
                "B_theta = 0.1",
                "B_zeta = 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing Fourier-mode section"):
        io.load_magnetic_configuration_surface(magnetic)

    magnetic_rows = tmp_path / "surface_rows.dat"
    magnetic_rows.write_text(
        "\n".join(
            [
                "Number of periods = 5",
                "psi_p = 1",
                "chi_p = -0.5",
                "iota = 0.5",
                "B00 = 1",
                "B_theta = 0.1",
                "B_zeta = 1.0",
                "*** Magnetic field strength Fourier modes",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no Fourier rows found"):
        io.load_magnetic_configuration_surface(magnetic_rows)


def test_database_stack_with_rho_branch():
    first = ntx.build_monoenergetic_database_arrays(
        example_surface(),
        GridSpec(5, 5, 4),
        jnp.asarray([1e-2]),
        rho=jnp.asarray([0.25]),
    )
    second = ntx.build_monoenergetic_database_arrays(
        example_surface(),
        GridSpec(5, 5, 4),
        jnp.asarray([2e-2]),
        rho=jnp.asarray([0.5]),
    )
    stacked = ntx.stack_monoenergetic_database_arrays((first, second))
    assert stacked.rho is not None
    assert stacked.rho.shape == (2, 1)


def test_cli_additional_branches(monkeypatch, capsys, tmp_path):
    vmec_sentinel = object()
    monkeypatch.setattr(cli, "load_vmec_surface", lambda *args, **kwargs: vmec_sentinel)
    args = SimpleNamespace(
        example=False,
        dkes=None,
        vmec=tmp_path / "wout.nc",
        psi_n=0.25,
        vmec_radial_option=0,
        vmec_nyquist_option=1,
        vmec_mode_convention="reduced",
        min_bmn_to_load=0.0,
    )
    assert cli._load_surface(args) is vmec_sentinel

    original_parse_args = cli.argparse.ArgumentParser.parse_args
    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, args=None, namespace=None: SimpleNamespace(command="other"),
    )
    assert cli.main(["solve"]) == 1
    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", original_parse_args)

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("ntx.cli", run_name="__main__")
    assert excinfo.value.code == 0
    assert json.loads(capsys.readouterr().out)["D11"] > 0.0


def test_inputfile_case_table_vmec_branches():
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
    assert "dpsi_hat/dr_hat" in rendered
    assert "coefficient_psi_scale" in rendered


def test_neopax_additional_shape_errors():
    surface = example_surface()
    with pytest.raises(ValueError, match="first dimension must match rho"):
        build_ntx_neopax_scan(
            lambda rho: surface,
            rho=jnp.asarray([0.25, 0.5]),
            nu_v=jnp.asarray([1.0e-3]),
            Es=jnp.asarray([[0.0]]),
            Er=jnp.asarray([[0.0]]),
            drds=jnp.asarray([1.0, 1.1]),
            grid=GridSpec(5, 5, 4),
        )
    with pytest.raises(ValueError, match="same length as rho"):
        build_ntx_neopax_scan(
            lambda rho: surface,
            rho=jnp.asarray([0.25]),
            nu_v=jnp.asarray([1.0e-3]),
            Es=jnp.asarray([[0.0]]),
            Er=jnp.asarray([[0.0]]),
            drds=jnp.asarray([1.0, 2.0]),
            grid=GridSpec(5, 5, 4),
        )
    with pytest.raises(ValueError, match="first dimension must match rho"):
        build_ntx_neopax_scan_from_surfaces(
            (surface, surface),
            rho=jnp.asarray([0.25, 0.5]),
            nu_v=jnp.asarray([1.0e-3]),
            Es=jnp.asarray([[0.0]]),
            Er=jnp.asarray([[0.0]]),
            drds=jnp.asarray([1.0, 1.1]),
            grid=GridSpec(5, 5, 4),
        )
    with pytest.raises(ValueError, match="same length as rho"):
        build_ntx_neopax_scan_from_surfaces(
            (surface,),
            rho=jnp.asarray([0.25]),
            nu_v=jnp.asarray([1.0e-3]),
            Es=jnp.asarray([[0.0]]),
            Er=jnp.asarray([[0.0]]),
            drds=jnp.asarray([1.0, 2.0]),
            grid=GridSpec(5, 5, 4),
        )
    with pytest.raises(ValueError, match="same shape"):
        build_ntx_neopax_scan_from_surfaces(
            (surface, surface),
            rho=jnp.asarray([0.25, 0.5]),
            nu_v=jnp.asarray([1.0e-3]),
            Es=jnp.asarray([[0.0], [1.0e-3]]),
            Er=jnp.asarray([[0.0, 1.0e-3], [0.0, 1.0e-3]]),
            drds=jnp.asarray([1.0, 1.1]),
            grid=GridSpec(5, 5, 4),
        )


def test_operator_and_sfincs_error_branches(monkeypatch, tmp_path):
    d_out, u_out = operators_mod.apply_nullspace_condition(
        jnp.eye(2),
        None,
    )
    assert u_out is None
    assert d_out.shape == (2, 2)

    monkeypatch.setattr(sfincs_geometry_mod, "find_sfincs_jax_root", lambda: None)
    with pytest.raises(FileNotFoundError, match="checkout not found"):
        ntx.compare_vmec_geometry_to_sfincs(
            wout_path=Path("wout.nc"),
            psi_n=0.25,
            grid=GridSpec(5, 5, 4),
        )

    missing_repo = tmp_path / "sfincs"
    with pytest.raises(FileNotFoundError, match=str(missing_repo)):
        ntx.compare_vmec_geometry_to_sfincs(
            wout_path=Path("wout.nc"),
            psi_n=0.25,
            grid=GridSpec(5, 5, 4),
            sfincs_repo=missing_repo,
        )


def test_vmec_jax_backend_and_reference_additional_branches(monkeypatch, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class FakeCfg:
        ns: int
        mpol: int
        ntor: int

    cfg = FakeCfg(ns=5, mpol=3, ntor=2)
    indata = object()
    wout = _base_wout(ns=6, mpol=4, ntor=3)
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.load_config = lambda path: (cfg, indata)
    vmec_pkg.build_static = lambda cfg_obj: cfg_obj
    vmec_api = ModuleType("vmec_jax.api")
    vmec_api.read_wout = lambda path: wout
    vmec_api.state_from_wout = lambda w: "state"
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", vmec_api)
    captured = {}

    def fake_surface_from_state(**kwargs):
        captured.update(kwargs)
        return "surface"

    monkeypatch.setattr(
        vmec_jax_backend_mod,
        "surface_from_vmec_jax_state",
        fake_surface_from_state,
    )
    result = ntx.surface_from_vmec_jax_wout(
        input_path=tmp_path / "input.vmec",
        wout_path=tmp_path / "wout.nc",
        s=0.25,
    )
    assert result == "surface"
    assert captured["static"].mpol == 4
    assert captured["static"].ntor == 3

    with pytest.raises(ValueError, match="zero magnetic-field strength"):
        surface_from_vmec_jax_vmec_wout(_base_wout(bmnc=np.zeros((3, 2))), s=0.25)
    original_read_wout = vmec_reference_mod._read_vmec_wout_netcdf
    original_read_booz = vmec_reference_mod._read_booz_netcdf
    monkeypatch.setattr(
        vmec_reference_mod,
        "_read_vmec_wout_netcdf",
        lambda path: {
            "ns": 3,
            "nfp": 5,
            "mpol": 3,
            "ntor": 2,
            "Aminor_p": 2.0,
            "volume_p": 1.0,
            "phi": np.asarray([0.0, 0.5, 1.0]),
            "iotaf": np.asarray([0.6, 0.65, 0.7]),
            "xm_nyq": np.asarray([0, 1]),
            "xn_nyq": np.asarray([0, 5]),
            "gmnc": np.asarray([[1.0, 0.0], [1.1, 0.0], [1.2, 0.0]]),
            "bmnc": np.asarray([[1.0, 0.0], [1.1, 0.0], [1.2, 0.0]]),
            "bsupumnc": np.asarray([[0.3, 0.0], [0.31, 0.0], [0.32, 0.0]]),
            "bsupvmnc": np.asarray([[1.1, 0.0], [1.11, 0.0], [1.12, 0.0]]),
            "bsubumnc": np.asarray([[0.2, 0.0], [0.21, 0.0], [0.22, 0.0]]),
            "bsubvmnc": np.asarray([[1.0, 0.0], [1.01, 0.0], [1.02, 0.0]]),
        },
    )
    monkeypatch.setattr(
        vmec_reference_mod,
        "_read_booz_netcdf",
        lambda path: {
            "bmnc_b": np.asarray([[1.0, 0.1], [1.1, 0.1], [1.2, 0.1]]),
            "ixm_b": np.asarray([0, 1]),
            "ixn_b": np.asarray([0, 5]),
            "jlist": np.asarray([1, 2, 3]),
            "buco_b": np.asarray([0.2, 0.3, 0.4]),
            "bvco_b": np.asarray([1.0, 1.1, 1.2]),
        },
    )
    with pytest.raises(ValueError, match="rho must be a 1D array"):
        vmec_reference_mod.vmec_reference_factors(
            tmp_path / "wout.nc", tmp_path / "booz.nc", jnp.asarray([[0.25]])
        )

    monkeypatch.setattr(vmec_reference_mod, "_read_vmec_wout_netcdf", original_read_wout)
    monkeypatch.setattr(vmec_reference_mod, "_read_booz_netcdf", original_read_booz)
    monkeypatch.setattr(builtins, "__import__", _import_blocker("netCDF4"))
    with pytest.raises(ModuleNotFoundError, match="netCDF4"):
        vmec_reference_mod._read_vmec_wout_netcdf(tmp_path / "wout.nc")
    with pytest.raises(ModuleNotFoundError, match="netCDF4"):
        vmec_reference_mod._read_booz_netcdf(tmp_path / "booz.nc")


def test_vmec_reference_error_branches(monkeypatch, tmp_path):
    missing = tmp_path / "missing.nc"
    with pytest.raises(FileNotFoundError):
        load_vmec_surface_reference(missing, s=0.25)

    present = tmp_path / "present.nc"
    present.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="s must be between 0 and 1"):
        load_vmec_surface_reference(present, s=2.0)

    monkeypatch.setattr(
        vmec_reference_mod,
        "_read_vmec_wout_netcdf",
        lambda path: {
            "ns": 1,
            "nfp": 5,
            "mpol": 3,
            "ntor": 2,
            "Aminor_p": 2.0,
            "volume_p": 1.0,
            "phi": np.asarray([0.0]),
            "iotaf": np.asarray([0.6]),
            "xm_nyq": np.asarray([0]),
            "xn_nyq": np.asarray([0]),
            "gmnc": np.asarray([[1.0]]),
            "bmnc": np.asarray([[1.0]]),
            "bsupumnc": np.asarray([[0.3]]),
            "bsupvmnc": np.asarray([[1.1]]),
            "bsubumnc": np.asarray([[0.2]]),
            "bsubvmnc": np.asarray([[1.0]]),
        },
    )
    with pytest.raises(ValueError, match="at least two radial surfaces"):
        load_vmec_surface_reference(present, s=0.25)

    monkeypatch.setattr(
        vmec_reference_mod,
        "_read_vmec_wout_netcdf",
        lambda path: {
            "ns": 3,
            "nfp": 5,
            "mpol": 3,
            "ntor": 2,
            "Aminor_p": 2.0,
            "volume_p": 1.0,
            "phi": np.asarray([0.0, 0.5, 1.0]),
            "iotaf": np.asarray([0.6, 0.65, 0.7]),
            "xm_nyq": np.asarray([0, 1]),
            "xn_nyq": np.asarray([0, 5]),
            "gmnc": np.asarray([[1.0, 0.0], [1.1, 0.0], [1.2, 0.0]]),
            "bmnc": np.zeros((3, 2)),
            "bsupumnc": np.asarray([[0.3, 0.0], [0.31, 0.0], [0.32, 0.0]]),
            "bsupvmnc": np.asarray([[1.1, 0.0], [1.11, 0.0], [1.12, 0.0]]),
            "bsubumnc": np.asarray([[0.2, 0.0], [0.21, 0.0], [0.22, 0.0]]),
            "bsubvmnc": np.asarray([[1.0, 0.0], [1.01, 0.0], [1.02, 0.0]]),
        },
    )
    with pytest.raises(ValueError, match="zero magnetic-field strength"):
        load_vmec_surface_reference(present, s=0.25)
