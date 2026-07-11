from __future__ import annotations

import builtins
import dataclasses
import sys
from types import ModuleType, SimpleNamespace

import jax.numpy as jnp
import pytest

from ntx import surface_from_vmec_jax_state, surface_from_vmec_jax_wout
from ntx._vmec_jax_boozer import _apply_boozer_sign_convention_profiles
from ntx.geometry import BoozerSurface
from ntx.vmec_jax_backend import (
    _apply_boozer_sign_convention,
    _booz_xform_bundle_from_vmec_jax_state,
    _import_booz_xform_jax_api,
    _import_vmec_jax,
    _prepend_checkout,
    build_vmec_jax_boundary_context,
    initial_guess_vmec_jax_boundary_state,
    relax_vmec_jax_boundary_state_explicit,
    solve_vmec_jax_boundary_state,
    surfaces_from_vmec_jax_boundary_params,
)


def test_apply_boozer_sign_convention_returns_right_handed_values():
    iota, b_theta, b_zeta = _apply_boozer_sign_convention(
        iota=0.5,
        b_theta=0.2,
        b_zeta=1.3,
    )
    assert iota == -0.5
    assert b_zeta + iota * b_theta >= 0.0

    flipped_iota, flipped_b_theta, flipped_b_zeta = _apply_boozer_sign_convention(
        iota=0.5,
        b_theta=-2.0,
        b_zeta=0.3,
    )
    assert flipped_iota == -0.5
    assert flipped_b_zeta + flipped_iota * flipped_b_theta >= 0.0
    assert jnp.allclose(
        flipped_b_zeta + flipped_iota * flipped_b_theta,
        jnp.asarray(0.7),
    )


def test_apply_boozer_sign_convention_profiles_keeps_positive_jacobian():
    iota, b_theta, b_zeta, gmnc = _apply_boozer_sign_convention_profiles(
        iotaf=jnp.asarray([0.0, 0.4, 0.5]),
        buco=jnp.asarray([0.0, -2.0, 0.2]),
        bvco=jnp.asarray([0.0, 0.3, 1.0]),
        gmnc_b=jnp.asarray([[2.0, 0.1], [3.0, 0.2]]),
    )

    assert jnp.allclose(iota, jnp.asarray([0.0, -0.4, -0.5]))
    assert jnp.all(b_zeta[1:] + iota[1:] * b_theta[1:] >= 0.0)
    assert jnp.allclose(gmnc[0], jnp.asarray([-2.0, -0.1]))
    assert jnp.allclose(gmnc[1], jnp.asarray([3.0, 0.2]))


def test_surface_from_vmec_jax_state_builds_boozer_surface(monkeypatch):
    jax_api = ModuleType("booz_xform_jax.jax_api")
    jax_api.prepare_booz_xform_constants_from_inputs = lambda **kwargs: ("constants", "grids")
    jax_api.booz_xform_from_inputs = lambda **kwargs: {
        "bmnc_b": jnp.asarray([[2.0, 0.1]]),
        "ixm_b": jnp.asarray([0, 1]),
        "ixn_b": jnp.asarray([0, 2]),
        "iota_b": jnp.asarray([0.4]),
        "buco_b": jnp.asarray([0.2]),
        "bvco_b": jnp.asarray([1.2]),
    }
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.booz_xform_inputs_from_state = lambda **kwargs: SimpleNamespace(nfp=2)
    vmec_pkg.surface_indices_from_static = lambda static, s_values: ([0], s_values)
    monkeypatch.setitem(sys.modules, "booz_xform_jax.jax_api", jax_api)
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)

    surface = surface_from_vmec_jax_state(
        state="state",
        static=SimpleNamespace(cfg=SimpleNamespace(lasym=False)),
        indata=SimpleNamespace(input_filename="sample.vmec"),
        signgs=1,
        s=0.25,
    )
    assert isinstance(surface, BoozerSurface)
    assert surface.nfp == 2
    assert surface.b0 == 2.0


def test_surface_from_current_vmec_jax_state_uses_core_boozer_tables(monkeypatch):
    requested_rows = []
    vmec_pkg = ModuleType("vmec_jax")
    vmec_core = ModuleType("vmec_jax.core")
    boozer_tables = ModuleType("vmec_jax.core.boozer_tables")

    def fake_tables(state, runtime, row):
        requested_rows.append(row)
        return {
            "xm": jnp.asarray([0, 1]),
            "xn": jnp.asarray([0, 2]),
            "rmnc": jnp.asarray([1.0, 0.1]),
            "zmns": jnp.asarray([0.0, 0.1]),
            "lmns": jnp.asarray([0.0, 0.01]),
            "bmnc": jnp.asarray([2.0, 0.1]),
            "bsubumnc": jnp.asarray([0.2, 0.0]),
            "bsubvmnc": jnp.asarray([1.2, 0.0]),
            "iota": jnp.asarray(0.4),
        }

    boozer_tables.boozer_input_tables = fake_tables
    jax_api = ModuleType("booz_xform_jax.jax_api")
    jax_api.prepare_booz_xform_constants_from_inputs = lambda **kwargs: ("constants", "grids")

    def fake_transform(**kwargs):
        inputs = kwargs["inputs"]
        assert inputs.bmnc.shape == (1, 2)
        assert jnp.allclose(inputs.xm, inputs.xm_nyq)
        return {
            "bmnc_b": inputs.bmnc,
            "ixm_b": inputs.xm,
            "ixn_b": inputs.xn,
            "iota_b": inputs.iota,
            "buco_b": jnp.asarray([0.2]),
            "bvco_b": jnp.asarray([1.2]),
        }

    jax_api.booz_xform_from_inputs = fake_transform
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.core", vmec_core)
    monkeypatch.setitem(sys.modules, "vmec_jax.core.boozer_tables", boozer_tables)
    monkeypatch.setitem(sys.modules, "booz_xform_jax.jax_api", jax_api)

    runtime = SimpleNamespace(
        resolution=SimpleNamespace(nfp=2, lasym=False),
        setup=SimpleNamespace(signgs=1, s_full=jnp.asarray([0.0, 0.5, 1.0])),
    )
    surface = surface_from_vmec_jax_state(
        state=SimpleNamespace(R_cos=jnp.ones((3, 2))),
        static=runtime,
        indata=SimpleNamespace(source_path="sample.vmec"),
        signgs=1,
        s=0.3,
    )

    assert requested_rows == [1]
    assert surface.nfp == 2
    assert surface.b0 == 2.0


def test_surface_from_vmec_jax_wout_uses_current_root_api(monkeypatch, tmp_path):
    wout = SimpleNamespace(ns=3, mpol=2, ntor=1, signgs=1)
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.read_wout = lambda path: wout
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)

    fallback = {}

    def fake_wout_fallback(wout_obj, **kwargs):
        fallback["wout"] = wout_obj
        fallback["kwargs"] = kwargs
        return "surface"

    monkeypatch.setattr(
        "ntx._vmec_jax_surfaces._surface_from_booz_xform_wout_data",
        fake_wout_fallback,
    )

    input_path = tmp_path / "input.vmec"
    wout_path = tmp_path / "wout.nc"
    input_path.touch()
    wout_path.touch()
    result = surface_from_vmec_jax_wout(
        input_path=input_path,
        wout_path=wout_path,
        s=0.5,
        mboz=5,
        nboz=4,
        profile_source="wout",
    )

    assert result == "surface"
    assert fallback["wout"] is wout
    assert fallback["kwargs"]["mboz"] == 5
    assert fallback["kwargs"]["nboz"] == 4


def test_surface_from_vmec_jax_wout_rejects_removed_state_reconstruction(monkeypatch, tmp_path):
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.read_wout = lambda path: object()
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    input_path = tmp_path / "input.vmec"
    wout_path = tmp_path / "wout.nc"
    input_path.touch()
    wout_path.touch()

    with pytest.raises(NotImplementedError, match="removed legacy"):
        surface_from_vmec_jax_wout(
            input_path=input_path,
            wout_path=wout_path,
            s=0.5,
            profile_source="input",
        )


def test_vmec_jax_boundary_context_and_state_helpers(monkeypatch, tmp_path):
    calls = {}
    vmec_pkg = ModuleType("vmec_jax")
    cfg = SimpleNamespace()
    indata = SimpleNamespace(get_float=lambda key, default: 1.5)
    static = SimpleNamespace(
        modes="modes",
        s=jnp.asarray([0.0, 0.5, 1.0]),
    )
    boundary = "boundary"
    specs = ("rc10", "zs10")
    state0 = SimpleNamespace(
        Rcos=jnp.asarray([[0.0, 0.0], [1.0, 2.0]]),
        Rsin=jnp.asarray([[0.0, 0.0], [3.0, 4.0]]),
        Zcos=jnp.asarray([[0.0, 0.0], [5.0, 6.0]]),
        Zsin=jnp.asarray([[0.0, 0.0], [7.0, 8.0]]),
    )
    solved_state = SimpleNamespace(name="solved")
    relaxed_state = SimpleNamespace(name="relaxed")

    vmec_pkg.load_config = lambda path: (cfg, indata)
    vmec_pkg.build_static = lambda cfg_obj: static
    vmec_pkg.boundary_input_from_indata = lambda indata_obj, modes: boundary
    vmec_pkg.boundary_param_specs = lambda *args, **kwargs: specs
    vmec_pkg.apply_boundary_params = lambda boundary_obj, specs_obj, params: (
        boundary_obj,
        tuple(specs_obj),
        tuple(jnp.asarray(params).tolist()),
    )
    vmec_pkg.initial_guess_from_boundary = (
        lambda static_obj, boundary_obj, indata_obj, vmec_project: state0
    )

    def fake_solve(state, static_obj, **kwargs):
        calls["implicit"] = kwargs
        return solved_state

    vmec_pkg.implicit = SimpleNamespace(
        solve_fixed_boundary_state_implicit_vmec_residual=fake_solve
    )
    vmec_pkg.flux_profiles_from_indata = lambda indata_obj, s, signgs: SimpleNamespace(
        phipf=jnp.ones_like(s),
        chipf=2.0 * jnp.ones_like(s),
        lamscale=3.0,
    )

    def fake_relax(state, static_obj, **kwargs):
        calls["relax"] = kwargs
        return SimpleNamespace(state=relaxed_state)

    vmec_pkg.solve_fixed_boundary_gd = fake_relax
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)

    context = build_vmec_jax_boundary_context(tmp_path / "input.vmec", signgs=-1)
    assert context.specs == specs
    assert context.signgs == -1

    guess = initial_guess_vmec_jax_boundary_state(context, jnp.asarray([0.0, 1.0]))
    assert guess is state0

    solved = solve_vmec_jax_boundary_state(
        context,
        jnp.asarray([0.0, 1.0]),
        vmec_project=False,
        max_iter=3,
        step_size=0.25,
        ftol=1.0e-8,
        implicit="implicit",
    )
    assert solved is solved_state
    assert calls["implicit"]["max_iter"] == 3
    assert jnp.allclose(calls["implicit"]["edge_Rcos"], jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(calls["implicit"]["edge_Rsin"], jnp.asarray([3.0, 4.0]))
    assert jnp.allclose(calls["implicit"]["edge_Zcos"], jnp.asarray([5.0, 6.0]))
    assert jnp.allclose(calls["implicit"]["edge_Zsin"], jnp.asarray([7.0, 8.0]))

    relaxed = relax_vmec_jax_boundary_state_explicit(
        context,
        jnp.asarray([0.0, 1.0]),
        pressure=jnp.asarray([1.0, 2.0, 3.0]),
        max_iter=4,
        step_size=1.0e-7,
        stop_grad_in_update=True,
    )
    assert relaxed is relaxed_state
    assert calls["relax"]["max_iter"] == 4
    assert calls["relax"]["gamma"] == 1.5
    assert jnp.allclose(calls["relax"]["pressure"], jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.allclose(calls["relax"]["edge_Rcos"], jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(calls["relax"]["edge_Rsin"], jnp.asarray([3.0, 4.0]))
    assert jnp.allclose(calls["relax"]["edge_Zcos"], jnp.asarray([5.0, 6.0]))
    assert jnp.allclose(calls["relax"]["edge_Zsin"], jnp.asarray([7.0, 8.0]))

    relaxed_zero_pressure = relax_vmec_jax_boundary_state_explicit(
        context,
        jnp.asarray([0.0, 1.0]),
        pressure=None,
    )
    assert relaxed_zero_pressure is relaxed_state
    assert jnp.allclose(calls["relax"]["pressure"], jnp.asarray([0.0, 0.0, 0.0]))


def test_current_vmec_jax_boundary_context_uses_implicit_params(monkeypatch, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class FakeParams:
        rbc: object
        rbs: object
        zbc: object
        zbs: object

    @dataclasses.dataclass(frozen=True)
    class FakeConfig:
        max_iterations: int = 20
        ftol: float = 1.0e-8

    inp = SimpleNamespace(
        mpol=2,
        ntor=1,
        lasym=False,
        rbc=jnp.zeros((3, 2)),
        rbs=jnp.zeros((3, 2)),
        zbc=jnp.zeros((3, 2)),
        zbs=jnp.zeros((3, 2)),
    )
    base = FakeParams(inp.rbc, inp.rbs, inp.zbc, inp.zbs)
    runtime = SimpleNamespace(setup=SimpleNamespace(signgs=-1, s_full=jnp.asarray([0.0, 0.5, 1.0])))
    calls = {}

    class FakeVmecInput:
        @staticmethod
        def from_file(path):
            calls["path"] = path
            return inp

    def solve_implicit(params, cfg):
        calls["params"] = params
        calls["cfg"] = cfg
        return "solved"

    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.VmecInput = FakeVmecInput
    vmec_pkg.implicit = SimpleNamespace(
        make_config=lambda value: FakeConfig(),
        params_from_input=lambda value: base,
        runtime_from_params=lambda params, cfg: runtime,
        solve_implicit=solve_implicit,
    )
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    input_path = tmp_path / "input.vmec"
    input_path.touch()

    context = build_vmec_jax_boundary_context(
        input_path,
        max_mode=1,
        include=("rc",),
        fix=("rc00",),
    )
    assert context.backend == "core"
    assert context.signgs == -1
    assert [spec.name for spec in context.specs] == ["rc01", "rc1-1", "rc10", "rc11"]

    updates = jnp.asarray([0.1, 0.2, 0.3, 0.4])
    assert solve_vmec_jax_boundary_state(context, updates, max_iter=3) == "solved"
    assert calls["cfg"].max_iterations == 20
    assert jnp.allclose(calls["params"].rbc[:, 0], jnp.asarray([0.0, 0.0, 0.1]))
    assert jnp.allclose(calls["params"].rbc[:, 1], jnp.asarray([0.2, 0.3, 0.4]))

    with pytest.raises(NotImplementedError, match="removed the experimental explicit"):
        relax_vmec_jax_boundary_state_explicit(context, updates)


def test_surfaces_from_boundary_params_delegates(monkeypatch):
    import ntx._vmec_jax_surfaces as surfaces

    context = SimpleNamespace(static="static", indata="indata", signgs=1)
    calls = {}

    def fake_solve(ctx, params, **kwargs):
        calls["solve"] = (ctx, params, kwargs)
        return "state"

    def fake_surfaces(**kwargs):
        calls["surfaces"] = kwargs
        return ("surface",)

    monkeypatch.setattr(surfaces, "solve_vmec_jax_boundary_state", fake_solve)
    monkeypatch.setattr(surfaces, "surfaces_from_vmec_jax_state", fake_surfaces)

    result = surfaces_from_vmec_jax_boundary_params(
        context,
        jnp.asarray([0.0]),
        s_values=(0.25,),
        max_iter=2,
        step_size=0.5,
        psi_p=1.7,
    )
    assert result == ("surface",)
    assert calls["solve"][2]["max_iter"] == 2
    assert calls["surfaces"]["state"] == "state"
    assert calls["surfaces"]["psi_p"] == 1.7


def test_booz_xform_bundle_accepts_all_surfaces(monkeypatch):
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.booz_xform_inputs_from_state = lambda **kwargs: SimpleNamespace(nfp=2)
    jax_api = ModuleType("booz_xform_jax.jax_api")
    jax_api.prepare_booz_xform_constants_from_inputs = lambda **kwargs: ("constants", "grids")

    def fake_booz_xform_from_inputs(**kwargs):
        assert kwargs["surface_indices"] is None
        return {"bmnc_b": jnp.asarray([[2.0]])}

    jax_api.booz_xform_from_inputs = fake_booz_xform_from_inputs
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.setitem(sys.modules, "booz_xform_jax.jax_api", jax_api)

    inputs, out = _booz_xform_bundle_from_vmec_jax_state(
        state="state",
        static=SimpleNamespace(cfg=SimpleNamespace(lasym=False)),
        indata="indata",
        signgs=1,
        s_values=None,
        mboz=2,
        nboz=2,
    )
    assert inputs.nfp == 2
    assert jnp.allclose(out["bmnc_b"], jnp.asarray([[2.0]]))


def test_prepend_checkout_adds_existing_root_once(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    root_str = str(root)
    try:
        while root_str in sys.path:
            sys.path.remove(root_str)
        _prepend_checkout(root)
        _prepend_checkout(root)
        assert sys.path.count(root_str) == 1
        _prepend_checkout(None)
        assert sys.path.count(root_str) == 1
    finally:
        while root_str in sys.path:
            sys.path.remove(root_str)


def test_import_helpers_use_checkout_fallback(monkeypatch, tmp_path):
    import ntx._vmec_jax_boozer as boozer_backend

    real_import = builtins.__import__
    fake_vmec = ModuleType("vmec_jax")
    fake_jax_api = ModuleType("booz_xform_jax.jax_api")
    fake_booz_pkg = ModuleType("booz_xform_jax")
    fake_booz_pkg.jax_api = fake_jax_api
    attempts = {"vmec": 0, "booz": 0}

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vmec_jax":
            attempts["vmec"] += 1
            if attempts["vmec"] == 1:
                raise ModuleNotFoundError(name)
            return fake_vmec
        if name == "booz_xform_jax":
            attempts["booz"] += 1
            if attempts["booz"] == 1:
                raise ModuleNotFoundError(name)
            return fake_booz_pkg
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "vmec_jax", raising=False)
    monkeypatch.delitem(sys.modules, "booz_xform_jax.jax_api", raising=False)
    monkeypatch.setattr(boozer_backend, "find_vmec_jax_root", lambda: tmp_path / "vmec_jax")
    monkeypatch.setattr(
        boozer_backend,
        "find_booz_xform_jax_root",
        lambda: tmp_path / "booz_xform_jax",
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        assert _import_vmec_jax() is fake_vmec
        assert _import_booz_xform_jax_api() is fake_jax_api
        assert attempts == {"vmec": 2, "booz": 2}
    finally:
        for path in (tmp_path / "vmec_jax", tmp_path / "booz_xform_jax"):
            path_str = str(path)
            while path_str in sys.path:
                sys.path.remove(path_str)
