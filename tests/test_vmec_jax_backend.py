from __future__ import annotations

import builtins
import dataclasses
import sys
from types import ModuleType, SimpleNamespace

import jax.numpy as jnp

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


def test_surface_from_vmec_jax_wout_updates_static_from_wout(monkeypatch, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class FakeCfg:
        ns: int
        mpol: int
        ntor: int

    cfg = FakeCfg(ns=3, mpol=2, ntor=1)
    indata = object()
    wout = SimpleNamespace(ns=5, mpol=3, ntor=2, signgs=1)
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
        "ntx._vmec_jax_surfaces.surface_from_vmec_jax_state",
        fake_surface_from_state,
    )
    result = surface_from_vmec_jax_wout(
        input_path=tmp_path / "input.vmec",
        wout_path=tmp_path / "wout.nc",
        s=0.25,
    )
    assert result == "surface"
    assert captured["static"].ns == 5
    assert captured["static"].mpol == 3
    assert captured["static"].ntor == 2


def test_surface_from_vmec_jax_wout_keeps_matching_static(monkeypatch, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class FakeCfg:
        ns: int
        mpol: int
        ntor: int

    cfg = FakeCfg(ns=5, mpol=3, ntor=2)
    wout = SimpleNamespace(ns=5, mpol=3, ntor=2, signgs=-1)
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.load_config = lambda path: (cfg, "indata")
    vmec_pkg.build_static = lambda cfg_obj: cfg_obj
    vmec_api = ModuleType("vmec_jax.api")
    vmec_api.read_wout = lambda path: wout
    vmec_api.state_from_wout = lambda w: "state"
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", vmec_api)

    captured = {}

    def fake_surface_from_state(**kwargs):
        captured["kwargs"] = kwargs
        return "surface"

    monkeypatch.setattr(
        "ntx._vmec_jax_surfaces.surface_from_vmec_jax_state",
        fake_surface_from_state,
    )

    result = surface_from_vmec_jax_wout(
        input_path=tmp_path / "input.vmec",
        wout_path=tmp_path / "wout.nc",
        s=0.25,
    )
    assert result == "surface"
    assert captured["kwargs"]["static"] is cfg
    assert captured["kwargs"]["signgs"] == -1


def test_surface_from_vmec_jax_wout_keeps_flux_scale_explicit(monkeypatch, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class FakeCfg:
        ns: int
        mpol: int
        ntor: int

    cfg = FakeCfg(ns=5, mpol=3, ntor=2)
    wout = SimpleNamespace(
        ns=5,
        mpol=3,
        ntor=2,
        signgs=1,
        phi=jnp.asarray([0.0, 0.25 * jnp.pi, 0.5 * jnp.pi]),
    )
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.load_config = lambda path: (cfg, "indata")
    vmec_pkg.build_static = lambda cfg_obj: cfg_obj
    vmec_api = ModuleType("vmec_jax.api")
    vmec_api.read_wout = lambda path: wout
    vmec_api.state_from_wout = lambda w: "state"
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", vmec_api)

    captured = {}

    def fake_surface_from_state(**kwargs):
        captured["kwargs"] = kwargs
        return "surface"

    monkeypatch.setattr(
        "ntx._vmec_jax_surfaces.surface_from_vmec_jax_state",
        fake_surface_from_state,
    )

    result = surface_from_vmec_jax_wout(
        input_path=tmp_path / "input.vmec",
        wout_path=tmp_path / "wout.nc",
        s=0.25,
    )

    assert result == "surface"
    assert float(captured["kwargs"]["psi_p"]) == 1.0


def test_surface_from_vmec_jax_wout_can_use_wout_profiles(monkeypatch, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class FakeCfg:
        ns: int
        mpol: int
        ntor: int

    class FakeFluxProfiles(SimpleNamespace):
        pass

    cfg = FakeCfg(ns=4, mpol=3, ntor=2)
    wout = SimpleNamespace(
        ns=4,
        mpol=3,
        ntor=2,
        signgs=-1,
        phipf=jnp.asarray([1.0, 1.1, 1.2, 1.3]),
        chipf=jnp.asarray([2.0, 2.1, 2.2, 2.3]),
        phips=jnp.asarray([0.0, 3.1, 3.2, 3.3]),
        iotas=jnp.asarray([0.0, 0.4, 0.5, 0.6]),
        pres=jnp.asarray([0.0, 10.0, 8.0, 1.0]),
    )
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.load_config = lambda path: (cfg, "indata")
    vmec_pkg.build_static = lambda cfg_obj: SimpleNamespace(
        cfg=cfg_obj,
        s=jnp.linspace(0.0, 1.0, cfg_obj.ns),
    )
    vmec_api = ModuleType("vmec_jax.api")
    vmec_api.read_wout = lambda path: wout
    vmec_api.state_from_wout = lambda w: "state"
    vmec_energy = ModuleType("vmec_jax.energy")
    vmec_energy.FluxProfiles = FakeFluxProfiles
    vmec_energy.lamscale_from_phips = lambda phips, s: jnp.asarray(9.0)
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", vmec_api)
    monkeypatch.setitem(sys.modules, "vmec_jax.energy", vmec_energy)

    captured = {}

    def fake_surface_from_state(**kwargs):
        captured["kwargs"] = kwargs
        return "surface"

    monkeypatch.setattr(
        "ntx._vmec_jax_surfaces.surface_from_vmec_jax_state",
        fake_surface_from_state,
    )

    result = surface_from_vmec_jax_wout(
        input_path=tmp_path / "input.vmec",
        wout_path=tmp_path / "wout.nc",
        s=0.25,
        profile_source="state_wout_profiles",
    )

    assert result == "surface"
    flux = captured["kwargs"]["flux_profiles"]
    profiles_half = captured["kwargs"]["profiles_half"]
    assert isinstance(flux, FakeFluxProfiles)
    assert jnp.allclose(flux.phipf, -wout.phipf / (2.0 * jnp.pi))
    assert jnp.allclose(flux.chipf, -wout.chipf / (2.0 * jnp.pi))
    assert jnp.allclose(flux.phips, wout.phips)
    assert flux.signgs == -1
    assert jnp.allclose(flux.lamscale, jnp.asarray(9.0))
    assert jnp.allclose(profiles_half["iota"], wout.iotas)
    assert jnp.allclose(profiles_half["pressure"], wout.pres)


def test_surface_from_vmec_jax_wout_auto_falls_back_to_wout_backend(
    monkeypatch,
    tmp_path,
):
    @dataclasses.dataclass(frozen=True)
    class FakeCfg:
        ns: int
        mpol: int
        ntor: int

    cfg = FakeCfg(ns=3, mpol=2, ntor=1)
    wout = SimpleNamespace(
        ns=3,
        mpol=2,
        ntor=1,
        signgs=1,
    )
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.load_config = lambda path: (cfg, "indata")
    vmec_pkg.build_static = lambda cfg_obj: SimpleNamespace(
        cfg=cfg_obj,
        s=jnp.linspace(0.0, 1.0, cfg_obj.ns),
    )
    vmec_api = ModuleType("vmec_jax.api")
    vmec_api.read_wout = lambda path: wout
    vmec_api.state_from_wout = lambda w: "state"
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", vmec_api)

    calls = []

    def fake_surface_from_state(**kwargs):
        calls.append(kwargs)
        raise NotImplementedError("unsupported VMEC input profile")

    fallback = {}

    def fake_wout_fallback(wout_obj, **kwargs):
        fallback["wout"] = wout_obj
        fallback["kwargs"] = kwargs
        return "surface"

    monkeypatch.setattr(
        "ntx._vmec_jax_surfaces.surface_from_vmec_jax_state",
        fake_surface_from_state,
    )
    monkeypatch.setattr(
        "ntx._vmec_jax_surfaces._surface_from_booz_xform_wout_data",
        fake_wout_fallback,
    )

    result = surface_from_vmec_jax_wout(
        input_path=tmp_path / "input.vmec",
        wout_path=tmp_path / "wout.nc",
        s=0.25,
    )

    assert result == "surface"
    assert len(calls) == 1
    assert calls[0]["flux_profiles"] is None
    assert fallback["wout"] is wout
    assert fallback["kwargs"]["s"] == 0.25


def test_surface_from_vmec_jax_wout_current_api_uses_finalized_wout(
    monkeypatch,
    tmp_path,
):
    wout = SimpleNamespace(ns=3, mpol=2, ntor=1, signgs=1)
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.read_wout = lambda path: wout
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.delitem(sys.modules, "vmec_jax.api", raising=False)

    captured = {}

    def fake_wout_fallback(wout_obj, **kwargs):
        captured["wout"] = wout_obj
        captured["kwargs"] = kwargs
        return "surface"

    monkeypatch.setattr(
        "ntx._vmec_jax_surfaces._surface_from_booz_xform_wout_data",
        fake_wout_fallback,
    )

    result = surface_from_vmec_jax_wout(
        input_path=tmp_path / "input.vmec",
        wout_path=tmp_path / "wout.nc",
        s=0.5,
    )

    assert result == "surface"
    assert captured["wout"] is wout
    assert captured["kwargs"]["s"] == 0.5


def test_surface_from_vmec_jax_wout_source_uses_wout_backend(monkeypatch, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class FakeCfg:
        ns: int
        mpol: int
        ntor: int

    cfg = FakeCfg(ns=3, mpol=2, ntor=1)
    wout = SimpleNamespace(ns=3, mpol=2, ntor=1, signgs=1)
    vmec_pkg = ModuleType("vmec_jax")
    vmec_pkg.load_config = lambda path: (cfg, "indata")
    vmec_pkg.build_static = lambda cfg_obj: SimpleNamespace(
        cfg=cfg_obj,
        s=jnp.linspace(0.0, 1.0, cfg_obj.ns),
    )
    vmec_api = ModuleType("vmec_jax.api")
    vmec_api.read_wout = lambda path: wout
    vmec_api.state_from_wout = lambda w: "state"
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.api", vmec_api)

    fallback = {}

    def fake_wout_fallback(wout_obj, **kwargs):
        fallback["wout"] = wout_obj
        fallback["kwargs"] = kwargs
        return "surface"

    monkeypatch.setattr(
        "ntx._vmec_jax_surfaces._surface_from_booz_xform_wout_data",
        fake_wout_fallback,
    )

    result = surface_from_vmec_jax_wout(
        input_path=tmp_path / "input.vmec",
        wout_path=tmp_path / "wout.nc",
        s=0.5,
        mboz=5,
        nboz=4,
        profile_source="wout",
    )

    assert result == "surface"
    assert fallback["wout"] is wout
    assert fallback["kwargs"]["mboz"] == 5
    assert fallback["kwargs"]["nboz"] == 4


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
