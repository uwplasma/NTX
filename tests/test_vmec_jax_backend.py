from __future__ import annotations

import dataclasses
import sys
from types import ModuleType, SimpleNamespace

import jax.numpy as jnp

from ntx import surface_from_vmec_jax_state, surface_from_vmec_jax_wout
from ntx.geometry import BoozerSurface
from ntx.vmec_jax_backend import _apply_boozer_sign_convention


def test_apply_boozer_sign_convention_returns_right_handed_values():
    iota, b_theta, b_zeta = _apply_boozer_sign_convention(
        iota=0.5,
        b_theta=0.2,
        b_zeta=1.3,
    )
    assert iota == -0.5
    assert b_zeta + iota * b_theta >= 0.0


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

    monkeypatch.setattr("ntx.vmec_jax_backend.surface_from_vmec_jax_state", fake_surface_from_state)
    result = surface_from_vmec_jax_wout(
        input_path=tmp_path / "input.vmec",
        wout_path=tmp_path / "wout.nc",
        s=0.25,
    )
    assert result == "surface"
    assert captured["static"].ns == 5
    assert captured["static"].mpol == 3
    assert captured["static"].ntor == 2
