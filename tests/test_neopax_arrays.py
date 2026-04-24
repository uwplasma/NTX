from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

import ntx._neopax_scan as neopax_scan_module
import ntx.neopax as neopax_module
from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    build_ntx_neopax_scan_from_surfaces,
    example_surface,
    load_neopax_reference_scan,
    load_vmec_surface,
    neopax_scan_requires_rebuild,
    scan_to_neopax_arrays,
    write_neopax_scan_hdf5,
)
from ntx.neopax import _surface_reference_bridge

from .fixture_data import SAMPLE_WOUT


def test_build_ntx_neopax_scan_from_surfaces_matches_callback_builder():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    explicit = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
        source_name="explicit",
    )
    callback = build_ntx_neopax_scan(
        lambda rho_value: example_surface(),
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
        source_name="callback",
    )

    assert jnp.allclose(explicit.D11, callback.D11)
    assert jnp.allclose(explicit.D13, callback.D13)
    assert jnp.allclose(explicit.D33, callback.D33)
    assert explicit.D33_spitzer is not None
    assert callback.D33_spitzer is not None
    assert jnp.allclose(explicit.D33_spitzer, callback.D33_spitzer)
    assert explicit.fac_reference_to_sfincs_11 is not None
    assert explicit.fac_reference_to_sfincs_31 is not None
    assert explicit.fac_reference_to_sfincs_33 is not None
    assert explicit.fac_sfincs_to_dkes_11 is not None
    assert explicit.fac_sfincs_to_dkes_31 is not None
    assert explicit.fac_sfincs_to_dkes_33 is not None
    assert jnp.all(explicit.fac_reference_to_sfincs_11 > 0)
    assert jnp.all(explicit.fac_reference_to_sfincs_31 > 0)
    assert jnp.all(explicit.fac_reference_to_sfincs_33 > 0)
    assert jnp.all(explicit.fac_sfincs_to_dkes_11 > 0)
    assert jnp.all(explicit.fac_sfincs_to_dkes_31 > 0)
    assert jnp.all(explicit.fac_sfincs_to_dkes_33 > 0)


def test_build_ntx_neopax_scan_validates_basic_shapes():
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    grid = GridSpec(5, 5, 4)
    with pytest.raises(ValueError, match="drds must have the same length as rho"):
        build_ntx_neopax_scan(
            lambda _: example_surface(),
            rho=rho,
            nu_v=nu_v,
            Er=jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]]),
            drds=jnp.asarray([1.0]),
            grid=grid,
        )
    with pytest.raises(ValueError, match="set at least one of Es or Er"):
        build_ntx_neopax_scan(
            lambda _: example_surface(),
            rho=rho,
            nu_v=nu_v,
            drds=jnp.asarray([1.0, 1.5]),
            grid=grid,
        )
    with pytest.raises(ValueError, match="Es and Er must have the same shape"):
        build_ntx_neopax_scan(
            lambda _: example_surface(),
            rho=rho,
            nu_v=nu_v,
            Es=jnp.asarray([[0.0, 1.0e-3]]),
            Er=jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]]),
            drds=jnp.asarray([1.0, 1.5]),
            grid=grid,
        )
    with pytest.raises(ValueError, match="Es/Er first dimension must match rho"):
        build_ntx_neopax_scan(
            lambda _: example_surface(),
            rho=rho,
            nu_v=nu_v,
            Es=jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3], [0.0, 3.0e-3]]),
            Er=jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3], [0.0, 3.0e-3]]),
            drds=jnp.asarray([1.0, 1.5]),
            grid=grid,
        )


def test_build_ntx_neopax_scan_from_vmec_jax_state_forwards_owned_surfaces(monkeypatch):
    surfaces = (example_surface(), example_surface())
    calls: dict[str, object] = {}

    def fake_surfaces_from_state(**kwargs):
        calls["surface_kwargs"] = kwargs
        return surfaces

    def fake_build_from_surfaces(surface_tuple, **kwargs):
        calls["build_surfaces"] = surface_tuple
        calls["build_kwargs"] = kwargs
        return SimpleNamespace(rho=kwargs["rho"], source_name=kwargs["source_name"])

    monkeypatch.setattr(
        neopax_scan_module,
        "surfaces_from_vmec_jax_state",
        fake_surfaces_from_state,
    )
    monkeypatch.setattr(
        neopax_scan_module,
        "build_ntx_neopax_scan_from_surfaces",
        fake_build_from_surfaces,
    )

    rho = jnp.asarray([0.25, 0.5])
    result = neopax_module.build_ntx_neopax_scan_from_vmec_jax_state(
        state="state",
        static="static",
        indata="indata",
        signgs=-1,
        rho=rho,
        nu_v=jnp.asarray([1.0e-3]),
        Er=jnp.ones((2, 1)),
        drds=jnp.ones(2),
        grid=GridSpec(5, 5, 4),
        source_name="state-scan",
        mboz=4,
        nboz=5,
        psi_p=2.0,
        min_bmn_to_load=1.0e-6,
    )

    assert result.source_name == "state-scan"
    assert calls["build_surfaces"] is surfaces
    assert calls["surface_kwargs"]["s_values"] == pytest.approx((0.0625, 0.25))
    assert calls["surface_kwargs"]["mboz"] == 4
    assert calls["surface_kwargs"]["nboz"] == 5
    assert calls["surface_kwargs"]["psi_p"] == 2.0
    assert calls["surface_kwargs"]["min_bmn_to_load"] == 1.0e-6
    assert jnp.allclose(calls["build_kwargs"]["rho"], rho)


def test_build_ntx_neopax_scan_from_vmec_jax_boundary_params_forwards_owned_surfaces(monkeypatch):
    surfaces = (example_surface(),)
    context = SimpleNamespace(static="static", indata="indata", signgs=1)
    calls: dict[str, object] = {}

    def fake_surfaces_from_boundary(context_arg, params_arg, **kwargs):
        calls["context"] = context_arg
        calls["params"] = params_arg
        calls["surface_kwargs"] = kwargs
        return surfaces

    def fake_build_from_surfaces(surface_tuple, **kwargs):
        calls["build_surfaces"] = surface_tuple
        calls["build_kwargs"] = kwargs
        return SimpleNamespace(rho=kwargs["rho"], source_name=kwargs["source_name"])

    monkeypatch.setattr(
        neopax_scan_module,
        "surfaces_from_vmec_jax_boundary_params",
        fake_surfaces_from_boundary,
    )
    monkeypatch.setattr(
        neopax_scan_module,
        "build_ntx_neopax_scan_from_surfaces",
        fake_build_from_surfaces,
    )

    rho = jnp.asarray([0.4])
    params = jnp.asarray([0.01, -0.02])
    result = neopax_module.build_ntx_neopax_scan_from_vmec_jax_boundary_params(
        context,
        params,
        rho=rho,
        nu_v=jnp.asarray([1.0e-3]),
        Es=jnp.ones((1, 1)),
        drds=jnp.ones(1),
        grid=GridSpec(5, 5, 4),
        source_name="boundary-scan",
        vmec_project=False,
        max_iter=3,
        step_size=0.25,
        ftol=1.0e-8,
        implicit="implicit",
        mboz=6,
        nboz=7,
        psi_p=1.5,
        min_bmn_to_load=2.0e-6,
    )

    assert result.source_name == "boundary-scan"
    assert calls["context"] is context
    assert jnp.allclose(calls["params"], params)
    assert calls["build_surfaces"] is surfaces
    assert calls["surface_kwargs"]["s_values"] == pytest.approx((0.16,))
    assert calls["surface_kwargs"]["vmec_project"] is False
    assert calls["surface_kwargs"]["max_iter"] == 3
    assert calls["surface_kwargs"]["step_size"] == 0.25
    assert calls["surface_kwargs"]["ftol"] == 1.0e-8
    assert calls["surface_kwargs"]["implicit"] == "implicit"
    assert calls["surface_kwargs"]["mboz"] == 6
    assert calls["surface_kwargs"]["nboz"] == 7
    assert calls["surface_kwargs"]["psi_p"] == 1.5
    assert calls["surface_kwargs"]["min_bmn_to_load"] == 2.0e-6
    assert jnp.allclose(calls["build_kwargs"]["rho"], rho)


def test_build_ntx_neopax_scan_derives_missing_field_channel():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    from_er = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Er=er,
        drds=drds,
        grid=grid,
    )
    from_es = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        drds=drds,
        grid=grid,
    )

    assert jnp.allclose(from_er.Er, er)
    assert jnp.allclose(from_es.Es, es)
    assert from_er.Es.shape == er.shape
    assert from_es.Er.shape == es.shape

    callback_from_er = build_ntx_neopax_scan(
        lambda _: example_surface(),
        rho=rho,
        nu_v=nu_v,
        Er=er,
        drds=drds,
        grid=grid,
    )
    callback_from_es = build_ntx_neopax_scan(
        lambda _: example_surface(),
        rho=rho,
        nu_v=nu_v,
        Es=es,
        drds=drds,
        grid=grid,
    )
    assert callback_from_er.Es.shape == er.shape
    assert callback_from_es.Er.shape == es.shape


def test_build_ntx_neopax_scan_from_surfaces_validates_shape_mismatches():
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)
    surfaces = (example_surface(),)
    with pytest.raises(ValueError, match="number of surfaces must match rho length"):
        build_ntx_neopax_scan_from_surfaces(
            surfaces,
            rho=rho,
            nu_v=nu_v,
            Er=jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]]),
            drds=drds,
            grid=grid,
        )
    with pytest.raises(ValueError, match="drds must have the same length as rho"):
        build_ntx_neopax_scan_from_surfaces(
            (example_surface(), example_surface()),
            rho=rho,
            nu_v=nu_v,
            Er=jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]]),
            drds=jnp.asarray([1.0]),
            grid=grid,
        )
    with pytest.raises(ValueError, match="set at least one of Es or Er"):
        build_ntx_neopax_scan_from_surfaces(
            (example_surface(), example_surface()),
            rho=rho,
            nu_v=nu_v,
            drds=drds,
            grid=grid,
        )
    with pytest.raises(ValueError, match="Es and Er must have the same shape"):
        build_ntx_neopax_scan_from_surfaces(
            (example_surface(), example_surface()),
            rho=rho,
            nu_v=nu_v,
            Es=jnp.asarray([[0.0, 1.0e-3]]),
            Er=jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]]),
            drds=drds,
            grid=grid,
        )
    with pytest.raises(ValueError, match="Es/Er first dimension must match rho"):
        build_ntx_neopax_scan_from_surfaces(
            (example_surface(), example_surface()),
            rho=rho,
            nu_v=nu_v,
            Es=jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3], [0.0, 3.0e-3]]),
            Er=jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3], [0.0, 3.0e-3]]),
            drds=drds,
            grid=grid,
        )


def test_scan_to_neopax_arrays_matches_expected_scalings():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
    )
    mapped = scan_to_neopax_arrays(scan, a_b=1.0)

    assert mapped.D11_log.shape == scan.D11.shape
    assert mapped.D13.shape == scan.D13.shape
    assert mapped.D33.shape == scan.D33.shape
    assert jnp.allclose(mapped.nu_log, jnp.log10(nu_v))
    assert jnp.allclose(mapped.D11_log, jnp.log10(scan.D11 * drds[:, None, None] ** 2))
    assert jnp.allclose(mapped.D13, scan.D13 * drds[:, None, None])
    assert scan.D33_spitzer is not None
    assert jnp.allclose(mapped.D33, scan.D33_spitzer * nu_v[None, :, None])


def test_scan_to_neopax_arrays_keeps_d13_database_convention_without_bridge_metadata():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
    )
    scan = scan.__class__(
        **{
            **scan.__dict__,
            "fac_reference_to_sfincs_31": None,
            "fac_sfincs_to_dkes_31": None,
            "D33_spitzer": None,
        }
    )
    mapped = scan_to_neopax_arrays(scan, a_b=1.0)

    assert jnp.allclose(mapped.D11_log, jnp.log10(scan.D11 * drds[:, None, None] ** 2))
    assert jnp.allclose(mapped.D13, scan.D13 * drds[:, None, None])
    assert jnp.allclose(mapped.D33, scan.D33 * nu_v[None, :, None])


def test_scan_to_neopax_arrays_supports_raw_d33_mode():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
    )
    mapped = scan_to_neopax_arrays(scan, a_b=1.0, d33_mode="raw")

    assert jnp.allclose(mapped.D33, scan.D33 * nu_v[None, :, None])


def test_scan_to_neopax_arrays_supports_conductivity_difference_d33_mode():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
    )
    mapped = scan_to_neopax_arrays(scan, a_b=1.0, d33_mode="conductivity_difference")

    assert scan.D33_spitzer is not None
    assert jnp.allclose(
        mapped.D33,
        (scan.D33_spitzer - scan.D33) * nu_v[None, :, None],
    )


def test_conductivity_difference_d33_mode_requires_spitzer_branch():
    surfaces = (example_surface(),)
    rho = jnp.asarray([0.25])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3]])
    drds = jnp.asarray([1.0])
    grid = GridSpec(5, 5, 4)

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
    )
    scan = scan.__class__(**{**scan.__dict__, "D33_spitzer": None})

    try:
        scan_to_neopax_arrays(scan, a_b=1.0, d33_mode="conductivity_difference")
    except ValueError as exc:
        assert "requires D33_spitzer" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for conductivity_difference without D33_spitzer")

def test_scan_to_neopax_arrays_is_differentiable_in_es():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    er = jnp.asarray([[0.0, 1.0e-3], [0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    grid = GridSpec(5, 5, 4)

    def objective(flat_es):
        es = flat_es.reshape(2, 2)
        scan = build_ntx_neopax_scan_from_surfaces(
            surfaces,
            rho=rho,
            nu_v=nu_v,
            Es=es,
            Er=er,
            drds=drds,
            grid=grid,
        )
        mapped = scan_to_neopax_arrays(scan, a_b=1.0)
        return jnp.sum(mapped.D13) + jnp.sum(mapped.D11_log)

    grad = jax.grad(objective)(jnp.asarray([0.0, 1.0e-3, 0.0, 2.0e-3]))
    assert grad.shape == (4,)
    assert jnp.all(jnp.isfinite(grad))


def test_vmec_bridge_uses_covariant_boozer_zero_mode():
    surface = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25)
    zero_mode = jnp.asarray((surface.m == 0) & (surface.n == 0))
    idx = int(jnp.argmax(zero_mode))

    bridge = _surface_reference_bridge(surface)

    assert bridge["boozer_i"] == jnp.asarray(surface.b_sub_theta_cos[idx])
    assert bridge["boozer_g"] == jnp.asarray(surface.b_sub_zeta_cos[idx])


def test_vmec_scan_derives_es_from_er_using_transport_scale():
    surface = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25)
    rho = jnp.asarray([0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    er = jnp.asarray([[0.0, 1.0e-3, 2.0e-3]])
    drds = jnp.asarray([1.0])
    grid = GridSpec(5, 5, 4)

    implicit_es = build_ntx_neopax_scan_from_surfaces(
        (surface,),
        rho=rho,
        nu_v=nu_v,
        Er=er,
        drds=drds,
        grid=grid,
    )
    explicit_es = er / jnp.asarray(surface.transport_psi_scale)
    explicit = build_ntx_neopax_scan_from_surfaces(
        (surface,),
        rho=rho,
        nu_v=nu_v,
        Es=explicit_es,
        Er=er,
        drds=drds,
        grid=grid,
    )

    assert jnp.allclose(implicit_es.Es, explicit_es)
    assert jnp.allclose(implicit_es.D11, explicit.D11)
    assert jnp.allclose(implicit_es.D13, explicit.D13)
    assert jnp.allclose(implicit_es.D33, explicit.D33)


def test_vmec_scan_reference_bridge_is_jax_traceable():
    surface = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25)
    rho = jnp.asarray([0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    er = jnp.asarray([[0.0, 1.0e-3, 2.0e-3]])
    drds = jnp.asarray([1.0])
    grid = GridSpec(5, 5, 4)

    def objective(scale):
        scaled_surface = replace(
            surface,
            b_cos=surface.b_cos * (1.0 + 0.05 * scale),
        )
        scan = build_ntx_neopax_scan_from_surfaces(
            (scaled_surface,),
            rho=rho,
            nu_v=nu_v,
            Er=er,
            drds=drds,
            grid=grid,
        )
        return jnp.sum(scan.D11) + jnp.sum(scan.D13) + jnp.sum(scan.D33)

    grad = jax.grad(objective)(0.0)
    assert jnp.isfinite(grad)


def test_neopax_scan_requires_rebuild_for_legacy_cache_without_d33_spitzer(tmp_path: Path):
    surfaces = (example_surface(),)
    rho = jnp.asarray([0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3]])
    drds = jnp.asarray([1.0])
    grid = GridSpec(5, 5, 4)

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
    )
    path = tmp_path / "scan.h5"
    write_neopax_scan_hdf5(scan, path)
    assert not neopax_scan_requires_rebuild(path)

    import h5py

    with h5py.File(path, "a") as handle:
        del handle["D33_spitzer"]
        del handle.attrs["format_version"]
    assert neopax_scan_requires_rebuild(path)


def test_neopax_scan_roundtrip_preserves_optional_attrs_and_missing_path_requires_rebuild(
    tmp_path: Path,
):
    surfaces = (example_surface(),)
    rho = jnp.asarray([0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2])
    es = jnp.asarray([[0.0, 1.0e-3]])
    er = jnp.asarray([[0.0, 1.0e-3]])
    drds = jnp.asarray([1.0])
    grid = GridSpec(5, 5, 4)

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=grid,
    )
    scan = replace(scan, a_b=1.7, psia=2.3, source_name="roundtrip-unit")
    path = tmp_path / "scan_attrs.h5"
    write_neopax_scan_hdf5(scan, path)

    restored = load_neopax_reference_scan(path)
    assert restored.source_name == "roundtrip-unit"
    assert jnp.allclose(restored.D11, scan.D11)
    assert jnp.allclose(restored.D33, scan.D33)

    import h5py

    with h5py.File(path, "r") as handle:
        assert handle.attrs["a_b"] == pytest.approx(1.7)
        assert handle.attrs["psia"] == pytest.approx(2.3)

    assert neopax_scan_requires_rebuild(tmp_path / "missing_scan.h5")
