from __future__ import annotations

import jax
import jax.numpy as jnp

from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    build_ntx_neopax_scan_from_surfaces,
    example_surface,
    load_vmec_surface,
    scan_to_neopax_arrays,
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
    assert scan.fac_reference_to_sfincs_31 is not None
    assert scan.fac_sfincs_to_dkes_31 is not None
    expected_d13 = -scan.D13 * (
        scan.fac_reference_to_sfincs_31[:, None, None]
        * scan.fac_sfincs_to_dkes_31[:, None, None]
    )
    assert jnp.allclose(mapped.D13, expected_d13)
    assert scan.D33_spitzer is not None
    assert jnp.allclose(mapped.D33, scan.D33_spitzer * nu_v[None, :, None])


def test_scan_to_neopax_arrays_falls_back_to_legacy_scalings_without_bridge_metadata():
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
    assert jnp.allclose(mapped.D13, -scan.D13 * drds[:, None, None])
    assert jnp.allclose(mapped.D33, scan.D33 * nu_v[None, :, None])


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
