from __future__ import annotations

import jax.numpy as jnp
import pytest

from ntx import (
    BootstrapSpeciesProfile,
    GridSpec,
    PrimitiveSpeciesProfile,
    build_bootstrap_species_profile,
    build_ntx_neopax_scan_from_surfaces,
    evaluate_bootstrap_current,
    example_surface,
)


def _example_scan():
    surfaces = (example_surface(), example_surface())
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-2, 2.0e-2, 4.0e-2])
    es = jnp.asarray([[-1.0e-3, 0.0, 1.0e-3], [-2.0e-3, 0.0, 2.0e-3]])
    er = jnp.asarray([[-1.0e-3, 0.0, 1.0e-3], [-2.0e-3, 0.0, 2.0e-3]])
    drds = jnp.asarray([1.0, 1.5])
    return build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=GridSpec(5, 5, 4),
    )


def test_evaluate_bootstrap_current_vanishes_for_zero_forces():
    scan = _example_scan()
    rho = jnp.asarray(scan.rho)
    species_profiles = (
        BootstrapSpeciesProfile(
            mass_mp=1.0 / 1836.15267343,
            charge_qp=-1.0,
            density=jnp.asarray([3.0e19, 2.8e19]),
            temperature=jnp.asarray([1200.0, 1100.0]),
            A1=jnp.zeros_like(rho),
            A2=jnp.zeros_like(rho),
            A3=jnp.zeros_like(rho),
            name="e",
        ),
        BootstrapSpeciesProfile(
            mass_mp=1.0,
            charge_qp=1.0,
            density=jnp.asarray([3.0e19, 2.8e19]),
            temperature=jnp.asarray([1200.0, 1100.0]),
            A1=jnp.zeros_like(rho),
            A2=jnp.zeros_like(rho),
            A3=jnp.zeros_like(rho),
            name="i",
        ),
    )
    result = evaluate_bootstrap_current(
        scan,
        species_profiles,
        a_b=1.0,
        er_profile=jnp.zeros_like(rho),
    )
    assert jnp.allclose(result.particle_flux, 0.0)
    assert jnp.allclose(result.heat_flux, 0.0)
    assert jnp.allclose(result.parallel_flow, 0.0)
    assert jnp.allclose(result.current_density, 0.0)
    assert jnp.allclose(result.jdotb, 0.0)


def test_build_bootstrap_species_profile_from_primitives_is_finite():
    rho = jnp.asarray([0.2, 0.4, 0.6, 0.8])
    primitive = PrimitiveSpeciesProfile(
        charge=-1.0,
        nu_v=jnp.full_like(rho, 1.0e-3),
        density=jnp.asarray([3.1e19, 3.0e19, 2.7e19, 2.3e19]),
        temperature=jnp.asarray([1400.0, 1300.0, 1150.0, 950.0]),
        name="e",
    )
    species = build_bootstrap_species_profile(
        rho,
        primitive,
        mass_mp=1.0 / 1836.15267343,
        er_profile=jnp.asarray([0.6, 0.4, 0.2, 0.1]),
        a_b=0.55,
    )
    assert species.A1.shape == rho.shape
    assert species.A2.shape == rho.shape
    assert species.A3.shape == rho.shape
    assert jnp.all(jnp.isfinite(species.A1))
    assert jnp.all(jnp.isfinite(species.A2))
    assert jnp.allclose(species.A3, 0.0)


def test_evaluate_bootstrap_current_matches_frozen_regression_values():
    scan = _example_scan()
    rho = jnp.asarray(scan.rho)
    species_profiles = (
        BootstrapSpeciesProfile(
            mass_mp=1.0 / 1836.15267343,
            charge_qp=-1.0,
            density=jnp.asarray([3.0e19, 2.7e19]),
            temperature=jnp.asarray([1300.0, 1000.0]),
            A1=jnp.asarray([0.6, 0.4]),
            A2=jnp.asarray([-0.2, -0.1]),
            A3=jnp.zeros_like(rho),
            name="e",
        ),
        BootstrapSpeciesProfile(
            mass_mp=1.0,
            charge_qp=1.0,
            density=jnp.asarray([3.0e19, 2.7e19]),
            temperature=jnp.asarray([1300.0, 1000.0]),
            A1=jnp.asarray([0.25, 0.15]),
            A2=jnp.asarray([-0.08, -0.03]),
            A3=jnp.zeros_like(rho),
            name="i",
        ),
    )
    result = evaluate_bootstrap_current(
        scan,
        species_profiles,
        a_b=1.0,
        er_profile=jnp.asarray([0.0, 0.0]),
        n_x=16,
    )
    assert result.current_density == pytest.approx(
        jnp.asarray([-1.6737020812012965e4, -2.9713447976445940e3]),
        rel=5.0e-6,
        abs=5.0e-6,
    )
