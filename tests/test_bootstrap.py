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


def _compat_scan():
    surfaces = tuple(example_surface() for _ in range(4))
    rho = jnp.asarray([0.2, 0.4, 0.6, 0.8])
    nu_v = jnp.asarray([1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2])
    er_grid = jnp.stack(
        [jnp.asarray([-2.0e-3, -5.0e-4, 0.0, 5.0e-4, 2.0e-3])] * rho.size
    )
    return build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=er_grid,
        Er=er_grid,
        drds=0.5 / rho,
        grid=GridSpec(7, 7, 6),
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
        jnp.asarray([-1.8367934148385500e3, -2.8382846219716356e2]),
        rel=5.0e-6,
        abs=5.0e-6,
    )


def test_evaluate_bootstrap_current_neopax_compat_matches_frozen_profile():
    scan = _compat_scan()
    rho = jnp.asarray(scan.rho)
    species_profiles = (
        BootstrapSpeciesProfile(
            mass_mp=1.0 / 1836.15267343,
            charge_qp=-1.0,
            density=jnp.asarray([4.0e19, 3.6e19, 3.0e19, 2.4e19]),
            temperature=jnp.asarray([1500.0, 1300.0, 1100.0, 900.0]),
            A1=jnp.asarray([0.55, 0.42, 0.28, 0.18]),
            A2=jnp.asarray([-0.18, -0.13, -0.08, -0.04]),
            A3=jnp.zeros_like(rho),
            name="e",
        ),
        BootstrapSpeciesProfile(
            mass_mp=1.0,
            charge_qp=1.0,
            density=jnp.asarray([4.0e19, 3.6e19, 3.0e19, 2.4e19]),
            temperature=jnp.asarray([1400.0, 1200.0, 1000.0, 850.0]),
            A1=jnp.asarray([0.22, 0.16, 0.10, 0.06]),
            A2=jnp.asarray([-0.07, -0.05, -0.03, -0.015]),
            A3=jnp.zeros_like(rho),
            name="i",
        ),
    )
    result = evaluate_bootstrap_current(
        scan,
        species_profiles,
        a_b=1.0,
        er_profile=jnp.asarray([0.8, 0.5, 0.25, 0.1]),
        n_x=16,
        neopax_compat_boundary=True,
    )
    assert result.current_density == pytest.approx(
        jnp.asarray(
            [
                -1.4913648270543364e2,
                -3.1888869289650327e2,
                -1.6436991273249596e2,
                -1.2578820907120081e2,
            ]
        ),
        rel=5.0e-8,
        abs=5.0e-8,
    )


def test_evaluate_bootstrap_current_neopax_compat_overwrites_first_radial_block():
    scan = _compat_scan()
    rho = jnp.asarray(scan.rho)
    species_profiles = (
        BootstrapSpeciesProfile(
            mass_mp=1.0 / 1836.15267343,
            charge_qp=-1.0,
            density=jnp.asarray([4.0e19, 3.6e19, 3.0e19, 2.4e19]),
            temperature=jnp.asarray([1500.0, 1300.0, 1100.0, 900.0]),
            A1=jnp.asarray([0.55, 0.42, 0.28, 0.18]),
            A2=jnp.asarray([-0.18, -0.13, -0.08, -0.04]),
            A3=jnp.zeros_like(rho),
            name="e",
        ),
        BootstrapSpeciesProfile(
            mass_mp=1.0,
            charge_qp=1.0,
            density=jnp.asarray([4.0e19, 3.6e19, 3.0e19, 2.4e19]),
            temperature=jnp.asarray([1400.0, 1200.0, 1000.0, 850.0]),
            A1=jnp.asarray([0.22, 0.16, 0.10, 0.06]),
            A2=jnp.asarray([-0.07, -0.05, -0.03, -0.015]),
            A3=jnp.zeros_like(rho),
            name="i",
        ),
    )
    plain = evaluate_bootstrap_current(
        scan,
        species_profiles,
        a_b=1.0,
        er_profile=jnp.asarray([0.8, 0.5, 0.25, 0.1]),
        n_x=16,
    )
    compat = evaluate_bootstrap_current(
        scan,
        species_profiles,
        a_b=1.0,
        er_profile=jnp.asarray([0.8, 0.5, 0.25, 0.1]),
        n_x=16,
        neopax_compat_boundary=True,
    )
    assert compat.Lij[:, 0, :, :] == pytest.approx(compat.Lij[:, 1, :, :], rel=1.0e-10, abs=1.0e-10)
    assert plain.Lij[:, 0, :, :] != pytest.approx(plain.Lij[:, 1, :, :], rel=1.0e-4, abs=1.0e-4)
