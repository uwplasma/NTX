from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from ntx import (
    AmbipolarProfileResult,
    GridSpec,
    MonoenergeticSpeciesProfile,
    ambipolar_residual_profile,
    build_ntx_neopax_scan_from_surfaces,
    evaluate_scan_channel,
    evaluate_species_current_response,
    evaluate_species_particle_flux,
    example_surface,
    solve_ambipolar_er_profile,
)


def _example_scan():
    base = example_surface()
    rho = jnp.asarray([0.25, 0.5, 0.75])
    surfaces = tuple(
        replace(base, b_cos=base.b_cos.at[1].set(base.b_cos[1] * (1.0 + 0.15 * float(r))))
        for r in rho
    )
    nu_v = jnp.asarray([3.0e-4, 1.0e-3, 3.0e-3])
    er_axis = jnp.asarray([-2.0e-3, -5.0e-4, 0.0, 5.0e-4, 2.0e-3])
    er = jnp.tile(er_axis[None, :], (rho.size, 1))
    return build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=er,
        Er=er,
        drds=jnp.ones_like(rho),
        grid=GridSpec(5, 5, 4),
        source_name="profile_test",
    )


def _species_profiles():
    return (
        MonoenergeticSpeciesProfile(
            charge=-1.0,
            nu_v=jnp.asarray([4.0e-4, 6.0e-4, 8.0e-4]),
            A1=jnp.asarray([1.1, 1.0, 0.9]),
            A3=jnp.asarray([0.55, 0.5, 0.45]),
            current_weight=-1.0,
            name="electron",
        ),
        MonoenergeticSpeciesProfile(
            charge=1.0,
            nu_v=jnp.asarray([2.0e-3, 2.5e-3, 3.0e-3]),
            A1=jnp.asarray([0.7, 0.8, 0.9]),
            A3=jnp.asarray([0.25, 0.25, 0.25]),
            particle_weight=1.1,
            current_weight=1.0,
            name="ion",
        ),
    )


def test_profile_interpolators_return_finite_arrays():
    scan = _example_scan()
    rho = jnp.asarray(scan.rho)
    er_profile = jnp.asarray([-4.0e-4, 0.0, 4.0e-4])
    values = evaluate_scan_channel(
        scan,
        "D33",
        rho,
        jnp.asarray([5.0e-4, 1.0e-3, 2.0e-3]),
        er_profile,
    )
    assert values.shape == rho.shape
    assert jnp.all(jnp.isfinite(values))


def test_species_flux_and_current_shapes_are_consistent():
    scan = _example_scan()
    species = _species_profiles()[0]
    er_profile = jnp.asarray([-4.0e-4, 0.0, 4.0e-4])
    flux = evaluate_species_particle_flux(scan, species, er_profile=er_profile)
    current = evaluate_species_current_response(scan, species, er_profile=er_profile)
    assert flux.shape == scan.rho.shape
    assert current.shape == scan.rho.shape
    assert jnp.all(jnp.isfinite(flux))
    assert jnp.all(jnp.isfinite(current))


def test_ambipolar_profile_solver_returns_finite_result_and_reduces_loss():
    scan = _example_scan()
    result = solve_ambipolar_er_profile(
        scan,
        _species_profiles(),
        steps=8,
    )
    assert isinstance(result, AmbipolarProfileResult)
    assert result.er_profile.shape == scan.rho.shape
    assert result.bootstrap_current_proxy.shape == scan.rho.shape
    assert result.species_particle_flux.shape[1] == scan.rho.shape[0]
    assert jnp.all(jnp.isfinite(result.er_profile))
    assert jnp.all(jnp.isfinite(result.bootstrap_current_proxy))
    assert result.loss_history[-1] <= result.loss_history[0] + 1e-12


def test_ambipolar_residual_and_solver_are_differentiable():
    scan = _example_scan()
    species_profiles = _species_profiles()

    def objective(scale):
        scaled_species = (
            replace(species_profiles[0], A1=species_profiles[0].A1 * scale),
            species_profiles[1],
        )
        result = solve_ambipolar_er_profile(scan, scaled_species, steps=6)
        return jnp.sum(result.bootstrap_current_proxy**2) + jnp.sum(result.ambipolar_residual**2)

    gradient = jax.grad(objective)(1.0)
    assert jnp.isfinite(gradient)


def test_ambipolar_residual_profile_has_expected_shape():
    scan = _example_scan()
    er_profile = jnp.asarray([-3.0e-4, 0.0, 3.0e-4])
    residual = ambipolar_residual_profile(scan, _species_profiles(), er_profile=er_profile)
    assert residual.shape == scan.rho.shape
    assert jnp.all(jnp.isfinite(residual))
