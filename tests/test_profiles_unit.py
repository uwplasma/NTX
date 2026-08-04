from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import pytest

from ntx import (
    PrimitiveSpeciesProfile,
    ProfileBasisControlSpec,
    ProfileControlSpec,
    ProfileTransportClosureSpec,
    ambipolar_residual_profile,
    apply_profile_basis_control,
    apply_profile_control,
    bootstrap_current_objective,
    build_species_profile_from_primitives,
    build_species_profiles_from_primitives,
    current_response_objective,
    evaluate_scan_channel,
    evaluate_species_current_response,
    evaluate_species_particle_flux,
)
from ntx._profiles import _channel_data
from ntx._profiles import _single_radius_profile, _smooth_radial_profile
from ntx._profiles_transport import (
    _broadcast_species_transport_field,
    _scaled_transport_closure,
)

from ._profile_test_helpers import example_scan, species_profiles


def test_profile_interpolators_return_finite_arrays():
    scan = example_scan()
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
    scan = example_scan()
    species = species_profiles()[0]
    er_profile = jnp.asarray([-4.0e-4, 0.0, 4.0e-4])
    flux = evaluate_species_particle_flux(scan, species, er_profile=er_profile)
    current = evaluate_species_current_response(scan, species, er_profile=er_profile)
    assert flux.shape == scan.rho.shape
    assert current.shape == scan.rho.shape
    assert jnp.all(jnp.isfinite(flux))
    assert jnp.all(jnp.isfinite(current))


def test_charge_symmetric_pair_has_zero_ambipolar_residual():
    scan = example_scan()
    base_species = species_profiles()[0]
    ion = replace(base_species, charge=1.0, name="ion")
    electron = replace(base_species, charge=-1.0, name="electron")
    er_profile = jnp.asarray([-4.0e-4, 0.0, 4.0e-4])

    residual = ambipolar_residual_profile(
        scan,
        (ion, electron),
        er_profile=er_profile,
    )

    assert residual.shape == scan.rho.shape
    assert jnp.allclose(residual, 0.0, atol=1.0e-14)


def test_profile_helpers_cover_error_branches_and_d31_fallback():
    scan = example_scan()
    scan = replace(scan, D31=None)
    species = species_profiles()[0]
    er_profile = jnp.asarray([-4.0e-4, 0.0, 4.0e-4])
    current = evaluate_species_current_response(scan, species, er_profile=er_profile)
    assert current.shape == scan.rho.shape
    with pytest.raises(ValueError, match="unsupported channel"):
        evaluate_scan_channel(
            scan,
            "bad",
            jnp.asarray(scan.rho),
            species.nu_v,
            er_profile,
        )
    with pytest.raises(ValueError, match="rho must match scan.rho shape"):
        evaluate_scan_channel(
            scan,
            "D11",
            jnp.asarray([0.5]),
            jnp.asarray([1.0e-3]),
            jnp.asarray([0.0]),
        )
    with pytest.raises(ValueError, match="profile field must be scalar or match rho shape"):
        evaluate_species_particle_flux(
            scan,
            replace(species, A1=jnp.asarray([1.0, 2.0])),
            er_profile=er_profile,
        )
    with pytest.raises(ValueError, match="current_response must match rho shape"):
        bootstrap_current_objective(
            scan.rho,
            jnp.asarray([1.0, 2.0]),
        )


def test_profile_control_spec_shape_mismatch_raises():
    with pytest.raises(ValueError, match="control_spec must match the number of species"):
        apply_profile_control(
            species_profiles(),
            0.0,
            ProfileControlSpec(
                a1_response=jnp.asarray([1.0]),
                a3_response=jnp.asarray([1.0]),
            ),
        )


def test_profile_basis_control_shape_mismatch_raises():
    basis = jnp.asarray([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="control must match the number of basis functions"):
        apply_profile_basis_control(
            species_profiles(),
            jnp.asarray([0.1, 0.2]),
            ProfileBasisControlSpec(
                basis=basis,
                a1_response=jnp.asarray([[0.0], [0.0]]),
                a3_response=jnp.asarray([[1.0], [0.0]]),
            ),
        )


def test_profile_control_a3_shape_mismatch_raises():
    with pytest.raises(ValueError, match="control_spec must match the number of species"):
        apply_profile_control(
            species_profiles(),
            0.0,
            ProfileControlSpec(
                a1_response=jnp.asarray([1.0, 1.0]),
                a3_response=jnp.asarray([1.0]),
            ),
        )


def test_profile_scalar_control_is_identity_at_zero_and_linear_in_response():
    profiles = species_profiles()
    control_spec = ProfileControlSpec(
        a1_response=jnp.asarray([0.2, -0.3]),
        a3_response=jnp.asarray([0.5, 0.0]),
    )

    identity = apply_profile_control(profiles, 0.0, control_spec)
    controlled = apply_profile_control(profiles, 0.25, control_spec)

    for baseline, same in zip(profiles, identity, strict=True):
        assert jnp.allclose(same.A1, baseline.A1)
        assert jnp.allclose(same.A3, baseline.A3)
    assert jnp.allclose(controlled[0].A1, profiles[0].A1 * (1.0 + 0.25 * 0.2))
    assert jnp.allclose(controlled[0].A3, profiles[0].A3 * (1.0 + 0.25 * 0.5))
    assert jnp.allclose(controlled[1].A1, profiles[1].A1 * (1.0 - 0.25 * 0.3))
    assert jnp.allclose(controlled[1].A3, profiles[1].A3)


def test_profile_basis_control_is_identity_at_zero_and_linear_in_basis():
    profiles = species_profiles()
    basis = jnp.asarray(
        [
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0],
        ]
    )
    control = jnp.asarray([0.2, -0.1])
    control_spec = ProfileBasisControlSpec(
        basis=basis,
        a1_response=jnp.asarray([[0.3, 0.0], [0.0, -0.2]]),
        a3_response=jnp.asarray([[0.5, 0.1], [0.0, 0.0]]),
    )

    identity = apply_profile_basis_control(profiles, jnp.zeros(2), control_spec)
    controlled = apply_profile_basis_control(profiles, control, control_spec)

    for baseline, same in zip(profiles, identity, strict=True):
        assert jnp.allclose(same.A1, baseline.A1)
        assert jnp.allclose(same.A3, baseline.A3)
    species0_a1_modifier = jnp.tensordot(control * control_spec.a1_response[0], basis, axes=1)
    species0_a3_modifier = jnp.tensordot(control * control_spec.a3_response[0], basis, axes=1)
    species1_a1_modifier = jnp.tensordot(control * control_spec.a1_response[1], basis, axes=1)
    assert jnp.allclose(controlled[0].A1, profiles[0].A1 * (1.0 + species0_a1_modifier))
    assert jnp.allclose(controlled[0].A3, profiles[0].A3 * (1.0 + species0_a3_modifier))
    assert jnp.allclose(controlled[1].A1, profiles[1].A1 * (1.0 + species1_a1_modifier))
    assert jnp.allclose(controlled[1].A3, profiles[1].A3)


def test_profile_basis_species_and_response_shape_mismatch_raises():
    profiles = species_profiles()
    basis = jnp.asarray([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="control_spec must match the number of species"):
        apply_profile_basis_control(
            profiles,
            jnp.asarray([0.1]),
            ProfileBasisControlSpec(
                basis=basis,
                a1_response=jnp.asarray([[0.0], [0.0]]),
                a3_response=jnp.asarray([[1.0]]),
            ),
        )
    with pytest.raises(
        ValueError,
        match="response matrices must match the number of basis functions",
    ):
        apply_profile_basis_control(
            profiles,
            jnp.asarray([0.1]),
            ProfileBasisControlSpec(
                basis=basis,
                a1_response=jnp.asarray([[0.0, 0.0], [0.0, 0.0]]),
                a3_response=jnp.asarray([[1.0, 0.0], [0.0, 0.0]]),
            ),
        )


def test_build_species_profiles_from_primitives_returns_finite_forces():
    scan = example_scan()
    rho = jnp.asarray(scan.rho)
    primitive = PrimitiveSpeciesProfile(
        charge=-1.0,
        nu_v=jnp.asarray([4.0e-4, 6.0e-4, 8.0e-4]),
        density=jnp.asarray([1.0, 0.95, 0.90]),
        temperature=jnp.asarray([1.1, 1.0, 0.92]),
        electrostatic_prefactor=0.2,
        name="electron",
    )
    species = build_species_profile_from_primitives(
        rho,
        primitive,
        er_profile=jnp.asarray([-3.0e-4, 0.0, 3.0e-4]),
    )
    family = build_species_profiles_from_primitives(
        rho,
        (primitive,),
        er_profile=jnp.asarray([-3.0e-4, 0.0, 3.0e-4]),
    )
    assert species.A1.shape == rho.shape
    assert species.A3.shape == rho.shape
    assert jnp.all(jnp.isfinite(species.A1))
    assert jnp.all(jnp.isfinite(species.A3))
    assert len(family) == 1


def test_primitive_profile_force_reconstruction_matches_thermodynamic_formula():
    rho = jnp.linspace(0.2, 0.8, 5)
    er_profile = jnp.linspace(-2.0e-3, 2.0e-3, rho.size)
    electrostatic_prefactor = 0.25
    charge = -2.0
    primitive = PrimitiveSpeciesProfile(
        charge=charge,
        nu_v=1.0e-3,
        density=jnp.ones_like(rho),
        temperature=jnp.ones_like(rho),
        electrostatic_prefactor=electrostatic_prefactor,
    )

    species = build_species_profile_from_primitives(
        rho,
        primitive,
        er_profile=er_profile,
    )

    assert jnp.allclose(species.A3, 0.0)
    assert jnp.allclose(species.A1, electrostatic_prefactor * charge * er_profile)

    density_gradient = 0.7
    temperature_gradient = -0.2
    exponential_primitive = PrimitiveSpeciesProfile(
        charge=charge,
        nu_v=1.0e-3,
        density=jnp.exp(density_gradient * rho),
        temperature=jnp.exp(temperature_gradient * rho),
        electrostatic_prefactor=0.0,
    )
    exponential_species = build_species_profile_from_primitives(
        rho,
        exponential_primitive,
        er_profile=jnp.zeros_like(rho),
    )
    center = rho.size // 2

    assert exponential_species.A3[center] == pytest.approx(temperature_gradient)
    assert exponential_species.A1[center] == pytest.approx(
        density_gradient - 1.5 * temperature_gradient,
    )


def test_bootstrap_objective_accepts_explicit_weight():
    rho = jnp.asarray([0.25, 0.5, 0.75])
    current = jnp.asarray([1.0, -0.5, 0.25])
    weight = jnp.asarray([1.0, 2.0, 3.0])
    value = bootstrap_current_objective(rho, current, weight=weight)
    alias_value = current_response_objective(rho, current, weight=weight)
    expected = jnp.trapezoid(weight * current**2, rho)
    assert jnp.isclose(value, expected)
    assert jnp.isclose(alias_value, expected)


def test_profile_eval_internal_helpers_cover_remaining_branches():
    scan = example_scan()
    d31 = _channel_data(scan, "D31")
    assert d31.shape == scan.D13.shape

    rho = jnp.asarray([0.25, 0.5, 0.75])
    profile = jnp.asarray([0.1, 0.2, 0.3])
    updated = _single_radius_profile(rho, jnp.asarray(0.48), profile, jnp.asarray(-0.4))
    assert jnp.allclose(updated, jnp.asarray([0.1, -0.4, 0.3]))

    short = jnp.asarray([1.0, 2.0])
    assert jnp.allclose(_smooth_radial_profile(short, jnp.asarray(0.7)), short)
    with pytest.raises(ValueError, match="values must be one-dimensional"):
        _smooth_radial_profile(jnp.ones((2, 2)), jnp.asarray(0.1))


def test_broadcast_species_transport_field_covers_vector_branches():
    rho = jnp.asarray([0.25, 0.5, 0.75])
    per_species = _broadcast_species_transport_field(jnp.asarray([1.0, 2.0]), 2, rho)
    per_radius = _broadcast_species_transport_field(jnp.asarray([0.1, 0.2, 0.3]), 2, rho)
    assert per_species.shape == (2, 3)
    assert per_radius.shape == (2, 3)
    assert jnp.allclose(per_species[1], jnp.asarray([2.0, 2.0, 2.0]))
    assert jnp.allclose(per_radius[0], jnp.asarray([0.1, 0.2, 0.3]))


def test_transport_closure_scaling_only_changes_relaxation_terms():
    closure = ProfileTransportClosureSpec(
        particle_relaxation=jnp.asarray([0.2, 0.4]),
        current_relaxation=0.3,
        particle_target=0.1,
        density_relaxation=0.05,
        temperature_relaxation=0.07,
        closure_name="unit closure",
    )

    scaled = _scaled_transport_closure(closure, jnp.asarray(0.5))

    assert jnp.allclose(scaled.particle_relaxation, jnp.asarray([0.1, 0.2]))
    assert jnp.allclose(scaled.current_relaxation, 0.15)
    assert jnp.allclose(scaled.density_relaxation, 0.025)
    assert jnp.allclose(scaled.temperature_relaxation, 0.035)
    assert scaled.particle_target == closure.particle_target
    assert scaled.closure_name == closure.closure_name
