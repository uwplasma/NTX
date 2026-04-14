from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from ntx import (
    AmbipolarProfileFamilyResult,
    AmbipolarProfileResult,
    GridSpec,
    MonoenergeticSpeciesProfile,
    PrimitiveProfileTransportIterationResult,
    PrimitiveSpeciesProfile,
    ProfileBasisControlSpec,
    ProfileBasisOptimizationResult,
    ProfileControlOptimizationResult,
    ProfileControlSpec,
    ProfileTransportClosureSpec,
    ProfileTransportIterationResult,
    advance_primitive_profile_transport,
    advance_profile_transport,
    ambipolar_residual_profile,
    apply_profile_basis_control,
    apply_profile_control,
    bootstrap_current_objective,
    build_ntx_neopax_scan_from_surfaces,
    build_species_profile_from_primitives,
    build_species_profiles_from_primitives,
    evaluate_scan_channel,
    evaluate_species_current_response,
    evaluate_species_particle_flux,
    example_surface,
    optimize_profile_basis_control,
    optimize_profile_control,
    profile_transport_loss,
    solve_ambipolar_er_profile,
    solve_ambipolar_profile_family,
    solve_primitive_profile_transport_loop,
    solve_profile_transport_loop,
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


def test_profile_family_solver_and_bootstrap_objective_return_finite_results():
    scan = _example_scan()
    control = jnp.asarray([-0.2, 0.0, 0.2])
    electron, ion = _species_profiles()
    family = tuple(
        (
            replace(electron, A3=electron.A3 * (1.0 + scale)),
            replace(ion, A1=ion.A1 * (1.0 - 0.5 * scale)),
        )
        for scale in control
    )
    result = solve_ambipolar_profile_family(scan, family, control=control, steps=6)
    assert isinstance(result, AmbipolarProfileFamilyResult)
    assert result.er_profile.shape == (control.size, scan.rho.size)
    assert result.bootstrap_current_proxy.shape == (control.size, scan.rho.size)
    objective = bootstrap_current_objective(scan.rho, result.bootstrap_current_proxy[1])
    assert jnp.isfinite(objective)


def test_profile_helpers_cover_error_branches_and_d31_fallback():
    scan = _example_scan()
    scan = replace(scan, D31=None)
    species = _species_profiles()[0]
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
    with pytest.raises(ValueError, match="bootstrap_current_proxy must match rho shape"):
        bootstrap_current_objective(
            scan.rho,
            jnp.asarray([1.0, 2.0]),
        )


def test_profile_control_application_and_optimization_return_finite_results():
    scan = _example_scan()
    species_profiles = _species_profiles()
    control_spec = ProfileControlSpec(
        a1_response=jnp.asarray([0.0, -0.5]),
        a3_response=jnp.asarray([1.0, 0.0]),
        control_name="shape",
    )
    controlled = apply_profile_control(species_profiles, 0.1, control_spec)
    assert controlled[0].A3.shape == species_profiles[0].A3.shape
    assert controlled[1].A1.shape == species_profiles[1].A1.shape
    result = optimize_profile_control(
        scan,
        species_profiles,
        control_spec,
        control_initial=0.2,
        optimization_steps=5,
        solve_steps=6,
        damping=0.7,
        residual_penalty=0.5,
        control_bound=0.3,
    )
    assert isinstance(result, ProfileControlOptimizationResult)
    assert result.control_history.shape == (5,)
    assert result.best_profile.er_profile.shape == scan.rho.shape
    assert jnp.all(jnp.isfinite(result.objective_history))
    assert jnp.isfinite(result.best_control)
    assert jnp.max(jnp.abs(result.control_history)) <= 0.3 + 1.0e-12


def test_profile_control_spec_shape_mismatch_raises():
    with pytest.raises(ValueError, match="control_spec must match the number of species"):
        apply_profile_control(
            _species_profiles(),
            0.0,
            ProfileControlSpec(
                a1_response=jnp.asarray([1.0]),
                a3_response=jnp.asarray([1.0]),
            ),
        )


def test_profile_basis_control_application_and_optimization_return_finite_results():
    scan = _example_scan()
    species_profiles = _species_profiles()
    basis = jnp.asarray(
        [
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0],
        ]
    )
    control_spec = ProfileBasisControlSpec(
        basis=basis,
        a1_response=jnp.asarray([[0.0, 0.0], [-0.4, -0.2]]),
        a3_response=jnp.asarray([[0.8, 0.3], [0.0, 0.0]]),
        control_name="basis",
    )
    controlled = apply_profile_basis_control(
        species_profiles,
        jnp.asarray([0.1, -0.05]),
        control_spec,
    )
    assert controlled[0].A3.shape == species_profiles[0].A3.shape
    assert controlled[1].A1.shape == species_profiles[1].A1.shape
    result = optimize_profile_basis_control(
        scan,
        species_profiles,
        control_spec,
        control_initial=jnp.asarray([0.15, -0.05]),
        optimization_steps=5,
        solve_steps=6,
        damping=0.7,
        residual_penalty=0.5,
        control_penalty=1.0e-2,
        control_bound=0.25,
    )
    assert isinstance(result, ProfileBasisOptimizationResult)
    assert result.control_history.shape == (5, 2)
    assert result.best_control.shape == (2,)
    assert result.best_profile.er_profile.shape == scan.rho.shape
    assert jnp.all(jnp.isfinite(result.objective_history))
    assert jnp.max(jnp.abs(result.control_history)) <= 0.25 + 1.0e-12


def test_profile_basis_control_shape_mismatch_raises():
    species_profiles = _species_profiles()
    basis = jnp.asarray([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="control must match the number of basis functions"):
        apply_profile_basis_control(
            species_profiles,
            jnp.asarray([0.1, 0.2]),
            ProfileBasisControlSpec(
                basis=basis,
                a1_response=jnp.asarray([[0.0], [0.0]]),
                a3_response=jnp.asarray([[1.0], [0.0]]),
            ),
        )


def test_profile_transport_loop_returns_finite_histories():
    scan = _example_scan()
    species_profiles = _species_profiles()
    closure = ProfileTransportClosureSpec(
        particle_relaxation=jnp.asarray([[0.10, 0.11, 0.12], [0.05, 0.06, 0.07]]),
        current_relaxation=jnp.asarray([[0.08, 0.07, 0.06], [0.03, 0.03, 0.03]]),
        particle_target=jnp.asarray([[0.01, 0.01, 0.01], [0.0, 0.0, 0.0]]),
        current_target=0.0,
        particle_source=jnp.asarray([[0.0, 0.0, 0.0], [0.005, 0.005, 0.005]]),
        normalization_floor=0.1,
        max_normalized_update=0.2,
    )
    profile = solve_ambipolar_er_profile(scan, species_profiles, steps=6)
    loss = profile_transport_loss(profile, closure)
    advanced = advance_profile_transport(species_profiles, profile, closure)
    result = solve_profile_transport_loop(
        scan,
        species_profiles,
        closure,
        iterations=4,
        solve_steps=6,
        damping=0.7,
    )
    assert jnp.isfinite(loss)
    assert advanced[0].A1.shape == species_profiles[0].A1.shape
    assert isinstance(result, ProfileTransportIterationResult)
    assert result.er_profile_history.shape == (4, scan.rho.size)
    assert result.bootstrap_current_proxy_history.shape == (4, scan.rho.size)
    assert result.species_a1_history.shape == (4, 2, scan.rho.size)
    assert result.species_a3_history.shape == (4, 2, scan.rho.size)
    assert jnp.all(jnp.isfinite(result.transport_loss_history))
    assert result.transport_loss_history[-1] <= result.transport_loss_history[0] + 1.0e-12


def test_profile_transport_closure_shape_mismatch_raises():
    scan = _example_scan()
    species_profiles = _species_profiles()
    profile = solve_ambipolar_er_profile(scan, species_profiles, steps=4)
    with pytest.raises(
        ValueError,
        match="transport field must be scalar, per-species, per-radius, or species-by-radius",
    ):
        advance_profile_transport(
            species_profiles,
            profile,
            ProfileTransportClosureSpec(
                particle_relaxation=jnp.asarray([0.1, 0.2, 0.3, 0.4]),
                current_relaxation=0.1,
            ),
        )


def test_build_species_profiles_from_primitives_returns_finite_forces():
    scan = _example_scan()
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


def test_primitive_profile_transport_loop_returns_finite_histories():
    scan = _example_scan()
    rho = jnp.asarray(scan.rho)
    primitives = (
        PrimitiveSpeciesProfile(
            charge=-1.0,
            nu_v=jnp.asarray([4.0e-4, 6.0e-4, 8.0e-4]),
            density=jnp.asarray([1.0, 0.95, 0.92]),
            temperature=jnp.asarray([1.1, 1.0, 0.96]),
            electrostatic_prefactor=0.15,
            name="electron",
        ),
        PrimitiveSpeciesProfile(
            charge=1.0,
            nu_v=jnp.asarray([2.0e-3, 2.5e-3, 3.0e-3]),
            density=jnp.asarray([0.92, 0.90, 0.88]),
            temperature=jnp.asarray([0.85, 0.82, 0.80]),
            electrostatic_prefactor=0.10,
            name="ion",
        ),
    )
    closure = ProfileTransportClosureSpec(
        particle_relaxation=0.04,
        current_relaxation=0.03,
        particle_target=0.0,
        current_target=0.0,
        particle_source=0.0,
        current_source=0.0,
        normalization_floor=0.05,
        max_normalized_update=0.15,
    )
    initial_profile = solve_ambipolar_er_profile(
        scan,
        build_species_profiles_from_primitives(rho, primitives, er_profile=jnp.zeros_like(rho)),
        steps=4,
        smoothing_strength=0.35,
    )
    advanced = advance_primitive_profile_transport(primitives, initial_profile, closure)
    result = solve_primitive_profile_transport_loop(
        scan,
        primitives,
        closure,
        iterations=4,
        solve_steps=4,
        damping=0.7,
        smoothing_strength=0.35,
    )
    assert advanced[0].density.shape == rho.shape
    assert isinstance(result, PrimitiveProfileTransportIterationResult)
    assert result.er_profile_history.shape == (4, rho.size)
    assert result.species_density_history.shape == (4, 2, rho.size)
    assert result.species_temperature_history.shape == (4, 2, rho.size)
    assert jnp.all(jnp.isfinite(result.transport_loss_history))
