"""Ambipolar-profile solvers built on NTX scan data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ._profiles_ambipolar_types import (
    AmbipolarProfileFamilyResult,
    AmbipolarProfileResult,
)
from ._profiles_channels import (
    _channel_data,
    ambipolar_residual_profile,
    evaluate_scan_channel,
    evaluate_species_current_response,
    evaluate_species_particle_flux,
)
from ._profiles_primitives import (
    build_species_profile_from_primitives,
    build_species_profiles_from_primitives,
)
from ._profiles_radial import (
    _broadcast_profile_field,
    _single_radius_profile,
    _smooth_radial_profile,
)
from ._profiles_species_types import (
    MonoenergeticSpeciesProfile,
)
from .neopax import NeopaxScan

__all__ = [
    "_broadcast_profile_field",
    "_channel_data",
    "_single_radius_profile",
    "_smooth_radial_profile",
    "ambipolar_residual_profile",
    "bootstrap_current_objective",
    "build_species_profile_from_primitives",
    "build_species_profiles_from_primitives",
    "current_response_objective",
    "evaluate_scan_channel",
    "evaluate_species_current_response",
    "evaluate_species_particle_flux",
    "solve_ambipolar_er_profile",
    "solve_ambipolar_profile_family",
]


def solve_ambipolar_er_profile(
    scan: NeopaxScan,
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    *,
    er_initial: Array | None = None,
    steps: int = 16,
    damping: float = 0.8,
    smoothing_strength: float = 0.0,
) -> AmbipolarProfileResult:
    """Solve a smooth ambipolar `E_r(r)` profile from an NTX scan."""

    rho = jnp.asarray(scan.rho)
    dtype = rho.dtype
    er_min = jnp.min(jnp.asarray(scan.Er), axis=1)
    er_max = jnp.max(jnp.asarray(scan.Er), axis=1)
    damping_value = jnp.asarray(damping, dtype=dtype)
    smoothing_value = jnp.clip(jnp.asarray(smoothing_strength, dtype=dtype), 0.0, 1.0)
    er_scale = jnp.maximum(jnp.mean(er_max - er_min), jnp.asarray(1.0e-8, dtype=dtype))
    if er_initial is None:
        er0 = 0.5 * (er_min + er_max)
    else:
        er0 = jnp.clip(_broadcast_profile_field(er_initial, rho), er_min, er_max)

    def residual_at_profile(er_profile):
        return ambipolar_residual_profile(
            scan,
            species_profiles,
            er_profile=er_profile,
        )

    def smoothness_penalty(er_profile):
        if er_profile.shape[0] < 3:
            return jnp.asarray(0.0, dtype=dtype)
        first_diff = jnp.diff(er_profile)
        second_diff = jnp.diff(first_diff)
        return (
            jnp.mean(first_diff**2) / (er_scale**2)
            + 0.5 * jnp.mean(second_diff**2) / (er_scale**2)
        )

    def profile_loss(er_profile):
        residual = residual_at_profile(er_profile)
        return jnp.mean(residual**2) + smoothing_value * smoothness_penalty(er_profile)

    def profile_update(carry, _):
        er_profile = carry
        loss, gradient = jax.value_and_grad(profile_loss)(er_profile)
        grad_norm = jnp.maximum(jnp.linalg.norm(gradient), jnp.asarray(1.0e-12, dtype=dtype))

        def backtrack_step(step_index, state):
            best_profile, best_loss, accepted = state
            factor = 0.5**step_index
            candidate = jnp.clip(
                er_profile - factor * damping_value * er_scale * gradient / grad_norm,
                er_min,
                er_max,
            )
            candidate = _smooth_radial_profile(candidate, 0.35 * smoothing_value)
            candidate_loss = profile_loss(candidate)
            take = (~accepted) & (candidate_loss <= loss)
            next_profile = jnp.where(take, candidate, best_profile)
            next_loss = jnp.where(take, candidate_loss, best_loss)
            next_accepted = accepted | take
            return next_profile, next_loss, next_accepted

        initial_candidate = jnp.clip(
            er_profile - damping_value * er_scale * gradient / grad_norm,
            er_min,
            er_max,
        )
        initial_candidate = _smooth_radial_profile(initial_candidate, 0.35 * smoothing_value)
        initial_loss = profile_loss(initial_candidate)
        next_profile, next_loss, accepted = jax.lax.fori_loop(
            1,
            6,
            backtrack_step,
            (initial_candidate, initial_loss, initial_loss <= loss),
        )
        next_profile = jnp.where(accepted, next_profile, er_profile)
        next_loss = jnp.where(accepted, next_loss, loss)
        return next_profile, (next_profile, next_loss)

    solved_profile, history = jax.lax.scan(profile_update, er0, xs=None, length=steps)
    _, loss_history = history
    residual = residual_at_profile(solved_profile)
    species_flux = jnp.stack(
        [
            evaluate_species_particle_flux(scan, species, rho=rho, er_profile=solved_profile)
            for species in species_profiles
        ]
    )
    species_current = jnp.stack(
        [
            evaluate_species_current_response(scan, species, rho=rho, er_profile=solved_profile)
            for species in species_profiles
        ]
    )
    bootstrap_current = jnp.sum(species_current, axis=0)
    return AmbipolarProfileResult(
        rho=rho,
        er_profile=solved_profile,
        ambipolar_residual=residual,
        bootstrap_current_proxy=bootstrap_current,
        species_particle_flux=species_flux,
        species_current_response=species_current,
        loss_history=loss_history,
    )


def solve_ambipolar_profile_family(
    scan: NeopaxScan,
    species_profiles_family: tuple[tuple[MonoenergeticSpeciesProfile, ...], ...],
    *,
    control: Array | None = None,
    er_initial: Array | None = None,
    steps: int = 16,
    damping: float = 0.8,
    smoothing_strength: float = 0.0,
) -> AmbipolarProfileFamilyResult:
    """Solve a family of ambipolar profiles across explicit profile controls."""

    family_results = [
        solve_ambipolar_er_profile(
            scan,
            species_profiles,
            er_initial=er_initial,
            steps=steps,
            damping=damping,
            smoothing_strength=smoothing_strength,
        )
        for species_profiles in species_profiles_family
    ]
    if control is None:
        control_array = jnp.arange(len(family_results), dtype=jnp.asarray(scan.rho).dtype)
    else:
        control_array = jnp.asarray(control)
    return AmbipolarProfileFamilyResult(
        control=control_array,
        er_profile=jnp.stack([result.er_profile for result in family_results]),
        ambipolar_residual=jnp.stack([result.ambipolar_residual for result in family_results]),
        bootstrap_current_proxy=jnp.stack(
            [result.bootstrap_current_response for result in family_results]
        ),
        loss_history=jnp.stack([result.loss_history for result in family_results]),
    )


def bootstrap_current_objective(
    rho: Array,
    current_response: Array,
    *,
    weight: Array | None = None,
) -> Array:
    """Return a weighted quadratic objective for a reduced current response."""

    rho_arr = jnp.asarray(rho)
    profile = jnp.asarray(current_response)
    if profile.shape != rho_arr.shape:
        raise ValueError("current_response must match rho shape")
    if weight is None:
        weight_arr = jnp.ones_like(rho_arr)
    else:
        weight_arr = _broadcast_profile_field(weight, rho_arr)
    return jnp.trapezoid(weight_arr * profile**2, rho_arr)


def current_response_objective(
    rho: Array,
    current_response: Array,
    *,
    weight: Array | None = None,
) -> Array:
    """Return a weighted quadratic objective for a reduced current response."""

    return bootstrap_current_objective(rho, current_response, weight=weight)
