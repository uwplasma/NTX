"""Evaluation and ambipolar-profile solvers built on NTX scan data."""

from __future__ import annotations

import interpax
import jax
import jax.numpy as jnp
from jax import Array

from ._profiles_types import (
    AmbipolarProfileFamilyResult,
    AmbipolarProfileResult,
    MonoenergeticSpeciesProfile,
    PrimitiveSpeciesProfile,
)
from .neopax import NeopaxScan


def evaluate_scan_channel(
    scan: NeopaxScan,
    channel: str,
    rho: Array,
    nu_v: Array,
    er_profile: Array,
) -> Array:
    """Interpolate one NTX scan channel over `(rho, nu_v, E_r)`."""

    rho_arr = jnp.asarray(rho)
    nu_arr = _broadcast_profile_field(nu_v, rho_arr)
    er_arr = _broadcast_profile_field(er_profile, rho_arr)
    if rho_arr.shape != jnp.asarray(scan.rho).shape:
        raise ValueError("rho must match scan.rho shape")
    data = _channel_data(scan, channel)
    log_nu_axis = jnp.log10(jnp.asarray(scan.nu_v))

    def per_radius(index, nu_value, er_value):
        er_axis = jnp.asarray(scan.Er[index])
        values = data[index]
        interpolator = interpax.Interpolator2D(
            log_nu_axis,
            er_axis,
            values,
            extrap=True,
        )
        if channel == "D11":
            return 10.0 ** interpolator(jnp.log10(jnp.maximum(nu_value, 1e-30)), er_value)
        return interpolator(jnp.log10(jnp.maximum(nu_value, 1e-30)), er_value)

    return jax.vmap(per_radius)(jnp.arange(rho_arr.size), nu_arr, er_arr)


def evaluate_species_particle_flux(
    scan: NeopaxScan,
    species: MonoenergeticSpeciesProfile,
    *,
    rho: Array | None = None,
    er_profile: Array,
) -> Array:
    """Return the monoenergetic particle-flux proxy for one species."""

    rho_eval = jnp.asarray(scan.rho) if rho is None else jnp.asarray(rho)
    d11 = evaluate_scan_channel(scan, "D11", rho_eval, species.nu_v, er_profile)
    d13 = evaluate_scan_channel(scan, "D13", rho_eval, species.nu_v, er_profile)
    a1 = _broadcast_profile_field(species.A1, rho_eval)
    a3 = _broadcast_profile_field(species.A3, rho_eval)
    particle_weight = _broadcast_profile_field(species.particle_weight, rho_eval)
    return -particle_weight * (d11 * a1 + d13 * a3)


def evaluate_species_current_response(
    scan: NeopaxScan,
    species: MonoenergeticSpeciesProfile,
    *,
    rho: Array | None = None,
    er_profile: Array,
) -> Array:
    """Return the monoenergetic bootstrap-current proxy for one species."""

    rho_eval = jnp.asarray(scan.rho) if rho is None else jnp.asarray(rho)
    d31 = evaluate_scan_channel(scan, "D31", rho_eval, species.nu_v, er_profile)
    d33 = evaluate_scan_channel(scan, "D33", rho_eval, species.nu_v, er_profile)
    a1 = _broadcast_profile_field(species.A1, rho_eval)
    a3 = _broadcast_profile_field(species.A3, rho_eval)
    current_weight = _broadcast_profile_field(species.current_weight, rho_eval)
    return -current_weight * (d31 * a1 + d33 * a3)


def ambipolar_residual_profile(
    scan: NeopaxScan,
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    *,
    er_profile: Array,
) -> Array:
    """Return the charge-weighted monoenergetic ambipolar residual profile."""

    rho = jnp.asarray(scan.rho)
    er_arr = _broadcast_profile_field(er_profile, rho)
    residual = jnp.zeros_like(rho)
    for species in species_profiles:
        charge = _broadcast_profile_field(species.charge, rho)
        residual = residual + charge * evaluate_species_particle_flux(
            scan,
            species,
            rho=rho,
            er_profile=er_arr,
        )
    return residual


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
            [result.bootstrap_current_proxy for result in family_results]
        ),
        loss_history=jnp.stack([result.loss_history for result in family_results]),
    )


def bootstrap_current_objective(
    rho: Array,
    bootstrap_current_proxy: Array,
    *,
    weight: Array | None = None,
) -> Array:
    """Return a weighted quadratic radial objective for a bootstrap-current profile."""

    rho_arr = jnp.asarray(rho)
    profile = jnp.asarray(bootstrap_current_proxy)
    if profile.shape != rho_arr.shape:
        raise ValueError("bootstrap_current_proxy must match rho shape")
    if weight is None:
        weight_arr = jnp.ones_like(rho_arr)
    else:
        weight_arr = _broadcast_profile_field(weight, rho_arr)
    return jnp.trapezoid(weight_arr * profile**2, rho_arr)


def build_species_profile_from_primitives(
    rho: Array,
    primitive: PrimitiveSpeciesProfile,
    *,
    er_profile: Array,
) -> MonoenergeticSpeciesProfile:
    """Construct `A1(r)` and `A3(r)` from primitive density/temperature profiles."""

    rho_arr = jnp.asarray(rho)
    density = _broadcast_profile_field(primitive.density, rho_arr)
    temperature = _broadcast_profile_field(primitive.temperature, rho_arr)
    charge = _broadcast_profile_field(primitive.charge, rho_arr)
    er_arr = _broadcast_profile_field(er_profile, rho_arr)
    prefactor = _broadcast_profile_field(primitive.electrostatic_prefactor, rho_arr)

    def grad(values):
        safe_values = _smooth_radial_profile(values, jnp.asarray(0.35, dtype=rho_arr.dtype))
        return jnp.gradient(safe_values, rho_arr)

    log_density_grad = grad(
        jnp.log(jnp.maximum(density, jnp.asarray(1.0e-12, dtype=rho_arr.dtype)))
    )
    log_temperature_grad = grad(
        jnp.log(jnp.maximum(temperature, jnp.asarray(1.0e-12, dtype=rho_arr.dtype)))
    )
    a3 = log_temperature_grad
    a1 = log_density_grad - 1.5 * log_temperature_grad + prefactor * charge * er_arr
    return MonoenergeticSpeciesProfile(
        charge=primitive.charge,
        nu_v=_broadcast_profile_field(primitive.nu_v, rho_arr),
        A1=a1,
        A3=a3,
        particle_weight=primitive.particle_weight,
        current_weight=primitive.current_weight,
        name=primitive.name,
    )


def build_species_profiles_from_primitives(
    rho: Array,
    primitives: tuple[PrimitiveSpeciesProfile, ...],
    *,
    er_profile: Array,
) -> tuple[MonoenergeticSpeciesProfile, ...]:
    """Vectorized helper for primitive-to-monoenergetic profile construction."""

    return tuple(
        build_species_profile_from_primitives(rho, primitive, er_profile=er_profile)
        for primitive in primitives
    )


def _broadcast_profile_field(values, rho: Array) -> Array:
    array = jnp.asarray(values)
    if array.ndim == 0:
        return jnp.full_like(rho, array)
    if array.shape == rho.shape:
        return array
    raise ValueError("profile field must be scalar or match rho shape")


def _smooth_radial_profile(values: Array, strength: Array) -> Array:
    if jnp.asarray(values).ndim != 1:
        raise ValueError("values must be one-dimensional for radial smoothing")
    if jnp.asarray(values).shape[0] < 3:
        return values
    strength_value = jnp.clip(jnp.asarray(strength), 0.0, 1.0)
    left = jnp.concatenate([values[:1], values[:-1]])
    right = jnp.concatenate([values[1:], values[-1:]])
    smoothed = 0.25 * left + 0.5 * values + 0.25 * right
    return (1.0 - strength_value) * values + strength_value * smoothed


def _channel_data(scan: NeopaxScan, channel: str) -> Array:
    if channel == "D11":
        return jnp.log10(jnp.maximum(jnp.asarray(scan.D11), 1.0e-30))
    if channel == "D13":
        return jnp.asarray(scan.D13)
    if channel == "D33":
        return jnp.asarray(scan.D33)
    if channel == "D31":
        if scan.D31 is None:
            return -jnp.asarray(scan.D13)
        return jnp.asarray(scan.D31)
    raise ValueError(f"unsupported channel '{channel}'")


def _single_radius_profile(
    rho: Array,
    rho_value: Array,
    er_profile: Array,
    er_trial: Array,
) -> Array:
    index = jnp.argmin(jnp.abs(rho - rho_value))
    return er_profile.at[index].set(er_trial)

