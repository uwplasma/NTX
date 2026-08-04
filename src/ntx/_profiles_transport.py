"""Transport closures and their flux terms.

The closure relations that turn profiles and geometry into fluxes, with the
term-by-term decomposition used by the transport diagnostics. Types live in
_profiles.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
from jax import Array

from ._profiles import (
    AmbipolarProfileResult,
    MonoenergeticSpeciesProfile,
    PrimitiveProfileTransportIterationResult,
    PrimitiveSpeciesProfile,
    ProfileTransportClosureSpec,
    ProfileTransportIterationResult,
    _broadcast_profile_field,
    _smooth_radial_profile,
    build_species_profiles_from_primitives,
    solve_ambipolar_er_profile,
)
from .neopax import NeopaxScan

__all__ = [
    "_broadcast_species_transport_field",
    "_normalized_primitive_updates",
    "_normalized_transport_updates",
    "_primitive_mismatch",
    "_scaled_transport_closure",
    "_transport_mismatch",
    "advance_primitive_profile_transport",
    "advance_profile_transport",
    "primitive_profile_transport_loss",
    "profile_transport_loss",
]


# --- _profiles_transport_terms: Profile transport mismatch, normalization, and scaling terms. ---


def _broadcast_species_transport_field(
    values,
    species_count: int,
    rho: Array,
) -> Array:
    array = jnp.asarray(values)
    radial_size = int(jnp.asarray(rho).size)
    if array.ndim == 0:
        return jnp.full((species_count, radial_size), array)
    if array.ndim == 1 and array.shape == (species_count,):
        return jnp.repeat(array[:, None], radial_size, axis=1)
    if array.ndim == 1 and array.shape == (radial_size,):
        return jnp.repeat(array[None, :], species_count, axis=0)
    if array.shape == (species_count, radial_size):
        return array
    raise ValueError(
        "transport field must be scalar, per-species, per-radius, or species-by-radius"
    )


def _transport_mismatch(
    profile: AmbipolarProfileResult,
    closure_spec: ProfileTransportClosureSpec,
) -> tuple[Array, Array]:
    species_flux = jnp.asarray(profile.species_particle_flux)
    species_current = jnp.asarray(profile.species_current_response)
    rho = jnp.asarray(profile.rho)
    particle_target = _broadcast_species_transport_field(
        closure_spec.particle_target,
        species_flux.shape[0],
        rho,
    )
    current_target = _broadcast_species_transport_field(
        closure_spec.current_target,
        species_current.shape[0],
        rho,
    )
    particle_source = _broadcast_species_transport_field(
        closure_spec.particle_source,
        species_flux.shape[0],
        rho,
    )
    current_source = _broadcast_species_transport_field(
        closure_spec.current_source,
        species_current.shape[0],
        rho,
    )
    particle_mismatch = species_flux - particle_target - particle_source
    current_mismatch = species_current - current_target - current_source
    return particle_mismatch, current_mismatch


def _normalized_transport_updates(
    profile: AmbipolarProfileResult,
    closure_spec: ProfileTransportClosureSpec,
    *,
    normalization_floor: Array,
    max_update: Array,
) -> tuple[Array, Array]:
    particle_mismatch, current_mismatch = _transport_mismatch(profile, closure_spec)
    particle_scale = jnp.maximum(
        jnp.sqrt(jnp.mean(particle_mismatch**2, axis=1, keepdims=True)),
        normalization_floor,
    )
    current_scale = jnp.maximum(
        jnp.sqrt(jnp.mean(current_mismatch**2, axis=1, keepdims=True)),
        normalization_floor,
    )
    normalized_particle = jnp.clip(
        particle_mismatch / particle_scale,
        -max_update,
        max_update,
    )
    normalized_current = jnp.clip(
        current_mismatch / current_scale,
        -max_update,
        max_update,
    )
    return normalized_particle, normalized_current


def _scaled_transport_closure(
    closure_spec: ProfileTransportClosureSpec,
    factor: Array,
) -> ProfileTransportClosureSpec:
    return replace(
        closure_spec,
        particle_relaxation=jnp.asarray(closure_spec.particle_relaxation) * factor,
        current_relaxation=jnp.asarray(closure_spec.current_relaxation) * factor,
        density_relaxation=jnp.asarray(closure_spec.density_relaxation) * factor,
        temperature_relaxation=jnp.asarray(closure_spec.temperature_relaxation) * factor,
    )


def _primitive_mismatch(
    primitive_profiles: tuple[PrimitiveSpeciesProfile, ...],
    closure_spec: ProfileTransportClosureSpec,
    rho: Array,
) -> tuple[Array, Array]:
    species_count = len(primitive_profiles)
    density_target = _broadcast_species_transport_field(
        closure_spec.density_target,
        species_count,
        rho,
    )
    temperature_target = _broadcast_species_transport_field(
        closure_spec.temperature_target,
        species_count,
        rho,
    )
    density_source = _broadcast_species_transport_field(
        closure_spec.density_source,
        species_count,
        rho,
    )
    temperature_source = _broadcast_species_transport_field(
        closure_spec.temperature_source,
        species_count,
        rho,
    )
    density = jnp.stack(
        [_broadcast_profile_field(primitive.density, rho) for primitive in primitive_profiles]
    )
    temperature = jnp.stack(
        [_broadcast_profile_field(primitive.temperature, rho) for primitive in primitive_profiles]
    )
    density_mismatch = density - density_target - density_source
    temperature_mismatch = temperature - temperature_target - temperature_source
    return density_mismatch, temperature_mismatch


def _normalized_primitive_updates(
    primitive_profiles: tuple[PrimitiveSpeciesProfile, ...],
    closure_spec: ProfileTransportClosureSpec,
    *,
    rho: Array,
    normalization_floor: Array,
    max_update: Array,
) -> tuple[Array, Array]:
    density_mismatch, temperature_mismatch = _primitive_mismatch(
        primitive_profiles,
        closure_spec,
        jnp.asarray(rho),
    )
    density_scale = jnp.maximum(
        jnp.sqrt(jnp.mean(density_mismatch**2, axis=1, keepdims=True)),
        normalization_floor,
    )
    temperature_scale = jnp.maximum(
        jnp.sqrt(jnp.mean(temperature_mismatch**2, axis=1, keepdims=True)),
        normalization_floor,
    )
    normalized_density = jnp.clip(density_mismatch / density_scale, -max_update, max_update)
    normalized_temperature = jnp.clip(
        temperature_mismatch / temperature_scale,
        -max_update,
        max_update,
    )
    return normalized_density, normalized_temperature


# --- _profiles_transport_closure: Profile transport closure losses and explicit update algebra. ---


def profile_transport_loss(
    profile: AmbipolarProfileResult,
    closure_spec: ProfileTransportClosureSpec,
) -> Array:
    """Quadratic transport mismatch loss for a solved ambipolar profile."""

    particle_mismatch, current_mismatch = _transport_mismatch(profile, closure_spec)
    return jnp.mean(particle_mismatch**2 + current_mismatch**2)


def advance_profile_transport(
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    profile: AmbipolarProfileResult,
    closure_spec: ProfileTransportClosureSpec,
) -> tuple[MonoenergeticSpeciesProfile, ...]:
    """Apply one explicit transport-relaxation update to `A1` and `A3`."""

    species_count = len(species_profiles)
    rho = jnp.asarray(profile.rho)
    species_flux = jnp.asarray(profile.species_particle_flux)
    species_current = jnp.asarray(profile.species_current_response)
    if species_flux.shape[0] != species_count or species_current.shape[0] != species_count:
        raise ValueError("profile species arrays must match the number of species")
    particle_relaxation = _broadcast_species_transport_field(
        closure_spec.particle_relaxation,
        species_count,
        rho,
    )
    current_relaxation = _broadcast_species_transport_field(
        closure_spec.current_relaxation,
        species_count,
        rho,
    )
    normalization_floor = _broadcast_species_transport_field(
        closure_spec.normalization_floor,
        species_count,
        rho,
    )
    max_update = _broadcast_species_transport_field(
        closure_spec.max_normalized_update,
        species_count,
        rho,
    )
    normalized_particle, normalized_current = _normalized_transport_updates(
        profile,
        closure_spec,
        normalization_floor=normalization_floor,
        max_update=max_update,
    )
    smoothing_strength = _broadcast_species_transport_field(
        closure_spec.radial_smoothing_strength,
        species_count,
        rho,
    )
    return tuple(
        replace(
            species,
            A1=_smooth_radial_profile(
                jnp.asarray(species.A1) - particle_relaxation[index] * normalized_particle[index],
                jnp.mean(smoothing_strength[index]),
            ),
            A3=_smooth_radial_profile(
                jnp.asarray(species.A3) - current_relaxation[index] * normalized_current[index],
                jnp.mean(smoothing_strength[index]),
            ),
        )
        for index, species in enumerate(species_profiles)
    )


def advance_primitive_profile_transport(
    primitive_profiles: tuple[PrimitiveSpeciesProfile, ...],
    profile: AmbipolarProfileResult,
    closure_spec: ProfileTransportClosureSpec,
) -> tuple[PrimitiveSpeciesProfile, ...]:
    """Apply one explicit transport-relaxation update to primitive profiles."""

    species_count = len(primitive_profiles)
    rho = jnp.asarray(profile.rho)
    normalization_floor = _broadcast_species_transport_field(
        closure_spec.normalization_floor,
        species_count,
        rho,
    )
    max_update = _broadcast_species_transport_field(
        closure_spec.max_normalized_update,
        species_count,
        rho,
    )
    particle_relaxation = _broadcast_species_transport_field(
        closure_spec.particle_relaxation,
        species_count,
        rho,
    )
    current_relaxation = _broadcast_species_transport_field(
        closure_spec.current_relaxation,
        species_count,
        rho,
    )
    density_relaxation = _broadcast_species_transport_field(
        closure_spec.density_relaxation,
        species_count,
        rho,
    )
    temperature_relaxation = _broadcast_species_transport_field(
        closure_spec.temperature_relaxation,
        species_count,
        rho,
    )
    primitive_normalization_floor = _broadcast_species_transport_field(
        closure_spec.primitive_normalization_floor,
        species_count,
        rho,
    )
    max_primitive_update = _broadcast_species_transport_field(
        closure_spec.max_primitive_normalized_update,
        species_count,
        rho,
    )
    radial_smoothing = _broadcast_species_transport_field(
        closure_spec.radial_smoothing_strength,
        species_count,
        rho,
    )
    normalized_particle, normalized_current = _normalized_transport_updates(
        profile,
        closure_spec,
        normalization_floor=normalization_floor,
        max_update=max_update,
    )
    normalized_density, normalized_temperature = _normalized_primitive_updates(
        primitive_profiles,
        closure_spec,
        rho=rho,
        normalization_floor=primitive_normalization_floor,
        max_update=max_primitive_update,
    )
    return tuple(
        replace(
            primitive,
            density=jnp.maximum(
                _smooth_radial_profile(
                    jnp.asarray(primitive.density)
                    * jnp.exp(
                        -particle_relaxation[index] * normalized_particle[index]
                        - density_relaxation[index] * normalized_density[index]
                    ),
                    jnp.mean(radial_smoothing[index]),
                ),
                jnp.asarray(1.0e-8, dtype=rho.dtype),
            ),
            temperature=jnp.maximum(
                _smooth_radial_profile(
                    jnp.asarray(primitive.temperature)
                    * jnp.exp(
                        -current_relaxation[index] * normalized_current[index]
                        - temperature_relaxation[index] * normalized_temperature[index]
                    ),
                    jnp.mean(radial_smoothing[index]),
                ),
                jnp.asarray(1.0e-8, dtype=rho.dtype),
            ),
        )
        for index, primitive in enumerate(primitive_profiles)
    )


def primitive_profile_transport_loss(
    profile: AmbipolarProfileResult,
    primitive_profiles: tuple[PrimitiveSpeciesProfile, ...],
    closure_spec: ProfileTransportClosureSpec,
) -> Array:
    """Combined profile-transport loss including primitive source/target closure."""

    base_loss = profile_transport_loss(profile, closure_spec)
    species_count = len(primitive_profiles)
    rho = jnp.asarray(profile.rho)
    normalization_floor = _broadcast_species_transport_field(
        closure_spec.primitive_normalization_floor,
        species_count,
        rho,
    )
    max_update = _broadcast_species_transport_field(
        closure_spec.max_primitive_normalized_update,
        species_count,
        rho,
    )
    normalized_density, normalized_temperature = _normalized_primitive_updates(
        primitive_profiles,
        closure_spec,
        rho=rho,
        normalization_floor=normalization_floor,
        max_update=max_update,
    )
    smoothing_strength = _broadcast_species_transport_field(
        closure_spec.radial_smoothing_strength,
        species_count,
        rho,
    )
    smoothness = jnp.asarray(0.0, dtype=rho.dtype)
    for index, primitive in enumerate(primitive_profiles):
        density = jnp.asarray(primitive.density)
        temperature = jnp.asarray(primitive.temperature)
        density_smooth = density - _smooth_radial_profile(
            density,
            jnp.mean(smoothing_strength[index]),
        )
        temperature_smooth = temperature - _smooth_radial_profile(
            temperature,
            jnp.mean(smoothing_strength[index]),
        )
        smoothness = smoothness + jnp.mean(density_smooth**2 + temperature_smooth**2)
    primitive_loss = jnp.mean(normalized_density**2 + normalized_temperature**2)
    return base_loss + primitive_loss + 0.25 * smoothness


# --- _profiles_transport: Transport-loop helpers for profile-grade NTX workflows. ---


def solve_profile_transport_loop(
    scan: NeopaxScan,
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    closure_spec: ProfileTransportClosureSpec,
    *,
    iterations: int = 8,
    er_initial: Array | None = None,
    solve_steps: int = 16,
    damping: float = 0.8,
    smoothing_strength: float = 0.0,
) -> ProfileTransportIterationResult:
    """Iterate a simple self-consistent profile transport closure."""

    species_state = species_profiles
    er_seed = er_initial
    profile_history: list[AmbipolarProfileResult] = []
    loss_history: list[Array] = []
    a1_history: list[Array] = []
    a3_history: list[Array] = []
    best_profile: AmbipolarProfileResult | None = None
    best_loss: Array | None = None

    for _ in range(iterations):
        profile = solve_ambipolar_er_profile(
            scan,
            species_state,
            er_initial=er_seed,
            steps=solve_steps,
            damping=damping,
            smoothing_strength=smoothing_strength,
        )
        profile_history.append(profile)
        current_loss = profile_transport_loss(profile, closure_spec)
        loss_history.append(current_loss)
        a1_history.append(jnp.stack([jnp.asarray(species.A1) for species in species_state]))
        a3_history.append(jnp.stack([jnp.asarray(species.A3) for species in species_state]))
        if best_profile is None or best_loss is None or bool(current_loss < best_loss):
            best_profile = profile
            best_loss = current_loss

        next_species_state = species_state
        next_er_seed = profile.er_profile
        accepted = False
        for step_index in range(5):
            factor = jnp.asarray(0.5**step_index, dtype=jnp.asarray(scan.rho).dtype)
            scaled_closure = _scaled_transport_closure(closure_spec, factor)
            candidate_species = advance_profile_transport(species_state, profile, scaled_closure)
            candidate_profile = solve_ambipolar_er_profile(
                scan,
                candidate_species,
                er_initial=profile.er_profile,
                steps=solve_steps,
                damping=damping,
                smoothing_strength=smoothing_strength,
            )
            candidate_loss = profile_transport_loss(candidate_profile, closure_spec)
            if bool(candidate_loss <= current_loss + 1.0e-12):
                next_species_state = candidate_species
                next_er_seed = candidate_profile.er_profile
                accepted = True
                if best_loss is None or bool(candidate_loss < best_loss):
                    best_profile = candidate_profile
                    best_loss = candidate_loss
                break
        if not accepted:
            next_er_seed = profile.er_profile
        species_state = next_species_state
        er_seed = next_er_seed

    loss_array = jnp.stack(loss_history)
    best_index = int(jnp.argmin(loss_array))
    return ProfileTransportIterationResult(
        er_profile_history=jnp.stack([profile.er_profile for profile in profile_history]),
        ambipolar_residual_history=jnp.stack(
            [profile.ambipolar_residual for profile in profile_history]
        ),
        bootstrap_current_response_history=jnp.stack(
            [profile.bootstrap_current_response for profile in profile_history]
        ),
        transport_loss_history=loss_array,
        species_a1_history=jnp.stack(a1_history),
        species_a3_history=jnp.stack(a3_history),
        best_profile=best_profile if best_profile is not None else profile_history[best_index],
    )


def solve_primitive_profile_transport_loop(
    scan: NeopaxScan,
    primitive_profiles: tuple[PrimitiveSpeciesProfile, ...],
    closure_spec: ProfileTransportClosureSpec,
    *,
    iterations: int = 8,
    er_initial: Array | None = None,
    solve_steps: int = 16,
    damping: float = 0.8,
    smoothing_strength: float = 0.0,
) -> PrimitiveProfileTransportIterationResult:
    """Iterate a primitive density/temperature transport closure."""

    primitive_state = primitive_profiles
    rho = jnp.asarray(scan.rho)
    er_seed = (
        0.5 * (jnp.min(jnp.asarray(scan.Er), axis=1) + jnp.max(jnp.asarray(scan.Er), axis=1))
        if er_initial is None
        else _broadcast_profile_field(er_initial, rho)
    )
    profile_history: list[AmbipolarProfileResult] = []
    density_history: list[Array] = []
    temperature_history: list[Array] = []
    loss_history: list[Array] = []
    best_profile: AmbipolarProfileResult | None = None
    best_loss: Array | None = None

    for _ in range(iterations):
        species_profiles = build_species_profiles_from_primitives(
            rho,
            primitive_state,
            er_profile=er_seed,
        )
        profile = solve_ambipolar_er_profile(
            scan,
            species_profiles,
            er_initial=er_seed,
            steps=solve_steps,
            damping=damping,
            smoothing_strength=smoothing_strength,
        )
        profile_history.append(profile)
        density_history.append(
            jnp.stack(
                [_broadcast_profile_field(primitive.density, rho) for primitive in primitive_state]
            )
        )
        temperature_history.append(
            jnp.stack(
                [
                    _broadcast_profile_field(primitive.temperature, rho)
                    for primitive in primitive_state
                ]
            )
        )
        current_loss = primitive_profile_transport_loss(profile, primitive_state, closure_spec)
        loss_history.append(current_loss)
        if best_profile is None or best_loss is None or bool(current_loss < best_loss):
            best_profile = profile
            best_loss = current_loss

        next_primitive_state = primitive_state
        next_er_seed = profile.er_profile
        accepted = False
        for step_index in range(5):
            factor = jnp.asarray(0.5**step_index, dtype=rho.dtype)
            scaled_closure = _scaled_transport_closure(closure_spec, factor)
            candidate_primitive = advance_primitive_profile_transport(
                primitive_state,
                profile,
                scaled_closure,
            )
            candidate_species = build_species_profiles_from_primitives(
                rho,
                candidate_primitive,
                er_profile=profile.er_profile,
            )
            candidate_profile = solve_ambipolar_er_profile(
                scan,
                candidate_species,
                er_initial=profile.er_profile,
                steps=solve_steps,
                damping=damping,
                smoothing_strength=smoothing_strength,
            )
            candidate_loss = primitive_profile_transport_loss(
                candidate_profile,
                candidate_primitive,
                closure_spec,
            )
            if bool(candidate_loss <= current_loss + 1.0e-12):
                next_primitive_state = candidate_primitive
                next_er_seed = candidate_profile.er_profile
                accepted = True
                if best_loss is None or bool(candidate_loss < best_loss):
                    best_profile = candidate_profile
                    best_loss = candidate_loss
                break
        if not accepted:
            next_er_seed = profile.er_profile
        primitive_state = next_primitive_state
        er_seed = next_er_seed

    loss_array = jnp.stack(loss_history)
    best_index = int(jnp.argmin(loss_array))
    return PrimitiveProfileTransportIterationResult(
        er_profile_history=jnp.stack([profile.er_profile for profile in profile_history]),
        ambipolar_residual_history=jnp.stack(
            [profile.ambipolar_residual for profile in profile_history]
        ),
        bootstrap_current_response_history=jnp.stack(
            [profile.bootstrap_current_response for profile in profile_history]
        ),
        transport_loss_history=loss_array,
        species_density_history=jnp.stack(density_history),
        species_temperature_history=jnp.stack(temperature_history),
        best_profile=best_profile if best_profile is not None else profile_history[best_index],
    )
