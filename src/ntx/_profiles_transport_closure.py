"""Profile transport closure losses and explicit update algebra."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
from jax import Array

from ._profiles_radial import _broadcast_profile_field, _smooth_radial_profile
from ._profiles_types import (
    AmbipolarProfileResult,
    MonoenergeticSpeciesProfile,
    PrimitiveSpeciesProfile,
    ProfileTransportClosureSpec,
)


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
        [
            _broadcast_profile_field(primitive.temperature, rho)
            for primitive in primitive_profiles
        ]
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
