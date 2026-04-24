"""Profile transport mismatch, normalization, and scaling terms."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
from jax import Array

from ._profiles_radial import _broadcast_profile_field
from ._profiles_types import (
    AmbipolarProfileResult,
    PrimitiveSpeciesProfile,
    ProfileTransportClosureSpec,
)


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
