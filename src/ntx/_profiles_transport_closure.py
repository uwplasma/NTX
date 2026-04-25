"""Profile transport closure losses and explicit update algebra."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
from jax import Array

from ._profiles_ambipolar_types import (
    AmbipolarProfileResult,
)
from ._profiles_radial import _smooth_radial_profile
from ._profiles_species_types import (
    MonoenergeticSpeciesProfile,
    PrimitiveSpeciesProfile,
)
from ._profiles_transport_terms import (
    _broadcast_species_transport_field,
    _normalized_primitive_updates,
    _normalized_transport_updates,
    _primitive_mismatch,
    _scaled_transport_closure,
    _transport_mismatch,
)
from ._profiles_transport_types import (
    ProfileTransportClosureSpec,
)

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
