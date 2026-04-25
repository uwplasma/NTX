"""Primitive density/temperature profiles mapped to NTX force proxies."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ._profiles_radial import _broadcast_profile_field, _smooth_radial_profile
from ._profiles_species_types import (
    MonoenergeticSpeciesProfile,
    PrimitiveSpeciesProfile,
)


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
