"""Small radial-profile array helpers shared by profile workflows."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


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


def _single_radius_profile(
    rho: Array,
    rho_value: Array,
    er_profile: Array,
    er_trial: Array,
) -> Array:
    index = jnp.argmin(jnp.abs(rho - rho_value))
    return er_profile.at[index].set(er_trial)
