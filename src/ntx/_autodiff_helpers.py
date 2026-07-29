"""Private helpers shared by autodiff workflow examples."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import Array

from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .neopax import NeopaxMonoenergeticArrays
from .solver import solve_monoenergetic_scan
from ._interp import interp2d_at


def surface_with_amplitude(
    surface: BoozerSurface,
    coefficient_index: int,
    amplitude: float | Array,
) -> BoozerSurface:
    return replace(surface, b_cos=surface.b_cos.at[coefficient_index].set(amplitude))


def inverse_problem_response(
    surface: BoozerSurface,
    grid: GridSpec,
    nu_hat: Array,
    er_hat: float,
) -> Array:
    coeffs = solve_monoenergetic_scan(surface, grid, nu_hat, er_hat=jnp.full_like(nu_hat, er_hat))
    return coeffs["D11"]


def er_profile(rho: Array, params: Array) -> Array:
    parameters = jnp.asarray(params)
    powers = 2 * jnp.arange(parameters.size, dtype=jnp.asarray(rho).dtype) + 1
    return jnp.sum(parameters[:, None] * jnp.asarray(rho)[None, :] ** powers[:, None], axis=0)


def evaluate_d33_profile(
    arrays: NeopaxMonoenergeticArrays,
    rho: Array,
    nu_value: Array,
    er_profile_value: Array,
) -> Array:
    log_nu = jnp.log10(jnp.maximum(nu_value, 1e-12))

    def per_radius(index, er_value):
        radius_scale = jnp.maximum(arrays.a_b * rho[index], 1e-8)
        er_log = jnp.log10(jnp.maximum(1e-8, jnp.abs(er_value / radius_scale)))
        # saturating and positive: the monotone rule cannot overshoot its knee.
        value = interp2d_at(
            arrays.nu_log,
            arrays.Er_list[index],
            arrays.D33[index],
            log_nu,
            er_log,
            method="pchip",
        )
        return value

    return jax.vmap(per_radius)(jnp.arange(rho.size), er_profile_value)


def evaluate_d11_profile(
    arrays: NeopaxMonoenergeticArrays,
    rho: Array,
    nu_value: Array,
    er_profile_value: Array,
) -> Array:
    log_nu = jnp.log10(jnp.maximum(nu_value, 1e-12))

    def per_radius(index, er_value):
        radius_scale = jnp.maximum(arrays.a_b * rho[index], 1e-8)
        er_log = jnp.log10(jnp.maximum(1e-8, jnp.abs(er_value / radius_scale)))
        # log-log regime knees: the monotone rule cannot overshoot them.
        value = interp2d_at(
            arrays.nu_log,
            arrays.Er_list[index],
            arrays.D11_log[index],
            log_nu,
            er_log,
            method="pchip",
        )
        return 10.0 ** value

    return jax.vmap(per_radius)(jnp.arange(rho.size), er_profile_value)


def evaluate_d13_profile(
    arrays: NeopaxMonoenergeticArrays,
    rho: Array,
    nu_value: Array,
    er_profile_value: Array,
) -> Array:
    log_nu = jnp.log10(jnp.maximum(nu_value, 1e-12))

    def per_radius(index, er_value):
        radius_scale = jnp.maximum(arrays.a_b * rho[index], 1e-8)
        er_log = jnp.log10(jnp.maximum(1e-8, jnp.abs(er_value / radius_scale)))
        # sign-changing with a smooth extremum, which a monotone limiter would flatten.
        value = interp2d_at(
            arrays.nu_log,
            arrays.Er_list[index],
            arrays.D13[index],
            log_nu,
            er_log,
            method="parabolic",
        )
        return value

    return jax.vmap(per_radius)(jnp.arange(rho.size), er_profile_value)


def dominant_nonaxisymmetric_mode(surface: BoozerSurface | VmecSurface) -> tuple[int, int]:
    mask = jnp.logical_not(jnp.logical_and(surface.m == 0, surface.n == 0))
    masked_amplitude = jnp.where(mask, jnp.abs(surface.b_cos), -1.0)
    index = int(jnp.argmax(masked_amplitude))
    return int(surface.m[index]), int(surface.n[index])


def mode_value_for_surface(
    surface: BoozerSurface | VmecSurface,
    harmonic_m: int,
    harmonic_n: int,
) -> Array:
    matches = jnp.logical_and(surface.m == harmonic_m, surface.n == harmonic_n)
    index = jnp.argmax(matches)
    return surface.b_cos[index]


def scale_surface_mode(
    surface: BoozerSurface | VmecSurface,
    harmonic_m: int,
    harmonic_n: int,
    scale: Array,
) -> BoozerSurface | VmecSurface:
    matches = jnp.logical_and(surface.m == harmonic_m, surface.n == harmonic_n)
    index = jnp.argmax(matches)
    scaled = surface.b_cos.at[index].set(surface.b_cos[index] * scale)
    return replace(surface, b_cos=jnp.where(matches, scaled, surface.b_cos))
