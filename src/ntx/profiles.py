"""Profile-grade imported transport workflows built on NTX scan data."""

from __future__ import annotations

from dataclasses import dataclass

import interpax
import jax
import jax.numpy as jnp
from jax import Array, tree_util

from .neopax import NeopaxScan


@dataclass(frozen=True)
class MonoenergeticSpeciesProfile:
    """One-species radial profile inputs for ambipolar and current proxies."""

    charge: float | Array
    nu_v: Array
    A1: Array
    A3: Array
    particle_weight: float | Array = 1.0
    current_weight: float | Array = 1.0
    name: str | None = None


tree_util.register_dataclass(
    MonoenergeticSpeciesProfile,
    data_fields=("charge", "nu_v", "A1", "A3", "particle_weight", "current_weight"),
    meta_fields=("name",),
)


@dataclass(frozen=True)
class AmbipolarProfileResult:
    """Solved electric-field profile and derived monoenergetic proxy quantities."""

    rho: Array
    er_profile: Array
    ambipolar_residual: Array
    bootstrap_current_proxy: Array
    species_particle_flux: Array
    species_current_response: Array
    loss_history: Array


tree_util.register_dataclass(
    AmbipolarProfileResult,
    data_fields=(
        "rho",
        "er_profile",
        "ambipolar_residual",
        "bootstrap_current_proxy",
        "species_particle_flux",
        "species_current_response",
        "loss_history",
    ),
    meta_fields=(),
)


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
) -> AmbipolarProfileResult:
    """Solve a per-radius ambipolar `E_r(r)` profile from an NTX scan."""

    rho = jnp.asarray(scan.rho)
    dtype = rho.dtype
    er_min = jnp.min(jnp.asarray(scan.Er), axis=1)
    er_max = jnp.max(jnp.asarray(scan.Er), axis=1)
    damping_value = jnp.asarray(damping, dtype=dtype)
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

    def radius_update(carry, _):
        er_profile = carry

        def update_one(er_value, rho_value, er_lo, er_hi):
            def local_residual(er_trial):
                return ambipolar_residual_profile(
                    scan,
                    species_profiles,
                    er_profile=_single_radius_profile(rho, rho_value, er_profile, er_trial),
                )[jnp.argmin(jnp.abs(rho - rho_value))]

            residual = local_residual(er_value)
            derivative = jax.grad(local_residual)(er_value)
            safe_derivative = jnp.where(
                jnp.abs(derivative) > jnp.asarray(1.0e-12, dtype=dtype),
                derivative,
                jnp.where(
                    derivative >= 0.0,
                    jnp.asarray(1.0e-12, dtype=dtype),
                    jnp.asarray(-1.0e-12, dtype=dtype),
                ),
            )
            updated = er_value - damping_value * residual / safe_derivative
            return jnp.clip(updated, er_lo, er_hi)

        next_profile = jax.vmap(update_one)(er_profile, rho, er_min, er_max)
        residual = residual_at_profile(next_profile)
        loss = jnp.mean(residual**2)
        return next_profile, (next_profile, loss)

    solved_profile, history = jax.lax.scan(radius_update, er0, xs=None, length=steps)
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


def _broadcast_profile_field(values, rho: Array) -> Array:
    array = jnp.asarray(values)
    if array.ndim == 0:
        return jnp.full_like(rho, array)
    if array.shape == rho.shape:
        return array
    raise ValueError("profile field must be scalar or match rho shape")


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
