"""Plasma profiles: species, channels, radial evaluation, and the shared specs.

The profile containers and the primitives that evaluate them on a radial grid,
together with the specification and result types that the transport and control
layers both build on. Those two layers import from here, never the other way.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array, tree_util

from ._interp import interp2d_at
from .neopax import NeopaxScan

__all__ = [
    "AmbipolarProfileFamilyResult",
    "AmbipolarProfileResult",
    "MonoenergeticSpeciesProfile",
    "PrimitiveProfileTransportIterationResult",
    "PrimitiveSpeciesProfile",
    "ProfileBasisControlSpec",
    "ProfileBasisOptimizationResult",
    "ProfileControlOptimizationResult",
    "ProfileControlSpec",
    "ProfileTransportClosureSpec",
    "ProfileTransportIterationResult",
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


# --- _profiles_ambipolar_types: Ambipolar-profile result dataclasses. ---


@dataclass(frozen=True)
class AmbipolarProfileResult:
    """Solved electric-field profile and reduced monoenergetic responses."""

    rho: Array
    er_profile: Array
    ambipolar_residual: Array
    bootstrap_current_response: Array
    species_particle_flux: Array
    species_current_response: Array
    loss_history: Array

    @property
    def bootstrap_current_proxy(self) -> Array:
        """Compatibility alias for the reduced bootstrap-current response.

        New code should use ``bootstrap_current_response``. The alias is kept
        so NTX 0.2.x readers do not break, but it is no longer the registered
        runtime field because the quantity is a reduced monoenergetic response,
        not a fitted bootstrap-current closure.
        """

        return self.bootstrap_current_response


tree_util.register_dataclass(
    AmbipolarProfileResult,
    data_fields=(
        "rho",
        "er_profile",
        "ambipolar_residual",
        "bootstrap_current_response",
        "species_particle_flux",
        "species_current_response",
        "loss_history",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class AmbipolarProfileFamilyResult:
    """Stacked ambipolar-profile solutions across a control-parameter family."""

    control: Array
    er_profile: Array
    ambipolar_residual: Array
    bootstrap_current_response: Array
    loss_history: Array

    @property
    def bootstrap_current_proxy(self) -> Array:
        """Compatibility alias for the reduced bootstrap-current response family."""

        return self.bootstrap_current_response


tree_util.register_dataclass(
    AmbipolarProfileFamilyResult,
    data_fields=(
        "control",
        "er_profile",
        "ambipolar_residual",
        "bootstrap_current_response",
        "loss_history",
    ),
    meta_fields=(),
)


# --- _profiles_radial: Small radial-profile array helpers shared by profile workflows. ---


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


# --- _profiles_species_types: Species-profile dataclasses for profile transport workflows. ---


@dataclass(frozen=True)
class MonoenergeticSpeciesProfile:
    """One-species radial inputs for reduced ambipolar/current responses."""

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
class PrimitiveSpeciesProfile:
    """Primitive radial inputs used to construct monoenergetic force profiles."""

    charge: float | Array
    nu_v: Array
    density: Array
    temperature: Array
    electrostatic_prefactor: float | Array = 1.0
    particle_weight: float | Array = 1.0
    current_weight: float | Array = 1.0
    name: str | None = None


tree_util.register_dataclass(
    PrimitiveSpeciesProfile,
    data_fields=(
        "charge",
        "nu_v",
        "density",
        "temperature",
        "electrostatic_prefactor",
        "particle_weight",
        "current_weight",
    ),
    meta_fields=("name",),
)


# --- _profiles_channels: Scan-channel interpolation and species response helpers. ---


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

    # D11 and D33 are positive with a knee where the collisionality regime
    # changes, so the monotone rule -- which cannot overshoot a knee -- is the
    # accurate choice. D13 and D31 change sign and have a smooth extremum in
    # E_r, which that same limiter would flatten; they take the unlimited
    # parabolic slope instead. See `_interp` for the measurements.
    method = "pchip" if channel in ("D11", "D33") else "parabolic"

    def per_radius(index, nu_value, er_value):
        value = interp2d_at(
            log_nu_axis,
            jnp.asarray(scan.Er[index]),
            data[index],
            jnp.log10(jnp.maximum(nu_value, 1e-30)),
            er_value,
            method=method,
        )
        if channel == "D11":
            return 10.0**value
        return value

    return jax.vmap(per_radius)(jnp.arange(rho_arr.size), nu_arr, er_arr)


def evaluate_species_particle_flux(
    scan: NeopaxScan,
    species: MonoenergeticSpeciesProfile,
    *,
    rho: Array | None = None,
    er_profile: Array,
) -> Array:
    """Return the reduced monoenergetic particle-flux response for one species."""

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
    """Return the reduced monoenergetic parallel-current response for one species."""

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


# --- _profiles_control_types: Profile-control dataclasses. ---


@dataclass(frozen=True)
class ProfileControlSpec:
    """Linear control map applied to `A1` and `A3` for each species."""

    a1_response: Array
    a3_response: Array
    control_name: str = "control"


tree_util.register_dataclass(
    ProfileControlSpec,
    data_fields=("a1_response", "a3_response"),
    meta_fields=("control_name",),
)


@dataclass(frozen=True)
class ProfileControlOptimizationResult:
    """Optimization history for a scalar profile control."""

    control_history: Array
    objective_history: Array
    bootstrap_objective_history: Array
    residual_norm_history: Array
    best_control: Array
    best_profile: AmbipolarProfileResult


tree_util.register_dataclass(
    ProfileControlOptimizationResult,
    data_fields=(
        "control_history",
        "objective_history",
        "bootstrap_objective_history",
        "residual_norm_history",
        "best_control",
        "best_profile",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class ProfileBasisControlSpec:
    """Low-dimensional radial basis control applied to `A1` and `A3`."""

    basis: Array
    a1_response: Array
    a3_response: Array
    control_name: str = "basis control"


tree_util.register_dataclass(
    ProfileBasisControlSpec,
    data_fields=("basis", "a1_response", "a3_response"),
    meta_fields=("control_name",),
)


@dataclass(frozen=True)
class ProfileBasisOptimizationResult:
    """Optimization history for a vector profile-basis control."""

    control_history: Array
    objective_history: Array
    bootstrap_objective_history: Array
    residual_norm_history: Array
    best_control: Array
    best_profile: AmbipolarProfileResult


tree_util.register_dataclass(
    ProfileBasisOptimizationResult,
    data_fields=(
        "control_history",
        "objective_history",
        "bootstrap_objective_history",
        "residual_norm_history",
        "best_control",
        "best_profile",
    ),
    meta_fields=(),
)


# --- _profiles_primitives: Primitive density/temperature profiles mapped to NTX force channels. ---


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


# --- _profiles_eval: Ambipolar-profile solvers built on NTX scan data. ---


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
        return jnp.mean(first_diff**2) / (er_scale**2) + 0.5 * jnp.mean(second_diff**2) / (
            er_scale**2
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
        bootstrap_current_response=bootstrap_current,
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
        bootstrap_current_response=jnp.stack(
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


# --- _profiles_transport_types: Profile-transport closure and result dataclasses. ---


@dataclass(frozen=True)
class ProfileTransportClosureSpec:
    """Relaxation closure for iterating reduced responses toward transport targets."""

    particle_relaxation: Array
    current_relaxation: Array
    particle_target: float | Array = 0.0
    current_target: float | Array = 0.0
    particle_source: float | Array = 0.0
    current_source: float | Array = 0.0
    normalization_floor: float | Array = 1.0
    max_normalized_update: float | Array = 0.35
    density_relaxation: float | Array = 0.0
    temperature_relaxation: float | Array = 0.0
    density_target: float | Array = 0.0
    temperature_target: float | Array = 0.0
    density_source: float | Array = 0.0
    temperature_source: float | Array = 0.0
    primitive_normalization_floor: float | Array = 1.0
    max_primitive_normalized_update: float | Array = 0.20
    radial_smoothing_strength: float | Array = 0.0
    closure_name: str = "transport loop"


tree_util.register_dataclass(
    ProfileTransportClosureSpec,
    data_fields=(
        "particle_relaxation",
        "current_relaxation",
        "particle_target",
        "current_target",
        "particle_source",
        "current_source",
        "normalization_floor",
        "max_normalized_update",
        "density_relaxation",
        "temperature_relaxation",
        "density_target",
        "temperature_target",
        "density_source",
        "temperature_source",
        "primitive_normalization_floor",
        "max_primitive_normalized_update",
        "radial_smoothing_strength",
    ),
    meta_fields=("closure_name",),
)


@dataclass(frozen=True)
class ProfileTransportIterationResult:
    """History of a simple self-consistent profile transport relaxation loop."""

    er_profile_history: Array
    ambipolar_residual_history: Array
    bootstrap_current_response_history: Array
    transport_loss_history: Array
    species_a1_history: Array
    species_a3_history: Array
    best_profile: AmbipolarProfileResult

    @property
    def bootstrap_current_proxy_history(self) -> Array:
        """Compatibility alias for the reduced bootstrap-current response history."""

        return self.bootstrap_current_response_history


tree_util.register_dataclass(
    ProfileTransportIterationResult,
    data_fields=(
        "er_profile_history",
        "ambipolar_residual_history",
        "bootstrap_current_response_history",
        "transport_loss_history",
        "species_a1_history",
        "species_a3_history",
        "best_profile",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class PrimitiveProfileTransportIterationResult:
    """History of a primitive-profile transport relaxation workflow."""

    er_profile_history: Array
    ambipolar_residual_history: Array
    bootstrap_current_response_history: Array
    transport_loss_history: Array
    species_density_history: Array
    species_temperature_history: Array
    best_profile: AmbipolarProfileResult

    @property
    def bootstrap_current_proxy_history(self) -> Array:
        """Compatibility alias for the reduced bootstrap-current response history."""

        return self.bootstrap_current_response_history


tree_util.register_dataclass(
    PrimitiveProfileTransportIterationResult,
    data_fields=(
        "er_profile_history",
        "ambipolar_residual_history",
        "bootstrap_current_response_history",
        "transport_loss_history",
        "species_density_history",
        "species_temperature_history",
        "best_profile",
    ),
    meta_fields=(),
)
