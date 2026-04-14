"""Profile-grade imported transport workflows built on NTX scan data."""

from __future__ import annotations

from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class AmbipolarProfileFamilyResult:
    """Stacked ambipolar-profile solutions across a control-parameter family."""

    control: Array
    er_profile: Array
    ambipolar_residual: Array
    bootstrap_current_proxy: Array
    loss_history: Array


tree_util.register_dataclass(
    AmbipolarProfileFamilyResult,
    data_fields=(
        "control",
        "er_profile",
        "ambipolar_residual",
        "bootstrap_current_proxy",
        "loss_history",
    ),
    meta_fields=(),
)


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


@dataclass(frozen=True)
class ProfileTransportClosureSpec:
    """Relaxation closure for iterating profile proxies toward transport targets."""

    particle_relaxation: Array
    current_relaxation: Array
    particle_target: float | Array = 0.0
    current_target: float | Array = 0.0
    particle_source: float | Array = 0.0
    current_source: float | Array = 0.0
    normalization_floor: float | Array = 1.0
    max_normalized_update: float | Array = 0.35
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
    ),
    meta_fields=("closure_name",),
)


@dataclass(frozen=True)
class ProfileTransportIterationResult:
    """History of a simple self-consistent profile transport relaxation loop."""

    er_profile_history: Array
    ambipolar_residual_history: Array
    bootstrap_current_proxy_history: Array
    transport_loss_history: Array
    species_a1_history: Array
    species_a3_history: Array
    best_profile: AmbipolarProfileResult


tree_util.register_dataclass(
    ProfileTransportIterationResult,
    data_fields=(
        "er_profile_history",
        "ambipolar_residual_history",
        "bootstrap_current_proxy_history",
        "transport_loss_history",
        "species_a1_history",
        "species_a3_history",
        "best_profile",
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


def solve_ambipolar_profile_family(
    scan: NeopaxScan,
    species_profiles_family: tuple[tuple[MonoenergeticSpeciesProfile, ...], ...],
    *,
    control: Array | None = None,
    er_initial: Array | None = None,
    steps: int = 16,
    damping: float = 0.8,
) -> AmbipolarProfileFamilyResult:
    """Solve a family of ambipolar profiles across explicit profile controls."""

    family_results = [
        solve_ambipolar_er_profile(
            scan,
            species_profiles,
            er_initial=er_initial,
            steps=steps,
            damping=damping,
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


def apply_profile_control(
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    control: Array | float,
    control_spec: ProfileControlSpec,
) -> tuple[MonoenergeticSpeciesProfile, ...]:
    """Apply a scalar control to the `A1` and `A3` profiles for each species."""

    if len(species_profiles) != int(jnp.asarray(control_spec.a1_response).shape[0]):
        raise ValueError("control_spec must match the number of species")
    if len(species_profiles) != int(jnp.asarray(control_spec.a3_response).shape[0]):
        raise ValueError("control_spec must match the number of species")
    control_value = jnp.asarray(control)
    a1_response = jnp.asarray(control_spec.a1_response)
    a3_response = jnp.asarray(control_spec.a3_response)
    return tuple(
        replace(
            species,
            A1=jnp.asarray(species.A1) * (1.0 + control_value * a1_response[index]),
            A3=jnp.asarray(species.A3) * (1.0 + control_value * a3_response[index]),
        )
        for index, species in enumerate(species_profiles)
    )


def optimize_profile_control(
    scan: NeopaxScan,
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    control_spec: ProfileControlSpec,
    *,
    control_initial: float | Array = 0.0,
    learning_rate: float = 0.15,
    optimization_steps: int = 12,
    solve_steps: int = 16,
    damping: float = 0.8,
    weight: Array | None = None,
    residual_penalty: float = 1.0,
    control_bound: float | Array | None = 0.6,
    backtracking_steps: int = 5,
) -> ProfileControlOptimizationResult:
    """Optimize a scalar profile control against the bootstrap-current objective."""

    rho = jnp.asarray(scan.rho)
    dtype = rho.dtype
    lr = jnp.asarray(learning_rate, dtype=dtype)
    residual_scale = jnp.asarray(residual_penalty, dtype=dtype)
    control0 = jnp.asarray(control_initial, dtype=dtype)
    weight_arr = None if weight is None else _broadcast_profile_field(weight, rho)

    def objective_and_profile(control_value, er_seed):
        controlled = apply_profile_control(species_profiles, control_value, control_spec)
        profile = solve_ambipolar_er_profile(
            scan,
            controlled,
            er_initial=er_seed,
            steps=solve_steps,
            damping=damping,
        )
        bootstrap_obj = bootstrap_current_objective(
            rho,
            profile.bootstrap_current_proxy,
            weight=weight_arr,
        )
        residual_obj = residual_scale * jnp.mean(profile.ambipolar_residual**2)
        objective = bootstrap_obj + residual_obj
        residual_norm = jnp.linalg.norm(profile.ambipolar_residual)
        return objective, (profile, bootstrap_obj, residual_norm)

    def optimization_step(carry, _):
        control_value, er_seed = carry

        def scalar_objective(control_trial):
            return objective_and_profile(control_trial, er_seed)

        (objective, (profile, bootstrap_obj, residual_norm)), gradient = jax.value_and_grad(
            scalar_objective,
            has_aux=True,
        )(control_value)
        grad_scale = jnp.maximum(jnp.abs(gradient), jnp.asarray(1.0, dtype=dtype))

        def clip_control(value):
            if control_bound is None:
                return value
            bound = jnp.asarray(control_bound, dtype=dtype)
            return jnp.clip(value, -bound, bound)

        def backtrack_step(step_index, state):
            best_control, best_objective, accepted = state
            factor = 0.5**step_index
            candidate = clip_control(control_value - factor * lr * gradient / grad_scale)
            candidate_objective, _ = objective_and_profile(candidate, er_seed)
            take = (~accepted) & (candidate_objective <= objective)
            next_best_control = jnp.where(take, candidate, best_control)
            next_best_objective = jnp.where(take, candidate_objective, best_objective)
            next_accepted = accepted | take
            return next_best_control, next_best_objective, next_accepted

        initial_candidate = clip_control(control_value - lr * gradient / grad_scale)
        initial_objective, _ = objective_and_profile(initial_candidate, er_seed)
        next_control, _, accepted = jax.lax.fori_loop(
            1,
            backtracking_steps,
            backtrack_step,
            (
                initial_candidate,
                initial_objective,
                initial_objective <= objective,
            ),
        )
        next_control = jnp.where(accepted, next_control, control_value)
        return (next_control, profile.er_profile), (
            control_value,
            objective,
            bootstrap_obj,
            residual_norm,
            profile,
        )

    er_seed0 = 0.5 * (
        jnp.min(jnp.asarray(scan.Er), axis=1) + jnp.max(jnp.asarray(scan.Er), axis=1)
    )
    (_, _), history = jax.lax.scan(
        optimization_step,
        (control0, er_seed0),
        xs=None,
        length=optimization_steps,
    )
    (
        control_history,
        objective_history,
        bootstrap_objective_history,
        residual_norm_history,
        profile_history,
    ) = history
    best_index = jnp.argmin(objective_history)
    return ProfileControlOptimizationResult(
        control_history=control_history,
        objective_history=objective_history,
        bootstrap_objective_history=bootstrap_objective_history,
        residual_norm_history=residual_norm_history,
        best_control=control_history[best_index],
        best_profile=jax.tree.map(lambda x: x[best_index], profile_history),
    )


def apply_profile_basis_control(
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    control: Array,
    control_spec: ProfileBasisControlSpec,
) -> tuple[MonoenergeticSpeciesProfile, ...]:
    """Apply a low-dimensional radial basis control to `A1` and `A3`."""

    basis = jnp.asarray(control_spec.basis)
    a1_response = jnp.asarray(control_spec.a1_response)
    a3_response = jnp.asarray(control_spec.a3_response)
    control_value = jnp.asarray(control)
    if len(species_profiles) != int(a1_response.shape[0]):
        raise ValueError("control_spec must match the number of species")
    if len(species_profiles) != int(a3_response.shape[0]):
        raise ValueError("control_spec must match the number of species")
    if control_value.shape != (basis.shape[0],):
        raise ValueError("control must match the number of basis functions")
    if a1_response.shape[1] != basis.shape[0] or a3_response.shape[1] != basis.shape[0]:
        raise ValueError("response matrices must match the number of basis functions")
    return tuple(
        replace(
            species,
            A1=jnp.asarray(species.A1)
            * (1.0 + _basis_profile_modifier(control_value, a1_response[index], basis)),
            A3=jnp.asarray(species.A3)
            * (1.0 + _basis_profile_modifier(control_value, a3_response[index], basis)),
        )
        for index, species in enumerate(species_profiles)
    )


def optimize_profile_basis_control(
    scan: NeopaxScan,
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    control_spec: ProfileBasisControlSpec,
    *,
    control_initial: Array,
    learning_rate: float = 0.12,
    optimization_steps: int = 12,
    solve_steps: int = 16,
    damping: float = 0.8,
    weight: Array | None = None,
    residual_penalty: float = 1.0,
    control_penalty: float = 1.0e-2,
    control_bound: float | Array | None = 0.4,
    backtracking_steps: int = 5,
) -> ProfileBasisOptimizationResult:
    """Optimize a vector radial-basis control against the profile objective."""

    rho = jnp.asarray(scan.rho)
    dtype = rho.dtype
    lr = jnp.asarray(learning_rate, dtype=dtype)
    residual_scale = jnp.asarray(residual_penalty, dtype=dtype)
    control_scale = jnp.asarray(control_penalty, dtype=dtype)
    control0 = jnp.asarray(control_initial, dtype=dtype)
    weight_arr = None if weight is None else _broadcast_profile_field(weight, rho)

    def objective_and_profile(control_value, er_seed):
        controlled = apply_profile_basis_control(species_profiles, control_value, control_spec)
        profile = solve_ambipolar_er_profile(
            scan,
            controlled,
            er_initial=er_seed,
            steps=solve_steps,
            damping=damping,
        )
        bootstrap_obj = bootstrap_current_objective(
            rho,
            profile.bootstrap_current_proxy,
            weight=weight_arr,
        )
        residual_obj = residual_scale * jnp.mean(profile.ambipolar_residual**2)
        regularization = control_scale * jnp.sum(control_value**2)
        objective = bootstrap_obj + residual_obj + regularization
        residual_norm = jnp.linalg.norm(profile.ambipolar_residual)
        return objective, (profile, bootstrap_obj, residual_norm)

    def optimization_step(carry, _):
        control_value, er_seed = carry

        def vector_objective(control_trial):
            return objective_and_profile(control_trial, er_seed)

        (objective, (profile, bootstrap_obj, residual_norm)), gradient = jax.value_and_grad(
            vector_objective,
            has_aux=True,
        )(control_value)
        grad_norm = jnp.maximum(jnp.linalg.norm(gradient), jnp.asarray(1.0, dtype=dtype))

        def clip_control(value):
            if control_bound is None:
                return value
            bound = jnp.asarray(control_bound, dtype=dtype)
            return jnp.clip(value, -bound, bound)

        def backtrack_step(step_index, state):
            best_control, best_objective, accepted = state
            factor = 0.5**step_index
            candidate = clip_control(control_value - factor * lr * gradient / grad_norm)
            candidate_objective, _ = objective_and_profile(candidate, er_seed)
            take = (~accepted) & (candidate_objective <= objective)
            next_best_control = jnp.where(take, candidate, best_control)
            next_best_objective = jnp.where(take, candidate_objective, best_objective)
            next_accepted = accepted | take
            return next_best_control, next_best_objective, next_accepted

        initial_candidate = clip_control(control_value - lr * gradient / grad_norm)
        initial_objective, _ = objective_and_profile(initial_candidate, er_seed)
        next_control, _, accepted = jax.lax.fori_loop(
            1,
            backtracking_steps,
            backtrack_step,
            (
                initial_candidate,
                initial_objective,
                initial_objective <= objective,
            ),
        )
        next_control = jnp.where(accepted, next_control, control_value)
        return (next_control, profile.er_profile), (
            control_value,
            objective,
            bootstrap_obj,
            residual_norm,
            profile,
        )

    er_seed0 = 0.5 * (
        jnp.min(jnp.asarray(scan.Er), axis=1) + jnp.max(jnp.asarray(scan.Er), axis=1)
    )
    (_, _), history = jax.lax.scan(
        optimization_step,
        (control0, er_seed0),
        xs=None,
        length=optimization_steps,
    )
    (
        control_history,
        objective_history,
        bootstrap_objective_history,
        residual_norm_history,
        profile_history,
    ) = history
    best_index = jnp.argmin(objective_history)
    return ProfileBasisOptimizationResult(
        control_history=control_history,
        objective_history=objective_history,
        bootstrap_objective_history=bootstrap_objective_history,
        residual_norm_history=residual_norm_history,
        best_control=control_history[best_index],
        best_profile=jax.tree.map(lambda x: x[best_index], profile_history),
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
    return tuple(
        replace(
            species,
            A1=jnp.asarray(species.A1)
            - particle_relaxation[index] * normalized_particle[index],
            A3=jnp.asarray(species.A3)
            - current_relaxation[index] * normalized_current[index],
        )
        for index, species in enumerate(species_profiles)
    )


def solve_profile_transport_loop(
    scan: NeopaxScan,
    species_profiles: tuple[MonoenergeticSpeciesProfile, ...],
    closure_spec: ProfileTransportClosureSpec,
    *,
    iterations: int = 8,
    er_initial: Array | None = None,
    solve_steps: int = 16,
    damping: float = 0.8,
) -> ProfileTransportIterationResult:
    """Iterate a simple self-consistent profile transport closure."""

    species_state = species_profiles
    er_seed = er_initial
    profile_history: list[AmbipolarProfileResult] = []
    loss_history: list[Array] = []
    a1_history: list[Array] = []
    a3_history: list[Array] = []

    for _ in range(iterations):
        profile = solve_ambipolar_er_profile(
            scan,
            species_state,
            er_initial=er_seed,
            steps=solve_steps,
            damping=damping,
        )
        profile_history.append(profile)
        loss_history.append(profile_transport_loss(profile, closure_spec))
        a1_history.append(jnp.stack([jnp.asarray(species.A1) for species in species_state]))
        a3_history.append(jnp.stack([jnp.asarray(species.A3) for species in species_state]))
        species_state = advance_profile_transport(species_state, profile, closure_spec)
        er_seed = profile.er_profile

    loss_array = jnp.stack(loss_history)
    best_index = int(jnp.argmin(loss_array))
    return ProfileTransportIterationResult(
        er_profile_history=jnp.stack([profile.er_profile for profile in profile_history]),
        ambipolar_residual_history=jnp.stack(
            [profile.ambipolar_residual for profile in profile_history]
        ),
        bootstrap_current_proxy_history=jnp.stack(
            [profile.bootstrap_current_proxy for profile in profile_history]
        ),
        transport_loss_history=loss_array,
        species_a1_history=jnp.stack(a1_history),
        species_a3_history=jnp.stack(a3_history),
        best_profile=profile_history[best_index],
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


def _broadcast_profile_field(values, rho: Array) -> Array:
    array = jnp.asarray(values)
    if array.ndim == 0:
        return jnp.full_like(rho, array)
    if array.shape == rho.shape:
        return array
    raise ValueError("profile field must be scalar or match rho shape")


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


def _basis_profile_modifier(control: Array, response: Array, basis: Array) -> Array:
    return jnp.tensordot(control * response, basis, axes=1)


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
