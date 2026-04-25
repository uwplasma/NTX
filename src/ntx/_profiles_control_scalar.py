"""Scalar control helpers for profile-grade NTX workflows."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import Array

from ._profiles_control_types import (
    ProfileControlOptimizationResult,
    ProfileControlSpec,
)
from ._profiles_eval import (
    bootstrap_current_objective,
    solve_ambipolar_er_profile,
)
from ._profiles_radial import _broadcast_profile_field
from ._profiles_species_types import (
    MonoenergeticSpeciesProfile,
)
from .neopax import NeopaxScan


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
    smoothing_strength: float = 0.0,
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
            smoothing_strength=smoothing_strength,
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


__all__ = ["apply_profile_control", "optimize_profile_control"]
