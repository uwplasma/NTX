"""Robust bootstrap-current autodiff optimization workflow."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ._autodiff_bootstrap_common import (
    bounded_surface_scale as _bounded_surface_scale,
)
from ._autodiff_bootstrap_common import (
    build_bootstrap_profile_context as _build_bootstrap_profile_context,
)
from ._autodiff_bootstrap_common import (
    raw_scale_from_bounded_scale as _raw_scale_from_bounded_scale,
)
from ._autodiff_bootstrap_common import (
    transport_profiles_from_raw_scale as _transport_profiles_from_raw_scale,
)
from ._autodiff_types import RobustBootstrapOptimizationResult


def _gauss_hermite_rule(
    quadrature_order: int,
    *,
    dtype,
) -> tuple[Array, Array]:
    if quadrature_order == 3:
        nodes = jnp.asarray([-1.7320508075688772, 0.0, 1.7320508075688772], dtype=dtype)
        weights = jnp.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=dtype)
    elif quadrature_order == 5:
        nodes = jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        weights = jnp.asarray(
            [0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868],
            dtype=dtype,
        )
    else:
        raise ValueError("quadrature_order must be 3 or 5")
    return nodes, weights


def example_bootstrap_current_robust_optimization(
    surfaces: tuple,
    *,
    rho: Array,
    nu_v: Array,
    Es: Array,
    Er: Array,
    drds: Array,
    grid,
    a_b: float = 1.0,
    nu_index: int = 1,
    learning_rate: float = 0.2,
    steps: int = 40,
    regularization: float = 5.0e-3,
    uncertainty_sigma: float = 6.0e-2,
    risk_aversion: float = 0.35,
    scale_grid_size: int = 29,
    quadrature_order: int = 5,
) -> RobustBootstrapOptimizationResult:
    """Optimize a bootstrap-current proxy under prescribed control uncertainty."""

    del Es, Er, drds
    if scale_grid_size < 2:
        raise ValueError("scale_grid_size must be at least 2")
    context = _build_bootstrap_profile_context(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        nu_index=nu_index,
    )
    dtype = context.rho.dtype
    scale_grid = jnp.linspace(0.65, 1.35, scale_grid_size, dtype=dtype)
    sigma = jnp.asarray(uncertainty_sigma, dtype=dtype)
    risk = jnp.asarray(risk_aversion, dtype=dtype)
    quadrature_nodes, quadrature_weights = _gauss_hermite_rule(
        quadrature_order,
        dtype=dtype,
    )

    def current_profile_from_raw(raw_scale: Array) -> Array:
        current, _, _ = _transport_profiles_from_raw_scale(
            context,
            raw_scale,
            grid=grid,
            a_b=a_b,
            source_name="bootstrap_current_robust_example",
        )
        return current

    def scalar_objective(current: Array) -> Array:
        return jnp.trapezoid(current * context.objective_weight, context.rho)

    def robust_moments(raw_scale: Array) -> tuple[Array, Array, Array]:
        sample_raw = raw_scale + sigma * quadrature_nodes
        sample_currents = jax.vmap(current_profile_from_raw)(sample_raw)
        sample_objectives = jax.vmap(scalar_objective)(sample_currents)
        mean_current = jnp.tensordot(quadrature_weights, sample_currents, axes=1)
        mean_objective = jnp.dot(quadrature_weights, sample_objectives)
        variance = jnp.dot(
            quadrature_weights,
            (sample_objectives - mean_objective) ** 2,
        )
        return mean_current, mean_objective, jnp.sqrt(jnp.maximum(variance, 0.0))

    def deterministic_objective(raw_scale: Array) -> Array:
        current = current_profile_from_raw(raw_scale)
        scale = _bounded_surface_scale(raw_scale)
        return scalar_objective(current) - regularization * (scale - 1.0) ** 2 * 1.0e18

    def robust_objective(raw_scale: Array) -> Array:
        _, mean_objective, objective_std = robust_moments(raw_scale)
        scale = _bounded_surface_scale(raw_scale)
        return (
            mean_objective
            - risk * objective_std
            - regularization * (scale - 1.0) ** 2 * 1.0e18
        )

    baseline_raw_scale = jnp.asarray(0.0, dtype=dtype)
    initial_raw_scale = jnp.asarray(-0.35, dtype=dtype)

    def step(raw_scale: Array, _):
        value, gradient = jax.value_and_grad(robust_objective)(raw_scale)
        next_raw = raw_scale + learning_rate * gradient / jnp.maximum(
            jnp.abs(value),
            1.0e18,
        )
        return next_raw, (_bounded_surface_scale(raw_scale), value, gradient)

    fitted_raw_scale, history = jax.lax.scan(step, initial_raw_scale, xs=None, length=steps)
    scale_history, objective_history, gradient_history = history
    baseline_scale = _bounded_surface_scale(baseline_raw_scale)
    optimized_scale = _bounded_surface_scale(fitted_raw_scale)
    baseline_current_profile = current_profile_from_raw(baseline_raw_scale)
    optimized_current_profile = current_profile_from_raw(fitted_raw_scale)
    sample_raw = fitted_raw_scale + sigma * quadrature_nodes
    sample_currents = jax.vmap(current_profile_from_raw)(sample_raw)
    optimized_current_mean = jnp.tensordot(quadrature_weights, sample_currents, axes=1)
    centered_current = sample_currents - optimized_current_mean[None, :]
    optimized_current_std = jnp.sqrt(
        jnp.tensordot(quadrature_weights, centered_current**2, axes=1)
    )
    optimized_current_quantile_low = jnp.quantile(sample_currents, 0.16, axis=0)
    optimized_current_quantile_high = jnp.quantile(sample_currents, 0.84, axis=0)
    deterministic_objective_landscape = jax.vmap(
        lambda scale: deterministic_objective(_raw_scale_from_bounded_scale(scale))
    )(scale_grid)
    robust_objective_landscape = jax.vmap(
        lambda scale: robust_objective(_raw_scale_from_bounded_scale(scale))
    )(scale_grid)
    return RobustBootstrapOptimizationResult(
        scale_history=scale_history,
        gradient_history=gradient_history,
        objective_history=objective_history,
        scale_grid=scale_grid,
        deterministic_objective_landscape=deterministic_objective_landscape,
        robust_objective_landscape=robust_objective_landscape,
        rho=context.rho,
        baseline_scale=baseline_scale,
        optimized_scale=optimized_scale,
        baseline_current_profile=baseline_current_profile,
        optimized_current_profile=optimized_current_profile,
        optimized_current_mean=optimized_current_mean,
        optimized_current_std=optimized_current_std,
        optimized_current_quantile_low=optimized_current_quantile_low,
        optimized_current_quantile_high=optimized_current_quantile_high,
        harmonic_m=jnp.asarray(context.harmonic_m),
        harmonic_n=jnp.asarray(context.harmonic_n),
        harmonic_reference_value=jnp.asarray(context.harmonic_reference_value),
        nu_value=jnp.asarray(context.nu_value),
        uncertainty_sigma=sigma,
        risk_aversion=risk,
    )
