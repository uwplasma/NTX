"""Deterministic bootstrap-current autodiff optimization workflow."""

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
from ._autodiff_types import BootstrapOptimizationResult


def example_bootstrap_current_optimization(
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
    steps: int = 48,
    regularization: float = 5.0e-3,
    serial_seconds: float = 0.0,
    parallel_seconds: float = 0.0,
) -> BootstrapOptimizationResult:
    """Optimize a reduced bootstrap-current proxy through one geometry control."""

    del Es, Er, drds
    context = _build_bootstrap_profile_context(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        nu_index=nu_index,
    )
    scale_grid = jnp.linspace(0.65, 1.35, 29, dtype=context.rho.dtype)

    def transport_profiles(raw_scale: Array) -> tuple[Array, Array, Array]:
        return _transport_profiles_from_raw_scale(
            context,
            raw_scale,
            grid=grid,
            a_b=a_b,
            source_name="bootstrap_current_example",
        )

    def objective(raw_scale: Array) -> Array:
        current, _, _ = transport_profiles(raw_scale)
        scale = _bounded_surface_scale(raw_scale)
        weighted_current = jnp.trapezoid(current * context.objective_weight, context.rho)
        return weighted_current - regularization * (scale - 1.0) ** 2 * 1.0e18

    baseline_raw_scale = jnp.asarray(0.0, dtype=context.rho.dtype)
    initial_raw_scale = jnp.asarray(-0.35, dtype=context.rho.dtype)

    def step(raw_scale: Array, _):
        value, gradient = jax.value_and_grad(objective)(raw_scale)
        next_raw = raw_scale + learning_rate * gradient / jnp.maximum(
            jnp.abs(value),
            1.0e18,
        )
        return next_raw, (_bounded_surface_scale(raw_scale), value, gradient)

    fitted_raw_scale, history = jax.lax.scan(step, initial_raw_scale, xs=None, length=steps)
    scale_history, objective_history, gradient_history = history
    baseline_scale = _bounded_surface_scale(baseline_raw_scale)
    optimized_scale = _bounded_surface_scale(fitted_raw_scale)
    baseline_current_profile, baseline_d13_profile, baseline_d33_profile = (
        transport_profiles(baseline_raw_scale)
    )
    optimized_current_profile, optimized_d13_profile, optimized_d33_profile = (
        transport_profiles(fitted_raw_scale)
    )
    objective_landscape = jax.vmap(
        lambda scale: objective(_raw_scale_from_bounded_scale(scale))
    )(scale_grid)
    current_sensitivity = jax.grad(
        lambda raw_scale: jnp.sum(
            transport_profiles(raw_scale)[0] * context.objective_weight
        )
    )(fitted_raw_scale)[None]
    return BootstrapOptimizationResult(
        scale_history=scale_history,
        gradient_history=gradient_history,
        objective_history=objective_history,
        scale_grid=scale_grid,
        objective_landscape=objective_landscape,
        rho=context.rho,
        baseline_scale=baseline_scale,
        optimized_scale=optimized_scale,
        baseline_current_profile=baseline_current_profile,
        optimized_current_profile=optimized_current_profile,
        baseline_d13_profile=baseline_d13_profile,
        optimized_d13_profile=optimized_d13_profile,
        baseline_d33_profile=baseline_d33_profile,
        optimized_d33_profile=optimized_d33_profile,
        current_sensitivity=current_sensitivity,
        harmonic_m=jnp.asarray(context.harmonic_m),
        harmonic_n=jnp.asarray(context.harmonic_n),
        harmonic_reference_value=jnp.asarray(context.harmonic_reference_value),
        nu_value=jnp.asarray(context.nu_value),
        serial_seconds=jnp.asarray(serial_seconds),
        parallel_seconds=jnp.asarray(parallel_seconds),
    )
