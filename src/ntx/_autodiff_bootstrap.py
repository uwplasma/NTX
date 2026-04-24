"""Bootstrap-current autodiff optimization workflows."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ._autodiff_helpers import (
    dominant_nonaxisymmetric_mode as _dominant_nonaxisymmetric_mode,
)
from ._autodiff_helpers import mode_value_for_surface as _mode_value_for_surface
from ._autodiff_helpers import scale_surface_mode as _scale_surface_mode
from ._autodiff_types import (
    BootstrapOptimizationResult,
    RobustBootstrapOptimizationResult,
)
from .neopax import (
    build_ntx_neopax_scan_from_surfaces,
    scan_to_neopax_arrays,
)


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
    rho_grid = jnp.asarray(rho)
    dtype = rho_grid.dtype
    nu_value = jnp.asarray(nu_v[nu_index], dtype=dtype)
    density = 3.2e19 * (1.0 - rho_grid**4) + 0.45e19
    temperature = 3.0e3 * (1.0 - rho_grid**2) + 0.8e3
    density_gradient = jnp.asarray(jnp.gradient(jnp.log(density), rho_grid))
    temperature_gradient = jnp.asarray(jnp.gradient(jnp.log(temperature), rho_grid))
    objective_weight = jnp.exp(-0.5 * ((rho_grid - 0.45) / 0.16) ** 2)
    scale_grid = jnp.linspace(0.65, 1.35, 29, dtype=dtype)
    harmonic_m, harmonic_n = _dominant_nonaxisymmetric_mode(surfaces[len(surfaces) // 2])
    harmonic_reference_value = _mode_value_for_surface(
        surfaces[len(surfaces) // 2],
        harmonic_m,
        harmonic_n,
    )
    zero_scan = jnp.zeros((rho_grid.size, 1), dtype=dtype)
    unit_drds = jnp.ones_like(rho_grid)

    def bounded_scale(raw_scale: Array) -> Array:
        return 1.0 + 0.35 * jnp.tanh(raw_scale)

    def transport_profiles(raw_scale: Array) -> tuple[Array, Array, Array]:
        scale = bounded_scale(raw_scale)
        perturbed_surfaces = tuple(
            _scale_surface_mode(surface, harmonic_m, harmonic_n, scale) for surface in surfaces
        )
        scan = build_ntx_neopax_scan_from_surfaces(
            perturbed_surfaces,
            rho=rho_grid,
            nu_v=jnp.asarray([nu_value]),
            Es=zero_scan,
            Er=zero_scan,
            drds=unit_drds,
            grid=grid,
            source_name="bootstrap_current_example",
        )
        arrays = scan_to_neopax_arrays(scan, a_b=a_b)
        d13 = arrays.D13[:, 0, 0]
        d33 = arrays.D33[:, 0, 0]
        current = density * (-density_gradient * d13 - 0.75 * temperature_gradient * d33)
        return current, d13, d33

    def objective(raw_scale: Array) -> Array:
        current, _, _ = transport_profiles(raw_scale)
        scale = bounded_scale(raw_scale)
        weighted_current = jnp.trapezoid(current * objective_weight, rho_grid)
        return weighted_current - regularization * (scale - 1.0) ** 2 * 1.0e18

    baseline_raw_scale = jnp.asarray(0.0, dtype=dtype)
    initial_raw_scale = jnp.asarray(-0.35, dtype=dtype)

    def step(raw_scale: Array, _):
        value, gradient = jax.value_and_grad(objective)(raw_scale)
        next_raw = raw_scale + learning_rate * gradient / jnp.maximum(jnp.abs(value), 1.0e18)
        return next_raw, (bounded_scale(raw_scale), value, gradient)

    fitted_raw_scale, history = jax.lax.scan(step, initial_raw_scale, xs=None, length=steps)
    scale_history, objective_history, gradient_history = history
    baseline_scale = bounded_scale(baseline_raw_scale)
    optimized_scale = bounded_scale(fitted_raw_scale)
    baseline_current_profile, baseline_d13_profile, baseline_d33_profile = transport_profiles(
        baseline_raw_scale
    )
    optimized_current_profile, optimized_d13_profile, optimized_d33_profile = transport_profiles(
        fitted_raw_scale
    )
    objective_landscape = jax.vmap(
        lambda scale: objective(
            jnp.arctanh(jnp.clip((scale - 1.0) / 0.35, -0.999, 0.999))
        )
    )(scale_grid)
    current_sensitivity = jax.grad(
        lambda raw_scale: jnp.sum(transport_profiles(raw_scale)[0] * objective_weight)
    )(fitted_raw_scale)[None]
    return BootstrapOptimizationResult(
        scale_history=scale_history,
        gradient_history=gradient_history,
        objective_history=objective_history,
        scale_grid=scale_grid,
        objective_landscape=objective_landscape,
        rho=rho_grid,
        baseline_scale=baseline_scale,
        optimized_scale=optimized_scale,
        baseline_current_profile=baseline_current_profile,
        optimized_current_profile=optimized_current_profile,
        baseline_d13_profile=baseline_d13_profile,
        optimized_d13_profile=optimized_d13_profile,
        baseline_d33_profile=baseline_d33_profile,
        optimized_d33_profile=optimized_d33_profile,
        current_sensitivity=current_sensitivity,
        harmonic_m=jnp.asarray(harmonic_m),
        harmonic_n=jnp.asarray(harmonic_n),
        harmonic_reference_value=jnp.asarray(harmonic_reference_value),
        nu_value=jnp.asarray(nu_value),
        serial_seconds=jnp.asarray(serial_seconds),
        parallel_seconds=jnp.asarray(parallel_seconds),
    )


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
    if quadrature_order not in (3, 5):
        raise ValueError("quadrature_order must be 3 or 5")
    rho_grid = jnp.asarray(rho)
    dtype = rho_grid.dtype
    nu_value = jnp.asarray(nu_v[nu_index], dtype=dtype)
    density = 3.2e19 * (1.0 - rho_grid**4) + 0.45e19
    temperature = 3.0e3 * (1.0 - rho_grid**2) + 0.8e3
    density_gradient = jnp.asarray(jnp.gradient(jnp.log(density), rho_grid))
    temperature_gradient = jnp.asarray(jnp.gradient(jnp.log(temperature), rho_grid))
    objective_weight = jnp.exp(-0.5 * ((rho_grid - 0.45) / 0.16) ** 2)
    scale_grid = jnp.linspace(0.65, 1.35, scale_grid_size, dtype=dtype)
    harmonic_m, harmonic_n = _dominant_nonaxisymmetric_mode(surfaces[len(surfaces) // 2])
    harmonic_reference_value = _mode_value_for_surface(
        surfaces[len(surfaces) // 2],
        harmonic_m,
        harmonic_n,
    )
    zero_scan = jnp.zeros((rho_grid.size, 1), dtype=dtype)
    unit_drds = jnp.ones_like(rho_grid)
    sigma = jnp.asarray(uncertainty_sigma, dtype=dtype)
    risk = jnp.asarray(risk_aversion, dtype=dtype)
    if quadrature_order == 3:
        quadrature_nodes = jnp.asarray([-1.7320508075688772, 0.0, 1.7320508075688772], dtype=dtype)
        quadrature_weights = jnp.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=dtype)
    else:
        quadrature_nodes = jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        quadrature_weights = jnp.asarray(
            [0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868],
            dtype=dtype,
        )

    def bounded_scale(raw_scale: Array) -> Array:
        return 1.0 + 0.35 * jnp.tanh(raw_scale)

    def current_profile_from_raw(raw_scale: Array) -> Array:
        scale = bounded_scale(raw_scale)
        perturbed_surfaces = tuple(
            _scale_surface_mode(surface, harmonic_m, harmonic_n, scale) for surface in surfaces
        )
        scan = build_ntx_neopax_scan_from_surfaces(
            perturbed_surfaces,
            rho=rho_grid,
            nu_v=jnp.asarray([nu_value]),
            Es=zero_scan,
            Er=zero_scan,
            drds=unit_drds,
            grid=grid,
            source_name="bootstrap_current_robust_example",
        )
        arrays = scan_to_neopax_arrays(scan, a_b=a_b)
        d13 = arrays.D13[:, 0, 0]
        d33 = arrays.D33[:, 0, 0]
        return density * (-density_gradient * d13 - 0.75 * temperature_gradient * d33)

    def scalar_objective(current: Array) -> Array:
        return jnp.trapezoid(current * objective_weight, rho_grid)

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
        scale = bounded_scale(raw_scale)
        return scalar_objective(current) - regularization * (scale - 1.0) ** 2 * 1.0e18

    def robust_objective(raw_scale: Array) -> Array:
        _, mean_objective, objective_std = robust_moments(raw_scale)
        scale = bounded_scale(raw_scale)
        return (
            mean_objective
            - risk * objective_std
            - regularization * (scale - 1.0) ** 2 * 1.0e18
        )

    baseline_raw_scale = jnp.asarray(0.0, dtype=dtype)
    initial_raw_scale = jnp.asarray(-0.35, dtype=dtype)

    def step(raw_scale: Array, _):
        value, gradient = jax.value_and_grad(robust_objective)(raw_scale)
        next_raw = raw_scale + learning_rate * gradient / jnp.maximum(jnp.abs(value), 1.0e18)
        return next_raw, (bounded_scale(raw_scale), value, gradient)

    fitted_raw_scale, history = jax.lax.scan(step, initial_raw_scale, xs=None, length=steps)
    scale_history, objective_history, gradient_history = history
    baseline_scale = bounded_scale(baseline_raw_scale)
    optimized_scale = bounded_scale(fitted_raw_scale)
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
        lambda scale: deterministic_objective(
            jnp.arctanh(jnp.clip((scale - 1.0) / 0.35, -0.999, 0.999))
        )
    )(scale_grid)
    robust_objective_landscape = jax.vmap(
        lambda scale: robust_objective(
            jnp.arctanh(jnp.clip((scale - 1.0) / 0.35, -0.999, 0.999))
        )
    )(scale_grid)
    return RobustBootstrapOptimizationResult(
        scale_history=scale_history,
        gradient_history=gradient_history,
        objective_history=objective_history,
        scale_grid=scale_grid,
        deterministic_objective_landscape=deterministic_objective_landscape,
        robust_objective_landscape=robust_objective_landscape,
        rho=rho_grid,
        baseline_scale=baseline_scale,
        optimized_scale=optimized_scale,
        baseline_current_profile=baseline_current_profile,
        optimized_current_profile=optimized_current_profile,
        optimized_current_mean=optimized_current_mean,
        optimized_current_std=optimized_current_std,
        optimized_current_quantile_low=optimized_current_quantile_low,
        optimized_current_quantile_high=optimized_current_quantile_high,
        harmonic_m=jnp.asarray(harmonic_m),
        harmonic_n=jnp.asarray(harmonic_n),
        harmonic_reference_value=jnp.asarray(harmonic_reference_value),
        nu_value=jnp.asarray(nu_value),
        uncertainty_sigma=sigma,
        risk_aversion=risk,
    )
