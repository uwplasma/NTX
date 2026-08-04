"""Bootstrap-current derivatives, deterministic and robust.

The bootstrap response is the noisiest derivative NTX reports; this keeps the
deterministic path and the robust estimator side by side with their shared
machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from ._autodiff import BootstrapOptimizationResult, RobustBootstrapOptimizationResult
from ._autodiff import (
    dominant_nonaxisymmetric_mode as _dominant_nonaxisymmetric_mode,
)
from ._autodiff import mode_value_for_surface as _mode_value_for_surface
from ._autodiff import scale_surface_mode as _scale_surface_mode
from .neopax import (
    build_ntx_neopax_scan_from_surfaces,
    scan_to_neopax_arrays,
)

__all__ = [
    "example_bootstrap_current_optimization",
    "example_bootstrap_current_robust_optimization",
]


# --- _autodiff_bootstrap_common: Shared helpers for bootstrap-current autodiff workflows. ---


@dataclass(frozen=True)
class BootstrapProfileContext:
    """Precomputed profile and harmonic data for bootstrap-current examples."""

    surfaces: tuple
    rho: Array
    nu_value: Array
    density: Array
    density_gradient: Array
    temperature_gradient: Array
    objective_weight: Array
    harmonic_m: int
    harmonic_n: int
    harmonic_reference_value: Array
    zero_scan: Array
    unit_drds: Array


def build_bootstrap_profile_context(
    surfaces: tuple,
    *,
    rho: Array,
    nu_v: Array,
    nu_index: int,
) -> BootstrapProfileContext:
    """Build the shared radial profiles and controlled harmonic metadata."""

    rho_grid = jnp.asarray(rho)
    dtype = rho_grid.dtype
    nu_value = jnp.asarray(nu_v[nu_index], dtype=dtype)
    density = 3.2e19 * (1.0 - rho_grid**4) + 0.45e19
    temperature = 3.0e3 * (1.0 - rho_grid**2) + 0.8e3
    density_gradient = jnp.asarray(jnp.gradient(jnp.log(density), rho_grid))
    temperature_gradient = jnp.asarray(jnp.gradient(jnp.log(temperature), rho_grid))
    objective_weight = jnp.exp(-0.5 * ((rho_grid - 0.45) / 0.16) ** 2)
    harmonic_m, harmonic_n = _dominant_nonaxisymmetric_mode(surfaces[len(surfaces) // 2])
    harmonic_reference_value = _mode_value_for_surface(
        surfaces[len(surfaces) // 2],
        harmonic_m,
        harmonic_n,
    )
    zero_scan = jnp.zeros((rho_grid.size, 1), dtype=dtype)
    unit_drds = jnp.ones_like(rho_grid)
    return BootstrapProfileContext(
        surfaces=surfaces,
        rho=rho_grid,
        nu_value=nu_value,
        density=density,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
        objective_weight=objective_weight,
        harmonic_m=harmonic_m,
        harmonic_n=harmonic_n,
        harmonic_reference_value=jnp.asarray(harmonic_reference_value),
        zero_scan=zero_scan,
        unit_drds=unit_drds,
    )


def bounded_surface_scale(raw_scale: Array) -> Array:
    """Map an unconstrained scalar control to the bounded harmonic scale."""

    return 1.0 + 0.35 * jnp.tanh(raw_scale)


def raw_scale_from_bounded_scale(scale: Array) -> Array:
    """Invert :func:`bounded_surface_scale` on the plotted scale interval."""

    return jnp.arctanh(jnp.clip((scale - 1.0) / 0.35, -0.999, 0.999))


def transport_profiles_from_raw_scale(
    context: BootstrapProfileContext,
    raw_scale: Array,
    *,
    grid,
    a_b: float,
    source_name: str,
) -> tuple[Array, Array, Array]:
    """Evaluate the reduced current response and selected monoenergetic profiles."""

    scale = bounded_surface_scale(raw_scale)
    perturbed_surfaces = tuple(
        _scale_surface_mode(
            surface,
            context.harmonic_m,
            context.harmonic_n,
            scale,
        )
        for surface in context.surfaces
    )
    scan = build_ntx_neopax_scan_from_surfaces(
        perturbed_surfaces,
        rho=context.rho,
        nu_v=jnp.asarray([context.nu_value]),
        Es=context.zero_scan,
        Er=context.zero_scan,
        drds=context.unit_drds,
        grid=grid,
        source_name=source_name,
    )
    arrays = scan_to_neopax_arrays(scan, a_b=a_b)
    d13 = arrays.D13[:, 0, 0]
    d33 = arrays.D33[:, 0, 0]
    current = context.density * (
        -context.density_gradient * d13 - 0.75 * context.temperature_gradient * d33
    )
    return current, d13, d33


# --- _autodiff_bootstrap_deterministic ---
# Deterministic bootstrap-current autodiff optimization workflow.


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
    """Optimize a reduced bootstrap-current response through one geometry control."""

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
    baseline_current_profile, baseline_d13_profile, baseline_d33_profile = transport_profiles(
        baseline_raw_scale
    )
    optimized_current_profile, optimized_d13_profile, optimized_d33_profile = transport_profiles(
        fitted_raw_scale
    )
    objective_landscape = jax.vmap(lambda scale: objective(_raw_scale_from_bounded_scale(scale)))(
        scale_grid
    )
    current_sensitivity = jax.grad(
        lambda raw_scale: jnp.sum(transport_profiles(raw_scale)[0] * context.objective_weight)
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


# --- _autodiff_bootstrap_robust: Robust bootstrap-current autodiff optimization workflow. ---


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
    """Optimize a reduced bootstrap-current response under control uncertainty."""

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
        return mean_objective - risk * objective_std - regularization * (scale - 1.0) ** 2 * 1.0e18

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
    optimized_current_std = jnp.sqrt(jnp.tensordot(quadrature_weights, centered_current**2, axes=1))
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


# Underscore aliases kept for callers that imported these from the modules
# fused into this one; the public names above are the definitions.
_bounded_surface_scale = bounded_surface_scale
_build_bootstrap_profile_context = build_bootstrap_profile_context
_raw_scale_from_bounded_scale = raw_scale_from_bounded_scale
_transport_profiles_from_raw_scale = transport_profiles_from_raw_scale
