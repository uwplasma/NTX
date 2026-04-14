"""Autodiff-oriented NTX demonstrations and analysis helpers."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jax import Array, tree_util

from ._checkout_paths import find_neopax_root
from .geometry import BoozerSurface, VmecSurface, example_surface
from .grids import GridSpec
from .neopax import (
    NeopaxMonoenergeticArrays,
    build_ntx_neopax_scan_from_surfaces,
    scan_to_neopax_arrays,
)
from .solver import solve_monoenergetic_scan


@dataclass(frozen=True)
class InverseProblemResult:
    amplitude_history: Array
    gradient_history: Array
    loss_history: Array
    nu_hat: Array
    target_response: Array
    initial_response: Array
    fitted_response: Array
    sensitivity: Array
    inferred_amplitude: Array
    target_amplitude: Array


tree_util.register_dataclass(
    InverseProblemResult,
    data_fields=(
        "amplitude_history",
        "gradient_history",
        "loss_history",
        "nu_hat",
        "target_response",
        "initial_response",
        "fitted_response",
        "sensitivity",
        "inferred_amplitude",
        "target_amplitude",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class NeopaxProfileAutodiffResult:
    parameter_history: Array
    loss_history: Array
    rho: Array
    target_er_profile: Array
    fitted_er_profile: Array
    target_d11_profile: Array
    fitted_d11_profile: Array
    target_d33_profile: Array
    fitted_d33_profile: Array
    sensitivity_matrix: Array
    nu_value: Array


tree_util.register_dataclass(
    NeopaxProfileAutodiffResult,
    data_fields=(
        "parameter_history",
        "loss_history",
        "rho",
        "target_er_profile",
        "fitted_er_profile",
        "target_d11_profile",
        "fitted_d11_profile",
        "target_d33_profile",
        "fitted_d33_profile",
        "sensitivity_matrix",
        "nu_value",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class BootstrapOptimizationResult:
    scale_history: Array
    gradient_history: Array
    objective_history: Array
    scale_grid: Array
    objective_landscape: Array
    rho: Array
    baseline_scale: Array
    optimized_scale: Array
    baseline_current_profile: Array
    optimized_current_profile: Array
    baseline_d13_profile: Array
    optimized_d13_profile: Array
    baseline_d33_profile: Array
    optimized_d33_profile: Array
    current_sensitivity: Array
    harmonic_m: Array
    harmonic_n: Array
    harmonic_reference_value: Array
    nu_value: Array
    serial_seconds: Array
    parallel_seconds: Array


tree_util.register_dataclass(
    BootstrapOptimizationResult,
    data_fields=(
        "scale_history",
        "gradient_history",
        "objective_history",
        "scale_grid",
        "objective_landscape",
        "rho",
        "baseline_scale",
        "optimized_scale",
        "baseline_current_profile",
        "optimized_current_profile",
        "baseline_d13_profile",
        "optimized_d13_profile",
        "baseline_d33_profile",
        "optimized_d33_profile",
        "current_sensitivity",
        "harmonic_m",
        "harmonic_n",
        "harmonic_reference_value",
        "nu_value",
        "serial_seconds",
        "parallel_seconds",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class DerivativeAuditResult:
    nu_hat: Array
    er_hat_scan: Array
    amplitude_value: Array
    er_reference: Array
    nu_reference: Array
    autodiff_d11_da: Array
    finite_difference_d11_da: Array
    autodiff_d33_da: Array
    finite_difference_d33_da: Array
    autodiff_d11_der: Array
    finite_difference_d11_der: Array
    autodiff_d33_der: Array
    finite_difference_d33_der: Array


tree_util.register_dataclass(
    DerivativeAuditResult,
    data_fields=(
        "nu_hat",
        "er_hat_scan",
        "amplitude_value",
        "er_reference",
        "nu_reference",
        "autodiff_d11_da",
        "finite_difference_d11_da",
        "autodiff_d33_da",
        "finite_difference_d33_da",
        "autodiff_d11_der",
        "finite_difference_d11_der",
        "autodiff_d33_der",
        "finite_difference_d33_der",
    ),
    meta_fields=(),
)


def example_inverse_problem(
    *,
    grid: GridSpec | None = None,
    nu_hat: Array | None = None,
    er_hat: float = 1e-3,
    target_amplitude: float = 0.085,
    initial_amplitude: float = 0.03,
    learning_rate: float = 0.5,
    steps: int = 24,
    coefficient_index: int = 1,
) -> InverseProblemResult:
    """Recover one Boozer-harmonic amplitude from synthetic transport data."""

    grid = GridSpec(7, 9, 6) if grid is None else grid
    nu_hat = (
        jnp.logspace(-4, -2, 8)
        if nu_hat is None
        else jnp.asarray(nu_hat, dtype=grid.jax_dtype)
    )
    base_surface = example_surface(dtype=grid.jax_dtype)
    target_surface = _surface_with_amplitude(
        base_surface,
        coefficient_index,
        target_amplitude,
    )
    target_response = _inverse_problem_response(target_surface, grid, nu_hat, er_hat)

    def loss_fn(amplitude):
        surface = _surface_with_amplitude(base_surface, coefficient_index, amplitude)
        response = _inverse_problem_response(surface, grid, nu_hat, er_hat)
        return jnp.mean(
            (
                jnp.log10(jnp.maximum(response, 1e-12))
                - jnp.log10(jnp.maximum(target_response, 1e-12))
            )
            ** 2
        )

    initial_response = _inverse_problem_response(
        _surface_with_amplitude(base_surface, coefficient_index, initial_amplitude),
        grid,
        nu_hat,
        er_hat,
    )

    def step(amplitude, _):
        loss, gradient = jax.value_and_grad(loss_fn)(amplitude)
        trial_amplitude = jnp.clip(amplitude - learning_rate * gradient, 1e-3, 0.3)
        trial_loss = loss_fn(trial_amplitude)

        def cond_fn(state):
            _, step_size, candidate_loss, count = state
            return jnp.logical_and(candidate_loss > loss, count < 6)

        def body_fn(state):
            _, step_size, _, count = state
            next_step = step_size * 0.5
            candidate = jnp.clip(amplitude - next_step * gradient, 1e-3, 0.3)
            candidate_loss = loss_fn(candidate)
            return candidate, next_step, candidate_loss, count + 1

        next_amplitude, _, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (trial_amplitude, jnp.asarray(learning_rate), trial_loss, jnp.asarray(0)),
        )
        return next_amplitude, (next_amplitude, loss, gradient)

    inferred_amplitude, history = jax.lax.scan(
        step,
        jnp.asarray(initial_amplitude, dtype=grid.jax_dtype),
        xs=None,
        length=steps,
    )
    amplitude_history, loss_history, gradient_history = history
    fitted_surface = _surface_with_amplitude(base_surface, coefficient_index, inferred_amplitude)
    fitted_response = _inverse_problem_response(fitted_surface, grid, nu_hat, er_hat)
    sensitivity = jax.grad(
        lambda amplitude: jnp.sum(
            _inverse_problem_response(
                _surface_with_amplitude(base_surface, coefficient_index, amplitude),
                grid,
                nu_hat,
                er_hat,
            )
        )
    )(inferred_amplitude)
    return InverseProblemResult(
        amplitude_history=amplitude_history,
        gradient_history=gradient_history,
        loss_history=loss_history,
        nu_hat=nu_hat,
        target_response=target_response,
        initial_response=initial_response,
        fitted_response=fitted_response,
        sensitivity=sensitivity[None],
        inferred_amplitude=inferred_amplitude,
        target_amplitude=jnp.asarray(target_amplitude, dtype=grid.jax_dtype),
    )


def example_neopax_profile_autodiff(
    surfaces: tuple,
    *,
    rho: Array,
    nu_v: Array,
    Es: Array,
    Er: Array,
    drds: Array,
    grid: GridSpec,
    a_b: float = 1.0,
    nu_index: int = 1,
    learning_rate: float = 0.25,
    steps: int = 32,
    use_neopax_package: bool = False,
) -> NeopaxProfileAutodiffResult:
    """Infer a low-dimensional electric-field profile on a NEOPAX-style scan."""

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=jnp.asarray(rho),
        nu_v=jnp.asarray(nu_v),
        Es=jnp.asarray(Es),
        Er=jnp.asarray(Er),
        drds=jnp.asarray(drds),
        grid=grid,
        source_name="autodiff_profile_example",
    )
    arrays = scan_to_neopax_arrays(scan, a_b=a_b)
    rho_grid = jnp.asarray(arrays.rho)
    nu_value = 10.0 ** arrays.nu_log[nu_index]
    target_params = jnp.asarray([1.4e-3, -6.0e-4], dtype=rho_grid.dtype)
    initial_params = jnp.asarray([5.0e-4, 2.0e-4], dtype=rho_grid.dtype)
    target_er_profile = _er_profile(rho_grid, target_params)
    target_d11_profile = _evaluate_d11_profile(arrays, rho_grid, nu_value, target_er_profile)
    target_d33_profile = _evaluate_d33_profile(arrays, rho_grid, nu_value, target_er_profile)

    def loss_fn(params):
        trial_er = _er_profile(rho_grid, params)
        fitted_d11 = _evaluate_d11_profile(arrays, rho_grid, nu_value, trial_er)
        fitted = _evaluate_d33_profile(
            arrays,
            rho_grid,
            nu_value,
            trial_er,
        )
        d11_residual = (
            jnp.log10(jnp.maximum(fitted_d11, 1e-30))
            - jnp.log10(jnp.maximum(target_d11_profile, 1e-30))
        )
        d33_residual = (fitted - target_d33_profile) / jnp.maximum(
            jnp.abs(target_d33_profile),
            1e-12,
        )
        return jnp.mean(d11_residual**2) + jnp.mean(d33_residual**2)

    def step(params, _):
        loss, grad = jax.value_and_grad(loss_fn)(params)
        next_params = params - learning_rate * grad
        return next_params, (next_params, loss)

    fitted_params, history = jax.lax.scan(
        step,
        initial_params,
        xs=None,
        length=steps,
    )
    parameter_history, loss_history = history
    fitted_er_profile = _er_profile(rho_grid, fitted_params)
    fitted_d11_profile = _evaluate_d11_profile(arrays, rho_grid, nu_value, fitted_er_profile)
    fitted_d33_profile = _evaluate_d33_profile(arrays, rho_grid, nu_value, fitted_er_profile)
    sensitivity_matrix = jax.jacrev(
        lambda params: _evaluate_d33_profile(
            arrays,
            rho_grid,
            nu_value,
            _er_profile(rho_grid, params),
        )
    )(fitted_params)
    if use_neopax_package:
        _ = _maybe_import_neopax()
    return NeopaxProfileAutodiffResult(
        parameter_history=parameter_history,
        loss_history=loss_history,
        rho=rho_grid,
        target_er_profile=target_er_profile,
        fitted_er_profile=fitted_er_profile,
        target_d11_profile=target_d11_profile,
        fitted_d11_profile=fitted_d11_profile,
        target_d33_profile=target_d33_profile,
        fitted_d33_profile=fitted_d33_profile,
        sensitivity_matrix=sensitivity_matrix,
        nu_value=jnp.asarray(nu_value),
    )


def example_derivative_audit(
    *,
    grid: GridSpec | None = None,
    coefficient_index: int = 1,
    amplitude_value: float = 0.085,
    nu_hat: Array | None = None,
    er_hat_scan: Array | None = None,
    er_reference: float = 1.0e-3,
    nu_reference: float = 3.0e-4,
    fd_step_amplitude: float = 1.0e-4,
    fd_step_er: float = 1.0e-5,
) -> DerivativeAuditResult:
    """Compare direct JAX gradients against finite differences.

    This is the first validation layer for the future implicit/adjoint solve.
    It keeps the current reverse-mode path honest on a compact test problem
    before introducing custom VJP logic for larger optimization loops.
    """

    grid = GridSpec(7, 9, 6) if grid is None else grid
    nu_hat = (
        jnp.logspace(-4.5, -1.5, 9)
        if nu_hat is None
        else jnp.asarray(nu_hat, dtype=grid.jax_dtype)
    )
    er_hat_scan = (
        jnp.logspace(-6, -2.5, 8)
        if er_hat_scan is None
        else jnp.asarray(er_hat_scan, dtype=grid.jax_dtype)
    )
    base_surface = example_surface(dtype=grid.jax_dtype)

    def d11_curve(amplitude):
        surface = _surface_with_amplitude(base_surface, coefficient_index, amplitude)
        return solve_monoenergetic_scan(
            surface,
            grid,
            nu_hat,
            er_hat=jnp.full_like(nu_hat, er_reference),
        )["D11"].reshape(-1)

    def d33_curve(amplitude):
        surface = _surface_with_amplitude(base_surface, coefficient_index, amplitude)
        return solve_monoenergetic_scan(
            surface,
            grid,
            nu_hat,
            er_hat=jnp.full_like(nu_hat, er_reference),
        )["D33"].reshape(-1)

    autodiff_d11_da = jax.jacrev(d11_curve)(amplitude_value)
    finite_difference_d11_da = (
        d11_curve(amplitude_value + fd_step_amplitude)
        - d11_curve(amplitude_value - fd_step_amplitude)
    ) / (2.0 * fd_step_amplitude)
    autodiff_d33_da = jax.jacrev(d33_curve)(amplitude_value)
    finite_difference_d33_da = (
        d33_curve(amplitude_value + fd_step_amplitude)
        - d33_curve(amplitude_value - fd_step_amplitude)
    ) / (2.0 * fd_step_amplitude)

    fixed_surface = _surface_with_amplitude(base_surface, coefficient_index, amplitude_value)

    def d11_at_er(er_value):
        return solve_monoenergetic_scan(
            fixed_surface,
            grid,
            jnp.asarray([nu_reference], dtype=grid.jax_dtype),
            er_hat=jnp.asarray([er_value], dtype=grid.jax_dtype),
        )["D11"].reshape(-1)[0]

    def d33_at_er(er_value):
        return solve_monoenergetic_scan(
            fixed_surface,
            grid,
            jnp.asarray([nu_reference], dtype=grid.jax_dtype),
            er_hat=jnp.asarray([er_value], dtype=grid.jax_dtype),
        )["D33"].reshape(-1)[0]

    autodiff_d11_der = jax.vmap(jax.grad(d11_at_er))(er_hat_scan)
    finite_difference_d11_der = jax.vmap(
        lambda value: (d11_at_er(value + fd_step_er) - d11_at_er(value - fd_step_er))
        / (2.0 * fd_step_er)
    )(er_hat_scan)
    autodiff_d33_der = jax.vmap(jax.grad(d33_at_er))(er_hat_scan)
    finite_difference_d33_der = jax.vmap(
        lambda value: (d33_at_er(value + fd_step_er) - d33_at_er(value - fd_step_er))
        / (2.0 * fd_step_er)
    )(er_hat_scan)

    return DerivativeAuditResult(
        nu_hat=nu_hat,
        er_hat_scan=er_hat_scan,
        amplitude_value=jnp.asarray(amplitude_value, dtype=grid.jax_dtype),
        er_reference=jnp.asarray(er_reference, dtype=grid.jax_dtype),
        nu_reference=jnp.asarray(nu_reference, dtype=grid.jax_dtype),
        autodiff_d11_da=autodiff_d11_da,
        finite_difference_d11_da=finite_difference_d11_da,
        autodiff_d33_da=autodiff_d33_da,
        finite_difference_d33_da=finite_difference_d33_da,
        autodiff_d11_der=autodiff_d11_der,
        finite_difference_d11_der=finite_difference_d11_der,
        autodiff_d33_der=autodiff_d33_der,
        finite_difference_d33_der=finite_difference_d33_der,
    )


def example_bootstrap_current_optimization(
    surfaces: tuple,
    *,
    rho: Array,
    nu_v: Array,
    Es: Array,
    Er: Array,
    drds: Array,
    grid: GridSpec,
    a_b: float = 1.0,
    nu_index: int = 1,
    learning_rate: float = 0.2,
    steps: int = 48,
    regularization: float = 5.0e-3,
    serial_seconds: float = 0.0,
    parallel_seconds: float = 0.0,
) -> BootstrapOptimizationResult:
    """Optimize a reduced bootstrap-current proxy through one geometry control.

    The optimized control is a single non-axisymmetric magnetic-field harmonic
    that is present across the provided radial family. NTX differentiates a
    weighted bootstrap-current proxy with respect to that harmonic scale and
    performs a compact gradient-based optimization.
    """

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

    fitted_raw_scale, history = jax.lax.scan(
        step,
        initial_raw_scale,
        xs=None,
        length=steps,
    )
    scale_history, objective_history, gradient_history = history
    baseline_scale = bounded_scale(baseline_raw_scale)
    optimized_scale = bounded_scale(fitted_raw_scale)
    baseline_current_profile, baseline_d13_profile, baseline_d33_profile = transport_profiles(
        baseline_raw_scale
    )
    optimized_current_profile, optimized_d13_profile, optimized_d33_profile = (
        transport_profiles(fitted_raw_scale)
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


def _surface_with_amplitude(
    surface: BoozerSurface,
    coefficient_index: int,
    amplitude: float | Array,
) -> BoozerSurface:
    return replace(surface, b_cos=surface.b_cos.at[coefficient_index].set(amplitude))


def _inverse_problem_response(
    surface: BoozerSurface,
    grid: GridSpec,
    nu_hat: Array,
    er_hat: float,
) -> Array:
    coeffs = solve_monoenergetic_scan(surface, grid, nu_hat, er_hat=jnp.full_like(nu_hat, er_hat))
    return coeffs["D11"]


def _er_profile(rho: Array, params: Array) -> Array:
    return params[0] * rho + params[1] * rho**3


def _evaluate_d33_profile(
    arrays: NeopaxMonoenergeticArrays,
    rho: Array,
    nu_value: Array,
    er_profile: Array,
) -> Array:
    import interpax

    log_nu = jnp.log10(jnp.maximum(nu_value, 1e-12))

    def per_radius(index, er_value):
        radius_scale = jnp.maximum(arrays.a_b * rho[index], 1e-8)
        er_log = jnp.log10(jnp.maximum(1e-8, jnp.abs(er_value / radius_scale)))
        interpolator = interpax.Interpolator2D(
            arrays.nu_log,
            arrays.Er_list[index],
            arrays.D33[index],
            extrap=True,
        )
        return interpolator(log_nu, er_log)

    return jax.vmap(per_radius)(jnp.arange(rho.size), er_profile)


def _evaluate_d11_profile(
    arrays: NeopaxMonoenergeticArrays,
    rho: Array,
    nu_value: Array,
    er_profile: Array,
) -> Array:
    import interpax

    log_nu = jnp.log10(jnp.maximum(nu_value, 1e-12))

    def per_radius(index, er_value):
        radius_scale = jnp.maximum(arrays.a_b * rho[index], 1e-8)
        er_log = jnp.log10(jnp.maximum(1e-8, jnp.abs(er_value / radius_scale)))
        interpolator = interpax.Interpolator2D(
            arrays.nu_log,
            arrays.Er_list[index],
            arrays.D11_log[index],
            extrap=True,
        )
        return 10.0 ** interpolator(log_nu, er_log)

    return jax.vmap(per_radius)(jnp.arange(rho.size), er_profile)


def _evaluate_d13_profile(
    arrays: NeopaxMonoenergeticArrays,
    rho: Array,
    nu_value: Array,
    er_profile: Array,
) -> Array:
    import interpax

    log_nu = jnp.log10(jnp.maximum(nu_value, 1e-12))

    def per_radius(index, er_value):
        radius_scale = jnp.maximum(arrays.a_b * rho[index], 1e-8)
        er_log = jnp.log10(jnp.maximum(1e-8, jnp.abs(er_value / radius_scale)))
        interpolator = interpax.Interpolator2D(
            arrays.nu_log,
            arrays.Er_list[index],
            arrays.D13[index],
            extrap=True,
        )
        return interpolator(log_nu, er_log)

    return jax.vmap(per_radius)(jnp.arange(rho.size), er_profile)


def _dominant_nonaxisymmetric_mode(surface: BoozerSurface | VmecSurface) -> tuple[int, int]:
    mask = jnp.logical_not(jnp.logical_and(surface.m == 0, surface.n == 0))
    masked_amplitude = jnp.where(mask, jnp.abs(surface.b_cos), -1.0)
    index = int(jnp.argmax(masked_amplitude))
    return int(surface.m[index]), int(surface.n[index])


def _mode_value_for_surface(
    surface: BoozerSurface | VmecSurface,
    harmonic_m: int,
    harmonic_n: int,
) -> Array:
    matches = jnp.logical_and(surface.m == harmonic_m, surface.n == harmonic_n)
    index = jnp.argmax(matches)
    return surface.b_cos[index]


def _scale_surface_mode(
    surface: BoozerSurface | VmecSurface,
    harmonic_m: int,
    harmonic_n: int,
    scale: Array,
) -> BoozerSurface | VmecSurface:
    matches = jnp.logical_and(surface.m == harmonic_m, surface.n == harmonic_n)
    index = jnp.argmax(matches)
    scaled = surface.b_cos.at[index].set(surface.b_cos[index] * scale)
    return replace(surface, b_cos=jnp.where(matches, scaled, surface.b_cos))


def _maybe_import_neopax():
    try:
        import NEOPAX

        return NEOPAX
    except ModuleNotFoundError:
        root = find_neopax_root()
        if root is None:
            raise
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import NEOPAX

        return NEOPAX
