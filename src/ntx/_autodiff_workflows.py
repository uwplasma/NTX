"""Autodiff workflow helpers and example routines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from ._autodiff_bootstrap import (  # noqa: F401
    example_bootstrap_current_optimization,
    example_bootstrap_current_robust_optimization,
)
from ._autodiff_helpers import er_profile as _er_profile
from ._autodiff_helpers import evaluate_d11_profile as _evaluate_d11_profile
from ._autodiff_helpers import evaluate_d13_profile as _evaluate_d13_profile  # noqa: F401
from ._autodiff_helpers import evaluate_d33_profile as _evaluate_d33_profile
from ._autodiff_helpers import inverse_problem_response as _inverse_problem_response
from ._autodiff_helpers import surface_with_amplitude as _surface_with_amplitude
from ._autodiff_types import (
    DerivativeAuditResult,
    InverseProblemResult,
    NeopaxProfileAutodiffResult,
    NeopaxProfileUncertaintyResult,
)
from .geometry import example_surface
from .grids import GridSpec
from .neopax import (
    build_ntx_neopax_scan_from_surfaces,
    scan_to_neopax_arrays,
)
from .solver import solve_monoenergetic_scan


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
    target_surface = _surface_with_amplitude(base_surface, coefficient_index, target_amplitude)
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
    maybe_import_neopax: Callable[[], Any] | None = None,
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
        fitted = _evaluate_d33_profile(arrays, rho_grid, nu_value, trial_er)
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

    fitted_params, history = jax.lax.scan(step, initial_params, xs=None, length=steps)
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
        if maybe_import_neopax is None:
            raise RuntimeError("use_neopax_package=True requires a NEOPAX importer callback")
        _ = maybe_import_neopax()
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
    """Compare direct JAX gradients against finite differences."""

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


def example_neopax_profile_uncertainty(
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
    parameter_std: Array | None = None,
    parameter_correlation: float = -0.35,
    monte_carlo_samples: int = 64,
    random_seed: int = 0,
) -> NeopaxProfileUncertaintyResult:
    """Compare linearized covariance propagation against Monte Carlo on a profile fit."""

    fit = example_neopax_profile_autodiff(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=grid,
        a_b=a_b,
        nu_index=nu_index,
        learning_rate=learning_rate,
        steps=steps,
    )
    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=jnp.asarray(rho),
        nu_v=jnp.asarray(nu_v),
        Es=jnp.asarray(Es),
        Er=jnp.asarray(Er),
        drds=jnp.asarray(drds),
        grid=grid,
        source_name="autodiff_profile_uncertainty",
    )
    arrays = scan_to_neopax_arrays(scan, a_b=a_b)
    rho_grid = jnp.asarray(arrays.rho)
    nu_value = 10.0 ** arrays.nu_log[nu_index]

    params = fit.parameter_history[-1]
    parameter_std = (
        jnp.asarray([5.0e-5, 2.0e-5], dtype=rho_grid.dtype)
        if parameter_std is None
        else jnp.asarray(parameter_std, dtype=rho_grid.dtype)
    )
    parameter_covariance = jnp.diag(parameter_std**2)
    correlation_entry = parameter_correlation * parameter_std[0] * parameter_std[1]
    parameter_covariance = parameter_covariance.at[0, 1].set(correlation_entry)
    parameter_covariance = parameter_covariance.at[1, 0].set(correlation_entry)
    parameter_correlation_matrix = parameter_covariance / (
        parameter_std[:, None] * parameter_std[None, :]
    )
    linearized_profile_covariance = (
        fit.sensitivity_matrix @ parameter_covariance @ fit.sensitivity_matrix.T
    )
    linearized_d33_std = jnp.sqrt(jnp.maximum(jnp.diag(linearized_profile_covariance), 0.0))

    key = jax.random.PRNGKey(random_seed)
    standard_draws = jax.random.normal(
        key,
        shape=(monte_carlo_samples, parameter_covariance.shape[0]),
        dtype=rho_grid.dtype,
    )
    parameter_cholesky = jnp.linalg.cholesky(
        parameter_covariance
        + 1.0e-20 * jnp.eye(parameter_covariance.shape[0], dtype=rho_grid.dtype)
    )
    parameter_samples = params + standard_draws @ parameter_cholesky.T
    monte_carlo_d33 = jax.vmap(
        lambda sample: _evaluate_d33_profile(
            arrays,
            rho_grid,
            nu_value,
            _er_profile(rho_grid, sample),
        )
    )(parameter_samples)
    monte_carlo_d33_mean = jnp.mean(monte_carlo_d33, axis=0)
    monte_carlo_d33_std = jnp.std(monte_carlo_d33, axis=0, ddof=1)
    monte_carlo_d33_quantile_low = jnp.quantile(monte_carlo_d33, 0.16, axis=0)
    monte_carlo_d33_quantile_high = jnp.quantile(monte_carlo_d33, 0.84, axis=0)
    return NeopaxProfileUncertaintyResult(
        rho=rho_grid,
        fitted_er_profile=fit.fitted_er_profile,
        fitted_d33_profile=fit.fitted_d33_profile,
        target_d33_profile=fit.target_d33_profile,
        sensitivity_matrix=fit.sensitivity_matrix,
        parameter_covariance=parameter_covariance,
        parameter_std=parameter_std,
        parameter_correlation=parameter_correlation_matrix,
        linearized_d33_std=linearized_d33_std,
        monte_carlo_d33_mean=monte_carlo_d33_mean,
        monte_carlo_d33_std=monte_carlo_d33_std,
        monte_carlo_d33_quantile_low=monte_carlo_d33_quantile_low,
        monte_carlo_d33_quantile_high=monte_carlo_d33_quantile_high,
        sample_count=jnp.asarray(monte_carlo_samples),
    )
