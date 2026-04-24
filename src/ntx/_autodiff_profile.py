"""Profile-level autodiff and uncertainty workflow helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from ._autodiff_helpers import (
    er_profile,
    evaluate_d11_profile,
    evaluate_d33_profile,
)
from ._autodiff_types import NeopaxProfileAutodiffResult, NeopaxProfileUncertaintyResult
from .grids import GridSpec
from .neopax import (
    build_ntx_neopax_scan_from_surfaces,
    scan_to_neopax_arrays,
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
    target_er_profile = er_profile(rho_grid, target_params)
    target_d11_profile = evaluate_d11_profile(arrays, rho_grid, nu_value, target_er_profile)
    target_d33_profile = evaluate_d33_profile(arrays, rho_grid, nu_value, target_er_profile)

    def loss_fn(params):
        trial_er = er_profile(rho_grid, params)
        fitted_d11 = evaluate_d11_profile(arrays, rho_grid, nu_value, trial_er)
        fitted = evaluate_d33_profile(arrays, rho_grid, nu_value, trial_er)
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
    fitted_er_profile = er_profile(rho_grid, fitted_params)
    fitted_d11_profile = evaluate_d11_profile(arrays, rho_grid, nu_value, fitted_er_profile)
    fitted_d33_profile = evaluate_d33_profile(arrays, rho_grid, nu_value, fitted_er_profile)
    sensitivity_matrix = jax.jacrev(
        lambda params: evaluate_d33_profile(
            arrays,
            rho_grid,
            nu_value,
            er_profile(rho_grid, params),
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
        lambda sample: evaluate_d33_profile(
            arrays,
            rho_grid,
            nu_value,
            er_profile(rho_grid, sample),
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
