"""Autodiff-oriented NTX demonstrations and analysis helpers."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jax import Array, tree_util

from ._checkout_paths import find_neopax_root
from .geometry import BoozerSurface, example_surface
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
        "target_d33_profile",
        "fitted_d33_profile",
        "sensitivity_matrix",
        "nu_value",
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
    target_d33_profile = _evaluate_d33_profile(arrays, rho_grid, nu_value, target_er_profile)

    def loss_fn(params):
        fitted = _evaluate_d33_profile(
            arrays,
            rho_grid,
            nu_value,
            _er_profile(rho_grid, params),
        )
        return jnp.mean((fitted - target_d33_profile) ** 2)

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
        target_d33_profile=target_d33_profile,
        fitted_d33_profile=fitted_d33_profile,
        sensitivity_matrix=sensitivity_matrix,
        nu_value=jnp.asarray(nu_value),
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
