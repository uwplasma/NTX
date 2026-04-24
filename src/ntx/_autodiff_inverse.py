"""Synthetic inverse-problem workflow helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ._autodiff_helpers import inverse_problem_response, surface_with_amplitude
from ._autodiff_types import InverseProblemResult
from .geometry import example_surface
from .grids import GridSpec


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
    target_surface = surface_with_amplitude(base_surface, coefficient_index, target_amplitude)
    target_response = inverse_problem_response(target_surface, grid, nu_hat, er_hat)

    def loss_fn(amplitude):
        surface = surface_with_amplitude(base_surface, coefficient_index, amplitude)
        response = inverse_problem_response(surface, grid, nu_hat, er_hat)
        return jnp.mean(
            (
                jnp.log10(jnp.maximum(response, 1e-12))
                - jnp.log10(jnp.maximum(target_response, 1e-12))
            )
            ** 2
        )

    initial_response = inverse_problem_response(
        surface_with_amplitude(base_surface, coefficient_index, initial_amplitude),
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
    fitted_surface = surface_with_amplitude(base_surface, coefficient_index, inferred_amplitude)
    fitted_response = inverse_problem_response(fitted_surface, grid, nu_hat, er_hat)
    sensitivity = jax.grad(
        lambda amplitude: jnp.sum(
            inverse_problem_response(
                surface_with_amplitude(base_surface, coefficient_index, amplitude),
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
