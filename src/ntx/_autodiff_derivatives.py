"""Finite-difference derivative audit workflow helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ._autodiff_helpers import surface_with_amplitude
from ._autodiff_types import DerivativeAuditResult
from .geometry import example_surface
from .grids import GridSpec
from .solver import solve_monoenergetic_scan


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
        surface = surface_with_amplitude(base_surface, coefficient_index, amplitude)
        return solve_monoenergetic_scan(
            surface,
            grid,
            nu_hat,
            er_hat=jnp.full_like(nu_hat, er_reference),
        )["D11"].reshape(-1)

    def d33_curve(amplitude):
        surface = surface_with_amplitude(base_surface, coefficient_index, amplitude)
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

    fixed_surface = surface_with_amplitude(base_surface, coefficient_index, amplitude_value)

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
