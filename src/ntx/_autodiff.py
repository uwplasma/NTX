"""Differentiating the transport solve.

The derivative types, profile-space helpers, forward and inverse derivative
workflows. Bootstrap-current derivatives, which need their own robustness
treatment, live in _autodiff_bootstrap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, tree_util
from solvax import chunked_jacobian

from ._interp import interp2d_at
from .geometry import BoozerSurface, VmecSurface, example_surface
from .grids import GridSpec
from .neopax import (
    NeopaxMonoenergeticArrays,
    build_ntx_neopax_scan_from_surfaces,
    scan_to_neopax_arrays,
)
from .solver import solve_monoenergetic_scan

__all__ = [
    "_er_profile",
    "_evaluate_d11_profile",
    "_evaluate_d13_profile",
    "_evaluate_d33_profile",
    "_inverse_problem_response",
    "_surface_with_amplitude",
    "example_derivative_audit",
    "example_inverse_problem",
    "example_neopax_profile_autodiff",
    "example_neopax_profile_uncertainty",
]


# --- _autodiff_helpers: Private helpers shared by autodiff workflow examples. ---


def surface_with_amplitude(
    surface: BoozerSurface,
    coefficient_index: int,
    amplitude: float | Array,
) -> BoozerSurface:
    """Return `surface` with one Boozer coefficient replaced by `amplitude`.

    Functional update rather than mutation, so the result is differentiable
    with respect to `amplitude` and safe to call inside a traced loop.
    """
    return replace(surface, b_cos=surface.b_cos.at[coefficient_index].set(amplitude))


def inverse_problem_response(
    surface: BoozerSurface,
    grid: GridSpec,
    nu_hat: Array,
    er_hat: float,
) -> Array:
    """Forward map of the inverse problem: surface geometry to D11.

    The quantity an inverse solve tries to match. Wrapping the scan in a
    single-output function is what lets JAX differentiate the response with
    respect to a geometry coefficient.
    """
    coeffs = solve_monoenergetic_scan(surface, grid, nu_hat, er_hat=jnp.full_like(nu_hat, er_hat))
    return coeffs["D11"]


def er_profile(rho: Array, params: Array) -> Array:
    """Evaluate a radial electric field profile from its polynomial parameters.

    Uses odd powers of rho only, so the profile is odd about the magnetic
    axis and E_r(0) = 0, which is the physical boundary condition; a general
    polynomial would have to be constrained afterwards to achieve it.
    """
    parameters = jnp.asarray(params)
    # Odd powers only: E_r must vanish on axis, and building that into the
    # basis is more robust than constraining a general polynomial afterwards.
    powers = 2 * jnp.arange(parameters.size, dtype=jnp.asarray(rho).dtype) + 1
    return jnp.sum(parameters[:, None] * jnp.asarray(rho)[None, :] ** powers[:, None], axis=0)


def evaluate_d33_profile(
    arrays: NeopaxMonoenergeticArrays,
    rho: Array,
    nu_value: Array,
    er_profile_value: Array,
) -> Array:
    """Interpolate D33 across a radial profile at fixed collisionality.

    Monotone (pchip) interpolation: D33 is positive and saturating, so a
    monotone rule cannot overshoot its knee. Interpolated directly rather
    than in log space, unlike D11.
    """
    log_nu = jnp.log10(jnp.maximum(nu_value, 1e-12))

    def per_radius(index, er_value):
        radius_scale = jnp.maximum(arrays.a_b * rho[index], 1e-8)
        er_log = jnp.log10(jnp.maximum(1e-8, jnp.abs(er_value / radius_scale)))
        # saturating and positive: the monotone rule cannot overshoot its knee.
        value = interp2d_at(
            arrays.nu_log,
            arrays.Er_list[index],
            arrays.D33[index],
            log_nu,
            er_log,
            method="pchip",
        )
        return value

    return jax.vmap(per_radius)(jnp.arange(rho.size), er_profile_value)


def evaluate_d11_profile(
    arrays: NeopaxMonoenergeticArrays,
    rho: Array,
    nu_value: Array,
    er_profile_value: Array,
) -> Array:
    """Interpolate D11 across a radial profile at fixed collisionality.

    Interpolated in log10 and exponentiated back, because D11 spans several
    decades and its regime transitions appear as knees in log-log; a monotone
    (pchip) rule cannot overshoot them.
    """
    log_nu = jnp.log10(jnp.maximum(nu_value, 1e-12))

    def per_radius(index, er_value):
        radius_scale = jnp.maximum(arrays.a_b * rho[index], 1e-8)
        er_log = jnp.log10(jnp.maximum(1e-8, jnp.abs(er_value / radius_scale)))
        # log-log regime knees: the monotone rule cannot overshoot them.
        value = interp2d_at(
            arrays.nu_log,
            arrays.Er_list[index],
            arrays.D11_log[index],
            log_nu,
            er_log,
            method="pchip",
        )
        return 10.0**value

    return jax.vmap(per_radius)(jnp.arange(rho.size), er_profile_value)


def evaluate_d13_profile(
    arrays: NeopaxMonoenergeticArrays,
    rho: Array,
    nu_value: Array,
    er_profile_value: Array,
) -> Array:
    """Interpolate D13 across a radial profile at fixed collisionality.

    Parabolic rather than pchip, and the exception to the rule the other two
    coefficients follow: D13 changes sign and has a smooth extremum, which a
    monotone limiter would flatten into a plateau.
    """
    log_nu = jnp.log10(jnp.maximum(nu_value, 1e-12))

    def per_radius(index, er_value):
        radius_scale = jnp.maximum(arrays.a_b * rho[index], 1e-8)
        er_log = jnp.log10(jnp.maximum(1e-8, jnp.abs(er_value / radius_scale)))
        # sign-changing with a smooth extremum, which a monotone limiter would flatten.
        value = interp2d_at(
            arrays.nu_log,
            arrays.Er_list[index],
            arrays.D13[index],
            log_nu,
            er_log,
            method="parabolic",
        )
        return value

    return jax.vmap(per_radius)(jnp.arange(rho.size), er_profile_value)


def dominant_nonaxisymmetric_mode(surface: BoozerSurface | VmecSurface) -> tuple[int, int]:
    """Return the (m, n) of the largest-amplitude non-axisymmetric harmonic.

    The (0, 0) component is masked out because it carries the field strength
    rather than the shaping, and would otherwise always win.
    """
    mask = jnp.logical_not(jnp.logical_and(surface.m == 0, surface.n == 0))
    masked_amplitude = jnp.where(mask, jnp.abs(surface.b_cos), -1.0)
    index = int(jnp.argmax(masked_amplitude))
    return int(surface.m[index]), int(surface.n[index])


def mode_value_for_surface(
    surface: BoozerSurface | VmecSurface,
    harmonic_m: int,
    harmonic_n: int,
) -> Array:
    """Return the b_cos amplitude of one (m, n) harmonic.

    Selects by boolean match rather than by index arithmetic so the lookup
    survives an arbitrary harmonic ordering.
    """
    matches = jnp.logical_and(surface.m == harmonic_m, surface.n == harmonic_n)
    index = jnp.argmax(matches)
    return surface.b_cos[index]


def scale_surface_mode(
    surface: BoozerSurface | VmecSurface,
    harmonic_m: int,
    harmonic_n: int,
    scale: Array,
) -> BoozerSurface | VmecSurface:
    """Return `surface` with one harmonic's amplitude multiplied by `scale`.

    The knob a geometry optimization turns: it perturbs a single shaping
    harmonic while leaving every other harmonic, and the mode table itself,
    untouched.
    """
    matches = jnp.logical_and(surface.m == harmonic_m, surface.n == harmonic_n)
    index = jnp.argmax(matches)
    scaled = surface.b_cos.at[index].set(surface.b_cos[index] * scale)
    return replace(surface, b_cos=jnp.where(matches, scaled, surface.b_cos))


# --- _autodiff_types: Autodiff result dataclasses. ---


@dataclass(frozen=True)
class InverseProblemResult:
    """Per-iteration history of an inverse-problem solve.

    Keeps the whole trajectory rather than the final answer alone, so a run
    can be shown to have converged rather than merely stopped.
    """
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
    """Fitted profile, its target, and the optimization trajectory.

    Carries both the fitted and target profiles so the residual can be
    recomputed without rerunning the fit.
    """
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
class NeopaxProfileUncertaintyResult:
    """Fitted profile with its sensitivity matrix and parameter covariance.

    The covariance is what turns a fit into an error bar; it is propagated
    from the sensitivity matrix rather than estimated by resampling.
    """
    rho: Array
    fitted_er_profile: Array
    fitted_d33_profile: Array
    target_d33_profile: Array
    sensitivity_matrix: Array
    parameter_covariance: Array
    parameter_std: Array
    parameter_correlation: Array
    linearized_d33_std: Array
    monte_carlo_d33_mean: Array
    monte_carlo_d33_std: Array
    monte_carlo_d33_quantile_low: Array
    monte_carlo_d33_quantile_high: Array
    fisher_matrix: Array
    fisher_eigenvalues: Array
    hessian_vector_probe: Array
    gauss_newton_vector_probe: Array
    hessian_probe_relative_error: Array
    sample_count: Array


tree_util.register_dataclass(
    NeopaxProfileUncertaintyResult,
    data_fields=(
        "rho",
        "fitted_er_profile",
        "fitted_d33_profile",
        "target_d33_profile",
        "sensitivity_matrix",
        "parameter_covariance",
        "parameter_std",
        "parameter_correlation",
        "linearized_d33_std",
        "monte_carlo_d33_mean",
        "monte_carlo_d33_std",
        "monte_carlo_d33_quantile_low",
        "monte_carlo_d33_quantile_high",
        "fisher_matrix",
        "fisher_eigenvalues",
        "hessian_vector_probe",
        "gauss_newton_vector_probe",
        "hessian_probe_relative_error",
        "sample_count",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class BootstrapOptimizationResult:
    """Bootstrap-current optimization history plus the objective landscape.

    The landscape is swept alongside the descent path so a reported optimum
    can be checked against the surrounding objective rather than trusted from
    the final iterate.
    """
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
class RobustBootstrapOptimizationResult:
    """Robust bootstrap optimization, with both objective landscapes.

    Keeps the deterministic and robust landscapes side by side: the point of
    the robust objective is that its optimum sits somewhere the deterministic
    one does not, and that is only visible with both.
    """
    scale_history: Array
    gradient_history: Array
    objective_history: Array
    scale_grid: Array
    deterministic_objective_landscape: Array
    robust_objective_landscape: Array
    rho: Array
    baseline_scale: Array
    optimized_scale: Array
    baseline_current_profile: Array
    optimized_current_profile: Array
    optimized_current_mean: Array
    optimized_current_std: Array
    optimized_current_quantile_low: Array
    optimized_current_quantile_high: Array
    harmonic_m: Array
    harmonic_n: Array
    harmonic_reference_value: Array
    nu_value: Array
    uncertainty_sigma: Array
    risk_aversion: Array


tree_util.register_dataclass(
    RobustBootstrapOptimizationResult,
    data_fields=(
        "scale_history",
        "gradient_history",
        "objective_history",
        "scale_grid",
        "deterministic_objective_landscape",
        "robust_objective_landscape",
        "rho",
        "baseline_scale",
        "optimized_scale",
        "baseline_current_profile",
        "optimized_current_profile",
        "optimized_current_mean",
        "optimized_current_std",
        "optimized_current_quantile_low",
        "optimized_current_quantile_high",
        "harmonic_m",
        "harmonic_n",
        "harmonic_reference_value",
        "nu_value",
        "uncertainty_sigma",
        "risk_aversion",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class DerivativeAuditResult:
    """Autodiff derivatives beside finite-difference references.

    The audit exists because an autodiff gradient through an interpolated
    table can be silently wrong where the interpolant is not differentiable;
    holding both lets the comparison be asserted rather than assumed.
    """
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


# --- _autodiff_derivatives: Finite-difference derivative audit workflow helpers. ---


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
        jnp.logspace(-4.5, -1.5, 9) if nu_hat is None else jnp.asarray(nu_hat, dtype=grid.jax_dtype)
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
        lambda value: (
            (d11_at_er(value + fd_step_er) - d11_at_er(value - fd_step_er)) / (2.0 * fd_step_er)
        )
    )(er_hat_scan)
    autodiff_d33_der = jax.vmap(jax.grad(d33_at_er))(er_hat_scan)
    finite_difference_d33_der = jax.vmap(
        lambda value: (
            (d33_at_er(value + fd_step_er) - d33_at_er(value - fd_step_er)) / (2.0 * fd_step_er)
        )
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


# --- _autodiff_inverse: Synthetic inverse-problem workflow helpers. ---


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
        jnp.logspace(-4, -2, 8) if nu_hat is None else jnp.asarray(nu_hat, dtype=grid.jax_dtype)
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


# --- _autodiff_profile: Profile-level autodiff and uncertainty workflow helpers. ---

JacobianChunkSize = int | str | None


def _profile_jacobian(
    function: Callable[[Array], Array],
    parameters: Array,
    *,
    chunk_size: JacobianChunkSize,
) -> Array:
    """Evaluate a profile Jacobian, optionally bounding tangent-batch memory."""

    if chunk_size is None:
        return jax.jacrev(function)(parameters)
    return chunked_jacobian(
        function,
        mode="auto",
        chunk_size=chunk_size,
    )(parameters)


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
    target_params: Array | None = None,
    initial_params: Array | None = None,
    use_neopax_package: bool = False,
    maybe_import_neopax: Callable[[], Any] | None = None,
    jacobian_chunk_size: JacobianChunkSize = None,
) -> NeopaxProfileAutodiffResult:
    """Infer a low-dimensional electric-field profile on a NEOPAX-style scan.

    Set ``jacobian_chunk_size`` to a positive integer or ``"auto"`` to use
    SOLVAX's bounded-memory Jacobian assembly for a large profile basis. The
    default preserves native JAX reverse mode for small control sets.
    """

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
    target_params = (
        jnp.asarray([1.4e-3, -6.0e-4], dtype=rho_grid.dtype)
        if target_params is None
        else jnp.asarray(target_params, dtype=rho_grid.dtype)
    )
    initial_params = (
        jnp.asarray([5.0e-4, 2.0e-4], dtype=rho_grid.dtype)
        if initial_params is None
        else jnp.asarray(initial_params, dtype=rho_grid.dtype)
    )
    if target_params.shape != initial_params.shape:
        raise ValueError("target_params and initial_params must have the same shape")
    target_er_profile = er_profile(rho_grid, target_params)
    target_d11_profile = evaluate_d11_profile(arrays, rho_grid, nu_value, target_er_profile)
    target_d33_profile = evaluate_d33_profile(arrays, rho_grid, nu_value, target_er_profile)

    def loss_fn(params):
        trial_er = er_profile(rho_grid, params)
        fitted_d11 = evaluate_d11_profile(arrays, rho_grid, nu_value, trial_er)
        fitted = evaluate_d33_profile(arrays, rho_grid, nu_value, trial_er)
        d11_residual = jnp.log10(jnp.maximum(fitted_d11, 1e-30)) - jnp.log10(
            jnp.maximum(target_d11_profile, 1e-30)
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
    sensitivity_matrix = _profile_jacobian(
        lambda parameters: evaluate_d33_profile(
            arrays,
            rho_grid,
            nu_value,
            er_profile(rho_grid, parameters),
        ),
        fitted_params,
        chunk_size=jacobian_chunk_size,
    )
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
    target_params: Array | None = None,
    initial_params: Array | None = None,
    hessian_probe: Array | None = None,
    monte_carlo_samples: int = 64,
    random_seed: int = 0,
    jacobian_chunk_size: JacobianChunkSize = None,
) -> NeopaxProfileUncertaintyResult:
    """Compare linearized covariance propagation against Monte Carlo on a profile fit.

    ``jacobian_chunk_size`` controls both the profile sensitivity and local
    residual Jacobians. Use it for large profile bases after measuring the
    runtime-memory tradeoff on the target workload.
    """

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
        target_params=target_params,
        initial_params=initial_params,
        jacobian_chunk_size=jacobian_chunk_size,
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
    if parameter_std.shape != params.shape:
        raise ValueError("parameter_std must have the same shape as the fitted parameters")
    parameter_covariance = jnp.diag(parameter_std**2)
    indices = jnp.arange(parameter_std.size)
    off_diagonal_correlation = parameter_correlation ** jnp.abs(indices[:, None] - indices[None, :])
    off_diagonal_correlation = off_diagonal_correlation.at[
        jnp.diag_indices(parameter_std.size)
    ].set(1.0)
    parameter_covariance = (
        off_diagonal_correlation * parameter_std[:, None] * parameter_std[None, :]
    )
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

    d33_sensitivity_scale = jnp.maximum(jnp.abs(fit.fitted_d33_profile), 1.0e-12)

    def local_profile_residual(trial_params: Array) -> Array:
        trial_er_profile = er_profile(rho_grid, trial_params)
        trial_d11 = evaluate_d11_profile(arrays, rho_grid, nu_value, trial_er_profile)
        trial_d33 = evaluate_d33_profile(arrays, rho_grid, nu_value, trial_er_profile)
        d11_residual = jnp.log10(jnp.maximum(trial_d11, 1e-30)) - jnp.log10(
            jnp.maximum(fit.fitted_d11_profile, 1e-30)
        )
        d33_residual = (trial_d33 - fit.fitted_d33_profile) / d33_sensitivity_scale
        return jnp.concatenate((d11_residual, d33_residual))

    residual_jacobian = _profile_jacobian(
        local_profile_residual,
        params,
        chunk_size=jacobian_chunk_size,
    )
    fisher_matrix = residual_jacobian.T @ residual_jacobian / residual_jacobian.shape[0]
    fisher_eigenvalues = jnp.linalg.eigvalsh(fisher_matrix)
    hessian_probe = (
        jnp.linspace(1.0, -0.5, params.size, dtype=rho_grid.dtype)
        if hessian_probe is None
        else jnp.asarray(hessian_probe, dtype=rho_grid.dtype)
    )
    if hessian_probe.shape != params.shape:
        raise ValueError("hessian_probe must have the same shape as the fitted parameters")

    def local_profile_loss(trial_params: Array) -> Array:
        residual = local_profile_residual(trial_params)
        return 0.5 * jnp.mean(residual**2)

    _, hessian_vector_probe = jax.jvp(
        jax.grad(local_profile_loss),
        (params,),
        (hessian_probe,),
    )
    gauss_newton_vector_probe = fisher_matrix @ hessian_probe
    hessian_probe_relative_error = jnp.linalg.norm(
        hessian_vector_probe - gauss_newton_vector_probe
    ) / jnp.maximum(jnp.linalg.norm(gauss_newton_vector_probe), 1.0e-30)
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
        fisher_matrix=fisher_matrix,
        fisher_eigenvalues=fisher_eigenvalues,
        hessian_vector_probe=hessian_vector_probe,
        gauss_newton_vector_probe=gauss_newton_vector_probe,
        hessian_probe_relative_error=hessian_probe_relative_error,
        sample_count=jnp.asarray(monte_carlo_samples),
    )


# Underscore aliases kept for callers that imported these from the modules
# fused into this one; the public names above are the definitions.
_er_profile = er_profile
_evaluate_d11_profile = evaluate_d11_profile
_evaluate_d13_profile = evaluate_d13_profile
_evaluate_d33_profile = evaluate_d33_profile
_inverse_problem_response = inverse_problem_response
_surface_with_amplitude = surface_with_amplitude
