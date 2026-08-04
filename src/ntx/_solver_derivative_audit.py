"""Prepared primal/adjoint residual and coefficient-derivative audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array, tree_util

from ._solver import (
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    _coefficient_mode_pullback,
    _full_mode_relative_residual_norm,
    _full_mode_transpose_relative_residual_norm,
    _operator_context,
    _prepared_implicit_vjp_primal,
    _solve_factorized_adjoint,
    solve_prepared_coefficient_vector,
    solve_prepared_coefficient_vector_vjp,
)
from .operators import source_modes

CoefficientParameter = Literal["nu_hat", "epsi_hat", "er_hat"]
COEFFICIENT_NAMES = ("D11", "D31", "D13", "D33", "D33_spitzer")


@dataclass(frozen=True)
class PreparedDerivativeAuditResult:
    """Residual and independent-gradient evidence for one scalar derivative."""

    primal_relative_residual: Array
    transpose_relative_residual: Array
    direct_reverse_gradient: Array
    prepared_adjoint_gradient: Array
    forward_gradient: Array
    finite_difference_gradient: Array
    prepared_adjoint_relative_error: Array
    forward_relative_error: Array
    finite_difference_relative_error: Array
    finite_difference_step: Array
    valid: Array
    coefficient: str
    parameter: str
    residual_tolerance: float
    gradient_tolerance: float

    def as_dict(self) -> dict[str, float | bool | str]:
        """Materialize the audit as JSON-friendly scalar metadata."""

        return {
            "coefficient": self.coefficient,
            "parameter": self.parameter,
            "primal_relative_residual": float(self.primal_relative_residual),
            "transpose_relative_residual": float(self.transpose_relative_residual),
            "direct_reverse_gradient": float(self.direct_reverse_gradient),
            "prepared_adjoint_gradient": float(self.prepared_adjoint_gradient),
            "forward_gradient": float(self.forward_gradient),
            "finite_difference_gradient": float(self.finite_difference_gradient),
            "prepared_adjoint_relative_error": float(self.prepared_adjoint_relative_error),
            "forward_relative_error": float(self.forward_relative_error),
            "finite_difference_relative_error": float(self.finite_difference_relative_error),
            "finite_difference_step": float(self.finite_difference_step),
            "valid": bool(self.valid),
            "residual_tolerance": self.residual_tolerance,
            "gradient_tolerance": self.gradient_tolerance,
        }


tree_util.register_dataclass(
    PreparedDerivativeAuditResult,
    data_fields=(
        "primal_relative_residual",
        "transpose_relative_residual",
        "direct_reverse_gradient",
        "prepared_adjoint_gradient",
        "forward_gradient",
        "finite_difference_gradient",
        "prepared_adjoint_relative_error",
        "forward_relative_error",
        "finite_difference_relative_error",
        "finite_difference_step",
        "valid",
    ),
    meta_fields=(
        "coefficient",
        "parameter",
        "residual_tolerance",
        "gradient_tolerance",
    ),
)


def audit_prepared_coefficient_derivative(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    *,
    coefficient: str = "D11",
    parameter: CoefficientParameter = "er_hat",
    finite_difference_step: float | Array | None = None,
    residual_tolerance: float = 1.0e-10,
    gradient_tolerance: float = 1.0e-4,
) -> PreparedDerivativeAuditResult:
    """Audit one prepared coefficient derivative before accepting it as valid.

    The audit compares direct reverse mode, forward mode, the factor-reusing
    prepared adjoint, and a centered finite difference. It independently
    evaluates the full primal and algebraic-transpose block equations.
    """

    if coefficient not in COEFFICIENT_NAMES:
        names = ", ".join(COEFFICIENT_NAMES)
        raise ValueError(f"coefficient must be one of: {names}")
    if parameter not in ("nu_hat", "epsi_hat", "er_hat"):
        raise ValueError("parameter must be 'nu_hat', 'epsi_hat', or 'er_hat'")
    if residual_tolerance <= 0.0 or gradient_tolerance <= 0.0:
        raise ValueError("audit tolerances must be positive")

    coefficient_index = COEFFICIENT_NAMES.index(coefficient)
    value = _case_parameter_value(prepared, case, parameter)
    step = _finite_difference_step(value, finite_difference_step)

    def direct_objective(parameter_value):
        varied = _case_with_parameter(case, parameter, parameter_value)
        return solve_prepared_coefficient_vector(prepared, varied)[coefficient_index]

    def adjoint_objective(parameter_value):
        varied = _case_with_parameter(case, parameter, parameter_value)
        return solve_prepared_coefficient_vector_vjp(prepared, varied)[coefficient_index]

    direct_reverse = jax.grad(direct_objective)(value)
    prepared_adjoint = jax.grad(adjoint_objective)(value)
    forward = jax.jacfwd(direct_objective)(value)
    finite_difference = (direct_objective(value + step) - direct_objective(value - step)) / (
        2.0 * step
    )

    primal_residual, transpose_residual = _prepared_primal_transpose_residuals(
        prepared,
        case,
        coefficient_index,
    )
    adjoint_error = _relative_scalar_error(prepared_adjoint, direct_reverse)
    forward_error = _relative_scalar_error(forward, direct_reverse)
    finite_difference_error = _relative_scalar_error(finite_difference, direct_reverse)
    values = jnp.asarray(
        [
            direct_reverse,
            prepared_adjoint,
            forward,
            finite_difference,
            primal_residual,
            transpose_residual,
        ]
    )
    valid = (
        jnp.all(jnp.isfinite(values))
        & (primal_residual <= residual_tolerance)
        & (transpose_residual <= residual_tolerance)
        & (adjoint_error <= gradient_tolerance)
        & (forward_error <= gradient_tolerance)
        & (finite_difference_error <= gradient_tolerance)
    )
    return PreparedDerivativeAuditResult(
        primal_relative_residual=primal_residual,
        transpose_relative_residual=transpose_residual,
        direct_reverse_gradient=direct_reverse,
        prepared_adjoint_gradient=prepared_adjoint,
        forward_gradient=forward,
        finite_difference_gradient=finite_difference,
        prepared_adjoint_relative_error=adjoint_error,
        forward_relative_error=forward_error,
        finite_difference_relative_error=finite_difference_error,
        finite_difference_step=step,
        valid=valid,
        coefficient=coefficient,
        parameter=parameter,
        residual_tolerance=residual_tolerance,
        gradient_tolerance=gradient_tolerance,
    )


def _prepared_primal_transpose_residuals(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_index: int,
) -> tuple[Array, Array]:
    """Residual norms of the primal and transposed solves.

    An adjoint gradient is only as good as the transposed solve behind it, so
    both residuals are reported: a converged primal with a poorly solved
    transpose produces a plausible and wrong derivative.
    """
    epsi_hat = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        epsi_hat,
    )
    source1, source3 = source_modes(ctx, prepared.grid.n_xi)
    coefficients, f1, f3, lu, piv, lower, upper = _prepared_implicit_vjp_primal(
        prepared,
        case.nu_hat,
        epsi_hat,
    )
    coefficient_bar = jnp.zeros_like(coefficients).at[coefficient_index].set(1.0)
    f1_bar_low, f3_bar_low, _ = _coefficient_mode_pullback(
        prepared.geometry,
        f1[:3],
        f3[:3],
        ctx.nu_hat,
        coefficient_bar,
    )
    source_bar1 = jnp.zeros_like(f1).at[:3].set(f1_bar_low)
    source_bar3 = jnp.zeros_like(f3).at[:3].set(f3_bar_low)
    lambda1 = _solve_factorized_adjoint(lu, piv, lower, upper, source_bar1)
    lambda3 = _solve_factorized_adjoint(lu, piv, lower, upper, source_bar3)

    primal = jnp.maximum(
        _full_mode_relative_residual_norm(
            ctx,
            prepared.grid.n_xi,
            prepared.d_theta,
            prepared.d_zeta,
            source1,
            f1,
        ),
        _full_mode_relative_residual_norm(
            ctx,
            prepared.grid.n_xi,
            prepared.d_theta,
            prepared.d_zeta,
            source3,
            f3,
        ),
    )
    transpose = jnp.maximum(
        _full_mode_transpose_relative_residual_norm(
            ctx,
            prepared.grid.n_xi,
            prepared.d_theta,
            prepared.d_zeta,
            source_bar1,
            lambda1,
        ),
        _full_mode_transpose_relative_residual_norm(
            ctx,
            prepared.grid.n_xi,
            prepared.d_theta,
            prepared.d_zeta,
            source_bar3,
            lambda3,
        ),
    )
    return primal, transpose


def _case_parameter_value(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    parameter: CoefficientParameter,
) -> Array:
    """Current value of the parameter being differentiated."""
    if parameter == "nu_hat":
        return jnp.asarray(case.nu_hat)
    if parameter == "epsi_hat":
        if case.er_hat is not None:
            raise ValueError("epsi_hat audit requires a case without er_hat")
        return jnp.asarray(0.0 if case.epsi_hat is None else case.epsi_hat)
    if case.epsi_hat is not None:
        raise ValueError("er_hat audit requires a case without epsi_hat")
    if prepared.geometry.transport_psi_scale is None:
        raise ValueError("er_hat audit requires a transport normalization scale")
    return jnp.asarray(0.0 if case.er_hat is None else case.er_hat)


def _case_with_parameter(
    case: MonoenergeticCase,
    parameter: CoefficientParameter,
    value: Array,
) -> MonoenergeticCase:
    """Copy a case with one parameter replaced.

    Used to step a parameter for the finite-difference reference without
    mutating the case under audit.
    """
    if parameter == "nu_hat":
        return MonoenergeticCase(value, epsi_hat=case.epsi_hat, er_hat=case.er_hat)
    if parameter == "epsi_hat":
        return MonoenergeticCase(case.nu_hat, epsi_hat=value)
    return MonoenergeticCase(case.nu_hat, er_hat=value)


def _finite_difference_step(value: Array, requested: float | Array | None) -> Array:
    """Choose a finite-difference step, defaulting from the value's magnitude.

    A step scaled to the value keeps the difference in the range where
    truncation and round-off are balanced; a fixed absolute step would be far
    too large or too small depending on units.
    """
    if requested is not None:
        step = jnp.asarray(requested, dtype=value.dtype)
        if float(step) <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        return step
    epsilon = jnp.finfo(value.dtype).eps
    return epsilon ** (1.0 / 3.0) * jnp.maximum(jnp.abs(value), 1.0)


def _relative_scalar_error(candidate: Array, reference: Array) -> Array:
    """Relative error between two scalars, floored so zeros compare finitely.

    Scaled by the larger magnitude rather than the reference, so the comparison
    stays symmetric and does not blow up when the reference is near zero.
    """
    scale = jnp.maximum(jnp.maximum(jnp.abs(candidate), jnp.abs(reference)), 1.0e-30)
    return jnp.abs(candidate - reference) / scale


__all__ = [
    "PreparedDerivativeAuditResult",
    "audit_prepared_coefficient_derivative",
]
