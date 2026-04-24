"""Autodiff workflow compatibility exports."""

from __future__ import annotations

from ._autodiff_bootstrap import (  # noqa: F401
    example_bootstrap_current_optimization,
    example_bootstrap_current_robust_optimization,
)
from ._autodiff_derivatives import example_derivative_audit
from ._autodiff_helpers import er_profile as _er_profile
from ._autodiff_helpers import evaluate_d11_profile as _evaluate_d11_profile
from ._autodiff_helpers import evaluate_d13_profile as _evaluate_d13_profile
from ._autodiff_helpers import evaluate_d33_profile as _evaluate_d33_profile
from ._autodiff_helpers import inverse_problem_response as _inverse_problem_response
from ._autodiff_helpers import surface_with_amplitude as _surface_with_amplitude
from ._autodiff_inverse import example_inverse_problem
from ._autodiff_profile import (
    example_neopax_profile_autodiff,
    example_neopax_profile_uncertainty,
)

__all__ = [
    "_er_profile",
    "_evaluate_d11_profile",
    "_evaluate_d13_profile",
    "_evaluate_d33_profile",
    "_inverse_problem_response",
    "_surface_with_amplitude",
    "example_bootstrap_current_optimization",
    "example_bootstrap_current_robust_optimization",
    "example_derivative_audit",
    "example_inverse_problem",
    "example_neopax_profile_autodiff",
    "example_neopax_profile_uncertainty",
]
