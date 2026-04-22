"""Autodiff-oriented NTX demonstrations and analysis helpers."""

from __future__ import annotations

import sys

from jax import Array

from ._autodiff_types import (
    BootstrapOptimizationResult,
    DerivativeAuditResult,
    InverseProblemResult,
    NeopaxProfileAutodiffResult,
    NeopaxProfileUncertaintyResult,
)
from ._autodiff_workflows import (
    example_bootstrap_current_optimization,
    example_derivative_audit,
    example_inverse_problem,
    example_neopax_profile_uncertainty,
)
from ._autodiff_workflows import (
    example_neopax_profile_autodiff as _example_neopax_profile_autodiff,
)
from ._checkout_paths import find_neopax_root

__all__ = [
    "BootstrapOptimizationResult",
    "DerivativeAuditResult",
    "InverseProblemResult",
    "NeopaxProfileAutodiffResult",
    "NeopaxProfileUncertaintyResult",
    "example_bootstrap_current_optimization",
    "example_derivative_audit",
    "example_inverse_problem",
    "example_neopax_profile_autodiff",
    "example_neopax_profile_uncertainty",
    "_maybe_import_neopax",
]


def example_neopax_profile_autodiff(
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
    learning_rate: float = 0.25,
    steps: int = 32,
    use_neopax_package: bool = False,
) -> NeopaxProfileAutodiffResult:
    """Infer a low-dimensional electric-field profile on a NEOPAX-style scan."""

    return _example_neopax_profile_autodiff(
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
        use_neopax_package=use_neopax_package,
        maybe_import_neopax=_maybe_import_neopax,
    )

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
