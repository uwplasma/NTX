"""Autodiff-oriented NTX demonstrations and analysis helpers."""

from __future__ import annotations

import sys

from jax import Array

from ._autodiff import (
    BootstrapOptimizationResult,
    DerivativeAuditResult,
    InverseProblemResult,
    NeopaxProfileAutodiffResult,
    NeopaxProfileUncertaintyResult,
    RobustBootstrapOptimizationResult,
    example_derivative_audit,
    example_inverse_problem,
    example_neopax_profile_uncertainty,
)
from ._autodiff import (
    example_neopax_profile_autodiff as _example_neopax_profile_autodiff,
)
from ._autodiff_bootstrap import (
    example_bootstrap_current_optimization,
    example_bootstrap_current_robust_optimization,
)
from ._checkout_paths import find_neopax_root

__all__ = [
    "BootstrapOptimizationResult",
    "DerivativeAuditResult",
    "InverseProblemResult",
    "NeopaxProfileAutodiffResult",
    "NeopaxProfileUncertaintyResult",
    "RobustBootstrapOptimizationResult",
    "example_bootstrap_current_optimization",
    "example_bootstrap_current_robust_optimization",
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
    jacobian_chunk_size: int | str | None = None,
) -> NeopaxProfileAutodiffResult:
    """Infer an electric-field profile with optional chunked sensitivities."""

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
        jacobian_chunk_size=jacobian_chunk_size,
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
