"""Bootstrap-current autodiff optimization compatibility exports."""

from __future__ import annotations

from ._autodiff_bootstrap_deterministic import (
    example_bootstrap_current_optimization,
)
from ._autodiff_bootstrap_robust import (
    example_bootstrap_current_robust_optimization,
)

__all__ = [
    "example_bootstrap_current_optimization",
    "example_bootstrap_current_robust_optimization",
]
