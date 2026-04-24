"""Compatibility facade for profile-control helpers."""

from ._profiles_control_basis import (
    _basis_profile_modifier,
    apply_profile_basis_control,
    optimize_profile_basis_control,
)
from ._profiles_control_scalar import (
    apply_profile_control,
    optimize_profile_control,
)

__all__ = [
    "_basis_profile_modifier",
    "apply_profile_basis_control",
    "apply_profile_control",
    "optimize_profile_basis_control",
    "optimize_profile_control",
]
