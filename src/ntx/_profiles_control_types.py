"""Profile-control dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array, tree_util

from ._profiles_ambipolar_types import AmbipolarProfileResult


@dataclass(frozen=True)
class ProfileControlSpec:
    """Linear control map applied to `A1` and `A3` for each species."""

    a1_response: Array
    a3_response: Array
    control_name: str = "control"


tree_util.register_dataclass(
    ProfileControlSpec,
    data_fields=("a1_response", "a3_response"),
    meta_fields=("control_name",),
)


@dataclass(frozen=True)
class ProfileControlOptimizationResult:
    """Optimization history for a scalar profile control."""

    control_history: Array
    objective_history: Array
    bootstrap_objective_history: Array
    residual_norm_history: Array
    best_control: Array
    best_profile: AmbipolarProfileResult


tree_util.register_dataclass(
    ProfileControlOptimizationResult,
    data_fields=(
        "control_history",
        "objective_history",
        "bootstrap_objective_history",
        "residual_norm_history",
        "best_control",
        "best_profile",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class ProfileBasisControlSpec:
    """Low-dimensional radial basis control applied to `A1` and `A3`."""

    basis: Array
    a1_response: Array
    a3_response: Array
    control_name: str = "basis control"


tree_util.register_dataclass(
    ProfileBasisControlSpec,
    data_fields=("basis", "a1_response", "a3_response"),
    meta_fields=("control_name",),
)


@dataclass(frozen=True)
class ProfileBasisOptimizationResult:
    """Optimization history for a vector profile-basis control."""

    control_history: Array
    objective_history: Array
    bootstrap_objective_history: Array
    residual_norm_history: Array
    best_control: Array
    best_profile: AmbipolarProfileResult


tree_util.register_dataclass(
    ProfileBasisOptimizationResult,
    data_fields=(
        "control_history",
        "objective_history",
        "bootstrap_objective_history",
        "residual_norm_history",
        "best_control",
        "best_profile",
    ),
    meta_fields=(),
)


__all__ = [
    "ProfileBasisControlSpec",
    "ProfileBasisOptimizationResult",
    "ProfileControlOptimizationResult",
    "ProfileControlSpec",
]
