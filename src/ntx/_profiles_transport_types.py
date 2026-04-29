"""Profile-transport closure and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array, tree_util

from ._profiles_ambipolar_types import AmbipolarProfileResult


@dataclass(frozen=True)
class ProfileTransportClosureSpec:
    """Relaxation closure for iterating reduced responses toward transport targets."""

    particle_relaxation: Array
    current_relaxation: Array
    particle_target: float | Array = 0.0
    current_target: float | Array = 0.0
    particle_source: float | Array = 0.0
    current_source: float | Array = 0.0
    normalization_floor: float | Array = 1.0
    max_normalized_update: float | Array = 0.35
    density_relaxation: float | Array = 0.0
    temperature_relaxation: float | Array = 0.0
    density_target: float | Array = 0.0
    temperature_target: float | Array = 0.0
    density_source: float | Array = 0.0
    temperature_source: float | Array = 0.0
    primitive_normalization_floor: float | Array = 1.0
    max_primitive_normalized_update: float | Array = 0.20
    radial_smoothing_strength: float | Array = 0.0
    closure_name: str = "transport loop"


tree_util.register_dataclass(
    ProfileTransportClosureSpec,
    data_fields=(
        "particle_relaxation",
        "current_relaxation",
        "particle_target",
        "current_target",
        "particle_source",
        "current_source",
        "normalization_floor",
        "max_normalized_update",
        "density_relaxation",
        "temperature_relaxation",
        "density_target",
        "temperature_target",
        "density_source",
        "temperature_source",
        "primitive_normalization_floor",
        "max_primitive_normalized_update",
        "radial_smoothing_strength",
    ),
    meta_fields=("closure_name",),
)


@dataclass(frozen=True)
class ProfileTransportIterationResult:
    """History of a simple self-consistent profile transport relaxation loop."""

    er_profile_history: Array
    ambipolar_residual_history: Array
    bootstrap_current_response_history: Array
    transport_loss_history: Array
    species_a1_history: Array
    species_a3_history: Array
    best_profile: AmbipolarProfileResult

    @property
    def bootstrap_current_proxy_history(self) -> Array:
        """Compatibility alias for the reduced bootstrap-current response history."""

        return self.bootstrap_current_response_history


tree_util.register_dataclass(
    ProfileTransportIterationResult,
    data_fields=(
        "er_profile_history",
        "ambipolar_residual_history",
        "bootstrap_current_response_history",
        "transport_loss_history",
        "species_a1_history",
        "species_a3_history",
        "best_profile",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class PrimitiveProfileTransportIterationResult:
    """History of a primitive-profile transport relaxation workflow."""

    er_profile_history: Array
    ambipolar_residual_history: Array
    bootstrap_current_response_history: Array
    transport_loss_history: Array
    species_density_history: Array
    species_temperature_history: Array
    best_profile: AmbipolarProfileResult

    @property
    def bootstrap_current_proxy_history(self) -> Array:
        """Compatibility alias for the reduced bootstrap-current response history."""

        return self.bootstrap_current_response_history


tree_util.register_dataclass(
    PrimitiveProfileTransportIterationResult,
    data_fields=(
        "er_profile_history",
        "ambipolar_residual_history",
        "bootstrap_current_response_history",
        "transport_loss_history",
        "species_density_history",
        "species_temperature_history",
        "best_profile",
    ),
    meta_fields=(),
)


__all__ = [
    "PrimitiveProfileTransportIterationResult",
    "ProfileTransportClosureSpec",
    "ProfileTransportIterationResult",
]
