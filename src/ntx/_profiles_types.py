"""Dataclasses for profile transport workflows."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array, tree_util


@dataclass(frozen=True)
class MonoenergeticSpeciesProfile:
    """One-species radial profile inputs for ambipolar and current proxies."""

    charge: float | Array
    nu_v: Array
    A1: Array
    A3: Array
    particle_weight: float | Array = 1.0
    current_weight: float | Array = 1.0
    name: str | None = None


tree_util.register_dataclass(
    MonoenergeticSpeciesProfile,
    data_fields=("charge", "nu_v", "A1", "A3", "particle_weight", "current_weight"),
    meta_fields=("name",),
)


@dataclass(frozen=True)
class PrimitiveSpeciesProfile:
    """Primitive radial inputs used to construct monoenergetic force profiles."""

    charge: float | Array
    nu_v: Array
    density: Array
    temperature: Array
    electrostatic_prefactor: float | Array = 1.0
    particle_weight: float | Array = 1.0
    current_weight: float | Array = 1.0
    name: str | None = None


tree_util.register_dataclass(
    PrimitiveSpeciesProfile,
    data_fields=(
        "charge",
        "nu_v",
        "density",
        "temperature",
        "electrostatic_prefactor",
        "particle_weight",
        "current_weight",
    ),
    meta_fields=("name",),
)


@dataclass(frozen=True)
class AmbipolarProfileResult:
    """Solved electric-field profile and derived monoenergetic proxy quantities."""

    rho: Array
    er_profile: Array
    ambipolar_residual: Array
    bootstrap_current_proxy: Array
    species_particle_flux: Array
    species_current_response: Array
    loss_history: Array


tree_util.register_dataclass(
    AmbipolarProfileResult,
    data_fields=(
        "rho",
        "er_profile",
        "ambipolar_residual",
        "bootstrap_current_proxy",
        "species_particle_flux",
        "species_current_response",
        "loss_history",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class AmbipolarProfileFamilyResult:
    """Stacked ambipolar-profile solutions across a control-parameter family."""

    control: Array
    er_profile: Array
    ambipolar_residual: Array
    bootstrap_current_proxy: Array
    loss_history: Array


tree_util.register_dataclass(
    AmbipolarProfileFamilyResult,
    data_fields=(
        "control",
        "er_profile",
        "ambipolar_residual",
        "bootstrap_current_proxy",
        "loss_history",
    ),
    meta_fields=(),
)


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


@dataclass(frozen=True)
class ProfileTransportClosureSpec:
    """Relaxation closure for iterating profile proxies toward transport targets."""

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
    bootstrap_current_proxy_history: Array
    transport_loss_history: Array
    species_a1_history: Array
    species_a3_history: Array
    best_profile: AmbipolarProfileResult


tree_util.register_dataclass(
    ProfileTransportIterationResult,
    data_fields=(
        "er_profile_history",
        "ambipolar_residual_history",
        "bootstrap_current_proxy_history",
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
    bootstrap_current_proxy_history: Array
    transport_loss_history: Array
    species_density_history: Array
    species_temperature_history: Array
    best_profile: AmbipolarProfileResult


tree_util.register_dataclass(
    PrimitiveProfileTransportIterationResult,
    data_fields=(
        "er_profile_history",
        "ambipolar_residual_history",
        "bootstrap_current_proxy_history",
        "transport_loss_history",
        "species_density_history",
        "species_temperature_history",
        "best_profile",
    ),
    meta_fields=(),
)

