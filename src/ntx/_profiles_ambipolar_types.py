"""Ambipolar-profile result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array, tree_util


@dataclass(frozen=True)
class AmbipolarProfileResult:
    """Solved electric-field profile and reduced monoenergetic responses."""

    rho: Array
    er_profile: Array
    ambipolar_residual: Array
    bootstrap_current_proxy: Array
    species_particle_flux: Array
    species_current_response: Array
    loss_history: Array

    @property
    def bootstrap_current_response(self) -> Array:
        """Reduced monoenergetic bootstrap-current response.

        The stored ``bootstrap_current_proxy`` field is retained for API
        compatibility with NTX 0.2.x examples. New code should prefer this
        property because the quantity is a reduced response built from
        monoenergetic coefficients, not a fitted bootstrap-current closure.
        """

        return self.bootstrap_current_proxy


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

    @property
    def bootstrap_current_response(self) -> Array:
        """Reduced monoenergetic bootstrap-current response family."""

        return self.bootstrap_current_proxy


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


__all__ = ["AmbipolarProfileFamilyResult", "AmbipolarProfileResult"]
