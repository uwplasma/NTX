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
    bootstrap_current_response: Array
    species_particle_flux: Array
    species_current_response: Array
    loss_history: Array

    @property
    def bootstrap_current_proxy(self) -> Array:
        """Compatibility alias for the reduced bootstrap-current response.

        New code should use ``bootstrap_current_response``. The alias is kept
        so NTX 0.2.x readers do not break, but it is no longer the registered
        runtime field because the quantity is a reduced monoenergetic response,
        not a fitted bootstrap-current closure.
        """

        return self.bootstrap_current_response


tree_util.register_dataclass(
    AmbipolarProfileResult,
    data_fields=(
        "rho",
        "er_profile",
        "ambipolar_residual",
        "bootstrap_current_response",
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
    bootstrap_current_response: Array
    loss_history: Array

    @property
    def bootstrap_current_proxy(self) -> Array:
        """Compatibility alias for the reduced bootstrap-current response family."""

        return self.bootstrap_current_response


tree_util.register_dataclass(
    AmbipolarProfileFamilyResult,
    data_fields=(
        "control",
        "er_profile",
        "ambipolar_residual",
        "bootstrap_current_response",
        "loss_history",
    ),
    meta_fields=(),
)


__all__ = ["AmbipolarProfileFamilyResult", "AmbipolarProfileResult"]
