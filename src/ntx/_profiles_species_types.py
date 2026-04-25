"""Species-profile dataclasses for profile transport workflows."""

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


__all__ = ["MonoenergeticSpeciesProfile", "PrimitiveSpeciesProfile"]
