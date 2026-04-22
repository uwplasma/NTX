"""Core monoenergetic solver dataclasses and result helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, tree_util

from .geometry import BoozerSurface, GeometryOnGrid, VmecSurface
from .grids import GridSpec


@dataclass(frozen=True)
class MonoenergeticCase:
    """Monoenergetic DKE parameters."""

    nu_hat: float | Array
    epsi_hat: float | Array | None = None
    er_hat: float | Array | None = None

    def resolved_epsi_hat(self, transport_psi_scale: float | Array | None) -> Array:
        if self.epsi_hat is not None and self.er_hat is not None:
            msg = "set only one of epsi_hat or er_hat"
            raise ValueError(msg)
        if self.epsi_hat is not None:
            return jnp.asarray(self.epsi_hat)
        if self.er_hat is not None:
            if transport_psi_scale is None:
                msg = "er_hat requires a surface with a transport normalization scale"
                raise ValueError(msg)
            return jnp.asarray(self.er_hat) / jnp.asarray(transport_psi_scale)
        if transport_psi_scale is not None:
            return jnp.zeros_like(jnp.asarray(transport_psi_scale))
        return jnp.asarray(0.0)


tree_util.register_dataclass(MonoenergeticCase)


@dataclass(frozen=True)
class TransportResult:
    D11: Array
    D31: Array
    D13: Array
    D33: Array
    D33_spitzer: Array
    f1_modes: Array
    f3_modes: Array
    residual_l2: Array
    onsager_residual: Array

    def as_dict(self) -> dict[str, float]:
        return {
            "D11": float(self.D11),
            "D31": float(self.D31),
            "D13": float(self.D13),
            "D33": float(self.D33),
            "D33_spitzer": float(self.D33_spitzer),
            "residual_l2": float(self.residual_l2),
            "onsager_residual": float(self.onsager_residual),
        }


tree_util.register_dataclass(TransportResult)


@dataclass(frozen=True)
class PreparedMonoenergeticSystem:
    """Cached geometry and derivative operators for repeated solves."""

    surface: BoozerSurface | VmecSurface
    grid: GridSpec
    geometry: GeometryOnGrid
    d_theta: Array
    d_zeta: Array


tree_util.register_dataclass(PreparedMonoenergeticSystem)


CompiledPreparedSolver = Callable[[MonoenergeticCase], TransportResult]


def transport_result_from_arrays(values: tuple[Array, ...]) -> TransportResult:
    return TransportResult(
        D11=values[0],
        D31=values[1],
        D13=values[2],
        D33=values[3],
        D33_spitzer=values[4],
        f1_modes=values[5],
        f3_modes=values[6],
        residual_l2=values[7],
        onsager_residual=values[8],
    )
