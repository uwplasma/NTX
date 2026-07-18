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
    """Monoenergetic coefficients, retained modes, and solver diagnostics.

    ``residual_l2`` is retained for API compatibility. It is the RMS residual
    of the tail-eliminated Schur system, not the residual of every original
    Legendre block row. Use ``schur_residual_l2`` for explicit new code and
    :func:`ntx.audit_prepared_residuals` when a full-system residual is needed.
    """

    D11: Array
    D31: Array
    D13: Array
    D33: Array
    D33_spitzer: Array
    f1_modes: Array
    f3_modes: Array
    residual_l2: Array
    onsager_residual: Array

    @property
    def schur_residual_l2(self) -> Array:
        """RMS algebraic residual of the complete tail-eliminated system."""

        return self.residual_l2

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
class ResidualAuditResult:
    """Opt-in comparison of the low-memory and full Legendre solves."""

    tail_eliminated_l2: Array
    full_system_l2: Array
    retained_mode_max_abs_error: Array
    n_modes: int

    @property
    def schur_residual_l2(self) -> Array:
        """RMS residual of the tail-eliminated Schur system."""

        return self.tail_eliminated_l2

    @property
    def full_system_residual_l2(self) -> Array:
        """RMS residual obtained by applying every original Legendre row."""

        return self.full_system_l2


tree_util.register_dataclass(
    ResidualAuditResult,
    data_fields=(
        "tail_eliminated_l2",
        "full_system_l2",
        "retained_mode_max_abs_error",
    ),
    meta_fields=("n_modes",),
)


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
