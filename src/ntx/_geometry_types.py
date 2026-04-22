"""Geometry dataclasses for Boozer and VMEC surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jax import Array, tree_util

from .grids import AngularGrid


@dataclass(frozen=True)
class BoozerSurface:
    """Single flux-surface representation in Boozer coordinates."""

    m: Array
    n: Array
    b_cos: Array
    nfp: int
    iota: float
    psi_p: float
    b_theta: float
    b_zeta: float
    chi_p: float | None = None
    b0: float | None = None
    b_sin: Array | None = None
    stellarator_symmetric: bool = True
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if len(self.m) != len(self.n) or len(self.m) != len(self.b_cos):
            msg = "m, n, and b_cos must have the same length"
            raise ValueError(msg)


tree_util.register_dataclass(
    BoozerSurface,
    data_fields=(
        "m",
        "n",
        "b_cos",
        "iota",
        "psi_p",
        "b_theta",
        "b_zeta",
        "chi_p",
        "b0",
        "b_sin",
    ),
    meta_fields=("nfp", "stellarator_symmetric", "source_path"),
)


@dataclass(frozen=True)
class VmecSurface:
    """Single flux-surface representation loaded from a VMEC `wout` file."""

    path: Path
    requested_psi_n: float
    psi_n: float
    nfp: int
    ns: int
    mpol: int
    ntor: int
    total_mode_count: int
    loaded_mode_count: int
    iota: float
    m: Array
    n: Array
    b_cos: Array
    jacobian_cos: Array
    b_sub_theta_cos: Array
    b_sub_zeta_cos: Array
    b_sup_theta_cos: Array
    b_sup_zeta_cos: Array
    b0: float
    psi_a_hat: float
    phi_edge: float
    r_n: float
    r_hat: float
    dpsi_hat_dr_hat: float
    dr_hat_dpsi_hat: float
    aminor_p: float | None = None
    psi_p: float | None = None
    transport_psi_scale: float = 1.0
    stellarator_symmetric: bool = True

    def __post_init__(self) -> None:
        size = len(self.m)
        for name in (
            "n",
            "b_cos",
            "jacobian_cos",
            "b_sub_theta_cos",
            "b_sub_zeta_cos",
            "b_sup_theta_cos",
            "b_sup_zeta_cos",
        ):
            if len(getattr(self, name)) != size:
                msg = f"{name} must have the same length as m"
                raise ValueError(msg)


tree_util.register_dataclass(
    VmecSurface,
    data_fields=(
        "requested_psi_n",
        "psi_n",
        "nfp",
        "iota",
        "m",
        "n",
        "b_cos",
        "jacobian_cos",
        "b_sub_theta_cos",
        "b_sub_zeta_cos",
        "b_sup_theta_cos",
        "b_sup_zeta_cos",
        "b0",
        "psi_a_hat",
        "phi_edge",
        "r_n",
        "r_hat",
        "dpsi_hat_dr_hat",
        "dr_hat_dpsi_hat",
        "aminor_p",
        "psi_p",
        "transport_psi_scale",
    ),
    meta_fields=(
        "path",
        "ns",
        "mpol",
        "ntor",
        "total_mode_count",
        "loaded_mode_count",
        "stellarator_symmetric",
    ),
)


@dataclass(frozen=True)
class GeometryOnGrid:
    surface_type: str
    surface_path: Path | None
    nfp: int
    iota: float
    psi_p: float | None
    transport_psi_scale: float
    coefficient_psi_scale: float
    grid: AngularGrid
    theta_2d: Array
    zeta_2d: Array
    b: Array
    d_b_dtheta: Array
    d_b_dzeta: Array
    jacobian: Array
    b_sub_theta: Array
    b_sub_zeta: Array
    b_sup_theta: Array
    b_sup_zeta: Array
    volume_prime: Array
    b2_mean: Array
    radial_drift_spatial: Array
    b0: Array


tree_util.register_dataclass(
    GeometryOnGrid,
    data_fields=(
        "nfp",
        "iota",
        "psi_p",
        "transport_psi_scale",
        "coefficient_psi_scale",
        "grid",
        "theta_2d",
        "zeta_2d",
        "b",
        "d_b_dtheta",
        "d_b_dzeta",
        "jacobian",
        "b_sub_theta",
        "b_sub_zeta",
        "b_sup_theta",
        "b_sup_zeta",
        "volume_prime",
        "b2_mean",
        "radial_drift_spatial",
        "b0",
    ),
    meta_fields=("surface_type", "surface_path"),
)
