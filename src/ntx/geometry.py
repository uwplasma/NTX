"""Boozer-surface geometry for the monoenergetic DKE."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from .grids import AngularGrid, flux_surface_average, periodic_grid


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
    b0: float | None = None
    b_sin: Array | None = None
    stellarator_symmetric: bool = True

    def __post_init__(self) -> None:
        if len(self.m) != len(self.n) or len(self.m) != len(self.b_cos):
            msg = "m, n, and b_cos must have the same length"
            raise ValueError(msg)


@dataclass(frozen=True)
class GeometryOnGrid:
    grid: AngularGrid
    theta_2d: Array
    zeta_2d: Array
    b: Array
    d_b_dtheta: Array
    d_b_dzeta: Array
    jacobian: Array
    volume_prime: Array
    b2_mean: Array
    radial_drift_spatial: Array
    b0: Array


def example_surface(dtype=jnp.float64) -> BoozerSurface:
    """Return a small stellarator-symmetric test surface."""

    return BoozerSurface(
        m=jnp.asarray([0, 1, 1, 2], dtype=jnp.int32),
        n=jnp.asarray([0, 0, 1, -1], dtype=jnp.int32),
        b_cos=jnp.asarray([1.0, 0.06, 0.025, 0.01], dtype=dtype),
        nfp=5,
        iota=0.85,
        psi_p=1.0,
        b_theta=0.05,
        b_zeta=1.0,
    )


def evaluate_boozer_modes(
    surface: BoozerSurface,
    theta: Array,
    zeta: Array,
) -> tuple[Array, Array, Array]:
    """Evaluate `B`, `dB/dtheta`, and `dB/dzeta` on broadcastable arrays."""

    m = jnp.asarray(surface.m)
    n = jnp.asarray(surface.n)
    b_cos = jnp.asarray(surface.b_cos)
    phase = theta[..., None] * m + zeta[..., None] * (n * surface.nfp)
    b = jnp.sum(b_cos * jnp.cos(phase), axis=-1)
    d_b_dtheta = jnp.sum(-b_cos * m * jnp.sin(phase), axis=-1)
    d_b_dzeta = jnp.sum(-b_cos * (n * surface.nfp) * jnp.sin(phase), axis=-1)
    if surface.b_sin is not None:
        b_sin = jnp.asarray(surface.b_sin)
        b = b + jnp.sum(b_sin * jnp.sin(phase), axis=-1)
        d_b_dtheta = d_b_dtheta + jnp.sum(b_sin * m * jnp.cos(phase), axis=-1)
        d_b_dzeta = d_b_dzeta + jnp.sum(b_sin * (n * surface.nfp) * jnp.cos(phase), axis=-1)
    return b, d_b_dtheta, d_b_dzeta


def geometry_on_grid(surface: BoozerSurface, spec) -> GeometryOnGrid:
    """Evaluate all geometric quantities needed by the solver."""

    grid = periodic_grid(spec, surface.nfp)
    theta_2d, zeta_2d = jnp.meshgrid(grid.theta, grid.zeta, indexing="ij")
    b, d_b_dtheta, d_b_dzeta = evaluate_boozer_modes(surface, theta_2d, zeta_2d)
    denominator = surface.b_zeta + surface.iota * surface.b_theta
    jacobian = jnp.abs(denominator) / b**2
    volume_prime = jnp.sum(jacobian) * grid.dtheta * grid.dzeta
    b2_mean = flux_surface_average(b**2, jacobian, grid.dtheta, grid.dzeta)
    radial_drift_spatial = (
        (surface.b_theta * d_b_dzeta - surface.b_zeta * d_b_dtheta)
        / (jacobian * b**3)
    )
    if surface.b0 is None:
        b0 = jnp.sum(b) / b.size
    else:
        b0 = jnp.asarray(surface.b0, dtype=b.dtype)
    return GeometryOnGrid(
        grid=grid,
        theta_2d=theta_2d,
        zeta_2d=zeta_2d,
        b=b,
        d_b_dtheta=d_b_dtheta,
        d_b_dzeta=d_b_dzeta,
        jacobian=jacobian,
        volume_prime=volume_prime,
        b2_mean=b2_mean,
        radial_drift_spatial=radial_drift_spatial,
        b0=b0,
    )
