"""Fourier evaluation and grid projection helpers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ._geometry_types import BoozerSurface, GeometryOnGrid, VmecSurface
from .grids import flux_surface_average, periodic_grid


def example_surface(dtype=jnp.float64) -> BoozerSurface:
    """Return a small stellarator-symmetric test surface."""

    return BoozerSurface(
        m=jnp.asarray([0, 1, 1, 2], dtype=jnp.int32),
        n=jnp.asarray([0, 0, 1, -1], dtype=jnp.int32),
        b_cos=jnp.asarray([1.0, 0.06, 0.025, 0.01], dtype=dtype),
        nfp=5,
        iota=0.85,
        psi_p=1.0,
        chi_p=0.85,
        b_theta=0.05,
        b_zeta=1.0,
    )


def evaluate_boozer_modes(
    surface: BoozerSurface,
    theta: Array,
    zeta: Array,
) -> tuple[Array, Array, Array]:
    """Evaluate `B`, `dB/dtheta`, and `dB/dzeta` on broadcastable arrays."""

    return evaluate_fourier_series(
        surface.m,
        surface.n,
        surface.b_cos,
        theta,
        zeta,
        nfp=surface.nfp,
        sin_coeffs=surface.b_sin,
    )


def evaluate_fourier_series(
    m: Array,
    n: Array,
    cos_coeffs: Array,
    theta: Array,
    zeta: Array,
    *,
    nfp: int,
    sin_coeffs: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Evaluate a stellarator-symmetric or general Fourier series and its derivatives."""

    m = jnp.asarray(m)
    n = jnp.asarray(n)
    cos_coeffs = jnp.asarray(cos_coeffs)
    phase = theta[..., None] * m + zeta[..., None] * (n * nfp)
    value = jnp.sum(cos_coeffs * jnp.cos(phase), axis=-1)
    d_dtheta = jnp.sum(-cos_coeffs * m * jnp.sin(phase), axis=-1)
    d_dzeta = jnp.sum(-cos_coeffs * (n * nfp) * jnp.sin(phase), axis=-1)
    if sin_coeffs is not None:
        sin_coeffs = jnp.asarray(sin_coeffs)
        value = value + jnp.sum(sin_coeffs * jnp.sin(phase), axis=-1)
        d_dtheta = d_dtheta + jnp.sum(sin_coeffs * m * jnp.cos(phase), axis=-1)
        d_dzeta = d_dzeta + jnp.sum(sin_coeffs * (n * nfp) * jnp.cos(phase), axis=-1)
    return value, d_dtheta, d_dzeta


def geometry_on_grid(surface: BoozerSurface | VmecSurface, spec) -> GeometryOnGrid:
    """Evaluate all geometric quantities needed by the solver."""

    if isinstance(surface, VmecSurface):
        return _vmec_geometry_on_grid(surface, spec)
    return _boozer_geometry_on_grid(surface, spec)


def _boozer_geometry_on_grid(surface: BoozerSurface, spec) -> GeometryOnGrid:
    grid = periodic_grid(spec, surface.nfp)
    theta_2d, zeta_2d = jnp.meshgrid(grid.theta, grid.zeta, indexing="ij")
    b, d_b_dtheta, d_b_dzeta = evaluate_boozer_modes(surface, theta_2d, zeta_2d)
    denominator = surface.b_zeta + surface.iota * surface.b_theta
    jacobian = jnp.abs(denominator) / b**2
    b_sub_theta = jnp.full_like(b, surface.b_theta)
    b_sub_zeta = jnp.full_like(b, surface.b_zeta)
    b_sup_theta = surface.iota / jacobian
    b_sup_zeta = 1.0 / jacobian
    volume_prime = jnp.sum(jacobian) * grid.dtheta * grid.dzeta
    b2_mean = flux_surface_average(b**2, jacobian, grid.dtheta, grid.dzeta)
    radial_drift_spatial = (b_sub_theta * d_b_dzeta - b_sub_zeta * d_b_dtheta) / (jacobian * b**3)
    if surface.b0 is None:
        b0 = jnp.sum(b) / b.size
    else:
        b0 = jnp.asarray(surface.b0, dtype=b.dtype)
    return GeometryOnGrid(
        surface_type="boozer",
        surface_path=None,
        nfp=surface.nfp,
        iota=surface.iota,
        psi_p=surface.psi_p,
        transport_psi_scale=surface.psi_p,
        coefficient_psi_scale=surface.psi_p,
        grid=grid,
        theta_2d=theta_2d,
        zeta_2d=zeta_2d,
        b=b,
        d_b_dtheta=d_b_dtheta,
        d_b_dzeta=d_b_dzeta,
        jacobian=jacobian,
        b_sub_theta=b_sub_theta,
        b_sub_zeta=b_sub_zeta,
        b_sup_theta=b_sup_theta,
        b_sup_zeta=b_sup_zeta,
        volume_prime=volume_prime,
        b2_mean=b2_mean,
        radial_drift_spatial=radial_drift_spatial,
        b0=b0,
    )


def _vmec_geometry_on_grid(surface: VmecSurface, spec) -> GeometryOnGrid:
    grid = periodic_grid(spec, surface.nfp)
    theta_2d, zeta_2d = jnp.meshgrid(grid.theta, grid.zeta, indexing="ij")
    b, d_b_dtheta, d_b_dzeta = evaluate_fourier_series(
        surface.m,
        surface.n,
        surface.b_cos,
        theta_2d,
        zeta_2d,
        nfp=surface.nfp,
    )
    jacobian, _, _ = evaluate_fourier_series(
        surface.m,
        surface.n,
        surface.jacobian_cos,
        theta_2d,
        zeta_2d,
        nfp=surface.nfp,
    )
    b_sub_theta, _, _ = evaluate_fourier_series(
        surface.m,
        surface.n,
        surface.b_sub_theta_cos,
        theta_2d,
        zeta_2d,
        nfp=surface.nfp,
    )
    b_sub_zeta, _, _ = evaluate_fourier_series(
        surface.m,
        surface.n,
        surface.b_sub_zeta_cos,
        theta_2d,
        zeta_2d,
        nfp=surface.nfp,
    )
    b_sup_theta, _, _ = evaluate_fourier_series(
        surface.m,
        surface.n,
        surface.b_sup_theta_cos,
        theta_2d,
        zeta_2d,
        nfp=surface.nfp,
    )
    b_sup_zeta, _, _ = evaluate_fourier_series(
        surface.m,
        surface.n,
        surface.b_sup_zeta_cos,
        theta_2d,
        zeta_2d,
        nfp=surface.nfp,
    )
    volume_prime = jnp.sum(jacobian) * grid.dtheta * grid.dzeta
    b2_mean = flux_surface_average(b**2, jacobian, grid.dtheta, grid.dzeta)
    radial_drift_spatial = (b_sub_theta * d_b_dzeta - b_sub_zeta * d_b_dtheta) / (jacobian * b**3)
    return GeometryOnGrid(
        surface_type="vmec",
        surface_path=surface.path,
        nfp=surface.nfp,
        iota=surface.iota,
        psi_p=surface.psi_p,
        transport_psi_scale=surface.transport_psi_scale,
        coefficient_psi_scale=1.0,
        grid=grid,
        theta_2d=theta_2d,
        zeta_2d=zeta_2d,
        b=b,
        d_b_dtheta=d_b_dtheta,
        d_b_dzeta=d_b_dzeta,
        jacobian=jacobian,
        b_sub_theta=b_sub_theta,
        b_sub_zeta=b_sub_zeta,
        b_sup_theta=b_sup_theta,
        b_sup_zeta=b_sup_zeta,
        volume_prime=volume_prime,
        b2_mean=b2_mean,
        radial_drift_spatial=radial_drift_spatial,
        b0=jnp.asarray(surface.b0, dtype=b.dtype),
    )
