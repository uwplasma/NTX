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


def fourier_series_coefficient_bars_multi_rhs(
    m: Array,
    n: Array,
    theta: Array,
    zeta: Array,
    *,
    nfp: int,
    value_bar: Array,
    d_theta_bar: Array | None = None,
    d_zeta_bar: Array | None = None,
) -> tuple[Array, Array]:
    """Native RHS-axis transpose of :func:`evaluate_fourier_series`.

    Bars have shape ``(rhs, theta, zeta)`` and the returned cosine/sine
    coefficient bars have shape ``(rhs, mode)``.  This is pure Fourier-basis
    algebra and does not construct geometry or invoke any solver.
    """
    phase = theta[..., None] * m + zeta[..., None] * (n * nfp)
    cosine = jnp.cos(phase)
    sine = jnp.sin(phase)
    if d_theta_bar is None:
        d_theta_bar = jnp.zeros_like(value_bar)
    if d_zeta_bar is None:
        d_zeta_bar = jnp.zeros_like(value_bar)
    mode_m = jnp.asarray(m)[None, None, None, :]
    mode_nfp = (jnp.asarray(n) * nfp)[None, None, None, :]
    value_bar = value_bar[..., None]
    d_theta_bar = d_theta_bar[..., None]
    d_zeta_bar = d_zeta_bar[..., None]
    cosine_bar = jnp.sum(
        value_bar * cosine
        - d_theta_bar * mode_m * sine
        - d_zeta_bar * mode_nfp * sine,
        axis=(1, 2),
    )
    sine_bar = jnp.sum(
        value_bar * sine
        + d_theta_bar * mode_m * cosine
        + d_zeta_bar * mode_nfp * cosine,
        axis=(1, 2),
    )
    return cosine_bar, sine_bar


def volume_prime_bars_multi_rhs(volume_prime_bar: Array, jacobian: Array, grid) -> Array:
    """Add the RHS-axis transpose of ``volume_prime = sum(J) dtheta dzeta``."""
    return volume_prime_bar[:, None, None] * jnp.ones_like(jacobian)[None] * (
        grid.dtheta * grid.dzeta
    )


def b2_mean_bars_multi_rhs(b: Array, jacobian: Array, b2_mean_bar: Array, grid):
    """RHS-axis transpose of the flux-surface average defining ``b2_mean``."""
    delta = grid.dtheta * grid.dzeta
    volume = jnp.sum(jacobian) * delta
    b2 = jnp.sum(b**2 * jacobian) * delta / volume
    weight = b2_mean_bar[:, None, None]
    return (
        weight * 2.0 * b[None] * jacobian[None] * delta / volume,
        weight * (b[None] ** 2 - b2) * delta / volume,
    )


def radial_drift_bars_multi_rhs(
    b: Array, d_b_dtheta: Array, d_b_dzeta: Array, jacobian: Array,
    b_sub_theta: Array, b_sub_zeta: Array, radial_drift_bar: Array,
):
    """RHS-axis transpose of the sampled radial-drift expression."""
    drift = (b_sub_theta * d_b_dzeta - b_sub_zeta * d_b_dtheta) / (jacobian * b**3)
    weight = radial_drift_bar
    denom = jacobian[None] * b[None] ** 3
    return {
        "b": -3.0 * weight * drift[None] / b[None],
        "d_b_dtheta": -weight * b_sub_zeta[None] / denom,
        "d_b_dzeta": weight * b_sub_theta[None] / denom,
        "jacobian": -weight * drift[None] / jacobian[None],
        "b_sub_theta": weight * d_b_dzeta[None] / denom,
        "b_sub_zeta": -weight * d_b_dtheta[None] / denom,
    }


def add_sampled_field_bars_multi_rhs(*bar_dicts: dict[str, Array]) -> dict[str, Array]:
    """Combine named sampled-geometry cotangents without reducing RHS."""
    result: dict[str, Array] = {}
    for bars in bar_dicts:
        for name, bar in bars.items():
            result[name] = bar if name not in result else result[name] + bar
    return result


def vmec_sampled_field_bars_to_coefficients_multi_rhs(surface: VmecSurface, geometry: GeometryOnGrid, bars: dict[str, Array]) -> dict[str, Array]:
    """Map RHS-batched sampled VMEC field bars to Fourier coefficient bars."""
    zeros = lambda: jnp.zeros_like(bars["b"])
    result: dict[str, Array] = {}
    result["b_cos"], _ = fourier_series_coefficient_bars_multi_rhs(
        surface.m, surface.n, geometry.theta_2d, geometry.zeta_2d, nfp=surface.nfp,
        value_bar=bars.get("b", zeros()), d_theta_bar=bars.get("d_b_dtheta"),
        d_zeta_bar=bars.get("d_b_dzeta"),
    )
    for field, coeff in (("jacobian", "jacobian_cos"), ("b_sub_theta", "b_sub_theta_cos"), ("b_sub_zeta", "b_sub_zeta_cos"), ("b_sup_theta", "b_sup_theta_cos"), ("b_sup_zeta", "b_sup_zeta_cos")):
        if field in bars:
            result[coeff], _ = fourier_series_coefficient_bars_multi_rhs(
                surface.m, surface.n, geometry.theta_2d, geometry.zeta_2d,
                nfp=surface.nfp, value_bar=bars[field]
            )
    return result


def vmec_geometry_bars_to_coefficients_multi_rhs(
    surface: VmecSurface, geometry: GeometryOnGrid, primitive_bars: dict[str, Array]
) -> dict[str, Array]:
    """Reverse derived VMEC geometry fields to the compact surface contract.

    The six Fourier arrays are obtained by reversing their sampled fields.
    ``b0`` is an independent scalar leaf of :class:`VmecSurface`, however: it
    is consumed directly by the transport-coefficient formulas and is not a
    Fourier coefficient.  Preserve its RHS-batched bar here so callers that
    replace the generic prepared VJP do not silently drop that surface path.
    """
    template = primitive_bars["b"]
    zero_field = jnp.zeros_like(template)
    sampled = dict(primitive_bars)
    volume_bar = sampled.pop("volume_prime", None)
    if volume_bar is not None:
        sampled = add_sampled_field_bars_multi_rhs(
            sampled, {"jacobian": volume_prime_bars_multi_rhs(volume_bar, geometry.jacobian, geometry.grid)}
        )
    b2_bar = sampled.pop("b2_mean", None)
    if b2_bar is not None:
        b_bar, jacobian_bar = b2_mean_bars_multi_rhs(
            geometry.b, geometry.jacobian, b2_bar, geometry.grid
        )
        sampled = add_sampled_field_bars_multi_rhs(sampled, {"b": b_bar, "jacobian": jacobian_bar})
    drift_bar = sampled.pop("radial_drift_spatial", None)
    if drift_bar is not None:
        sampled = add_sampled_field_bars_multi_rhs(
            sampled,
            radial_drift_bars_multi_rhs(
                geometry.b, geometry.d_b_dtheta, geometry.d_b_dzeta,
                geometry.jacobian, geometry.b_sub_theta, geometry.b_sub_zeta,
                drift_bar,
            ),
        )
    sampled.setdefault("b", zero_field)
    result = vmec_sampled_field_bars_to_coefficients_multi_rhs(surface, geometry, sampled)
    if "b0" in primitive_bars:
        result["b0"] = jnp.asarray(primitive_bars["b0"])
    return result


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
