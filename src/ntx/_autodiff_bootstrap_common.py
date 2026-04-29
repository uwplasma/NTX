"""Shared helpers for bootstrap-current autodiff workflows."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from ._autodiff_helpers import (
    dominant_nonaxisymmetric_mode as _dominant_nonaxisymmetric_mode,
)
from ._autodiff_helpers import mode_value_for_surface as _mode_value_for_surface
from ._autodiff_helpers import scale_surface_mode as _scale_surface_mode
from .neopax import (
    build_ntx_neopax_scan_from_surfaces,
    scan_to_neopax_arrays,
)


@dataclass(frozen=True)
class BootstrapProfileContext:
    """Precomputed profile and harmonic data for bootstrap-current examples."""

    surfaces: tuple
    rho: Array
    nu_value: Array
    density: Array
    density_gradient: Array
    temperature_gradient: Array
    objective_weight: Array
    harmonic_m: int
    harmonic_n: int
    harmonic_reference_value: Array
    zero_scan: Array
    unit_drds: Array


def build_bootstrap_profile_context(
    surfaces: tuple,
    *,
    rho: Array,
    nu_v: Array,
    nu_index: int,
) -> BootstrapProfileContext:
    """Build the shared radial profiles and controlled harmonic metadata."""

    rho_grid = jnp.asarray(rho)
    dtype = rho_grid.dtype
    nu_value = jnp.asarray(nu_v[nu_index], dtype=dtype)
    density = 3.2e19 * (1.0 - rho_grid**4) + 0.45e19
    temperature = 3.0e3 * (1.0 - rho_grid**2) + 0.8e3
    density_gradient = jnp.asarray(jnp.gradient(jnp.log(density), rho_grid))
    temperature_gradient = jnp.asarray(jnp.gradient(jnp.log(temperature), rho_grid))
    objective_weight = jnp.exp(-0.5 * ((rho_grid - 0.45) / 0.16) ** 2)
    harmonic_m, harmonic_n = _dominant_nonaxisymmetric_mode(surfaces[len(surfaces) // 2])
    harmonic_reference_value = _mode_value_for_surface(
        surfaces[len(surfaces) // 2],
        harmonic_m,
        harmonic_n,
    )
    zero_scan = jnp.zeros((rho_grid.size, 1), dtype=dtype)
    unit_drds = jnp.ones_like(rho_grid)
    return BootstrapProfileContext(
        surfaces=surfaces,
        rho=rho_grid,
        nu_value=nu_value,
        density=density,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
        objective_weight=objective_weight,
        harmonic_m=harmonic_m,
        harmonic_n=harmonic_n,
        harmonic_reference_value=jnp.asarray(harmonic_reference_value),
        zero_scan=zero_scan,
        unit_drds=unit_drds,
    )


def bounded_surface_scale(raw_scale: Array) -> Array:
    """Map an unconstrained scalar control to the bounded harmonic scale."""

    return 1.0 + 0.35 * jnp.tanh(raw_scale)


def raw_scale_from_bounded_scale(scale: Array) -> Array:
    """Invert :func:`bounded_surface_scale` on the plotted scale interval."""

    return jnp.arctanh(jnp.clip((scale - 1.0) / 0.35, -0.999, 0.999))


def transport_profiles_from_raw_scale(
    context: BootstrapProfileContext,
    raw_scale: Array,
    *,
    grid,
    a_b: float,
    source_name: str,
) -> tuple[Array, Array, Array]:
    """Evaluate the reduced current response and selected monoenergetic profiles."""

    scale = bounded_surface_scale(raw_scale)
    perturbed_surfaces = tuple(
        _scale_surface_mode(
            surface,
            context.harmonic_m,
            context.harmonic_n,
            scale,
        )
        for surface in context.surfaces
    )
    scan = build_ntx_neopax_scan_from_surfaces(
        perturbed_surfaces,
        rho=context.rho,
        nu_v=jnp.asarray([context.nu_value]),
        Es=context.zero_scan,
        Er=context.zero_scan,
        drds=context.unit_drds,
        grid=grid,
        source_name=source_name,
    )
    arrays = scan_to_neopax_arrays(scan, a_b=a_b)
    d13 = arrays.D13[:, 0, 0]
    d33 = arrays.D33[:, 0, 0]
    current = context.density * (
        -context.density_gradient * d13 - 0.75 * context.temperature_gradient * d33
    )
    return current, d13, d33
