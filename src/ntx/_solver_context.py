"""Shared solver context construction."""

from __future__ import annotations

import jax.numpy as jnp

from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .operators import OperatorContext


def _operator_context(
    surface: BoozerSurface | VmecSurface,
    geom,
    grid: GridSpec,
    nu_hat,
    epsi_hat,
) -> OperatorContext:
    return OperatorContext(
        surface=surface,
        geometry=geom,
        nu_hat=jnp.asarray(nu_hat, dtype=grid.jax_dtype),
        epsi_hat=jnp.asarray(epsi_hat, dtype=grid.jax_dtype),
    )


__all__ = ["_operator_context"]
