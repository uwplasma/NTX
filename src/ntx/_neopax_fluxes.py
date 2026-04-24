"""Differentiable NEOPAX flux observable helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def get_differentiable_neopax_fluxes(species, grid, field, database):
    """Evaluate NEOPAX no-momentum fluxes with an axis-safe radial block.

    The reference monoenergetic databases do not include the magnetic axis
    exactly. For integrated objectives such as total bootstrap current, the
    axis contribution is weighted by `Vprime[0] = 0`, so copying the first
    interior radial block into the axis block removes an AD-only singularity
    without changing the physical integral.
    """

    try:
        from NEOPAX._neoclassical import get_Lij_matrix
    except ImportError as exc:  # pragma: no cover - exercised with local NEOPAX
        raise ImportError("NEOPAX is required for differentiable flux evaluation") from exc

    def _fluxes_internal(species_internal, species_index, lij):
        a1 = species_internal.A1[species_index]
        a2 = species_internal.A2[species_index]
        a3 = species_internal.A3
        temperature = species_internal.temperature[species_index]
        density = species_internal.density[species_index]
        gamma = -density * (
            lij[species_index, :, 0, 0] * a1
            + lij[species_index, :, 0, 1] * a2
            + lij[species_index, :, 0, 2] * a3
        )
        heat = -temperature * density * (
            lij[species_index, :, 1, 0] * a1
            + lij[species_index, :, 1, 1] * a2
            + lij[species_index, :, 1, 2] * a3
        )
        upar = -density * (
            lij[species_index, :, 2, 0] * a1
            + lij[species_index, :, 2, 1] * a2
            + lij[species_index, :, 2, 2] * a3
        )
        return gamma, heat, upar

    radial_indices = jnp.asarray(grid.full_grid_indeces)
    interior_indices = radial_indices[1:]
    lij_interior = jax.vmap(
        jax.vmap(get_Lij_matrix, in_axes=(None, None, None, None, None, 0)),
        in_axes=(None, None, None, None, 0, None),
    )(species, grid, field, database, species.species_indeces, interior_indices)
    lij = jnp.concatenate([lij_interior[:, :1, :, :], lij_interior], axis=1)
    gamma, heat, upar = jax.vmap(_fluxes_internal, in_axes=(None, 0, None))(
        species,
        species.species_indeces,
        lij,
    )
    return lij, gamma, heat, upar
