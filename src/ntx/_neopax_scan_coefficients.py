"""Coefficient and normalization-block assembly for NEOPAX scans."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from ._neopax_bridge import _surface_reference_bridge
from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from ._solver_core import prepare_monoenergetic_system
from ._solver_scan_execution import _resolved_scan_inputs, _scan_coefficients_serial
from ._solver_prepared import pullback_prepared_coefficient_vector_case_and_prepared
from ._solver_types import MonoenergeticCase
from .solver import solve_monoenergetic_scan


@dataclass(frozen=True)
class NeopaxScanCoefficientBlocks:
    """Solved monoenergetic blocks plus reference-normalization metadata."""

    D11: Array
    D13: Array
    D33: Array
    D33_spitzer: Array
    b00: Array
    boozer_i: Array
    boozer_g: Array
    iota: Array
    fac_reference_to_sfincs_11: Array
    fac_reference_to_sfincs_31: Array
    fac_reference_to_sfincs_33: Array
    fac_sfincs_to_dkes_11: Array
    fac_sfincs_to_dkes_31: Array
    fac_sfincs_to_dkes_33: Array


_COEFFICIENT_BLOCK_NAMES = tuple(NeopaxScanCoefficientBlocks.__dataclass_fields__)


def solve_neopax_scan_coefficient_blocks(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
) -> NeopaxScanCoefficientBlocks:
    """Solve all surface/electric-field blocks used by a NEOPAX scan."""

    d11_list = []
    d13_list = []
    d33_list = []
    d33_spitzer_list = []
    b00_list = []
    boozer_i_list = []
    boozer_g_list = []
    iota_list = []
    fac_11_list = []
    fac_31_list = []
    fac_33_list = []
    sfincs_to_dkes_11_list = []
    sfincs_to_dkes_31_list = []
    sfincs_to_dkes_33_list = []
    for surface, es_row in zip(surfaces, Es, strict=True):
        nu_grid, es_grid = jnp.meshgrid(nu_v, es_row, indexing="ij")
        coeffs = solve_monoenergetic_scan(surface, grid, nu_grid, epsi_hat=es_grid)
        d11_list.append(coeffs["D11"])
        d13_list.append(coeffs["D13"])
        d33_list.append(coeffs["D33"])
        d33_spitzer_list.append(coeffs["D33_spitzer"])
        bridge = _surface_reference_bridge(surface)
        b00_list.append(bridge["b00"])
        boozer_i_list.append(bridge["boozer_i"])
        boozer_g_list.append(bridge["boozer_g"])
        iota_list.append(bridge["iota"])
        fac_11_list.append(bridge["fac_11"])
        fac_31_list.append(bridge["fac_31"])
        fac_33_list.append(bridge["fac_33"])
        sfincs_to_dkes_11_list.append(bridge["fac_sfincs_to_dkes_11"])
        sfincs_to_dkes_31_list.append(bridge["fac_sfincs_to_dkes_31"])
        sfincs_to_dkes_33_list.append(bridge["fac_sfincs_to_dkes_33"])

    return NeopaxScanCoefficientBlocks(
        D11=jnp.stack(d11_list),
        D13=jnp.stack(d13_list),
        D33=jnp.stack(d33_list),
        D33_spitzer=jnp.stack(d33_spitzer_list),
        b00=jnp.asarray(b00_list),
        boozer_i=jnp.asarray(boozer_i_list),
        boozer_g=jnp.asarray(boozer_g_list),
        iota=jnp.asarray(iota_list),
        fac_reference_to_sfincs_11=jnp.asarray(fac_11_list),
        fac_reference_to_sfincs_31=jnp.asarray(fac_31_list),
        fac_reference_to_sfincs_33=jnp.asarray(fac_33_list),
        fac_sfincs_to_dkes_11=jnp.asarray(sfincs_to_dkes_11_list),
        fac_sfincs_to_dkes_31=jnp.asarray(sfincs_to_dkes_31_list),
        fac_sfincs_to_dkes_33=jnp.asarray(sfincs_to_dkes_33_list),
    )


def solve_neopax_scan_coefficient_blocks_prepared(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
) -> NeopaxScanCoefficientBlocks:
    """Build scan blocks through explicit per-surface prepared systems.

    The values are identical to :func:`solve_neopax_scan_coefficient_blocks`.
    This intentionally exposes the prepared-system boundary required by the
    opt-in structured scan reverse rule: a later backward implementation can
    reuse each surface's coefficient adjoint rather than reverse-tracing the
    whole ``solve_monoenergetic_scan`` call.

    This helper is not selected by the production generic scan path yet.
    """

    d11_list = []
    d13_list = []
    d33_list = []
    d33_spitzer_list = []
    b00_list = []
    boozer_i_list = []
    boozer_g_list = []
    iota_list = []
    fac_11_list = []
    fac_31_list = []
    fac_33_list = []
    sfincs_to_dkes_11_list = []
    sfincs_to_dkes_31_list = []
    sfincs_to_dkes_33_list = []
    for surface, es_row in zip(surfaces, Es, strict=True):
        prepared = prepare_monoenergetic_system(surface, grid)
        nu_grid, es_grid = jnp.meshgrid(nu_v, es_row, indexing="ij")
        nu_values, epsi_values, output_shape = _resolved_scan_inputs(
            prepared,
            grid,
            nu_grid,
            es_grid,
            None,
        )
        coefficients = _scan_coefficients_serial(
            prepared,
            nu_values.ravel(),
            epsi_values.ravel(),
        ).reshape((*output_shape, 5))
        d11_list.append(coefficients[..., 0])
        d13_list.append(coefficients[..., 2])
        d33_list.append(coefficients[..., 3])
        d33_spitzer_list.append(coefficients[..., 4])
        bridge = _surface_reference_bridge(surface)
        b00_list.append(bridge["b00"])
        boozer_i_list.append(bridge["boozer_i"])
        boozer_g_list.append(bridge["boozer_g"])
        iota_list.append(bridge["iota"])
        fac_11_list.append(bridge["fac_11"])
        fac_31_list.append(bridge["fac_31"])
        fac_33_list.append(bridge["fac_33"])
        sfincs_to_dkes_11_list.append(bridge["fac_sfincs_to_dkes_11"])
        sfincs_to_dkes_31_list.append(bridge["fac_sfincs_to_dkes_31"])
        sfincs_to_dkes_33_list.append(bridge["fac_sfincs_to_dkes_33"])

    return NeopaxScanCoefficientBlocks(
        D11=jnp.stack(d11_list),
        D13=jnp.stack(d13_list),
        D33=jnp.stack(d33_list),
        D33_spitzer=jnp.stack(d33_spitzer_list),
        b00=jnp.asarray(b00_list),
        boozer_i=jnp.asarray(boozer_i_list),
        boozer_g=jnp.asarray(boozer_g_list),
        iota=jnp.asarray(iota_list),
        fac_reference_to_sfincs_11=jnp.asarray(fac_11_list),
        fac_reference_to_sfincs_31=jnp.asarray(fac_31_list),
        fac_reference_to_sfincs_33=jnp.asarray(fac_33_list),
        fac_sfincs_to_dkes_11=jnp.asarray(sfincs_to_dkes_11_list),
        fac_sfincs_to_dkes_31=jnp.asarray(sfincs_to_dkes_31_list),
        fac_sfincs_to_dkes_33=jnp.asarray(sfincs_to_dkes_33_list),
    )


def pullback_neopax_scan_coefficient_blocks_prepared(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
    coefficient_blocks_bar: NeopaxScanCoefficientBlocks,
) -> tuple[tuple[BoozerSurface | VmecSurface, ...], Array]:
    """Explicit scan coefficient transpose for the structured reverse mode.

    ``coefficient_blocks_bar`` is a cotangent of the output of
    :func:`solve_neopax_scan_coefficient_blocks_prepared`.  For each fixed
    scan surface, this contracts all ``(nu, Es)`` coefficient bars through
    NTX's prepared coefficient adjoint and accumulates one prepared-system
    cotangent.  Only that compact accumulated tree is then pulled back through
    surface preparation.  The reference-normalization fields use their small
    ordinary JAX VJP.

    This is intentionally an explicit API for parity testing before it is
    installed as a custom VJP for the live scan builder.
    """

    def _zero_float_tree(tree):
        def _zero(leaf):
            value = jnp.asarray(leaf)
            dtype = value.dtype if jnp.issubdtype(value.dtype, jnp.inexact) else jnp.float64
            return jnp.zeros(value.shape, dtype=dtype)

        return jax.tree_util.tree_map(_zero, tree)

    def _add_trees(left, right):
        def _add(left_leaf, right_leaf):
            left_value = jnp.asarray(left_leaf)
            right_value = jnp.asarray(right_leaf)
            if right_value.dtype == jax.dtypes.float0:
                return left_leaf
            if left_value.dtype == jax.dtypes.float0:
                return right_leaf
            return left_leaf + right_leaf

        return jax.tree_util.tree_map(_add, left, right)

    surface_bars = []
    es_bars = []
    for surface_index, (surface, es_row) in enumerate(zip(surfaces, Es, strict=True)):
        prepared = prepare_monoenergetic_system(surface, grid)
        nu_grid, es_grid = jnp.meshgrid(nu_v, es_row, indexing="ij")
        nu_values, epsi_values, _output_shape = _resolved_scan_inputs(
            prepared,
            grid,
            nu_grid,
            es_grid,
            None,
        )
        coefficient_bars = jnp.stack(
            (
                coefficient_blocks_bar.D11[surface_index],
                jnp.zeros_like(coefficient_blocks_bar.D11[surface_index]),
                coefficient_blocks_bar.D13[surface_index],
                coefficient_blocks_bar.D33[surface_index],
                coefficient_blocks_bar.D33_spitzer[surface_index],
            ),
            axis=-1,
        ).reshape((-1, 5))

        def _accumulate(prepared_bar, values):
            nu_value, epsi_value, coefficient_bar = values
            case_bar, local_prepared_bar = (
                pullback_prepared_coefficient_vector_case_and_prepared(
                    prepared,
                    MonoenergeticCase(nu_hat=nu_value, epsi_hat=epsi_value),
                    coefficient_bar,
                )
            )
            return _add_trees(prepared_bar, local_prepared_bar), case_bar.epsi_hat

        prepared_bar, epsi_flat_bar = jax.lax.scan(
            _accumulate,
            _zero_float_tree(prepared),
            (
                nu_values.reshape((-1,)),
                epsi_values.reshape((-1,)),
                coefficient_bars,
            ),
        )
        # ``lax.scan`` carries a scalar/vector accumulator.  Its epsi
        # contribution is per scan case, so reconstruct the original field
        # grid and sum the collisionality axis back to the supplied Es row.
        epsi_bar_grid = epsi_flat_bar.reshape(es_grid.shape)
        es_bars.append(jnp.sum(epsi_bar_grid, axis=0))

        _, prepare_pullback = jax.vjp(
            lambda surface_value: prepare_monoenergetic_system(surface_value, grid),
            surface,
        )
        surface_bar = prepare_pullback(prepared_bar)[0]
        reference_bar = {
            "b00": coefficient_blocks_bar.b00[surface_index],
            "boozer_i": coefficient_blocks_bar.boozer_i[surface_index],
            "boozer_g": coefficient_blocks_bar.boozer_g[surface_index],
            "iota": coefficient_blocks_bar.iota[surface_index],
            "fac_11": coefficient_blocks_bar.fac_reference_to_sfincs_11[surface_index],
            "fac_31": coefficient_blocks_bar.fac_reference_to_sfincs_31[surface_index],
            "fac_33": coefficient_blocks_bar.fac_reference_to_sfincs_33[surface_index],
            "fac_sfincs_to_dkes_11": coefficient_blocks_bar.fac_sfincs_to_dkes_11[surface_index],
            "fac_sfincs_to_dkes_31": coefficient_blocks_bar.fac_sfincs_to_dkes_31[surface_index],
            "fac_sfincs_to_dkes_33": coefficient_blocks_bar.fac_sfincs_to_dkes_33[surface_index],
        }
        _, reference_pullback = jax.vjp(_surface_reference_bridge, surface)
        reference_surface_bar = reference_pullback(reference_bar)[0]
        surface_bars.append(_add_trees(surface_bar, reference_surface_bar))

    return tuple(surface_bars), jnp.stack(es_bars)


def _coefficient_block_values(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
) -> tuple[Array, ...]:
    blocks = solve_neopax_scan_coefficient_blocks_prepared(
        surfaces, Es=Es, nu_v=nu_v, grid=grid
    )
    return tuple(getattr(blocks, name) for name in _COEFFICIENT_BLOCK_NAMES)


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def _structured_scan_coefficient_block_values(
    surface_tree,
    surface_leaves: tuple[Array, ...],
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
) -> tuple[Array, ...]:
    """Prepared scan values with a bounded coefficient-adjoint VJP."""

    surfaces = jax.tree_util.tree_unflatten(surface_tree, surface_leaves)
    return _coefficient_block_values(surfaces, Es, nu_v, grid)


def _structured_scan_coefficient_block_values_fwd(
    surface_tree,
    surface_leaves: tuple[Array, ...],
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
) -> tuple[tuple[Array, ...], tuple[tuple[BoozerSurface | VmecSurface, ...], Array]]:
    surfaces = jax.tree_util.tree_unflatten(surface_tree, surface_leaves)
    values = _coefficient_block_values(surfaces, Es, nu_v, grid)
    return values, (surfaces, Es)


def _structured_scan_coefficient_block_values_bwd(
    surface_tree,
    nu_v: Array,
    grid: GridSpec,
    residuals: tuple[tuple[BoozerSurface | VmecSurface, ...], Array],
    output_bars: tuple[Array, ...],
) -> tuple[tuple[Array, ...], Array]:
    surfaces, Es = residuals
    blocks_bar = NeopaxScanCoefficientBlocks(
        **dict(zip(_COEFFICIENT_BLOCK_NAMES, output_bars, strict=True))
    )
    surface_bars, es_bar = pullback_neopax_scan_coefficient_blocks_prepared(
        surfaces,
        Es=Es,
        nu_v=nu_v,
        grid=grid,
        coefficient_blocks_bar=blocks_bar,
    )
    del surface_tree
    return tuple(jax.tree_util.tree_leaves(surface_bars)), es_bar


_structured_scan_coefficient_block_values.defvjp(
    _structured_scan_coefficient_block_values_fwd,
    _structured_scan_coefficient_block_values_bwd,
)


def solve_neopax_scan_coefficient_blocks_prepared_structured_vjp(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
) -> NeopaxScanCoefficientBlocks:
    """Prepared scan builder with an opt-in compact reverse rule."""

    surface_leaves, surface_tree = jax.tree_util.tree_flatten(surfaces)
    values = _structured_scan_coefficient_block_values(
        surface_tree, tuple(surface_leaves), Es, nu_v, grid
    )
    return NeopaxScanCoefficientBlocks(
        **dict(zip(_COEFFICIENT_BLOCK_NAMES, values, strict=True))
    )


__all__ = [
    "NeopaxScanCoefficientBlocks",
    "solve_neopax_scan_coefficient_blocks",
    "solve_neopax_scan_coefficient_blocks_prepared",
    "pullback_neopax_scan_coefficient_blocks_prepared",
    "solve_neopax_scan_coefficient_blocks_prepared_structured_vjp",
]
