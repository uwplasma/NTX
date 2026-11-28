"""Recorded scan lifetimes and bounded transposes across surfaces and scan points."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, cast, overload

import jax
import jax.numpy as jnp
from jax import Array

from .._geometry import BoozerSurface, VmecSurface, geometry_on_grid
from .._solver import MonoenergeticCase, PreparedMonoenergeticSystem, prepare_monoenergetic_system
from .._solver_scan import _resolved_scan_inputs, _scan_coefficients_serial
from ..grids import GridSpec
from .adjoints import (
    pullback_prepared_coefficient_vector_case_and_prepared,
    pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs,
)

if TYPE_CHECKING:
    from .._neopax_scan import NeopaxScanCoefficientBlocks


@dataclass(frozen=True)
class NeopaxScanCoefficientPrimalRecord:
    """Reusable primal state for a live structured NEOPAX scan transpose.

    The coefficient tables remain in the database.  This record retains only
    the prepared NTX systems that produced those tables, plus their matching
    scan inputs.  It is intentionally opt-in because prepared systems can be
    materially larger than the interpolated database itself.
    """

    surfaces: tuple[BoozerSurface | VmecSurface, ...]
    prepared: tuple[PreparedMonoenergeticSystem, ...]
    Es: Array
    nu_v: Array
    grid: GridSpec


_COEFFICIENT_CHANNELS = {"D11": 0, "D13": 2, "D33": 3, "D33_spitzer": 4}


_REFERENCE_BLOCKS = (
    ("b00", "b00"),
    ("boozer_i", "boozer_i"),
    ("boozer_g", "boozer_g"),
    ("iota", "iota"),
    ("fac_11", "fac_reference_to_sfincs_11"),
    ("fac_31", "fac_reference_to_sfincs_31"),
    ("fac_33", "fac_reference_to_sfincs_33"),
    ("fac_sfincs_to_dkes_11", "fac_sfincs_to_dkes_11"),
    ("fac_sfincs_to_dkes_31", "fac_sfincs_to_dkes_31"),
    ("fac_sfincs_to_dkes_33", "fac_sfincs_to_dkes_33"),
)


_COEFFICIENT_BLOCK_NAMES = (*_COEFFICIENT_CHANNELS, *(field for _, field in _REFERENCE_BLOCKS))


def _reference_bar_from_values(values):
    """Match reference names to the fixed scan cotangent field order."""
    return {name: values[i] for i, (name, _) in enumerate(_REFERENCE_BLOCKS)}


def _reference_bar_from_blocks(blocks, index):
    """Select one scan point without mixing coefficient and reference bars."""
    return {name: getattr(blocks, field)[index] for name, field in _REFERENCE_BLOCKS}


def _coefficient_bars_from_blocks(blocks, index):
    """Embed database channels in the five-coefficient solver cotangent order."""
    d11 = blocks.D11[index]
    return jnp.stack(
        (d11, jnp.zeros_like(d11), blocks.D13[index], blocks.D33[index], blocks.D33_spitzer[index]),
        axis=-1,
    )


def _zero_float_tree(tree):
    """Zeros compatible with a prepared-system PyTree, including static leaves."""

    def _zero(leaf):
        value = jnp.asarray(leaf)
        dtype = value.dtype if jnp.issubdtype(value.dtype, jnp.inexact) else jnp.float64
        return jnp.zeros(value.shape, dtype=dtype)

    return jax.tree_util.tree_map(_zero, tree)


def _add_float_trees(left, right):
    """Add prepared-system bars while preserving JAX's float0 static leaves."""

    def _add(left_leaf, right_leaf):
        left_value = jnp.asarray(left_leaf)
        right_value = jnp.asarray(right_leaf)
        if right_value.dtype == jax.dtypes.float0:
            return left_leaf
        if left_value.dtype == jax.dtypes.float0:
            return right_leaf
        return left_leaf + right_leaf

    return jax.tree_util.tree_map(_add, left, right)


@jax.jit
def _prepared_scan_coefficient_bar_kernel(
    prepared,
    nu_values: Array,
    epsi_values: Array,
    coefficient_bars: Array,
):
    """One reusable, bounded NTX coefficient-adjoint scan.

    Keeping this as a JIT call rather than expanding it in the outer database
    VJP is essential: the database has several surfaces, but every surface
    uses the same scan shape.  XLA can then compile this kernel once and
    execute it for each prepared surface.
    """

    def _accumulate(prepared_bar, values):
        nu_value, epsi_value, coefficient_bar = values
        case_bar, local_prepared_bar = pullback_prepared_coefficient_vector_case_and_prepared(
            prepared,
            MonoenergeticCase(nu_hat=nu_value, epsi_hat=epsi_value),
            coefficient_bar,
        )
        return _add_float_trees(prepared_bar, local_prepared_bar), case_bar.epsi_hat

    return jax.lax.scan(
        _accumulate,
        _zero_float_tree(prepared),
        (nu_values, epsi_values, coefficient_bars),
    )


@partial(jax.jit, static_argnums=(2,), inline=False)
def _prepared_scan_coefficient_bar_rows_to_numeric_surface_leaves_kernel(
    prepared,
    surface,
    grid: GridSpec,
    nu_values: Array,
    epsi_values: Array,
    coefficient_bars: Array,
    reference_bars: Array,
):
    """Run a compiled RHS scan while keeping VMEC metadata out of its output.

    The output contains only numerical leaves.  Reconstructing a surface
    cotangent inside ``vmap``/``scan`` would expose static mode metadata to
    JAX's tangent machinery, so the caller restores the original surface tree
    only after this compiled scan completes.
    """
    from .._neopax import _surface_reference_bridge

    _, prepare_pullback = jax.vjp(
        lambda surface_value: prepare_monoenergetic_system(surface_value, grid),
        surface,
    )
    _, reference_pullback = jax.vjp(_surface_reference_bridge, surface)

    def _one_row(values):
        coefficient_bar, reference_values = values
        prepared_bar, epsi_flat_bar = _prepared_scan_coefficient_bar_kernel(
            prepared, nu_values, epsi_values, coefficient_bar
        )
        prepared_surface_bar = prepare_pullback(prepared_bar)[0]
        reference_bar = _reference_bar_from_values(reference_values)
        surface_bar = _add_float_trees(prepared_surface_bar, reference_pullback(reference_bar)[0])
        return tuple(jax.tree_util.tree_leaves(surface_bar)), epsi_flat_bar

    _, (surface_leaf_rows, epsi_flat_rows) = jax.lax.scan(
        lambda carry, values: (carry, _one_row(values)),
        None,
        (coefficient_bars, reference_bars),
    )
    return surface_leaf_rows, epsi_flat_rows


@partial(jax.jit, static_argnums=(2,), inline=False)
def _prepared_scan_coefficient_bar_rows_to_vmec_surface_leaves_native_kernel(
    prepared,
    surface: VmecSurface,
    grid: GridSpec,
    nu_values: Array,
    epsi_values: Array,
    coefficient_bars: Array,
    reference_bars: Array,
):
    """Database-only matrix-RHS scan transpose directly to VMEC leaves.

    Unlike the established generic row kernel above, this never materialises
    a RHS-batched ``PreparedMonoenergeticSystem`` cotangent.  It is used only
    by the runtime-database boundary; the Lij and generic scan paths retain
    their existing prepared-system pullbacks.
    """
    from .._neopax import _surface_reference_bridge

    rhs_count = coefficient_bars.shape[0]
    geometry_bars0 = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (rhs_count, *jnp.asarray(value).shape)),
        _zero_float_tree(prepared.geometry),
    )

    def _accumulate(geometry_bars, values):
        nu_value, epsi_value, coefficient_bar = values
        case_bars, local_geometry_bars = (
            pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs(
                prepared,
                MonoenergeticCase(nu_hat=nu_value, epsi_hat=epsi_value),
                coefficient_bar,
            )
        )
        return (
            _add_float_trees(geometry_bars, local_geometry_bars),
            case_bars.epsi_hat,
        )

    geometry_bars, epsi_case_bars = jax.lax.scan(
        _accumulate,
        geometry_bars0,
        (nu_values, epsi_values, jnp.moveaxis(coefficient_bars, 0, 1)),
    )
    _, reference_pullback = jax.vjp(_surface_reference_bridge, surface)
    _, geometry_pullback = jax.vjp(
        lambda surface_value: geometry_on_grid(surface_value, grid), surface
    )

    def _surface_leaves(values):
        reference_values, local_geometry_bars = values
        reference_bar = _reference_bar_from_values(reference_values)
        surface_bar = _add_float_trees(
            geometry_pullback(local_geometry_bars)[0],
            reference_pullback(reference_bar)[0],
        )
        return tuple(jax.tree_util.tree_leaves(surface_bar))

    _, surface_leaf_rows = jax.lax.scan(
        lambda carry, values: (carry, _surface_leaves(values)),
        None,
        (reference_bars, geometry_bars),
    )
    return surface_leaf_rows, jnp.moveaxis(epsi_case_bars, 0, 1)


@overload
def solve_neopax_scan_coefficient_blocks_prepared(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
    return_primal_record: Literal[False] = False,
) -> NeopaxScanCoefficientBlocks:
    """Describe coefficient-only callers that do not request reusable primal records."""
    ...


@overload
def solve_neopax_scan_coefficient_blocks_prepared(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
    return_primal_record: Literal[True],
) -> tuple[NeopaxScanCoefficientBlocks, NeopaxScanCoefficientPrimalRecord]:
    """Describe callers that retain primal records for a later scan transpose."""
    ...


@overload
def solve_neopax_scan_coefficient_blocks_prepared(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
    return_primal_record: bool,
) -> (
    NeopaxScanCoefficientBlocks
    | tuple[NeopaxScanCoefficientBlocks, NeopaxScanCoefficientPrimalRecord]
):
    """Cover callers whose primal-record flag is not a literal."""
    ...


def solve_neopax_scan_coefficient_blocks_prepared(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
    return_primal_record: bool = False,
) -> (
    NeopaxScanCoefficientBlocks
    | tuple[NeopaxScanCoefficientBlocks, NeopaxScanCoefficientPrimalRecord]
):
    """Build the same scan tables as the generic builder, retaining per-surface
    prepared systems only when ``return_primal_record`` is requested.
    """
    from .._neopax import _surface_reference_bridge
    from .._neopax_scan import NeopaxScanCoefficientBlocks

    values: dict[str, list[Array]] = {name: [] for name in _COEFFICIENT_BLOCK_NAMES}
    prepared_records = []
    for surface, es_row in zip(surfaces, Es, strict=True):
        prepared = prepare_monoenergetic_system(surface, grid)
        if return_primal_record:
            prepared_records.append(prepared)
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
        for name, channel in _COEFFICIENT_CHANNELS.items():
            values[name].append(coefficients[..., channel])
        bridge = _surface_reference_bridge(surface)
        for name, field in _REFERENCE_BLOCKS:
            values[field].append(bridge[name])

    blocks = NeopaxScanCoefficientBlocks(
        **{
            name: jnp.stack(rows) if name in _COEFFICIENT_CHANNELS else jnp.asarray(rows)
            for name, rows in values.items()
        }
    )
    if not return_primal_record:
        return blocks
    return blocks, NeopaxScanCoefficientPrimalRecord(
        surfaces=tuple(surfaces),
        prepared=tuple(prepared_records),
        Es=jnp.asarray(Es),
        nu_v=jnp.asarray(nu_v),
        grid=grid,
    )


def pullback_neopax_scan_coefficient_blocks_prepared(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
    coefficient_blocks_bar: NeopaxScanCoefficientBlocks,
    prepared_systems: tuple[PreparedMonoenergeticSystem, ...] | None = None,
) -> tuple[tuple[BoozerSurface | VmecSurface, ...], Array]:
    """Transpose coefficient blocks through one prepared system per surface.

    Accumulate all coefficient-case bars before pulling them through surface
    preparation. Reference-normalization fields use their ordinary JAX VJP.
    ``prepared_systems`` optionally reuses the systems saved by the primal.
    """
    from .._neopax import _surface_reference_bridge

    surface_bars = []
    es_bars = []
    if prepared_systems is not None and len(prepared_systems) != len(surfaces):
        raise ValueError("prepared_systems must contain one entry per scan surface")

    for surface_index, (surface, es_row) in enumerate(zip(surfaces, Es, strict=True)):
        prepared = (
            prepare_monoenergetic_system(surface, grid)
            if prepared_systems is None
            else prepared_systems[surface_index]
        )
        nu_grid, es_grid = jnp.meshgrid(nu_v, es_row, indexing="ij")
        nu_values, epsi_values, _output_shape = _resolved_scan_inputs(
            prepared,
            grid,
            nu_grid,
            es_grid,
            None,
        )
        coefficient_bars = _coefficient_bars_from_blocks(
            coefficient_blocks_bar, surface_index
        ).reshape((-1, 5))

        prepared_bar, epsi_flat_bar = _prepared_scan_coefficient_bar_kernel(
            prepared,
            nu_values.reshape((-1,)),
            epsi_values.reshape((-1,)),
            coefficient_bars,
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
        reference_bar = _reference_bar_from_blocks(coefficient_blocks_bar, surface_index)
        _, reference_pullback = jax.vjp(_surface_reference_bridge, surface)
        reference_surface_bar = reference_pullback(reference_bar)[0]
        surface_bars.append(_add_float_trees(surface_bar, reference_surface_bar))

    return tuple(surface_bars), jnp.stack(es_bars)


def pullback_neopax_scan_coefficient_blocks_from_primal_record(
    record: NeopaxScanCoefficientPrimalRecord,
    *,
    coefficient_blocks_bar: NeopaxScanCoefficientBlocks,
) -> tuple[tuple[BoozerSurface | VmecSurface, ...], Array]:
    """Structured scan transpose reusing the prepared systems from its primal."""

    return pullback_neopax_scan_coefficient_blocks_prepared(
        record.surfaces,
        Es=record.Es,
        nu_v=record.nu_v,
        grid=record.grid,
        coefficient_blocks_bar=coefficient_blocks_bar,
        prepared_systems=record.prepared,
    )


def pullback_neopax_scan_coefficient_blocks_from_primal_record_batched_vmec_native(
    record: NeopaxScanCoefficientPrimalRecord,
    *,
    coefficient_blocks_bar: NeopaxScanCoefficientBlocks,
) -> tuple[tuple[VmecSurface, ...], Array]:
    """Transpose recorded VMEC table bars using compact matrix-RHS geometry bars."""

    return cast(
        tuple[tuple[VmecSurface, ...], Array],
        _pullback_recorded_scan_batched(record, coefficient_blocks_bar, vmec_native=True),
    )


def _pullback_recorded_scan_batched(record, coefficient_blocks_bar, *, vmec_native):
    """Share surface traversal; retain the distinct row-scan and native RHS kernels."""
    rhs_count = int(coefficient_blocks_bar.D11.shape[0])
    if rhs_count < 1:
        raise ValueError("Batched scan coefficient bars must contain at least one RHS row.")
    if vmec_native and not all(isinstance(surface, VmecSurface) for surface in record.surfaces):
        raise ValueError("The compact database recorded transpose requires VMEC scan surfaces.")
    kernel = (
        _prepared_scan_coefficient_bar_rows_to_vmec_surface_leaves_native_kernel
        if vmec_native
        else _prepared_scan_coefficient_bar_rows_to_numeric_surface_leaves_kernel
    )
    surface_bars, es_bars = [], []
    for i, (surface, prepared, es_row) in enumerate(
        zip(record.surfaces, record.prepared, record.Es, strict=True)
    ):
        nu_grid, es_grid = jnp.meshgrid(record.nu_v, es_row, indexing="ij")
        nu_values, epsi_values, _ = _resolved_scan_inputs(
            prepared, record.grid, nu_grid, es_grid, None
        )
        coefficient_bars = _coefficient_bars_from_blocks(
            coefficient_blocks_bar, (slice(None), i)
        ).reshape((rhs_count, -1, 5))
        reference_bars = _reference_bar_from_blocks(coefficient_blocks_bar, (slice(None), i))
        reference_rows = jnp.stack(tuple(reference_bars.values()), axis=1)
        surface_leaves, epsi_rows = kernel(
            prepared,
            surface,
            record.grid,
            nu_values.reshape(-1),
            epsi_values.reshape(-1),
            coefficient_bars,
            reference_rows,
        )
        es_bars.append(jnp.sum(epsi_rows.reshape((rhs_count, *es_grid.shape)), axis=1))
        surface_bars.append(
            jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(surface), surface_leaves)
        )
    return tuple(surface_bars), jnp.stack(es_bars, axis=1)


def _coefficient_block_values(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
) -> tuple[Array, ...]:
    """Expose numeric scan leaves at the structured custom-VJP boundary."""
    blocks = solve_neopax_scan_coefficient_blocks_prepared(surfaces, Es=Es, nu_v=nu_v, grid=grid)
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
    """Save prepared scan systems for reuse by the structured reverse rule."""
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
    """Transpose saved systems while keeping discrete scan grids nondifferentiable."""
    from .._neopax_scan import NeopaxScanCoefficientBlocks

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
    from .._neopax_scan import NeopaxScanCoefficientBlocks

    surface_leaves, surface_tree = jax.tree_util.tree_flatten(surfaces)
    values = _structured_scan_coefficient_block_values(
        surface_tree, tuple(surface_leaves), Es, nu_v, grid
    )
    return NeopaxScanCoefficientBlocks(**dict(zip(_COEFFICIENT_BLOCK_NAMES, values, strict=True)))
