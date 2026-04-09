"""In-memory monoenergetic scan builders for higher-level JAX workflows."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, tree_util

from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .solver import solve_monoenergetic_scan


@dataclass(frozen=True)
class MonoenergeticDatabaseArrays:
    """In-memory monoenergetic transport arrays.

    This container is intended for imported JAX workflows that need NTX scan
    data as arrays rather than through files or the terminal CLI.
    """

    nu_hat: Array
    scan_field: Array
    scan_field_name: str
    D11: Array
    D31: Array
    D13: Array
    D33: Array
    D33_spitzer: Array
    rho: Array | None = None


tree_util.register_dataclass(
    MonoenergeticDatabaseArrays,
    data_fields=(
        "nu_hat",
        "scan_field",
        "D11",
        "D31",
        "D13",
        "D33",
        "D33_spitzer",
        "rho",
    ),
    meta_fields=("scan_field_name",),
)


def build_monoenergetic_database_arrays(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    nu_hat: Array,
    *,
    epsi_hat: Array | None = None,
    er_hat: Array | None = None,
    rho: Array | None = None,
) -> MonoenergeticDatabaseArrays:
    """Build in-memory monoenergetic scan arrays for one surface."""
    if epsi_hat is not None and er_hat is not None:
        msg = "set only one of epsi_hat or er_hat"
        raise ValueError(msg)
    nu_values = jnp.atleast_1d(jnp.asarray(nu_hat))
    if epsi_hat is not None:
        scan_field = jnp.atleast_1d(jnp.asarray(epsi_hat))
        scan_field_name = "epsi_hat"
        nu_grid, field_grid = jnp.meshgrid(nu_values, scan_field, indexing="ij")
        coeffs = solve_monoenergetic_scan(
            surface,
            grid,
            nu_grid,
            epsi_hat=field_grid,
        )
    elif er_hat is not None:
        scan_field = jnp.atleast_1d(jnp.asarray(er_hat))
        scan_field_name = "er_hat"
        nu_grid, field_grid = jnp.meshgrid(nu_values, scan_field, indexing="ij")
        coeffs = solve_monoenergetic_scan(
            surface,
            grid,
            nu_grid,
            er_hat=field_grid,
        )
    else:
        scan_field = jnp.asarray([0.0])
        scan_field_name = "epsi_hat"
        nu_grid, field_grid = jnp.meshgrid(nu_values, scan_field, indexing="ij")
        coeffs = solve_monoenergetic_scan(
            surface,
            grid,
            nu_grid,
            epsi_hat=field_grid,
        )
    return MonoenergeticDatabaseArrays(
        nu_hat=nu_values,
        scan_field=scan_field,
        scan_field_name=scan_field_name,
        D11=coeffs["D11"],
        D31=coeffs["D31"],
        D13=coeffs["D13"],
        D33=coeffs["D33"],
        D33_spitzer=coeffs["D33_spitzer"],
        rho=None if rho is None else jnp.asarray(rho),
    )


def stack_monoenergetic_database_arrays(
    databases: tuple[MonoenergeticDatabaseArrays, ...],
) -> MonoenergeticDatabaseArrays:
    """Stack per-surface scan arrays into a leading surface axis."""

    if not databases:
        raise ValueError("expected at least one database to stack")
    first = databases[0]
    if any(db.scan_field_name != first.scan_field_name for db in databases[1:]):
        raise ValueError("all databases must use the same scan_field_name")
    stacked_rho = None
    if all(db.rho is not None for db in databases):
        stacked_rho = jnp.stack([jnp.asarray(db.rho) for db in databases])
    return MonoenergeticDatabaseArrays(
        nu_hat=jnp.stack([db.nu_hat for db in databases]),
        scan_field=jnp.stack([db.scan_field for db in databases]),
        scan_field_name=first.scan_field_name,
        D11=jnp.stack([db.D11 for db in databases]),
        D31=jnp.stack([db.D31 for db in databases]),
        D13=jnp.stack([db.D13 for db in databases]),
        D33=jnp.stack([db.D33 for db in databases]),
        D33_spitzer=jnp.stack([db.D33_spitzer for db in databases]),
        rho=stacked_rho,
    )
