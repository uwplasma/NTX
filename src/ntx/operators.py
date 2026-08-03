"""Block operators for the Legendre-space monoenergetic DKE."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from .geometry import BoozerSurface, GeometryOnGrid, VmecSurface
from .grids import flatten_fs


@dataclass(frozen=True)
class OperatorContext:
    """Everything the block generator needs to build one row of the chain.

    Carries the surface, its geometry on the grid, and the two case parameters.
    Held as a frozen dataclass so it can be closed over by the row generator and
    still be replaced cheaply for a different collisionality.
    """
    surface: BoozerSurface | VmecSurface
    geometry: GeometryOnGrid
    nu_hat: Array
    epsi_hat: Array


def derivative_blocks(geom: GeometryOnGrid) -> tuple[Array, Array]:
    """Return lifted theta and zeta derivative matrices for flattened fields."""

    n_theta = geom.grid.theta.size
    n_zeta = geom.grid.zeta.size
    eye_theta = jnp.eye(n_theta, dtype=geom.b.dtype)
    eye_zeta = jnp.eye(n_zeta, dtype=geom.b.dtype)
    d_theta = jnp.kron(eye_zeta, geom.grid.dtheta_matrix)
    d_zeta = jnp.kron(geom.grid.dzeta_matrix, eye_theta)
    return d_theta, d_zeta


#: The geometry and physics arrays every block row is built from. Naming them
#: in one place is what lets the rows be generated from an explicit parameter
#: pytree, which in turn is what lets the reverse pass be bounded: see
#: :func:`block_parameters`.
BLOCK_PARAMETER_NAMES = (
    "b",
    "b_sup_theta",
    "b_sup_zeta",
    "d_b_dtheta",
    "d_b_dzeta",
    "b_sub_theta",
    "b_sub_zeta",
    "jacobian",
    "b2_mean",
    "nu_hat",
    "epsi_hat",
)


def block_parameters(ctx: OperatorContext) -> dict[str, Array]:
    """Extract the compact arrays the block rows are generated from.

    The Legendre chain has one row per mode, but every row is a formula in the
    same handful of surface quantities. Handing those to the solver explicitly,
    rather than closing over them, lets it differentiate with respect to them
    without recording the elimination -- the rows are regenerated on demand in
    both directions. The result is an ordinary pytree, so a cotangent for it
    chains back through the geometry to whatever produced it.
    """

    g = ctx.geometry
    return {
        "b": g.b,
        "b_sup_theta": g.b_sup_theta,
        "b_sup_zeta": g.b_sup_zeta,
        "d_b_dtheta": g.d_b_dtheta,
        "d_b_dzeta": g.d_b_dzeta,
        "b_sub_theta": g.b_sub_theta,
        "b_sub_zeta": g.b_sub_zeta,
        "jacobian": g.jacobian,
        "b2_mean": g.b2_mean,
        "nu_hat": ctx.nu_hat,
        "epsi_hat": ctx.epsi_hat,
    }


def coefficients_from_parameters(
    params: dict[str, Array], k: int | Array
) -> tuple[Array, Array, Array]:
    """Coefficient arrays for row ``k``, read from an explicit parameter set.

    This is the single implementation; :func:`coefficients_for_k` is the same
    thing sourced from an :class:`OperatorContext`.
    """

    b = params["b"]
    bu = params["b_sup_theta"]
    bv = params["b_sup_zeta"]
    bdot_grad_b = bv * params["d_b_dzeta"] + bu * params["d_b_dtheta"]

    kf = jnp.asarray(k, dtype=b.dtype)
    lower = {
        "theta": kf * bu / (b * (2.0 * kf - 1.0)),
        "zeta": kf * bv / (b * (2.0 * kf - 1.0)),
        "value": (kf * (kf - 1.0)) * bdot_grad_b / (2.0 * (2.0 * kf - 1.0) * b**2),
    }
    diagonal = {
        "theta": -params["epsi_hat"]
        * params["b_sub_zeta"]
        / (params["jacobian"] * params["b2_mean"]),
        "zeta": params["epsi_hat"]
        * params["b_sub_theta"]
        / (params["jacobian"] * params["b2_mean"]),
        "value": 0.5 * params["nu_hat"] * kf * (kf + 1.0) * jnp.ones_like(b),
    }
    upper = {
        "theta": (kf + 1.0) * bu / (b * (2.0 * kf + 3.0)),
        "zeta": (kf + 1.0) * bv / (b * (2.0 * kf + 3.0)),
        "value": -((kf + 1.0) * (kf + 2.0)) * bdot_grad_b / (
            2.0 * (2.0 * kf + 3.0) * b**2
        ),
    }
    return _pack(lower), _pack(diagonal), _pack(upper)


def coefficients_for_k(ctx: OperatorContext, k: int | Array) -> tuple[Array, Array, Array]:
    """Return coefficient arrays for lower, diagonal, and upper blocks."""

    return coefficients_from_parameters(block_parameters(ctx), k)


def _pack(coefficients: dict[str, Array]) -> Array:
    return jnp.stack(
        [
            flatten_fs(coefficients["theta"]),
            flatten_fs(coefficients["zeta"]),
            flatten_fs(coefficients["value"]),
        ]
    )


def build_block(coefficients: Array, d_theta: Array, d_zeta: Array) -> Array:
    """Construct a dense spatial block matrix."""

    c_theta, c_zeta, c_value = coefficients
    return c_theta[:, None] * d_theta + c_zeta[:, None] * d_zeta + jnp.diag(c_value)


def operator_blocks_from_parameters(
    params: dict[str, Array],
    k: int | Array,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array]:
    """Construct `(L_k, D_k, U_k)` from an explicit parameter set."""

    lower, diagonal, upper = coefficients_from_parameters(params, k)
    return (
        build_block(lower, d_theta, d_zeta),
        build_block(diagonal, d_theta, d_zeta),
        build_block(upper, d_theta, d_zeta),
    )


def operator_blocks(
    ctx: OperatorContext,
    k: int | Array,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array]:
    """Construct `(L_k, D_k, U_k)` dense blocks."""

    lower, diagonal, upper = coefficients_for_k(ctx, k)
    return (
        build_block(lower, d_theta, d_zeta),
        build_block(diagonal, d_theta, d_zeta),
        build_block(upper, d_theta, d_zeta),
    )


def parameter_derivative_blocks(
    ctx: OperatorContext,
    k: int | Array,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array]:
    """Construct dense `dD_k/dnu_hat` and `dD_k/depsi_hat` blocks."""

    g = ctx.geometry
    kf = jnp.asarray(k, dtype=g.b.dtype)
    zeros = jnp.zeros_like(g.b)
    nu_coefficients = _pack(
        {
            "theta": zeros,
            "zeta": zeros,
            "value": 0.5 * kf * (kf + 1.0) * jnp.ones_like(g.b),
        }
    )
    epsi_coefficients = _pack(
        {
            "theta": -g.b_sub_zeta / (g.jacobian * g.b2_mean),
            "zeta": g.b_sub_theta / (g.jacobian * g.b2_mean),
            "value": zeros,
        }
    )
    return (
        build_block(nu_coefficients, d_theta, d_zeta),
        build_block(epsi_coefficients, d_theta, d_zeta),
    )


def apply_nullspace_condition(
    d_block: Array,
    u_block: Array | None = None,
) -> tuple[Array, Array | None]:
    """Replace the first row by the `f^(0)(0,0)=0` constraint."""

    n = d_block.shape[0]
    row = jnp.zeros((n,), dtype=d_block.dtype).at[0].set(1.0)
    d_out = d_block.at[0, :].set(row)
    if u_block is None:
        return d_out, None
    return d_out, u_block.at[0, :].set(jnp.zeros((n,), dtype=u_block.dtype))


def source_modes(ctx: OperatorContext, n_xi: int) -> tuple[Array, Array]:
    """Build source arrays with shape `(n_xi + 1, n_fs)` for `s1` and `s3`."""

    b = ctx.geometry.b
    vm0 = ctx.geometry.radial_drift_spatial * (2.0 / 3.0)
    vm2 = ctx.geometry.radial_drift_spatial / 3.0
    n_fs = b.size
    s1 = jnp.zeros((n_xi + 1, n_fs), dtype=b.dtype)
    s3 = jnp.zeros((n_xi + 1, n_fs), dtype=b.dtype)
    s1 = s1.at[0].set(-flatten_fs(vm0))
    s1 = s1.at[2].set(-flatten_fs(vm2))
    s1 = s1.at[0, 0].set(0.0)
    s3 = s3.at[1].set(flatten_fs(b))
    return s1, s3
