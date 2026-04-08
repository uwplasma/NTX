"""Dense JAX block-tridiagonal monoenergetic DKE solver."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import lu_factor, lu_solve

from .config import enable_x64
from .geometry import BoozerSurface, VmecSurface, geometry_on_grid
from .grids import GridSpec
from .operators import (
    OperatorContext,
    apply_nullspace_condition,
    derivative_blocks,
    operator_blocks,
    source_modes,
)
from .transport import coefficients_from_modes, onsager_error


@dataclass(frozen=True)
class MonoenergeticCase:
    """Monoenergetic DKE parameters."""

    nu_hat: float
    epsi_hat: float | None = None
    er_hat: float | None = None

    def resolved_epsi_hat(self, psi_p: float | None) -> float:
        if self.epsi_hat is not None and self.er_hat is not None:
            msg = "set only one of epsi_hat or er_hat"
            raise ValueError(msg)
        if self.epsi_hat is not None:
            return float(self.epsi_hat)
        if self.er_hat is not None:
            if psi_p is None:
                msg = "er_hat requires a surface with psi_p; use epsi_hat for VMEC inputs"
                raise ValueError(msg)
            return float(self.er_hat) / float(psi_p)
        return 0.0


@dataclass(frozen=True)
class TransportResult:
    D11: Array
    D31: Array
    D13: Array
    D33: Array
    D33_spitzer: Array
    f1_modes: Array
    f3_modes: Array
    residual_l2: Array
    onsager_residual: Array

    def as_dict(self) -> dict[str, float]:
        return {
            "D11": float(self.D11),
            "D31": float(self.D31),
            "D13": float(self.D13),
            "D33": float(self.D33),
            "D33_spitzer": float(self.D33_spitzer),
            "residual_l2": float(self.residual_l2),
            "onsager_residual": float(self.onsager_residual),
        }


def solve_monoenergetic(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    case: MonoenergeticCase,
) -> TransportResult:
    """Solve one monoenergetic DKE case."""

    enable_x64(grid.x64)
    geom = geometry_on_grid(surface, grid)
    ctx = OperatorContext(
        surface=surface,
        geometry=geom,
        nu_hat=jnp.asarray(case.nu_hat, dtype=grid.jax_dtype),
        epsi_hat=jnp.asarray(case.resolved_epsi_hat(geom.psi_p), dtype=grid.jax_dtype),
    )
    d_theta, d_zeta = derivative_blocks(geom)
    s1, s3 = source_modes(ctx, grid.n_xi)
    f1_modes, f3_modes = _solve_modes(ctx, grid.n_xi, d_theta, d_zeta, s1, s3)
    d11, d31, d13, d33, d33_spitzer = coefficients_from_modes(
        geom, f1_modes, f3_modes, ctx.nu_hat
    )
    residual = _residual_norm(ctx, grid.n_xi, d_theta, d_zeta, s1, f1_modes)
    return TransportResult(
        D11=d11,
        D31=d31,
        D13=d13,
        D33=d33,
        D33_spitzer=d33_spitzer,
        f1_modes=f1_modes,
        f3_modes=f3_modes,
        residual_l2=residual,
        onsager_residual=onsager_error(d31, d13),
    )


def solve_scan(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    cases: tuple[MonoenergeticCase, ...],
) -> list[TransportResult]:
    """Solve a Python-level scan of monoenergetic cases."""

    return [solve_monoenergetic(surface, grid, case) for case in cases]


def solve_monoenergetic_scan(
    surface: BoozerSurface | VmecSurface,
    grid: GridSpec,
    nu_hat: Array,
    *,
    epsi_hat: Array | None = None,
    er_hat: Array | None = None,
) -> dict[str, Array]:
    """Vectorized scan over collisionality and radial electric field."""

    enable_x64(grid.x64)
    if epsi_hat is not None and er_hat is not None:
        msg = "set only one of epsi_hat or er_hat"
        raise ValueError(msg)
    geom = geometry_on_grid(surface, grid)
    d_theta, d_zeta = derivative_blocks(geom)
    nu_values = jnp.asarray(nu_hat, dtype=grid.jax_dtype)
    if epsi_hat is None:
        if er_hat is None:
            epsi_values = jnp.zeros_like(nu_values)
        else:
            if geom.psi_p is None:
                msg = "er_hat scans require a surface with psi_p; use epsi_hat for VMEC inputs"
                raise ValueError(msg)
            epsi_values = jnp.asarray(er_hat, dtype=grid.jax_dtype) / geom.psi_p
    else:
        epsi_values = jnp.asarray(epsi_hat, dtype=grid.jax_dtype)
    nu_values, epsi_values = jnp.broadcast_arrays(nu_values, epsi_values)
    output_shape = nu_values.shape

    def solve_one(nu_value, epsi_value):
        ctx = OperatorContext(
            surface=surface,
            geometry=geom,
            nu_hat=nu_value,
            epsi_hat=epsi_value,
        )
        s1, s3 = source_modes(ctx, grid.n_xi)
        f1_modes, f3_modes = _solve_modes(ctx, grid.n_xi, d_theta, d_zeta, s1, s3)
        return jnp.stack(coefficients_from_modes(geom, f1_modes, f3_modes, nu_value))

    coeffs = jax.vmap(solve_one)(nu_values.ravel(), epsi_values.ravel())
    coeffs = coeffs.reshape((*output_shape, 5))
    return {
        "D11": coeffs[..., 0],
        "D31": coeffs[..., 1],
        "D13": coeffs[..., 2],
        "D33": coeffs[..., 3],
        "D33_spitzer": coeffs[..., 4],
    }


def _solve_modes(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    s1: Array,
    s3: Array,
) -> tuple[Array, Array]:
    """Return source solutions for modes 0, 1, and 2."""

    lower_terminal, delta, lower_next = _terminal_delta(ctx, n_xi, d_theta, d_zeta)
    x = lu_solve(lu_factor(delta), lower_next)

    n_fs = delta.shape[0]
    saved_delta_init = jnp.zeros((3, n_fs, n_fs), dtype=delta.dtype)
    saved_lower_init = jnp.zeros((3, n_fs, n_fs), dtype=delta.dtype)
    saved_upper_init = jnp.zeros((3, n_fs, n_fs), dtype=delta.dtype)
    if n_xi == 2:
        saved_delta_init = saved_delta_init.at[2].set(delta)
        saved_lower_init = saved_lower_init.at[2].set(lower_terminal)

    def scan_step(carry, k):
        x_prev, saved_delta, saved_lower, saved_upper = carry
        lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)

        def fix_nullspace(args):
            diagonal_in, upper_in = args
            diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal_in, upper_in)
            assert upper_fixed is not None
            return diagonal_fixed, upper_fixed

        diagonal, upper = jax.lax.cond(
            k == 0,
            fix_nullspace,
            lambda args: args,
            (diagonal, upper),
        )
        delta_k = diagonal - upper @ x_prev

        def save_needed(args):
            saved_delta_in, saved_lower_in, saved_upper_in = args
            return (
                saved_delta_in.at[k].set(delta_k),
                saved_lower_in.at[k].set(lower),
                saved_upper_in.at[k].set(upper),
            )

        saved_delta, saved_lower, saved_upper = jax.lax.cond(
            k <= 2,
            save_needed,
            lambda args: args,
            (saved_delta, saved_lower, saved_upper),
        )
        x_next = jax.lax.cond(
            k > 0,
            lambda _: lu_solve(lu_factor(delta_k), lower),
            lambda _: x_prev,
            operand=None,
        )
        return (x_next, saved_delta, saved_lower, saved_upper), None

    ks = jnp.arange(n_xi - 1, -1, -1)
    (_, saved_delta, saved_lower, saved_upper), _ = jax.lax.scan(
        scan_step,
        (x, saved_delta_init, saved_lower_init, saved_upper_init),
        ks,
    )

    sigma1 = {
        2: s1[2],
        1: s1[1],
        0: s1[0],
    }
    sigma3 = {
        2: s3[2],
        1: s3[1],
        0: s3[0],
    }

    y1 = lu_solve(lu_factor(saved_delta[2]), sigma1[2])
    sigma1[1] = s1[1] - saved_upper[1] @ y1

    y1 = lu_solve(lu_factor(saved_delta[1]), sigma1[1])
    y3 = lu_solve(lu_factor(saved_delta[1]), sigma3[1])
    sigma1[0] = s1[0] - saved_upper[0] @ y1
    sigma3[0] = s3[0] - saved_upper[0] @ y3

    f1 = []
    f3 = []
    f1_0 = lu_solve(lu_factor(saved_delta[0]), sigma1[0])
    f3_0 = lu_solve(lu_factor(saved_delta[0]), sigma3[0])
    f1.append(f1_0)
    f3.append(f3_0)
    for k in (1, 2):
        lu = lu_factor(saved_delta[k])
        f1.append(lu_solve(lu, sigma1[k] - saved_lower[k] @ f1[k - 1]))
        f3.append(lu_solve(lu, sigma3[k] - saved_lower[k] @ f3[k - 1]))
    return jnp.stack(f1), jnp.stack(f3)


def _terminal_delta(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
) -> tuple[Array, Array, Array]:
    lower, diagonal, _ = operator_blocks(ctx, n_xi, d_theta, d_zeta)
    return lower, diagonal, lower


def _residual_norm(
    ctx: OperatorContext,
    n_xi: int,
    d_theta: Array,
    d_zeta: Array,
    source: Array,
    modes: Array,
) -> Array:
    """Residual norm for the solved low Legendre modes."""

    residuals = []
    for k in range(3):
        lower, diagonal, upper = operator_blocks(ctx, k, d_theta, d_zeta)
        if k == 0:
            diagonal_fixed, upper_fixed = apply_nullspace_condition(diagonal, upper)
            assert upper_fixed is not None
            diagonal = diagonal_fixed
            upper = upper_fixed
        value = diagonal @ modes[k] - source[k]
        if k > 0:
            value = value + lower @ modes[k - 1]
        if k < 2:
            value = value + upper @ modes[k + 1]
        residuals.append(value)
    residual = jnp.concatenate(residuals)
    _ = n_xi
    return jnp.linalg.norm(residual) / jnp.sqrt(residual.size)


jit_solve_monoenergetic = jax.jit(
    solve_monoenergetic,
    static_argnames=("surface", "grid", "case"),
)
