"""Prepared-solver adjoint and custom-VJP helper algebra."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ._solver_context import _operator_context
from ._solver_factorization import (
    _factorize_prepared_modes,
    _solve_factorized_modes,
)
from ._solver_types import PreparedMonoenergeticSystem
from ._geometry_types import VmecSurface
from ._geometry_eval import vmec_geometry_bars_to_coefficients_multi_rhs
from .grids import flatten_fs
from .operators import (
    OperatorContext,
    apply_nullspace_condition,
    block_parameters,
    operator_blocks,
    operator_blocks_from_parameters,
    parameter_derivative_blocks,
    source_modes,
)
from .transport import coefficients_from_modes


def _prepared_implicit_vjp_primal(
    prepared: PreparedMonoenergeticSystem,
    nu_hat,
    epsi_hat,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    geom = prepared.geometry
    grid = prepared.grid
    ctx = _operator_context(prepared.surface, geom, grid, nu_hat, epsi_hat)
    s1, s3 = source_modes(ctx, grid.n_xi)
    saved_lu, saved_piv, saved_lower, saved_upper = _factorize_prepared_modes(
        ctx,
        grid.n_xi,
        prepared.d_theta,
        prepared.d_zeta,
    )
    f1_full = _solve_factorized_modes(saved_lu, saved_piv, saved_lower, saved_upper, s1)
    f3_full = _solve_factorized_modes(saved_lu, saved_piv, saved_lower, saved_upper, s3)

    def coefficient_fn(modes1, modes3, nu_value):
        return jnp.stack(coefficients_from_modes(geom, modes1, modes3, nu_value))

    coefficients = coefficient_fn(f1_full[:3], f3_full[:3], ctx.nu_hat)
    return coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper


def _coefficient_mode_pullback(
    geom,
    f1_low: Array,
    f3_low: Array,
    nu_hat: Array,
    coefficient_bar: Array,
) -> tuple[Array, Array, Array]:
    def coefficient_fn(modes1, modes3, nu_value):
        return jnp.stack(coefficients_from_modes(geom, modes1, modes3, nu_value))

    _, pullback = jax.vjp(coefficient_fn, f1_low, f3_low, nu_hat)
    f1_bar, f3_bar, nu_bar = pullback(coefficient_bar)
    return f1_bar, f3_bar, nu_bar


def _coefficient_mode_pullback_multi_rhs_direct(
    geom,
    nu_hat: Array,
    coefficient_bars: Array,
) -> tuple[Array, Array, Array]:
    """Exact RHS-batched transpose of :func:`coefficients_from_modes`.

    This is deliberately below the implicit NTX adjoint.  The five transport
    coefficients are affine in the retained ``f1``/``f3`` modes, and only the
    Spitzer coefficient depends on ``nu_hat``.  Keeping the RHS axis explicit
    avoids staging ``vmap(_coefficient_mode_pullback)`` and, more
    importantly, makes its two low-dot directional tangents available without
    a JVP through a generic VJP.

    The returned mode bars have shape ``(n_rhs, 3, n_fs)`` and the scalar
    ``nu_hat`` bar has shape ``(n_rhs,)``.
    """
    coefficient_bars = jnp.asarray(coefficient_bars)
    if coefficient_bars.ndim != 2 or coefficient_bars.shape[1] != 5:
        raise ValueError("coefficient_bars must have shape (n_rhs, 5).")

    rhs_count = coefficient_bars.shape[0]
    dtype = coefficient_bars.dtype
    psi_scale = jnp.asarray(geom.coefficient_psi_scale, dtype=dtype)
    b0 = jnp.asarray(geom.b0, dtype=dtype)
    weight = (
        jnp.asarray(geom.jacobian, dtype=dtype)
        / jnp.asarray(geom.volume_prime, dtype=dtype)
        * jnp.asarray(geom.grid.dtheta * geom.grid.dzeta, dtype=dtype)
    )
    drift = jnp.asarray(geom.radial_drift_spatial, dtype=dtype)
    b = jnp.asarray(geom.b, dtype=dtype)

    def _flat(fields):
        return jax.vmap(flatten_fs)(fields)

    q0, q1, q2, q3, q4 = (
        coefficient_bars[:, index, None, None] for index in range(5)
    )
    zeros = jnp.zeros((rhs_count,) + b.shape, dtype=dtype)
    drift_weight = weight * drift
    f1_bar = jnp.stack(
        (
            _flat(q0 * (-4.0 / 3.0) * drift_weight / psi_scale**2),
            _flat(zeros),
            _flat(q0 * (-2.0 / 15.0) * drift_weight / psi_scale**2),
        ),
        axis=1,
    )
    f3_bar = jnp.stack(
        (
            _flat(q2 * (-4.0 / 3.0) * drift_weight / (psi_scale * b0)),
            _flat(
                q1 * weight * (2.0 * b / (3.0 * b0 * psi_scale))
                + q3 * weight * (2.0 * b / (3.0 * b0**2))
            ),
            _flat(q2 * (-2.0 / 15.0) * drift_weight / (psi_scale * b0)),
        ),
        axis=1,
    )
    spitzer_without_nu = jnp.sum(weight * 2.0 * b**2 / (3.0 * b0**2))
    nu_bar = -coefficient_bars[:, 4] * spitzer_without_nu / nu_hat**2
    return f1_bar, f3_bar, nu_bar


def _directional_coefficient_mode_pullback_multi_rhs_direct(
    geom,
    nu_hat: Array,
    nu_hat_dot: Array,
    coefficient_bars: Array,
    coefficient_bars_dot: Array,
) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array]]:
    """Exact low-dot tangent of the direct RHS-batched coefficient transpose.

    ``coefficients_from_modes`` is linear in the retained modes, so the
    coefficient transpose itself has no mode tangent.  Its only physical
    tangent is the explicit ``nu_hat**-1`` Spitzer term.  This is therefore
    algebraically the same result as the existing JVP of
    ``_coefficient_mode_pullback`` but contains no nested AD transform.
    """
    base = _coefficient_mode_pullback_multi_rhs_direct(
        geom, nu_hat, coefficient_bars
    )
    bar_tangent = _coefficient_mode_pullback_multi_rhs_direct(
        geom, nu_hat, coefficient_bars_dot
    )
    f1_dot, f3_dot, nu_bar_from_bar_dot = bar_tangent
    spitzer_without_nu = jnp.sum(
        (geom.jacobian / geom.volume_prime)
        * (geom.grid.dtheta * geom.grid.dzeta)
        * 2.0 * geom.b**2
        / (3.0 * geom.b0**2)
    )
    nu_dot = nu_bar_from_bar_dot + (
        2.0
        * coefficient_bars[:, 4]
        * spitzer_without_nu
        * nu_hat_dot
        / nu_hat**3
    )
    return base, (f1_dot, f3_dot, nu_dot)


def _parameter_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
) -> tuple[Array, Array]:
    def zero_first_row(block: Array) -> Array:
        return block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))

    nu_bar = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    epsi_bar = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = zero_first_row(diagonal_nu)
            diagonal_epsi = zero_first_row(diagonal_epsi)
        nu_bar = nu_bar - (
            jnp.vdot(lambda1[k], diagonal_nu @ f1_full[k])
            + jnp.vdot(lambda3[k], diagonal_nu @ f3_full[k])
        )
        epsi_bar = epsi_bar - (
            jnp.vdot(lambda1[k], diagonal_epsi @ f1_full[k])
            + jnp.vdot(lambda3[k], diagonal_epsi @ f3_full[k])
        )
    return nu_bar, epsi_bar


def _parameter_gradient_from_adjoint_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
) -> tuple[Array, Array]:
    """Case bars for a trailing batch of already-solved adjoint fields.

    This is the matrix-RHS counterpart of
    :func:`_parameter_gradient_from_adjoint`.  It only contracts existing
    primal and adjoint fields; it neither factorizes nor solves.  Keeping the
    RHS column explicit avoids ``vmap`` tracing one complete parameter-JVP
    graph per objective in the experimental native support path.
    """
    if lambda1.ndim != 3 or lambda3.shape != lambda1.shape:
        raise ValueError(
            "lambda1 and lambda3 must have shape (mode, unknown, n_rhs)."
        )

    def zero_first_row(block: Array) -> Array:
        return block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))

    n_rhs = lambda1.shape[-1]
    nu_bar = jnp.zeros((n_rhs,), dtype=prepared.grid.jax_dtype)
    epsi_bar = jnp.zeros((n_rhs,), dtype=prepared.grid.jax_dtype)
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = zero_first_row(diagonal_nu)
            diagonal_epsi = zero_first_row(diagonal_epsi)
        f1_nu = diagonal_nu @ f1_full[k]
        f3_nu = diagonal_nu @ f3_full[k]
        f1_epsi = diagonal_epsi @ f1_full[k]
        f3_epsi = diagonal_epsi @ f3_full[k]
        nu_bar = nu_bar - (
            jnp.sum(lambda1[k] * f1_nu[:, None], axis=0)
            + jnp.sum(lambda3[k] * f3_nu[:, None], axis=0)
        )
        epsi_bar = epsi_bar - (
            jnp.sum(lambda1[k] * f1_epsi[:, None], axis=0)
            + jnp.sum(lambda3[k] * f3_epsi[:, None], axis=0)
        )
    return nu_bar, epsi_bar


def _fixed_residual_block_coefficient_bars_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
) -> tuple[Array, Array, Array]:
    """Transpose the fixed block residual directly along its RHS axis.

    The return values are cotangents for the lower, diagonal and upper packed
    coefficient arrays, with shape ``(mode, rhs, 3, n_fs)``.  They are the
    exact contribution of ``-lambda.T @ A @ f`` before the coefficient arrays
    are chained to ``block_parameters``.  No factorisation, solve, generic VJP
    or objective ``vmap`` occurs here.

    This is deliberately a small, independently testable first part of the
    native prepared-support transpose.  Source and direct-coefficient terms
    have different geometry dependencies and are added by the later stages.
    """
    if lambda1.ndim != 3 or lambda3.shape != lambda1.shape:
        raise ValueError(
            "lambda1 and lambda3 must have shape (mode, unknown, n_rhs)."
        )
    if f1_full.shape != f3_full.shape or f1_full.ndim != 2:
        raise ValueError("primal mode fields must have shape (mode, unknown).")

    n_modes, n_fs = f1_full.shape
    if lambda1.shape[:2] != (n_modes, n_fs):
        raise ValueError("primal and adjoint mode dimensions must agree.")

    # The fixed residual pullback is called with residual cotangents
    # ``(-lambda1, -lambda3)``.  Keep that sign explicit so this helper can be
    # compared directly against the established compact residual VJP.
    residual_bar1 = -lambda1
    residual_bar3 = -lambda3

    def _coefficient_bar(residual_bar: Array, mode_values: Array) -> Array:
        """Return the packed block-coefficient bar for one ``A @ f`` term."""
        theta_value = prepared.d_theta @ mode_values
        zeta_value = prepared.d_zeta @ mode_values
        # ``residual_bar`` is (n_fs, rhs); make the RHS axis leading once and
        # contract it with the three pointwise block contributions.
        bar = jnp.swapaxes(residual_bar, 0, 1)
        return jnp.stack(
            (
                bar * theta_value[None, :],
                bar * zeta_value[None, :],
                bar * mode_values[None, :],
            ),
            axis=1,
        )

    lower_bars = []
    diagonal_bars = []
    upper_bars = []
    for mode_index in range(n_modes):
        diagonal_bar1 = residual_bar1[mode_index]
        diagonal_bar3 = residual_bar3[mode_index]
        upper_bar1 = residual_bar1[mode_index]
        upper_bar3 = residual_bar3[mode_index]
        if mode_index == 0:
            # ``apply_nullspace_condition`` replaces the diagonal first row
            # and zeros the upper first row.  The original coefficient arrays
            # therefore receive no cotangent from that row.
            diagonal_bar1 = diagonal_bar1.at[0].set(0)
            diagonal_bar3 = diagonal_bar3.at[0].set(0)
            upper_bar1 = upper_bar1.at[0].set(0)
            upper_bar3 = upper_bar3.at[0].set(0)

        diagonal_bars.append(
            _coefficient_bar(diagonal_bar1, f1_full[mode_index])
            + _coefficient_bar(diagonal_bar3, f3_full[mode_index])
        )
        if mode_index > 0:
            lower_bars.append(
                _coefficient_bar(residual_bar1[mode_index], f1_full[mode_index - 1])
                + _coefficient_bar(residual_bar3[mode_index], f3_full[mode_index - 1])
            )
        else:
            lower_bars.append(
                jnp.zeros_like(diagonal_bars[-1])
            )
        if mode_index < n_modes - 1:
            upper_bars.append(
                _coefficient_bar(upper_bar1, f1_full[mode_index + 1])
                + _coefficient_bar(upper_bar3, f3_full[mode_index + 1])
            )
        else:
            upper_bars.append(jnp.zeros_like(diagonal_bars[-1]))

    return jnp.stack(lower_bars), jnp.stack(diagonal_bars), jnp.stack(upper_bars)


def _block_parameters_bar_from_coefficient_bars_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    params: dict[str, Array],
    lower_bars: Array,
    diagonal_bars: Array,
    upper_bars: Array,
) -> dict[str, Array]:
    """Direct RHS-axis transpose of :func:`coefficients_from_parameters`.

    This is the analytic coefficient-to-block-parameter portion of the native
    prepared-support transpose.  It retains one leading RHS axis and performs
    only elementwise reductions; importantly, it does not invoke a generic
    VJP over the dense operator construction.
    """
    n_modes, n_rhs, _, n_fs = lower_bars.shape
    if diagonal_bars.shape != lower_bars.shape or upper_bars.shape != lower_bars.shape:
        raise ValueError("all packed coefficient bars must have the same shape.")
    if n_modes != prepared.grid.n_xi + 1 or n_fs != prepared.grid.n_fs:
        raise ValueError("coefficient-bar shape does not match the prepared grid.")
    nt, nz = prepared.geometry.b.shape

    def field(values):
        # Inverse of the solver's Fortran-order spatial flattening, retaining
        # mode/RHS/component leading axes without an objective vmap.
        return jnp.swapaxes(values.reshape(n_modes, n_rhs, 3, nz, nt), -1, -2)

    lower = field(lower_bars)
    diagonal = field(diagonal_bars)
    upper = field(upper_bars)
    b = params["b"]
    bu = params["b_sup_theta"]
    bv = params["b_sup_zeta"]
    dbt = params["d_b_dtheta"]
    dbz = params["d_b_dzeta"]
    bsubt = params["b_sub_theta"]
    bsubz = params["b_sub_zeta"]
    jacobian = params["jacobian"]
    b2_mean = params["b2_mean"]
    nu_hat = params["nu_hat"]
    epsi_hat = params["epsi_hat"]
    del nu_hat
    mode = jnp.arange(n_modes, dtype=b.dtype)[:, None, None, None]
    lower_theta_scale = mode / (2.0 * mode - 1.0)
    upper_theta_scale = (mode + 1.0) / (2.0 * mode + 3.0)
    lower_value_scale = mode * (mode - 1.0) / (2.0 * (2.0 * mode - 1.0))
    upper_value_scale = -((mode + 1.0) * (mode + 2.0)) / (2.0 * (2.0 * mode + 3.0))

    inv_b = 1.0 / b
    h = bv * dbz + bu * dbt
    theta_bar = lower[:, :, 0] * lower_theta_scale + upper[:, :, 0] * upper_theta_scale
    zeta_bar = lower[:, :, 1] * lower_theta_scale + upper[:, :, 1] * upper_theta_scale
    h_bar = (lower[:, :, 2] * lower_value_scale + upper[:, :, 2] * upper_value_scale) * inv_b**2
    b_bar = jnp.sum(
        -theta_bar * bu[None, None] * inv_b**2
        - zeta_bar * bv[None, None] * inv_b**2
        - 2.0 * h_bar * h[None, None] * inv_b,
        axis=0,
    )
    bu_bar = jnp.sum(theta_bar * inv_b[None, None] + h_bar * dbt[None, None], axis=0)
    bv_bar = jnp.sum(zeta_bar * inv_b[None, None] + h_bar * dbz[None, None], axis=0)
    dbt_bar = jnp.sum(h_bar * bu[None, None], axis=0)
    dbz_bar = jnp.sum(h_bar * bv[None, None], axis=0)

    diagonal_theta_bar = diagonal[:, :, 0]
    diagonal_zeta_bar = diagonal[:, :, 1]
    denominator = jacobian * b2_mean
    theta_factor = -epsi_hat * bsubz / denominator
    zeta_factor = epsi_hat * bsubt / denominator
    bsubz_bar = jnp.sum(diagonal_theta_bar * (-epsi_hat / denominator)[None, None], axis=0)
    bsubt_bar = jnp.sum(diagonal_zeta_bar * (epsi_hat / denominator)[None, None], axis=0)
    jacobian_bar = jnp.sum(
        (diagonal_theta_bar * (-theta_factor / jacobian)[None, None]
         + diagonal_zeta_bar * (-zeta_factor / jacobian)[None, None]), axis=0
    )
    b2_mean_bar = jnp.sum(
        (diagonal_theta_bar * (-theta_factor / b2_mean)[None, None]
         + diagonal_zeta_bar * (-zeta_factor / b2_mean)[None, None]), axis=(0, 2, 3)
    )
    epsi_bar = jnp.sum(
        diagonal_theta_bar * (-bsubz / denominator)[None, None]
        + diagonal_zeta_bar * (bsubt / denominator)[None, None], axis=(0, 2, 3)
    )
    nu_bar = jnp.sum(
        diagonal[:, :, 2] * (0.5 * mode * (mode + 1.0)), axis=(0, 2, 3)
    )
    return {
        "b": b_bar,
        "b_sup_theta": bu_bar,
        "b_sup_zeta": bv_bar,
        "d_b_dtheta": dbt_bar,
        "d_b_dzeta": dbz_bar,
        "b_sub_theta": bsubt_bar,
        "b_sub_zeta": bsubz_bar,
        "jacobian": jacobian_bar,
        "b2_mean": b2_mean_bar,
        "nu_hat": nu_bar,
        "epsi_hat": epsi_bar,
    }


def _fixed_residual_source_geometry_bars_multi_rhs(lambda1: Array, lambda3: Array):
    """Direct RHS-axis transpose of the two fixed residual source arrays.

    Returns bars for the flattened ``b`` and ``radial_drift_spatial`` source
    inputs.  This is pure post-adjoint algebra: it neither builds nor solves
    an NTX system.
    """
    if lambda1.ndim != 3 or lambda3.shape != lambda1.shape:
        raise ValueError("source adjoints must have shape (mode, unknown, n_rhs).")
    if lambda1.shape[0] < 3:
        raise ValueError("the NTX source requires modes 0, 1 and 2.")
    # The residual pullback receives -lambda and r=A f-s, hence +lambda ds.
    b_bar = jnp.swapaxes(lambda3[1], 0, 1)
    drift_bar = jnp.swapaxes(
        -(2.0 / 3.0) * lambda1[0] - (1.0 / 3.0) * lambda1[2], 0, 1
    )
    return b_bar, drift_bar


def _direct_coefficient_geometry_bars_multi_rhs(prepared, f1_low, f3_low, nu_hat, coefficient_bars):
    """Direct RHS-axis transpose of transport coefficients at fixed modes."""
    g = prepared.geometry
    if coefficient_bars.ndim != 2 or coefficient_bars.shape[1] != 5:
        raise ValueError("coefficient_bars must have shape (n_rhs, 5).")
    nt, nz = g.b.shape
    def unflat(values):
        return jnp.swapaxes(values.reshape(nz, nt), -1, -2)
    f10, f11, f12 = (unflat(f1_low[k]) for k in range(3))
    f30, f31, f32 = (unflat(f3_low[k]) for k in range(3))
    q = coefficient_bars
    delta = g.grid.dtheta * g.grid.dzeta
    pref = g.jacobian / g.volume_prime * delta
    drift11 = -4.0 * f10 / 3.0 - 2.0 * f12 / 15.0
    drift13 = -4.0 * f30 / 3.0 - 2.0 * f32 / 15.0
    d11_density = pref * g.radial_drift_spatial * drift11 / g.coefficient_psi_scale**2
    d31_density = pref * 2.0 * f11 * g.b / (3.0 * g.b0 * g.coefficient_psi_scale)
    d13_density = pref * g.radial_drift_spatial * drift13 / (g.coefficient_psi_scale * g.b0)
    d33_density = pref * 2.0 * g.b * f31 / (3.0 * g.b0**2)
    dsp_density = pref * 2.0 * g.b**2 / (3.0 * g.b0**2 * nu_hat)
    densities = jnp.stack((d11_density, d31_density, d13_density, d33_density, dsp_density))
    values = jnp.sum(densities, axis=(1, 2))
    weight = q[:, :, None, None]
    jacobian_bar = jnp.sum(weight * densities[None] / g.jacobian, axis=1)
    volume_prime_bar = -jnp.sum(q * values[None] / g.volume_prime, axis=1)
    drift_bar = q[:, 0, None, None] * pref * drift11[None] / g.coefficient_psi_scale**2 + q[:, 2, None, None] * pref * drift13[None] / (g.coefficient_psi_scale * g.b0)
    b_bar = (q[:, 1, None, None] * pref * 2.0 * f11[None] / (3.0*g.b0*g.coefficient_psi_scale) + q[:, 3, None, None] * pref * 2.0*f31[None]/(3.0*g.b0**2) + q[:, 4, None, None] * pref * 4.0*g.b[None]/(3.0*g.b0**2*nu_hat))
    psi_bar = -2.0*q[:,0]*values[0]/g.coefficient_psi_scale - q[:,1]*values[1]/g.coefficient_psi_scale - q[:,2]*values[2]/g.coefficient_psi_scale
    b0_bar = -q[:,1]*values[1]/g.b0 -q[:,2]*values[2]/g.b0 -2.0*q[:,3]*values[3]/g.b0 -2.0*q[:,4]*values[4]/g.b0
    nu_bar = -q[:,4]*values[4]/nu_hat
    return dict(radial_drift_spatial=drift_bar, jacobian=jacobian_bar, volume_prime=volume_prime_bar, coefficient_psi_scale=psi_bar, b=b_bar, b0=b0_bar, nu_hat=nu_bar)


def _add_native_rhs_bar_dicts(*bar_dicts):
    """Add matching RHS-preserving primitive cotangent dictionaries."""
    result = {}
    for bars in bar_dicts:
        for key, value in bars.items():
            result[key] = value if key not in result else result[key] + value
    return result


def _native_vmec_coefficient_bars_from_fixed_adjoint_multi_rhs(
    prepared: PreparedMonoenergeticSystem, ctx: OperatorContext, f1_full: Array,
    f3_full: Array, lambda1: Array, lambda3: Array, coefficient_bars: Array,
) -> dict[str, Array]:
    """Native VMEC coefficient bars from existing fixed primal/adjoint fields."""
    primitive_bars = _native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
        coefficient_bars,
    )
    return vmec_geometry_bars_to_coefficients_multi_rhs(
        prepared.surface, prepared.geometry, primitive_bars
    )


def _native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs(
    prepared: PreparedMonoenergeticSystem, ctx: OperatorContext, f1_full: Array,
    f3_full: Array, lambda1: Array, lambda3: Array, coefficient_bars: Array,
) -> dict[str, Array]:
    """RHS-batched VMEC primitive bars before the Fourier transpose.

    Keeping this intermediate form allows the fused low-dot helper to add its
    base and directional cotangents before one VMEC sampled-field transpose.
    That is the same linear contraction order as a combined prepared VJP.
    """
    if not isinstance(prepared.surface, VmecSurface):
        raise ValueError("native VMEC geometry transpose requires VmecSurface.")
    lower, diagonal, upper = _fixed_residual_block_coefficient_bars_multi_rhs(
        prepared, f1_full, f3_full, lambda1, lambda3
    )
    parameter_bars = _block_parameters_bar_from_coefficient_bars_multi_rhs(
        prepared, block_parameters(ctx), lower, diagonal, upper
    )
    source_b, source_drift = _fixed_residual_source_geometry_bars_multi_rhs(lambda1, lambda3)
    nt, nz = prepared.geometry.b.shape
    unflat = lambda value: jnp.swapaxes(value.reshape(value.shape[0], nz, nt), -1, -2)
    source_bars = {"b": unflat(source_b), "radial_drift_spatial": unflat(source_drift)}
    direct_bars = _direct_coefficient_geometry_bars_multi_rhs(
        prepared, f1_full[:3], f3_full[:3], ctx.nu_hat, coefficient_bars
    )
    return _add_native_rhs_bar_dicts(parameter_bars, source_bars, direct_bars)


def _directional_native_vmec_coefficient_bars_from_fixed_adjoint_multi_rhs(
    prepared: PreparedMonoenergeticSystem, *, nu_hat: Array, epsi_hat: Array,
    nu_hat_dot: Array, epsi_hat_dot: Array, f1_full: Array, f3_full: Array,
    f1_dot: Array, f3_dot: Array, lambda1: Array, lambda3: Array,
    lambda1_dot: Array, lambda3_dot: Array, coefficient_bars: Array,
    coefficient_bars_dot: Array,
):
    """Base/directional VMEC coefficient bars without an NTX re-solve."""
    def native(nu_value, epsi_value, f1_value, f3_value, l1_value, l3_value, bar_value):
        ctx = _operator_context(prepared.surface, prepared.geometry, prepared.grid,
                                nu_value, epsi_value)
        return _native_vmec_coefficient_bars_from_fixed_adjoint_multi_rhs(
            prepared, ctx, f1_value, f3_value, l1_value, l3_value, bar_value
        )
    return jax.jvp(
        native,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3, coefficient_bars),
        (nu_hat_dot, epsi_hat_dot, f1_dot, f3_dot, lambda1_dot, lambda3_dot,
         coefficient_bars_dot),
    )


def _directional_native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs_jvp(
    prepared: PreparedMonoenergeticSystem, *, nu_hat: Array, epsi_hat: Array,
    nu_hat_dot: Array, epsi_hat_dot: Array, f1_full: Array, f3_full: Array,
    f1_dot: Array, f3_dot: Array, lambda1: Array, lambda3: Array,
    lambda1_dot: Array, lambda3_dot: Array, coefficient_bars: Array,
    coefficient_bars_dot: Array,
) -> dict[str, Array]:
    """Existing generic-JVP primitive tangent, exposed only as an oracle."""
    def native(nu_value, epsi_value, f1_value, f3_value, l1_value, l3_value, bar_value):
        ctx = _operator_context(
            prepared.surface, prepared.geometry, prepared.grid, nu_value, epsi_value
        )
        return _native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs(
            prepared, ctx, f1_value, f3_value, l1_value, l3_value, bar_value
        )

    return jax.jvp(
        native,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3, coefficient_bars),
        (nu_hat_dot, epsi_hat_dot, f1_dot, f3_dot, lambda1_dot, lambda3_dot,
         coefficient_bars_dot),
    )[1]


def _directional_native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs_direct(
    prepared: PreparedMonoenergeticSystem, *, nu_hat: Array, epsi_hat: Array,
    nu_hat_dot: Array, epsi_hat_dot: Array, f1_full: Array, f3_full: Array,
    f1_dot: Array, f3_dot: Array, lambda1: Array, lambda3: Array,
    lambda1_dot: Array, lambda3_dot: Array, coefficient_bars: Array,
    coefficient_bars_dot: Array,
) -> dict[str, Array]:
    """Exact product-rule form of the directional native VMEC primitive bars.

    This private helper is deliberately the same derivative returned as the
    tangent from :func:`_directional_native_vmec_coefficient_bars_from_fixed_adjoint_multi_rhs`.
    It replaces only the generic forward-mode transformation of the
    post-adjoint primitive-bar algebra: the NTX primal, factorisation and
    matrix-RHS adjoint fields are inputs and are not rebuilt here.

    The VMEC sampled-field/Fourier transpose is linear for the fixed prepared
    surface, so all primitive directional bars are accumulated first and
    converted once at the end.
    """
    def _add(*bar_dicts: dict[str, Array]) -> dict[str, Array]:
        return _add_native_rhs_bar_dicts(*bar_dicts)

    def _subtract(first: dict[str, Array], second: dict[str, Array]) -> dict[str, Array]:
        return {
            name: first.get(name, 0.0) - second.get(name, 0.0)
            for name in first.keys() | second.keys()
        }

    def _scale(bar_dict: dict[str, Array], scalar: Array) -> dict[str, Array]:
        return {name: scalar * value for name, value in bar_dict.items()}

    ctx = _operator_context(
        prepared.surface, prepared.geometry, prepared.grid, nu_hat, epsi_hat
    )

    # ``_fixed_residual_block_coefficient_bars_multi_rhs`` is bilinear in the
    # primal modes and adjoint fields.  Its tangent is therefore exactly the
    # two product-rule contractions below, including its existing nullspace
    # row handling.
    block_from_primal = _fixed_residual_block_coefficient_bars_multi_rhs(
        prepared, f1_dot, f3_dot, lambda1, lambda3
    )
    block_from_adjoint = _fixed_residual_block_coefficient_bars_multi_rhs(
        prepared, f1_full, f3_full, lambda1_dot, lambda3_dot
    )
    block_dot = tuple(
        primal_part + adjoint_part
        for primal_part, adjoint_part in zip(
            block_from_primal, block_from_adjoint, strict=True
        )
    )
    params = block_parameters(ctx)
    parameter_dot = _block_parameters_bar_from_coefficient_bars_multi_rhs(
        prepared, params, *block_dot
    )

    # The explicit epsilon term multiplies the base block bars, which have
    # lambda and f at their primal values.  Do not obtain this slope by
    # subtracting two complete parameter maps: their large epsilon-independent
    # terms can lose precision before the final VMEC Fourier cancellation.
    block_base = _fixed_residual_block_coefficient_bars_multi_rhs(
        prepared, f1_full, f3_full, lambda1, lambda3
    )
    _lower_base, diagonal_base, _upper_base = block_base
    n_modes, n_rhs, _, n_fs = diagonal_base.shape
    nt, nz = prepared.geometry.b.shape
    diagonal_field = jnp.swapaxes(
        diagonal_base.reshape(n_modes, n_rhs, 3, nz, nt), -1, -2
    )
    diagonal_theta_bar = diagonal_field[:, :, 0]
    diagonal_zeta_bar = diagonal_field[:, :, 1]
    denominator = (
        prepared.geometry.jacobian * prepared.geometry.b2_mean
    )
    parameter_epsi_slope = {
        "b_sub_zeta": jnp.sum(
            diagonal_theta_bar * (-1.0 / denominator)[None, None], axis=0
        ),
        "b_sub_theta": jnp.sum(
            diagonal_zeta_bar * (1.0 / denominator)[None, None], axis=0
        ),
        "jacobian": jnp.sum(
            (
                diagonal_theta_bar
                * (prepared.geometry.b_sub_zeta / (denominator * prepared.geometry.jacobian))[None, None]
                + diagonal_zeta_bar
                * (-prepared.geometry.b_sub_theta / (denominator * prepared.geometry.jacobian))[None, None]
            ),
            axis=0,
        ),
        "b2_mean": jnp.sum(
            (
                diagonal_theta_bar
                * (prepared.geometry.b_sub_zeta / denominator)[None, None]
                + diagonal_zeta_bar
                * (-prepared.geometry.b_sub_theta / denominator)[None, None]
            ) / prepared.geometry.b2_mean,
            axis=(0, 2, 3),
        ),
    }
    parameter_dot = _add(
        parameter_dot,
        _scale(parameter_epsi_slope, epsi_hat_dot),
    )

    source_b_dot, source_drift_dot = _fixed_residual_source_geometry_bars_multi_rhs(
        lambda1_dot, lambda3_dot
    )
    source_dot = {
        "b": jnp.swapaxes(
            source_b_dot.reshape(source_b_dot.shape[0], *prepared.geometry.b.shape[::-1]),
            -1,
            -2,
        ),
        "radial_drift_spatial": jnp.swapaxes(
            source_drift_dot.reshape(
                source_drift_dot.shape[0], *prepared.geometry.b.shape[::-1]
            ),
            -1,
            -2,
        ),
    }

    # The direct coefficient map is linear in f and coefficient bars, while
    # its q_4 contribution is proportional to nu_hat**-1.  Split its product
    # rule explicitly; no forward-mode transform is staged.
    f1_zero = jnp.zeros_like(f1_full[:3])
    f3_zero = jnp.zeros_like(f3_full[:3])
    direct_zero = _direct_coefficient_geometry_bars_multi_rhs(
        prepared, f1_zero, f3_zero, nu_hat, coefficient_bars
    )
    direct_from_primal = _subtract(
        _direct_coefficient_geometry_bars_multi_rhs(
            prepared, f1_dot[:3], f3_dot[:3], nu_hat, coefficient_bars
        ),
        direct_zero,
    )
    direct_from_bar = _direct_coefficient_geometry_bars_multi_rhs(
        prepared, f1_full[:3], f3_full[:3], nu_hat, coefficient_bars_dot
    )
    direct_from_nu = _scale(direct_zero, -nu_hat_dot / nu_hat)
    # The returned nu_hat cotangent is itself proportional to nu_hat**-2.
    direct_from_nu["nu_hat"] = (
        -2.0 * nu_hat_dot / nu_hat * direct_zero["nu_hat"]
    )
    direct_dot = _add(direct_from_primal, direct_from_bar, direct_from_nu)

    primitive_dot = _add(parameter_dot, source_dot, direct_dot)
    return primitive_dot


def _directional_native_vmec_coefficient_bars_from_fixed_adjoint_multi_rhs_direct(
    prepared: PreparedMonoenergeticSystem, **kwargs,
) -> dict[str, Array]:
    """Convert the direct directional primitive bars to VMEC coefficients.

    The primitive helper is used by the low-dot path so base and both
    directional contributions can be summed before one conversion.  This
    wrapper is retained as the direct unit-test boundary.
    """
    primitive_bars = _directional_native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs_direct(
        prepared, **kwargs
    )
    return vmec_geometry_bars_to_coefficients_multi_rhs(
        prepared.surface, prepared.geometry, primitive_bars
    )


def _geometry_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
):
    """Return the exact geometry cotangent from prepared primal/adjoint modes.

    This is the geometry counterpart of :func:`_parameter_gradient_from_adjoint`.
    It deliberately differentiates the *fixed* block residual, not the factorized
    solve.  Consequently it reuses ``f1_full``, ``f3_full``, ``lambda1`` and
    ``lambda3`` already computed by the implicit case adjoint:

    ``dL/dg = dL_direct/dg - lambda.T d(A(g) f - s(g))/dg``.

    The result is a cotangent for ``prepared.geometry``.  The caller is
    responsible for chaining it to its higher-level support/geometry payload.
    """

    n_xi = int(prepared.grid.n_xi)

    def _conditioned_blocks(local_ctx: OperatorContext, mode_index: int):
        lower, diagonal, upper = operator_blocks(
            local_ctx,
            mode_index,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if mode_index == 0:
            diagonal, upper = apply_nullspace_condition(diagonal, upper)
            assert upper is not None
        return lower, diagonal, upper

    def _residual_and_direct_coefficients(geometry):
        local_ctx = OperatorContext(
            surface=prepared.surface,
            geometry=geometry,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
        )
        source1, source3 = source_modes(local_ctx, n_xi)

        def _residual(modes, source):
            rows = []
            for mode_index in range(n_xi + 1):
                lower, diagonal, upper = _conditioned_blocks(local_ctx, mode_index)
                row = diagonal @ modes[mode_index] - source[mode_index]
                if mode_index > 0:
                    row = row + lower @ modes[mode_index - 1]
                if mode_index < n_xi:
                    row = row + upper @ modes[mode_index + 1]
                rows.append(row)
            return jnp.stack(rows)

        direct_coefficients = jnp.stack(
            coefficients_from_modes(
                geometry,
                f1_full[:3],
                f3_full[:3],
                local_ctx.nu_hat,
            )
        )
        return (
            direct_coefficients,
            _residual(f1_full, source1),
            _residual(f3_full, source3),
        )

    _, pullback = jax.vjp(
        _residual_and_direct_coefficients,
        prepared.geometry,
    )
    (geometry_bar,) = pullback((coefficient_bar, -lambda1, -lambda3))
    return geometry_bar


def _prepared_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
):
    """Exact fixed-primal cotangent for every differentiable prepared leaf.

    This is the full-prepared counterpart of
    :func:`_geometry_gradient_from_adjoint`.  Crucially, its VJP is only over
    the fixed residual and direct coefficient contraction: it does *not*
    differentiate a factorization or execute a second primal/adjoint solve.
    Thus the already available ``f`` and ``lambda`` modes are shared for the
    surface, geometry, derivative-operator, and grid-dependent support leaves.
    """
    n_xi = int(prepared.grid.n_xi)

    def _residual_and_direct_coefficients(prepared_value):
        local_ctx = _operator_context(
            prepared_value.surface,
            prepared_value.geometry,
            prepared_value.grid,
            ctx.nu_hat,
            ctx.epsi_hat,
        )
        source1, source3 = source_modes(local_ctx, n_xi)

        def _residual(modes, source):
            rows = []
            for mode_index in range(n_xi + 1):
                lower, diagonal, upper = operator_blocks(
                    local_ctx,
                    mode_index,
                    prepared_value.d_theta,
                    prepared_value.d_zeta,
                )
                if mode_index == 0:
                    diagonal, upper = apply_nullspace_condition(diagonal, upper)
                    assert upper is not None
                row = diagonal @ modes[mode_index] - source[mode_index]
                if mode_index > 0:
                    row = row + lower @ modes[mode_index - 1]
                if mode_index < n_xi:
                    row = row + upper @ modes[mode_index + 1]
                rows.append(row)
            return jnp.stack(rows)

        direct_coefficients = jnp.stack(
            coefficients_from_modes(
                prepared_value.geometry,
                f1_full[:3],
                f3_full[:3],
                local_ctx.nu_hat,
            )
        )
        return (
            direct_coefficients,
            _residual(f1_full, source1),
            _residual(f3_full, source3),
        )

    _, pullback = jax.vjp(_residual_and_direct_coefficients, prepared)
    (prepared_bar,) = pullback((coefficient_bar, -lambda1, -lambda3))
    return prepared_bar


def _compact_prepared_residual_inputs(
    prepared: PreparedMonoenergeticSystem,
    nu_hat: Array,
    epsi_hat: Array,
):
    """Return the dynamic inputs to the fixed prepared residual.

    The dense residual depends on the complete prepared pytree only through
    the geometry used by the coefficient contraction, the two derivative
    matrices, and the compact block-parameter arrays.  Keeping this tuple
    explicit lets the expensive residual transpose stop there; a separate,
    small VJP subsequently chains its cotangent back to ``prepared``.
    """
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        nu_hat,
        epsi_hat,
    )
    return (
        prepared.geometry,
        prepared.d_theta,
        prepared.d_zeta,
        block_parameters(ctx),
    )


def _compact_residual_and_direct_coefficients(
    compact_inputs,
    *,
    surface,
    n_xi: int,
    f1_full: Array,
    f3_full: Array,
):
    """Fixed-primal residual expressed in compact prepared inputs only."""
    geometry, d_theta, d_zeta, params = compact_inputs
    local_ctx = OperatorContext(
        surface=surface,
        geometry=geometry,
        nu_hat=params["nu_hat"],
        epsi_hat=params["epsi_hat"],
    )
    source1, source3 = source_modes(local_ctx, n_xi)

    def _residual(modes, source):
        rows = []
        for mode_index in range(n_xi + 1):
            lower, diagonal, upper = operator_blocks_from_parameters(
                params,
                mode_index,
                d_theta,
                d_zeta,
            )
            if mode_index == 0:
                diagonal, upper = apply_nullspace_condition(diagonal, upper)
                assert upper is not None
            row = diagonal @ modes[mode_index] - source[mode_index]
            if mode_index > 0:
                row = row + lower @ modes[mode_index - 1]
            if mode_index < n_xi:
                row = row + upper @ modes[mode_index + 1]
            rows.append(row)
        return jnp.stack(rows)

    direct_coefficients = jnp.stack(
        coefficients_from_modes(
            geometry,
            f1_full[:3],
            f3_full[:3],
            params["nu_hat"],
        )
    )
    return (
        direct_coefficients,
        _residual(f1_full, source1),
        _residual(f3_full, source3),
    )


def _compact_prepared_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
):
    """Exact fixed-primal cotangent for compact prepared residual inputs.

    This has the same mathematical residual/direct-coefficient contraction as
    :func:`_prepared_gradient_from_adjoint`, but deliberately stops at the
    compact input tuple.  It neither factorizes nor solves.  The caller uses
    :func:`_compact_prepared_bar_to_prepared` to apply the cheap remaining
    prepared-input chain once to the final combined cotangent.
    """
    compact_inputs = _compact_prepared_residual_inputs(
        prepared,
        ctx.nu_hat,
        ctx.epsi_hat,
    )
    _, pullback = jax.vjp(
        lambda inputs: _compact_residual_and_direct_coefficients(
            inputs,
            surface=prepared.surface,
            n_xi=int(prepared.grid.n_xi),
            f1_full=f1_full,
            f3_full=f3_full,
        ),
        compact_inputs,
    )
    (compact_bar,) = pullback((coefficient_bar, -lambda1, -lambda3))
    return compact_bar


def _compact_prepared_bar_to_prepared(
    prepared: PreparedMonoenergeticSystem,
    *,
    nu_hat: Array,
    epsi_hat: Array,
    compact_bar,
):
    """Chain a compact residual-input cotangent back to ``prepared``.

    This transpose intentionally contains no dense operator construction.  In
    particular, shared geometry leaves receive the sum of their direct
    geometry and block-parameter contributions here, exactly as in the old
    all-in-one prepared VJP.
    """
    _, pullback = jax.vjp(
        lambda prepared_value: _compact_prepared_residual_inputs(
            prepared_value,
            nu_hat,
            epsi_hat,
        ),
        prepared,
    )
    (prepared_bar,) = pullback(compact_bar)
    return prepared_bar


def _directional_compact_prepared_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    *,
    nu_hat: Array,
    epsi_hat: Array,
    nu_hat_dot: Array,
    epsi_hat_dot: Array,
    f1_full: Array,
    f3_full: Array,
    f1_dot: Array,
    f3_dot: Array,
    lambda1: Array,
    lambda3: Array,
    lambda1_dot: Array,
    lambda3_dot: Array,
    coefficient_bar: Array,
    coefficient_bar_dot: Array | None = None,
):
    """Directional compact-residual cotangent without a prepared VJP.

    The two outputs are the base and case-directional compact cotangents.
    The caller should retain only the directional cotangent for the primal
    prepared bar, combine it with the base compact bar, then execute one
    compact-to-prepared pullback.
    """
    def _compact_bar_from_dynamic_terms(
        nu_hat_value,
        epsi_hat_value,
        f1_value,
        f3_value,
        lambda1_value,
        lambda3_value,
        coefficient_bar_value,
    ):
        local_ctx = _operator_context(
            prepared.surface,
            prepared.geometry,
            prepared.grid,
            nu_hat_value,
            epsi_hat_value,
        )
        return _compact_prepared_gradient_from_adjoint(
            prepared,
            local_ctx,
            f1_value,
            f3_value,
            lambda1_value,
            lambda3_value,
            coefficient_bar_value,
        )

    if coefficient_bar_dot is None:
        coefficient_bar_dot = jnp.zeros_like(coefficient_bar)

    return jax.jvp(
        _compact_bar_from_dynamic_terms,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3, coefficient_bar),
        (
            nu_hat_dot,
            epsi_hat_dot,
            f1_dot,
            f3_dot,
            lambda1_dot,
            lambda3_dot,
            coefficient_bar_dot,
        ),
    )


def _case_and_geometry_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
    nu_hat_direct_bar: Array,
):
    """Return exact case and prepared-geometry bars from one implicit adjoint.

    ``nu_hat_direct_bar`` is the direct coefficient contribution returned by
    :func:`_coefficient_mode_pullback`; the factorized adjoint solutions are
    shared by the case and geometry terms.
    """

    nu_hat_implicit_bar, epsi_hat_bar = _parameter_gradient_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
    )
    geometry_bar = _geometry_gradient_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
        coefficient_bar,
    )
    return (
        nu_hat_direct_bar + nu_hat_implicit_bar,
        epsi_hat_bar,
        geometry_bar,
    )


def _directional_geometry_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    *,
    nu_hat: Array,
    epsi_hat: Array,
    nu_hat_dot: Array,
    epsi_hat_dot: Array,
    f1_full: Array,
    f3_full: Array,
    f1_dot: Array,
    f3_dot: Array,
    lambda1: Array,
    lambda3: Array,
    lambda1_dot: Array,
    lambda3_dot: Array,
    coefficient_bar: Array,
    coefficient_bar_dot: Array | None = None,
):
    """Return base and directional exact geometry bars without re-solving.

    The directional result is the derivative of the fixed-residual geometry
    pullback along the supplied case direction.  Primal and adjoint tangents
    are inputs, so the nested JVP never differentiates through an LU
    factorization or invokes another implicit solve.
    """

    def _geometry_bar_from_dynamic_terms(
        nu_hat_value,
        epsi_hat_value,
        f1_value,
        f3_value,
        lambda1_value,
        lambda3_value,
        coefficient_bar_value,
    ):
        local_ctx = OperatorContext(
            surface=prepared.surface,
            geometry=prepared.geometry,
            nu_hat=nu_hat_value,
            epsi_hat=epsi_hat_value,
        )
        return _geometry_gradient_from_adjoint(
            prepared,
            local_ctx,
            f1_value,
            f3_value,
            lambda1_value,
            lambda3_value,
            coefficient_bar_value,
        )

    if coefficient_bar_dot is None:
        coefficient_bar_dot = jnp.zeros_like(coefficient_bar)

    return jax.jvp(
        _geometry_bar_from_dynamic_terms,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3, coefficient_bar),
        (
            nu_hat_dot,
            epsi_hat_dot,
            f1_dot,
            f3_dot,
            lambda1_dot,
            lambda3_dot,
            coefficient_bar_dot,
        ),
    )


def _directional_prepared_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    *,
    nu_hat: Array,
    epsi_hat: Array,
    nu_hat_dot: Array,
    epsi_hat_dot: Array,
    f1_full: Array,
    f3_full: Array,
    f1_dot: Array,
    f3_dot: Array,
    lambda1: Array,
    lambda3: Array,
    lambda1_dot: Array,
    lambda3_dot: Array,
    coefficient_bar: Array,
    coefficient_bar_dot: Array | None = None,
):
    """Directional counterpart of :func:`_prepared_gradient_from_adjoint`."""
    def _prepared_bar_from_dynamic_terms(
        nu_hat_value,
        epsi_hat_value,
        f1_value,
        f3_value,
        lambda1_value,
        lambda3_value,
        coefficient_bar_value,
    ):
        local_ctx = _operator_context(
            prepared.surface, prepared.geometry, prepared.grid, nu_hat_value, epsi_hat_value
        )
        return _prepared_gradient_from_adjoint(
            prepared,
            local_ctx,
            f1_value,
            f3_value,
            lambda1_value,
            lambda3_value,
            coefficient_bar_value,
        )

    if coefficient_bar_dot is None:
        coefficient_bar_dot = jnp.zeros_like(coefficient_bar)

    return jax.jvp(
        _prepared_bar_from_dynamic_terms,
        (nu_hat, epsi_hat, f1_full, f3_full, lambda1, lambda3, coefficient_bar),
        (
            nu_hat_dot,
            epsi_hat_dot,
            f1_dot,
            f3_dot,
            lambda1_dot,
            lambda3_dot,
            coefficient_bar_dot,
        ),
    )


def _combined_prepared_gradient_from_adjoint_multi_rhs_oracle(
    prepared: PreparedMonoenergeticSystem,
    *,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    first_f1_dot: Array,
    first_f3_dot: Array,
    second_f1_dot: Array,
    second_f3_dot: Array,
    nu_dots: Array,
    epsi_dots: Array,
    base_lambda1: Array,
    base_lambda3: Array,
    directional_lambda1: Array,
    directional_lambda3: Array,
    directional_lambda1_dot: Array,
    directional_lambda3_dot: Array,
    base_coefficient_bars: Array,
    first_coefficient_bars: Array,
    second_coefficient_bars: Array,
    first_coefficient_bars_dot: Array,
    second_coefficient_bars_dot: Array,
) -> tuple[Array, ...]:
    """Oracle contract for the native combined prepared-support transpose.

    The grouped native low-dot path already supplies one factorization and its
    base/directional adjoint fields with a trailing RHS axis.  This helper
    defines the *final combined prepared-bar* contract that a future explicit
    RHS-axis transpose must reproduce.  It deliberately uses the established
    scalar gradient routines under ``vmap`` for now; it is a numerical oracle,
    not the production optimisation.  Keeping it separate prevents any change
    to current public or reverse-mode behaviour while the native contraction
    is derived and tested.

    The return is a tuple of dynamic prepared leaves, each shaped
    ``(n_rhs, *prepared_leaf.shape)``.  Static prepared metadata is never
    reconstructed or batched here.
    """

    def _one_rhs(
        base_coefficient_bar,
        first_coefficient_bar,
        second_coefficient_bar,
        first_coefficient_bar_dot,
        second_coefficient_bar_dot,
        base_lambda1_value,
        base_lambda3_value,
        directional_lambda1_value,
        directional_lambda3_value,
        directional_lambda1_dot_value,
        directional_lambda3_dot_value,
    ):
        base_prepared = _prepared_gradient_from_adjoint(
            prepared,
            ctx,
            f1_full,
            f3_full,
            base_lambda1_value,
            base_lambda3_value,
            base_coefficient_bar,
        )
        _, first_directional_prepared = _directional_prepared_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_dots[0],
            epsi_hat_dot=epsi_dots[0],
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=first_f1_dot,
            f3_dot=first_f3_dot,
            lambda1=directional_lambda1_value[..., 0],
            lambda3=directional_lambda3_value[..., 0],
            lambda1_dot=directional_lambda1_dot_value[..., 0],
            lambda3_dot=directional_lambda3_dot_value[..., 0],
            coefficient_bar=first_coefficient_bar,
            coefficient_bar_dot=first_coefficient_bar_dot,
        )
        _, second_directional_prepared = _directional_prepared_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_dots[1],
            epsi_hat_dot=epsi_dots[1],
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=second_f1_dot,
            f3_dot=second_f3_dot,
            lambda1=directional_lambda1_value[..., 1],
            lambda3=directional_lambda3_value[..., 1],
            lambda1_dot=directional_lambda1_dot_value[..., 1],
            lambda3_dot=directional_lambda3_dot_value[..., 1],
            coefficient_bar=second_coefficient_bar,
            coefficient_bar_dot=second_coefficient_bar_dot,
        )

        combined_leaves = []
        for primal_leaf, base_leaf, first_leaf, second_leaf in zip(
            jax.tree_util.tree_leaves(prepared),
            jax.tree_util.tree_leaves(base_prepared),
            jax.tree_util.tree_leaves(first_directional_prepared),
            jax.tree_util.tree_leaves(second_directional_prepared),
            strict=True,
        ):
            primal_value = jnp.asarray(primal_leaf)
            if not jnp.issubdtype(primal_value.dtype, jnp.inexact):
                combined_leaves.append(
                    jnp.zeros(primal_value.shape, dtype=jnp.float64)
                )
            elif (
                jnp.asarray(base_leaf).dtype == jax.dtypes.float0
                or jnp.asarray(first_leaf).dtype == jax.dtypes.float0
                or jnp.asarray(second_leaf).dtype == jax.dtypes.float0
            ):
                combined_leaves.append(jnp.zeros_like(primal_value))
            else:
                combined_leaves.append(base_leaf + first_leaf + second_leaf)
        return tuple(combined_leaves)

    return jax.vmap(_one_rhs)(
        base_coefficient_bars,
        first_coefficient_bars,
        second_coefficient_bars,
        first_coefficient_bars_dot,
        second_coefficient_bars_dot,
        jnp.moveaxis(base_lambda1, 2, 0),
        jnp.moveaxis(base_lambda3, 2, 0),
        jnp.moveaxis(directional_lambda1, 2, 0),
        jnp.moveaxis(directional_lambda3, 2, 0),
        jnp.moveaxis(directional_lambda1_dot, 2, 0),
        jnp.moveaxis(directional_lambda3_dot, 2, 0),
    )
