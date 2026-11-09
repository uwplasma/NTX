"""Custom AD rules and derivative utilities used by NEOPAX.

Geometry transposes, factorized solver derivatives and recorded-scan pullbacks
share this module. The core NTX solve and the generic scan remain in their
existing modules; scan-specific imports are deferred to avoid import cycles.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, cast, overload

import jax
import jax.numpy as jnp
from jax import Array

from ._geometry import BoozerSurface, GeometryOnGrid, VmecSurface, geometry_on_grid
from ._solver import (
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    _coefficient_mode_pullback,
    _operator_context,
    _parameter_gradient_from_adjoint,
    _prepared_implicit_vjp_primal,
    _solve_factorized_adjoint,
    _solve_factorized_modes,
    prepare_monoenergetic_system,
)
from ._solver_scan import _resolved_scan_inputs, _scan_coefficients_serial
from .grids import GridSpec, flatten_fs
from .operators import (
    OperatorContext,
    apply_nullspace_condition,
    block_parameters,
    operator_blocks_from_parameters,
    parameter_derivative_blocks,
    source_modes,
)
from .transport import coefficients_from_modes

if TYPE_CHECKING:
    from ._neopax_scan import NeopaxScanCoefficientBlocks


# VMEC geometry transposes
# ------------------------


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
        value_bar * cosine - d_theta_bar * mode_m * sine - d_zeta_bar * mode_nfp * sine,
        axis=(1, 2),
    )
    sine_bar = jnp.sum(
        value_bar * sine + d_theta_bar * mode_m * cosine + d_zeta_bar * mode_nfp * cosine,
        axis=(1, 2),
    )
    return cosine_bar, sine_bar


def volume_prime_bars_multi_rhs(volume_prime_bar: Array, jacobian: Array, grid) -> Array:
    """Add the RHS-axis transpose of ``volume_prime = sum(J) dtheta dzeta``."""
    return (
        volume_prime_bar[:, None, None] * jnp.ones_like(jacobian)[None] * (grid.dtheta * grid.dzeta)
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
    b: Array,
    d_b_dtheta: Array,
    d_b_dzeta: Array,
    jacobian: Array,
    b_sub_theta: Array,
    b_sub_zeta: Array,
    radial_drift_bar: Array,
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


def vmec_sampled_field_bars_to_coefficients_multi_rhs(
    surface: VmecSurface, geometry: GeometryOnGrid, bars: dict[str, Array]
) -> dict[str, Array]:
    """Map RHS-batched sampled VMEC field bars to Fourier coefficient bars."""

    def zeros():
        return jnp.zeros_like(bars["b"])

    result: dict[str, Array] = {}
    result["b_cos"], _ = fourier_series_coefficient_bars_multi_rhs(
        surface.m,
        surface.n,
        geometry.theta_2d,
        geometry.zeta_2d,
        nfp=surface.nfp,
        value_bar=bars.get("b", zeros()),
        d_theta_bar=bars.get("d_b_dtheta"),
        d_zeta_bar=bars.get("d_b_dzeta"),
    )
    for field, coeff in (
        ("jacobian", "jacobian_cos"),
        ("b_sub_theta", "b_sub_theta_cos"),
        ("b_sub_zeta", "b_sub_zeta_cos"),
        ("b_sup_theta", "b_sup_theta_cos"),
        ("b_sup_zeta", "b_sup_zeta_cos"),
    ):
        if field in bars:
            result[coeff], _ = fourier_series_coefficient_bars_multi_rhs(
                surface.m,
                surface.n,
                geometry.theta_2d,
                geometry.zeta_2d,
                nfp=surface.nfp,
                value_bar=bars[field],
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
    if not isinstance(surface, VmecSurface):
        raise ValueError("The native geometry transpose requires a VMEC surface.")
    template = primitive_bars["b"]
    zero_field = jnp.zeros_like(template)
    sampled = dict(primitive_bars)
    volume_bar = sampled.pop("volume_prime", None)
    if volume_bar is not None:
        sampled = add_sampled_field_bars_multi_rhs(
            sampled,
            {"jacobian": volume_prime_bars_multi_rhs(volume_bar, geometry.jacobian, geometry.grid)},
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
                geometry.b,
                geometry.d_b_dtheta,
                geometry.d_b_dzeta,
                geometry.jacobian,
                geometry.b_sub_theta,
                geometry.b_sub_zeta,
                drift_bar,
            ),
        )
    sampled.setdefault("b", zero_field)
    result = vmec_sampled_field_bars_to_coefficients_multi_rhs(surface, geometry, sampled)
    if "b0" in primitive_bars:
        result["b0"] = jnp.asarray(primitive_bars["b0"])
    return result


# Fixed-residual adjoint contractions
# -----------------------------------


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

    q0, q1, q2, q3, q4 = (coefficient_bars[:, index, None, None] for index in range(5))
    drift_weight = weight * drift
    f1_bar = jnp.stack(
        (
            _flat(q0 * (-4.0 / 3.0) * drift_weight / psi_scale**2),
            _flat(q1 * weight * (2.0 * b / (3.0 * b0 * psi_scale))),
            _flat(q0 * (-2.0 / 15.0) * drift_weight / psi_scale**2),
        ),
        axis=1,
    )
    f3_bar = jnp.stack(
        (
            _flat(q2 * (-4.0 / 3.0) * drift_weight / (psi_scale * b0)),
            _flat(q3 * weight * (2.0 * b / (3.0 * b0**2))),
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
    base = _coefficient_mode_pullback_multi_rhs_direct(geom, nu_hat, coefficient_bars)
    bar_tangent = _coefficient_mode_pullback_multi_rhs_direct(geom, nu_hat, coefficient_bars_dot)
    f1_dot, f3_dot, nu_bar_from_bar_dot = bar_tangent
    spitzer_without_nu = jnp.sum(
        (geom.jacobian / geom.volume_prime)
        * (geom.grid.dtheta * geom.grid.dzeta)
        * 2.0
        * geom.b**2
        / (3.0 * geom.b0**2)
    )
    nu_dot = nu_bar_from_bar_dot + (
        2.0 * coefficient_bars[:, 4] * spitzer_without_nu * nu_hat_dot / nu_hat**3
    )
    return base, (f1_dot, f3_dot, nu_dot)


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
        raise ValueError("lambda1 and lambda3 must have shape (mode, unknown, n_rhs).")

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
        raise ValueError("lambda1 and lambda3 must have shape (mode, unknown, n_rhs).")
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
            lower_bars.append(jnp.zeros_like(diagonal_bars[-1]))
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
        (
            diagonal_theta_bar * (-theta_factor / jacobian)[None, None]
            + diagonal_zeta_bar * (-zeta_factor / jacobian)[None, None]
        ),
        axis=0,
    )
    b2_mean_bar = jnp.sum(
        (
            diagonal_theta_bar * (-theta_factor / b2_mean)[None, None]
            + diagonal_zeta_bar * (-zeta_factor / b2_mean)[None, None]
        ),
        axis=(0, 2, 3),
    )
    epsi_bar = jnp.sum(
        diagonal_theta_bar * (-bsubz / denominator)[None, None]
        + diagonal_zeta_bar * (bsubt / denominator)[None, None],
        axis=(0, 2, 3),
    )
    nu_bar = jnp.sum(diagonal[:, :, 2] * (0.5 * mode * (mode + 1.0)), axis=(0, 2, 3))
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
    drift_bar = jnp.swapaxes(-(2.0 / 3.0) * lambda1[0] - (1.0 / 3.0) * lambda1[2], 0, 1)
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
    drift_bar = q[:, 0, None, None] * pref * drift11[None] / g.coefficient_psi_scale**2 + q[
        :, 2, None, None
    ] * pref * drift13[None] / (g.coefficient_psi_scale * g.b0)
    b_bar = (
        q[:, 1, None, None] * pref * 2.0 * f11[None] / (3.0 * g.b0 * g.coefficient_psi_scale)
        + q[:, 3, None, None] * pref * 2.0 * f31[None] / (3.0 * g.b0**2)
        + q[:, 4, None, None] * pref * 4.0 * g.b[None] / (3.0 * g.b0**2 * nu_hat)
    )
    psi_bar = (
        -2.0 * q[:, 0] * values[0] / g.coefficient_psi_scale
        - q[:, 1] * values[1] / g.coefficient_psi_scale
        - q[:, 2] * values[2] / g.coefficient_psi_scale
    )
    b0_bar = (
        -q[:, 1] * values[1] / g.b0
        - q[:, 2] * values[2] / g.b0
        - 2.0 * q[:, 3] * values[3] / g.b0
        - 2.0 * q[:, 4] * values[4] / g.b0
    )
    nu_bar = -q[:, 4] * values[4] / nu_hat
    return dict(
        radial_drift_spatial=drift_bar,
        jacobian=jacobian_bar,
        volume_prime=volume_prime_bar,
        coefficient_psi_scale=psi_bar,
        b=b_bar,
        b0=b0_bar,
        nu_hat=nu_bar,
    )


def _native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bars: Array,
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

    def unflat(value):
        return jnp.swapaxes(value.reshape(value.shape[0], nz, nt), -1, -2)

    source_bars = {"b": unflat(source_b), "radial_drift_spatial": unflat(source_drift)}
    direct_bars = _direct_coefficient_geometry_bars_multi_rhs(
        prepared, f1_full[:3], f3_full[:3], ctx.nu_hat, coefficient_bars
    )
    return add_sampled_field_bars_multi_rhs(parameter_bars, source_bars, direct_bars)


def _directional_native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs_direct(
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
    coefficient_bars: Array,
    coefficient_bars_dot: Array,
) -> dict[str, Array]:
    """Exact product-rule form of the directional native VMEC primitive bars.

    This is the product-rule derivative of
    :func:`_native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs`.
    It replaces only the generic forward-mode transformation of the
    post-adjoint primitive-bar algebra: the NTX primal, factorisation and
    matrix-RHS adjoint fields are inputs and are not rebuilt here.

    The VMEC sampled-field/Fourier transpose is linear for the fixed prepared
    surface, so all primitive directional bars are accumulated first and
    converted once at the end.
    """

    def _add(*bar_dicts: dict[str, Array]) -> dict[str, Array]:
        return add_sampled_field_bars_multi_rhs(*bar_dicts)

    def _subtract(first: dict[str, Array], second: dict[str, Array]) -> dict[str, Array]:
        return {
            name: jnp.asarray(first.get(name, 0.0) - second.get(name, 0.0))
            for name in first.keys() | second.keys()
        }

    def _scale(bar_dict: dict[str, Array], scalar: Array) -> dict[str, Array]:
        return {name: scalar * value for name, value in bar_dict.items()}

    ctx = _operator_context(prepared.surface, prepared.geometry, prepared.grid, nu_hat, epsi_hat)

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
        for primal_part, adjoint_part in zip(block_from_primal, block_from_adjoint, strict=True)
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
    diagonal_field = jnp.swapaxes(diagonal_base.reshape(n_modes, n_rhs, 3, nz, nt), -1, -2)
    diagonal_theta_bar = diagonal_field[:, :, 0]
    diagonal_zeta_bar = diagonal_field[:, :, 1]
    denominator = prepared.geometry.jacobian * prepared.geometry.b2_mean
    parameter_epsi_slope = {
        "b_sub_zeta": jnp.sum(diagonal_theta_bar * (-1.0 / denominator)[None, None], axis=0),
        "b_sub_theta": jnp.sum(diagonal_zeta_bar * (1.0 / denominator)[None, None], axis=0),
        "jacobian": jnp.sum(
            (
                diagonal_theta_bar
                * (prepared.geometry.b_sub_zeta / (denominator * prepared.geometry.jacobian))[
                    None, None
                ]
                + diagonal_zeta_bar
                * (-prepared.geometry.b_sub_theta / (denominator * prepared.geometry.jacobian))[
                    None, None
                ]
            ),
            axis=0,
        ),
        "b2_mean": jnp.sum(
            (
                diagonal_theta_bar * (prepared.geometry.b_sub_zeta / denominator)[None, None]
                + diagonal_zeta_bar * (-prepared.geometry.b_sub_theta / denominator)[None, None]
            )
            / prepared.geometry.b2_mean,
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
            source_drift_dot.reshape(source_drift_dot.shape[0], *prepared.geometry.b.shape[::-1]),
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
    direct_from_nu["nu_hat"] = -2.0 * nu_hat_dot / nu_hat * direct_zero["nu_hat"]
    direct_dot = _add(direct_from_primal, direct_from_bar, direct_from_nu)

    primitive_dot = _add(parameter_dot, source_dot, direct_dot)
    return primitive_dot


def _geometry_gradient_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx: OperatorContext,
    f1_full: Array,
    f3_full: Array,
    lambda1: Array,
    lambda3: Array,
    coefficient_bar: Array,
):
    """Transpose the fixed residual and direct coefficients to geometry only.

    Reuse the solved modes and adjoints: ``g_bar = direct_bar - lambda.T F_g``.
    Resolved epsilon is held fixed; callers own the outer support chain.
    """

    def _residual_and_direct_coefficients(geometry):
        local_ctx = OperatorContext(
            surface=prepared.surface, geometry=geometry, nu_hat=ctx.nu_hat, epsi_hat=ctx.epsi_hat
        )
        inputs = (geometry, prepared.d_theta, prepared.d_zeta, block_parameters(local_ctx))
        return _compact_residual_and_direct_coefficients(
            inputs,
            surface=prepared.surface,
            n_xi=int(prepared.grid.n_xi),
            f1_full=f1_full,
            f3_full=f3_full,
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
    """Transpose the same fixed residual to all differentiable prepared leaves.

    This VJP includes surface, geometry and derivative operators without
    differentiating a factorization or solving another primal/adjoint system.
    """

    def _residual_and_direct_coefficients(prepared_value):
        return _compact_residual_and_direct_coefficients(
            _compact_prepared_residual_inputs(prepared_value, ctx.nu_hat, ctx.epsi_hat),
            surface=prepared_value.surface,
            n_xi=int(prepared.grid.n_xi),
            f1_full=f1_full,
            f3_full=f3_full,
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
    """Transpose the fixed residual to the compact input tuple.

    Call ``_compact_prepared_bar_to_prepared`` once after combining bars to
    finish the prepared-input chain without another dense residual transpose.
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
    """Chain compact-input bars to prepared leaves, summing shared geometry paths."""
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


# Factorized derivative kernels and public views
# ----------------------------------------------


def _solve_factorized_adjoint_field_pair(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    first_field_rhs: Array,
    second_field_rhs: Array,
) -> tuple[Array, Array]:
    """Solve two compatible adjoint field RHS through one matrix-RHS call.

    The leading block and unknown axes are shared and a trailing RHS-column
    axis is required. It is deliberately independent of the scalar low-dot
    helper; scalar vector RHS must use the established scalar solve.
    """

    if first_field_rhs.ndim != 3 or second_field_rhs.shape != first_field_rhs.shape:
        raise ValueError("Paired adjoint RHS must both have shape (mode, unknown, n_rhs).")
    packed = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.concatenate([first_field_rhs, second_field_rhs], axis=-1),
    )
    first_solution, second_solution = jnp.split(packed, 2, axis=-1)
    return first_solution, second_solution


def _solve_factorized_directional_adjoint_field_pair(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    first_field_adjoint: Array,
    second_field_adjoint: Array,
    first_field_rhs_dot: Array,
    second_field_rhs_dot: Array,
    diagonal_dot: Array,
) -> tuple[Array, Array]:
    """Solve paired directional adjoints for a trailing direction/RHS axis.

    ``diagonal_dot`` has layout ``(mode, direction, row, column)`` and the
    four field arrays have layout ``(mode, unknown, direction)``.  The RHS is
    the exact scalar formula ``g_dot - diagonal_dot.T @ lambda`` evaluated
    independently for each direction, then packed only for the factorized
    transpose solve.
    """

    expected_shape = first_field_adjoint.shape
    if (
        second_field_adjoint.shape != expected_shape
        or first_field_rhs_dot.shape != expected_shape
        or second_field_rhs_dot.shape != expected_shape
        or diagonal_dot.shape
        != (expected_shape[0], expected_shape[-1], expected_shape[1], expected_shape[1])
    ):
        raise ValueError("Directional paired-adjoint inputs have incompatible shapes.")
    first_rhs = first_field_rhs_dot - jnp.einsum("mdji,mjd->mid", diagonal_dot, first_field_adjoint)
    second_rhs = second_field_rhs_dot - jnp.einsum(
        "mdji,mjd->mid", diagonal_dot, second_field_adjoint
    )
    return _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        first_rhs,
        second_rhs,
    )


def _solve_factorized_multi_rhs_directional_adjoint_field_pair(
    saved_lu: Array,
    saved_piv: Array,
    saved_lower: Array,
    saved_upper: Array,
    first_field_adjoint: Array,
    second_field_adjoint: Array,
    first_field_rhs_dot: Array,
    second_field_rhs_dot: Array,
    diagonal_dot: Array,
) -> tuple[Array, Array]:
    """Solve directional adjoints for objective RHS columns in one call.

    This is the matrix-RHS counterpart of
    :func:`_solve_factorized_directional_adjoint_field_pair`.  The trailing
    axes are ``(rhs, direction)`` rather than only ``(direction,)``:

    - fields: ``(mode, unknown, rhs, direction)``;
    - diagonal derivatives: ``(mode, direction, row, column)``.

    The physical directions are shared by every objective RHS.  They are
    packed with the objective columns only at the block-tridiagonal transpose
    solve, then restored immediately.  In particular, this is not a ``vmap``
    around scalar implicit solves and it keeps no factorisation outside the
    caller's local anchor calculation.
    """

    expected_shape = first_field_adjoint.shape
    if (
        first_field_adjoint.ndim != 4
        or second_field_adjoint.shape != expected_shape
        or first_field_rhs_dot.shape != expected_shape
        or second_field_rhs_dot.shape != expected_shape
        or diagonal_dot.shape
        != (
            expected_shape[0],
            expected_shape[-1],
            expected_shape[1],
            expected_shape[1],
        )
    ):
        raise ValueError("Multi-RHS directional paired-adjoint inputs have incompatible shapes.")

    # ``diagonal_dot`` has no RHS axis: each physical case direction acts on
    # every objective adjoint column.  Keep that broadcasting explicit so a
    # later caller cannot accidentally introduce an objective-dependent
    # physical direction.
    first_rhs = first_field_rhs_dot - jnp.einsum(
        "mdji,mjrd->mird", diagonal_dot, first_field_adjoint
    )
    second_rhs = second_field_rhs_dot - jnp.einsum(
        "mdji,mjrd->mird", diagonal_dot, second_field_adjoint
    )
    mode_count, unknown_count, rhs_count, direction_count = first_rhs.shape
    first_packed = jnp.reshape(first_rhs, (mode_count, unknown_count, rhs_count * direction_count))
    second_packed = jnp.reshape(
        second_rhs, (mode_count, unknown_count, rhs_count * direction_count)
    )
    first_solution, second_solution = _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        first_packed,
        second_packed,
    )
    return (
        jnp.reshape(first_solution, first_rhs.shape),
        jnp.reshape(second_solution, second_rhs.shape),
    )


def _prepared_primal_context(prepared, case):
    """Share local primal setup without adding solves or JIT boundaries."""
    transport_scale = prepared.geometry.transport_psi_scale
    epsi = case.resolved_epsi_hat(transport_scale)
    primal = _prepared_implicit_vjp_primal(prepared, case.nu_hat, epsi)
    ctx = _operator_context(prepared.surface, prepared.geometry, prepared.grid, case.nu_hat, epsi)
    return transport_scale, ctx, primal


def pullback_prepared_coefficient_vector_case_and_geometry(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, GeometryOnGrid]:
    """Exact grouped implicit pullback for case/profile and NTX geometry.

    Unlike :func:`solve_prepared_coefficient_vector_vjp`, this is an explicit
    pullback API. It retains the existing custom-VJP contract (``prepared`` is
    non-differentiable there), while exposing the exact prepared-geometry
    cotangent required by callers that already manage a higher-level support
    payload. Geometry bars hold resolved epsilon fixed; the complete prepared
    API also includes the ``er_hat / transport_psi_scale`` chain rule.
    No primal or adjoint system is solved twice.
    """

    transport_scale, ctx, primal = _prepared_primal_context(prepared, case)
    (_coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper) = primal
    f1_bar_low, f3_bar_low, nu_hat_direct_bar = _coefficient_mode_pullback(
        prepared.geometry,
        f1_full[:3],
        f3_full[:3],
        ctx.nu_hat,
        coefficient_bar,
    )
    g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
    g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
    lambda1 = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        g1,
    )
    lambda3 = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        g3,
    )
    nu_hat_implicit_bar, epsi_hat_bar = _parameter_gradient_from_adjoint(
        prepared, ctx, f1_full, f3_full, lambda1, lambda3
    )
    geometry_bar = _geometry_gradient_from_adjoint(
        prepared, ctx, f1_full, f3_full, lambda1, lambda3, coefficient_bar
    )
    case_bar = _case_cotangent_from_resolved_epsi_bar(
        case,
        nu_hat_bar=nu_hat_direct_bar + nu_hat_implicit_bar,
        epsi_hat_bar=epsi_hat_bar,
        transport_scale=transport_scale,
    )
    return case_bar, geometry_bar


def pullback_prepared_coefficient_vector_case_and_prepared(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, PreparedMonoenergeticSystem]:
    """Exact grouped implicit pullback for case and complete prepared support.

    The returned prepared cotangent includes every differentiable leaf of the
    prepared NTX system.  It reuses the primal factorization and transpose
    solutions exactly as the geometry-only API does.
    """
    transport_scale, ctx, primal = _prepared_primal_context(prepared, case)
    (_coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper) = primal
    f1_bar_low, f3_bar_low, nu_hat_direct_bar = _coefficient_mode_pullback(
        prepared.geometry, f1_full[:3], f3_full[:3], ctx.nu_hat, coefficient_bar
    )
    g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
    g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
    lambda1 = _solve_factorized_adjoint(saved_lu, saved_piv, saved_lower, saved_upper, g1)
    lambda3 = _solve_factorized_adjoint(saved_lu, saved_piv, saved_lower, saved_upper, g3)
    nu_hat_implicit_bar, epsi_hat_bar = _parameter_gradient_from_adjoint(
        prepared, ctx, f1_full, f3_full, lambda1, lambda3
    )
    prepared_bar = _prepared_gradient_from_adjoint(
        prepared, ctx, f1_full, f3_full, lambda1, lambda3, coefficient_bar
    )
    prepared_bar = _add_er_coordinate_scale_bar(prepared_bar, case, epsi_hat_bar, transport_scale)
    case_bar = _case_cotangent_from_resolved_epsi_bar(
        case,
        nu_hat_bar=nu_hat_direct_bar + nu_hat_implicit_bar,
        epsi_hat_bar=epsi_hat_bar,
        transport_scale=transport_scale,
    )
    return case_bar, prepared_bar


def pullback_prepared_coefficient_vector_case_and_prepared_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_bars: Array,
) -> tuple[MonoenergeticCase, PreparedMonoenergeticSystem]:
    """Transpose coefficient rows with one primal and matrix-RHS adjoint solve.

    Case and prepared bars have the leading objective-RHS axis supplied in
    ``coefficient_bars``. Only base coefficients are handled here; the low-dot
    helpers additionally include their directional terms.
    """

    coefficient_bars = jnp.asarray(coefficient_bars)
    if coefficient_bars.ndim != 2:
        raise ValueError("coefficient_bars must have shape (n_rhs, n_coefficients).")

    transport_scale, ctx, primal = _prepared_primal_context(prepared, case)
    (_coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper) = primal

    def _coefficient_bar_to_low_modes(one_coefficient_bar):
        return _coefficient_mode_pullback(
            prepared.geometry,
            f1_full[:3],
            f3_full[:3],
            ctx.nu_hat,
            one_coefficient_bar,
        )

    f1_bar_low, f3_bar_low, nu_hat_direct_bars = jax.vmap(_coefficient_bar_to_low_modes)(
        coefficient_bars
    )
    rhs_count = coefficient_bars.shape[0]
    g1 = (
        jnp.zeros((*f1_full.shape, rhs_count), dtype=f1_full.dtype)
        .at[:3]
        .set(jnp.moveaxis(f1_bar_low, 0, -1))
    )
    g3 = (
        jnp.zeros((*f3_full.shape, rhs_count), dtype=f3_full.dtype)
        .at[:3]
        .set(jnp.moveaxis(f3_bar_low, 0, -1))
    )
    lambda1, lambda3 = _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        g1,
        g3,
    )

    def _parameter_bars(carry, mode_index):
        nu_bar, epsi_bar = carry
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            mode_index,
            prepared.d_theta,
            prepared.d_zeta,
        )
        diagonal_nu = jax.lax.cond(
            mode_index == 0, _zero_first_row, lambda value: value, diagonal_nu
        )
        diagonal_epsi = jax.lax.cond(
            mode_index == 0, _zero_first_row, lambda value: value, diagonal_epsi
        )
        f1_mode = jax.lax.dynamic_index_in_dim(f1_full, mode_index, axis=0, keepdims=False)
        f3_mode = jax.lax.dynamic_index_in_dim(f3_full, mode_index, axis=0, keepdims=False)
        lambda1_mode = jax.lax.dynamic_index_in_dim(lambda1, mode_index, axis=0, keepdims=False)
        lambda3_mode = jax.lax.dynamic_index_in_dim(lambda3, mode_index, axis=0, keepdims=False)
        nu_bar = nu_bar - (
            jnp.einsum("nr,n->r", lambda1_mode, diagonal_nu @ f1_mode)
            + jnp.einsum("nr,n->r", lambda3_mode, diagonal_nu @ f3_mode)
        )
        epsi_bar = epsi_bar - (
            jnp.einsum("nr,n->r", lambda1_mode, diagonal_epsi @ f1_mode)
            + jnp.einsum("nr,n->r", lambda3_mode, diagonal_epsi @ f3_mode)
        )
        return (nu_bar, epsi_bar), None

    (nu_hat_implicit_bars, epsi_hat_bars), _ = jax.lax.scan(
        _parameter_bars,
        (
            jnp.zeros((rhs_count,), dtype=prepared.grid.jax_dtype),
            jnp.zeros((rhs_count,), dtype=prepared.grid.jax_dtype),
        ),
        jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32),
    )
    # ``PreparedMonoenergeticSystem`` has static metadata leaves.  Batch only
    # its dynamic cotangent leaves, then rebuild the original pytree, rather
    # than asking ``vmap`` to construct a batched instance with static fields.
    prepared_bar_tree = jax.tree_util.tree_structure(prepared)

    def _prepared_bar_leaves(one_lambda1, one_lambda3, one_coefficient_bar):
        prepared_bar = _prepared_gradient_from_adjoint(
            prepared,
            ctx,
            f1_full,
            f3_full,
            one_lambda1,
            one_lambda3,
            one_coefficient_bar,
        )
        return tuple(jax.tree_util.tree_leaves(prepared_bar))

    batched_prepared_bar_leaves = jax.vmap(
        _prepared_bar_leaves,
        in_axes=(2, 2, 0),
    )(lambda1, lambda3, coefficient_bars)
    prepared_bars = jax.tree_util.tree_unflatten(
        prepared_bar_tree,
        batched_prepared_bar_leaves,
    )

    prepared_bars = _add_er_coordinate_scale_bar(
        prepared_bars, case, epsi_hat_bars, transport_scale
    )
    case_bars = _case_cotangent_from_resolved_epsi_bar(
        case,
        nu_hat_bar=nu_hat_direct_bars + nu_hat_implicit_bars,
        epsi_hat_bar=epsi_hat_bars,
        transport_scale=transport_scale,
    )
    return case_bars, prepared_bars


def pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_bars: Array,
) -> tuple[MonoenergeticCase, dict[str, Array]]:
    """Database-only base-coefficient transpose to GeometryOnGrid bars.

    This is intentionally separate from the Lij low-dot helpers.  It shares
    one prepared primal and factorization across a leading table-bar RHS axis.
    Returning geometry bars lets a scan caller add every ``(nu, Er)`` case
    before one exact geometry-to-surface VJP, without materialising an
    RHS-batched prepared-system cotangent.
    """
    coefficient_bars = jnp.asarray(coefficient_bars)
    if coefficient_bars.ndim != 2:
        raise ValueError("coefficient_bars must have shape (n_rhs, n_coefficients).")
    if case.epsi_hat is None:
        raise ValueError("The database VMEC multi-RHS transpose requires explicit epsi_hat.")
    _, ctx, primal = _prepared_primal_context(prepared, case)
    (_coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper) = primal

    def _low_modes(one_bar):
        return _coefficient_mode_pullback(
            prepared.geometry, f1_full[:3], f3_full[:3], ctx.nu_hat, one_bar
        )

    f1_bar_low, f3_bar_low, nu_direct = jax.vmap(_low_modes)(coefficient_bars)
    rhs_count = coefficient_bars.shape[0]
    g1 = (
        jnp.zeros((*f1_full.shape, rhs_count), dtype=f1_full.dtype)
        .at[:3]
        .set(jnp.moveaxis(f1_bar_low, 0, -1))
    )
    g3 = (
        jnp.zeros((*f3_full.shape, rhs_count), dtype=f3_full.dtype)
        .at[:3]
        .set(jnp.moveaxis(f3_bar_low, 0, -1))
    )
    lambda1, lambda3 = _solve_factorized_adjoint_field_pair(
        saved_lu, saved_piv, saved_lower, saved_upper, g1, g3
    )

    def _parameter_bars(carry, mode_index):
        nu_bar, epsi_bar = carry
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx, mode_index, prepared.d_theta, prepared.d_zeta
        )
        diagonal_nu = jax.lax.cond(
            mode_index == 0, _zero_first_row, lambda value: value, diagonal_nu
        )
        diagonal_epsi = jax.lax.cond(
            mode_index == 0, _zero_first_row, lambda value: value, diagonal_epsi
        )
        f1_mode = jax.lax.dynamic_index_in_dim(f1_full, mode_index, axis=0, keepdims=False)
        f3_mode = jax.lax.dynamic_index_in_dim(f3_full, mode_index, axis=0, keepdims=False)
        lambda1_mode = jax.lax.dynamic_index_in_dim(lambda1, mode_index, axis=0, keepdims=False)
        lambda3_mode = jax.lax.dynamic_index_in_dim(lambda3, mode_index, axis=0, keepdims=False)
        return (
            nu_bar
            - jnp.einsum("nr,n->r", lambda1_mode, diagonal_nu @ f1_mode)
            - jnp.einsum("nr,n->r", lambda3_mode, diagonal_nu @ f3_mode),
            epsi_bar
            - jnp.einsum("nr,n->r", lambda1_mode, diagonal_epsi @ f1_mode)
            - jnp.einsum("nr,n->r", lambda3_mode, diagonal_epsi @ f3_mode),
        ), None

    (nu_implicit, epsi_bars), _ = jax.lax.scan(
        _parameter_bars,
        (
            jnp.zeros((rhs_count,), dtype=prepared.grid.jax_dtype),
            jnp.zeros((rhs_count,), dtype=prepared.grid.jax_dtype),
        ),
        jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32),
    )
    # Keep the fixed primal/factorisation and matrix-RHS adjoints above, but
    # use the exact geometry-only residual VJP here.  The earlier analytic
    # primitive contraction is fast for one case yet accumulates a measurable
    # Jacobian error across the different epsi values of a database scan.
    # This VJP is over ``GeometryOnGrid`` only: it does not materialise a
    # prepared-system cotangent or re-solve NTX.
    geometry_bars = jax.vmap(
        lambda lambda1_value, lambda3_value, coefficient_bar_value: _geometry_gradient_from_adjoint(
            prepared,
            ctx,
            f1_full,
            f3_full,
            lambda1_value,
            lambda3_value,
            coefficient_bar_value,
        ),
        in_axes=(2, 2, 0),
    )(lambda1, lambda3, coefficient_bars)
    return (
        MonoenergeticCase(
            nu_hat=nu_direct + nu_implicit,
            epsi_hat=epsi_bars,
            er_hat=None,
        ),
        geometry_bars,
    )


def pullback_prepared_database_coefficients_to_vmec_multi_rhs(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    coefficient_bars: Array,
) -> tuple[MonoenergeticCase, dict[str, Array]]:
    """Database-only base-coefficient transpose directly to VMEC bars.

    This convenience wrapper preserves the compact per-case API.  Recorded
    scan transposes should instead use the primitive helper above and defer
    their Fourier contraction until all scan cases have been accumulated.
    """

    case_bars, geometry_bars = pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs(
        prepared, case, coefficient_bars
    )
    return (
        case_bars,
        vmec_geometry_bars_to_coefficients_multi_rhs(
            cast(VmecSurface, prepared.surface),
            prepared.geometry,
            {
                name: getattr(geometry_bars, name)
                for name in (
                    "b",
                    "d_b_dtheta",
                    "d_b_dzeta",
                    "jacobian",
                    "b_sub_theta",
                    "b_sub_zeta",
                    "b_sup_theta",
                    "b_sup_zeta",
                    "volume_prime",
                    "b2_mean",
                    "radial_drift_spatial",
                    "b0",
                )
            },
        ),
    )


def _zero_first_row(block: Array) -> Array:
    return block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))


def _parameter_gradient_directional_from_adjoint(
    prepared: PreparedMonoenergeticSystem,
    ctx,
    f1_full: Array,
    f3_full: Array,
    f1_dot: Array,
    f3_dot: Array,
    lambda1: Array,
    lambda3: Array,
    lambda1_dot: Array,
    lambda3_dot: Array,
) -> tuple[Array, Array]:
    nu_bar_dot = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    epsi_bar_dot = jnp.asarray(0.0, dtype=prepared.grid.jax_dtype)
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = _zero_first_row(diagonal_nu)
            diagonal_epsi = _zero_first_row(diagonal_epsi)
        nu_bar_dot = nu_bar_dot - (
            jnp.vdot(lambda1_dot[k], diagonal_nu @ f1_full[k])
            + jnp.vdot(lambda1[k], diagonal_nu @ f1_dot[k])
            + jnp.vdot(lambda3_dot[k], diagonal_nu @ f3_full[k])
            + jnp.vdot(lambda3[k], diagonal_nu @ f3_dot[k])
        )
        epsi_bar_dot = epsi_bar_dot - (
            jnp.vdot(lambda1_dot[k], diagonal_epsi @ f1_full[k])
            + jnp.vdot(lambda1[k], diagonal_epsi @ f1_dot[k])
            + jnp.vdot(lambda3_dot[k], diagonal_epsi @ f3_full[k])
            + jnp.vdot(lambda3[k], diagonal_epsi @ f3_dot[k])
        )
    return nu_bar_dot, epsi_bar_dot


def _pullback_prepared_coefficient_vector_case_and_geometry_derivative_core(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    case_dot: MonoenergeticCase,
    coefficient_bar: Array,
    *,
    include_geometry: bool,
) -> tuple[
    MonoenergeticCase,
    MonoenergeticCase,
    GeometryOnGrid | None,
    GeometryOnGrid | None,
]:
    """Return exact case/profile and geometry bars for a coefficient direction.

    This is the reverse-mode companion needed by callers that differentiate
    coefficient-solve derivative fields. It avoids tracing a JVP through the LU
    factorization by differentiating the implicit primal/adjoint systems
    directly for one monoenergetic case. The geometry bars reuse those same
    primal and adjoint quantities.
    """

    transport_scale = prepared.geometry.transport_psi_scale
    resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
    nu_hat_dot = jnp.asarray(case_dot.nu_hat)
    if case.epsi_hat is not None:
        epsi_hat_dot = jnp.asarray(case_dot.epsi_hat)
    elif case.er_hat is not None:
        assert transport_scale is not None
        epsi_hat_dot = jnp.asarray(case_dot.er_hat) / jnp.asarray(transport_scale)
    else:
        epsi_hat_dot = jnp.zeros_like(resolved_epsi_hat)

    (
        _coefficients,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
    ) = _prepared_implicit_vjp_primal(
        prepared,
        case.nu_hat,
        resolved_epsi_hat,
    )
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        resolved_epsi_hat,
    )

    source1_dot = []
    source3_dot = []
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = _zero_first_row(diagonal_nu)
            diagonal_epsi = _zero_first_row(diagonal_epsi)
        diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
        source1_dot.append(-(diagonal_dot @ f1_full[k]))
        source3_dot.append(-(diagonal_dot @ f3_full[k]))

    f1_dot = _solve_factorized_modes(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(source1_dot),
    )
    f3_dot = _solve_factorized_modes(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(source3_dot),
    )

    def _coefficient_pullback(modes1, modes3, nu_value):
        return _coefficient_mode_pullback(
            prepared.geometry,
            modes1,
            modes3,
            nu_value,
            coefficient_bar,
        )

    (
        (
            f1_bar_low,
            f3_bar_low,
            nu_bar_direct,
        ),
        (
            f1_bar_low_dot,
            f3_bar_low_dot,
            nu_bar_direct_dot,
        ),
    ) = jax.jvp(
        _coefficient_pullback,
        (f1_full[:3], f3_full[:3], ctx.nu_hat),
        (f1_dot[:3], f3_dot[:3], nu_hat_dot),
    )

    g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
    g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
    g1_dot = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low_dot)
    g3_dot = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low_dot)

    lambda1 = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        g1,
    )
    lambda3 = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        g3,
    )

    adjoint_rhs1_dot = []
    adjoint_rhs3_dot = []
    for k in range(prepared.grid.n_xi + 1):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        if k == 0:
            diagonal_nu = _zero_first_row(diagonal_nu)
            diagonal_epsi = _zero_first_row(diagonal_epsi)
        diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
        adjoint_rhs1_dot.append(g1_dot[k] - diagonal_dot.T @ lambda1[k])
        adjoint_rhs3_dot.append(g3_dot[k] - diagonal_dot.T @ lambda3[k])

    lambda1_dot = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(adjoint_rhs1_dot),
    )
    lambda3_dot = _solve_factorized_adjoint(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.stack(adjoint_rhs3_dot),
    )

    nu_bar_implicit, epsi_bar = _parameter_gradient_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        lambda1,
        lambda3,
    )
    nu_bar_implicit_dot, epsi_bar_dot = _parameter_gradient_directional_from_adjoint(
        prepared,
        ctx,
        f1_full,
        f3_full,
        f1_dot,
        f3_dot,
        lambda1,
        lambda3,
        lambda1_dot,
        lambda3_dot,
    )

    nu_bar = nu_bar_direct + nu_bar_implicit
    nu_bar_dot = nu_bar_direct_dot + nu_bar_implicit_dot
    if include_geometry:
        base_geometry_bar, directional_geometry_bar = _directional_geometry_gradient_from_adjoint(
            prepared,
            nu_hat=ctx.nu_hat,
            epsi_hat=ctx.epsi_hat,
            nu_hat_dot=nu_hat_dot,
            epsi_hat_dot=epsi_hat_dot,
            f1_full=f1_full,
            f3_full=f3_full,
            f1_dot=f1_dot,
            f3_dot=f3_dot,
            lambda1=lambda1,
            lambda3=lambda3,
            lambda1_dot=lambda1_dot,
            lambda3_dot=lambda3_dot,
            coefficient_bar=coefficient_bar,
        )
    else:
        base_geometry_bar = None
        directional_geometry_bar = None
    case_bars = (
        _case_cotangent_from_resolved_epsi_bar(
            case, nu_hat_bar=nu_bar, epsi_hat_bar=epsi_bar, transport_scale=transport_scale
        ),
        _case_cotangent_from_resolved_epsi_bar(
            case, nu_hat_bar=nu_bar_dot, epsi_hat_bar=epsi_bar_dot, transport_scale=transport_scale
        ),
    )
    return (
        case_bars[0],
        case_bars[1],
        base_geometry_bar,
        directional_geometry_bar,
    )


def pullback_prepared_coefficient_vector_case_and_geometry_derivative(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    case_dot: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, MonoenergeticCase, GeometryOnGrid, GeometryOnGrid]:
    """Grouped directional pullback with geometry bars at fixed resolved epsilon."""

    base, tangent, base_geometry, tangent_geometry = (
        _pullback_prepared_coefficient_vector_case_and_geometry_derivative_core(
            prepared,
            case,
            case_dot,
            coefficient_bar,
            include_geometry=True,
        )
    )
    assert base_geometry is not None and tangent_geometry is not None
    return base, tangent, base_geometry, tangent_geometry


def solve_prepared_coefficient_vector_derivative_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    case_dot: MonoenergeticCase,
    coefficient_bar: Array,
) -> tuple[MonoenergeticCase, MonoenergeticCase]:
    """Return the coefficient pullback and its directional derivative.

    With a fixed ``coefficient_bar`` the result is ``(J.T @ bar,
    D(J.T @ bar)[case_dot])``. This preserves the established NEOPAX order;
    it is the reverse of the two input bars of ``vjp(jvp(f, case_dot))``.
    """

    base_case_bar, tangent_case_bar, _base_geometry_bar, _tangent_geometry_bar = (
        _pullback_prepared_coefficient_vector_case_and_geometry_derivative_core(
            prepared,
            case,
            case_dot,
            coefficient_bar,
            include_geometry=False,
        )
    )
    return base_case_bar, tangent_case_bar


def _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_fn,
    *,
    include_geometry: bool,
    include_prepared: bool = False,
    return_coefficient_aux: bool,
    packed_support_directional_adjoint: bool = False,
    support_only: bool = False,
    primal_residuals=None,
    return_primal_residuals: bool = False,
):
    """Fused exact pullback for two coefficient-derivative contractions.

    Higher-level transport models own the mapping from coefficient vectors to
    objective-specific coefficient cotangents, so this helper receives that
    mapping as a Python callback. NTX still owns the expensive implicit-solve
    pullback algebra and reuses one primal factorization for the base VJP and
    the two low-mode directional pullbacks. ``include_geometry`` is static:
    the legacy public API keeps its existing case-only output and does not
    build the prepared-geometry cotangents. When ``return_coefficient_aux``
    is true, ``coefficient_bar_fn`` must additionally return a pytree after
    the three coefficient cotangents; it is forwarded unchanged. This lets a
    transport caller carry exact direct prefactor bars (for example ``drds``)
    without another NTX solve.
    """

    if include_geometry and include_prepared:
        raise ValueError("Only one prepared-support cotangent representation may be requested.")
    include_support = include_geometry or include_prepared
    if support_only and not include_support:
        raise ValueError("support_only requires a prepared-support representation.")
    if packed_support_directional_adjoint and not include_support:
        raise ValueError("Packed support directional adjoint requires a support pullback mode.")

    from jax.scipy.linalg import lu_solve

    if primal_residuals is None:
        transport_scale = prepared.geometry.transport_psi_scale
        resolved_epsi_hat = case.resolved_epsi_hat(transport_scale)
        first_nu_dot = jnp.asarray(first_case_dot.nu_hat)
        second_nu_dot = jnp.asarray(second_case_dot.nu_hat)
        if first_case_dot.epsi_hat is not None:
            first_epsi_dot = jnp.asarray(first_case_dot.epsi_hat)
        elif first_case_dot.er_hat is not None:
            assert transport_scale is not None
            first_epsi_dot = jnp.asarray(first_case_dot.er_hat) / jnp.asarray(transport_scale)
        else:
            first_epsi_dot = jnp.zeros_like(resolved_epsi_hat)
        if second_case_dot.epsi_hat is not None:
            second_epsi_dot = jnp.asarray(second_case_dot.epsi_hat)
        elif second_case_dot.er_hat is not None:
            assert transport_scale is not None
            second_epsi_dot = jnp.asarray(second_case_dot.er_hat) / jnp.asarray(transport_scale)
        else:
            second_epsi_dot = jnp.zeros_like(resolved_epsi_hat)

        (
            coefficients,
            f1_full,
            f3_full,
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
        ) = _prepared_implicit_vjp_primal(
            prepared,
            case.nu_hat,
            resolved_epsi_hat,
        )
        ctx = _operator_context(
            prepared.surface,
            prepared.geometry,
            prepared.grid,
            case.nu_hat,
            resolved_epsi_hat,
        )
    else:
        (
            transport_scale,
            resolved_epsi_hat,
            first_nu_dot,
            first_epsi_dot,
            second_nu_dot,
            second_epsi_dot,
            coefficients,
            f1_full,
            f3_full,
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            packed_f_dot_low_matrix,
            first_coefficient_dot,
            second_coefficient_dot,
        ) = primal_residuals
        ctx = _operator_context(
            prepared.surface,
            prepared.geometry,
            prepared.grid,
            case.nu_hat,
            resolved_epsi_hat,
        )
    mode_indices = jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32)

    def _zero_first_row_if_needed(block, k):
        zeroed = block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))
        return jnp.where(jnp.asarray(k) == 0, zeroed, block)

    def _take_mode(values, k):
        return jax.lax.dynamic_index_in_dim(values, k, axis=0, keepdims=False)

    def _solve_factorized_low_modes_scan(source_for_mode):
        mode_count = saved_lu.shape[0]
        last_index = mode_count - 1
        source0 = source_for_mode(jnp.asarray(0, dtype=jnp.int32))
        zero = jnp.zeros_like(source0)
        y_last = lu_solve(
            (_take_mode(saved_lu, last_index), _take_mode(saved_piv, last_index)),
            source_for_mode(jnp.asarray(last_index, dtype=jnp.int32)),
        )
        y0 = jnp.where(last_index == 0, y_last, zero)
        y1 = jnp.where(last_index == 1, y_last, zero)
        y2 = jnp.where(last_index == 2, y_last, zero)

        def _backward_y(carry, k):
            y_next, y0_value, y1_value, y2_value = carry
            rhs = source_for_mode(k) - _take_mode(saved_upper, k) @ y_next
            y_k = lu_solve(
                (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                rhs,
            )
            y0_value = jnp.where(k == 0, y_k, y0_value)
            y1_value = jnp.where(k == 1, y_k, y1_value)
            y2_value = jnp.where(k == 2, y_k, y2_value)
            return (y_k, y0_value, y1_value, y2_value), None

        (_, y0, y1, y2), _ = jax.lax.scan(
            _backward_y,
            (y_last, y0, y1, y2),
            jnp.arange(last_index, dtype=jnp.int32),
            reverse=True,
        )
        mode0 = y0
        mode1 = y1 - lu_solve(
            (_take_mode(saved_lu, 1), _take_mode(saved_piv, 1)),
            _take_mode(saved_lower, 1) @ mode0,
        )
        mode2 = y2 - lu_solve(
            (_take_mode(saved_lu, 2), _take_mode(saved_piv, 2)),
            _take_mode(saved_lower, 2) @ mode1,
        )
        return jnp.stack([mode0, mode1, mode2], axis=0)

    def _solve_factorized_adjoint_scan(source_bar):
        mode_count = source_bar.shape[0]
        last_index = mode_count - 1
        mu_last = _take_mode(source_bar, last_index)

        def _backward_mu(mu_next, k):
            propagated = lu_solve(
                (_take_mode(saved_lu, k + 1), _take_mode(saved_piv, k + 1)),
                mu_next,
                trans=1,
            )
            mu_k = _take_mode(source_bar, k) - _take_mode(saved_lower, k + 1).T @ propagated
            return mu_k, mu_k

        _, mu_tail = jax.lax.scan(
            _backward_mu,
            mu_last,
            jnp.arange(last_index, dtype=jnp.int32),
            reverse=True,
        )
        mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
        adjoint0 = lu_solve(
            (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
            _take_mode(mu, 0),
            trans=1,
        )

        def _forward_adjoint(adjoint_prev, k):
            rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
            adjoint_k = lu_solve(
                (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                rhs,
                trans=1,
            )
            return adjoint_k, adjoint_k

        _, adjoint_tail = jax.lax.scan(
            _forward_adjoint,
            adjoint0,
            jnp.arange(1, mode_count, dtype=jnp.int32),
        )
        return jnp.concatenate([adjoint0[None, ...], adjoint_tail], axis=0)

    def _parameter_source_matrix_for_mode(k):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
        diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
        f1_k = _take_mode(f1_full, k)
        f3_k = _take_mode(f3_full, k)
        return jnp.stack(
            [
                jnp.stack([diagonal_nu @ f1_k, diagonal_nu @ f3_k], axis=-1),
                jnp.stack([diagonal_epsi @ f1_k, diagonal_epsi @ f3_k], axis=-1),
            ],
            axis=-1,
        )

    def _contract_factorized_parameter_sources_scan(source_bar_matrix_for_mode):
        mode_count = saved_lu.shape[0]
        last_index = mode_count - 1
        mu_last = source_bar_matrix_for_mode(jnp.asarray(last_index, dtype=jnp.int32))

        def _backward_mu(mu_next, k):
            propagated = lu_solve(
                (_take_mode(saved_lu, k + 1), _take_mode(saved_piv, k + 1)),
                mu_next,
                trans=1,
            )
            mu_k = source_bar_matrix_for_mode(k) - _take_mode(saved_lower, k + 1).T @ propagated
            return mu_k, mu_k

        _, mu_tail = jax.lax.scan(
            _backward_mu,
            mu_last,
            jnp.arange(last_index, dtype=jnp.int32),
            reverse=True,
        )
        mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
        adjoint0 = lu_solve(
            (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
            _take_mode(mu, 0),
            trans=1,
        )
        source0 = _parameter_source_matrix_for_mode(jnp.asarray(0, dtype=jnp.int32))
        contract0 = jnp.sum(adjoint0[..., None] * source0, axis=(0, 1))

        def _forward_adjoint(carry, k):
            adjoint_prev, contract = carry
            rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
            adjoint_k = lu_solve(
                (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                rhs,
                trans=1,
            )
            source_k = _parameter_source_matrix_for_mode(k)
            contract = contract + jnp.sum(adjoint_k[..., None] * source_k, axis=(0, 1))
            return (adjoint_k, contract), None

        (_, contracted), _ = jax.lax.scan(
            _forward_adjoint,
            (adjoint0, contract0),
            jnp.arange(1, mode_count, dtype=jnp.int32),
        )
        return contracted

    def _base_pullback(coefficient_bar):
        f1_bar_low, f3_bar_low, nu_bar_direct = _coefficient_mode_pullback(
            prepared.geometry,
            f1_full[:3],
            f3_full[:3],
            ctx.nu_hat,
            coefficient_bar,
        )
        g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
        g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)

        def _source_bar_matrix_for_mode(k):
            return jnp.stack([_take_mode(g1, k), _take_mode(g3, k)], axis=-1)

        if include_support:
            lambda1 = _solve_factorized_adjoint_scan(g1)
            lambda3 = _solve_factorized_adjoint_scan(g3)
            nu_bar_implicit, epsi_bar = _parameter_gradient_from_adjoint(
                prepared,
                ctx,
                f1_full,
                f3_full,
                lambda1,
                lambda3,
            )
            support_bar = (
                _geometry_gradient_from_adjoint(
                    prepared, ctx, f1_full, f3_full, lambda1, lambda3, coefficient_bar
                )
                if include_geometry
                else _prepared_gradient_from_adjoint(
                    prepared, ctx, f1_full, f3_full, lambda1, lambda3, coefficient_bar
                )
            )
            return nu_bar_direct + nu_bar_implicit, epsi_bar, support_bar

        nu_bar_implicit, epsi_bar = -_contract_factorized_parameter_sources_scan(
            _source_bar_matrix_for_mode
        )
        return nu_bar_direct + nu_bar_implicit, epsi_bar

    def _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx,
            k,
            prepared.d_theta,
            prepared.d_zeta,
        )
        diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
        diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
        diagonal_dot = nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi
        return (
            -(diagonal_dot @ _take_mode(f1_full, k)),
            -(diagonal_dot @ _take_mode(f3_full, k)),
        )

    def _packed_source_dot_matrix_for_mode(k):
        first_source1, first_source3 = _source_dot_pair_for_direction(
            k,
            first_nu_dot,
            first_epsi_dot,
        )
        second_source1, second_source3 = _source_dot_pair_for_direction(
            k,
            second_nu_dot,
            second_epsi_dot,
        )
        return jnp.stack(
            [first_source1, first_source3, second_source1, second_source3],
            axis=-1,
        )

    if primal_residuals is None:
        packed_f_dot_low_matrix = _solve_factorized_low_modes_scan(
            _packed_source_dot_matrix_for_mode
        )

        def _coefficient_vector_from_low_modes(modes1, modes3, nu_value):
            return jnp.stack(coefficients_from_modes(prepared.geometry, modes1, modes3, nu_value))

        _, first_coefficient_dot = jax.jvp(
            _coefficient_vector_from_low_modes,
            (f1_full[:3], f3_full[:3], ctx.nu_hat),
            (
                packed_f_dot_low_matrix[..., 0],
                packed_f_dot_low_matrix[..., 1],
                first_nu_dot,
            ),
        )
        _, second_coefficient_dot = jax.jvp(
            _coefficient_vector_from_low_modes,
            (f1_full[:3], f3_full[:3], ctx.nu_hat),
            (
                packed_f_dot_low_matrix[..., 2],
                packed_f_dot_low_matrix[..., 3],
                second_nu_dot,
            ),
        )
    local_primal_residuals = (
        transport_scale,
        resolved_epsi_hat,
        first_nu_dot,
        first_epsi_dot,
        second_nu_dot,
        second_epsi_dot,
        coefficients,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        packed_f_dot_low_matrix,
        first_coefficient_dot,
        second_coefficient_dot,
    )
    # This is also useful on its own for callers that need the primal
    # coefficient vector and two exact parameter directions.  Keep the
    # factorization-local calculation here: it is the same established
    # algebra used by the grouped pullback below, but stops before any
    # reverse contractions are formed.
    if coefficient_bar_fn is None:
        primal_outputs = (coefficients, first_coefficient_dot, second_coefficient_dot)
        return (
            (primal_outputs, local_primal_residuals) if return_primal_residuals else primal_outputs
        )
    coefficient_bar_result = (
        coefficient_bar_fn(
            coefficients,
            first_coefficient_dot,
            second_coefficient_dot,
        )
        if return_coefficient_aux
        else coefficient_bar_fn(coefficients)
    )
    base_coefficient_bar, first_coefficient_bar, second_coefficient_bar = coefficient_bar_result[:3]
    coefficient_aux = coefficient_bar_result[3] if return_coefficient_aux else None
    # A differentiated transport-moment pullback can make its coefficient
    # cotangent depend on the NTX coefficient primal.  Carry that mixed term
    # into the two low-dot reverse directions when supplied.  The optional
    # fifth result keeps every established caller on the previous zero-tangent
    # contract.
    if return_coefficient_aux and len(coefficient_bar_result) > 4:
        first_coefficient_bar_dot, second_coefficient_bar_dot = coefficient_bar_result[4]
    else:
        first_coefficient_bar_dot = jnp.zeros_like(first_coefficient_bar)
        second_coefficient_bar_dot = jnp.zeros_like(second_coefficient_bar)

    def _scan_direction_pullbacks(
        coefficient_bar_pair,
        coefficient_bar_dot_pair,
        nu_hat_dot_pair,
        epsi_hat_dot_pair,
        f1_dot_low_pair,
        f3_dot_low_pair,
    ):
        def _contract_factorized_source_bar_pair_scan(source_dot_for_mode, source_bar_for_mode):
            mode_count = saved_lu.shape[0]
            last_index = mode_count - 1
            mu_last = source_bar_for_mode(jnp.asarray(last_index, dtype=jnp.int32))

            def _backward_mu(mu_next, k):
                propagated = lu_solve(
                    (_take_mode(saved_lu, k + 1), _take_mode(saved_piv, k + 1)),
                    mu_next,
                    trans=1,
                )
                mu_k = source_bar_for_mode(k) - _take_mode(saved_lower, k + 1).T @ propagated
                return mu_k, mu_k

            _, mu_tail = jax.lax.scan(
                _backward_mu,
                mu_last,
                jnp.arange(last_index, dtype=jnp.int32),
                reverse=True,
            )
            mu = jnp.concatenate([mu_tail, mu_last[None, ...]], axis=0)
            adjoint0 = lu_solve(
                (_take_mode(saved_lu, 0), _take_mode(saved_piv, 0)),
                _take_mode(mu, 0),
                trans=1,
            )
            source0 = source_dot_for_mode(jnp.asarray(0, dtype=jnp.int32))
            contract0 = jnp.sum(adjoint0 * source0[:, None], axis=0)

            def _forward_adjoint(carry, k):
                adjoint_prev, contract = carry
                rhs = _take_mode(mu, k) - _take_mode(saved_upper, k - 1).T @ adjoint_prev
                adjoint_k = lu_solve(
                    (_take_mode(saved_lu, k), _take_mode(saved_piv, k)),
                    rhs,
                    trans=1,
                )
                source_k = source_dot_for_mode(k)
                contract = contract + jnp.sum(adjoint_k * source_k[:, None], axis=0)
                return (adjoint_k, contract), None

            (_, contracted), _ = jax.lax.scan(
                _forward_adjoint,
                (adjoint0, contract0),
                jnp.arange(1, mode_count, dtype=jnp.int32),
            )
            return contracted

        def _one_direction_pullback(xs):
            (
                coefficient_bar,
                coefficient_bar_dot,
                nu_hat_dot,
                epsi_hat_dot,
                f1_dot_low,
                f3_dot_low,
            ) = xs

            def _coefficient_pullback(modes1, modes3, nu_value, coefficient_bar_value):
                return _coefficient_mode_pullback(
                    prepared.geometry,
                    modes1,
                    modes3,
                    nu_value,
                    coefficient_bar_value,
                )

            (
                (
                    f1_bar_low,
                    f3_bar_low,
                    nu_bar_direct,
                ),
                (
                    f1_bar_low_dot,
                    f3_bar_low_dot,
                    nu_bar_direct_dot,
                ),
            ) = jax.jvp(
                _coefficient_pullback,
                (f1_full[:3], f3_full[:3], ctx.nu_hat, coefficient_bar),
                (f1_dot_low, f3_dot_low, nu_hat_dot, coefficient_bar_dot),
            )

            g1 = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low)
            g3 = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low)
            g1_dot = jnp.zeros_like(f1_full).at[:3].set(f1_bar_low_dot)
            g3_dot = jnp.zeros_like(f3_full).at[:3].set(f3_bar_low_dot)

            lambda1 = _solve_factorized_adjoint_scan(g1)
            lambda3 = _solve_factorized_adjoint_scan(g3)

            def _diagonal_dot(k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                return nu_hat_dot * diagonal_nu + epsi_hat_dot * diagonal_epsi

            def _adjoint_rhs_dot_for_mode(lambda_field, g_dot, k):
                return _take_mode(g_dot, k) - _diagonal_dot(k).T @ _take_mode(lambda_field, k)

            if support_only:
                # The support-only caller holds the local case fixed.  Do not
                # form any case/profile contractions: only the forward
                # directional fields and prepared-support adjoints are needed.
                source1_dot, source3_dot = jax.lax.map(
                    lambda k: _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot),
                    mode_indices,
                )
                f1_dot = _solve_factorized_modes(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source1_dot,
                )
                f3_dot = _solve_factorized_modes(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source3_dot,
                )
                lambda1_dot = _solve_factorized_adjoint_scan(
                    jax.lax.map(
                        lambda k: _adjoint_rhs_dot_for_mode(lambda1, g1_dot, k),
                        mode_indices,
                    )
                )
                lambda3_dot = _solve_factorized_adjoint_scan(
                    jax.lax.map(
                        lambda k: _adjoint_rhs_dot_for_mode(lambda3, g3_dot, k),
                        mode_indices,
                    )
                )
                base_support_bar, directional_support_bar = (
                    _directional_geometry_gradient_from_adjoint(
                        prepared,
                        nu_hat=ctx.nu_hat,
                        epsi_hat=ctx.epsi_hat,
                        nu_hat_dot=nu_hat_dot,
                        epsi_hat_dot=epsi_hat_dot,
                        f1_full=f1_full,
                        f3_full=f3_full,
                        f1_dot=f1_dot,
                        f3_dot=f3_dot,
                        lambda1=lambda1,
                        lambda3=lambda3,
                        lambda1_dot=lambda1_dot,
                        lambda3_dot=lambda3_dot,
                        coefficient_bar=coefficient_bar,
                        coefficient_bar_dot=coefficient_bar_dot,
                    )
                    if include_geometry
                    else _directional_prepared_gradient_from_adjoint(
                        prepared,
                        nu_hat=ctx.nu_hat,
                        epsi_hat=ctx.epsi_hat,
                        nu_hat_dot=nu_hat_dot,
                        epsi_hat_dot=epsi_hat_dot,
                        f1_full=f1_full,
                        f3_full=f3_full,
                        f1_dot=f1_dot,
                        f3_dot=f3_dot,
                        lambda1=lambda1,
                        lambda3=lambda3,
                        lambda1_dot=lambda1_dot,
                        lambda3_dot=lambda3_dot,
                        coefficient_bar=coefficient_bar,
                        coefficient_bar_dot=coefficient_bar_dot,
                    )
                )
                return base_support_bar, directional_support_bar

            def _accumulate_base_bars(carry, k):
                nu_bar, epsi_bar = carry
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                f1_k = _take_mode(f1_full, k)
                f3_k = _take_mode(f3_full, k)
                lambda1_k = _take_mode(lambda1, k)
                lambda3_k = _take_mode(lambda3, k)
                nu_bar = (
                    nu_bar - lambda1_k @ (diagonal_nu @ f1_k) - lambda3_k @ (diagonal_nu @ f3_k)
                )
                epsi_bar = (
                    epsi_bar
                    - lambda1_k @ (diagonal_epsi @ f1_k)
                    - lambda3_k @ (diagonal_epsi @ f3_k)
                )
                return (nu_bar, epsi_bar), None

            (
                (
                    nu_bar_implicit,
                    epsi_bar,
                ),
                _,
            ) = jax.lax.scan(
                _accumulate_base_bars,
                (
                    jnp.asarray(0, dtype=prepared.grid.jax_dtype),
                    jnp.asarray(0, dtype=prepared.grid.jax_dtype),
                ),
                mode_indices,
            )

            def _adjoint_rhs_dot_matrix_for_mode(k):
                return jnp.stack(
                    [
                        _adjoint_rhs_dot_for_mode(lambda1, g1_dot, k),
                        _adjoint_rhs_dot_for_mode(lambda3, g3_dot, k),
                    ],
                    axis=-1,
                )

            nu_bar_implicit_dot, epsi_bar_dot = -_contract_factorized_parameter_sources_scan(
                _adjoint_rhs_dot_matrix_for_mode
            )

            def _source_bar_pair_for_mode(lambdas, k):
                diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
                    ctx,
                    k,
                    prepared.d_theta,
                    prepared.d_zeta,
                )
                diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
                diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
                lambda_k = _take_mode(lambdas, k)
                return jnp.stack([diagonal_nu.T @ lambda_k, diagonal_epsi.T @ lambda_k], axis=-1)

            def source1_dot_for_mode(k):
                return _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot)[0]

            def source3_dot_for_mode(k):
                return _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot)[1]

            f1_field_dot = _contract_factorized_source_bar_pair_scan(
                source1_dot_for_mode,
                lambda k: _source_bar_pair_for_mode(lambda1, k),
            )
            f3_field_dot = _contract_factorized_source_bar_pair_scan(
                source3_dot_for_mode,
                lambda k: _source_bar_pair_for_mode(lambda3, k),
            )
            nu_bar_implicit_dot = nu_bar_implicit_dot - f1_field_dot[0] - f3_field_dot[0]
            epsi_bar_dot = epsi_bar_dot - f1_field_dot[1] - f3_field_dot[1]
            if include_support:
                source1_dot, source3_dot = jax.lax.map(
                    lambda k: _source_dot_pair_for_direction(k, nu_hat_dot, epsi_hat_dot),
                    mode_indices,
                )
                f1_dot = _solve_factorized_modes(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source1_dot,
                )
                f3_dot = _solve_factorized_modes(
                    saved_lu,
                    saved_piv,
                    saved_lower,
                    saved_upper,
                    source3_dot,
                )
                if packed_support_directional_adjoint:
                    diagonal_dot = jax.lax.map(_diagonal_dot, mode_indices)[:, None, :, :]
                    lambda1_dot_pair, lambda3_dot_pair = (
                        _solve_factorized_directional_adjoint_field_pair(
                            saved_lu,
                            saved_piv,
                            saved_lower,
                            saved_upper,
                            lambda1[..., None],
                            lambda3[..., None],
                            g1_dot[..., None],
                            g3_dot[..., None],
                            diagonal_dot,
                        )
                    )
                    lambda1_dot = lambda1_dot_pair[..., 0]
                    lambda3_dot = lambda3_dot_pair[..., 0]
                else:
                    lambda1_dot = _solve_factorized_adjoint_scan(
                        jax.lax.map(
                            lambda k: _adjoint_rhs_dot_for_mode(lambda1, g1_dot, k),
                            mode_indices,
                        )
                    )
                    lambda3_dot = _solve_factorized_adjoint_scan(
                        jax.lax.map(
                            lambda k: _adjoint_rhs_dot_for_mode(lambda3, g3_dot, k),
                            mode_indices,
                        )
                    )
                base_support_bar, directional_support_bar = (
                    _directional_geometry_gradient_from_adjoint(
                        prepared,
                        nu_hat=ctx.nu_hat,
                        epsi_hat=ctx.epsi_hat,
                        nu_hat_dot=nu_hat_dot,
                        epsi_hat_dot=epsi_hat_dot,
                        f1_full=f1_full,
                        f3_full=f3_full,
                        f1_dot=f1_dot,
                        f3_dot=f3_dot,
                        lambda1=lambda1,
                        lambda3=lambda3,
                        lambda1_dot=lambda1_dot,
                        lambda3_dot=lambda3_dot,
                        coefficient_bar=coefficient_bar,
                        coefficient_bar_dot=coefficient_bar_dot,
                    )
                    if include_geometry
                    else _directional_prepared_gradient_from_adjoint(
                        prepared,
                        nu_hat=ctx.nu_hat,
                        epsi_hat=ctx.epsi_hat,
                        nu_hat_dot=nu_hat_dot,
                        epsi_hat_dot=epsi_hat_dot,
                        f1_full=f1_full,
                        f3_full=f3_full,
                        f1_dot=f1_dot,
                        f3_dot=f3_dot,
                        lambda1=lambda1,
                        lambda3=lambda3,
                        lambda1_dot=lambda1_dot,
                        lambda3_dot=lambda3_dot,
                        coefficient_bar=coefficient_bar,
                        coefficient_bar_dot=coefficient_bar_dot,
                    )
                )
                return (
                    nu_bar_direct + nu_bar_implicit,
                    epsi_bar,
                    nu_bar_direct_dot + nu_bar_implicit_dot,
                    epsi_bar_dot,
                    base_support_bar,
                    directional_support_bar,
                )
            return (
                nu_bar_direct + nu_bar_implicit,
                epsi_bar,
                nu_bar_direct_dot + nu_bar_implicit_dot,
                epsi_bar_dot,
            )

        _, outputs = jax.lax.scan(
            lambda carry, xs: (carry, _one_direction_pullback(xs)),
            jnp.asarray(0, dtype=jnp.int32),
            (
                coefficient_bar_pair,
                coefficient_bar_dot_pair,
                nu_hat_dot_pair,
                epsi_hat_dot_pair,
                f1_dot_low_pair,
                f3_dot_low_pair,
            ),
        )
        return outputs

    base = _base_pullback(base_coefficient_bar)
    if include_prepared and case.er_hat is not None:
        base = (
            base[0],
            base[1],
            _add_er_coordinate_scale_bar(base[2], case, base[1], transport_scale),
        )
    direction = _scan_direction_pullbacks(
        jnp.stack([first_coefficient_bar, second_coefficient_bar], axis=0),
        jnp.stack([first_coefficient_bar_dot, second_coefficient_bar_dot], axis=0),
        jnp.stack([first_nu_dot, second_nu_dot], axis=0),
        jnp.stack([first_epsi_dot, second_epsi_dot], axis=0),
        jnp.stack([packed_f_dot_low_matrix[..., 0], packed_f_dot_low_matrix[..., 2]], axis=0),
        jnp.stack([packed_f_dot_low_matrix[..., 1], packed_f_dot_low_matrix[..., 3]], axis=0),
    )
    if support_only:
        direction_base_support_bar, direction_support_bar_dot = direction
    elif include_support:
        (
            direction_nu_bar,
            direction_epsi_bar,
            direction_nu_bar_dot,
            direction_epsi_bar_dot,
            direction_base_support_bar,
            direction_support_bar_dot,
        ) = direction
    else:
        (
            direction_nu_bar,
            direction_epsi_bar,
            direction_nu_bar_dot,
            direction_epsi_bar_dot,
        ) = direction
    if include_support:
        base_nu_bar, base_epsi_bar, base_support_bar = base

        def _take_direction_support(support_bars, index):
            return jax.tree_util.tree_map(
                lambda value: jax.lax.dynamic_index_in_dim(
                    value,
                    index,
                    axis=0,
                    keepdims=False,
                ),
                support_bars,
            )

        if support_only:
            return (
                base_support_bar,
                _take_direction_support(direction_base_support_bar, 0),
                _take_direction_support(direction_support_bar_dot, 0),
                _take_direction_support(direction_base_support_bar, 1),
                _take_direction_support(direction_support_bar_dot, 1),
                coefficient_aux,
            )

        first: tuple = (
            direction_nu_bar[0],
            direction_epsi_bar[0],
            direction_nu_bar_dot[0],
            direction_epsi_bar_dot[0],
        )
        second: tuple = (
            direction_nu_bar[1],
            direction_epsi_bar[1],
            direction_nu_bar_dot[1],
            direction_epsi_bar_dot[1],
        )
        first = (
            *first,
            _take_direction_support(direction_base_support_bar, 0),
            _take_direction_support(direction_support_bar_dot, 0),
        )
        second = (
            *second,
            _take_direction_support(direction_base_support_bar, 1),
            _take_direction_support(direction_support_bar_dot, 1),
        )
        result = (base_nu_bar, base_epsi_bar, base_support_bar, *first, *second)
        return (*result, coefficient_aux) if return_coefficient_aux else result
    first = (
        direction_nu_bar[0],
        direction_epsi_bar[0],
        direction_nu_bar_dot[0],
        direction_epsi_bar_dot[0],
    )
    second = (
        direction_nu_bar[1],
        direction_epsi_bar[1],
        direction_nu_bar_dot[1],
        direction_epsi_bar_dot[1],
    )
    return (*base, *first, *second)


def _unpack_coefficient_bars(result):
    """Read the common callback contract, preserving caller-owned auxiliary data."""
    if len(result) not in {4, 5}:
        raise ValueError(
            "coefficient_bars_and_aux_fn must return "
            "(base_bars, first_bars, second_bars, auxiliary) optionally followed "
            "by (first_bars_dot, second_bars_dot)."
        )
    first_dot, second_dot = (
        result[4] if len(result) == 5 else (jnp.zeros_like(result[1]), jnp.zeros_like(result[2]))
    )
    return *result[:4], first_dot, second_dot


def _lowdot_two_pullback_native_multi_rhs_adjoint_fields(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    direct_coefficient_pullback: bool = False,
):
    """Build native matrix-RHS fields for the exact low-dot support pullback.

    This private building block deliberately stops before constructing any
    prepared-system cotangent pytree.  It evaluates the primal and the two
    physical forward directions once, then forms all base and directional
    implicit adjoints with trailing objective-RHS columns.  The caller can
    subsequently apply the prepared-gradient contractions to those fields
    without entering the scalar low-dot implicit helper once per objective.

    Returned field layouts are ``(mode, unknown, rhs)`` for base fields and
    ``(mode, unknown, rhs, direction)`` for directional fields.  The two
    physical direction fields themselves have no RHS axis.  This distinction
    is essential: physical forward directions are objective-independent, while
    cotangents are objective-dependent.
    """

    primal_outputs, primal_residuals = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        None,
        include_geometry=False,
        return_coefficient_aux=True,
        return_primal_residuals=True,
    )
    (
        transport_scale,
        resolved_epsi_hat,
        first_nu_dot,
        first_epsi_dot,
        second_nu_dot,
        second_epsi_dot,
        _coefficients,
        f1_full,
        f3_full,
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        packed_f_dot_low_matrix,
        _first_coefficient_dot,
        _second_coefficient_dot,
    ) = primal_residuals
    del transport_scale

    base, first, second, auxiliary, first_dot, second_dot = _unpack_coefficient_bars(
        coefficient_bars_and_aux_fn(*primal_outputs)
    )
    (
        base_coefficient_bars,
        first_coefficient_bars,
        second_coefficient_bars,
        first_coefficient_bars_dot,
        second_coefficient_bars_dot,
    ) = map(jnp.asarray, (base, first, second, first_dot, second_dot))
    if (
        base_coefficient_bars.ndim != 2
        or first_coefficient_bars.shape != base_coefficient_bars.shape
        or second_coefficient_bars.shape != base_coefficient_bars.shape
        or first_coefficient_bars_dot.shape != base_coefficient_bars.shape
        or second_coefficient_bars_dot.shape != base_coefficient_bars.shape
    ):
        raise ValueError(
            "All low-dot coefficient cotangent batches must have shape (n_rhs, n_coefficients)."
        )

    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        resolved_epsi_hat,
    )
    rhs_count = base_coefficient_bars.shape[0]
    direction_count = 2
    mode_count, unknown_count = f1_full.shape
    mode_indices = jnp.arange(mode_count, dtype=jnp.int32)

    def _zero_first_row_if_needed(block, k):
        zeroed = block.at[0, :].set(jnp.zeros((block.shape[1],), dtype=block.dtype))
        return jnp.where(jnp.asarray(k) == 0, zeroed, block)

    def _coefficient_pullback(one_coefficient_bar):
        return _coefficient_mode_pullback(
            prepared.geometry,
            f1_full[:3],
            f3_full[:3],
            ctx.nu_hat,
            one_coefficient_bar,
        )

    def _field_rhs_from_low(low_fields):
        # ``low_fields`` is (rhs, retained-mode, unknown).  The factorized
        # solver uses mode/unknown/RHS ordering.
        return (
            jnp.zeros((mode_count, unknown_count, rhs_count), dtype=f1_full.dtype)
            .at[:3]
            .set(jnp.moveaxis(low_fields, 0, -1))
        )

    if direct_coefficient_pullback:
        base_f1_low, base_f3_low, base_nu_direct = _coefficient_mode_pullback_multi_rhs_direct(
            prepared.geometry, ctx.nu_hat, base_coefficient_bars
        )
    else:
        base_f1_low, base_f3_low, base_nu_direct = jax.vmap(_coefficient_pullback)(
            base_coefficient_bars
        )
    base_lambda1, base_lambda3 = _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        _field_rhs_from_low(base_f1_low),
        _field_rhs_from_low(base_f3_low),
    )

    nu_dots = jnp.stack([first_nu_dot, second_nu_dot])
    epsi_dots = jnp.stack([first_epsi_dot, second_epsi_dot])
    f1_dot_low = jnp.stack(
        [packed_f_dot_low_matrix[..., 0], packed_f_dot_low_matrix[..., 2]], axis=-1
    )
    f3_dot_low = jnp.stack(
        [packed_f_dot_low_matrix[..., 1], packed_f_dot_low_matrix[..., 3]], axis=-1
    )

    def _directional_coefficient_pullback(
        coefficient_bars,
        coefficient_bars_dot,
        f1_dot_low_value,
        f3_dot_low_value,
        nu_dot_value,
    ):
        if direct_coefficient_pullback:
            # ``_coefficient_mode_pullback`` is independent of the retained
            # primal modes.  The direct helper therefore needs only the two
            # coefficient-bar batches and the explicit Spitzer ``nu``
            # direction; ``f*_dot_low_value`` are intentionally absent.
            return _directional_coefficient_mode_pullback_multi_rhs_direct(
                prepared.geometry,
                ctx.nu_hat,
                nu_dot_value,
                coefficient_bars,
                coefficient_bars_dot,
            )

        def _one_rhs(one_coefficient_bar, one_coefficient_bar_dot):
            def _pullback_from_primal(modes1, modes3, nu_value, coefficient_bar_value):
                return _coefficient_mode_pullback(
                    prepared.geometry,
                    modes1,
                    modes3,
                    nu_value,
                    coefficient_bar_value,
                )

            return jax.jvp(
                _pullback_from_primal,
                (f1_full[:3], f3_full[:3], ctx.nu_hat, one_coefficient_bar),
                (
                    f1_dot_low_value,
                    f3_dot_low_value,
                    nu_dot_value,
                    one_coefficient_bar_dot,
                ),
            )

        return jax.vmap(_one_rhs)(coefficient_bars, coefficient_bars_dot)

    first_directional = _directional_coefficient_pullback(
        first_coefficient_bars,
        first_coefficient_bars_dot,
        f1_dot_low[..., 0],
        f3_dot_low[..., 0],
        first_nu_dot,
    )
    second_directional = _directional_coefficient_pullback(
        second_coefficient_bars,
        second_coefficient_bars_dot,
        f1_dot_low[..., 1],
        f3_dot_low[..., 1],
        second_nu_dot,
    )
    directional_primal_low = tuple(
        jnp.stack([first_value, second_value], axis=-1)
        for first_value, second_value in zip(
            first_directional[0], second_directional[0], strict=True
        )
    )
    directional_dot_low = tuple(
        jnp.stack([first_value, second_value], axis=-1)
        for first_value, second_value in zip(
            first_directional[1], second_directional[1], strict=True
        )
    )
    # (rhs, retained-mode, unknown, direction) -> (mode, unknown, rhs, direction)
    directional_f1_low, directional_f3_low, directional_nu_direct = directional_primal_low
    directional_f1_low_dot, directional_f3_low_dot, directional_nu_direct_dot = directional_dot_low
    directional_f1_low = jnp.transpose(directional_f1_low, (1, 2, 0, 3))
    directional_f3_low = jnp.transpose(directional_f3_low, (1, 2, 0, 3))
    directional_f1_low_dot = jnp.transpose(directional_f1_low_dot, (1, 2, 0, 3))
    directional_f3_low_dot = jnp.transpose(directional_f3_low_dot, (1, 2, 0, 3))
    directional_shape = (mode_count, unknown_count, rhs_count, direction_count)
    directional_g1 = (
        jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(directional_f1_low)
    )
    directional_g3 = (
        jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(directional_f3_low)
    )
    directional_g1_dot = (
        jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(directional_f1_low_dot)
    )
    directional_g3_dot = (
        jnp.zeros(directional_shape, dtype=f1_full.dtype).at[:3].set(directional_f3_low_dot)
    )
    directional_lambda1, directional_lambda3 = _solve_factorized_adjoint_field_pair(
        saved_lu,
        saved_piv,
        saved_lower,
        saved_upper,
        jnp.reshape(directional_g1, (mode_count, unknown_count, rhs_count * direction_count)),
        jnp.reshape(directional_g3, (mode_count, unknown_count, rhs_count * direction_count)),
    )
    directional_lambda1 = jnp.reshape(directional_lambda1, directional_shape)
    directional_lambda3 = jnp.reshape(directional_lambda3, directional_shape)

    def _directional_sources_for_mode(k):
        diagonal_nu, diagonal_epsi = parameter_derivative_blocks(
            ctx, k, prepared.d_theta, prepared.d_zeta
        )
        diagonal_nu = _zero_first_row_if_needed(diagonal_nu, k)
        diagonal_epsi = _zero_first_row_if_needed(diagonal_epsi, k)
        diagonal_dot = (
            nu_dots[:, None, None] * diagonal_nu[None, :, :]
            + epsi_dots[:, None, None] * diagonal_epsi[None, :, :]
        )
        f1_value = jax.lax.dynamic_index_in_dim(f1_full, k, axis=0, keepdims=False)
        f3_value = jax.lax.dynamic_index_in_dim(f3_full, k, axis=0, keepdims=False)
        return (
            -jnp.einsum("dij,j->id", diagonal_dot, f1_value),
            -jnp.einsum("dij,j->id", diagonal_dot, f3_value),
            diagonal_dot,
        )

    source1_dot, source3_dot, diagonal_dot = jax.vmap(_directional_sources_for_mode)(mode_indices)
    f1_dot_full = _solve_factorized_modes(
        saved_lu, saved_piv, saved_lower, saved_upper, source1_dot
    )
    f3_dot_full = _solve_factorized_modes(
        saved_lu, saved_piv, saved_lower, saved_upper, source3_dot
    )
    directional_lambda1_dot, directional_lambda3_dot = (
        _solve_factorized_multi_rhs_directional_adjoint_field_pair(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            directional_lambda1,
            directional_lambda3,
            directional_g1_dot,
            directional_g3_dot,
            diagonal_dot,
        )
    )
    return (
        primal_outputs,
        base_coefficient_bars,
        first_coefficient_bars,
        second_coefficient_bars,
        first_coefficient_bars_dot,
        second_coefficient_bars_dot,
        auxiliary,
        f1_full,
        f3_full,
        f1_dot_full,
        f3_dot_full,
        base_lambda1,
        base_lambda3,
        directional_lambda1,
        directional_lambda3,
        directional_lambda1_dot,
        directional_lambda3_dot,
        nu_dots,
        epsi_dots,
        base_nu_direct,
        directional_nu_direct,
        directional_nu_direct_dot,
    )


def solve_prepared_coefficient_vector_two_directional_factorized(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Return a prepared solve and two exact case directions from one factorization.

    This is an isolated forward helper.  It evaluates the base coefficient
    vector and the exact linear responses to ``first_case_dot`` and
    ``second_case_dot`` by reusing the one prepared-system factorization.
    It does not install a custom derivative rule and does not alter the
    existing solve or pullback entry points.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        None,
        include_geometry=False,
        return_coefficient_aux=False,
    )


def solve_prepared_coefficient_vector_hessian_factorized(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return coefficient value, gradient, and Hessian from one factorization.

    The prepared NTX operator is affine in the resolved local coordinates
    ``(nu_hat, epsi_hat)`` and its source is parameter independent.  For
    ``A f = b`` this gives the exact directional systems

    ``A f_p = -A_p f`` and
    ``A f_pq = -(A_p f_q + A_q f_p)``.

    Consequently one base factorization can solve all base, first-, and
    second-direction fields.  This is intentionally an explicit implicit
    derivative primitive: it does not differentiate through the existing
    custom-VJP solver entry point.
    """

    _, ctx, primal = _prepared_primal_context(prepared, case)
    (coefficients, f1_full, f3_full, saved_lu, saved_piv, saved_lower, saved_upper) = primal
    mode_indices = jnp.arange(prepared.grid.n_xi + 1, dtype=jnp.int32)

    def _parameter_blocks(mode_index):
        dnu, depsi = parameter_derivative_blocks(
            ctx,
            mode_index,
            prepared.d_theta,
            prepared.d_zeta,
        )
        dnu = jax.lax.cond(
            mode_index == 0,
            lambda value: value.at[0, :].set(jnp.zeros_like(value[0, :])),
            lambda value: value,
            dnu,
        )
        depsi = jax.lax.cond(
            mode_index == 0,
            lambda value: value.at[0, :].set(jnp.zeros_like(value[0, :])),
            lambda value: value,
            depsi,
        )
        return dnu, depsi

    dnu_blocks, depsi_blocks = jax.vmap(_parameter_blocks)(mode_indices)

    def _solve_first_fields(field):
        rhs = jnp.stack(
            (
                -jnp.einsum("mij,mj->mi", dnu_blocks, field),
                -jnp.einsum("mij,mj->mi", depsi_blocks, field),
            ),
            axis=-1,
        )
        return _solve_factorized_modes(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            rhs,
        )

    f1_first = _solve_first_fields(f1_full)
    f3_first = _solve_first_fields(f3_full)

    def _solve_second_fields(first_fields):
        f_nu = first_fields[..., 0]
        f_epsi = first_fields[..., 1]
        rhs = jnp.stack(
            (
                -2.0 * jnp.einsum("mij,mj->mi", dnu_blocks, f_nu),
                -(
                    jnp.einsum("mij,mj->mi", dnu_blocks, f_epsi)
                    + jnp.einsum("mij,mj->mi", depsi_blocks, f_nu)
                ),
                -2.0 * jnp.einsum("mij,mj->mi", depsi_blocks, f_epsi),
            ),
            axis=-1,
        )
        return _solve_factorized_modes(
            saved_lu,
            saved_piv,
            saved_lower,
            saved_upper,
            rhs,
        )

    f1_second = _solve_second_fields(f1_first)
    f3_second = _solve_second_fields(f3_first)

    def _coefficient_first(f1_dot, f3_dot, nu_dot):
        output = jnp.stack(
            coefficients_from_modes(
                prepared.geometry,
                f1_dot[:3],
                f3_dot[:3],
                ctx.nu_hat,
            )
        )
        return output.at[4].set(-coefficients[4] * nu_dot / ctx.nu_hat)

    def _coefficient_second(f1_dotdot, f3_dotdot, nu_dot_a, nu_dot_b):
        output = jnp.stack(
            coefficients_from_modes(
                prepared.geometry,
                f1_dotdot[:3],
                f3_dotdot[:3],
                ctx.nu_hat,
            )
        )
        return output.at[4].set(2.0 * coefficients[4] * nu_dot_a * nu_dot_b / ctx.nu_hat**2)

    nu_one = jnp.asarray(1.0, dtype=ctx.nu_hat.dtype)
    zero = jnp.asarray(0.0, dtype=ctx.nu_hat.dtype)
    coefficient_nu = _coefficient_first(f1_first[..., 0], f3_first[..., 0], nu_one)
    coefficient_epsi = _coefficient_first(f1_first[..., 1], f3_first[..., 1], zero)
    coefficient_nunu = _coefficient_second(f1_second[..., 0], f3_second[..., 0], nu_one, nu_one)
    coefficient_nuepsi = _coefficient_second(f1_second[..., 1], f3_second[..., 1], nu_one, zero)
    coefficient_epsiepsi = _coefficient_second(f1_second[..., 2], f3_second[..., 2], zero, zero)
    return (
        coefficients,
        coefficient_nu,
        coefficient_epsi,
        coefficient_nunu,
        coefficient_nuepsi,
        coefficient_epsiepsi,
    )


def _case_cotangent_from_resolved_epsi_bar(
    case: MonoenergeticCase,
    *,
    nu_hat_bar: Array,
    epsi_hat_bar: Array,
    transport_scale: float | Array | None,
) -> MonoenergeticCase:
    """Rebuild a case cotangent in the input's public coordinate system."""

    if case.epsi_hat is not None:
        return MonoenergeticCase(nu_hat=nu_hat_bar, epsi_hat=epsi_hat_bar, er_hat=None)
    if case.er_hat is not None:
        assert transport_scale is not None
        return MonoenergeticCase(
            nu_hat=nu_hat_bar,
            epsi_hat=None,
            er_hat=epsi_hat_bar / transport_scale,
        )
    return MonoenergeticCase(nu_hat=nu_hat_bar, epsi_hat=None, er_hat=None)


def _add_er_coordinate_scale_bar(
    prepared_bar: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    epsi_hat_bar: Array,
    transport_scale: float | Array | None,
) -> PreparedMonoenergeticSystem:
    """Apply ``epsi_hat = er_hat / transport_scale`` to a prepared bar."""

    if case.er_hat is None:
        return prepared_bar
    assert transport_scale is not None
    return dataclasses.replace(
        prepared_bar,
        geometry=dataclasses.replace(
            prepared_bar.geometry,
            transport_psi_scale=(
                prepared_bar.geometry.transport_psi_scale
                - epsi_hat_bar * jnp.asarray(case.er_hat) / jnp.asarray(transport_scale) ** 2
            ),
        ),
    )


def _sum_prepared_cotangent_trees(*trees: PreparedMonoenergeticSystem):
    """Sum prepared cotangents while preserving JAX's static ``float0`` leaves."""

    def _sum_leaves(*values):
        if jnp.asarray(values[0]).dtype == jax.dtypes.float0:
            return values[0]
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result

    return jax.tree_util.tree_map(_sum_leaves, *trees)


def solve_prepared_coefficient_vector_two_directional_prepared_vjp(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    """Prepared-pytree wrapper around a local exact prepared-support custom VJP.

    This entry point is intentionally separate from the existing raw solve and
    grouped pullback APIs.  Its forward rule saves one local prepared-system
    factorization and the two directional low modes; its backward rule reuses
    those residuals instead of constructing a second primal NTX solve.
    """

    prepared_tree = jax.tree_util.tree_structure(prepared)
    prepared_leaves = tuple(jax.tree_util.tree_leaves(prepared))
    return _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp(
        prepared_tree,
        prepared_leaves,
        case,
        first_case_dot,
        second_case_dot,
    )


def _solve_prepared_coefficient_vector_two_directional_prepared_vjp_fwd(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
):
    outputs, primal_residuals = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        None,
        include_geometry=False,
        return_coefficient_aux=False,
        return_primal_residuals=True,
    )
    return outputs, (prepared, case, first_case_dot, second_case_dot, primal_residuals)


def _solve_prepared_coefficient_vector_two_directional_prepared_vjp_bwd(
    residuals,
    output_bars: tuple[Array, Array, Array],
):
    prepared, case, first_case_dot, second_case_dot, primal_residuals = residuals
    base_coefficient_bar, first_coefficient_bar, second_coefficient_bar = output_bars

    def _coefficient_bars_and_aux(_coefficients, _first_coefficients, _second_coefficients):
        return (
            base_coefficient_bar,
            first_coefficient_bar,
            second_coefficient_bar,
            jnp.asarray(0.0, dtype=base_coefficient_bar.dtype),
        )

    result = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        _coefficient_bars_and_aux,
        include_geometry=False,
        include_prepared=True,
        return_coefficient_aux=True,
        primal_residuals=primal_residuals,
    )
    (
        base_nu_hat_bar,
        base_epsi_hat_bar,
        base_prepared_bar,
        first_base_nu_hat_bar,
        first_base_epsi_hat_bar,
        first_nu_hat_bar,
        first_epsi_hat_bar,
        _first_base_prepared_bar,
        first_prepared_bar,
        second_base_nu_hat_bar,
        second_base_epsi_hat_bar,
        second_nu_hat_bar,
        second_epsi_hat_bar,
        _second_base_prepared_bar,
        second_prepared_bar,
        _auxiliary_bar,
    ) = result
    transport_scale = primal_residuals[0]
    prepared_bar = _sum_prepared_cotangent_trees(
        base_prepared_bar,
        first_prepared_bar,
        second_prepared_bar,
    )
    # The grouped core has already applied this chain rule to the base output
    # bar.  Apply only the two directional-output contributions here.
    prepared_bar = _add_er_coordinate_scale_bar(
        prepared_bar,
        case,
        first_epsi_hat_bar + second_epsi_hat_bar,
        transport_scale,
    )
    prepared_bar = _add_er_coordinate_scale_bar(
        prepared_bar,
        first_case_dot,
        first_base_epsi_hat_bar,
        transport_scale,
    )
    prepared_bar = _add_er_coordinate_scale_bar(
        prepared_bar,
        second_case_dot,
        second_base_epsi_hat_bar,
        transport_scale,
    )
    case_bar = _case_cotangent_from_resolved_epsi_bar(
        case,
        nu_hat_bar=(base_nu_hat_bar + first_nu_hat_bar + second_nu_hat_bar),
        epsi_hat_bar=(base_epsi_hat_bar + first_epsi_hat_bar + second_epsi_hat_bar),
        transport_scale=transport_scale,
    )
    first_case_dot_bar = _case_cotangent_from_resolved_epsi_bar(
        first_case_dot,
        nu_hat_bar=first_base_nu_hat_bar,
        epsi_hat_bar=first_base_epsi_hat_bar,
        transport_scale=transport_scale,
    )
    second_case_dot_bar = _case_cotangent_from_resolved_epsi_bar(
        second_case_dot,
        nu_hat_bar=second_base_nu_hat_bar,
        epsi_hat_bar=second_base_epsi_hat_bar,
        transport_scale=transport_scale,
    )
    return prepared_bar, case_bar, first_case_dot_bar, second_case_dot_bar


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp(
    prepared_tree,
    prepared_leaves,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
) -> tuple[Array, Array, Array]:
    prepared = jax.tree_util.tree_unflatten(prepared_tree, prepared_leaves)
    return solve_prepared_coefficient_vector_two_directional_factorized(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
    )


def _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp_fwd(
    prepared_tree,
    prepared_leaves,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
):
    prepared = jax.tree_util.tree_unflatten(prepared_tree, prepared_leaves)
    return _solve_prepared_coefficient_vector_two_directional_prepared_vjp_fwd(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
    )


def _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp_bwd(
    prepared_tree,
    residuals,
    output_bars: tuple[Array, Array, Array],
):
    prepared_bar, case_bar, first_case_dot_bar, second_case_dot_bar = (
        _solve_prepared_coefficient_vector_two_directional_prepared_vjp_bwd(
            residuals,
            output_bars,
        )
    )
    return (
        tuple(jax.tree_util.tree_leaves(prepared_bar)),
        case_bar,
        first_case_dot_bar,
        second_case_dot_bar,
    )


_solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp.defvjp(
    _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp_fwd,
    _solve_prepared_coefficient_vector_two_directional_prepared_leaves_vjp_bwd,
)


def solve_prepared_coefficient_vector_lowdot_two_pullbacks(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_fn,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Case-only public view of the fused low-order pullback kernel."""

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_fn,
        include_geometry=False,
        return_coefficient_aux=False,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_fn,
):
    """Fused exact pullback returning case and prepared-geometry cotangents.

    This opt-in API preserves the original 10 case cotangents and additionally
    returns one geometry cotangent for the base response plus base/directional
    geometry cotangents for each supplied case direction.  The primal
    factorization is shared across all three contractions.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_fn,
        include_geometry=True,
        return_coefficient_aux=False,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Geometry-returning fused pullback with caller-defined direct auxiliaries.

    ``coefficient_bar_and_aux_fn(coefficients, first_coeff_dot,
    second_coeff_dot)`` returns ``(base_bar, first_bar, second_bar,
    auxiliary)``. ``auxiliary`` is not interpreted by NTX; it is evaluated
    from the already available coefficient vector and its two exact tangent
    vectors, then returned with the implicit bars.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=True,
        return_coefficient_aux=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Return only geometry support bars of the fused low-dot pullback.

    This is the geometry-only counterpart of the prepared-support-only helper.
    It retains the exact base and two directional implicit adjoints, but omits
    unused case/profile contractions and does not construct complete prepared
    cotangent trees.  The result is ``(base_geometry,
    first_base_geometry, first_directional_geometry,
    second_base_geometry, second_directional_geometry, auxiliary)``.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=True,
        return_coefficient_aux=True,
        support_only=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Fused exact pullback returning a complete prepared-system cotangent.

    This is the support-payload-safe variant of the geometry-only helper. Its
    prepared bars include ``surface``, ``geometry``, and derivative-operator
    leaves while retaining the same one-factorization grouped adjoint.
    """
    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=False,
        include_prepared=True,
        return_coefficient_aux=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Return only the prepared-support bars of the fused low-dot pullback.

    This is an isolated API for callers differentiating a response with
    respect to a prepared NTX payload and an external direct auxiliary (for
    example ``drds``).  It intentionally discards the case/profile bars:
    callers in that lane hold the local case fixed.  The underlying grouped
    implicit algebra and primal factorization are exactly the established
    prepared-support implementation above, while the directional case/profile
    contractions are omitted rather than merely discarded.

    The result is ``(base_prepared, first_base_prepared,
    first_directional_prepared, second_base_prepared,
    second_directional_prepared, auxiliary)``.  This keeps the smaller
    support-only algebra isolated from every existing public helper.
    """

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=False,
        include_prepared=True,
        return_coefficient_aux=True,
        support_only=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux(
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
):
    """Return RHS-batched prepared-support bars with one local primal.

    The callback takes ``(coefficients, first_dot, second_dot)`` and returns
    ``(base_bars, first_bars, second_bars, auxiliary)`` with a shared RHS axis.
    An optional fifth item supplies ``(first_bars_dot, second_bars_dot)``.
    Return the unbatched primal outputs too when ``return_primal_outputs`` is
    true. Factors remain local to this call; they are not part of the result.
    """

    primal_outputs, primal_residuals = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        None,
        include_geometry=False,
        include_prepared=True,
        return_coefficient_aux=True,
        support_only=True,
        return_primal_residuals=True,
    )
    base_bars, first_bars, second_bars, auxiliary, first_bars_dot, second_bars_dot = (
        _unpack_coefficient_bars(coefficient_bars_and_aux_fn(*primal_outputs))
    )

    prepared_bar_tree = jax.tree_util.tree_structure(prepared)

    def _one_rhs(
        base_bar,
        first_bar,
        second_bar,
        auxiliary,
        first_bar_dot,
        second_bar_dot,
    ):
        result = _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
            prepared,
            case,
            first_case_dot,
            second_case_dot,
            lambda _base, _first, _second: (
                base_bar,
                first_bar,
                second_bar,
                auxiliary,
                (first_bar_dot, second_bar_dot),
            ),
            include_geometry=False,
            include_prepared=True,
            return_coefficient_aux=True,
            support_only=True,
            primal_residuals=primal_residuals,
        )
        # Keep static prepared-system metadata outside ``vmap``.  Returning a
        # PreparedMonoenergeticSystem directly would ask JAX to batch its
        # validated/static fields; batch only the dynamic cotangent leaves and
        # reconstruct the five original pytrees afterwards.
        return (
            tuple(tuple(jax.tree_util.tree_leaves(value)) for value in result[:5]),
            result[5],
        )

    prepared_bar_leaf_groups, auxiliary = jax.vmap(_one_rhs)(
        base_bars,
        first_bars,
        second_bars,
        auxiliary,
        first_bars_dot,
        second_bars_dot,
    )
    result = (
        *(
            jax.tree_util.tree_unflatten(prepared_bar_tree, leaves)
            for leaves in prepared_bar_leaf_groups
        ),
        auxiliary,
    )
    return (primal_outputs, result) if return_primal_outputs else result


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
    return_case_bars: bool = False,
    _compact_result: bool = False,
    _compact_residual_transpose: bool = False,
    return_vmec_coefficient_bars: bool = False,
    native_vmec_coefficient_bars_only: bool = False,
    native_vmec_direct_directional_product_rule: bool = False,
    native_direct_coefficient_pullback: bool = False,
):
    """Exact support-only low-dot pullback with native objective RHS solves.

    Unlike the older ``*_multi_rhs_and_aux`` experiment, this function does
    not ``vmap`` the scalar implicit pullback.  It builds all of the base and
    two directional adjoint fields with matrix-RHS factorized solves, then
    batches only the final prepared-gradient contractions.  Consequently the
    primal modes, physical forward directions, and block-tridiagonal adjoint
    factorization are each local to one call and are never recreated per
    objective RHS.

    It is intentionally opt-in and currently supports the explicit
    ``epsi_hat`` coordinate used by NEOPAX's local interpolated response. If
    ``return_case_bars`` is selected, it also returns the native matrix-RHS
    case contractions needed by NEOPAX's local support-coordinate chain.
    Those contractions reuse the adjoint fields formed here and do not create
    another NTX solve. The existing scalar APIs retain their broader public
    coordinate contracts.
    """

    if case.epsi_hat is None or first_case_dot.epsi_hat is None or second_case_dot.epsi_hat is None:
        raise ValueError(
            "The native multi-RHS prepared-support helper requires explicit epsi_hat cases."
        )
    (
        primal_outputs,
        base_coefficient_bars,
        first_coefficient_bars,
        second_coefficient_bars,
        first_coefficient_bars_dot,
        second_coefficient_bars_dot,
        auxiliary,
        f1_full,
        f3_full,
        f1_dot_full,
        f3_dot_full,
        base_lambda1,
        base_lambda3,
        directional_lambda1,
        directional_lambda3,
        directional_lambda1_dot,
        directional_lambda3_dot,
        nu_dots,
        epsi_dots,
        base_nu_direct,
        directional_nu_direct,
        directional_nu_direct_dot,
    ) = _lowdot_two_pullback_native_multi_rhs_adjoint_fields(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bars_and_aux_fn,
        direct_coefficient_pullback=native_direct_coefficient_pullback,
    )
    ctx = _operator_context(
        prepared.surface,
        prepared.geometry,
        prepared.grid,
        case.nu_hat,
        case.epsi_hat,
    )
    prepared_tree = jax.tree_util.tree_structure(prepared)

    if _compact_residual_transpose and not _compact_result:
        raise ValueError(
            "The compact residual transpose has only the compact prepared-bar contract."
        )

    if return_vmec_coefficient_bars and not _compact_result:
        raise ValueError(
            "Native VMEC coefficient bars require the combined compact prepared contract."
        )
    if native_vmec_coefficient_bars_only and not return_vmec_coefficient_bars:
        raise ValueError(
            "native_vmec_coefficient_bars_only requires return_vmec_coefficient_bars=True."
        )
    if native_vmec_coefficient_bars_only and _compact_residual_transpose:
        raise ValueError(
            "native_vmec_coefficient_bars_only does not support the compact residual contract."
        )
    if native_vmec_direct_directional_product_rule and not return_vmec_coefficient_bars:
        raise ValueError(
            "native_vmec_direct_directional_product_rule requires VMEC coefficient bars."
        )

    if _compact_residual_transpose:
        compact_tree = jax.tree_util.tree_structure(
            _compact_prepared_residual_inputs(
                prepared,
                ctx.nu_hat,
                ctx.epsi_hat,
            )
        )

        def _combine_compact_leaf(*leaves):
            if any(jnp.asarray(leaf).dtype == jax.dtypes.float0 for leaf in leaves):
                return leaves[0]
            return sum(leaves)

        def _one_rhs_compact_residual(
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
            base_compact = _compact_prepared_gradient_from_adjoint(
                prepared,
                ctx,
                f1_full,
                f3_full,
                base_lambda1_value,
                base_lambda3_value,
                base_coefficient_bar,
            )
            _, first_directional_compact = _directional_compact_prepared_gradient_from_adjoint(
                prepared,
                nu_hat=ctx.nu_hat,
                epsi_hat=ctx.epsi_hat,
                nu_hat_dot=nu_dots[0],
                epsi_hat_dot=epsi_dots[0],
                f1_full=f1_full,
                f3_full=f3_full,
                f1_dot=f1_dot_full[..., 0],
                f3_dot=f3_dot_full[..., 0],
                lambda1=directional_lambda1_value[..., 0],
                lambda3=directional_lambda3_value[..., 0],
                lambda1_dot=directional_lambda1_dot_value[..., 0],
                lambda3_dot=directional_lambda3_dot_value[..., 0],
                coefficient_bar=first_coefficient_bar,
                coefficient_bar_dot=first_coefficient_bar_dot,
            )
            _, second_directional_compact = _directional_compact_prepared_gradient_from_adjoint(
                prepared,
                nu_hat=ctx.nu_hat,
                epsi_hat=ctx.epsi_hat,
                nu_hat_dot=nu_dots[1],
                epsi_hat_dot=epsi_dots[1],
                f1_full=f1_full,
                f3_full=f3_full,
                f1_dot=f1_dot_full[..., 1],
                f3_dot=f3_dot_full[..., 1],
                lambda1=directional_lambda1_value[..., 1],
                lambda3=directional_lambda3_value[..., 1],
                lambda1_dot=directional_lambda1_dot_value[..., 1],
                lambda3_dot=directional_lambda3_dot_value[..., 1],
                coefficient_bar=second_coefficient_bar,
                coefficient_bar_dot=second_coefficient_bar_dot,
            )
            combined_compact = jax.tree_util.tree_map(
                _combine_compact_leaf,
                base_compact,
                first_directional_compact,
                second_directional_compact,
            )
            return tuple(jax.tree_util.tree_leaves(combined_compact))

        compact_bar_leaf_groups = jax.vmap(_one_rhs_compact_residual)(
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

        def _one_compact_bar_to_prepared(*compact_leaves):
            compact_bar = jax.tree_util.tree_unflatten(compact_tree, compact_leaves)
            prepared_bar = _compact_prepared_bar_to_prepared(
                prepared,
                nu_hat=ctx.nu_hat,
                epsi_hat=ctx.epsi_hat,
                compact_bar=compact_bar,
            )
            return tuple(jax.tree_util.tree_leaves(prepared_bar))

        prepared_bar_leaf_groups = jax.vmap(_one_compact_bar_to_prepared)(*compact_bar_leaf_groups)
        result = (jax.tree_util.tree_unflatten(prepared_tree, prepared_bar_leaf_groups), auxiliary)

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
        first_base_prepared, first_directional_prepared = (
            _directional_prepared_gradient_from_adjoint(
                prepared,
                nu_hat=ctx.nu_hat,
                epsi_hat=ctx.epsi_hat,
                nu_hat_dot=nu_dots[0],
                epsi_hat_dot=epsi_dots[0],
                f1_full=f1_full,
                f3_full=f3_full,
                f1_dot=f1_dot_full[..., 0],
                f3_dot=f3_dot_full[..., 0],
                lambda1=directional_lambda1_value[..., 0],
                lambda3=directional_lambda3_value[..., 0],
                lambda1_dot=directional_lambda1_dot_value[..., 0],
                lambda3_dot=directional_lambda3_dot_value[..., 0],
                coefficient_bar=first_coefficient_bar,
                coefficient_bar_dot=first_coefficient_bar_dot,
            )
        )
        second_base_prepared, second_directional_prepared = (
            _directional_prepared_gradient_from_adjoint(
                prepared,
                nu_hat=ctx.nu_hat,
                epsi_hat=ctx.epsi_hat,
                nu_hat_dot=nu_dots[1],
                epsi_hat_dot=epsi_dots[1],
                f1_full=f1_full,
                f3_full=f3_full,
                f1_dot=f1_dot_full[..., 1],
                f3_dot=f3_dot_full[..., 1],
                lambda1=directional_lambda1_value[..., 1],
                lambda3=directional_lambda3_value[..., 1],
                lambda1_dot=directional_lambda1_dot_value[..., 1],
                lambda3_dot=directional_lambda3_dot_value[..., 1],
                coefficient_bar=second_coefficient_bar,
                coefficient_bar_dot=second_coefficient_bar_dot,
            )
        )

        if _compact_result:
            # Only these three terms are cotangents of the primal prepared
            # system.  The two ``*_base`` trees belong to the physical case
            # directions and are consumed below for the compact case chain;
            # returning them as full prepared pytrees needlessly multiplies
            # the objective-batched temporary payload.
            def _combine_primal_prepared_leaf(primal_leaf, base_leaf, first_leaf, second_leaf):
                primal_arr = jnp.asarray(primal_leaf)
                if not jnp.issubdtype(primal_arr.dtype, jnp.inexact):
                    return jnp.zeros(primal_arr.shape, dtype=jnp.float64)
                return base_leaf + first_leaf + second_leaf

            combined_prepared = jax.tree_util.tree_map(
                _combine_primal_prepared_leaf,
                prepared,
                base_prepared,
                first_directional_prepared,
                second_directional_prepared,
            )
            return tuple(jax.tree_util.tree_leaves(combined_prepared))
        return tuple(
            tuple(jax.tree_util.tree_leaves(value))
            for value in (
                base_prepared,
                first_base_prepared,
                first_directional_prepared,
                second_base_prepared,
                second_directional_prepared,
            )
        )

    if native_vmec_coefficient_bars_only:
        # The caller routes the complete prepared-system contribution through
        # ``native_vmec_coefficient_bars``.  Do not stage the generic
        # prepared-gradient contraction merely to return a support payload
        # which NEOPAX will discard.  Keep a shape-compatible zero tree so
        # the established outer scan/carry ABI remains unchanged for now.
        rhs_count = int(jnp.asarray(base_coefficient_bars).shape[0])

        def _zero_prepared_leaf(primal_leaf):
            primal_arr = jnp.asarray(primal_leaf)
            return jnp.zeros((rhs_count,) + primal_arr.shape, dtype=jnp.float64)

        result = (
            jax.tree_util.tree_map(_zero_prepared_leaf, prepared),
            auxiliary,
        )
    elif not _compact_residual_transpose:
        prepared_bar_leaf_groups = jax.vmap(_one_rhs)(
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
        result = (
            (jax.tree_util.tree_unflatten(prepared_tree, prepared_bar_leaf_groups), auxiliary)
            if _compact_result
            else (
                *(
                    jax.tree_util.tree_unflatten(prepared_tree, leaves)
                    for leaves in prepared_bar_leaf_groups
                ),
                auxiliary,
            )
        )
    native_vmec_coefficient_bars = None
    if return_vmec_coefficient_bars:
        # The local support cotangent contains only the primal prepared-system
        # terms: base plus the two directional low-dot contributions.  The
        # two physical-direction *base* cotangents belong exclusively to the
        # case chain below and must not be added to this geometry result.
        #
        # Keep those three terms as primitive geometry bars until after their
        # sum.  Mapping each one separately through the sampled-field/Fourier
        # transpose changes floating-point reduction order relative to the
        # combined prepared VJP and needlessly stages that large transpose
        # three times inside the segment graph.
        native_vmec_base = _native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs(
            prepared,
            ctx,
            f1_full,
            f3_full,
            base_lambda1,
            base_lambda3,
            base_coefficient_bars,
        )

        def _directional_native_primitive_bars(
            direction_index,
            coefficient_bars,
            coefficient_bars_dot,
        ):
            def _primitive_from_dynamic_terms(
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
                return _native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs(
                    prepared,
                    local_ctx,
                    f1_value,
                    f3_value,
                    lambda1_value,
                    lambda3_value,
                    coefficient_bar_value,
                )

            _base, directional = jax.jvp(
                _primitive_from_dynamic_terms,
                (
                    ctx.nu_hat,
                    ctx.epsi_hat,
                    f1_full,
                    f3_full,
                    directional_lambda1[..., direction_index],
                    directional_lambda3[..., direction_index],
                    coefficient_bars,
                ),
                (
                    nu_dots[direction_index],
                    epsi_dots[direction_index],
                    f1_dot_full[..., direction_index],
                    f3_dot_full[..., direction_index],
                    directional_lambda1_dot[..., direction_index],
                    directional_lambda3_dot[..., direction_index],
                    coefficient_bars_dot,
                ),
            )
            return directional

        if native_vmec_direct_directional_product_rule:

            def _direct_directional_primitive_bars(
                direction_index,
                coefficient_bars,
                coefficient_bars_dot,
            ):
                return _directional_native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs_direct(
                    prepared,
                    nu_hat=ctx.nu_hat,
                    epsi_hat=ctx.epsi_hat,
                    nu_hat_dot=nu_dots[direction_index],
                    epsi_hat_dot=epsi_dots[direction_index],
                    f1_full=f1_full,
                    f3_full=f3_full,
                    f1_dot=f1_dot_full[..., direction_index],
                    f3_dot=f3_dot_full[..., direction_index],
                    lambda1=directional_lambda1[..., direction_index],
                    lambda3=directional_lambda3[..., direction_index],
                    lambda1_dot=directional_lambda1_dot[..., direction_index],
                    lambda3_dot=directional_lambda3_dot[..., direction_index],
                    coefficient_bars=coefficient_bars,
                    coefficient_bars_dot=coefficient_bars_dot,
                )

            native_vmec_first_directional = _direct_directional_primitive_bars(
                0, first_coefficient_bars, first_coefficient_bars_dot
            )
            native_vmec_second_directional = _direct_directional_primitive_bars(
                1, second_coefficient_bars, second_coefficient_bars_dot
            )
        else:
            native_vmec_first_directional = _directional_native_primitive_bars(
                0,
                first_coefficient_bars,
                first_coefficient_bars_dot,
            )
            native_vmec_second_directional = _directional_native_primitive_bars(
                1,
                second_coefficient_bars,
                second_coefficient_bars_dot,
            )
        native_vmec_primitive_bars = {
            name: (
                native_vmec_base[name]
                + native_vmec_first_directional[name]
                + native_vmec_second_directional[name]
            )
            for name in native_vmec_base
        }
        native_vmec_coefficient_bars = vmec_geometry_bars_to_coefficients_multi_rhs(
            cast(VmecSurface, prepared.surface),
            prepared.geometry,
            native_vmec_primitive_bars,
        )

    if return_case_bars:

        def _parameter_gradient_from_values(
            nu_hat_value,
            epsi_hat_value,
            f1_value,
            f3_value,
            lambda1_value,
            lambda3_value,
        ):
            return _parameter_gradient_from_adjoint_multi_rhs(
                prepared,
                _operator_context(
                    prepared.surface,
                    prepared.geometry,
                    prepared.grid,
                    nu_hat_value,
                    epsi_hat_value,
                ),
                f1_value,
                f3_value,
                lambda1_value,
                lambda3_value,
            )

        base_nu_implicit, base_epsi_bar = _parameter_gradient_from_values(
            ctx.nu_hat, ctx.epsi_hat, f1_full, f3_full, base_lambda1, base_lambda3
        )
        (
            (first_base_nu_implicit, first_base_epsi_bar),
            (first_nu_implicit_dot, first_epsi_bar_dot),
        ) = jax.jvp(
            _parameter_gradient_from_values,
            (
                ctx.nu_hat,
                ctx.epsi_hat,
                f1_full,
                f3_full,
                directional_lambda1[..., 0],
                directional_lambda3[..., 0],
            ),
            (
                nu_dots[0],
                epsi_dots[0],
                f1_dot_full[..., 0],
                f3_dot_full[..., 0],
                directional_lambda1_dot[..., 0],
                directional_lambda3_dot[..., 0],
            ),
        )
        (
            (second_base_nu_implicit, second_base_epsi_bar),
            (second_nu_implicit_dot, second_epsi_bar_dot),
        ) = jax.jvp(
            _parameter_gradient_from_values,
            (
                ctx.nu_hat,
                ctx.epsi_hat,
                f1_full,
                f3_full,
                directional_lambda1[..., 1],
                directional_lambda3[..., 1],
            ),
            (
                nu_dots[1],
                epsi_dots[1],
                f1_dot_full[..., 1],
                f3_dot_full[..., 1],
                directional_lambda1_dot[..., 1],
                directional_lambda3_dot[..., 1],
            ),
        )
        # Keep the established ten case components even in the compact view.
        # They are small compared with the prepared-system trees, and routing
        # them through NEOPAX's already-proven combination preserves exactly
        # the native mapped contract.  Only the five large prepared trees are
        # compacted to one.
        case_bar_components = (
            base_nu_direct + base_nu_implicit,
            base_epsi_bar,
            directional_nu_direct[..., 0] + first_base_nu_implicit,
            first_base_epsi_bar,
            directional_nu_direct_dot[..., 0] + first_nu_implicit_dot,
            first_epsi_bar_dot,
            directional_nu_direct[..., 1] + second_base_nu_implicit,
            second_base_epsi_bar,
            directional_nu_direct_dot[..., 1] + second_nu_implicit_dot,
            second_epsi_bar_dot,
        )
        if return_vmec_coefficient_bars:
            return (
                (primal_outputs, result, case_bar_components, native_vmec_coefficient_bars)
                if return_primal_outputs
                else (result, case_bar_components, native_vmec_coefficient_bars)
            )
        return (
            (primal_outputs, result, case_bar_components)
            if return_primal_outputs
            else (result, case_bar_components)
        )
    if return_vmec_coefficient_bars:
        return (
            (primal_outputs, result, native_vmec_coefficient_bars)
            if return_primal_outputs
            else (result, native_vmec_coefficient_bars)
        )
    return (primal_outputs, result) if return_primal_outputs else result


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_and_aux(  # noqa: E501 - preserve the existing public API name
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
    return_case_bars: bool = False,
):
    """Combine the base and two directional prepared bars into one compact tree.

    The native primal and matrix-RHS adjoints are shared. Optional case bars
    retain the ten-component contract used by the outer coordinate chain.
    """

    return solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bars_and_aux_fn,
        return_primal_outputs=return_primal_outputs,
        return_case_bars=return_case_bars,
        _compact_result=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_residual_and_aux(  # noqa: E501 - preserve the existing public API name
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bars_and_aux_fn,
    *,
    return_primal_outputs: bool = False,
    return_case_bars: bool = False,
):
    """Return compact bars with dense residual and prepared-input transposes split."""
    return solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bars_and_aux_fn,
        return_primal_outputs=return_primal_outputs,
        return_case_bars=return_case_bars,
        _compact_result=True,
        _compact_residual_transpose=True,
    )


def solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux_packed_support_adjoint(  # noqa: E501 - preserve the existing public API name
    prepared: PreparedMonoenergeticSystem,
    case: MonoenergeticCase,
    first_case_dot: MonoenergeticCase,
    second_case_dot: MonoenergeticCase,
    coefficient_bar_and_aux_fn,
):
    """Pair the two directional adjoint field solves using the shared factors."""

    return _solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
        prepared,
        case,
        first_case_dot,
        second_case_dot,
        coefficient_bar_and_aux_fn,
        include_geometry=False,
        include_prepared=True,
        return_coefficient_aux=True,
        packed_support_directional_adjoint=True,
    )


# Recorded scan transposes
# ------------------------


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
    return {name: values[i] for i, (name, _) in enumerate(_REFERENCE_BLOCKS)}


def _reference_bar_from_blocks(blocks, index):
    return {name: getattr(blocks, field)[index] for name, field in _REFERENCE_BLOCKS}


def _coefficient_bars_from_blocks(blocks, index):
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
    from ._neopax import _surface_reference_bridge

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
    from ._neopax import _surface_reference_bridge

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
) -> NeopaxScanCoefficientBlocks: ...


@overload
def solve_neopax_scan_coefficient_blocks_prepared(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
    return_primal_record: Literal[True],
) -> tuple[NeopaxScanCoefficientBlocks, NeopaxScanCoefficientPrimalRecord]: ...


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
): ...


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
    from ._neopax import _surface_reference_bridge
    from ._neopax_scan import NeopaxScanCoefficientBlocks

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
    from ._neopax import _surface_reference_bridge

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


def pullback_neopax_scan_coefficient_blocks_from_primal_record_batched(
    record: NeopaxScanCoefficientPrimalRecord,
    *,
    coefficient_blocks_bar: NeopaxScanCoefficientBlocks,
) -> tuple[tuple[BoozerSurface | VmecSurface, ...], Array]:
    """Transpose objective rows using one bounded-memory recorded scan.

    Surface metadata stays outside the mapped axis. The compiled row scan
    carries one objective RHS through the prepared transpose at a time and
    emits only numeric surface leaves, avoiding a full objective-batched tape.
    """

    return _pullback_recorded_scan_batched(record, coefficient_blocks_bar, vmec_native=False)


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
    from ._neopax_scan import NeopaxScanCoefficientBlocks

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
    from ._neopax_scan import NeopaxScanCoefficientBlocks

    surface_leaves, surface_tree = jax.tree_util.tree_flatten(surfaces)
    values = _structured_scan_coefficient_block_values(
        surface_tree, tuple(surface_leaves), Es, nu_v, grid
    )
    return NeopaxScanCoefficientBlocks(**dict(zip(_COEFFICIENT_BLOCK_NAMES, values, strict=True)))
