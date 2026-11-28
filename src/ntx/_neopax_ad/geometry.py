"""Analytic transposes of sampled VMEC fields and fixed-residual primitive bars."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .._geometry import GeometryOnGrid, VmecSurface
from .._solver import PreparedMonoenergeticSystem, _operator_context
from ..grids import flatten_fs
from ..operators import OperatorContext, block_parameters, parameter_derivative_blocks


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
