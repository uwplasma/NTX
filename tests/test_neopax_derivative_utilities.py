"""Tiny generated systems check the opt-in NEOPAX derivatives and scan records."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ntx
import ntx.solver
from ntx import _neopax_ad as ad
from ntx import _neopax_scan as scan
from ntx.geometry import VmecSurface, evaluate_fourier_series, geometry_on_grid

GRID = ntx.GridSpec(3, 3, 2)
CASE = ntx.MonoenergeticCase(1e-2, epsi_hat=1e-3)
FIRST = ntx.MonoenergeticCase(0.13, epsi_hat=-0.27)
SECOND = ntx.MonoenergeticCase(-0.19, epsi_hat=0.41)
BARS = (
    jnp.array([0.7, -0.2, 0.1, 0.3, -0.4]),
    jnp.array([-0.1, 0.3, 0.2, -0.4, 0.5]),
    jnp.array([0.4, 0.1, -0.6, 0.3, -0.2]),
)


def _prepared(surface=None):
    return ntx.prepare_monoenergetic_system(
        ntx.example_surface() if surface is None else surface, GRID
    )


def _vmec_surface():
    return VmecSurface(
        path=Path("generated.nc"),
        requested_psi_n=0.2,
        psi_n=0.2,
        nfp=2,
        ns=3,
        mpol=2,
        ntor=1,
        total_mode_count=2,
        loaded_mode_count=2,
        iota=0.6,
        m=jnp.array([0, 1]),
        n=jnp.array([0, 1]),
        b0=1.0,
        psi_a_hat=1.0,
        phi_edge=1.0,
        r_n=0.5,
        r_hat=0.5,
        dpsi_hat_dr_hat=1.0,
        dr_hat_dpsi_hat=1.0,
        transport_psi_scale=1.0,
        b_cos=jnp.array([1.0, 0.1]),
        jacobian_cos=jnp.array([1.0, 0.02]),
        b_sub_theta_cos=jnp.array([0.2, 0.01]),
        b_sub_zeta_cos=jnp.array([1.1, 0.03]),
        b_sup_theta_cos=jnp.array([0.3, 0.04]),
        b_sup_zeta_cos=jnp.array([1.2, 0.05]),
    )


def _close(actual, expected, rtol=1e-8, atol=1e-9):
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    assert actual_tree == expected_tree
    for left, right in zip(actual_leaves, expected_leaves, strict=True):
        if np.issubdtype(np.asarray(right).dtype, np.inexact):
            np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)


def _raw_directions(prepared, case, first, second):
    def value(local_case):
        return ntx.solve_prepared_coefficient_vector(prepared, local_case)

    return value(case), jax.jvp(value, (case,), (first,))[1], jax.jvp(value, (case,), (second,))[1]


def _bars_and_aux(*values):
    return *BARS, {"direct": jnp.sum(values[0])}


def _batched_bars_with_dot(*values):
    bars = tuple(jnp.stack([bar, -0.7 * bar]) for bar in BARS)
    return (
        *bars,
        {"direct": jnp.repeat(jnp.sum(values[0])[None], 2)},
        (0.2 * bars[1], -0.3 * bars[2]),
    )


def _add_bars(*trees):
    def add(*leaves):
        if any(np.asarray(x).dtype == jax.dtypes.float0 for x in leaves):
            return leaves[0]
        return sum(leaves)

    return jax.tree.map(add, *trees)


def _take_row(tree, i):
    def take(value):
        row = value[i]
        return np.asarray(row) if value.dtype == jax.dtypes.float0 else row

    return jax.tree.map(take, tree)


def _block_bars(blocks, batched=False):
    return ntx.NeopaxScanCoefficientBlocks(
        **{
            name: (
                jnp.stack(
                    [
                        jnp.full_like(getattr(blocks, name), 0.01 * (i + 1)),
                        jnp.full_like(getattr(blocks, name), -0.02 * (i + 1)),
                    ]
                )
                if batched
                else jnp.full_like(getattr(blocks, name), 0.01 * (i + 1))
            )
            for i, name in enumerate(blocks.__dataclass_fields__)
        }
    )


def _block_tuple(blocks):
    return tuple(getattr(blocks, name) for name in blocks.__dataclass_fields__)


def test_factorized_hessian_and_directions_match_raw_finite_differences():
    prepared = _prepared()

    def value(nu, epsi):
        return ntx.solve_prepared_coefficient_vector(
            prepared, ntx.MonoenergeticCase(nu, epsi_hat=epsi)
        )

    nu, epsi, hn, he = 0.01, 0.001, 1e-5, 1e-5
    base = value(nu, epsi)
    np_, nm = value(nu + hn, epsi), value(nu - hn, epsi)
    ep, em = value(nu, epsi + he), value(nu, epsi - he)
    reference = (
        base,
        (np_ - nm) / (2 * hn),
        (ep - em) / (2 * he),
        (np_ - 2 * base + nm) / hn**2,
        (
            value(nu + hn, epsi + he)
            - value(nu + hn, epsi - he)
            - value(nu - hn, epsi + he)
            + value(nu - hn, epsi - he)
        )
        / (4 * hn * he),
        (ep - 2 * base + em) / he**2,
    )
    actual = ntx.solve_prepared_coefficient_vector_hessian_factorized(prepared, CASE)
    _close(actual, reference, rtol=3e-3, atol=1e-7)
    result = ntx.solve_prepared_coefficient_vector_two_directional_factorized(
        prepared, CASE, FIRST, SECOND
    )
    _close(
        result,
        (
            actual[0],
            FIRST.nu_hat * actual[1] + FIRST.epsi_hat * actual[2],
            SECOND.nu_hat * actual[1] + SECOND.epsi_hat * actual[2],
        ),
    )


def test_directional_and_prepared_custom_vjps_match_raw_ad():
    prepared = _prepared()
    for coordinates in ("epsi", "er", "none"):
        case, first, second = CASE, FIRST, SECOND
        if coordinates == "er":
            case, first, second = (
                ntx.MonoenergeticCase(c.nu_hat, er_hat=c.epsi_hat) for c in (CASE, FIRST, SECOND)
            )
        elif coordinates == "none":
            case, first, second = (ntx.MonoenergeticCase(c.nu_hat) for c in (CASE, FIRST, SECOND))
        expected, expected_pullback = jax.vjp(_raw_directions, prepared, case, first, second)
        actual, actual_pullback = jax.vjp(
            ntx.solve_prepared_coefficient_vector_two_directional_prepared_vjp,
            prepared,
            case,
            first,
            second,
        )
        _close(actual, expected)
        _close(
            ntx.solve_prepared_coefficient_vector_two_directional_prepared_vjp(
                prepared, case, first, second
            ),
            expected,
        )
        _close(actual_pullback(BARS), expected_pullback(BARS), rtol=2e-8)

        def directional(local_case, direction):
            return jax.jvp(
                lambda value: ntx.solve_prepared_coefficient_vector(prepared, value),
                (local_case,),
                (direction,),
            )[1]

        _, pullback = jax.vjp(directional, case, first)
        base_direction_bar, coefficient_case_bar = pullback(BARS[1])
        _close(
            ad.solve_prepared_coefficient_vector_derivative_vjp(prepared, case, first, BARS[1]),
            (coefficient_case_bar, base_direction_bar),
        )
        _, pullback = jax.vjp(ntx.solve_prepared_coefficient_vector, prepared, case)
        reference_prepared, reference_case = pullback(BARS[0])
        _close(
            ad.pullback_prepared_coefficient_vector_case_and_prepared(prepared, case, BARS[0]),
            (reference_case, reference_prepared),
        )

        # Geometry-only bars hold resolved epsilon fixed; the full prepared API
        # above additionally owns the er_hat / transport_psi_scale chain rule.
        def geometry_solver(geometry, local_case):
            resolved_case = ntx.MonoenergeticCase(
                local_case.nu_hat,
                epsi_hat=local_case.resolved_epsi_hat(prepared.geometry.transport_psi_scale),
            )
            return ntx.solve_prepared_coefficient_vector(
                replace(prepared, geometry=geometry), resolved_case
            )

        _, geometry_pb = jax.vjp(geometry_solver, prepared.geometry, case)
        expected_geometry, expected_case = geometry_pb(BARS[0])
        _close(
            ad.pullback_prepared_coefficient_vector_case_and_geometry(prepared, case, BARS[0]),
            (expected_case, expected_geometry),
        )

        def geometry_pullback(local_case):
            _, pb = jax.vjp(geometry_solver, prepared.geometry, local_case)
            geometry_bar, case_bar = pb(BARS[1])
            return case_bar, geometry_bar

        base_bars, tangent_bars = jax.jvp(geometry_pullback, (case,), (first,))
        _close(
            ad.pullback_prepared_coefficient_vector_case_and_geometry_derivative(
                prepared, case, first, BARS[1]
            ),
            (base_bars[0], tangent_bars[0], base_bars[1], tangent_bars[1]),
            rtol=2e-8,
        )
        batched = jnp.stack([BARS[0], BARS[1]])
        for i in range(2):
            result = ad.pullback_prepared_coefficient_vector_case_and_prepared_multi_rhs(
                prepared, case, batched
            )
            expected_prepared, expected_case = pullback(batched[i])
            _close(jax.tree.map(lambda x, i=i: x[i], result), (expected_case, expected_prepared))


def test_lowdot_scalar_geometry_and_prepared_views_agree():
    prepared = _prepared()
    args = prepared, CASE, FIRST, SECOND
    plain = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks(*args, lambda *x: BARS)
    full = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux(
        *args, _bars_and_aux
    )
    _close(plain, (full[0], full[1], *full[3:7], *full[9:13]))
    support = (
        ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_and_aux(  # noqa: E501 - preserve the existing public API name
            *args, _bars_and_aux
        )
    )
    _close(support, (full[2], full[7], full[8], full[13], full[14], full[15]))
    _close(
        ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_prepared_and_aux_packed_support_adjoint(  # noqa: E501 - preserve the existing public API name
            *args, _bars_and_aux
        ),
        full,
    )
    geo = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry(
        *args, lambda *x: BARS
    )
    geo_aux = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_with_geometry_and_aux(
        *args, _bars_and_aux
    )
    _close(geo, geo_aux[:-1])
    _close(
        ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_geometry_support_only_and_aux(  # noqa: E501 - preserve the existing public API name
            *args, _bars_and_aux
        ),
        (geo[2], geo[7], geo[8], geo[13], geo[14], geo_aux[-1]),
    )

    def batched_bars(*values):
        return *(jnp.stack([bar, -0.7 * bar]) for bar in BARS), {
            "direct": jnp.repeat(jnp.sum(values[0])[None], 2)
        }

    generic = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
        *args, batched_bars
    )
    native = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
        *args, batched_bars
    )
    _close(native, generic)
    primal, with_dot = (
        ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
            *args, _batched_bars_with_dot, return_primal_outputs=True
        )
    )
    _close(primal, _raw_directions(*args))
    native_dot = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux(  # noqa: E501 - preserve the existing public API name
        *args, _batched_bars_with_dot, native_direct_coefficient_pullback=True
    )
    _close(native_dot, with_dot)
    expected_compact = _add_bars(with_dot[0], with_dot[2], with_dot[4])
    for helper in (
        ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_and_aux,  # noqa: E501 - preserve the existing public API name
        ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_compact_residual_and_aux,  # noqa: E501 - preserve the existing public API name
    ):
        primal, result, cases = helper(
            *args, _batched_bars_with_dot, return_primal_outputs=True, return_case_bars=True
        )
        _close(primal, _raw_directions(*args))
        _close(result, (expected_compact, with_dot[-1]), rtol=2e-8)
        assert len(cases) == 10
        _close(helper(*args, batched_bars)[0], _add_bars(generic[0], generic[2], generic[4]))


def test_recorded_boozer_scan_matches_generic_ad_and_reuses_primal_systems(monkeypatch):
    surfaces = (ntx.example_surface(),)
    es, nu = jnp.array([[0.0, 1e-3]]), jnp.array([0.01, 0.02])
    blocks, record = ad.solve_neopax_scan_coefficient_blocks_prepared(
        surfaces, Es=es, nu_v=nu, grid=GRID, return_primal_record=True
    )
    generic = scan.solve_neopax_scan_coefficient_blocks(surfaces, Es=es, nu_v=nu, grid=GRID)
    _close(_block_tuple(blocks), _block_tuple(generic))

    def oracle(local_surfaces, local_es):
        return _block_tuple(
            scan.solve_neopax_scan_coefficient_blocks(
                local_surfaces, Es=local_es, nu_v=nu, grid=GRID
            )
        )

    _, pullback = jax.vjp(oracle, surfaces, es)
    bars = _block_bars(blocks)
    reference = pullback(_block_tuple(bars))
    _close(
        ad.pullback_neopax_scan_coefficient_blocks_prepared(
            surfaces, Es=es, nu_v=nu, grid=GRID, coefficient_blocks_bar=bars
        ),
        reference,
    )
    original_prepare = ad.prepare_monoenergetic_system

    def prepare_chain_rule_only(surface, grid):
        assert isinstance(surface.b_cos, jax.core.Tracer), (
            "Recorded transpose reprepared an eager primal"
        )
        return original_prepare(surface, grid)

    monkeypatch.setattr(ad, "prepare_monoenergetic_system", prepare_chain_rule_only)
    _close(
        ad.pullback_neopax_scan_coefficient_blocks_from_primal_record(
            record, coefficient_blocks_bar=bars
        ),
        reference,
    )
    rows = _block_bars(blocks, batched=True)
    actual = ad.pullback_neopax_scan_coefficient_blocks_from_primal_record_batched(
        record, coefficient_blocks_bar=rows
    )
    for i in range(2):
        _close(
            jax.tree.map(lambda x, i=i: x[i], actual),
            pullback(tuple(x[i] for x in _block_tuple(rows))),
        )

    def structured(local_surfaces, local_es):
        return _block_tuple(
            ad.solve_neopax_scan_coefficient_blocks_prepared_structured_vjp(
                local_surfaces, Es=local_es, nu_v=nu, grid=GRID
            )
        )

    monkeypatch.undo()
    value, structured_pullback = jax.vjp(structured, surfaces, es)
    _close(value, _block_tuple(generic))
    _close(structured_pullback(_block_tuple(bars)), reference)
    kwargs = dict(rho=jnp.array([0.3]), nu_v=nu, Es=es, drds=jnp.array([1.0]), grid=GRID)
    before = ntx.build_ntx_neopax_scan_from_surfaces(surfaces, **kwargs)
    after, saved = ntx.build_ntx_neopax_scan_from_surfaces(
        surfaces, **kwargs, coefficient_reverse_mode="structured", return_primal_record=True
    )
    np.testing.assert_allclose(before.D11, after.D11)
    assert saved.prepared
    ntx.build_ntx_neopax_scan_from_surfaces(
        surfaces, **kwargs, coefficient_reverse_mode="structured"
    )


def test_recorded_vmec_scan_native_rows_match_generic_transpose():
    surfaces = (_vmec_surface(),)
    es, nu = jnp.array([[0.0, 1e-3]]), jnp.array([0.01, 0.02])
    blocks, record = ad.solve_neopax_scan_coefficient_blocks_prepared(
        surfaces, Es=es, nu_v=nu, grid=GRID, return_primal_record=True
    )
    rows = _block_bars(blocks, batched=True)
    expected = ad.pullback_neopax_scan_coefficient_blocks_from_primal_record_batched(
        record, coefficient_blocks_bar=rows
    )
    actual = ad.pullback_neopax_scan_coefficient_blocks_from_primal_record_batched_vmec_native(
        record, coefficient_blocks_bar=rows
    )
    _close(actual, expected, rtol=1e-8, atol=1e-8)
    prepared = record.prepared[0]
    bars = jnp.stack([BARS[0], BARS[1]])
    case_bars, primitive = ad.pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs(
        prepared, CASE, bars
    )
    coefficient_case, coefficient_bars = (
        ad.pullback_prepared_database_coefficients_to_vmec_multi_rhs(prepared, CASE, bars)
    )
    _close(case_bars, coefficient_case)
    assert "b_cos" in coefficient_bars and primitive
    args = prepared, CASE, FIRST, SECOND, _batched_bars_with_dot
    native = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux  # noqa: E501 - preserve the existing public API name
    full = native(*args)
    combined = _add_bars(full[0], full[2], full[4])
    _, prepare_pullback = jax.vjp(lambda s: ntx.prepare_monoenergetic_system(s, GRID), surfaces[0])
    expected_surface = [prepare_pullback(_take_row(combined, i))[0] for i in range(2)]
    for direct in (False, True):
        primal, result, cases, coefficients = native(
            *args,
            _compact_result=True,
            return_vmec_coefficient_bars=True,
            native_vmec_direct_directional_product_rule=direct,
            native_direct_coefficient_pullback=direct,
            return_primal_outputs=True,
            return_case_bars=True,
        )
        _close(result, (combined, full[-1]), rtol=2e-8)
        _close(primal, _raw_directions(prepared, CASE, FIRST, SECOND))
        assert len(cases) == 10
        for name, value in coefficients.items():
            np.testing.assert_allclose(
                value, jnp.stack([getattr(s, name) for s in expected_surface]), rtol=2e-8, atol=1e-8
            )
    result, native_only = native(
        *args,
        _compact_result=True,
        return_vmec_coefficient_bars=True,
        native_vmec_coefficient_bars_only=True,
    )
    _close(native_only, coefficients, rtol=2e-8)
    assert all(np.all(x == 0) for x in jax.tree.leaves(result[0]))


def test_native_geometry_transposes_match_fourier_vjps():
    from ntx.grids import fourier_derivative_matrix

    period = jnp.array(2.4)
    matrix, derivative = jax.jvp(
        lambda p: fourier_derivative_matrix(3, p), (period,), (jnp.array(1.0),)
    )
    np.testing.assert_allclose(derivative, -matrix / period, atol=1e-12)
    surface = _vmec_surface()
    geometry = geometry_on_grid(surface, GRID)
    values = jnp.arange(18.0).reshape(2, 3, 3) / 19.0
    field_bars = {
        name: values * (i + 1) / 10
        for i, name in enumerate(
            (
                "b",
                "d_b_dtheta",
                "d_b_dzeta",
                "jacobian",
                "b_sub_theta",
                "b_sub_zeta",
                "b_sup_theta",
                "b_sup_zeta",
            )
        )
    }
    actual = ad.vmec_sampled_field_bars_to_coefficients_multi_rhs(surface, geometry, field_bars)

    def field(coefficients):
        return evaluate_fourier_series(
            surface.m, surface.n, coefficients, geometry.theta_2d, geometry.zeta_2d, nfp=surface.nfp
        )

    _, pullback = jax.vjp(field, surface.b_cos)
    expected = jax.vmap(lambda b, bt, bz: pullback((b, bt, bz))[0])(
        field_bars["b"], field_bars["d_b_dtheta"], field_bars["d_b_dzeta"]
    )
    np.testing.assert_allclose(actual["b_cos"], expected, atol=1e-12)
    ad.fourier_series_coefficient_bars_multi_rhs(
        surface.m, surface.n, geometry.theta_2d, geometry.zeta_2d, nfp=surface.nfp, value_bar=values
    )
    volume_bars = jnp.array([0.2, -0.3])
    assert (
        ad.volume_prime_bars_multi_rhs(volume_bars, geometry.jacobian, geometry.grid).shape
        == values.shape
    )
    with pytest.raises(ValueError):
        ad.vmec_geometry_bars_to_coefficients_multi_rhs(ntx.example_surface(), geometry, {})


def test_new_utilities_validate_modes_shapes_and_static_cotangents():
    prepared = _prepared()
    assert ad._COEFFICIENT_BLOCK_NAMES == tuple(
        ntx.NeopaxScanCoefficientBlocks.__dataclass_fields__
    )
    assert (
        ntx.solver.solve_prepared_coefficient_vector_hessian_factorized
        is ntx.solve_prepared_coefficient_vector_hessian_factorized
    )
    args = prepared, CASE, FIRST, SECOND, _bars_and_aux
    for options in (
        {"include_geometry": True, "include_prepared": True},
        {"include_geometry": False, "support_only": True},
        {"include_geometry": False, "packed_support_directional_adjoint": True},
    ):
        with pytest.raises(ValueError):
            ad._solve_prepared_coefficient_vector_lowdot_two_pullbacks_core(
                *args, return_coefficient_aux=True, **options
            )
    for shape in ((5,), (1, 4)):
        with pytest.raises(ValueError):
            ad._coefficient_mode_pullback_multi_rhs_direct(
                prepared.geometry, 0.01, jnp.zeros(shape)
            )
    fields = jnp.zeros((3, 9))
    adjoints = jnp.zeros((3, 9, 2))
    for helper in (
        ad.pullback_prepared_coefficient_vector_case_and_prepared_multi_rhs,
        ad.pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs,
    ):
        with pytest.raises(ValueError):
            helper(prepared, CASE, BARS[0])
    with pytest.raises(ValueError, match="epsi_hat"):
        ad.pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs(
            prepared, ntx.MonoenergeticCase(0.01), jnp.stack([BARS[0]])
        )
    with pytest.raises(ValueError):
        ad._solve_factorized_adjoint_field_pair(None, None, None, None, fields, fields)
    for helper, value in (
        (ad._solve_factorized_directional_adjoint_field_pair, adjoints),
        (
            ad._solve_factorized_multi_rhs_directional_adjoint_field_pair,
            adjoints[..., None],
        ),
    ):
        with pytest.raises(ValueError):
            helper(None, None, None, None, value, value, value, value, jnp.zeros((1,)))
    with pytest.raises(ValueError):
        ad._parameter_gradient_from_adjoint_multi_rhs(
            prepared, None, fields, fields, fields, fields
        )
    for lam, primal in (
        (adjoints, jnp.zeros(9)),
        (fields, fields),
        (adjoints, jnp.zeros((2, 9))),
        (jnp.zeros((3, 8, 2)), fields),
    ):
        with pytest.raises(ValueError):
            ad._fixed_residual_block_coefficient_bars_multi_rhs(prepared, primal, primal, lam, lam)
    for lam in (fields, jnp.zeros((2, 9, 2))):
        with pytest.raises(ValueError):
            ad._fixed_residual_source_geometry_bars_multi_rhs(lam, lam)
    with pytest.raises(ValueError, match="VmecSurface"):
        ad._native_vmec_primitive_bars_from_fixed_adjoint_multi_rhs(
            prepared, None, fields, fields, adjoints, adjoints, jnp.stack([BARS[0], BARS[1]])
        )
    packed = jnp.zeros((3, 2, 3, 9))
    for lower, diagonal, upper in (
        (packed, packed[:2], packed),
        (packed[:2], packed[:2], packed[:2]),
    ):
        with pytest.raises(ValueError):
            ad._block_parameters_bar_from_coefficient_bars_multi_rhs(
                prepared, {}, lower, diagonal, upper
            )
    with pytest.raises(ValueError):
        ad._direct_coefficient_geometry_bars_multi_rhs(prepared, fields, fields, 0.01, BARS[0])

    # Omitted cotangent directions mean zero, for every residual view.
    kwargs_direction = dict(
        nu_hat=jnp.array(0.01),
        epsi_hat=jnp.array(0.001),
        nu_hat_dot=jnp.array(0.13),
        epsi_hat_dot=jnp.array(-0.27),
        f1_full=fields,
        f3_full=fields,
        f1_dot=fields,
        f3_dot=fields,
        lambda1=fields,
        lambda3=fields,
        lambda1_dot=fields,
        lambda3_dot=fields,
        coefficient_bar=BARS[0],
    )
    for helper in (
        ad._directional_geometry_gradient_from_adjoint,
        ad._directional_prepared_gradient_from_adjoint,
        ad._directional_compact_prepared_gradient_from_adjoint,
    ):
        _close(
            helper(prepared, **kwargs_direction),
            helper(prepared, **kwargs_direction, coefficient_bar_dot=jnp.zeros(5)),
        )

    native = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_native_multi_rhs_and_aux  # noqa: E501 - preserve the existing public API name
    for options in (
        {"_compact_residual_transpose": True},
        {"return_vmec_coefficient_bars": True},
        {"native_vmec_coefficient_bars_only": True},
        {
            "_compact_result": True,
            "_compact_residual_transpose": True,
            "return_vmec_coefficient_bars": True,
            "native_vmec_coefficient_bars_only": True,
        },
        {"native_vmec_direct_directional_product_rule": True},
    ):
        with pytest.raises(ValueError):
            native(prepared, CASE, FIRST, SECOND, _batched_bars_with_dot, **options)
    with pytest.raises(ValueError, match="epsi_hat"):
        native(prepared, ntx.MonoenergeticCase(0.01), FIRST, SECOND, _batched_bars_with_dot)
    generic = ad.solve_prepared_coefficient_vector_lowdot_two_pullbacks_prepared_support_only_multi_rhs_and_aux  # noqa: E501 - preserve the existing public API name
    for helper in (native, generic):
        for callback in (lambda *x: BARS, lambda *x: (*BARS, {}, (BARS[0], BARS[0]))):
            with pytest.raises(ValueError):
                helper(prepared, CASE, FIRST, SECOND, callback)
    kwargs = dict(
        rho=jnp.array([0.3]),
        nu_v=jnp.array([0.01]),
        Es=jnp.array([[0.0]]),
        drds=jnp.array([1.0]),
        grid=GRID,
    )
    for options in ({"coefficient_reverse_mode": "bad"}, {"return_primal_record": True}):
        with pytest.raises(ValueError):
            ntx.build_ntx_neopax_scan_from_surfaces((ntx.example_surface(),), **kwargs, **options)
    record = ntx.NeopaxScanCoefficientPrimalRecord(
        (ntx.example_surface(),), (prepared,), jnp.array([[0.0]]), jnp.array([0.01]), GRID
    )
    empty = ntx.NeopaxScanCoefficientBlocks(
        **{
            name: jnp.zeros((0, 1, 1, 1))
            for name in ntx.NeopaxScanCoefficientBlocks.__dataclass_fields__
        }
    )
    for helper in (
        ad.pullback_neopax_scan_coefficient_blocks_from_primal_record_batched,
        ad.pullback_neopax_scan_coefficient_blocks_from_primal_record_batched_vmec_native,
    ):
        with pytest.raises(ValueError):
            helper(record, coefficient_blocks_bar=empty)
    real = replace(empty, **{name: jnp.zeros((1, 1, 1, 1)) for name in empty.__dataclass_fields__})
    with pytest.raises(ValueError, match="one entry"):
        ad.pullback_neopax_scan_coefficient_blocks_prepared(
            record.surfaces,
            Es=record.Es,
            nu_v=record.nu_v,
            grid=GRID,
            coefficient_blocks_bar=real,
            prepared_systems=(),
        )
    with pytest.raises(ValueError, match="VMEC"):
        ad.pullback_neopax_scan_coefficient_blocks_from_primal_record_batched_vmec_native(
            record, coefficient_blocks_bar=real
        )
    assert ad._add_float_trees(jnp.array(1.0), np.zeros((), dtype=jax.dtypes.float0)) == 1.0
    assert ad._add_float_trees(np.zeros((), dtype=jax.dtypes.float0), jnp.array(2.0)) == 2.0
