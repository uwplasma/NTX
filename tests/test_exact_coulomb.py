from __future__ import annotations

import numpy as np

from ntx.exact_coulomb import (
    active_p2_like_species_reference_blocks,
    appendix_b_l1_a,
    appendix_b_l1_b,
    exact_l1_like_species_runtime_blocks,
    l1_c0_c1_from_flow_heat,
    l1_lambda_factors,
    l1_runtime_basis_factors,
    l1_runtime_coefficients_from_exact_moments,
    l1_sigma_factors,
)


def test_l1_sigma_and_runtime_basis_match_exact_low_order_values():
    assert np.allclose(l1_lambda_factors(3), np.array([1.5, 3.75, 105.0 / 16.0]))
    assert np.allclose(l1_sigma_factors(3), np.array([0.5, 1.25, 35.0 / 16.0]))
    assert np.allclose(l1_runtime_basis_factors(3), np.array([1.0, 0.4, 8.0 / 35.0]))


def test_runtime_coefficients_follow_exact_dimensionless_l1_moments():
    coeffs = l1_runtime_coefficients_from_exact_moments(
        2.0, np.array([3.0, 5.0, 7.0])
    )
    expected = 2.0 * np.array([1.0, 0.4, 8.0 / 35.0]) * np.array([3.0, 5.0, 7.0])
    assert np.allclose(coeffs, expected)


def test_first_two_runtime_coefficients_match_flow_heat_map():
    c0, c1 = l1_c0_c1_from_flow_heat(
        flow_velocity=3.5,
        heat_flux=10.0,
        density=4.0,
        temperature=2.0,
    )
    assert c0 == 3.5
    assert np.isclose(c1, 3.5 - 2.0 * 10.0 / (5.0 * 4.0 * 2.0))


def test_appendix_b_like_species_l1_coefficients_match_reference_values():
    a = appendix_b_l1_a(theta=1.0, mu=1.0)
    b = appendix_b_l1_b(theta=1.0, mu=1.0)
    assert np.allclose(
        a,
        np.array(
            [
                [1.0, 1.5, 15.0 / 8.0],
                [1.5, 59.0 / 4.0, 417.0 / 16.0],
                [15.0 / 8.0, 417.0 / 16.0, 8385.0 / 64.0],
            ]
        ),
    )
    assert np.allclose(
        b,
        np.array(
            [
                [1.0, 1.5, 15.0 / 8.0],
                [1.5, 27.0 / 4.0, 225.0 / 16.0],
                [15.0 / 8.0, 225.0 / 16.0, 2625.0 / 64.0],
            ]
        ),
    )


def test_exact_like_species_runtime_blocks_do_not_reduce_to_sign_only_reference():
    exact_a, exact_b = exact_l1_like_species_runtime_blocks()
    ref_a, ref_b = active_p2_like_species_reference_blocks()

    # The sign-only heat-flow transform is not enough: even after the exact
    # l=1 basis normalization, the current low-order closure does not collapse
    # onto the exact like-species Coulomb block.
    exact_a_n = exact_a / abs(exact_a[0, 0])
    exact_b_n = exact_b / abs(exact_b[0, 0])
    ref_a_n = ref_a / abs(ref_a[0, 0])
    ref_b_n = ref_b / abs(ref_b[0, 0])

    assert not np.allclose(exact_a_n, ref_a_n, rtol=0.05, atol=0.05)
    assert not np.allclose(exact_b_n, ref_b_n, rtol=0.05, atol=0.05)
