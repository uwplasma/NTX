from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from scipy.constants import elementary_charge

from examples import owned_finite_beta_bootstrap_comparison as example


def test_profile_values_use_radial_derivative_contract():
    contract = example.ProfileContract()
    rho = np.asarray([0.0, 0.5, 1.0])
    values = example._profile_values(rho, contract, a_b=2.0)

    assert values["density"][0] > values["density"][-1]
    assert values["temperature"][0] > values["temperature"][-1]
    assert values["d_density_dr"][0] == 0.0
    assert values["d_temperature_dr"][0] == 0.0
    assert np.all(np.isfinite(values["d_density_dr"]))
    assert np.all(np.isfinite(values["d_temperature_dr"]))


def test_neopax_current_conversion_matches_neopax_examples():
    species = SimpleNamespace(charge_qp=np.asarray([-1.0, 1.0, 2.0]))
    upar_species_first = np.asarray(
        [
            [1.0, 2.0],
            [4.0, 5.0],
            [7.0, 8.0],
        ]
    )
    upar_species_second = upar_species_first.T

    expected = elementary_charge * np.asarray([17.0, 19.0])
    np.testing.assert_allclose(
        example._neopax_current_from_upar(species, upar_species_first, species_axis=0),
        expected,
    )
    np.testing.assert_allclose(
        example._neopax_current_from_upar(species, upar_species_second, species_axis=1),
        expected,
    )


def test_neopax_parallel_current_observable_bridge_matches_fixed_field_convention():
    raw_current = np.asarray([2.0, -3.0])
    b0_over_bbar = np.asarray([0.8, 1.1])

    np.testing.assert_allclose(
        example._redl_observable_from_neopax_current(raw_current, b0_over_bbar),
        np.asarray([-1.6, 3.3]),
    )


def test_write_and_plot_synthetic_payload(tmp_path):
    output_prefix = tmp_path / "owned_finite_beta_bootstrap_comparison"
    rho_field = np.linspace(0.0, 1.0, 5)
    rho = rho_field[1:-1]
    payload = {
        "ntx_neopax": {
            "rho": rho_field.tolist(),
            "density": np.linspace(4.0e20, 0.5e20, rho_field.size).tolist(),
            "temperature": np.linspace(12.0e3, 0.5e3, rho_field.size).tolist(),
        },
        "redl": {
            "epsilon": np.linspace(0.02, 0.08, rho.size).tolist(),
            "trapped_fraction": np.linspace(0.20, 0.42, rho.size).tolist(),
            "L31": np.linspace(0.1, 0.4, rho.size).tolist(),
            "L32": np.linspace(0.05, 0.2, rho.size).tolist(),
            "density_gradient_term_over_root_fsab2": (-0.2e6 * np.arange(1, rho.size + 1)).tolist(),
            "electron_temperature_gradient_term_over_root_fsab2": (
                -0.3e6 * np.arange(1, rho.size + 1)
            ).tolist(),
            "ion_temperature_gradient_term_over_root_fsab2": (
                -0.5e6 * np.arange(1, rho.size + 1)
            ).tolist(),
            "temperature_gradient_term_over_root_fsab2": (
                -0.8e6 * np.arange(1, rho.size + 1)
            ).tolist(),
        },
        "comparison": {
            "rho": rho.tolist(),
            "redl_current_over_root_fsab2": (-1.0e6 * np.arange(1, rho.size + 1)).tolist(),
            "ntx_neopax_nomom_over_root_fsab2": (-0.8e6 * np.arange(1, rho.size + 1)).tolist(),
            "ntx_neopax_total_over_root_fsab2": (-0.9e6 * np.arange(1, rho.size + 1)).tolist(),
            "relative_error_total_vs_redl": np.full(rho.size, 0.1).tolist(),
        },
    }

    example.write_payload(payload, output_prefix)
    example.build_figure(payload, output_prefix)

    assert json.loads(output_prefix.with_suffix(".json").read_text())["comparison"]["rho"]
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
