from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import examples.owned_finite_beta_sfincs_jax_inputs as sfincs_inputs
from examples.owned_geometry_neopax_dataset import OwnedJaxGeometryCase
from ntx import GridSpec


def test_owned_finite_beta_sfincs_jax_inputs_write_same_grid_decks(tmp_path: Path):
    input_path = tmp_path / "input.finite_beta"
    wout_path = tmp_path / "wout_finite_beta.nc"
    input_path.write_text("&INDATA\n/\n")
    wout_path.write_text("placeholder")
    case = OwnedJaxGeometryCase(
        id="finite_beta_fake",
        label="Finite-beta fake",
        family="QA finite beta",
        source="unit test",
        input_path=input_path,
        wout_path=wout_path,
    )

    payload = sfincs_inputs.build_payload(
        case_specs=(case,),
        case_limit=None,
        rho=(0.25, 0.5),
        nu_v=(1.0e-3,),
        es_values=(0.0,),
        grid=GridSpec(5, 7, 9),
        output_dir=tmp_path / "sfincs_decks",
        run_sfincs_jax=False,
    )

    assert payload["benchmark"] == "owned_finite_beta_sfincs_jax_inputs"
    assert payload["summary_metrics"]["deck_count"] == 2
    assert payload["summary_metrics"]["input_written_count"] == 2
    assert payload["normalization_contract"]["rho_to_s"] == "s=rho^2"
    assert "VMECRadialOption=0" in payload["normalization_contract"]["radial_interpolation"]
    assert "nu_n*nuDHat" in payload["normalization_contract"]["pas_collision_frequency_bridge"]

    deck = payload["decks"][0]
    input_text = Path(deck["input_path"]).read_text()
    assert "RHSMode = 3" in input_text
    assert f'equilibriumFile = "{wout_path}"' in input_text
    assert "Ntheta = 5" in input_text
    assert "Nzeta = 7" in input_text
    assert "Nxi = 9" in input_text
    assert "nuPrime = 0.001" in input_text
    assert "VMECRadialOption = 0" in input_text

    output_prefix = tmp_path / "owned_sfincs_inputs"
    sfincs_inputs.write_payload(payload, output_prefix)
    sfincs_inputs.build_figure(payload, output_prefix)
    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()


def test_owned_finite_beta_sfincs_jax_inputs_resolves_relative_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_path = tmp_path / "input.finite_beta"
    wout_path = tmp_path / "wout_finite_beta.nc"
    input_path.write_text("&INDATA\n/\n")
    wout_path.write_text("placeholder")
    case = OwnedJaxGeometryCase(
        id="finite_beta_fake",
        label="Finite-beta fake",
        family="QA finite beta",
        source="unit test",
        input_path=input_path,
        wout_path=wout_path,
    )
    monkeypatch.chdir(tmp_path)

    payload = sfincs_inputs.build_payload(
        case_specs=(case,),
        case_limit=None,
        rho=(0.25,),
        nu_v=(1.0e-3,),
        es_values=(0.0,),
        grid=GridSpec(5, 7, 9),
        output_dir=Path("relative_sfincs_decks"),
        run_sfincs_jax=False,
    )

    deck = payload["decks"][0]
    assert Path(deck["input_path"]).is_absolute()
    assert Path(deck["output_path"]).is_absolute()
    assert Path(deck["input_path"]).is_file()
    assert str(Path(deck["input_path"])).startswith(str(tmp_path))


def test_owned_finite_beta_sfincs_jax_inputs_summarizes_completed_h5(tmp_path: Path):
    output_path = tmp_path / "sfincs_jax_output.h5"
    with h5py.File(output_path, "w") as handle:
        handle["transportMatrix"] = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        handle["B0OverBBar"] = 0.9
        handle["GHat"] = 1.2
        handle["IHat"] = 0.01
        handle["iota"] = 0.5
        handle["psiAHat"] = 0.25
        handle["nu_n"] = 8.0e-4
        handle["Ntheta"] = 5
        handle["x"] = np.asarray([1.0])
        handle["Zs"] = np.asarray([1.0])
        handle["mHats"] = np.asarray([1.0])
        handle["nHats"] = np.asarray([1.0])
        handle["THats"] = np.asarray([1.0])

    summary = sfincs_inputs._summarize_sfincs_output(output_path, nu_prime=1.0e-3)

    assert summary is not None
    assert summary["status"] == "complete"
    assert summary["transportMatrix_shape"] == [2, 2]
    assert summary["sfincs_runtime_nu_n_over_input_nuPrime"] == 0.8
    assert summary["sfincs_pas_nu_d_hat_first_species"] == pytest.approx(0.83602768)
    assert summary["scalars"]["Ntheta"] == 5


def test_owned_finite_beta_sfincs_jax_inputs_compares_ntx_same_grid(
    monkeypatch,
    tmp_path: Path,
):
    case = OwnedJaxGeometryCase(
        id="finite_beta_fake",
        label="Finite-beta fake",
        family="QA finite beta",
        source="unit test",
        input_path=tmp_path / "input.fake",
        wout_path=tmp_path / "wout_fake.nc",
    )
    monkeypatch.setattr(
        sfincs_inputs,
        "surface_from_vmec_jax_vmec_wout_file",
        lambda *args, **kwargs: SimpleNamespace(b0=2.0, psi_a_hat=0.25),
    )
    monkeypatch.setattr(
        sfincs_inputs,
        "solve_monoenergetic",
        lambda *args, **kwargs: SimpleNamespace(
            D11=1.0,
            D13=-2.0,
            D31=2.0,
            D33=3.0,
            D33_spitzer=4.0,
        ),
    )
    summary = {
        "transportMatrix": [
            [0.0, 8.0 / np.sqrt(np.pi)],
            [8.0 / np.sqrt(np.pi), 8.0],
        ],
        "scalars": {
            "GHat": 1.0,
            "IHat": 0.0,
            "iota": 0.5,
            "psiAHat": 0.5,
            "B0OverBBar": 2.0,
            "nu_n": 1.0e-3,
        },
    }

    comparison = sfincs_inputs._ntx_same_grid_transport_summary(
        case=case,
        rho=0.5,
        e_star=0.0,
        grid=GridSpec(5, 7, 9),
        min_bmn_to_load=0.0,
        transport_summary=summary,
    )

    assert comparison["status"] == "complete"
    assert comparison["relative_difference"]["L13_bridge_vs_sfincs"] == 0.0
    assert comparison["relative_difference"]["L31_bridge_vs_sfincs"] == 0.0
    assert comparison["relative_difference"]["L33_bridge_vs_sfincs"] == 0.0
    assert comparison["relative_difference"]["L33_spitzer_bridge_vs_sfincs"] > 0.0
