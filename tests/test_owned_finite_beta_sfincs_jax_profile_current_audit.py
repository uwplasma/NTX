from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import examples.owned_finite_beta_sfincs_jax_profile_current_audit as audit
from examples.owned_geometry_neopax_dataset import OwnedJaxGeometryCase
from ntx import GridSpec


def test_profile_current_audit_writes_rhs1_deck(tmp_path: Path):
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

    payload = audit.build_payload(
        case_id="finite_beta_fake",
        case_specs=(case,),
        rho=(0.5,),
        nu_n=(8.31565e-3,),
        grid=GridSpec(5, 7, 9),
        nx=3,
        output_dir=tmp_path / "profile_decks",
        run_sfincs_jax=False,
    )

    assert payload["benchmark"] == "owned_finite_beta_sfincs_jax_profile_current_audit"
    assert payload["summary_metrics"]["deck_count"] == 1
    assert payload["summary_metrics"]["input_written_count"] == 1
    assert payload["normalization_contract"]["rho_to_s"] == "s=rho^2"
    assert "FSABjHatOverRootFSAB2" in payload["normalization_contract"]["current_observable"]

    deck = payload["decks"][0]
    assert deck["n_hat"] == pytest.approx(3.996484375)
    assert deck["t_hat"] == pytest.approx(9.125)
    input_text = Path(deck["input_path"]).read_text()
    assert "RHSMode = 1" in input_text
    assert f'equilibriumFile = "{wout_path}"' in input_text
    assert "inputRadialCoordinateForGradients = 3" in input_text
    assert "Zs = 1.0 -1.0" in input_text
    assert "mHats = 2" in input_text
    assert "nu_n = 0.00831565" in input_text
    assert "includeXDotTerm = .true." in input_text
    assert "useDKESExBDrift = .false." in input_text
    assert "Ntheta = 5" in input_text
    assert "Nx = 3" in input_text


def test_profile_current_audit_summarizes_h5_current(tmp_path: Path):
    output_path = tmp_path / "sfincsOutput.h5"
    with h5py.File(output_path, "w") as handle:
        handle["FSABjHat"] = np.asarray([-0.25])
        handle["FSABjHatOverRootFSAB2"] = np.asarray([-0.5])
        handle["RHSMode"] = 1
        handle["collisionOperator"] = 1
        handle["Ntheta"] = 5
        handle["nu_n"] = 8.31565e-3

    summary = audit._summarize_profile_output(output_path)

    assert summary is not None
    assert summary["status"] == "complete"
    assert summary["fsab_jhat"] == pytest.approx(-0.25)
    assert summary["fsab_jhat_over_root_fsab2"] == pytest.approx(-0.5)
    assert summary["current_over_root_fsab2_am2"] == pytest.approx(
        -0.5 * audit.SFINCS_JHAT_TO_AM2
    )
    assert summary["scalars"]["RHSMode"] == 1
    assert summary["scalars"]["nu_n"] == pytest.approx(8.31565e-3)


def test_profile_current_audit_builds_empty_status_figure(tmp_path: Path):
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
    payload = audit.build_payload(
        case_id="finite_beta_fake",
        case_specs=(case,),
        rho=(0.25,),
        nu_n=(1.0e-3,),
        grid=GridSpec(5, 7, 9),
        nx=3,
        output_dir=tmp_path / "profile_decks",
        run_sfincs_jax=False,
    )

    output_prefix = tmp_path / "profile_current_audit"
    audit.write_payload(payload, output_prefix)
    audit.build_figure(payload, output_prefix)

    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    written = json.loads(output_prefix.with_suffix(".json").read_text())
    assert written["figure_png"] == str(output_prefix.with_suffix(".png"))
    assert written["figure_pdf"] == str(output_prefix.with_suffix(".pdf"))
