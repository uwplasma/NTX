from __future__ import annotations

from pathlib import Path

import pytest

import examples.owned_geometry_neopax_dataset as owned_dataset
from ntx import GridSpec, example_surface


def test_owned_geometry_neopax_dataset_records_provenance_without_external_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    case = owned_dataset.OwnedJaxGeometryCase(
        id="fake_owned_case",
        label="Fake owned case",
        family="test",
        source="unit test",
        input_path=tmp_path / "input.fake",
        wout_path=tmp_path / "wout_fake.nc",
        notes="schema-only smoke test",
    )

    def fake_build_surfaces_for_path(_case, *, rho, **_kwargs):
        return tuple(example_surface() for _ in rho)

    monkeypatch.setattr(
        owned_dataset,
        "_build_surfaces_for_path",
        fake_build_surfaces_for_path,
    )

    payload = owned_dataset.build_payload(
        case_specs=(case,),
        case_limit=None,
        rho=(0.25, 0.5),
        nu_v=(1.0e-3,),
        grid=GridSpec(5, 5, 4),
        output_dir=tmp_path,
        write_hdf5=False,
    )

    assert payload["benchmark"] == "owned_geometry_neopax_dataset"
    assert "same geometry" in payload["comparison_policy"]
    assert payload["normalization_contract"]["neopax_export"] == (
        "D11*drds^2, D13*drds, and nu_v*D33."
    )
    assert "booz_xform_psi_p" in payload["normalization_contract"]
    assert payload["inputs"]["booz_xform_psi_p_by_case"]["fake_owned_case"] == 1.0
    assert payload["summary_metrics"]["complete_case_count"] == 1

    case_payload = payload["cases"][0]
    assert case_payload["status"] == "complete"
    assert set(case_payload["scan_paths"]) == {"booz_xform_jax", "vmex_wout_cubic"}
    assert (
        case_payload["scan_paths"]["booz_xform_jax"]["geometry_path"]
        == "vmex.read_wout -> booz_xform_jax -> NTX BoozerSurface"
    )
    responses = case_payload["scan_paths"]["booz_xform_jax"]["profile_responses"]
    assert responses["profile_model"]
    assert len(responses["bootstrap_current_response_normalized"]) == 2
    assert np_is_finite_scalar(responses["current_response_objective"])
    assert case_payload["interpolation_audit"]["reference_path"] == "booz_xform_jax"
    assert case_payload["interpolation_audit"]["candidate_path"] == "vmex_wout_cubic"
    assert max(
        case_payload["interpolation_audit"]["max_relative_difference"].values()
    ) == pytest.approx(0.0)

    output_prefix = tmp_path / "owned_dataset"
    owned_dataset.write_payload(payload, output_prefix)
    owned_dataset.build_figure(payload, output_prefix)
    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()


def test_owned_geometry_uses_physical_psi_p_for_boozer_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    case = owned_dataset.OwnedJaxGeometryCase(
        id="fake_owned_case",
        label="Fake owned case",
        family="test",
        source="unit test",
        input_path=tmp_path / "input.fake",
        wout_path=tmp_path / "wout_fake.nc",
    )
    monkeypatch.setattr(
        owned_dataset,
        "_scalar_from_dataset",
        lambda _path, name: 4.0 * 3.141592653589793 if name == "phi" else None,
    )

    assert owned_dataset._case_psi_p_for_boozer(case) == pytest.approx(2.0)


def np_is_finite_scalar(value: object) -> bool:
    import numpy as np

    return bool(np.isfinite(float(value)))
