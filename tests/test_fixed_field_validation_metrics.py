from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _fixed_field_validation_metrics import (  # noqa: E402
    jsonify,
    least_squares_scale,
    relative_error_array,
    sign_mismatch_count,
)
from _fixed_field_validation_plotting import (  # noqa: E402
    display_label,
    plot_fixed_field_validation,
    plot_order,
    plot_styles,
)
from _fixed_field_validation_summary import build_fixed_field_summary_payload  # noqa: E402


def test_fixed_field_validation_metric_helpers_are_masked_and_scaled() -> None:
    reference = np.asarray([2.0, 4.0, -6.0, 8.0])
    model = np.asarray([1.0, 2.0, 3.0, -4.0])
    mask = np.asarray([True, True, False, False])

    assert least_squares_scale(reference, model, mask) == pytest.approx(2.0)
    assert math.isnan(least_squares_scale(reference, np.zeros_like(model), mask))

    relative = relative_error_array(reference, model)
    assert relative.tolist() == pytest.approx([0.5, 0.5, 1.5, 1.5])
    assert relative_error_array(np.asarray([0.0]), np.asarray([0.25])).item() == (
        pytest.approx(0.25)
    )

    assert sign_mismatch_count(reference, model, np.ones_like(mask, dtype=bool)) == 2
    assert sign_mismatch_count(
        np.asarray([0.0, 1.0, -1.0]),
        np.asarray([-1.0, -1.0, -2.0]),
        np.asarray([True, True, True]),
    ) == 1


def test_fixed_field_jsonify_converts_artifact_payload_types() -> None:
    payload = {
        "path": ROOT / "docs",
        "array": np.asarray([1.0, 2.0]),
        "scalar": np.float64(3.0),
        "nested": (np.int64(4),),
    }

    assert jsonify(payload) == {
        "path": str(ROOT / "docs"),
        "array": [1.0, 2.0],
        "scalar": 3.0,
        "nested": [4],
    }


def test_fixed_field_plotting_helper_writes_publication_artifacts(tmp_path: Path) -> None:
    rho = np.linspace(0.2, 0.9, 6)
    reference = 1.0e6 * (1.0 + rho)

    def profile(scale: float) -> dict[str, np.ndarray]:
        return {"rho": rho, "jdotb": reference * scale}

    qa_results = {
        "SFINCS": profile(1.0),
        "SFINCS-JAX": {
            "rho": rho,
            "jdotb": reference * 1.01,
            "rho_sample": rho[::2],
            "jdotb_sample": reference[::2] * 1.01,
        },
        "NTX+NEOPAX": profile(0.9),
        "Redl": profile(1.05),
    }
    qh_results = {
        "SFINCS": profile(1.0),
        "NTX+NEOPAX": profile(1.1),
        "Redl": profile(0.95),
    }
    output_prefix = tmp_path / "fixed_field"

    assert display_label("Redl") == "Redl (Boozer)"
    assert plot_order(qa_results) == ("SFINCS", "SFINCS-JAX", "NTX+NEOPAX", "Redl")
    assert plot_order(qh_results) == ("SFINCS", "NTX+NEOPAX", "Redl")
    assert "NTX+NEOPAX" in plot_styles()

    plot_fixed_field_validation(
        results={"qa": qa_results, "qh": qh_results},
        cases={
            "qa": SimpleNamespace(label="QA synthetic"),
            "qh": SimpleNamespace(label="QH synthetic"),
        },
        output_prefix=output_prefix,
        interior_rho_min=0.25,
        interior_rho_max=0.85,
        interp_profile=lambda x, y, xq: np.interp(xq, x, y),
    )

    assert output_prefix.with_suffix(".png").stat().st_size > 0
    assert output_prefix.with_suffix(".pdf").stat().st_size > 0


def test_fixed_field_summary_helper_builds_traceable_error_payload(tmp_path: Path) -> None:
    rho = np.asarray([0.2, 0.5, 0.9])
    reference = np.asarray([10.0, 20.0, 40.0])
    results = {
        "qa": {
            "SFINCS": {"rho": rho, "jdotb": reference},
            "NTX+NEOPAX": {"rho": rho, "jdotb": reference * np.asarray([1.2, 0.9, 1.1])},
            "Redl": {"rho": rho, "jdotb": reference * np.asarray([1.01, 1.02, 0.99])},
        }
    }
    cases = {"qa": SimpleNamespace(label="QA synthetic")}

    payload = build_fixed_field_summary_payload(
        results=results,
        cases=cases,
        output_prefix=tmp_path / "fixed_field",
        interior_rho_min=0.25,
        interior_rho_max=0.85,
        closure_diagnostics=lambda case, case_results: {
            "label": case.label,
            "model_count": len(case_results),
        },
    )

    assert payload["figure_png"] == str((tmp_path / "fixed_field").with_suffix(".png"))
    assert payload["figure_pdf"] == str((tmp_path / "fixed_field").with_suffix(".pdf"))
    assert payload["cases"]["qa"]["NTX+NEOPAX"]["jdotb"] == pytest.approx(
        [12.0, 18.0, 44.0]
    )
    assert payload["cases"]["qa"]["max_relative_error_vs_sfincs"]["NTX+NEOPAX"] == (
        pytest.approx(0.2)
    )
    assert payload["cases"]["qa"]["max_relative_error_vs_sfincs_interior"]["NTX+NEOPAX"] == (
        pytest.approx(0.1)
    )
    assert payload["cases"]["qa"]["max_relative_error_vs_sfincs_interior"]["Redl"] == (
        pytest.approx(0.02)
    )
    assert payload["cases"]["qa"]["closure_diagnostics"] == {
        "label": "QA synthetic",
        "model_count": 3,
    }
