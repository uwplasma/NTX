"""Variable-coefficient angular-oversampling audit gates."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ntx import MonoenergeticCase, load_vmec_surface
from ntx.validation import audit_angular_oversampling

from .fixture_data import SAMPLE_WOUT

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "angular_oversampling_audit.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("ntx_angular_oversampling_audit", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_angular_oversampling_audit_profiles_error_and_costs():
    surface = load_vmec_surface(
        SAMPLE_WOUT,
        psi_n=0.25,
        min_bmn_to_load=1.0e-4,
    )
    audit = audit_angular_oversampling(
        surface,
        MonoenergeticCase(1.0e-2, er_hat=0.0),
        ratios=(1.0, 1.25, 1.5),
        n_xi=4,
        recommended_oversampling=1.25,
        repeats=1,
    )

    assert len(audit.points) == 3
    assert audit.points[-1].max_relative_error == 0.0
    assert audit.recommended_point.requested_ratio == 1.25
    assert all(point.theta_oversampling >= point.requested_ratio for point in audit.points)
    assert all(point.zeta_oversampling >= point.requested_ratio for point in audit.points)
    assert all(point.schur_residual_l2 < 1.0e-10 for point in audit.points)
    assert all(point.prepare_seconds >= 0.0 for point in audit.points)
    assert all(point.compilation_seconds >= 0.0 for point in audit.points)
    assert all(point.warm_execution_seconds >= 0.0 for point in audit.points)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"ratios": (1.0,)}, "at least two"),
        ({"ratios": (1.0, 0.9)}, "no smaller"),
        ({"ratios": (1.0, 1.0, 2.0)}, "strictly increasing"),
        (
            {"ratios": (1.0, 1.5), "recommended_oversampling": 1.5},
            "reference ratio",
        ),
        ({"coefficient_atol": 0.0}, "positive"),
        ({"repeats": 0}, "positive"),
    ],
)
def test_angular_oversampling_audit_rejects_invalid_policy(kwargs, message):
    surface = load_vmec_surface(
        SAMPLE_WOUT,
        psi_n=0.25,
        min_bmn_to_load=1.0e-4,
    )
    options = {
        "ratios": (1.0, 1.5, 2.0),
        "n_xi": 4,
        "recommended_oversampling": 1.5,
        "repeats": 1,
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        audit_angular_oversampling(
            surface,
            MonoenergeticCase(1.0e-2),
            **options,
        )


def test_angular_oversampling_example_writes_artifacts(tmp_path):
    module = _load_example_module()
    case = module.AuditCase(
        id="fixture",
        label="Fixture",
        family="fixture",
        source="NTX tests",
        path=SAMPLE_WOUT,
    )
    payload = module.run_audits(
        (case,),
        ratios=(1.0, 1.25, 1.5),
        n_xi=4,
        recommended_oversampling=1.25,
        repeats=1,
    )
    output_prefix = tmp_path / "angular_oversampling_audit"
    module.plot_payload(payload, output_prefix)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    assert payload["schema_version"] == 1
    assert payload["summary_metrics"]["case_count"] == 1
    assert payload["cases"][0]["points"][-1]["max_relative_error"] == 0.0
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    assert output_prefix.with_suffix(".json").exists()
