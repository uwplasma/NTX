from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ntx import GridSpec

from .fixture_data import SAMPLE_BOOZMN, SAMPLE_WOUT

ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    path = ROOT / "examples" / "boozmn_backend_validation_audit.py"
    spec = importlib.util.spec_from_file_location("boozmn_backend_validation_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boozmn_backend_validation_audit_writes_artifacts(tmp_path: Path):
    module = _load_audit_module()
    payload = module.build_audit(
        wout_path=SAMPLE_WOUT,
        boozmn_path=SAMPLE_BOOZMN,
        rho=0.5,
        nu_hat=1.0e-2,
        epsi_hat=0.0,
        grid=GridSpec(5, 5, 4),
    )

    assert payload["benchmark"] == "boozmn_backend_validation_audit"
    assert set(payload["cases"]) == {
        "vmec_harmonic",
        "direct_boozer_unit_flux",
        "direct_boozer_vmec_edge_flux",
    }
    assert payload["summary_metrics"]["best_max_transport_relative_difference"] >= 0.0
    assert payload["summary_metrics"]["best_radial_drift_relative_l2"] >= 0.0
    assert isinstance(payload["summary_metrics"]["direct_boozer_backend_closed"], bool)
    assert "radial_drift_spatial" in payload["cases"]["vmec_harmonic"]["geometry"]
    assert "operator_k1" in payload["cases"]["direct_boozer_vmec_edge_flux"]

    output_prefix = tmp_path / "boozmn_backend_validation_audit"
    json_path = module.write_payload(payload, output_prefix)
    figure_paths = module.build_figure(payload, output_prefix)

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert "arrays" not in written["cases"]["vmec_harmonic"]
    assert written["summary_metrics"] == payload["summary_metrics"]
    assert all(path.exists() for path in figure_paths)
