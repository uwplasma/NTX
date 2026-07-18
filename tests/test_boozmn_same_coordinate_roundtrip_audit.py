from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from ntx import GridSpec, example_surface
from ntx.booz import BoozmnSurface

ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    path = ROOT / "examples" / "boozmn_same_coordinate_roundtrip_audit.py"
    spec = importlib.util.spec_from_file_location("boozmn_same_coordinate_roundtrip_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boozmn_same_coordinate_roundtrip_audit_writes_artifacts(monkeypatch, tmp_path: Path):
    module = _load_audit_module()
    input_path = tmp_path / "input.vmec"
    wout_path = tmp_path / "wout.nc"
    input_path.write_text("&INDATA\n/\n", encoding="utf-8")
    wout_path.write_text("placeholder", encoding="utf-8")

    def fake_write_boozmn_from_wout(**_kwargs):
        boozmn = tmp_path / "boozmn.nc"
        boozmn.write_text("placeholder", encoding="utf-8")
        return boozmn, np.asarray([0.125, 0.375])

    def fake_load_boozmn(path, *, s=None, surface_index=None, **_kwargs):
        s_value = float(s if s is not None else [0.125, 0.375][surface_index])
        if surface_index is not None:
            index = int(surface_index)
        else:
            half_grid = np.asarray([0.125, 0.375])
            index = int(np.argmin(abs(half_grid - s_value)))
        surface = example_surface()
        return BoozmnSurface(
            surface=surface,
            path=Path(path),
            s=s_value,
            rho=float(np.sqrt(s_value)),
            surface_index=index,
            mode_count=len(surface.m),
        )

    monkeypatch.setattr(module, "_write_boozmn_from_wout", fake_write_boozmn_from_wout)
    monkeypatch.setattr(module, "load_boozmn_surface", fake_load_boozmn)
    monkeypatch.setattr(module, "surface_from_vmex_wout", lambda **_kwargs: example_surface())

    payload = module.build_roundtrip_audit(
        input_path=input_path,
        wout_path=wout_path,
        surface_indices=(0, 1),
        mboz=4,
        nboz=4,
        psi_p=1.0,
        profile_source="auto",
        nu_hat=1.0e-2,
        epsi_hat=0.0,
        grid=GridSpec(5, 5, 4),
        output_dir=tmp_path,
    )

    assert payload["benchmark"] == "boozmn_same_coordinate_roundtrip_audit"
    assert payload["inputs"]["profile_source"] == "auto"
    assert payload["summary_metrics"]["roundtrip_closed"] is True
    assert payload["summary_metrics"]["max_transport_relative_difference"] == 0.0
    assert [surface["s"] for surface in payload["surfaces"]] == [0.125, 0.375]

    output_prefix = tmp_path / "boozmn_same_coordinate_roundtrip_audit"
    json_path = module.write_payload(payload, output_prefix)
    figure_paths = module.build_figure(payload, output_prefix)

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["summary_metrics"] == payload["summary_metrics"]
    assert all(path.exists() for path in figure_paths)
