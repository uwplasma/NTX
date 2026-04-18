from __future__ import annotations

import json

import numpy as np

from examples import bootstrap_current_with_neopax as example


def test_plot_and_summary_write_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(example, "OUTPUT_PREFIX", tmp_path / "bootstrap_current_with_neopax")
    data = {
        "rho": np.linspace(0.1, 0.9, 5),
        "ne": np.linspace(4.0e20, 1.0e20, 5),
        "te": np.linspace(8.0e3, 1.0e3, 5),
        "b0": np.full(5, 2.5),
        "iota": np.linspace(-0.9, -0.7, 5),
        "current_nomom": np.linspace(-3.0e6, -1.0e6, 5),
        "current_total": np.linspace(-3.2e6, -1.2e6, 5),
        "current_correction": np.linspace(-2.0e5, -2.0e5, 5),
        "d33_electron": np.linspace(0.2, 0.6, 5),
        "use_momentum_correction": True,
        "nu_v": np.asarray([1.0e-4, 1.0e-3]),
        "er_axis": np.zeros((5, 2)),
        "grid": np.array([17, 25, 32], dtype=int),
    }

    example.plot_profiles(data)
    example.write_summary(data)

    png = tmp_path / "bootstrap_current_with_neopax.png"
    pdf = tmp_path / "bootstrap_current_with_neopax.pdf"
    payload = json.loads((tmp_path / "bootstrap_current_with_neopax.json").read_text())

    assert png.exists()
    assert pdf.exists()
    assert payload["use_momentum_correction"] is True
    assert payload["grid"] == {"n_theta": 17, "n_zeta": 25, "n_xi": 32}
    assert len(payload["current_total_am2"]) == 5
