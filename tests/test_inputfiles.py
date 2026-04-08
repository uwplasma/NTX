from __future__ import annotations

from pathlib import Path

import numpy as np

from ntx.inputfiles import load_run_config, run_from_input_file

VMEC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wout_w7x_standardConfig.nc"


def test_load_run_config_vmec_parses_surface_options(tmp_path):
    input_path = tmp_path / "vmec.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "vmec"',
                'path = "wout_example.nc"',
                "psi_n = 0.4",
                "vmec_radial_option = 1",
                "vmec_nyquist_option = 2",
                "min_bmn_to_load = 1e-4",
                "",
                "[grid]",
                "n_theta = 9",
                "n_zeta = 11",
                "n_xi = 6",
                "",
                "[case]",
                "nu_hat = 1e-4",
                "epsi_hat = 2e-3",
            ]
        ),
        encoding="utf-8",
    )
    config = load_run_config(input_path)
    assert config.surface.type == "vmec"
    assert config.surface.path == (tmp_path / "wout_example.nc").resolve()
    assert config.surface.psi_n == 0.4
    assert config.surface.vmec_radial_option == 1
    assert config.surface.vmec_nyquist_option == 2
    assert config.surface.min_bmn_to_load == 1e-4
    assert config.case.epsi_hat == 2e-3


def test_load_run_config_requires_vmec_psi_n(tmp_path):
    input_path = tmp_path / "vmec.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "vmec"',
                'path = "wout_example.nc"',
                "",
                "[grid]",
                "n_theta = 9",
                "n_zeta = 11",
                "n_xi = 6",
                "",
                "[case]",
                "nu_hat = 1e-4",
                "epsi_hat = 2e-3",
            ]
        ),
        encoding="utf-8",
    )
    try:
        load_run_config(input_path)
    except ValueError as exc:
        assert "surface.psi_n" in str(exc)
    else:
        raise AssertionError("expected VMEC configuration without psi_n to fail")


def test_load_run_config_accepts_vmec_er_hat(tmp_path):
    input_path = tmp_path / "vmec.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "vmec"',
                'path = "wout_example.nc"',
                "psi_n = 0.4",
                "",
                "[grid]",
                "n_theta = 9",
                "n_zeta = 11",
                "n_xi = 6",
                "",
                "[case]",
                "nu_hat = 1e-4",
                "er_hat = 2e-3",
            ]
        ),
        encoding="utf-8",
    )
    config = load_run_config(input_path)
    assert config.case.er_hat == 2e-3
    assert config.case.epsi_hat is None


def test_run_from_input_file_vmec_writes_metadata_npz(tmp_path):
    input_path = tmp_path / "vmec.toml"
    output_path = tmp_path / "vmec.npz"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "vmec"',
                f'path = "{VMEC_FIXTURE}"',
                "psi_n = 0.25",
                "",
                "[grid]",
                "n_theta = 7",
                "n_zeta = 9",
                "n_xi = 4",
                "",
                "[case]",
                "nu_hat = 1e-3",
                "epsi_hat = 1e-3",
                "",
                "[output]",
                f'npz = "{output_path}"',
                "",
                "[logging]",
                "verbose = false",
            ]
        ),
        encoding="utf-8",
    )
    run_from_input_file(input_path)
    with np.load(output_path) as data:
        assert "vmec_ns" in data
        assert "vmec_total_mode_count" in data
        assert "surface_metadata_json" in data
        assert "geometry_metadata_json" in data
        assert "algorithm_metadata_json" in data
        assert np.isnan(data["surface_psi_p"])
        assert "vmec_r_n" in data
        assert "vmec_r_hat" in data
        assert "vmec_dpsi_hat_dr_hat" in data
        assert "vmec_dr_hat_dpsi_hat" in data
