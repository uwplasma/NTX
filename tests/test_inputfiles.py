from __future__ import annotations

from ntx.inputfiles import load_run_config


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
