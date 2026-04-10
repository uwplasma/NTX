from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ntx import cli

ROOT = Path(__file__).resolve().parents[1]
DKES = ROOT / "tests" / "fixtures" / "w7x_eim_sample.ddkes2.data"
VMEC = ROOT / "tests" / "fixtures" / "wout_w7x_standardConfig.nc"


def test_cli_main_example_branch(capsys):
    argv = [
        "solve",
        "--example",
        "--nu-hat",
        "1e-2",
        "--n-theta",
        "5",
        "--n-zeta",
        "5",
        "--n-xi",
        "4",
    ]
    assert cli.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["D11"] > 0.0


def test_cli_main_input_file_branch(tmp_path, capsys):
    input_path = tmp_path / "run.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "dkes"',
                f'path = "{DKES}"',
                "",
                "[grid]",
                "n_theta = 5",
                "n_zeta = 5",
                "n_xi = 4",
                "",
                "[case]",
                "nu_hat = 1e-5",
                "er_hat = 1e-3",
                "",
                "[output]",
                'npz = "result.npz"',
                "",
                "[logging]",
                "verbose = false",
            ]
        ),
        encoding="utf-8",
    )
    assert cli.main([str(input_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["D33"] > 0.0


def test_cli_main_inspect_surface_branch(capsys):
    assert cli.main(["inspect-surface", "--dkes", str(DKES)]) == 0
    assert "BoozerSurface" in capsys.readouterr().out


def test_cli_looks_like_input_file(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(
        "[surface]\ntype='example'\n[grid]\nn_theta=3\nn_zeta=3\nn_xi=2\n[case]\nnu_hat=1e-2\n",
        encoding="utf-8",
    )
    assert cli._looks_like_input_file([str(path)])
    assert not cli._looks_like_input_file([])
    assert not cli._looks_like_input_file([str(path), "extra"])
    assert not cli._looks_like_input_file([str(path.with_suffix(".txt"))])


def test_cli_load_surface_errors_without_vmec_psi_n():
    args = SimpleNamespace(example=False, dkes=None, vmec=VMEC, psi_n=None)
    with pytest.raises(ValueError, match="--psi-n"):
        cli._load_surface(args)


def test_cli_load_surface_errors_without_selection():
    args = SimpleNamespace(example=False, dkes=None, vmec=None)
    with pytest.raises(ValueError, match="select one"):
        cli._load_surface(args)


def test_python_m_ntx_module_entrypoint(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python",
            "solve",
            "--example",
            "--nu-hat",
            "1e-2",
            "--n-theta",
            "5",
            "--n-zeta",
            "5",
            "--n-xi",
            "4",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("ntx.__main__", run_name="__main__")
    assert excinfo.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["D11"] > 0.0
