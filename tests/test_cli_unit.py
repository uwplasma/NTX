from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ntx import cli

ROOT = Path(__file__).resolve().parents[1]
DKES = ROOT / "tests" / "fixtures" / "sample_surface.ddkes2.data"
VMEC = ROOT / "tests" / "fixtures" / "sample_wout.nc"


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
    assert cli._looks_like_input_file([str(path), "--plot"])
    assert not cli._looks_like_input_file([str(path.with_suffix(".txt"))])


def test_cli_parse_input_file_args(tmp_path):
    path = tmp_path / "run.toml"
    output = tmp_path / "run.nc"
    args = cli._parse_input_file_args(
        [str(path), "--output", str(output), "--plot", "--plot-output", str(tmp_path / "plot.pdf")]
    )
    assert args.input == path
    assert args.output == output
    assert args.plot


def test_cli_load_surface_errors_without_vmec_psi_n():
    args = SimpleNamespace(example=False, dkes=None, vmec=VMEC, psi_n=None)
    with pytest.raises(ValueError, match="--psi-n"):
        cli._load_surface(args)


def test_cli_load_surface_errors_without_selection():
    args = SimpleNamespace(example=False, dkes=None, vmec=None)
    with pytest.raises(ValueError, match="select one"):
        cli._load_surface(args)


def test_cli_load_surface_vmec_success(monkeypatch):
    fake_surface = object()
    called = {}

    def fake_load_vmec_surface(path, **kwargs):
        called["path"] = path
        called["kwargs"] = kwargs
        return fake_surface

    monkeypatch.setattr(cli, "load_vmec_surface", fake_load_vmec_surface)
    args = SimpleNamespace(
        example=False,
        dkes=None,
        vmec=VMEC,
        psi_n=0.25,
        vmec_radial_option=1,
        vmec_nyquist_option=2,
        vmec_mode_convention="reduced",
        min_bmn_to_load=1.0e-4,
    )
    assert cli._load_surface(args) is fake_surface
    assert called["path"] == VMEC
    assert called["kwargs"]["psi_n"] == 0.25


def test_cli_main_unknown_command_returns_one(monkeypatch):
    class DummyParser:
        def add_subparsers(self, **kwargs):
            class DummySubparsers:
                def add_parser(self, *args, **kwargs):
                    class DummyParserLeaf:
                        def add_mutually_exclusive_group(self, **kwargs):
                            class DummyGroup:
                                def add_argument(self, *args, **kwargs):
                                    return None

                            return DummyGroup()

                        def add_argument(self, *args, **kwargs):
                            return None

                    return DummyParserLeaf()

            return DummySubparsers()

        def parse_args(self, args_list):
            return SimpleNamespace(command="unexpected")

    monkeypatch.setattr(cli.argparse, "ArgumentParser", lambda prog=None: DummyParser())
    assert cli.main(["ignored"]) == 1


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


def test_python_m_ntx_cli_module_entrypoint(monkeypatch, capsys):
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
        runpy.run_module("ntx.cli", run_name="__main__")
    assert excinfo.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["D33"] > 0.0
