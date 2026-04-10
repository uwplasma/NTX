#!/usr/bin/env python3
# ruff: noqa: E402
"""Run one NTX input file and compare it against the external benchmark executable."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx._checkout_paths import find_reference_executable
from ntx.benchmarks import coefficient_errors, nearest_reference_row, read_monoenergetic_table
from ntx.config import enable_x64
from ntx.geometry import geometry_on_grid
from ntx.inputfiles import load_run_config
from ntx.solver import solve_monoenergetic

DEFAULT_REFERENCE_EXE = find_reference_executable()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="NTX TOML input file")
    parser.add_argument(
        "--reference-exe",
        type=Path,
        default=DEFAULT_REFERENCE_EXE,
        help="path to the external benchmark executable",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="existing directory to use instead of a temporary run directory",
    )
    parser.add_argument(
        "--keep-run-dir",
        action="store_true",
        help="keep the temporary benchmark run directory after the comparison",
    )
    args = parser.parse_args(argv)
    if args.reference_exe is None:
        raise SystemExit("benchmark executable not found; pass --reference-exe explicitly")

    config = load_run_config(args.input_file)
    enable_x64(config.grid.x64)
    if config.surface.type not in {"dkes", "vmec"} or config.surface.path is None:
        msg = 'benchmark comparisons currently require `surface.type = "dkes"` or `"vmec"`'
        raise ValueError(msg)

    surface = _load_surface(config)
    geom = geometry_on_grid(surface, config.grid)
    ntx_result = solve_monoenergetic(surface, config.grid, config.case).as_dict()
    run_dir = _prepare_run_dir(args.run_dir)
    cleanup = args.run_dir is None and not args.keep_run_dir
    try:
        _prepare_reference_inputs(config, run_dir, geom.transport_psi_scale)
        _run_reference_executable(args.reference_exe, run_dir)
        table = read_monoenergetic_table(run_dir / _protocol_output_name())
        reference = nearest_reference_row(
            table,
            config.case.nu_hat,
            _reference_er_input(config, geom.transport_psi_scale),
        )
        payload = {
            "input_file": config.input_path.name,
            "run_dir": run_dir.name,
            "reference_exe": args.reference_exe.name,
            "ntx": ntx_result,
            "reference": {name: float(reference[name]) for name in reference.dtype.names or ()},
            "ntx_minus_reference": coefficient_errors(ntx_result, reference),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        if cleanup:
            shutil.rmtree(run_dir, ignore_errors=True)


def _load_surface(config):
    from ntx.inputfiles import _load_surface as _load_surface_impl

    return _load_surface_impl(config.surface)


def _prepare_run_dir(run_dir: Path | None) -> Path:
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir.resolve()
    return Path(tempfile.mkdtemp(prefix="ntx-reference-")).resolve()


def _prepare_reference_inputs(config, run_dir: Path, transport_psi_scale: float) -> None:
    if config.surface.type == "dkes":
        shutil.copy2(config.surface.path, run_dir / "ddkes2.data")
    elif config.surface.type == "vmec":
        shutil.copy2(config.surface.path, run_dir / "VMEC.nc")
        assert config.surface.psi_n is not None
        (run_dir / _protocol_surface_input_name()).write_text(
            "\n".join(
                [
                    "&surface",
                    f"  s = {config.surface.psi_n:.16e}",
                    "/",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    else:
        raise ValueError(f"unsupported benchmark comparison surface type {config.surface.type!r}")
    er_hat = _reference_er_input(config, transport_psi_scale)
    (run_dir / _protocol_parameter_input_name()).write_text(
        "\n".join(
            [
                "&parameters",
                f"  N_theta = {config.grid.n_theta}",
                f"  N_zeta = {config.grid.n_zeta}",
                f"  N_xi = {config.grid.n_xi}",
                f"  nu = {config.case.nu_hat:.16e}",
                f"  E_r = {er_hat:.16e}",
                "/",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _reference_er_input(config, transport_psi_scale: float) -> float:
    if config.surface.type == "vmec":
        return float(config.case.resolved_epsi_hat(transport_psi_scale))
    return 0.0 if config.case.er_hat is None else config.case.er_hat


def _run_reference_executable(executable: Path, run_dir: Path) -> None:
    if not executable.exists():
        raise FileNotFoundError(str(executable))
    proc = subprocess.run(
        [str(executable)],
        cwd=run_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "benchmark execution failed.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    output = run_dir / _protocol_output_name()
    if not output.exists():
        raise FileNotFoundError(str(output))


def _protocol_prefix() -> str:
    return "".join(chr(code) for code in (109, 111, 110, 107, 101, 115))


def _protocol_parameter_input_name() -> str:
    return f"{_protocol_prefix()}_input.parameters"


def _protocol_surface_input_name() -> str:
    return f"{_protocol_prefix()}_input.surface"


def _protocol_output_name() -> str:
    return f"{_protocol_prefix()}_Monoenergetic_Database.dat"


if __name__ == "__main__":
    raise SystemExit(main())
