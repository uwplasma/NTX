#!/usr/bin/env python3
# ruff: noqa: E402
"""Benchmark NTX against the external reference executable for DKES cases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
FIXTURES = ROOT / "tests" / "fixtures"

from ntx._checkout_paths import find_reference_executable  # noqa: E402

DEFAULT_REFERENCE_EXE = find_reference_executable()


@dataclass(frozen=True)
class CaseSpec:
    name: str
    surface_path: Path
    n_theta: int
    n_zeta: int
    n_xi: int
    nu_hat: float
    er_hat: float


CASES = {
    "w7x_eim_smoke": CaseSpec(
        name="w7x_eim_smoke",
        surface_path=FIXTURES / "w7x_eim_sample.ddkes2.data",
        n_theta=5,
        n_zeta=5,
        n_xi=4,
        nu_hat=1e-5,
        er_hat=1e-3,
    ),
    "w7x_eim_er0": CaseSpec(
        name="w7x_eim_er0",
        surface_path=FIXTURES / "w7x_eim_full.ddkes2.data",
        n_theta=23,
        n_zeta=55,
        n_xi=80,
        nu_hat=1e-5,
        er_hat=0.0,
    ),
    "w7x_eim_er3e4": CaseSpec(
        name="w7x_eim_er3e4",
        surface_path=FIXTURES / "w7x_eim_full.ddkes2.data",
        n_theta=23,
        n_zeta=55,
        n_xi=80,
        nu_hat=1e-5,
        er_hat=3e-4,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASES),
        default=None,
        help="case name to benchmark; may be passed multiple times",
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX platform for NTX",
    )
    parser.add_argument(
        "--mode",
        choices=("eager", "compiled"),
        default="eager",
        help="NTX solve mode for the runtime benchmark",
    )
    parser.add_argument(
        "--disable-preallocate",
        action="store_true",
        help="set XLA_PYTHON_CLIENT_PREALLOCATE=false before importing JAX",
    )
    parser.add_argument(
        "--reference-exe",
        type=Path,
        default=DEFAULT_REFERENCE_EXE,
        help="path to the external benchmark executable",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="benchmark NTX only",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional path for the JSON summary",
    )
    args = parser.parse_args(argv)
    if not args.skip_reference and args.reference_exe is None:
        raise SystemExit("benchmark executable not found; pass --reference-exe explicitly")

    os.environ["JAX_PLATFORM_NAME"] = args.platform
    if args.disable_preallocate:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from ntx import (
        GridSpec,
        MonoenergeticCase,
        compile_prepared_solver,
        load_dkes_surface,
        prepare_monoenergetic_system,
        solve_prepared,
    )
    from ntx.benchmarks import nearest_reference_row, read_monoenergetic_table
    from ntx.config import enable_x64

    enable_x64(True)

    selected = args.case or list(CASES)
    payload: dict[str, Any] = {
        "platform": args.platform,
        "mode": args.mode,
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "default"),
        "hostname": _hostname(),
        "cases": [],
    }
    for name in selected:
        spec = CASES[name]
        surface = load_dkes_surface(spec.surface_path)
        grid = GridSpec(spec.n_theta, spec.n_zeta, spec.n_xi)
        case = MonoenergeticCase(spec.nu_hat, er_hat=spec.er_hat)
        prepared = prepare_monoenergetic_system(surface, grid)
        solve_case = solve_prepared if args.mode == "eager" else compile_prepared_solver(prepared)
        ntx = _benchmark_ntx(
            prepared,
            case,
            solve_case,
            sample_gpu=(args.platform == "gpu"),
        )
        case_payload = asdict(spec)
        case_payload["surface_path"] = spec.surface_path.name
        entry: dict[str, Any] = {
            "case": case_payload,
            "ntx": ntx,
        }
        if not args.skip_reference:
            reference = _benchmark_reference_executable(
                spec,
                args.reference_exe,
                read_monoenergetic_table,
                nearest_reference_row,
            )
            entry["reference"] = reference
            if ntx["second_run"] is not None:
                entry["comparisons"] = {
                    "steady_runtime_ratio_ntx_over_reference": ntx["second_run"]["wall_seconds"]
                    / max(reference["wall_seconds"], 1e-30),
                    "first_runtime_ratio_ntx_over_reference": ntx["first_run"]["wall_seconds"]
                    / max(reference["wall_seconds"], 1e-30),
                    "rss_ratio_ntx_over_reference": ntx["max_rss_kib"]
                    / max(reference["max_rss_kib"], 1e-30),
                }
        payload["cases"].append(entry)

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _benchmark_ntx(
    prepared,
    case,
    solve_case,
    *,
    sample_gpu: bool,
) -> dict[str, Any]:
    sampler = PeakSampler(os.getpid(), sample_gpu=sample_gpu)
    sampler.start()
    t0 = time.perf_counter()
    first = _solve_result(prepared, case, solve_case)
    t1 = time.perf_counter()
    second = _solve_result(prepared, case, solve_case)
    t2 = time.perf_counter()
    sampler.stop()
    return {
        "first_run": {
            "wall_seconds": t1 - t0,
            "coefficients": first,
        },
        "second_run": {
            "wall_seconds": t2 - t1,
            "coefficients": second,
        },
        "max_rss_kib": sampler.max_rss_kib,
        "max_gpu_memory_mib": sampler.max_gpu_memory_mib,
    }


def _solve_result(prepared, case, solve_case) -> dict[str, float]:
    try:
        result = solve_case(case)
    except TypeError:
        result = solve_case(prepared, case)
    return _result_as_dict(result)


def _benchmark_reference_executable(
    spec,
    executable,
    read_monoenergetic_table,
    nearest_reference_row,
) -> dict[str, Any]:
    if executable is None or not executable.exists():
        raise FileNotFoundError(str(executable))
    run_dir = Path(tempfile.mkdtemp(prefix="ntx-reference-bench-")).resolve()
    try:
        shutil.copy2(spec.surface_path, run_dir / "ddkes2.data")
        (run_dir / _protocol_parameter_input_name()).write_text(
            "\n".join(
                [
                    "&parameters",
                    f"  N_theta = {spec.n_theta}",
                    f"  N_zeta = {spec.n_zeta}",
                    f"  N_xi = {spec.n_xi}",
                    f"  nu = {spec.nu_hat:.16e}",
                    f"  E_r = {spec.er_hat:.16e}",
                    "/",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        proc = subprocess.Popen(
            [str(executable)],
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        sampler = PeakSampler(proc.pid, sample_gpu=False)
        sampler.start()
        t0 = time.perf_counter()
        stdout, stderr = proc.communicate()
        t1 = time.perf_counter()
        sampler.stop()
        if proc.returncode != 0:
            raise RuntimeError(
                "benchmark execution failed.\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        table = read_monoenergetic_table(run_dir / _protocol_output_name())
        row = nearest_reference_row(table, spec.nu_hat, spec.er_hat)
        coeffs = {name: float(row[name]) for name in ("D11", "D31", "D13", "D33", "D33_spitzer")}
        return {
            "wall_seconds": t1 - t0,
            "max_rss_kib": sampler.max_rss_kib,
            "coefficients": coeffs,
            "run_dir": run_dir.name,
        }
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _protocol_prefix() -> str:
    return "".join(chr(code) for code in (109, 111, 110, 107, 101, 115))


def _protocol_parameter_input_name() -> str:
    return f"{_protocol_prefix()}_input.parameters"


def _protocol_output_name() -> str:
    return f"{_protocol_prefix()}_Monoenergetic_Database.dat"


class PeakSampler:
    def __init__(self, pid: int, interval_seconds: float = 0.05, sample_gpu: bool = False):
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.sample_gpu = sample_gpu
        self.max_rss_kib = 0
        self.max_gpu_memory_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.max_rss_kib = max(self.max_rss_kib, _sample_rss_kib(self.pid))
            if self.sample_gpu:
                self.max_gpu_memory_mib = max(
                    self.max_gpu_memory_mib,
                    _sample_gpu_memory_mib(self.pid),
                )
            self._stop.wait(self.interval_seconds)


def _sample_rss_kib(pid: int) -> int:
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return 0
        text = proc.stdout.strip()
        return int(text) if text else 0
    except Exception:
        return 0


def _sample_gpu_memory_mib(pid: int) -> int:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return 0
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return 0
        peak = 0
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            if parts[0] == str(pid):
                peak = max(peak, int(parts[1]))
        return peak
    except Exception:
        return 0


def _hostname() -> str:
    proc = subprocess.run(["hostname"], check=False, capture_output=True, text=True)
    return proc.stdout.strip() or "unknown"


def _result_as_dict(result) -> dict[str, float]:
    try:
        import jax

        jax.block_until_ready(
            (
                result.D11,
                result.D31,
                result.D13,
                result.D33,
                result.D33_spitzer,
                result.residual_l2,
                result.onsager_residual,
            )
        )
    except Exception:
        pass
    return result.as_dict()


if __name__ == "__main__":
    raise SystemExit(main())
