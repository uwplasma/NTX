#!/usr/bin/env python3
"""Build a NEOPAX-style monoenergetic scan from an Er_tilde grid.

This mirrors a DKES-like database workflow: the user chooses rho, nu_v, and
Er_tilde, provides VMEC + Boozer files, and NTX computes coefficient tables and
conversion metadata from scratch before writing a NEOPAX-style HDF5 file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ntx._interp import Interpolator1D
import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import (  # noqa: E402
    GridSpec,
    NeopaxScan,
    load_boozmn_surface,
    solve_monoenergetic_parallel_scan,
    solve_monoenergetic_scan,
    surface_from_vmex_vmec_wout_file,
    write_neopax_scan_hdf5,
)
from ntx._checkout_paths import find_neopax_root  # noqa: E402

NEOPAX_ROOT = find_neopax_root()

# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
WOUT_PATH = (
    NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
    if NEOPAX_ROOT is not None
    else Path("/missing/wout_W7-X_standard_configuration.nc")
)
BOOZMN_PATH = (
    NEOPAX_ROOT / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
    if NEOPAX_ROOT is not None
    else Path("/missing/boozmn_wout_W7-X_standard_configuration.nc")
)
OUTPUT_PATH = (
    ROOT
    / "examples"
    / "outputs"
    / "neopax_scan_from_ertilde"
    / "ntx_scan_from_ertilde.h5"
)

DEFAULT_RHO = (0.12247, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
DEFAULT_NU_V = (
    3.0e-7,
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0e0,
    3.0e0,
    1.0e1,
)
DEFAULT_ER_TILDE = (
    0.0,
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
)
GRID = GridSpec(n_theta=25, n_zeta=25, n_xi=64)
ONSAGER_WARN_THRESHOLD = 1.0e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a NEOPAX-style NTX monoenergetic scan from an Er_tilde grid."
    )
    parser.add_argument(
        "--wout",
        type=Path,
        default=WOUT_PATH,
        help="Path to the VMEC wout file.",
    )
    parser.add_argument(
        "--booz",
        type=Path,
        default=BOOZMN_PATH,
        help="Path to the Boozer boozmn/boozermn file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to the output NEOPAX-style HDF5 file.",
    )
    parser.add_argument(
        "--surface-backend",
        choices=("auto", "vmec", "boozmn"),
        default="vmec",
        help=(
            "Geometry source used for the NTX surface solve. The VMEC backend "
            "is the validated default; boozmn is an explicit backend audit mode."
        ),
    )
    parser.add_argument(
        "--device-backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="JAX execution backend used for the solve.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Device index within the selected JAX backend.",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=GRID.n_theta,
        help="Poloidal grid resolution.",
    )
    parser.add_argument(
        "--n-zeta",
        type=int,
        default=GRID.n_zeta,
        help="Toroidal grid resolution.",
    )
    parser.add_argument(
        "--n-xi",
        type=int,
        default=GRID.n_xi,
        help="Pitch-angle / Legendre resolution.",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=None,
        help=(
            "Optional fixed batch size for the flattened (nu_v, Er_tilde) "
            "scan on each surface. Leave unset for full-surface batching; "
            "use smaller values to reduce CPU/GPU peak memory."
        ),
    )
    parser.add_argument(
        "--parallel-devices",
        type=int,
        default=None,
        help=(
            "Optional number of local JAX devices used to shard each surface "
            "scan. For CPU runs, expose host devices before launch, for example "
            "XLA_FLAGS=--xla_force_host_platform_device_count=4."
        ),
    )
    parser.add_argument(
        "--rho",
        type=str,
        default=None,
        help="Comma-separated rho grid. Default is the W7-X reference grid.",
    )
    parser.add_argument(
        "--rho-linspace",
        type=str,
        default=None,
        help=(
            "Convenience alternative to --rho: specify 'start,stop,count' to "
            "generate a uniform rho grid. Example: --rho-linspace 0.1,0.9,15"
        ),
    )
    parser.add_argument(
        "--nu-v",
        type=str,
        default=None,
        help="Comma-separated collisionality grid. Values must be positive.",
    )
    parser.add_argument(
        "--er-tilde",
        type=str,
        default=None,
        help="Comma-separated Er_tilde grid.",
    )
    parser.add_argument(
        "--er-tilde-logspace",
        type=str,
        default=None,
        help=(
            "Convenience alternative to --er-tilde: specify "
            "'start,stop,count' to generate a geometric Er_tilde grid over "
            "positive values. Example: --er-tilde-logspace 1e-6,1e-1,16"
        ),
    )
    parser.add_argument(
        "--er-tilde-include-zero",
        action="store_true",
        help=(
            "When used with --er-tilde-logspace, prepend 0.0 to the generated "
            "positive Er_tilde grid."
        ),
    )
    parser.add_argument(
        "--onsager-warn-threshold",
        type=float,
        default=ONSAGER_WARN_THRESHOLD,
        help="Warn when max |D13 + D31| exceeds this threshold.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Optional source_name attribute written to the HDF5 file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot D11, D13, D31, and D33 vs nu_v after the scan.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Plot output path or prefix. If omitted, uses the HDF5 output path stem.",
    )
    parser.add_argument(
        "--plot-rho-index",
        type=int,
        default=None,
        help="Optional rho index to plot. If omitted, plots all rho surfaces.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-surface progress and timing output.",
    )
    return parser.parse_args()


def _parse_float_grid(
    text: str | None,
    *,
    default: Sequence[float],
    name: str,
    positive: bool = False,
) -> jnp.ndarray:
    if text is None:
        values = tuple(default)
    else:
        try:
            values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of floats") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    array = jnp.asarray(values, dtype=jnp.float64)
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} contains non-finite values")
    if positive and not bool(jnp.all(array > 0.0)):
        raise ValueError(f"{name} values must be positive")
    return array


def _parse_rho_linspace(text: str | None) -> jnp.ndarray | None:
    if text is None:
        return None
    parts = [piece.strip() for piece in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError("rho-linspace must be in 'start,stop,count' format")
    try:
        start = float(parts[0])
        stop = float(parts[1])
        count = int(parts[2])
    except ValueError as exc:
        raise ValueError("rho-linspace must be in 'start,stop,count' format") from exc
    if count < 2:
        raise ValueError("rho-linspace count must be at least 2")
    grid = jnp.linspace(start, stop, count, dtype=jnp.float64)
    if not bool(jnp.all(jnp.isfinite(grid))):
        raise ValueError("rho-linspace produced non-finite values")
    return grid


def _resolve_rho_grid(args: argparse.Namespace) -> jnp.ndarray:
    if args.rho is not None and args.rho_linspace is not None:
        raise ValueError("set only one of --rho or --rho-linspace")
    if args.rho is not None:
        return _parse_float_grid(args.rho, default=DEFAULT_RHO, name="rho")
    rho_linspace = _parse_rho_linspace(args.rho_linspace)
    if rho_linspace is not None:
        return rho_linspace
    return _parse_float_grid(None, default=DEFAULT_RHO, name="rho")


def _parse_er_tilde_logspace(text: str | None, *, include_zero: bool) -> jnp.ndarray | None:
    if text is None:
        return None
    parts = [piece.strip() for piece in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError("er-tilde-logspace must be in 'start,stop,count' format")
    try:
        start = float(parts[0])
        stop = float(parts[1])
        count = int(parts[2])
    except ValueError as exc:
        raise ValueError("er-tilde-logspace must be in 'start,stop,count' format") from exc
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("er-tilde-logspace start and stop must be positive")
    if count < 2:
        raise ValueError("er-tilde-logspace count must be at least 2")
    grid = jnp.geomspace(start, stop, count, dtype=jnp.float64)
    if include_zero:
        grid = jnp.concatenate((jnp.asarray([0.0], dtype=jnp.float64), grid))
    if not bool(jnp.all(jnp.isfinite(grid))):
        raise ValueError("er-tilde-logspace produced non-finite values")
    return grid


def _resolve_er_tilde_grid(args: argparse.Namespace) -> jnp.ndarray:
    if args.er_tilde is not None and args.er_tilde_logspace is not None:
        raise ValueError("set only one of --er-tilde or --er-tilde-logspace")
    if args.er_tilde is not None:
        return _parse_float_grid(args.er_tilde, default=DEFAULT_ER_TILDE, name="er_tilde")
    er_tilde_logspace = _parse_er_tilde_logspace(
        args.er_tilde_logspace,
        include_zero=bool(args.er_tilde_include_zero),
    )
    if er_tilde_logspace is not None:
        return er_tilde_logspace
    return _parse_float_grid(None, default=DEFAULT_ER_TILDE, name="er_tilde")


def _validate_scan_axes(rho: jnp.ndarray, nu_v: jnp.ndarray, er_tilde: jnp.ndarray) -> None:
    if rho.ndim != 1 or nu_v.ndim != 1 or er_tilde.ndim != 1:
        raise ValueError("rho, nu_v, and er_tilde must be one-dimensional arrays")
    if not bool(jnp.all((rho > 0.0) & (rho <= 1.0))):
        raise ValueError("rho values must satisfy 0 < rho <= 1")
    if not bool(jnp.all(nu_v > 0.0)):
        raise ValueError("nu_v values must be positive")
    for name, values in (("rho", rho), ("nu_v", nu_v), ("er_tilde", er_tilde)):
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError(f"{name} contains non-finite values")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a regular file: {path}")


def _filled(variable) -> np.ndarray:
    values = variable[:]
    if hasattr(values, "filled"):
        values = values.filled()
    return np.asarray(values, dtype=float)


def _interpolator(x, y):
    return Interpolator1D(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(y, dtype=jnp.float64),
        method="akima",
    )


def _load_vmec_boozer_channels(
    wout_path: Path,
    boozmn_path: Path,
    rho: jnp.ndarray,
) -> dict[str, jnp.ndarray | float]:
    _validate_scan_axes(rho, jnp.asarray([1.0], dtype=jnp.float64), jnp.asarray([0.0]))
    _require_file(wout_path, "VMEC wout")
    _require_file(boozmn_path, "Boozer")

    with Dataset(wout_path, mode="r") as vfile:
        ns = int(np.asarray(vfile.variables["ns"][:]).reshape(-1)[0])
        s_full = jnp.linspace(0.0, 1.0, ns)
        s_half = jnp.asarray([(i - 0.5) / (ns - 1) for i in range(ns)], dtype=jnp.float64)
        rho_half = jnp.sqrt(s_half)
        rho_full = jnp.sqrt(s_full)

        volume_p = float(np.asarray(vfile.variables["volume_p"][:]).reshape(-1)[-1])
        phi = _filled(vfile.variables["phi"])
        iotaf = _filled(vfile.variables["iotaf"])
        psia = float(jnp.abs(phi[-1]) / (2.0 * jnp.pi))

    with Dataset(boozmn_path, mode="r") as bfile:
        bmnc_b = _filled(bfile.variables["bmnc_b"])
        rmnc_b = _filled(bfile.variables["rmnc_b"])
        xm_b = _filled(bfile.variables["ixm_b"])
        xn_b = _filled(bfile.variables["ixn_b"])
        buco = _filled(bfile.variables["buco_b"])
        bvco = _filled(bfile.variables["bvco_b"])

    zero_mode = np.where((xm_b == 0) & (xn_b == 0))[0]
    if zero_mode.size == 0:
        raise ValueError("could not find Boozer (m,n)=(0,0) mode in boozmn file")
    mode00 = int(zero_mode[0])

    r0_b = float(rmnc_b[-1, mode00])
    a_b = float(np.sqrt(volume_p / (2.0 * np.pi**2 * r0_b)))

    b00 = _interpolator(rho_half[1:], bmnc_b[:, mode00])
    r00 = _interpolator(rho_full[1:], rmnc_b[:, mode00])
    boozer_i = _interpolator(rho_half[1:], buco[1:])
    boozer_g = _interpolator(rho_half[1:], bvco[1:])
    iota = _interpolator(rho_full, iotaf)

    b00_rho = b00(rho)
    r00_rho = r00(rho)
    i_rho = boozer_i(rho)
    g_rho = boozer_g(rho)
    iota_rho = iota(rho)

    dpsidrtilde = rho * a_b * b00_rho
    drds = a_b / (2.0 * rho)
    dr_tildedr = 2.0 * psia / (a_b**2 * b00_rho)
    dr_tildeds = dr_tildedr * drds

    boozer_jacobian = g_rho + iota_rho * i_rho
    sqrt_pi = jnp.sqrt(jnp.pi)
    fac_reference_to_sfincs_11 = (
        8.0 * boozer_jacobian * b00_rho * psia**2 / (sqrt_pi * g_rho**2)
    )
    fac_reference_to_sfincs_31 = 4.0 * b00_rho * psia / (sqrt_pi * g_rho)
    fac_reference_to_sfincs_33 = -2.0 * b00_rho / (boozer_jacobian * sqrt_pi)

    fac_sfincs_to_dkes_11 = 1.0 / (
        8.0 * boozer_jacobian * dpsidrtilde**2 / (g_rho**2 * b00_rho * sqrt_pi)
    )
    fac_sfincs_to_dkes_31 = 1.0 / (4.0 * dpsidrtilde / (g_rho * sqrt_pi))
    fac_sfincs_to_dkes_33 = 1.0 / (-2.0 * b00_rho / (boozer_jacobian * sqrt_pi))

    epsilon_t = rho * a_b / r00_rho
    fac_dkes_to_d11star = -(8.0 / jnp.pi) * iota_rho * r00_rho
    fac_dkes_to_d31star = -(3.0 / 1.46) * iota_rho * jnp.sqrt(epsilon_t) / 2.0
    fac_dkes_to_d33star = jnp.asarray(1.0, dtype=jnp.float64)

    return {
        "a_b": a_b,
        "psia": psia,
        "b00": b00_rho,
        "r00": r00_rho,
        "boozer_i": i_rho,
        "boozer_g": g_rho,
        "iota": iota_rho,
        "drds": drds,
        "dr_tildedr": dr_tildedr,
        "dr_tildeds": dr_tildeds,
        "fac_reference_to_sfincs_11": fac_reference_to_sfincs_11,
        "fac_reference_to_sfincs_31": fac_reference_to_sfincs_31,
        "fac_reference_to_sfincs_33": fac_reference_to_sfincs_33,
        "fac_sfincs_to_dkes_11": fac_sfincs_to_dkes_11,
        "fac_sfincs_to_dkes_31": fac_sfincs_to_dkes_31,
        "fac_sfincs_to_dkes_33": fac_sfincs_to_dkes_33,
        "fac_dkes_to_d11star": fac_dkes_to_d11star,
        "fac_dkes_to_d31star": fac_dkes_to_d31star,
        "fac_dkes_to_d33star": fac_dkes_to_d33star,
    }


def _build_field_channels(
    rho: jnp.ndarray,
    er_tilde: jnp.ndarray,
    b00: jnp.ndarray,
    dr_tildedr: jnp.ndarray,
    dr_tildeds: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    er = er_tilde[None, :] * dr_tildedr[:, None] * b00[:, None]
    es = er_tilde[None, :] * dr_tildeds[:, None] * b00[:, None]
    er_to_ertilde = jnp.broadcast_to(1.0 / dr_tildedr[:, None], er.shape)
    return er, es, er_to_ertilde


def _surface_loader(wout_path: Path, rho_value: float):
    return surface_from_vmex_vmec_wout_file(wout_path, s=float(rho_value**2))


def _select_surface_loader(*, backend: str, wout_path: Path, boozmn_path: Path):
    if backend not in {"auto", "vmec", "boozmn"}:
        raise ValueError(f"unsupported backend {backend!r}")
    if backend == "boozmn":
        _require_file(boozmn_path, "Boozer")
        return (
            lambda rho_value: load_boozmn_surface(boozmn_path, rho=float(rho_value)).surface,
            "boozmn",
        )
    if backend == "vmec":
        _require_file(wout_path, "VMEC wout")
        return (
            lambda rho_value: _surface_loader(wout_path, float(rho_value)),
            "vmex",
        )
    if wout_path.exists():
        return (
            lambda rho_value: _surface_loader(wout_path, float(rho_value)),
            "vmex",
        )
    if boozmn_path.exists():
        return (
            lambda rho_value: load_boozmn_surface(boozmn_path, rho=float(rho_value)).surface,
            "boozmn",
        )
    raise FileNotFoundError("no usable backend input file was found")


def _select_jax_device(*, backend: str, device_index: int):
    if backend == "auto":
        devices = list(jax.devices())
    else:
        try:
            devices = list(jax.devices(backend))
        except RuntimeError as exc:
            available = ", ".join(sorted({device.platform for device in jax.devices()}))
            raise RuntimeError(
                f"JAX backend {backend!r} is not available; available platforms: "
                f"{available or 'none'}. Use --device-backend cpu on CPU-only laptops "
                "or run on a machine with a configured JAX GPU backend."
            ) from exc
    if not devices:
        raise RuntimeError(f"no JAX devices available for backend {backend!r}")
    if device_index < 0 or device_index >= len(devices):
        raise IndexError(
            f"device_index {device_index} is outside [0, {len(devices) - 1}] "
            f"for backend {backend!r}"
        )
    return devices[device_index]


def _report_scan_warnings(
    scan: NeopaxScan,
    *,
    onsager_warn_threshold: float = ONSAGER_WARN_THRESHOLD,
) -> None:
    d11 = np.asarray(scan.D11, dtype=float)
    d13 = np.asarray(scan.D13, dtype=float)
    d31 = np.asarray(scan.D31, dtype=float) if scan.D31 is not None else None
    d33 = np.asarray(scan.D33, dtype=float)
    er_tilde = (
        np.asarray(scan.Er_tilde, dtype=float)
        if scan.Er_tilde is not None
        else np.arange(d11.shape[2], dtype=float)
    )

    any_issue = False

    finite_mask = np.isfinite(d11) & np.isfinite(d13) & np.isfinite(d33)
    if d31 is not None:
        finite_mask = finite_mask & np.isfinite(d31)
    if not bool(np.all(finite_mask)):
        bad = int(np.size(finite_mask) - np.count_nonzero(finite_mask))
        print(f"warning: found {bad} non-finite coefficient entries in the scan")
        any_issue = True

    negative_d11 = np.argwhere(d11 < 0.0)
    if negative_d11.size > 0:
        print(f"warning: found {negative_d11.shape[0]} entries with D11 < 0")
        for idx in negative_d11[:10]:
            ir, inu, ier = (int(v) for v in idx)
            print(
                "  "
                f"rho={float(scan.rho[ir]):.5f}, "
                f"nu_v={float(scan.nu_v[inu]):.6e}, "
                f"Er_tilde={float(er_tilde[ier]):.6e}, "
                f"D11={d11[ir, inu, ier]:.6e}"
            )
        if negative_d11.shape[0] > 10:
            print(f"  ... plus {negative_d11.shape[0] - 10} more negative-D11 entries")
        any_issue = True

    if d31 is not None:
        onsager = np.abs(d13 + d31)
        max_onsager = float(np.max(onsager))
        if max_onsager > onsager_warn_threshold:
            worst = np.unravel_index(int(np.argmax(onsager)), onsager.shape)
            ir, inu, ier = (int(v) for v in worst)
            print(
                "warning: Onsager mismatch exceeded threshold "
                f"({onsager_warn_threshold:.1e}); max |D13 + D31| = {max_onsager:.6e}"
            )
            print(
                "  "
                f"rho={float(scan.rho[ir]):.5f}, "
                f"nu_v={float(scan.nu_v[inu]):.6e}, "
                f"Er_tilde={float(er_tilde[ier]):.6e}, "
                f"D13={d13[ir, inu, ier]:.6e}, "
                f"D31={d31[ir, inu, ier]:.6e}"
            )
            any_issue = True

    if not any_issue:
        print(
            "scan sanity checks: no negative D11, no non-finite values, "
            "Onsager mismatch within threshold"
        )


def _plot_scan_coefficients(
    scan: NeopaxScan,
    plot_output: Path | None,
    *,
    rho_index: int | None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    base = plot_output if plot_output is not None else OUTPUT_PATH.with_suffix(".png")
    base = base.expanduser().resolve()
    suffix = base.suffix if base.suffix else ".png"
    stem = base.stem if base.suffix else base.name
    parent = base.parent if base.suffix else base
    parent.mkdir(parents=True, exist_ok=True)

    rho = np.asarray(scan.rho, dtype=float)
    nu_v = np.asarray(scan.nu_v, dtype=float)
    if np.any(nu_v <= 0.0):
        raise ValueError("nu_v values must be positive for coefficient plots")
    ln_nu_v = np.log(nu_v)
    er_tilde = (
        np.asarray(scan.Er_tilde, dtype=float)
        if scan.Er_tilde is not None
        else np.arange(scan.D11.shape[2], dtype=float)
    )
    d11 = np.asarray(scan.D11, dtype=float)
    d13 = np.asarray(scan.D13, dtype=float)
    d31 = np.asarray(scan.D31, dtype=float) if scan.D31 is not None else None
    d33 = np.asarray(scan.D33, dtype=float)

    written: list[Path] = []
    rho_indices = list(range(rho.shape[0])) if rho_index is None else [rho_index]
    for ir in rho_indices:
        if ir < 0 or ir >= rho.shape[0]:
            raise IndexError(f"plot_rho_index {ir} is outside [0, {rho.shape[0] - 1}]")
        rho_value = rho[ir]
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
        panels = (
            ("D11", d11[ir], axes[0, 0]),
            ("D13", d13[ir], axes[0, 1]),
            ("D31", None if d31 is None else d31[ir], axes[1, 0]),
            ("D33", d33[ir], axes[1, 1]),
        )

        for label, values, ax in panels:
            if values is None:
                ax.set_visible(False)
                continue
            for ier, er_tilde_value in enumerate(er_tilde):
                if label in {"D11", "D33"}:
                    safe_values = np.maximum(np.abs(values[:, ier]), 1.0e-300)
                    ax.plot(
                        ln_nu_v,
                        np.log(safe_values),
                        lw=1.8,
                        label=rf"$\tilde E_r={er_tilde_value:.1e}$",
                    )
                else:
                    ax.plot(
                        ln_nu_v,
                        values[:, ier],
                        lw=1.8,
                        label=rf"$\tilde E_r={er_tilde_value:.1e}$",
                    )
            if label in {"D11", "D33"}:
                ax.set_title(f"ln({label}) vs ln(nu_v)")
                ax.set_xlabel("ln(nu_v)")
                ax.set_ylabel(f"ln({label})")
            else:
                ax.set_title(f"{label} vs ln(nu_v)")
                ax.set_xlabel("ln(nu_v)")
                ax.set_ylabel(label)
            ax.grid(alpha=0.24, lw=0.6, which="both")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center right", frameon=False)
        fig.suptitle(f"NTX monoenergetic scan coefficients at rho={rho_value:.5f}", fontsize=14)

        out_path = parent / f"{stem}_rho_{rho_value:.5f}{suffix}"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def build_scan(
    *,
    wout_path: Path,
    boozmn_path: Path,
    grid: GridSpec,
    backend: str,
    rho: jnp.ndarray,
    nu_v: jnp.ndarray,
    er_tilde: jnp.ndarray,
    source_name: str | None = None,
    scan_batch_size: int | None = None,
    parallel_devices: int | None = None,
    progress: bool = False,
) -> NeopaxScan:
    _validate_scan_axes(rho, nu_v, er_tilde)
    if scan_batch_size is not None and scan_batch_size < 1:
        raise ValueError("scan_batch_size must be a positive integer")
    if parallel_devices is not None and parallel_devices < 1:
        raise ValueError("parallel_devices must be a positive integer")
    channels = _load_vmec_boozer_channels(wout_path, boozmn_path, rho)
    load_surface, backend_name = _select_surface_loader(
        backend=backend,
        wout_path=wout_path,
        boozmn_path=boozmn_path,
    )
    er, es, er_to_ertilde = _build_field_channels(
        rho,
        er_tilde,
        channels["b00"],
        channels["dr_tildedr"],
        channels["dr_tildeds"],
    )

    n_r = int(rho.shape[0])
    n_nu = int(nu_v.shape[0])
    n_er = int(er_tilde.shape[0])
    d11 = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)
    d13 = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)
    d31 = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)
    d33 = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)
    d33_spitzer = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)
    case_count = n_nu * n_er
    batch_label = "full-surface" if scan_batch_size is None else str(scan_batch_size)
    parallel_label = "serial" if parallel_devices is None else str(parallel_devices)

    for idx, rho_value in enumerate(np.asarray(rho)):
        t0 = time.perf_counter()
        if progress:
            print(
                f"[{idx + 1}/{n_r}] rho={float(rho_value):.5f}: "
                f"{case_count} cases, grid={grid.n_theta}x{grid.n_zeta}x{grid.n_xi}, "
                f"scan_batch_size={batch_label}, parallel_devices_requested={parallel_label}",
                flush=True,
            )
        surface = load_surface(float(rho_value))
        nu_grid, es_grid = jnp.meshgrid(nu_v, es[idx], indexing="ij")
        if parallel_devices is None:
            coeffs = solve_monoenergetic_scan(
                surface,
                grid,
                nu_grid,
                epsi_hat=es_grid,
                scan_batch_size=scan_batch_size,
            )
        else:
            coeffs = solve_monoenergetic_parallel_scan(
                surface,
                grid,
                nu_grid,
                epsi_hat=es_grid,
                num_devices=parallel_devices,
                scan_batch_size=scan_batch_size,
            )
        if progress:
            jax.block_until_ready(tuple(coeffs.values()))
            print(
                f"[{idx + 1}/{n_r}] rho={float(rho_value):.5f}: "
                f"solved in {time.perf_counter() - t0:.2f} s",
                flush=True,
            )
        d11 = d11.at[idx].set(coeffs["D11"])
        d13 = d13.at[idx].set(coeffs["D13"])
        d31 = d31.at[idx].set(coeffs["D31"])
        d33 = d33.at[idx].set(coeffs["D33"])
        d33_spitzer = d33_spitzer.at[idx].set(coeffs["D33_spitzer"])

    return NeopaxScan(
        rho=rho,
        nu_v=nu_v,
        Er=er,
        Es=es,
        drds=channels["drds"],
        D11=d11,
        D13=d13,
        D33=d33,
        D33_spitzer=d33_spitzer,
        D31=d31,
        Er_tilde=er_tilde,
        Er_to_Ertilde=er_to_ertilde,
        dr_tildedr=channels["dr_tildedr"],
        dr_tildeds=channels["dr_tildeds"],
        a_b=channels["a_b"],
        psia=channels["psia"],
        b00=channels["b00"],
        r00=channels["r00"],
        boozer_i=channels["boozer_i"],
        boozer_g=channels["boozer_g"],
        iota=channels["iota"],
        fac_reference_to_sfincs_11=channels["fac_reference_to_sfincs_11"],
        fac_reference_to_sfincs_31=channels["fac_reference_to_sfincs_31"],
        fac_reference_to_sfincs_33=channels["fac_reference_to_sfincs_33"],
        fac_monkes_to_sfincs_11=channels["fac_reference_to_sfincs_11"],
        fac_monkes_to_sfincs_31=channels["fac_reference_to_sfincs_31"],
        fac_monkes_to_sfincs_33=channels["fac_reference_to_sfincs_33"],
        fac_sfincs_to_dkes_11=channels["fac_sfincs_to_dkes_11"],
        fac_sfincs_to_dkes_31=channels["fac_sfincs_to_dkes_31"],
        fac_sfincs_to_dkes_33=channels["fac_sfincs_to_dkes_33"],
        fac_dkes_to_d11star=channels["fac_dkes_to_d11star"],
        fac_dkes_to_d31star=channels["fac_dkes_to_d31star"],
        fac_dkes_to_d33star=channels["fac_dkes_to_d33star"],
        source_name=source_name or f"ntx_scan_from_ertilde_{backend_name}",
    )


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()
    wout_path = args.wout.expanduser().resolve()
    boozmn_path = args.booz.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    rho = _resolve_rho_grid(args)
    nu_v = _parse_float_grid(args.nu_v, default=DEFAULT_NU_V, name="nu_v", positive=True)
    er_tilde = _resolve_er_tilde_grid(args)
    _validate_scan_axes(rho, nu_v, er_tilde)
    _require_file(wout_path, "VMEC wout")
    _require_file(boozmn_path, "Boozer")
    grid = GridSpec(n_theta=args.n_theta, n_zeta=args.n_zeta, n_xi=args.n_xi)
    device = _select_jax_device(backend=args.device_backend, device_index=args.device_index)
    if args.parallel_devices is not None:
        available_devices = sum(
            1 for candidate in jax.devices() if candidate.platform == device.platform
        )
        if available_devices < args.parallel_devices:
            print(
                "warning: requested "
                f"{args.parallel_devices} parallel device(s), but only "
                f"{available_devices} {device.platform} device(s) are visible. "
                "For CPU runs, set "
                "XLA_FLAGS=--xla_force_host_platform_device_count=N before launch.",
                file=sys.stderr,
                flush=True,
            )

    with jax.default_device(device):
        scan = build_scan(
            wout_path=wout_path,
            boozmn_path=boozmn_path,
            grid=grid,
            backend=args.surface_backend,
            rho=rho,
            nu_v=nu_v,
            er_tilde=er_tilde,
            source_name=args.source_name,
            scan_batch_size=args.scan_batch_size,
            parallel_devices=args.parallel_devices,
            progress=not args.quiet,
        )
    _report_scan_warnings(scan, onsager_warn_threshold=args.onsager_warn_threshold)
    output = write_neopax_scan_hdf5(scan, output_path)
    print(f"wrote NEOPAX-style scan to: {output}")
    print(f"source name: {scan.source_name}")
    print(f"requested surface backend: {args.surface_backend}")
    print(f"device backend: {args.device_backend}")
    print(f"device: {device}")
    print(
        "scan batch size: "
        f"{args.scan_batch_size if args.scan_batch_size is not None else 'full-surface'}"
    )
    print(
        "parallel devices requested: "
        f"{args.parallel_devices if args.parallel_devices is not None else 'serial'}"
    )
    print(f"wout: {wout_path}")
    print(f"booz: {boozmn_path}")
    print(f"grid: n_theta={grid.n_theta}, n_zeta={grid.n_zeta}, n_xi={grid.n_xi}")
    print(f"rho points: {scan.rho.shape[0]}")
    print(f"nu_v points: {scan.nu_v.shape[0]}")
    print(f"Er_tilde points: {scan.Er_tilde.shape[0] if scan.Er_tilde is not None else 0}")
    print(f"D11 shape: {scan.D11.shape}")
    print(f"D31 shape: {scan.D31.shape if scan.D31 is not None else None}")
    print(f"total runtime: {time.perf_counter() - t0:.2f} s")
    if args.plot:
        plot_paths = _plot_scan_coefficients(
            scan,
            args.plot_output,
            rho_index=args.plot_rho_index,
        )
        print("plots:")
        for path in plot_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
