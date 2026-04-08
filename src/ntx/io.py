"""Input/output helpers for Boozer-surface files."""

from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .geometry import BoozerSurface
from .vmec import load_vmec_surface

__all__ = [
    "load_boozer_modes_csv",
    "load_dkes_surface",
    "load_magnetic_configuration_surface",
    "load_vmec_surface",
    "write_result_jsonable",
]


def load_boozer_modes_csv(
    path: str | Path,
    *,
    nfp: int,
    iota: float,
    psi_p: float,
    b_theta: float,
    b_zeta: float,
) -> BoozerSurface:
    """Load columns `m,n,b_cos[,b_sin]` from a CSV or whitespace-delimited file."""

    data = np.genfromtxt(path, names=True, delimiter=None)
    names = set(data.dtype.names or ())
    required = {"m", "n", "b_cos"}
    if not required.issubset(names):
        msg = f"expected columns {sorted(required)} in {path}"
        raise ValueError(msg)
    b_sin = data["b_sin"] if "b_sin" in names else None
    b0 = float(data["b_cos"][np.logical_and(data["m"] == 0, data["n"] == 0)][0])
    return BoozerSurface(
        m=jnp.asarray(data["m"], dtype=jnp.int32),
        n=jnp.asarray(data["n"], dtype=jnp.int32),
        b_cos=jnp.asarray(data["b_cos"], dtype=jnp.float64),
        b_sin=None if b_sin is None else jnp.asarray(b_sin, dtype=jnp.float64),
        nfp=nfp,
        iota=iota,
        psi_p=psi_p,
        b_theta=b_theta,
        b_zeta=b_zeta,
        chi_p=iota * psi_p,
        b0=b0,
        stellarator_symmetric=b_sin is None,
        source_path=Path(path).expanduser().resolve(),
    )


_SCALAR_PATTERN = r"(?im)\b{name}\s*=\s*([^,\n/]+)"
_BORBI_PATTERN = re.compile(
    r"(?im)\bborbi\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[deDE][+-]?\d+)?)"
)
_MAGNETIC_MODE_PATTERN = re.compile(
    r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[deDE][+-]?\d+)?)",
    re.MULTILINE,
)


def load_dkes_surface(path: str | Path) -> BoozerSurface:
    """Load a DKES-format `ddkes2.data` flux surface.

    REFERENCE_EXECUTABLE stores the scalar geometry using the `datain` namelist and interprets
    the Fourier entries as `borbi(n, m) = B_mn`.
    """

    text = Path(path).read_text(encoding="utf-8")
    nfp = int(_parse_scalar(text, "nzperiod"))
    psi_p = _parse_scalar(text, "psip")
    chip = _parse_scalar(text, "chip")
    b_theta = _parse_scalar(text, "btheta")
    b_zeta = _parse_scalar(text, "bzeta")

    modes: list[tuple[int, int, float]] = []
    for toroidal_mode, poloidal_mode, value in _BORBI_PATTERN.findall(text):
        modes.append((int(poloidal_mode), int(toroidal_mode), _parse_float(value)))
    if not modes:
        msg = f"no borbi(m,n) entries found in {path}"
        raise ValueError(msg)
    modes.sort()

    m = jnp.asarray([mode[0] for mode in modes], dtype=jnp.int32)
    n = jnp.asarray([mode[1] for mode in modes], dtype=jnp.int32)
    b_cos = jnp.asarray([mode[2] for mode in modes], dtype=jnp.float64)
    b0_matches = [value for mm, nn, value in modes if mm == 0 and nn == 0]
    if not b0_matches:
        msg = f"missing borbi(0,0) entry in {path}"
        raise ValueError(msg)
    return BoozerSurface(
        m=m,
        n=n,
        b_cos=b_cos,
        nfp=nfp,
        iota=(-chip) / psi_p,
        psi_p=psi_p,
        b_theta=b_theta,
        b_zeta=b_zeta,
        chi_p=-chip,
        b0=b0_matches[0],
        source_path=Path(path).expanduser().resolve(),
    )


def load_magnetic_configuration_surface(path: str | Path) -> BoozerSurface:
    """Load a text magnetic-configuration file with `m n B_mn` Fourier rows."""

    resolved = Path(path).expanduser().resolve()
    text = resolved.read_text(encoding="utf-8")
    nfp = int(_parse_scalar(text, "Number of periods"))
    psi_p = _parse_scalar(text, "psi_p")
    chi_p = _parse_scalar(text, "chi_p")
    iota = _parse_scalar(text, "iota")
    b0 = _parse_scalar(text, "B00")
    b_theta = _parse_scalar(text, "B_theta")
    b_zeta = _parse_scalar(text, "B_zeta")

    marker = "*** Magnetic field strength Fourier modes"
    if marker not in text:
        msg = f"missing Fourier-mode section in {resolved}"
        raise ValueError(msg)
    section = text.split(marker, maxsplit=1)[1]
    rows: list[tuple[int, int, float]] = []
    for m_text, n_text, value_text in _MAGNETIC_MODE_PATTERN.findall(section):
        rows.append((int(m_text), int(n_text), _parse_float(value_text)))
    if not rows:
        msg = f"no Fourier rows found in {resolved}"
        raise ValueError(msg)
    rows.sort()

    return BoozerSurface(
        m=jnp.asarray([row[0] for row in rows], dtype=jnp.int32),
        n=jnp.asarray([row[1] for row in rows], dtype=jnp.int32),
        b_cos=jnp.asarray([row[2] for row in rows], dtype=jnp.float64),
        nfp=nfp,
        iota=iota,
        psi_p=psi_p,
        b_theta=b_theta,
        b_zeta=b_zeta,
        chi_p=chi_p,
        b0=b0,
        source_path=resolved,
    )


def write_result_jsonable(result) -> dict[str, float]:
    """Convert a result to a small JSON-serializable mapping."""

    return result.as_dict()


def _parse_scalar(text: str, name: str) -> float:
    match = re.search(_SCALAR_PATTERN.format(name=re.escape(name)), text)
    if match is None:
        msg = f"missing `{name}` in DKES input"
        raise ValueError(msg)
    return _parse_float(match.group(1))


def _parse_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))
