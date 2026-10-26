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
    "load_geiger_boozer_surface",
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

    The scalar geometry is stored in the `datain` namelist and the Fourier
    entries are interpreted as `borbi(n, m) = B_mn`.
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
    # Sort so the harmonic ordering is a property of the data rather than of
    # the order rows happened to appear in the file.
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


_MU0 = 4.0e-7 * np.pi


def load_geiger_boozer_surface(
    path: str | Path,
    *,
    rho: float | None = None,
    s: float | None = None,
    min_bmn: float = 0.0,
) -> BoozerSurface:
    """Load one flux surface of a Geiger/IPP "CC Boozer-coordinate data file".

    The file (header ``m0b n0b nsurf nper flux a R ...``, then per surface
    ``s iota curr_pol/nper curr_tor pprime sqrt g`` and rows ``m n r z (phib-phi) bmn``)
    is the ``.bc`` format read by SFINCS and DKX with ``geometryScheme = 11``. The surface at
    ``rho = sqrt(s)`` (or ``s``) is the linear interpolation in ``s`` of the two bracketing file
    surfaces, mode by mode (``VMECRadialOption = 0``); a request outside the file is clamped to
    its first or last surface. ``G = -mu0 curr_pol/(2 pi)`` and ``I = mu0 curr_tor/(2 pi)`` in T m,
    ``iota -> -iota``
    (the ``theta -> -theta`` that takes the file's left-handed (r, theta, zeta) to the right-handed
    system of SFINCS and DKX, whose ``n`` is then NTX's ``n``). ``psi_p`` is
    ``d psi / d r = B0 r`` with ``r = sqrt(2 psi / B0)`` and ``psi = s |flux| / (2 pi)``, so
    ``er_hat = E_r / v`` in V m^-1 per m s^-1 and the coefficients are in the effective radius.
    ``min_bmn`` drops modes with ``|B_mn / B_00|`` below it.
    """

    if (rho is None) == (s is None):
        raise ValueError("set exactly one of rho or s")
    s_wish = float(rho) ** 2 if rho is not None else float(s)
    resolved = Path(path).expanduser().resolve()
    lines = (
        resolved.read_text(encoding="utf-8", errors="replace")
        .replace("D", "E")
        .replace("d", "E")
        .splitlines()
    )

    header: list[float] | None = None
    for i, line in enumerate(lines):
        if line.startswith("CC"):
            continue
        tokens = line.split()
        if (
            len(tokens) >= 6
            and all(_is_float(t) for t in tokens[:4])
            and not line.lstrip().startswith("s ")
        ):
            # The ten-number header may carry trailing text ("original: a=... m").
            header = [float(t) for t in tokens[4:] if _is_float(t)]
            n_periods = int(tokens[3])
            first = i
            break
    if header is None or len(header) < 2:
        raise ValueError(f"no Boozer-file header (m0b n0b nsurf nper flux a ...) in {resolved}")
    psi_edge = abs(header[0]) / (2.0 * np.pi)

    titles = [
        i
        for i in range(first, len(lines))
        if lines[i].lstrip().startswith("s ") and "iota" in lines[i]
    ]
    if not titles:
        raise ValueError(f"no surface blocks in {resolved}")
    scalars = np.array([[float(t) for t in lines[i + 2].split()[:6]] for i in titles])
    s_all = scalars[:, 0]
    hi = int(np.clip(np.searchsorted(s_all, s_wish), 1, len(s_all) - 1)) if len(s_all) > 1 else 0
    lo = max(hi - 1, 0)
    weight = (
        1.0
        if hi == lo
        else float(np.clip((s_wish - s_all[lo]) / (s_all[hi] - s_all[lo]), 0.0, 1.0))
    )
    s_used = (1.0 - weight) * s_all[lo] + weight * s_all[hi]

    def block(k: int) -> np.ndarray:
        """``(m, n, bmn)`` rows of the ``k``-th surface block."""

        end = titles[k + 1] if k + 1 < len(titles) else len(lines)
        return np.loadtxt(lines[titles[k] + 4 : end], usecols=(0, 1, 5), dtype=float, ndmin=2)

    blo, bhi = block(lo), block(hi)
    if not np.array_equal(blo[:, :2], bhi[:, :2]):
        raise ValueError("neighbouring surfaces list different (m, n) modes")
    bmn = (1.0 - weight) * blo[:, 2] + weight * bhi[:, 2]
    iota, curr_pol, curr_tor = (
        (1.0 - weight) * scalars[lo, j] + weight * scalars[hi, j] for j in (1, 2, 3)
    )
    m = blo[:, 0].astype(np.int32)
    n = blo[:, 1].astype(np.int32)
    b0 = float(bmn[(m == 0) & (n == 0)][0])
    keep = (np.abs(bmn / b0) >= min_bmn) | ((m == 0) & (n == 0))
    r_eff = float(np.sqrt(2.0 * s_used * psi_edge / b0))
    return BoozerSurface(
        m=jnp.asarray(m[keep]),
        n=jnp.asarray(n[keep]),
        b_cos=jnp.asarray(bmn[keep]),
        nfp=n_periods,
        iota=-float(iota),
        psi_p=b0 * r_eff,
        b_theta=curr_tor / (2.0 * np.pi) * _MU0,
        b_zeta=-curr_pol * n_periods / (2.0 * np.pi) * _MU0,
        b0=b0,
        source_path=resolved,
    )


def _is_float(token: str) -> bool:
    """Whether a whitespace-separated token is a number (``D`` exponents already rewritten)."""

    try:
        float(token)
    except ValueError:
        return False
    return True


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
    # Same ordering guarantee as the DKES loader, so the two produce
    # comparable surfaces from equivalent input.
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
    """Extract a named scalar from a DKES-format text file."""
    match = re.search(_SCALAR_PATTERN.format(name=re.escape(name)), text)
    if match is None:
        msg = f"missing `{name}` in DKES input"
        raise ValueError(msg)
    return _parse_float(match.group(1))


def _parse_float(value: str) -> float:
    """Parse a Fortran float, accepting D exponents.

    Fortran writes double precision as 1.0D+00; Python's float does not accept
    that spelling.
    """
    return float(value.replace("D", "E").replace("d", "e"))
