"""Boozer-surface helpers backed by booz_xform_jax-compatible files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .geometry import BoozerSurface


@dataclass(frozen=True)
class BoozmnSurface:
    """Boozer surface plus file-side metadata."""

    surface: BoozerSurface
    path: Path
    s: float
    rho: float
    surface_index: int
    mode_count: int


def load_boozmn_surface(
    path: str | Path,
    *,
    s: float | None = None,
    rho: float | None = None,
    surface_index: int | None = None,
    psi_p: float = 1.0,
    min_bmn_to_load: float = 0.0,
) -> BoozmnSurface:
    """Load one surface from a Boozer `boozmn` netCDF file.

    Parameters
    ----------
    path:
        Path to a `boozmn_*.nc` or equivalent file.
    s, rho, surface_index:
        Select exactly one surface using normalized toroidal flux, normalized
        minor radius, or the 0-based stored surface index.
    psi_p:
        Poloidal-flux normalization used when converting `er_hat` to
        `epsi_hat`. This quantity is not needed when solving directly in
        `epsi_hat`.
    min_bmn_to_load:
        Drop small `|B_{mn}|/B00` coefficients after loading.
    """

    booz_path = Path(path).expanduser().resolve()
    selectors = sum(value is not None for value in (s, rho, surface_index))
    if selectors != 1:
        raise ValueError("set exactly one of s, rho, or surface_index")

    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "load_boozmn_surface requires the optional 'io' dependencies. "
            "Install NTX with `pip install ntx[io]`."
        ) from exc

    with Dataset(booz_path) as ds:
        xm = np.asarray(ds.variables["ixm_b"][:]).reshape(-1)
        xn = np.asarray(ds.variables["ixn_b"][:]).reshape(-1)
        bmnc = np.asarray(ds.variables["bmnc_b"][:])
        iota = np.asarray(ds.variables["iota_b"][:]).reshape(-1)
        buco = np.asarray(ds.variables["buco_b"][:]).reshape(-1)
        bvco = np.asarray(ds.variables["bvco_b"][:]).reshape(-1)
        nfp = int(np.asarray(ds.variables["nfp_b"][:]).reshape(()))
        phi_b = (
            np.asarray(ds.variables["phi_b"][:]).reshape(-1)
            if "phi_b" in ds.variables
            else None
        )

    if bmnc.ndim != 2:
        raise ValueError("expected bmnc_b to be a 2D `(surface, mode)` array")
    ns_b, mode_count = bmnc.shape
    s_grid_full: NDArray[np.float64]
    if phi_b is not None and phi_b.shape[0] == ns_b + 1:
        s_grid_full = np.asarray(phi_b / float(phi_b[-1]), dtype=np.float64)
        s_grid_modes = s_grid_full[1:]
    elif phi_b is not None and phi_b.shape[0] == ns_b:
        s_grid_full = np.asarray(phi_b / float(phi_b[-1]), dtype=np.float64)
        s_grid_modes = s_grid_full
    else:
        s_grid_full = np.asarray(
            np.linspace(0.0, 1.0, ns_b + 1, dtype=np.float64),
            dtype=np.float64,
        )
        s_grid_modes = np.asarray(
            (np.arange(ns_b, dtype=np.float64) + 1.0) / float(ns_b),
            dtype=np.float64,
        )
    rho_grid = np.sqrt(np.clip(s_grid_modes, 0.0, None))

    idx: int
    if surface_index is not None:
        idx = int(surface_index)
        s_value = float(s_grid_modes[idx])
    elif s is not None:
        s_value = float(s)
        idx = int(np.argmin(np.abs(s_grid_modes - s_value)))
    else:
        assert rho is not None
        s_value = float(rho) ** 2
        idx = int(np.argmin(np.abs(rho_grid - float(rho))))

    if idx < 0 or idx >= ns_b:
        raise IndexError(f"surface_index {idx} is outside [0, {ns_b})")

    if surface_index is not None:
        bmn = bmnc[idx]
        full_idx = min(idx + 1, iota.shape[0] - 1)
        iota_value = float(iota[full_idx])
        buco_value = float(buco[full_idx])
        bvco_value = float(bvco[full_idx])
    else:
        bmn = np.vstack(
            [np.interp(s_value, s_grid_modes, bmnc[:, mode]) for mode in range(mode_count)]
        ).reshape(-1)
        iota_value = float(np.interp(s_value, s_grid_full, iota))
        buco_value = float(np.interp(s_value, s_grid_full, buco))
        bvco_value = float(np.interp(s_value, s_grid_full, bvco))

    # Match the right-handed sign convention used in the JAX REFERENCE_EXECUTABLE field loader.
    iota_value = -iota_value
    buco_value = -buco_value
    sign = 1.0 if (bvco_value + iota_value * buco_value) >= 0.0 else -1.0
    buco_value *= sign
    bvco_value *= sign

    b0 = float(bmn[0])
    if b0 == 0.0:
        raise ValueError("Boozer mode (m,n)=(0,0) is zero on the selected surface")

    include = np.abs(bmn / b0) >= float(min_bmn_to_load)
    include[0] = True

    surface = BoozerSurface(
        m=jnp.asarray(xm[include], dtype=jnp.int32),
        n=jnp.asarray(np.rint(xn[include] / nfp).astype(np.int32), dtype=jnp.int32),
        b_cos=jnp.asarray(bmn[include]),
        nfp=nfp,
        iota=iota_value,
        psi_p=psi_p,
        b_theta=buco_value,
        b_zeta=bvco_value,
        b0=b0,
        source_path=booz_path,
    )
    return BoozmnSurface(
        surface=surface,
        path=booz_path,
        s=s_value,
        rho=float(np.sqrt(max(s_value, 0.0))),
        surface_index=idx,
        mode_count=int(np.count_nonzero(include)),
    )
