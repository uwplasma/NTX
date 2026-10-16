"""Boozer-surface helpers backed by `booz_xform_jax`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

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
    """Load one surface from a Boozer `boozmn` file through `booz_xform_jax`."""

    booz_path = Path(path).expanduser().resolve()
    selectors = sum(value is not None for value in (s, rho, surface_index))
    if selectors != 1:
        raise ValueError("set exactly one of s, rho, or surface_index")

    try:
        from booz_xform_jax import read_boozmn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "load_boozmn_surface requires booz_xform_jax>=0.5.0. Install it with "
            "`pip install 'ntx[booz]'`."
        ) from exc

    booz = read_boozmn(booz_path)
    xm, xn, bmnc, nfp = booz["ixm_b"], booz["ixn_b"], booz["bmnc_b"], booz["nfp_b"]
    # Profiles are on the full grid with an axis placeholder; spectra are packed
    # half-grid surfaces with 1-based full-grid labels `jlist`.
    iota, buco, bvco = (booz[key][1:] for key in ("iota_b", "buco_b", "bvco_b"))
    s_profile = (np.arange(iota.size) + 0.5) / iota.size
    s_bmn = s_profile[booz["jlist"] - 2]
    ns_b, mode_count = bmnc.shape
    rho_grid = np.sqrt(s_bmn)

    idx: int
    if surface_index is not None:
        idx = int(surface_index)
        s_value = float(s_bmn[idx])
    elif s is not None:
        s_value = float(s)
        idx = int(np.argmin(np.abs(s_bmn - s_value)))
    else:
        assert rho is not None
        s_value = float(rho) ** 2
        idx = int(np.argmin(np.abs(rho_grid - float(rho))))

    if idx < 0 or idx >= ns_b:
        raise IndexError(f"surface_index {idx} is outside [0, {ns_b})")

    if surface_index is not None:
        bmn = bmnc[idx]
    else:
        bmn = np.vstack(
            [np.interp(s_value, s_bmn, bmnc[:, mode]) for mode in range(mode_count)]
        ).reshape(-1)
    iota_value = float(np.interp(s_value, s_profile, iota))
    buco_value = float(np.interp(s_value, s_profile, buco))
    bvco_value = float(np.interp(s_value, s_profile, bvco))

    # Match the right-handed Boozer convention used by the external benchmark lane.
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
