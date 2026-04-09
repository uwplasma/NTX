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
        from booz_xform_jax import Booz_xform
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "load_boozmn_surface requires booz_xform_jax. Install the local package with "
            "`pip install -e /Users/rogeriojorge/local/booz_xform_jax`."
        ) from exc

    bx = Booz_xform()
    bx.verbose = 0
    bx.read_boozmn(str(booz_path))

    xm = np.asarray(bx.xm_b, dtype=np.int32).reshape(-1)
    xn = np.asarray(bx.xn_b, dtype=np.int32).reshape(-1)
    bmnc = np.asarray(bx.bmnc_b, dtype=np.float64).T
    iota = np.asarray(bx.iota, dtype=np.float64).reshape(-1)
    buco = np.asarray(bx.Boozer_I_all, dtype=np.float64).reshape(-1)
    bvco = np.asarray(bx.Boozer_G_all, dtype=np.float64).reshape(-1)
    phi_attr = getattr(bx, "phi", None)
    phi = None if phi_attr is None else np.asarray(phi_attr, dtype=np.float64).reshape(-1)
    s_grid = (
        np.asarray(phi / float(phi[-1]), dtype=np.float64)
        if phi is not None and phi.size == bmnc.shape[0]
        else np.asarray(bx.s_in, dtype=np.float64).reshape(-1)
    )
    nfp = int(np.asarray(bx.nfp).reshape(()))

    if bmnc.ndim != 2:
        raise ValueError("expected booz_xform_jax bmnc_b to be a 2D `(surface, mode)` array")
    ns_b, mode_count = bmnc.shape
    if s_grid.shape[0] != ns_b:
        raise ValueError("booz_xform_jax radial grid length does not match bmnc_b")
    rho_grid = np.sqrt(np.clip(s_grid, 0.0, None))

    idx: int
    if surface_index is not None:
        idx = int(surface_index)
        s_value = float(s_grid[idx])
    elif s is not None:
        s_value = float(s)
        idx = int(np.argmin(np.abs(s_grid - s_value)))
    else:
        assert rho is not None
        s_value = float(rho) ** 2
        idx = int(np.argmin(np.abs(rho_grid - float(rho))))

    if idx < 0 or idx >= ns_b:
        raise IndexError(f"surface_index {idx} is outside [0, {ns_b})")

    if surface_index is not None:
        bmn = bmnc[idx]
        iota_value = float(iota[idx])
        buco_value = float(buco[idx])
        bvco_value = float(bvco[idx])
    else:
        bmn = np.vstack(
            [np.interp(s_value, s_grid, bmnc[:, mode]) for mode in range(mode_count)]
        ).reshape(-1)
        iota_value = float(np.interp(s_value, s_grid, iota))
        buco_value = float(np.interp(s_value, s_grid, buco))
        bvco_value = float(np.interp(s_value, s_grid, bvco))

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
