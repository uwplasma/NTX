"""VMEC `wout` helpers for NTX flux-surface inputs."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.io import netcdf_file

from .geometry import VmecSurface


def load_vmec_surface(
    path: str | Path,
    *,
    psi_n: float,
    vmec_radial_option: int = 0,
    vmec_nyquist_option: int = 1,
    min_bmn_to_load: float = 0.0,
) -> VmecSurface:
    """Load one VMEC flux surface from a `wout_*.nc` file.

    Notes
    -----
    This loader keeps the coefficient-selection logic close to established
    stellarator transport workflows while using a minimal netCDF reader.
    """

    wout_path = Path(path).expanduser().resolve()
    if not wout_path.exists():
        raise FileNotFoundError(str(wout_path))

    with netcdf_file(wout_path, "r", mmap=False) as handle:
        nfp = int(_read_scalar(handle, "nfp"))
        ns = int(_read_scalar(handle, "ns"))
        mpol = int(_read_scalar(handle, "mpol"))
        ntor = int(_read_scalar(handle, "ntor"))
        lasym = bool(int(np.asarray(_read_var(handle, "lasym__logical__")).reshape(())))
        if lasym:
            raise NotImplementedError("VMEC lasym=true inputs are not supported yet")

        phi = _read_var(handle, "phi").astype(np.float64)
        psi_n_grid = phi / float(phi[-1])
        iota_key = "iota_f" if "iota_f" in handle.variables else "iotas"
        iota_grid = _read_var(handle, iota_key).astype(np.float64)
        aminor_p = (
            float(_read_scalar(handle, "Aminor_p"))
            if "Aminor_p" in handle.variables
            else None
        )

        mode_m = _read_var(handle, "xm_nyq" if "xm_nyq" in handle.variables else "xm").astype(
            np.int32
        )
        mode_n = _read_var(handle, "xn_nyq" if "xn_nyq" in handle.variables else "xn").astype(
            np.int32
        )
        bmnc = _read_modes(handle, "bmnc")
        gmnc = _read_modes(handle, "gmnc")
        bsubumnc = _read_modes(handle, "bsubumnc")
        bsubvmnc = _read_modes(handle, "bsubvmnc")
        bsupumnc = _read_modes(handle, "bsupumnc")
        bsupvmnc = _read_modes(handle, "bsupvmnc")

    if ns < 2:
        raise ValueError("VMEC input must contain at least two radial surfaces")

    psi_a_hat = float(phi[-1]) / (2.0 * np.pi)
    target_psi_n = _resolve_psi_n(psi_n_grid, float(psi_n), int(vmec_radial_option))
    radial_grid = psi_n_grid[1:]
    b_interp = _interp_mode_columns(radial_grid, bmnc[:, 1:], target_psi_n)
    g_interp = _interp_mode_columns(radial_grid, gmnc[:, 1:], target_psi_n)
    b_sub_theta_interp = _interp_mode_columns(radial_grid, bsubumnc[:, 1:], target_psi_n)
    b_sub_zeta_interp = _interp_mode_columns(radial_grid, bsubvmnc[:, 1:], target_psi_n)
    b_sup_theta_interp = _interp_mode_columns(radial_grid, bsupumnc[:, 1:], target_psi_n)
    b_sup_zeta_interp = _interp_mode_columns(radial_grid, bsupvmnc[:, 1:], target_psi_n)
    iota = _interp_1d(radial_grid, iota_grid[1:], target_psi_n)

    if mode_m.shape[0] != b_interp.shape[0]:
        raise ValueError("VMEC mode-number arrays do not match Fourier coefficient arrays")
    if mode_m[0] != 0 or mode_n[0] != 0:
        raise ValueError("expected the first VMEC mode to be (m,n)=(0,0)")

    b0 = float(b_interp[0])
    if b0 == 0.0:
        raise ValueError("VMEC mode (0,0) has zero magnetic-field strength")

    include = np.abs(b_interp / b0) >= float(min_bmn_to_load)
    if int(vmec_nyquist_option) == 1:
        include &= (np.abs(mode_m) < mpol) & (np.abs(mode_n / nfp) <= ntor)
    elif int(vmec_nyquist_option) != 2:
        raise ValueError("vmec_nyquist_option must be 1 or 2")
    include[0] = True

    return VmecSurface(
        path=wout_path,
        requested_psi_n=float(psi_n),
        psi_n=target_psi_n,
        nfp=nfp,
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        total_mode_count=int(mode_m.size),
        loaded_mode_count=int(np.count_nonzero(include)),
        iota=float(iota),
        m=jnp.asarray(mode_m[include], dtype=jnp.int32),
        n=jnp.asarray(np.rint(mode_n[include] / nfp).astype(np.int32), dtype=jnp.int32),
        b_cos=jnp.asarray(b_interp[include], dtype=jnp.float64),
        jacobian_cos=jnp.asarray(-g_interp[include], dtype=jnp.float64),
        b_sub_theta_cos=jnp.asarray(b_sub_theta_interp[include], dtype=jnp.float64),
        b_sub_zeta_cos=jnp.asarray(b_sub_zeta_interp[include], dtype=jnp.float64),
        b_sup_theta_cos=jnp.asarray(b_sup_theta_interp[include], dtype=jnp.float64),
        b_sup_zeta_cos=jnp.asarray(b_sup_zeta_interp[include], dtype=jnp.float64),
        b0=b0,
        psi_a_hat=psi_a_hat,
        phi_edge=float(phi[-1]),
        aminor_p=aminor_p,
        psi_p=None,
        transport_psi_scale=1.0,
    )


def _read_var(handle, name: str) -> np.ndarray:
    if name not in handle.variables:
        raise KeyError(f"missing variable {name!r} in VMEC file")
    return np.asarray(handle.variables[name].data)


def _read_scalar(handle, name: str) -> float:
    return float(np.asarray(_read_var(handle, name)).reshape(()))


def _read_modes(handle, name: str) -> np.ndarray:
    values = _read_var(handle, name).astype(np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected {name} to be a 2D array")
    return values.T


def _resolve_psi_n(psi_n_grid: np.ndarray, psi_n: float, option: int) -> float:
    if not 0.0 <= psi_n <= 1.0:
        raise ValueError("surface.psi_n must be between 0 and 1")
    if option == 0:
        return psi_n
    if option == 1:
        interior = psi_n_grid[1:]
        return float(interior[int(np.argmin(np.abs(interior - psi_n)))])
    if option == 2:
        return float(psi_n_grid[int(np.argmin(np.abs(psi_n_grid - psi_n)))])
    raise ValueError("vmec_radial_option must be 0, 1, or 2")


def _interp_1d(x: np.ndarray, values: np.ndarray, xq: float) -> float:
    return float(np.interp(float(xq), x, values))


def _interp_mode_columns(x: np.ndarray, values: np.ndarray, xq: float) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("expected a 2D `(mode, radius)` array")
    return np.asarray([np.interp(float(xq), x, row) for row in values], dtype=np.float64)
