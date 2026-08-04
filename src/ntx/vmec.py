"""VMEC `wout` helpers for NTX flux-surface inputs backed by `vmex`."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .geometry import VmecSurface


def load_vmec_surface(
    path: str | Path,
    *,
    psi_n: float,
    vmec_radial_option: int = 0,
    vmec_nyquist_option: int = 1,
    vmec_mode_convention: str = "reduced",
    min_bmn_to_load: float = 0.0,
) -> VmecSurface:
    """Load one VMEC flux surface from a `wout_*.nc` file.

    NTX now sources the VMEC data through `vmex` rather than through a
    separate local netCDF parser.
    """

    wout_path = Path(path).expanduser().resolve()
    if not wout_path.exists():
        raise FileNotFoundError(str(wout_path))

    try:
        from vmex import read_wout
    except (ImportError, ModuleNotFoundError):
        try:
            from vmex.api import read_wout
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModuleNotFoundError(
                "load_vmec_surface requires vmex. Install it with "
                "`pip install vmex`, `pip install -e ../vmex`, "
                "or `pip install git+https://github.com/uwplasma/VMEX.git`."
            ) from exc

    wout = read_wout(wout_path)
    if bool(wout.lasym):
        raise NotImplementedError("VMEC lasym=true inputs are not supported yet")

    nfp = int(wout.nfp)
    ns = int(wout.ns)
    mpol = int(wout.mpol)
    ntor = int(wout.ntor)
    phi = np.asarray(wout.phi, dtype=np.float64)
    if float(phi[-1]) == 0.0:
        raise ValueError("VMEC transport normalization produced dpsi_hat/dr_hat = 0")
    psi_n_grid = phi / float(phi[-1])
    iota_full = _iota_grid_from_wout(wout)
    # reshape(()) rather than float() directly: VMEC writes some scalars as
    # zero-dimensional arrays and others as one-element ones.
    aminor_p = float(np.asarray(wout.Aminor_p).reshape(()))

    base_mode_m = np.asarray(wout.xm, dtype=np.int32)
    base_mode_n = np.asarray(wout.xn, dtype=np.int32)
    coeff_mode_m = np.asarray(
        getattr(wout, "xm_nyq", wout.xm),
        dtype=np.int32,
    )
    coeff_mode_n = np.asarray(
        getattr(wout, "xn_nyq", wout.xn),
        dtype=np.int32,
    )

    bmnc = _mode_major(wout.bmnc)
    gmnc = _mode_major(wout.gmnc)
    bsubumnc = _mode_major(wout.bsubumnc)
    bsubvmnc = _mode_major(wout.bsubvmnc)
    bsupumnc = _mode_major(wout.bsupumnc)
    bsupvmnc = _mode_major(wout.bsupvmnc)

    if ns < 2:
        raise ValueError("VMEC input must contain at least two radial surfaces")

    psi_a_hat = float(phi[-1]) / (2.0 * np.pi)
    target_psi_n = _resolve_psi_n(psi_n_grid, float(psi_n), int(vmec_radial_option))
    if target_psi_n <= 0.0:
        raise ValueError("VMEC transport normalization requires surface.psi_n > 0")

    radial_grid = psi_n_grid[1:]
    selected_mode_m, selected_mode_n, selected_indices = _select_mode_set(
        base_mode_m,
        base_mode_n,
        coeff_mode_m,
        coeff_mode_n,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        option=int(vmec_nyquist_option),
        mode_convention=vmec_mode_convention,
    )
    b_interp = _interp_mode_columns(radial_grid, bmnc[selected_indices, 1:], target_psi_n)
    g_interp = _interp_mode_columns(radial_grid, gmnc[selected_indices, 1:], target_psi_n)
    b_sub_theta_interp = _interp_mode_columns(
        radial_grid, bsubumnc[selected_indices, 1:], target_psi_n
    )
    b_sub_zeta_interp = _interp_mode_columns(
        radial_grid, bsubvmnc[selected_indices, 1:], target_psi_n
    )
    b_sup_theta_interp = _interp_mode_columns(
        radial_grid, bsupumnc[selected_indices, 1:], target_psi_n
    )
    b_sup_zeta_interp = _interp_mode_columns(
        radial_grid, bsupvmnc[selected_indices, 1:], target_psi_n
    )
    iota = -_interp_1d(radial_grid, iota_full[1:], target_psi_n)

    if selected_mode_m.shape[0] != b_interp.shape[0]:
        raise ValueError("VMEC mode-number arrays do not match Fourier coefficient arrays")
    if selected_mode_m[0] != 0 or selected_mode_n[0] != 0:
        raise ValueError("expected the first VMEC mode to be (m,n)=(0,0)")

    b0 = float(b_interp[0])
    if b0 == 0.0:
        raise ValueError("VMEC mode (0,0) has zero magnetic-field strength")
    if aminor_p == 0.0:
        raise ValueError("VMEC input must provide a nonzero Aminor_p for transport normalization")

    r_n = float(np.sqrt(target_psi_n))
    r_hat = float(aminor_p * r_n)
    dpsi_hat_dr_hat = float(2.0 * psi_a_hat * r_n / aminor_p)
    if dpsi_hat_dr_hat == 0.0:
        raise ValueError("VMEC transport normalization produced dpsi_hat/dr_hat = 0")

    include = np.abs(b_interp / b0) >= float(min_bmn_to_load)
    include[0] = True

    return VmecSurface(
        path=wout_path,
        requested_psi_n=float(psi_n),
        psi_n=target_psi_n,
        nfp=nfp,
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        total_mode_count=int(selected_mode_m.size),
        loaded_mode_count=int(np.count_nonzero(include)),
        iota=float(iota),
        m=jnp.asarray(selected_mode_m[include], dtype=jnp.int32),
        n=jnp.asarray(
            np.rint(-selected_mode_n[include] / nfp).astype(np.int32),
            dtype=jnp.int32,
        ),
        b_cos=jnp.asarray(b_interp[include], dtype=jnp.float64),
        jacobian_cos=jnp.asarray(g_interp[include], dtype=jnp.float64),
        b_sub_theta_cos=jnp.asarray(b_sub_theta_interp[include], dtype=jnp.float64),
        b_sub_zeta_cos=jnp.asarray(b_sub_zeta_interp[include], dtype=jnp.float64),
        b_sup_theta_cos=jnp.asarray(b_sup_theta_interp[include], dtype=jnp.float64),
        b_sup_zeta_cos=jnp.asarray(b_sup_zeta_interp[include], dtype=jnp.float64),
        b0=b0,
        psi_a_hat=psi_a_hat,
        phi_edge=float(phi[-1]),
        r_n=r_n,
        r_hat=r_hat,
        dpsi_hat_dr_hat=dpsi_hat_dr_hat,
        dr_hat_dpsi_hat=float(1.0 / dpsi_hat_dr_hat),
        aminor_p=aminor_p,
        psi_p=None,
        transport_psi_scale=dpsi_hat_dr_hat,
    )


def _mode_major(values) -> np.ndarray:
    """Transpose a vmex `(radius, mode)` array to mode-major order.

    NTX indexes harmonics first throughout; converting once at the boundary
    keeps that convention from leaking into the loaders.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("expected a 2D `(radius, mode)` array from vmex")
    return array.T


def _iota_grid_from_wout(wout) -> np.ndarray:
    """Read the rotational transform under whichever name the file uses.

    VMEC outputs differ in whether iota is stored full-mesh or half-mesh and
    under which spelling.
    """
    for name in ("iota_f", "iotaf", "iotas"):
        if hasattr(wout, name):
            values = np.asarray(getattr(wout, name), dtype=np.float64)
            if values.size > 0:
                return values
    raise ValueError("vmex wout data does not provide an iota profile")


def _resolve_psi_n(psi_n_grid: np.ndarray, psi_n: float, option: int) -> float:
    """Map a requested normalized flux onto a radial index or interpolation.

    `option` selects the convention: nearest stored surface, or interpolation
    between them. The bounds check is first because a psi_n outside [0, 1] is a
    configuration error, not something to clamp.
    """
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
    """Quadratic interpolation of a profile at one radius."""
    return float(_interpolated_value(x, values, float(xq), order=2))


def _interp_mode_columns(x: np.ndarray, values: np.ndarray, xq: float) -> np.ndarray:
    """Interpolate every harmonic column to one radius."""
    if values.ndim != 2:
        raise ValueError("expected a 2D `(mode, radius)` array")
    return np.asarray(
        [_interpolated_value(x, row, float(xq), order=2) for row in values],
        dtype=np.float64,
    )


def _interpolated_value(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    xq: float,
    *,
    order: int,
) -> float:
    """Polynomial interpolation of the requested order at a query point.

    Shapes are validated up front: a transposed input would otherwise
    interpolate along the wrong axis and return a plausible wrong number.
    """
    if x_nodes.ndim != 1 or y_nodes.ndim != 1:
        raise ValueError("interpolation inputs must be 1D")
    if x_nodes.shape[0] != y_nodes.shape[0]:
        raise ValueError("interpolation nodes and values must have the same length")
    if x_nodes.shape[0] == 0:
        raise ValueError("interpolation requires at least one node")
    if x_nodes.shape[0] <= order:
        return float(np.interp(xq, x_nodes, y_nodes))

    js = int(np.argmin(np.abs(x_nodes - xq)))
    start = js - (order - (order % 2)) // 2
    start = max(0, min(start, x_nodes.shape[0] - (order + 1)))
    indices = np.arange(start, start + order + 1, dtype=np.int32)
    x_sel = x_nodes[indices]
    y_sel = y_nodes[indices]

    weights = np.ones((order + 1,), dtype=np.float64)
    for i in range(order + 1):
        for j in range(order + 1):
            if i == j:
                continue
            weights[i] *= (xq - x_sel[j]) / (x_sel[i] - x_sel[j])
    return float(np.dot(y_sel, weights))


def _select_mode_set(
    base_mode_m: np.ndarray,
    base_mode_n: np.ndarray,
    coeff_mode_m: np.ndarray,
    coeff_mode_n: np.ndarray,
    *,
    nfp: int,
    mpol: int,
    ntor: int,
    option: int,
    mode_convention: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose which harmonics to keep, by option and mode convention.

    A reduced convention keeps only the harmonics the coefficient arrays
    actually populate, so the surface does not carry modes that are structurally
    zero.
    """
    selected_indices: np.ndarray
    if option == 1:
        if mode_convention == "reduced":
            if coeff_mode_m.shape[0] < base_mode_m.shape[0]:
                raise ValueError(
                    "VMEC Nyquist coefficient table is smaller than the reduced mode table"
                )
            selected_indices = np.arange(base_mode_m.size, dtype=np.int32)
            return base_mode_m, base_mode_n, selected_indices
        if mode_convention != "filtered_nyquist":
            raise ValueError("vmec_mode_convention must be 'reduced' or 'filtered_nyquist'")
        include = (np.abs(coeff_mode_m) < int(mpol)) & (
            np.abs(coeff_mode_n / float(nfp)) <= float(ntor)
        )
        selected_indices = np.nonzero(include)[0].astype(np.int32)
        return coeff_mode_m[include], coeff_mode_n[include], selected_indices
    if option == 2:
        selected_indices = np.arange(coeff_mode_m.size, dtype=np.int32)
        return coeff_mode_m, coeff_mode_n, selected_indices
    raise ValueError("vmec_nyquist_option must be 1 or 2")
