"""Direct VMEC-surface builders backed by `vmex`."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .geometry import VmecSurface


def surface_from_vmex_vmec_wout(
    wout,
    *,
    s: float,
    source_path: str | Path | None = None,
    min_bmn_to_load: float = 0.0,
) -> VmecSurface:
    """Build a VMEC harmonic surface from an in-memory `vmex` wout object."""

    if not 0.0 <= float(s) <= 1.0:
        raise ValueError("s must be between 0 and 1")

    ns = int(wout.ns)
    if ns < 2:
        raise ValueError("VMEC input must contain at least two radial surfaces")

    s_full = np.linspace(0.0, 1.0, ns, dtype=np.float64)
    hs = 1.0 / (ns - 1)
    s_half = s_full[:-1] + 0.5 * hs

    xm_nyq = np.asarray(wout.xm_nyq, dtype=np.int32)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=np.int32)
    phi = np.asarray(wout.phi, dtype=np.float64)

    bmnc = _interp_mode_columns(s_half, np.asarray(wout.bmnc, dtype=np.float64)[1:, :], s)
    gmnc = _interp_mode_columns(s_half, np.asarray(wout.gmnc, dtype=np.float64)[1:, :], s)
    bsupumnc = _interp_mode_columns(s_half, np.asarray(wout.bsupumnc, dtype=np.float64)[1:, :], s)
    bsupvmnc = _interp_mode_columns(s_half, np.asarray(wout.bsupvmnc, dtype=np.float64)[1:, :], s)
    bsubumnc = _interp_mode_columns(s_half, np.asarray(wout.bsubumnc, dtype=np.float64)[1:, :], s)
    bsubvmnc = _interp_mode_columns(s_half, np.asarray(wout.bsubvmnc, dtype=np.float64)[1:, :], s)
    iota = -float(_interp_profile(s_full, np.asarray(wout.iotaf, dtype=np.float64), s))

    b0 = float(np.max(np.abs(bmnc)))
    if b0 == 0.0:
        raise ValueError("selected VMEC surface has zero magnetic-field strength")

    include = np.abs(bmnc) >= float(min_bmn_to_load) * b0
    zero_mode = (xm_nyq == 0) & (xn_nyq == 0)
    if np.any(zero_mode):
        include[np.argmax(zero_mode)] = True

    aminor_p = float(np.asarray(wout.Aminor_p, dtype=np.float64).reshape(()))
    r_n = float(np.sqrt(float(s)))
    r_hat = float(aminor_p * r_n)
    psi_a_hat = float(phi[-1]) / (2.0 * np.pi)

    resolved_path = Path(
        source_path if source_path is not None else getattr(wout, "path", "vmex_wout")
    ).expanduser()

    return VmecSurface(
        path=resolved_path,
        requested_psi_n=float(s),
        psi_n=float(s),
        nfp=int(wout.nfp),
        ns=ns,
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        total_mode_count=int(xm_nyq.size),
        loaded_mode_count=int(np.count_nonzero(include)),
        iota=iota,
        m=jnp.asarray(xm_nyq[include], dtype=jnp.int32),
        n=jnp.asarray(np.rint(-xn_nyq[include] / int(wout.nfp)).astype(np.int32), dtype=jnp.int32),
        b_cos=jnp.asarray(bmnc[include], dtype=jnp.float64),
        jacobian_cos=jnp.asarray(gmnc[include], dtype=jnp.float64),
        b_sub_theta_cos=jnp.asarray(bsubumnc[include], dtype=jnp.float64),
        b_sub_zeta_cos=jnp.asarray(bsubvmnc[include], dtype=jnp.float64),
        b_sup_theta_cos=jnp.asarray(bsupumnc[include], dtype=jnp.float64),
        b_sup_zeta_cos=jnp.asarray(bsupvmnc[include], dtype=jnp.float64),
        b0=b0,
        psi_a_hat=psi_a_hat,
        phi_edge=float(phi[-1]),
        r_n=r_n,
        r_hat=r_hat,
        dpsi_hat_dr_hat=1.0,
        dr_hat_dpsi_hat=1.0,
        aminor_p=aminor_p,
        psi_p=None,
        transport_psi_scale=1.0,
    )


def surface_from_vmex_vmec_wout_file(
    path: str | Path,
    *,
    s: float,
    min_bmn_to_load: float = 0.0,
) -> VmecSurface:
    """Build a VMEC harmonic surface from a `wout` file through `vmex`."""

    try:
        from vmex import read_wout
    except (ImportError, ModuleNotFoundError):
        try:
            from vmex.api import read_wout
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModuleNotFoundError(
                "surface_from_vmex_vmec_wout_file requires vmex. "
                "Install it with `pip install vmex`."
            ) from exc

    wout_path = Path(path).expanduser().resolve()
    wout = read_wout(wout_path)
    return surface_from_vmex_vmec_wout(
        wout,
        s=s,
        source_path=wout_path,
        min_bmn_to_load=min_bmn_to_load,
    )


def _interp_mode_columns(x: np.ndarray, values: np.ndarray, xq: float) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("expected a 2D `(radius, mode)` array")
    return np.asarray(_interp_profile(x, values, xq), dtype=np.float64)


def _interp_profile(x: np.ndarray, values: np.ndarray, xq):
    import interpax

    return interpax.interp1d(
        jnp.asarray(xq, dtype=jnp.float64),
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(values, dtype=jnp.float64),
        method="cubic",
        extrap=True,
    )
