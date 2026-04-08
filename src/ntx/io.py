"""Input/output helpers for simple Boozer-surface files."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .geometry import BoozerSurface


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
        b0=b0,
        stellarator_symmetric=b_sin is None,
    )


def write_result_jsonable(result) -> dict[str, float]:
    """Convert a result to a small JSON-serializable mapping."""

    return result.as_dict()
