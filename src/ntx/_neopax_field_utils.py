"""Small shared helpers for differentiable NEOPAX field builders."""

from __future__ import annotations

import jax.numpy as jnp

from .geometry import evaluate_boozer_modes


def _safe_divide(num, den):
    num_arr = jnp.asarray(num)
    den_arr = jnp.asarray(den)
    den_safe = jnp.where(jnp.abs(den_arr) > 0.0, den_arr, 1.0)
    return jnp.where(jnp.abs(den_arr) > 0.0, num_arr / den_safe, 0.0)


def _safe_reciprocal(values):
    arr = jnp.asarray(values)
    return jnp.where(jnp.abs(arr) > 0.0, 1.0 / arr, 0.0)


def _surface_b10(surface):
    mask = (jnp.asarray(surface.m) == 1) & (jnp.asarray(surface.n) == 0)
    idx = jnp.argmax(mask.astype(jnp.int32))
    b10 = jnp.where(mask.any(), jnp.asarray(surface.b_cos)[idx], 0.0)
    b0 = jnp.asarray(surface.b0 if surface.b0 is not None else surface.b_cos[0])
    return _safe_divide(b10, b0)


def _surface_bsqav(surface, *, ntheta: int = 31, nzeta: int = 31):
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / int(surface.nfp), int(nzeta), endpoint=False)
    theta_2d, zeta_2d = jnp.meshgrid(theta, zeta, indexing="ij")
    b, _, _ = evaluate_boozer_modes(surface, theta_2d, zeta_2d)
    b0 = jnp.asarray(surface.b0 if surface.b0 is not None else surface.b_cos[0], dtype=b.dtype)
    inv_bsq_mean = jnp.mean(jnp.square(_safe_divide(b0, b)))
    return _safe_reciprocal(inv_bsq_mean)


def _find_mode_index(xm_b, xn_b, *, m_value: int, n_value: int) -> int | None:
    matches = (jnp.asarray(xm_b) == int(m_value)) & (jnp.asarray(xn_b) == int(n_value))
    if not bool(jnp.any(matches)):
        return None
    return int(jnp.argmax(matches))
