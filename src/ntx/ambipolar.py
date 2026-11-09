"""Ambipolar radial electric field and the electron-root condition.

The ambipolarity condition ``sum_a q_a Gamma_a(E_r) = 0`` has one or three
roots per surface. This module convolves NTX monoenergetic ``D11`` tables with
Maxwellians for electrons and one ion species, finds every root by bracketing,
and attaches the implicit-function derivative to each. Everything is written in
the normalized-flux coordinate ``s``: ``er_s = -dPhi/ds`` in volts, so no minor
radius enters. Profiles are callables of ``rho = sqrt(s)`` in SI and eV.

The thermal convolution follows the NEOPAX transport-matrix convention: with
``K = (v / vth)^2`` on Gauss-Laguerre nodes ``K_i`` and weights ``w_i``,

``Gamma_a = -(n_a / sqrt(pi)) (m_a / q_a)^2 vth_a^3 sum_i w_i K_i^2 D11_i X_i``,

where ``D11_i`` is read from the NTX table at the collisionality
``nu_D(v_i) / v_i`` and field ``|E_psi| / v_i``, and the force is
``X = dln n/ds - q er_s / T + (K - 3/2) dln T/ds``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import erf

from ._interp import interp2d_at
from .grids import GridSpec
from .solver import solve_monoenergetic_scan
from .vmex_backend import surfaces_from_vmex_state

__all__ = ["AmbipolarPlasma", "ElectronRoot", "ambipolar_roots"]

E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31
M_PROTON = 1.67262192369e-27
EPS0 = 8.8541878128e-12


@dataclass(frozen=True)
class AmbipolarPlasma:
    """Profiles of a quasi-neutral electron plus single-ion plasma.

    ``density`` (electron, m^-3), ``te`` and ``ti`` (eV) are functions of
    ``rho = sqrt(s)``. ``ion_mass_number`` is the ion mass in proton masses.
    """

    density: Callable[[Array], Array]
    te: Callable[[Array], Array]
    ti: Callable[[Array], Array]
    ion_mass_number: float = 2.0
    ion_charge: float = 1.0
    ln_lambda: float = 17.0


def _local_plasma(plasma: AmbipolarPlasma, rho: float) -> dict[str, Array]:
    """Species ``(electron, ion)`` state and logarithmic ``s`` gradients at ``rho``."""

    def at_s(s):
        r = jnp.sqrt(s)
        ne = plasma.density(r)
        return jnp.stack([ne, ne / plasma.ion_charge]), jnp.stack([plasma.te(r), plasma.ti(r)])

    s0 = jnp.asarray(float(rho) ** 2)
    n, t_ev = at_s(s0)
    return {
        "n": n,
        "t": t_ev * E_CHARGE,
        "dlnn": jax.jacfwd(lambda s: at_s(s)[0])(s0) / n,
        "dlnt": jax.jacfwd(lambda s: at_s(s)[1])(s0) / t_ev,
        "q": E_CHARGE * jnp.asarray([-1.0, plasma.ion_charge]),
        "m": jnp.asarray([M_ELECTRON, M_PROTON * plasma.ion_mass_number]),
    }


def _nu_over_v(p: dict[str, Array], ln_lambda: float, v: Array) -> Array:
    """Pitch-angle scattering rate over speed, ``nu_D(v) / v``, shape ``(2, nv)``."""
    vth = jnp.sqrt(2.0 * p["t"] / p["m"])
    rows = []
    for a in range(2):
        total = jnp.zeros_like(v[a])
        for b in range(2):
            x = v[a] / vth[b]
            g = (erf(x) - 2.0 * x * jnp.exp(-(x**2)) / jnp.sqrt(jnp.pi)) / (2.0 * x**2)
            pref = (
                p["n"][b]
                * (p["q"][a] * p["q"][b]) ** 2
                * ln_lambda
                / (4.0 * jnp.pi * EPS0**2 * p["m"][a] ** 2)
            )
            total = total + pref * (erf(x) - g) / v[a] ** 4
        rows.append(total)
    return jnp.stack(rows)


@dataclass(frozen=True)
class _Axes:
    """Concrete table axes and per-node kinematics for one surface."""

    nu: np.ndarray  # log10 nu_hat axis
    e: np.ndarray  # log10(|E_psi_hat| + floor) axis
    e_hat: np.ndarray  # |E_psi_hat| nodes, the first is zero
    floor: float
    nu_hat: np.ndarray  # (2, n_energy) nu_D / v at each species and energy node
    v: np.ndarray  # (2, n_energy) speeds


def _species_flux(d11_log, axes, p, er_s, psi_p, k_nodes, k_weights) -> Array:
    """Particle flux per species, shape ``(*er_s.shape, 2)``; ``d11_log`` is ``log10 D11``."""
    er_s = jnp.asarray(er_s)
    out = []
    for a in range(2):
        vth = jnp.sqrt(2.0 * p["t"][a] / p["m"][a])
        e_hat = jnp.abs(er_s)[..., None] / (psi_p * axes.v[a])
        y = jnp.log10(e_hat + axes.floor)
        x = jnp.broadcast_to(jnp.log10(axes.nu_hat[a]), y.shape)
        d11 = 10.0 ** jax.vmap(
            lambda xq, yq: interp2d_at(axes.nu, axes.e, d11_log, xq, yq, method="pchip")
        )(x.reshape(-1), y.reshape(-1)).reshape(y.shape)
        force = (
            p["dlnn"][a] - p["q"][a] * er_s[..., None] / p["t"][a] + (k_nodes - 1.5) * p["dlnt"][a]
        )
        scale = p["n"][a] / jnp.sqrt(jnp.pi) * (p["m"][a] / p["q"][a]) ** 2 * vth**3
        out.append(-scale * jnp.sum(k_weights * k_nodes**2 * d11 * force, axis=-1))
    return jnp.stack(out, axis=-1)


def ambipolar_roots(
    residual: Callable[[Array, object], Array], er_grid: Array, data: object
) -> tuple[Array, Array, Array]:
    """All roots of a one-dimensional ``residual(er, data)`` on ``er_grid``.

    Returns ``(roots, valid, stable)``, each of shape ``(len(er_grid) - 1,)``:
    one root per grid interval with a sign change, found by bisection and then
    corrected by a single Newton step built from ``stop_gradient`` slopes, so
    the derivative with respect to ``data`` is the implicit-function derivative
    ``-R_data / R_er``. A root is stable where ``dR/dEr > 0``; intervals
    without a root are NaN.
    """
    values = residual(er_grid, data)
    valid = values[:-1] * values[1:] < 0.0
    data_const = jax.lax.stop_gradient(data)

    def bisect(_, state):
        a, b = state
        m = 0.5 * (a + b)
        left = residual(a, data_const) * residual(m, data_const) <= 0.0
        return jnp.where(left, a, m), jnp.where(left, m, b)

    a, b = jax.lax.fori_loop(0, 50, bisect, (er_grid[:-1], er_grid[1:]))
    e0 = 0.5 * (a + b)
    slope = jax.vmap(jax.grad(lambda e: residual(e, data_const)))(e0)
    safe = jnp.where(valid, slope, 1.0)
    roots = e0 - residual(e0, data) / safe
    return jnp.where(valid, roots, jnp.nan), valid, valid & (slope > 0.0)


class ElectronRoot:
    """Electron-root objective and diagnostics for a VMEX equilibrium.

    ``__call__(state, runtime)`` is a VMEX objective term. Per surface it
    returns ``max(R(E_target) / sum_a |Gamma_a| + margin, 0)`` with
    ``R = Gamma_i - Gamma_e``. ``R`` rises through its largest root, so
    ``R(E_target) < 0`` exactly when a root above ``E_target`` exists; the
    hinge asks for that with a margin. It is smooth in the geometry and needs
    no root finder. :meth:`roots` and :meth:`top_root` return the roots
    with implicit-function derivatives for diagnostics and root objectives.

    ``phiedge`` is the edge toroidal flux in webers. ``e_target`` and the root
    search ``[-e_max, e_max]`` are ``-dPhi/ds`` in volts; ``e_max`` also sets
    the field axis of the ``D11`` tables.
    """

    def __init__(
        self,
        rho,
        plasma: AmbipolarPlasma,
        phiedge: float,
        *,
        e_target: float = 0.0,
        e_max: float = 1.0e4,
        margin: float = 0.25,
        grid: GridSpec | None = None,
        mboz: int = 10,
        nboz: int = 10,
        n_energy: int = 10,
        n_nu: int = 7,
        n_field: int = 6,
        n_er: int = 161,
        nu_hat_range: tuple[float, float] = (3.0e-5, 3.0),
    ):
        """Fix the profiles, the energy quadrature and the ``D11`` table axes.

        The axes depend on the plasma only, never on the geometry, so they stay
        constant while the equilibrium is optimized and the tables remain
        differentiable. The collisionality axis spans the per-energy-node
        ``nu_D / v`` of both species, clipped to ``nu_hat_range``; the field
        axis starts at zero and then spans three decades above a floor of
        ``1e-3 * e_max`` (in volts), where ``log10`` stays smooth.
        """
        self.rho = np.atleast_1d(np.asarray(rho, dtype=float))
        self.plasma = plasma
        self.psi_p = abs(float(phiedge)) / (2.0 * np.pi)
        self.e_target, self.margin = float(e_target), float(margin)
        self.grid = GridSpec(9, 9, 40) if grid is None else grid
        self.mboz, self.nboz = int(mboz), int(nboz)
        self.er_grid = jnp.linspace(-e_max, e_max, n_er)
        nodes, weights = np.polynomial.laguerre.laggauss(n_energy)
        self.k_nodes, self.k_weights = jnp.asarray(nodes), jnp.asarray(weights)
        self.local = [_local_plasma(plasma, r) for r in self.rho]
        self.axes = []
        for p in self.local:
            v = jnp.sqrt(2.0 * p["t"][:, None] / p["m"][:, None] * self.k_nodes[None, :])
            nu_hat = np.asarray(_nu_over_v(p, plasma.ln_lambda, v))
            lo = max(np.log10(nu_hat.min()) - 0.3, np.log10(nu_hat_range[0]))
            hi = min(np.log10(nu_hat.max()) + 0.3, np.log10(nu_hat_range[1]))
            floor = 1.0e-3 * e_max / (self.psi_p * float(v.min()))
            e_hat = np.concatenate([[0.0], np.geomspace(3.0 * floor, 1.0e3 * floor, n_field - 1)])
            self.axes.append(
                _Axes(
                    nu=np.linspace(lo, hi, n_nu),
                    e=np.log10(e_hat + floor),
                    e_hat=e_hat,
                    floor=floor,
                    nu_hat=np.clip(nu_hat, 10.0**lo, 10.0**hi),
                    v=np.asarray(v),
                )
            )

    def tables(self, state, runtime) -> Array:
        """``log10 D11`` over ``(surface, nu, E)``, differentiable in the equilibrium."""
        surfaces = surfaces_from_vmex_state(
            state=state,
            static=runtime,
            indata=None,
            signgs=int(runtime.setup.signgs),
            s_values=tuple(float(r) ** 2 for r in self.rho),
            mboz=self.mboz,
            nboz=self.nboz,
            psi_p=self.psi_p,
        )
        tables = []
        for surface, ax in zip(surfaces, self.axes, strict=True):
            nu, eh = jnp.meshgrid(10.0 ** jnp.asarray(ax.nu), jnp.asarray(ax.e_hat), indexing="ij")
            d11 = solve_monoenergetic_scan(surface, self.grid, nu, epsi_hat=eh)["D11"]
            tables.append(jnp.log10(jnp.maximum(d11, 1.0e-300)))
        return jnp.stack(tables)

    def flux(self, er_s: Array, tables: Array, surface: int) -> Array:
        """Electron and ion particle flux at ``er_s`` on one surface, shape ``(..., 2)``."""
        return _species_flux(
            tables[surface],
            self.axes[surface],
            self.local[surface],
            er_s,
            self.psi_p,
            self.k_nodes,
            self.k_weights,
        )

    def residual(self, er_s: Array, tables: Array, surface: int) -> Array:
        """Ambipolar residual ``Gamma_i - Gamma_e`` (charge-weighted) on one surface."""
        return jnp.sum(self.local[surface]["q"] / E_CHARGE * self.flux(er_s, tables, surface), -1)

    def roots(self, tables: Array) -> tuple[Array, Array, Array]:
        """Roots, validity and stability, each shaped ``(surface, n_er - 1)``."""
        per_surface = [
            ambipolar_roots(partial(self._residual_of, surface=i), self.er_grid, tables)
            for i in range(len(self.rho))
        ]
        roots, valid, stable = (jnp.stack(x) for x in zip(*per_surface, strict=True))
        return roots, valid, stable

    def _residual_of(self, er_s: Array, tables: Array, *, surface: int) -> Array:
        """:meth:`residual` with the surface fixed, in the form the root finder takes."""
        return self.residual(er_s, tables, surface)

    def top_root(self, tables: Array) -> Array:
        """Largest stable root per surface (the electron root when one exists), NaN if none."""
        roots, _, stable = self.roots(tables)
        return jnp.nanmax(jnp.where(stable, roots, jnp.nan), axis=-1)

    def __call__(self, state, runtime) -> Array:
        """Hinge residual per surface, zero once a root above ``e_target`` exists."""
        tables = self.tables(state, runtime)
        target = jnp.asarray(self.e_target)
        out = []
        for i in range(len(self.rho)):
            flux = self.flux(target, tables, i)
            q = self.local[i]["q"] / E_CHARGE
            out.append(jnp.maximum(jnp.sum(q * flux) / jnp.sum(jnp.abs(flux)) + self.margin, 0.0))
        return jnp.stack(out)
