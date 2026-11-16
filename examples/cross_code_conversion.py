"""Convert NTX monoenergetic coefficients to SFINCS, YANCC and MONKES conventions.

NTX, SFINCS and the DKES family (YANCC, MONKES) solve the same monoenergetic equation and
print different numbers.  This script writes out every factor between them and prints a
table: coefficient, NTX, reference, converted reference, relative error, against a 2% gate.

Data: only what is committed.  The SFINCS rows read the six same-grid decks of
``docs/_static/owned_finite_beta_sfincs_jax_production_ladder.json`` (NTX coefficients and the
SFINCS RHSMode 3 ``transportMatrix`` with its scalars).  No YANCC or MONKES table is committed
here, so those conventions are shown as conversions of the same NTX numbers; the same formulas
reproduce the YANCC and MONKES rows of DKX's ``validation/independent_cross_code_v1.json``.
A live NTX solve on the sample DKES-format surface checks the inline factors against NTX's own.

Run:              python examples/cross_code_conversion.py
Runtime:          about 5 s on a laptop CPU.
"""

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ntx import (
    GridSpec,
    MonoenergeticCase,
    build_ntx_neopax_scan,
    enable_x64,
    load_dkes_surface,
    solve_monoenergetic,
)

enable_x64()

# Inputs
ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "docs" / "_static" / "owned_finite_beta_sfincs_jax_production_ladder.json"
SURFACE = ROOT / "tests" / "fixtures" / "sample_surface.ddkes2.data"
GATE = 0.02  # relative-error gate per coefficient
RHO_MIN = 0.2  # skips rho = 1/7, the stress radius: 2.07% at nu' = 0.01 (resolution audit JSON)
GRID = GridSpec(n_theta=9, n_zeta=9, n_xi=8)  # live check only
SQRT_PI = math.sqrt(math.pi)

# Conversions.  G, I: Boozer currents; iota; b00: B(0,0); psi_a: edge flux scale; den = G + iota I.
#  NTX D_ij -> SFINCS RHSMode 3 transportMatrix L_ij (hat units; Delta, nu_n, psiHat are SFINCS's):
#    L11 = -D11 * 8 den b00 psi_a^2 / (sqrt(pi) G^2)   (D11 carries two radial gradients)
#    L13 = -D13 * 4 b00 psi_a / (sqrt(pi) G)           (one radial gradient; Onsager: L31 = -L13)
#    L31 = +D31 * 4 b00 psi_a / (sqrt(pi) G)
#    L33 = +D33 * 2 b00 / (sqrt(pi) den)
#  The 1/sqrt(pi) and the 2, 4, 8 deconvolve SFINCS's single speed node (x0 = 1, w0 = e, so
#  w0 x0^2 exp(-x0^2) = 1) from the energy integral (4/sqrt(pi)) int x^2 exp(-x^2) D dx.
#  Collisionality: NTX nu_hat is nu/v in 1/m, equal to nu_n nuDHat(x0) (0.83603 at x0 = 1); SFINCS
#  nu_n = nuPrime b00 / den.  A solve at the same nu_hat is the same equation.
#  SFINCS L -> DKES-scaled D (the YANCC and MONKES tables): multiply by
#    G^2 b00 sqrt(pi) / (8 den dpsi_dr^2) for 11, G sqrt(pi) / (4 dpsi_dr) for 13 and 31,
#    den sqrt(pi) / (2 b00) for 33.  dpsi_dr is the reference code's radial convention: b00 on a
#    Boozer surface; YANCC uses Phi_edge rho / (pi a_minor) = 2 psi_a rN / a, SFINCS's
#    dpsiHat/drHat; DKX uses b00 r_hat.  D11 changes with the square of the ratio of two
#    conventions, D31 and D13 with the first power.
#  Beidler D* (as DKX): D11* = D11 / [pi/4 v_d^2 R0 / (v iota)],
#    D31* = D31 / [2/3 v_d R0 1.46 sqrt(eps_t) / (iota eps_t)], D33* = D33 / D33_Spitzer, with
#    v_d = Delta / (2 R0 b00), R0 = |G| / b00, eps_t = r / R0, r = sqrt(2 psiHat / b00) and NTX's
#    D33_Spitzer = 2 / (3 nu_hat) <B^2 / b00^2>.
#  Electric field: NTX epsi_hat = E_r / (dpsi/dr); SFINCS EStar = E_r / (v b00 iota eps_t) and
#    dPhiHatdpsiHat = 2 EStar iota b00 / (alpha Delta G).  Every committed comparison has E = 0.

# Live check: the inline factors equal the ones NTX stores for a surface.
surface = load_dkes_surface(SURFACE)
names = ("b_zeta", "b_theta", "iota", "b0", "psi_p")
g, i, iota, b00, psi_a = (float(getattr(surface, name)) for name in names)
den = g + iota * i
inline = {
    "L11/D11": 8 * den * b00 * psi_a**2 / (SQRT_PI * g**2),
    "L31/D31": 4 * b00 * psi_a / (SQRT_PI * g),
    "L33/D33": 2 * b00 / (SQRT_PI * den),
    "D/L 11": g**2 * b00 * SQRT_PI / (8 * den * b00**2),
    "D/L 31": g * SQRT_PI / (4 * b00),
    "D/L 33": den * SQRT_PI / (2 * b00),
}
scan = build_ntx_neopax_scan(
    lambda rho: surface,
    rho=jnp.array([0.5]),
    nu_v=jnp.array([1e-3]),
    Es=jnp.array([[0.0]]),
    Er=jnp.array([[0.0]]),
    drds=jnp.array([1.0]),
    grid=GRID,
    source_name="sample",
)
stored = (
    scan.fac_reference_to_sfincs_11,
    scan.fac_reference_to_sfincs_31,
    scan.fac_reference_to_sfincs_33,
    scan.fac_sfincs_to_dkes_11,
    scan.fac_sfincs_to_dkes_31,
    scan.fac_sfincs_to_dkes_33,
)
for (name, value), factor in zip(inline.items(), stored, strict=True):
    assert abs(value - float(factor[0])) <= 1e-12 * abs(value), name
live = solve_monoenergetic(surface, GRID, MonoenergeticCase(nu_hat=1e-3, er_hat=0.0))

# SFINCS decks: NTX D (stored when the ladder was run) against the SFINCS transportMatrix.
rows, ladder = [], json.loads(LADDER.read_text())
for deck in ladder["decks"]:
    if deck["rho"] < RHO_MIN:
        continue
    summary = deck["transport_summary"]
    c, d = summary["scalars"], summary["ntx_same_grid"]["ntx"]
    tm = np.array(summary["transportMatrix"])
    b0, den = c["B0OverBBar"], c["GHat"] + c["iota"] * c["IHat"]
    f11 = 8 * den * b0 * c["psiAHat"] ** 2 / (SQRT_PI * c["GHat"] ** 2)
    f31 = 4 * b0 * c["psiAHat"] / (SQRT_PI * c["GHat"])
    f33 = 2 * b0 / (SQRT_PI * den)
    label = f"rho={deck['rho']:.2f} nu'={deck['nu_prime']:g}"
    # (SFINCS entry, NTX value, SFINCS value, factor, sign) with L = sign * factor * D
    for name, ntx, ref, factor, sign in (
        ("L11", d["D11_raw"], tm[0, 0], f11, -1),
        ("L13", d["D13_raw"], tm[0, 1], f31, -1),
        ("L31", d["D31_raw"], tm[1, 0], f31, 1),
        ("L33", d["D33_raw"], tm[1, 1], f33, 1),
    ):
        converted = ref / (sign * factor)  # the SFINCS number expressed as an NTX D
        rows.append((label, name, ntx, ref, converted, abs(ntx - converted) / abs(converted)))

    # The same deck in Beidler D*, and as YANCC and MONKES would tabulate it (last deck printed).
    r = math.sqrt(2 * c["psiAHat"] * c["rN"] ** 2 / b0)  # r_hat of the surface
    r_major, iota_abs = abs(c["GHat"]) / b0, abs(c["iota"])
    eps_t, v_d = r / r_major, c["Delta"] / (2 * r_major * b0)
    d11_plateau = math.pi / 4 * v_d**2 * r_major / iota_abs
    d31_banana = 2 / 3 * v_d * r_major / (iota_abs * eps_t) * 1.46 * math.sqrt(eps_t)
    c11 = (c["Delta"] ** 2 / 4) / (b0 * r) ** 2 * c["GHat"] ** 2 / (b0 * den)
    c13 = c["Delta"] / 2 * abs(c["GHat"]) / b0 / (b0 * r)
    sign = math.copysign(1.0, c["iota"])
    star = {
        "D11*": SQRT_PI / 4 * c11 * f11 * d["D11_raw"] / d11_plateau,
        "D31*": -sign * SQRT_PI / 4 * c13 * f31 * d["D31_raw"] / d31_banana,
        "D33*": d["D33_raw"] / d["D33_spitzer"],
    }
    ratio = 2 * c["psiAHat"] * c["rN"] / c["aHat"] / (b0 * r)  # YANCC dpsi/dr over b00 r_hat
    dkes = {
        "D11": star["D11*"] / (8 * r_major * b0**2 * iota_abs / math.pi) / ratio**2,
        "D31": star["D31*"] / (1.5 * iota_abs * eps_t * b0 / (1.46 * math.sqrt(eps_t))) / ratio,
    }
    last = (label, star, dkes, ratio)

# Print
print("=== Summary ===")
print(
    f"  live NTX on the sample DKES surface at nu_hat=1e-3: D11={float(live.D11):.4e} "
    f"D31={float(live.D31):.4e} D33={float(live.D33):.4e}; inline factors match NTX's own to 1e-12"
)
columns = f"{'NTX D':>12} {'SFINCS L':>12} {'L as NTX D':>12} {'rel. err':>9}"
print(f"  {'deck':<20} {'entry':<5} {columns}")
for label, name, ntx, ref, converted, error in rows:
    verdict = "ok" if error <= GATE else "FAIL"
    values = f"{ntx:>12.4e} {ref:>12.4e} {converted:>12.4e} {error:>9.2e}"
    print(f"  {label:<20} {name:<5} {values}  {verdict}")
worst = max(row[-1] for row in rows)
print(f"  largest relative error {worst:.2e} against the {GATE:.0%} gate: "
      f"{'pass' if worst <= GATE else 'FAIL'}")
label, star, dkes, ratio = last
print(f"\n  {label} in other conventions:")
print(f"    Beidler D*: D11*={star['D11*']:.4f} D31*={star['D31*']:.4f} D33*={star['D33*']:.4f}")
print(f"    YANCC/MONKES DKES-scaled: D11={dkes['D11']:.4e} D31={dkes['D31']:.4e}")
print(f"    YANCC dpsi/dr over b00 r_hat = {ratio:.4f}: leaving it out changes D11 by "
      f"{abs(ratio**2 - 1):.1%} and D31 by {abs(ratio - 1):.1%}")
