"""NTX-to-NEOPAX dataclasses and constants."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array, tree_util

NEOPAX_SCAN_FORMAT_VERSION = 2
D33_MODES = frozenset({"spitzer", "raw", "conductivity_difference"})


@dataclass(frozen=True)
class NeopaxScan:
    """Monoenergetic scan data shaped for NEOPAX."""

    rho: Array
    nu_v: Array
    Er: Array
    Es: Array
    drds: Array
    D11: Array
    D13: Array
    D33: Array
    D33_spitzer: Array | None = None
    D31: Array | None = None
    Er_tilde: Array | None = None
    Er_to_Ertilde: Array | None = None
    dr_tildedr: Array | None = None
    dr_tildeds: Array | None = None
    a_b: float | None = None
    psia: float | None = None
    b00: Array | None = None
    r00: Array | None = None
    boozer_i: Array | None = None
    boozer_g: Array | None = None
    iota: Array | None = None
    fac_reference_to_sfincs_11: Array | None = None
    fac_reference_to_sfincs_31: Array | None = None
    fac_reference_to_sfincs_33: Array | None = None
    fac_monkes_to_sfincs_11: Array | None = None
    fac_monkes_to_sfincs_31: Array | None = None
    fac_monkes_to_sfincs_33: Array | None = None
    fac_sfincs_to_dkes_11: Array | None = None
    fac_sfincs_to_dkes_31: Array | None = None
    fac_sfincs_to_dkes_33: Array | None = None
    fac_dkes_to_d11star: Array | None = None
    fac_dkes_to_d31star: Array | None = None
    fac_dkes_to_d33star: Array | None = None
    source_name: str | None = None


tree_util.register_dataclass(
    NeopaxScan,
    data_fields=(
        "rho",
        "nu_v",
        "Er",
        "Es",
        "drds",
        "D11",
        "D13",
        "D33",
        "D33_spitzer",
        "D31",
        "Er_tilde",
        "Er_to_Ertilde",
        "dr_tildedr",
        "dr_tildeds",
        "a_b",
        "psia",
        "b00",
        "r00",
        "boozer_i",
        "boozer_g",
        "iota",
        "fac_reference_to_sfincs_11",
        "fac_reference_to_sfincs_31",
        "fac_reference_to_sfincs_33",
        "fac_monkes_to_sfincs_11",
        "fac_monkes_to_sfincs_31",
        "fac_monkes_to_sfincs_33",
        "fac_sfincs_to_dkes_11",
        "fac_sfincs_to_dkes_31",
        "fac_sfincs_to_dkes_33",
        "fac_dkes_to_d11star",
        "fac_dkes_to_d31star",
        "fac_dkes_to_d33star",
    ),
    meta_fields=("source_name",),
)


@dataclass(frozen=True)
class NeopaxMonoenergeticArrays:
    """Pure-array NEOPAX mapping payload for differentiable imported workflows."""

    a_b: Array
    rho: Array
    nu_log: Array
    Er_list: Array
    D11_log: Array
    D13: Array
    D33: Array


tree_util.register_dataclass(
    NeopaxMonoenergeticArrays,
    data_fields=("a_b", "rho", "nu_log", "Er_list", "D11_log", "D13", "D33"),
    meta_fields=(),
)


@dataclass(frozen=True)
class DifferentiableNeopaxField:
    """JAX-safe field payload compatible with the NEOPAX flux routines."""

    n_r: int
    a_b: Array
    Psia_value: Array
    rho_grid: Array
    rho_grid_half: Array
    r_grid: Array
    r_grid_half: Array
    dr: Array
    Vprime: Array
    Vprime_half: Array
    overVprime: Array
    epsilon_t: Array
    B0: Array
    B_10: Array
    enlogation: Array
    iota: Array
    R0: Array
    B0prime: Array
    curvature: Array
    G_PS: Array
    sqrtg00_value: Array
    Bsqav: Array
    I_value: Array
    G_value: Array


tree_util.register_dataclass(
    DifferentiableNeopaxField,
    data_fields=(
        "a_b",
        "Psia_value",
        "rho_grid",
        "rho_grid_half",
        "r_grid",
        "r_grid_half",
        "dr",
        "Vprime",
        "Vprime_half",
        "overVprime",
        "epsilon_t",
        "B0",
        "B_10",
        "enlogation",
        "iota",
        "R0",
        "B0prime",
        "curvature",
        "G_PS",
        "sqrtg00_value",
        "Bsqav",
        "I_value",
        "G_value",
    ),
    meta_fields=("n_r",),
)
