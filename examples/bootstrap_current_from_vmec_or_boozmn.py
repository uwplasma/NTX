#!/usr/bin/env python3
"""Compute a bootstrap-current profile with NTX from VMEC or Boozer input.

This example shows two geometry entry points:

1. start from a VMEC `wout` file and build each NTX surface with `vmec_jax`
2. if a Boozer `boozmn` file already exists, load it directly with `booz_xform_jax`

If NEOPAX is available, the script builds a small monoenergetic database with
NTX, maps it into the NEOPAX database conventions, and computes a bootstrap-
current profile. When local reference files are available, the script also:

- compares that bootstrap-current profile against a reference HDF5 database
- compares one monoenergetic coefficient matrix against a local SFINCS-JAX
  `transportMatrix` output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    MonoenergeticCase,
    build_ntx_neopax_scan_from_surfaces,
    load_boozmn_surface,
    load_neopax_reference_scan,
    solve_monoenergetic,
    surface_from_vmec_jax_vmec_wout_file,
    to_neopax_monoenergetic,
)
from ntx._checkout_paths import (  # noqa: E402
    find_reference_root,
    find_neopax_root,
    find_sfincs_jax_root,
)
from ntx.neopax import NeopaxScan  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, default=None, help="VMEC `wout` file.")
    parser.add_argument("--boozmn", type=Path, default=None, help="Boozer `boozmn` file.")
    parser.add_argument(
        "--surface-source",
        choices=("auto", "vmec", "boozmn"),
        default="auto",
        help="Which NTX geometry lane to use when both `--wout` and `--boozmn` are available.",
    )
    parser.add_argument(
        "--reference-database",
        type=Path,
        default=None,
        help="Reference HDF5 monoenergetic database used for the bootstrap-current comparison.",
    )
    parser.add_argument(
        "--sfincs-output",
        type=Path,
        default=None,
        help="Optional local SFINCS-JAX `sfincsOutput_jax.h5` for a monoenergetic cross-check.",
    )
    parser.add_argument("--rho", type=float, default=0.5, help="Surface label `rho = sqrt(s)`.")
    parser.add_argument(
        "--nu-hat",
        type=float,
        default=1.0e-5,
        help="NTX collisionality used for the single-surface coefficient report.",
    )
    parser.add_argument(
        "--epsi-hat",
        type=float,
        default=0.0,
        help="NTX electric-field normalization used for the single-surface coefficient report.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NTHETA", "NZETA", "NXI"),
        default=(25, 25, 64),
        help="NTX grid for the single-surface solve.",
    )
    parser.add_argument(
        "--database-grid",
        type=int,
        nargs=3,
        metavar=("NTHETA", "NZETA", "NXI"),
        default=(13, 17, 16),
        help="NTX grid used to build the NEOPAX database subset.",
    )
    parser.add_argument(
        "--sfincs-grid",
        type=int,
        nargs=3,
        metavar=("NTHETA", "NZETA", "NXI"),
        default=(9, 17, 13),
        help="NTX grid for the SFINCS-JAX cross-check point.",
    )
    parser.add_argument(
        "--nu-stride",
        type=int,
        default=1,
        help="Stride through the reference collisionality grid when building the NTX database.",
    )
    parser.add_argument(
        "--er-stride",
        type=int,
        default=1,
        help="Stride through the reference electric-field grid when building the NTX database.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "docs" / "_static" / "bootstrap_current_from_vmec_or_boozmn",
        help="Prefix for the generated figure and JSON summary.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip the NTX + NEOPAX bootstrap-current workflow.",
    )
    parser.add_argument(
        "--skip-sfincs",
        action="store_true",
        help="Skip the optional SFINCS-JAX monoenergetic comparison.",
    )
    return parser.parse_args()


def _resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    neopax_root = find_neopax_root()
    sfincs_root = find_sfincs_jax_root()
    if args.wout is None and neopax_root is not None:
        candidate = neopax_root / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
        if candidate.exists():
            args.wout = candidate
    if args.boozmn is None and neopax_root is not None:
        candidate = neopax_root / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
        if candidate.exists():
            args.boozmn = candidate
    if args.reference_database is None and neopax_root is not None:
        candidate = neopax_root / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
        if candidate.exists():
            args.reference_database = candidate
    if args.sfincs_output is None and sfincs_root is not None:
        candidate = (
            sfincs_root
            / "tests"
            / "scaled_example_suite_fast_cpu_rtwindow_v4_part1"
            / "monoenergetic_geometryScheme5_netCDF"
            / "last_success"
            / "sfincsOutput_jax.h5"
        )
        if candidate.exists():
            args.sfincs_output = candidate
    return args


def _subset_reference_scan(scan: NeopaxScan, *, nu_stride: int, er_stride: int) -> NeopaxScan:
    nu_idx = _subsample_indices(scan.nu_v.shape[0], nu_stride)
    er_idx = _subsample_indices(scan.Er.shape[1], er_stride)
    return NeopaxScan(
        rho=scan.rho,
        nu_v=scan.nu_v[nu_idx],
        Er=scan.Er[:, er_idx],
        Es=scan.Es[:, er_idx],
        drds=scan.drds,
        D11=scan.D11[:, nu_idx][:, :, er_idx],
        D13=scan.D13[:, nu_idx][:, :, er_idx],
        D33=scan.D33[:, nu_idx][:, :, er_idx],
        D31=scan.D31[:, nu_idx][:, :, er_idx] if scan.D31 is not None else None,
        Er_tilde=scan.Er_tilde[er_idx] if scan.Er_tilde is not None else None,
        Er_to_Ertilde=scan.Er_to_Ertilde[:, er_idx] if scan.Er_to_Ertilde is not None else None,
        dr_tildedr=scan.dr_tildedr,
        dr_tildeds=scan.dr_tildeds,
        a_b=scan.a_b,
        psia=scan.psia,
        b00=scan.b00,
        r00=scan.r00,
        boozer_i=scan.boozer_i,
        boozer_g=scan.boozer_g,
        iota=scan.iota,
        fac_reference_to_sfincs_11=scan.fac_reference_to_sfincs_11,
        fac_reference_to_sfincs_31=scan.fac_reference_to_sfincs_31,
        fac_reference_to_sfincs_33=scan.fac_reference_to_sfincs_33,
        fac_sfincs_to_dkes_11=scan.fac_sfincs_to_dkes_11,
        fac_sfincs_to_dkes_31=scan.fac_sfincs_to_dkes_31,
        fac_sfincs_to_dkes_33=scan.fac_sfincs_to_dkes_33,
        fac_dkes_to_d11star=scan.fac_dkes_to_d11star,
        fac_dkes_to_d31star=scan.fac_dkes_to_d31star,
        fac_dkes_to_d33star=scan.fac_dkes_to_d33star,
        source_name=scan.source_name,
    )


def _subsample_indices(length: int, stride: int) -> np.ndarray:
    indices = np.arange(0, length, max(stride, 1), dtype=int)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return np.unique(indices)


def _surface_loader(wout: Path | None, boozmn: Path | None, *, preferred: str = "auto"):
    if preferred not in {"auto", "vmec", "boozmn"}:
        raise ValueError(f"unsupported surface source {preferred!r}")
    if preferred in {"auto", "vmec"} and wout is not None:
        def loader(rho_value: float):
            return surface_from_vmec_jax_vmec_wout_file(wout, s=float(rho_value**2))

        return loader, "vmec_jax"
    if preferred in {"auto", "boozmn"} and boozmn is not None:
        def loader(rho_value: float):
            return load_boozmn_surface(boozmn, rho=float(rho_value)).surface

        return loader, "boozmn"
    raise ValueError("need either `--wout` or `--boozmn`")


def _solve_single_surface(
    *,
    wout: Path | None,
    boozmn: Path | None,
    surface_source: str,
    rho: float,
    nu_hat: float,
    epsi_hat: float,
    grid_tuple: tuple[int, int, int],
):
    loader, mode = _surface_loader(wout, boozmn, preferred=surface_source)
    surface = loader(rho)
    result = solve_monoenergetic(
        surface,
        GridSpec(
            n_theta=int(grid_tuple[0]),
            n_zeta=int(grid_tuple[1]),
            n_xi=int(grid_tuple[2]),
        ),
        MonoenergeticCase(nu_hat=nu_hat, epsi_hat=epsi_hat),
    )
    return mode, surface, result


def _build_species_and_field(neopax_module, *, wout: Path, boozmn: Path, n_radial: int = 51):
    grid = neopax_module.Grid.create_standard(n_radial, 48, 3)
    field = neopax_module.Field.read_vmec_booz(n_radial, str(wout), str(boozmn))

    te0 = 17.8e3
    teb = 0.7e3
    ne0 = 4.21e20
    neb = 0.6e20
    deuterium_ratio = 0.5
    tritium_ratio = 0.5

    r = field.r_grid
    te = (te0 - teb) * (1.0 - (r / field.a_b) ** 2) + teb
    ne = (ne0 - neb) * (1.0 - (r / field.a_b) ** 10) + neb
    ti = (te0 - teb) * (1.0 - (r / field.a_b) ** 2) + teb
    nd = deuterium_ratio * ((ne0 - neb) * (1.0 - (r / field.a_b) ** 10) + neb)
    nt = tritium_ratio * ((ne0 - neb) * (1.0 - (r / field.a_b) ** 10) + neb)
    er = jnp.zeros_like(r)

    temperature = jnp.stack([te, ti, ti])
    density = jnp.stack([ne, nd, nt])
    mass = jnp.array([1.0 / 1836.15267343, 2.0, 3.0])
    charge = jnp.array([-1.0, 1.0, 1.0])
    n_edge = jnp.array([neb, deuterium_ratio * neb, tritium_ratio * neb])
    t_edge = jnp.array([teb, teb, teb])

    species = neopax_module.Species(
        3,
        n_radial,
        grid.species_indeces,
        mass,
        charge,
        temperature,
        density,
        er,
        field.r_grid,
        field.r_grid_half,
        field.dr,
        field.Vprime_half,
        field.overVprime,
        n_edge,
        t_edge,
    )
    return grid, field, species


def _bootstrap_current_profile(
    neopax_module,
    *,
    field,
    grid,
    species,
    database,
) -> tuple[np.ndarray, np.ndarray]:
    _lij, _gamma, _q, upar = neopax_module.get_Neoclassical_Fluxes(species, grid, field, database)
    current = neopax_module._constants.elementary_charge * jnp.sum(
        species.charge[:, None] * upar,
        axis=0,
    )
    return np.asarray(field.rho_grid, dtype=np.float64), np.asarray(current, dtype=np.float64)


def _load_reference_to_sfincs_factors(path: Path, rho: float) -> tuple[float, float, float] | None:
    with h5py.File(path, "r") as handle:
        rho_grid = np.asarray(handle["rho"], dtype=np.float64)
        rho_index = int(np.argmin(np.abs(rho_grid - rho)))
        names = [
            ("Fac_REFERENCE_TO_SFINCS_11", "Fac_REFERENCE_TO_SFINCS_11"),
            ("Fac_REFERENCE_TO_SFINCS_31", "Fac_REFERENCE_TO_SFINCS_31"),
            ("Fac_REFERENCE_TO_SFINCS_33", "Fac_REFERENCE_TO_SFINCS_33"),
        ]
        values: list[float] = []
        for aliases in names:
            for name in aliases:
                if name in handle:
                    values.append(float(np.asarray(handle[name])[rho_index]))
                    break
            else:
                return None
    return tuple(values)


def _compare_against_sfincs(
    *,
    sfincs_output: Path,
    boozmn: Path | None,
    wout: Path | None,
    surface_source: str,
    rho: float,
    reference_database: Path | None,
    grid_tuple: tuple[int, int, int],
) -> dict[str, object] | None:
    if reference_database is None:
        return None
    factors = _load_reference_to_sfincs_factors(reference_database, rho)
    if factors is None:
        return None
    with h5py.File(sfincs_output, "r") as handle:
        transport_matrix = np.asarray(handle["transportMatrix"], dtype=np.float64)
        nu_hat = float(np.asarray(handle["nuPrime"]).reshape(()))
        epsi_hat = float(np.asarray(handle["EStar"]).reshape(()))

    _mode, _surface, result = _solve_single_surface(
        wout=wout,
        boozmn=boozmn,
        surface_source=surface_source,
        rho=rho,
        nu_hat=nu_hat,
        epsi_hat=epsi_hat,
        grid_tuple=grid_tuple,
    )
    fac11, fac31, fac33 = factors
    ntx_matrix = np.array(
        [
            [result.D11 * fac11, -result.D13 * fac31],
            [result.D31 * fac31, result.D33 * fac33],
        ],
        dtype=np.float64,
    )
    rel = np.max(
        np.abs(ntx_matrix - transport_matrix)
        / np.maximum(np.abs(transport_matrix), 1.0e-12)
    )
    return {
        "nu_hat": nu_hat,
        "epsi_hat": epsi_hat,
        "transport_matrix": transport_matrix,
        "ntx_matrix": ntx_matrix,
        "max_relative_error": float(rel),
    }


def _reference_point_from_scan(
    scan: NeopaxScan,
    *,
    rho: float,
    nu_hat: float,
    epsi_hat: float,
) -> dict[str, float]:
    rho_index = int(np.argmin(np.abs(np.asarray(scan.rho) - rho)))
    nu_index = int(np.argmin(np.abs(np.asarray(scan.nu_v) - nu_hat)))
    es_index = int(np.argmin(np.abs(np.asarray(scan.Es[rho_index]) - epsi_hat)))
    values = {
        "rho": float(np.asarray(scan.rho)[rho_index]),
        "nu_hat": float(np.asarray(scan.nu_v)[nu_index]),
        "epsi_hat": float(np.asarray(scan.Es[rho_index])[es_index]),
        "D11": float(np.asarray(scan.D11)[rho_index, nu_index, es_index]),
        "D13": float(np.asarray(scan.D13)[rho_index, nu_index, es_index]),
        "D33": float(np.asarray(scan.D33)[rho_index, nu_index, es_index]),
    }
    d31 = scan.D31
    values["D31"] = (
        float(np.asarray(d31)[rho_index, nu_index, es_index]) if d31 is not None else values["D13"]
    )
    return values


def _plot_summary(
    *,
    output_prefix: Path,
    bootstrap_payload: dict[str, np.ndarray] | None,
    point_payload: dict[str, float],
    point_reference_payload: dict[str, float] | None,
    sfincs_payload: dict[str, object] | None,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True, dpi=160)

    ax = axes[0]
    coeff_labels = ["D11", "D13", "D31", "D33"]
    ntx_values = np.array(
        [
            point_payload["D11"],
            point_payload["D13"],
            point_payload["D31"],
            point_payload["D33"],
        ],
        dtype=np.float64,
    )
    x = np.arange(len(coeff_labels))
    width = 0.34
    ax.bar(x - 0.5 * width, ntx_values, width=width, label="NTX", color="#1f77b4")
    if point_reference_payload is not None:
        ref_values = np.array(
            [
                point_reference_payload["D11"],
                point_reference_payload["D13"],
                point_reference_payload["D31"],
                point_reference_payload["D33"],
            ],
            dtype=np.float64,
        )
        ax.bar(x + 0.5 * width, ref_values, width=width, label="Reference", color="#d62728")
        max_rel = np.max(np.abs(ntx_values - ref_values) / np.maximum(np.abs(ref_values), 1.0e-12))
        title = f"Pointwise Coefficients (max rel. error = {max_rel:.2e})"
    else:
        title = "NTX Monoenergetic Coefficients"
    ax.set_xticks(x, coeff_labels)
    ax.set_title(title)
    ax.set_ylabel("Coefficient value")
    ax.grid(alpha=0.25, axis="y")
    note = (
        f"rho={point_payload['rho']:.3f}\n"
        f"nu_hat={point_payload['nu_hat']:.2e}\n"
        f"epsi_hat={point_payload['epsi_hat']:.2e}"
    )
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left", fontsize=9)
    if point_reference_payload is not None:
        ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    if bootstrap_payload is None:
        ax.text(0.5, 0.5, "Bootstrap-current comparison skipped", ha="center", va="center")
        ax.axis("off")
    else:
        ax.plot(
            bootstrap_payload["rho"],
            bootstrap_payload["ntx_current"],
            label="NTX + NEOPAX",
            linewidth=2.4,
            color="#1f77b4",
        )
        ax.plot(
            bootstrap_payload["rho"],
            bootstrap_payload["reference_current"],
            label="Reference database + NEOPAX",
            linewidth=2.0,
            linestyle="--",
            color="#d62728",
        )
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$j_\parallel$ proxy")
        ax.set_title("Bootstrap-Current Profile")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
        ax.text(
            0.02,
            0.98,
            f"max rel. error = {bootstrap_payload['max_relative_error']:.2e}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    ax = axes[2]
    if sfincs_payload is None:
        ax.text(0.5, 0.5, "SFINCS-JAX comparison skipped", ha="center", va="center")
        ax.axis("off")
    else:
        sfincs_matrix = np.asarray(sfincs_payload["transport_matrix"], dtype=np.float64)
        ntx_matrix = np.asarray(sfincs_payload["ntx_matrix"], dtype=np.float64)
        error_matrix = ntx_matrix - sfincs_matrix
        im = ax.imshow(error_matrix, cmap="coolwarm")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1], labels=["11/13", "31/33"])
        ax.set_yticks([0, 1], labels=["row 1", "row 2"])
        ax.set_title("NTX - SFINCS-JAX Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{error_matrix[i, j]:+.2e}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )
        ax.text(
            0.02,
            0.98,
            f"max rel. error = {sfincs_payload['max_relative_error']:.2e}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _resolve_defaults(_parse_args())
    if args.wout is None and args.boozmn is None:
        raise SystemExit("Need either `--wout` or `--boozmn`.")

    if args.reference_database is not None:
        reference_scan = _subset_reference_scan(
            load_neopax_reference_scan(args.reference_database),
            nu_stride=args.nu_stride,
            er_stride=args.er_stride,
        )
    else:
        reference_scan = None

    mode, _surface, point_result = _solve_single_surface(
        wout=args.wout,
        boozmn=args.boozmn,
        surface_source=args.surface_source,
        rho=float(args.rho),
        nu_hat=float(args.nu_hat),
        epsi_hat=float(args.epsi_hat),
        grid_tuple=tuple(args.grid),
    )
    point_payload = {
        "rho": float(args.rho),
        "nu_hat": float(args.nu_hat),
        "epsi_hat": float(args.epsi_hat),
        "D11": float(point_result.D11),
        "D13": float(point_result.D13),
        "D31": float(point_result.D31),
        "D33": float(point_result.D33),
    }
    point_reference_payload = (
        _reference_point_from_scan(
            reference_scan,
            rho=float(args.rho),
            nu_hat=float(args.nu_hat),
            epsi_hat=float(args.epsi_hat),
        )
        if reference_scan is not None
        else None
    )

    bootstrap_payload: dict[str, np.ndarray] | None = None
    if (
        not args.skip_bootstrap
        and args.wout is not None
        and args.boozmn is not None
        and reference_scan is not None
    ):
        try:
            import NEOPAX
        except ImportError as exc:
            raise SystemExit(
                "NEOPAX is required for the bootstrap-current part of this example."
            ) from exc

        try:
            loader, _mode = _surface_loader(
                args.wout,
                args.boozmn,
                preferred=args.surface_source,
            )
            surfaces = tuple(
                loader(float(rho_value)) for rho_value in np.asarray(reference_scan.rho)
            )
            ntx_scan = build_ntx_neopax_scan_from_surfaces(
                surfaces,
                rho=reference_scan.rho,
                nu_v=reference_scan.nu_v,
                Es=reference_scan.Es,
                Er=reference_scan.Er,
                drds=reference_scan.drds,
                grid=GridSpec(
                    n_theta=int(args.database_grid[0]),
                    n_zeta=int(args.database_grid[1]),
                    n_xi=int(args.database_grid[2]),
                ),
                source_name="ntx_vmec_or_boozmn_example",
            )
            grid, field, species = _build_species_and_field(
                NEOPAX,
                wout=args.wout,
                boozmn=args.boozmn,
            )
            ntx_database = to_neopax_monoenergetic(ntx_scan, a_b=float(field.a_b))
            reference_database = to_neopax_monoenergetic(reference_scan, a_b=float(field.a_b))
            rho_grid, ntx_current = _bootstrap_current_profile(
                NEOPAX,
                field=field,
                grid=grid,
                species=species,
                database=ntx_database,
            )
            _rho_grid_ref, reference_current = _bootstrap_current_profile(
                NEOPAX,
                field=field,
                grid=grid,
                species=species,
                database=reference_database,
            )
            max_rel = np.max(
                np.abs(ntx_current - reference_current)
                / np.maximum(np.abs(reference_current), 1.0e-12)
            )
            bootstrap_payload = {
                "rho": rho_grid,
                "ntx_current": ntx_current,
                "reference_current": reference_current,
                "max_relative_error": np.asarray(max_rel),
            }
        except Exception as exc:  # pragma: no cover - depends on external VMEC/NEOPAX files
            print(f"Skipping bootstrap-current comparison: {exc}")

    sfincs_payload = None
    if not args.skip_sfincs and args.sfincs_output is not None:
        sfincs_payload = _compare_against_sfincs(
            sfincs_output=args.sfincs_output,
            boozmn=args.boozmn,
            wout=args.wout,
            surface_source=args.surface_source,
            rho=float(args.rho),
            reference_database=args.reference_database,
            grid_tuple=tuple(args.sfincs_grid),
        )

    _plot_summary(
        output_prefix=args.output_prefix,
        bootstrap_payload=bootstrap_payload,
        point_payload=point_payload,
        point_reference_payload=point_reference_payload,
        sfincs_payload=sfincs_payload,
    )

    summary = {
        "geometry_mode": mode,
        "wout": None if args.wout is None else str(args.wout),
        "boozmn": None if args.boozmn is None else str(args.boozmn),
        "reference_database": (
            None if args.reference_database is None else str(args.reference_database)
        ),
        "sfincs_output": None if args.sfincs_output is None else str(args.sfincs_output),
        "point": point_payload,
        "reference_point": point_reference_payload,
        "bootstrap": None
        if bootstrap_payload is None
        else {"max_relative_error": float(bootstrap_payload["max_relative_error"])},
        "sfincs": None
        if sfincs_payload is None
        else {
            "nu_hat": float(sfincs_payload["nu_hat"]),
            "epsi_hat": float(sfincs_payload["epsi_hat"]),
            "max_relative_error": float(sfincs_payload["max_relative_error"]),
        },
        "figure_png": str(args.output_prefix.with_suffix(".png")),
        "figure_pdf": str(args.output_prefix.with_suffix(".pdf")),
        "reference_source": (
            "reference-style NEOPAX database" if args.reference_database is not None else None
        ),
        "reference_checkout_found": find_reference_root() is not None,
    }
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Geometry path: {mode}")
    print(
        "NTX point coefficients:",
        f"D11={point_payload['D11']:.6e}",
        f"D13={point_payload['D13']:.6e}",
        f"D31={point_payload['D31']:.6e}",
        f"D33={point_payload['D33']:.6e}",
    )
    if point_reference_payload is not None:
        point_ref_values = np.array(
            [
                point_reference_payload["D11"],
                point_reference_payload["D13"],
                point_reference_payload["D31"],
                point_reference_payload["D33"],
            ],
            dtype=np.float64,
        )
        point_ntx_values = np.array(
            [
                point_payload["D11"],
                point_payload["D13"],
                point_payload["D31"],
                point_payload["D33"],
            ],
            dtype=np.float64,
        )
        point_rel = np.max(
            np.abs(point_ntx_values - point_ref_values)
            / np.maximum(np.abs(point_ref_values), 1.0e-12)
        )
        print("Reference point max relative error:", f"{point_rel:.3e}")
    if bootstrap_payload is not None:
        print(
            "Bootstrap-current max relative error:",
            f"{float(bootstrap_payload['max_relative_error']):.3e}",
        )
    if sfincs_payload is not None:
        print(
            "SFINCS-JAX matrix max relative error:",
            f"{float(sfincs_payload['max_relative_error']):.3e}",
        )
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
