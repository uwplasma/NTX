#!/usr/bin/env python3
"""Monoenergetic cross-code benchmark: NTX against MONKES and DKX.

    python benchmarks/bench_monoenergetic_crosscode.py                  # tables from the JSON
    python benchmarks/bench_monoenergetic_crosscode.py surfaces --work DIR
    python benchmarks/bench_monoenergetic_crosscode.py measure --work DIR --monkes BIN [--gpu 0]
    python benchmarks/bench_monoenergetic_crosscode.py lownu --work DIR

``surfaces`` downloads four public equilibria and writes one DKES ``ddkes2.data``
surface per device: the MONKES W7-X EIM example as published, and QA, QH and
LHD through booz_xform_jax at s = 0.25, keeping |B_mn| >= 1e-5 B_00. All three
codes read that same spectrum, so differences are numerical, not geometric.

``measure`` runs every code in a fresh process per (device, regime), pinned to
the least busy cores, with peak RSS from ``wait4``. Cold is the first solve
(NTX and DKX: surface load, tracing and compilation, no persistent cache;
MONKES: process wall time). Warm is the median of the later solves at the same
shapes. Errors are relative to an NTX reference at a finer grid, which is
checked against a second fine grid. DKX reports Beidler-normalized D* values;
they are mapped back to DKES units with DKX's own surface constants.

``lownu`` finds, per device and collisionality, the smallest Legendre and
angular resolution that keeps D11, D31 and D33 within 1% of the reference, and
fits the warm-solve cost against the angular size N_fs = N_theta N_zeta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "monoenergetic_crosscode.json"
SOURCES = {
    "W7-X": "https://raw.githubusercontent.com/JavierEscoto/MONKES/main/examples/W7X-EIM/dkes_input/ddkes2.data",
    "QA": "https://raw.githubusercontent.com/hiddenSymmetries/simsopt/master/tests/test_files/"
    "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
    "QH": "https://raw.githubusercontent.com/hiddenSymmetries/simsopt/master/tests/test_files/"
    "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc",
    "LHD": "https://raw.githubusercontent.com/PrincetonUniversity/STELLOPT/develop/BENCHMARKS/DIAGNO_TEST/wout_lhd.nc",
}
S_SURFACE, CUTOFF = 0.25, 1e-5
# nu* = nu R0 / iota and v_E = E_r / B00 (DKES units nu/v, E_r/v): same regimes on every device.
REGIMES = {"1/nu": 3e-4, "plateau": 1e-1, "collisional": 10.0}
V_E = (0.0, 1e-4, 5e-4)
GRID = {"1/nu": (23, 55, 140), "plateau": (19, 41, 48), "collisional": (15, 31, 16)}
REFERENCE = {"1/nu": (31, 71, 200), "plateau": (27, 63, 96), "collisional": (23, 55, 48)}
CHECK = {"1/nu": (27, 63, 170), "plateau": (23, 55, 80), "collisional": (19, 41, 32)}
NUD_X0 = 0.8360276804879032  # DKX nuDHat at the monoenergetic node x0 = 1
LOWNU = (1e-1, 1e-2, 1e-3, 3e-4, 1e-4)
NXI_LADDER = (8, 16, 24, 32, 48, 64, 96, 128, 160, 200, 256)
ANGLE_LADDER = ((11, 23), (15, 31), (19, 41), (23, 55), (27, 63), (31, 71))


# --------------------------------------------------------------------------- surfaces
def write_ddkes2(path: Path, nfp, psip, iota, b_theta, b_zeta, modes) -> None:
    """DKES surface file: ``borbi(n, m)`` with n per field period, ``iota = -chip/psip``."""
    rows = "\n".join(f" borbi({n},{m}) = {v:.16e}," for m, n, v in modes)
    path.write_text(
        f" &datain\n nzperiod = {nfp}, lalpha = 20, lfout = 20, nrun = 1,\n"
        f" cmul = 1.0, efield = 0.0, ibbi = 1,\n"
        f" mpolb = {1 + max(m for m, _, _ in modes)},\n"
        f" ntorb = {1 + max(abs(n) for _, n, _ in modes)},\n"
        f" chip = {-iota * psip:.16e}, psip = {psip:.16e},\n"
        f" btheta = {b_theta:.16e}, bzeta = {b_zeta:.16e},\n"
        f"{rows}\n /\n"
    )


def surface_from_wout(wout: Path, out: Path) -> float:
    """Boozer-transform one surface near ``S_SURFACE`` and write it as ``ddkes2``; return s."""
    import netCDF4
    from booz_xform_jax import Booz_xform

    import ntx

    bx = Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(wout), flux=True)
    idx = int(np.argmin(np.abs(np.asarray(bx.s_in) - S_SURFACE)))
    bx.compute_surfs, bx.mboz, bx.nboz = [idx], 32, 32
    bx.run()
    booz = out.with_suffix(".boozmn.nc")
    bx.write_boozmn(str(booz))
    with netCDF4.Dataset(wout) as f:
        torflux, a = float(f["phi"][-1]) / (2 * np.pi), float(f["Aminor_p"][:])
    s = float(np.asarray(bx.s_in)[idx])
    psip = 2 * torflux * np.sqrt(s) / a  # MONKES's d psi / d r with r = a sqrt(s)
    surf = ntx.load_boozmn_surface(booz, surface_index=0, min_bmn_to_load=CUTOFF).surface
    modes = list(
        zip(
            np.asarray(surf.m).tolist(),
            np.asarray(surf.n).tolist(),
            np.asarray(surf.b_cos).tolist(),
            strict=True,
        )
    )
    write_ddkes2(
        out, surf.nfp, psip, float(surf.iota), float(surf.b_theta), float(surf.b_zeta), modes
    )
    booz.unlink()
    return s


def surfaces(work: Path) -> dict:
    """Download the sources and write ``<device>.ddkes2``; return per-device metadata."""
    import ntx

    work.mkdir(parents=True, exist_ok=True)
    meta = {}
    for dev, url in SOURCES.items():
        src = work / url.rsplit("/", 1)[1]
        if not src.exists():
            urllib.request.urlretrieve(url, src)
        out = work / f"{dev}.ddkes2"
        s = None
        if src.suffix == ".nc":
            s = surface_from_wout(src, out)
        else:
            out.write_bytes(src.read_bytes())
        surf = ntx.load_dkes_surface(out)
        meta[dev] = dict(
            source=url,
            s=s,
            nfp=int(surf.nfp),
            modes=int(np.asarray(surf.m).size),
            iota=float(surf.iota),
            b00=float(surf.b0),
            r0=abs(float(surf.b_zeta)) / float(surf.b0),
            psip=float(surf.psi_p),
            sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        )
        print(dev, meta[dev])
    return meta


def points(m: dict, regime: str) -> list[tuple[float, float]]:
    """(nu/v, E_r/v) in DKES units for one regime of one device."""
    nu = REGIMES[regime] * abs(m["iota"]) / m["r0"]
    return [(nu, v * m["b00"]) for v in V_E]


# --------------------------------------------------------------------------- workers
def _timed(fn, p):
    t = time.perf_counter()
    out = fn(p)
    return out, time.perf_counter() - t


def _gpu_peak():
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use")


def worker(code: str, ddkes: str, res, pts) -> dict:
    """Solve ``pts`` in order, the first twice; return DKES-unit [D11, D31, D33] and times."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import ntx

    surf = ntx.load_dkes_surface(ddkes)
    if code == "ntx":
        state = {}

        def solve(p):
            if "solve" not in state:  # cold includes preparation and compilation
                prep = ntx.prepare_monoenergetic_system(surf, ntx.GridSpec(*res))
                state["solve"] = ntx.compile_prepared_solver(prep)
            r = state["solve"](ntx.MonoenergeticCase(nu_hat=p[0], er_hat=p[1]))
            return [float(r.D11), float(r.D31), float(r.D33)]
    else:
        from dkx.monoenergetic import monoenergetic_database

        iota, g, i, psip, b0 = (
            float(x) for x in (surf.iota, surf.b_zeta, surf.b_theta, surf.psi_p, surf.b0)
        )
        r0 = abs(g) / b0
        rows = "\n".join(  # DKX's Boozer toroidal mode number has the opposite sign
            f"  boozer_bmnc({m},{-n}) = {v:.16e}"
            for m, n, v in zip(surf.m.tolist(), surf.n.tolist(), surf.b_cos.tolist(), strict=True)
        )
        deck = Path(tempfile.mkdtemp()) / "input.namelist"
        deck.write_text(
            "&general\n  RHSMode = 3\n/\n&geometryParameters\n  geometryScheme = 13\n"
            "  inputRadialCoordinate = 3\n  rN_wish = 0.5\n  aHat = 1.0\n  psiAHat = 1.0\n"
            f"  Nperiods = {surf.nfp}\n  iota = {iota:.16e}\n  GHat = {g:.16e}\n  IHat = {i:.16e}\n"
            f"{rows}\n/\n&speciesParameters\n/\n&physicsParameters\n  collisionOperator = 1\n/\n"
            f"&resolutionParameters\n  Ntheta = {res[0]}\n  Nzeta = {res[1]}\n  Nxi = {res[2]}\n"
            "  Nx = 1\n  solverTolerance = 1d-10\n/\n"
        )

        def solve(p):
            # DKX's nu' and E* for (nu/v, E_r/v): E* = E_psi R0 / iota, E_psi = E_r / psip.
            db = monoenergetic_database(
                deck,
                [abs(g + iota * i) / b0 * p[0] / NUD_X0],
                [p[1] * r0 / (psip * iota)],
                tol=1e-10,
            )
            d11s, d31s, d33s = (
                float(np.ravel(getattr(db, k))[0]) for k in ("d11_star", "d31_star", "d33_star")
            )
            r, b2 = float(db.r_hat), float(np.ravel(db.fsab_hat2)[0]) / b0**2
            ft = 1.46 * np.sqrt(r / r0)
            # Beidler's D* are pitch averages (half the DKES xi integral), hence the 2.
            # DKX standardizes orientation (|G|, |iota|), so D31* carries -sign of DKES D31.
            return [
                2 * d11s * (np.pi / 16) / (abs(iota) * r0 * b0**2) * (b0 * r / psip) ** 2,
                -2 * d31s * r0 * ft / (3 * abs(iota * psip)),
                d33s * 2 * b2 / (3 * p[0]),
            ]

    runs = [_timed(solve, p) for p in [pts[0], *pts]]
    gpu = _gpu_peak() if jax.default_backend() == "gpu" else None
    return dict(
        values=[v for v, _ in runs[1:]],
        cold=runs[0][1],
        warm=float(np.median([t for _, t in runs[1:]])),
        gpu_peak_bytes=gpu,
    )


# --------------------------------------------------------------------------- launching
def idle_cores(n: int) -> str:
    """The ``n`` least busy logical CPUs over one second of ``/proc/stat`` (the host is shared)."""

    def ticks():
        rows = [ln.split() for ln in open("/proc/stat") if ln[:3] == "cpu" and ln[3] != " "]
        return {int(r[0][3:]): (sum(map(int, r[1:])), int(r[4]) + int(r[5])) for r in rows}

    a = ticks()
    time.sleep(1.0)
    b = ticks()
    busy = {c: 1 - (b[c][1] - a[c][1]) / max(b[c][0] - a[c][0], 1) for c in a}
    return ",".join(map(str, sorted(sorted(busy, key=busy.get)[:n])))


def rss_guard(p, done: threading.Event, limit_gb: float = 24.0) -> None:
    """Kill the child if its resident set exceeds ``limit_gb`` (the host is shared)."""
    status = Path(f"/proc/{p.pid}/status")
    while not done.is_set() and status.exists():
        try:
            kb = next(
                int(ln.split()[1]) for ln in status.read_text().splitlines() if ln[:6] == "VmRSS:"
            )
        except (OSError, StopIteration, ValueError):
            return
        if kb > limit_gb * 1e6:
            p.kill()
            return
        time.sleep(0.5)


def launch(
    cmd, *, ncores: int, gpu: str | None, timeout: float, cwd=None
) -> tuple[str, float, int, int]:
    """Run pinned; return (stdout, wall seconds, peak RSS bytes, status)."""
    env = dict(
        os.environ,
        OMP_NUM_THREADS=str(ncores),
        # MONKES threads through OpenBLAS; for the JAX codes a threaded OpenBLAS
        # inside LAPACK calls oversubscribes XLA's pool (measured 1.2-6x slower).
        OPENBLAS_NUM_THREADS=str(ncores if cmd[0].endswith(".x") else 1),
        XLA_FLAGS=f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={ncores}",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        CUDA_VISIBLE_DEVICES=gpu or "",
        JAX_PLATFORMS="cuda" if gpu else "cpu",
        DKX_CORES=str(ncores),
        DKX_DISABLE_COMPILATION_CACHE="1",
        NTX_DISABLE_COMPILATION_CACHE="1",
    )
    pin = ["taskset", "-c", idle_cores(ncores)] if sys.platform.startswith("linux") else []
    t = time.perf_counter()
    with tempfile.TemporaryFile() as f:
        p = subprocess.Popen([*pin, *cmd], stdout=f, stderr=subprocess.DEVNULL, env=env, cwd=cwd)
        timer = threading.Timer(timeout, p.kill)
        timer.start()
        done = threading.Event()
        threading.Thread(target=rss_guard, args=(p, done), daemon=True).start()
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = status
        done.set()
        timer.cancel()
        f.seek(0)
        rss = ru.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        return f.read().decode(), time.perf_counter() - t, rss, status


def run_jax(code, ddkes, res, pts, args, gpu=None) -> dict:
    cmd = [sys.executable, __file__, "worker", code, str(ddkes), json.dumps([list(res), pts])]
    out, wall, rss, st = launch(cmd, ncores=args.ncores, gpu=gpu, timeout=args.timeout)
    rec = json.loads(out.strip().splitlines()[-1]) if st == 0 else dict(error=f"status {st}")
    return dict(
        rec,
        code=code,
        res=list(res),
        rss_bytes=rss,
        process_wall=wall,
        device="gpu" if gpu else "cpu",
    )


def run_monkes(ddkes, res, pts, args) -> dict:
    """Cold: a fresh process for the first point. Warm: MONKES's own per-solve wall time
    for the later points of one process that solves every E_r at the regime's nu."""

    def once(ers):
        wd = Path(tempfile.mkdtemp(dir=args.work))
        (wd / "ddkes2.data").write_bytes(Path(ddkes).read_bytes())
        (wd / "monkes_input.parameters").write_text(
            f"&parameters\nN_theta={res[0]}\nN_zeta={res[1]}\nN_xi={res[2]}\n"
            f"nu={pts[0][0]!r}\nE_r={','.join(map(repr, ers))}\n/\n"
        )
        _, wall, rss, st = launch(
            [args.monkes], ncores=args.ncores, gpu=None, timeout=args.timeout, cwd=wd
        )
        out = wd / "monkes_Monoenergetic_Database.dat"
        rows = [ln.split() for ln in out.read_text().splitlines()[1:]] if st == 0 else []
        return np.asarray(rows, dtype=float), wall, rss, st

    _, cold, rss1, st1 = once([pts[0][1]])
    rows, _, rss, st = once([p[1] for p in pts])
    ok = st == st1 == 0 and np.allclose(rows[:, 1], [p[1] for p in pts])
    rec = dict(code="monkes", res=list(res), cold=cold, rss_bytes=max(rss, rss1), device="cpu")
    if not ok:
        return dict(rec, error=f"status {st1}/{st}")
    return dict(rec, values=rows[:, [5, 6, 8]].tolist(), warm=float(np.min(rows[1:, 10])))


# --------------------------------------------------------------------------- studies
def rel_err(vals, ref) -> list[float]:
    return np.max(
        np.abs(np.asarray(vals) - np.asarray(ref)) / np.abs(np.asarray(ref)), axis=0
    ).tolist()


def host() -> dict:
    import jax

    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            ln.split(":", 1)[1].strip()
            for ln in open("/proc/cpuinfo")
            if ln.startswith("model name")
        )
    gpu = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.splitlines()[:1]
        if sys.platform.startswith("linux")
        else []
    )
    return dict(
        cpu=cpu,
        gpu=gpu[0] if gpu else None,
        system=platform.system(),
        load_at_start=os.getloadavg()[0],
        python=platform.python_version(),
        jax=jax.__version__,
    )


def timed_runs(code, ddkes, res, pts, args, gpu=None) -> dict:
    """One code at one grid: values from the first run; best-of-``repeats`` times, max RSS."""
    if code == "monkes":
        runs = [run_monkes(ddkes, res, pts, args) for _ in range(args.repeats)]
    else:
        runs = [run_jax(code, ddkes, res, pts, args, gpu) for _ in range(args.repeats)]
    best = dict(runs[0])
    if "error" not in best:
        best.update(
            cold=min(r["cold"] for r in runs),
            warm=min(r["warm"] for r in runs),
            rss_bytes=max(r["rss_bytes"] for r in runs),
            repeats=len(runs),
        )
    return best


def measure(args) -> None:
    meta = json.loads((args.work / "surfaces.json").read_text())
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data.update(surfaces=meta, grids=dict(GRID=GRID, REFERENCE=REFERENCE, CHECK=CHECK))
    data.setdefault("hosts", {})["cpu" if args.gpu is None else "gpu"] = dict(
        host(), ncores=args.ncores
    )
    rows = data.setdefault("rows", [])
    for dev in args.devices.split(","):
        ddkes = args.work / f"{dev}.ddkes2"
        for regime in args.regimes.split(","):
            pts = points(meta[dev], regime)
            old = next((r for r in rows if (r["device"], r["regime"]) == (dev, regime)), None)
            if old is None or args.rerun_reference:
                ref = run_jax("ntx", ddkes, REFERENCE[regime], pts, args)
                chk = run_jax("ntx", ddkes, CHECK[regime], pts, args)
                row = dict(
                    device=dev,
                    regime=regime,
                    points=pts,
                    reference=ref["values"],
                    reference_check=rel_err(chk["values"], ref["values"]),
                    runs=[],
                )
            else:
                row = old
            gpu = args.gpu
            codes = [(c, gpu) for c in args.codes.split(",") if gpu is None or c != "monkes"]
            for code, gpu in codes:
                r = timed_runs(code, ddkes, GRID[regime], pts, args, gpu)
                if r.get("values"):
                    r["rel_err"] = rel_err(r["values"], row["reference"])  # D11, D31, D33
                row["runs"] = [
                    x for x in row["runs"] if (x["code"], x["device"]) != (code, r["device"])
                ]
                row["runs"].append(r)
            rows[:] = [r for r in rows if (r["device"], r["regime"]) != (dev, regime)] + [row]
            OUT.write_text(json.dumps(data, indent=1) + "\n")
            print(
                dev,
                regime,
                row["reference_check"],
                [
                    (r["code"], r["device"], r.get("rel_err"), r.get("cold"), r.get("warm"))
                    for r in row["runs"]
                ],
                flush=True,
            )


def first_within(tol, grids, ddkes, pts, ref, args):
    """The first grid whose D11, D31 and D33 are all within ``tol`` of ``ref``."""
    for g in grids:
        if max(rel_err(run_jax("ntx", ddkes, g, pts, args)["values"], ref)) <= tol:
            return g
    return None


def lownu(args) -> None:
    meta = json.loads((args.work / "surfaces.json").read_text())
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    rows = data.setdefault("lownu", [])
    n_ref, a_ref = NXI_LADDER[-1], ANGLE_LADDER[-1]
    for dev in args.devices.split(","):
        ddkes = args.work / f"{dev}.ddkes2"
        for nustar in LOWNU:
            nu = nustar * abs(meta[dev]["iota"]) / meta[dev]["r0"]
            pts = [(nu, 0.0)]
            ref = run_jax("ntx", ddkes, (*a_ref, n_ref), pts, args)["values"]

            g_xi = first_within(0.01, [(*a_ref, n) for n in NXI_LADDER], ddkes, pts, ref, args)
            g_ang = first_within(0.01, [(*a, n_ref) for a in ANGLE_LADDER], ddkes, pts, ref, args)
            rows[:] = [r for r in rows if (r["device"], r["nu_star"]) != (dev, nustar)]
            rows.append(
                dict(
                    device=dev,
                    nu_star=nustar,
                    nu=nu,
                    n_xi=g_xi and g_xi[2],
                    angles=g_ang and g_ang[:2],
                )
            )
            OUT.write_text(json.dumps(data, indent=1) + "\n")
            print(rows[-1], flush=True)


def cost(args) -> None:
    """NTX warm-solve time and RSS against N_fs at N_xi = 64 (W7-X, plateau point)."""
    meta = json.loads((args.work / "surfaces.json").read_text())
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    rows = []
    for a in ANGLE_LADDER:
        r = timed_runs(
            "ntx", args.work / "W7-X.ddkes2", (*a, 64), points(meta["W7-X"], "plateau")[:1], args
        )
        rows.append(
            dict(n_fs=a[0] * a[1], cold=r["cold"], warm=r["warm"], rss_bytes=r["rss_bytes"])
        )
    fs, t = np.log([c["n_fs"] for c in rows]), np.log([c["warm"] for c in rows])
    data["lownu_cost"] = dict(
        n_xi=64, rows=rows, exponent=float(np.polyfit(fs, t, 1)[0]), host=host()
    )
    OUT.write_text(json.dumps(data, indent=1) + "\n")


# --------------------------------------------------------------------------- report
def report() -> str:
    """Markdown tables: the cross-code matrix, the low-nu resolution table, the cost fit."""
    data = json.loads(OUT.read_text())
    lines = [
        "| Device | Regime | Max error NTX = MONKES / DKX | Warm [s] NTX / MONKES / DKX "
        "| Cold [s] NTX / MONKES / DKX | RSS [GB] NTX / MONKES / DKX |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in data.get("rows", []):
        runs = {(r["code"], r["device"]): r for r in row["runs"]}
        cpu = [runs.get((c, "cpu"), {}) for c in ("ntx", "monkes", "dkx")]

        def cells(key, fmt, runs=cpu):
            return " / ".join(fmt(r[key]) if r.get(key) is not None else "–" for r in runs)

        err = cells("rel_err", lambda e: f"{max(e):.0e}", [cpu[0], cpu[2]])
        lines.append(
            f"| {row['device']} | {row['regime']} | {err} | {cells('warm', '{:.2f}'.format)} | "
            f"{cells('cold', '{:.1f}'.format)} | {cells('rss_bytes', lambda b: f'{b / 1e9:.2f}')} |"
        )
    gpu = [(row, r) for row in data.get("rows", []) for r in row["runs"] if r["device"] == "gpu"]
    if gpu:
        lines += [
            "",
            "| Device | Regime | GPU code | Max error | Cold [s] | Warm [s] |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
        lines += [
            f"| {row['device']} | {row['regime']} | {r['code'].upper()} | "
            f"{max(r['rel_err']):.0e} | "
            f"{r['cold']:.1f} | {r['warm']:.2f} |"
            for row, r in gpu
            if r.get("rel_err")
        ]
    if data.get("lownu"):
        first = ANGLE_LADDER[0]
        lines += [
            "",
            "| Device | nu* | N_xi for 1% | (N_theta, N_zeta) for 1% |",
            "| --- | ---: | ---: | --- |",
        ]
        for r in data["lownu"]:
            ang = tuple(r["angles"]) if r["angles"] else None
            ang_s = f"≤ {first}" if ang == first else (str(ang) if ang else f"> {ANGLE_LADDER[-1]}")
            n_xi = r["n_xi"] or f"> {NXI_LADDER[-1]}"
            lines.append(f"| {r['device']} | {r['nu_star']:g} | {n_xi} | {ang_s} |")
    if data.get("lownu_cost"):
        c = data["lownu_cost"]
        lines += [
            "",
            f"NTX warm solve time grows as N_fs^{c['exponent']:.2f} at N_xi = {c['n_xi']}.",
        ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        res, pts = json.loads(sys.argv[4])
        print(json.dumps(worker(sys.argv[2], sys.argv[3], tuple(res), [tuple(p) for p in pts])))
        return
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "stage",
        nargs="?",
        choices=("report", "surfaces", "measure", "lownu", "cost"),
        default="report",
    )
    ap.add_argument("--work", type=Path, default=Path("monoenergetic-bench"))
    ap.add_argument("--monkes", help="MONKES main_monkes.x")
    ap.add_argument("--devices", default=",".join(SOURCES))
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--codes", default="ntx,monkes,dkx")
    ap.add_argument("--ncores", type=int, default=8)
    ap.add_argument("--gpu", help="CUDA device index for the GPU rows")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rerun-reference", action="store_true")
    args = ap.parse_args()
    if args.stage == "surfaces":
        (args.work / "surfaces.json").parent.mkdir(parents=True, exist_ok=True)
        (args.work / "surfaces.json").write_text(json.dumps(surfaces(args.work), indent=1))
    elif args.stage == "measure":
        measure(args)
    elif args.stage == "lownu":
        lownu(args)
    elif args.stage == "cost":
        cost(args)
    else:
        print(report())


if __name__ == "__main__":
    main()
