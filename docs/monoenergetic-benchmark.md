# Cross-code monoenergetic benchmark

NTX, [MONKES](https://github.com/JavierEscoto/MONKES) and
[DKX](https://github.com/uwplasma/DKX) (monoenergetic mode, `RHSMode = 3`)
solve the same monoenergetic drift-kinetic equation on the same flux surfaces.
This page compares their accuracy and cost.

## Setup

- **Surfaces.** W7-X EIM (the MONKES example surface as published), and the
  Landreman–Paul QA and QH and an LHD equilibrium at s = 0.25, transformed with
  booz_xform_jax and kept to |B_mn| ≥ 10⁻⁵ B₀₀. Each surface is written once as
  a DKES `ddkes2.data` file and read by all three codes.
- **Points.** Three regimes, ν* = νR₀/ι = 3×10⁻⁴ (1/ν, √ν with E_r), 0.1
  (plateau) and 10 (collisional), each at v_E = E_r/B₀₀ = 0, 10⁻⁴ and 5×10⁻⁴,
  in DKES units (ν/v, E_r/v).
- **Grids** `(N_theta, N_zeta, N_xi)`: (23, 55, 140) at 1/ν, (19, 41, 48) at
  plateau, (15, 31, 16) collisional, the same for every code.
- **Reference.** NTX on a finer grid: (31, 71, 200), (27, 63, 96) and
  (23, 55, 48). A second fine grid checks it; the largest change is listed per
  row in the JSON record.
- **Error** is the largest relative difference from the reference over the
  three E_r values, for D11, D31 and D33 separately.
- **Cost.** Every run is a fresh process. Cold is the first solve: surface
  load, tracing and compilation for NTX and DKX, with no persistent cache, and
  process wall time for MONKES. Warm is a later solve at the same shapes (MONKES:
  its own per-solve wall time). Peak RSS is the process maximum. Times are the
  best of three runs.
- **DKX units.** DKX reports Beidler-normalized D*. They are mapped back to DKES
  units with DKX's own surface constants. Beidler's coefficients are pitch
  averages, half the DKES ξ integral, and use the orientation-standardized field,
  hence a factor 2 for D11 and D31 and the sign of G for D31.

NTX and MONKES discretize identically (Fourier collocation in angles, Legendre
in pitch, block-tridiagonal elimination); on the same grid their coefficients
agree to about 10⁻¹¹. Their accuracy columns are therefore the same, and only
the cost differs.

## Results

Host: Intel Xeon W-2295 and RTX A4000, JAX 0.11.2, 8 pinned cores (GPU runs: 2).
The host was shared at load 25–65 during the runs, so times are best of two or
three and still carry noise of tens of percent; MONKES's cold times show it most.

| Device | Regime | Max error NTX = MONKES / DKX | Warm [s] NTX / MONKES / DKX | Cold [s] NTX / MONKES / DKX | RSS [GB] NTX / MONKES / DKX |
| --- | --- | --- | --- | --- | --- |
| W7-X | 1/nu | 5e-03 / 3e-03 | 36.85 / 6.33 / 100.86 | 43.5 / 6.9 / 156.5 | 1.14 / 0.22 / 23.90 |
| W7-X | plateau | 2e-07 / 2e-07 | 3.95 / 0.72 / 6.21 | 8.2 / 42.4 / 13.0 | 0.89 / 0.17 / 1.54 |
| W7-X | collisional | 9e-06 / 2e-05 | 0.41 / 0.09 / 0.83 | 4.1 / 1.2 / 7.5 | 0.83 / 0.17 / 0.96 |
| QA | 1/nu | 8e-04 / 8e-04 | 29.20 / 6.21 / 31.41 | 35.0 / 6.6 / 39.3 | 1.14 / 0.22 / 1.19 |
| QA | plateau | 7e-06 / 6e-06 | 4.10 / 0.71 / 6.29 | 8.6 / 2.0 / 13.1 | 0.87 / 0.17 / 1.54 |
| QA | collisional | 3e-05 / 3e-05 | 0.41 / 0.07 / 0.76 | 4.0 / 0.2 / 7.8 | 0.83 / 0.17 / 0.96 |
| QH | 1/nu | 7e-05 / 2e-04 | 33.85 / 6.51 / 111.71 | 40.2 / 6.9 / 163.6 | 1.14 / 0.22 / 22.86 |
| QH | plateau | 8e-09 / 8e-09 | 3.88 / 0.76 / 6.71 | 8.4 / 0.9 / 12.8 | 0.89 / 0.17 / 1.54 |
| QH | collisional | 8e-07 / 8e-07 | 0.36 / 0.08 / 0.74 | 3.9 / 0.2 / 7.0 | 0.86 / 0.17 / 0.93 |
| LHD | 1/nu | 1e-01 / 1e-01 | 29.25 / 6.21 / 88.80 | 34.2 / 6.6 / 130.3 | 1.15 / 0.22 / 22.99 |
| LHD | plateau | 4e-09 / 4e-09 | 3.59 / 0.74 / 5.18 | 7.6 / 7.8 / 11.3 | 0.89 / 0.17 / 1.54 |
| LHD | collisional | 4e-09 / 4e-09 | 0.40 / 0.09 / 0.80 | 3.7 / 0.3 / 7.3 | 0.84 / 0.17 / 0.96 |

| Device | Regime | GPU code | Max error | Cold [s] | Warm [s] |
| --- | --- | --- | --- | ---: | ---: |
| W7-X | 1/nu | NTX | 5e-03 | 30.7 | 6.36 |
| W7-X | 1/nu | DKX | 3e-03 | 158.8 | 45.24 |
| W7-X | plateau | NTX | 2e-07 | 17.5 | 0.81 |
| W7-X | plateau | DKX | 2e-07 | 31.5 | 1.15 |
| W7-X | collisional | NTX | 9e-06 | 15.7 | 0.09 |
| W7-X | collisional | DKX | 2e-05 | 21.8 | 0.25 |
| QA | 1/nu | NTX | 8e-04 | 27.6 | 6.38 |
| QA | 1/nu | DKX | 8e-04 | 38.4 | 9.66 |
| QA | plateau | NTX | 7e-06 | 16.4 | 0.81 |
| QA | plateau | DKX | 6e-06 | 30.8 | 1.16 |
| QA | collisional | NTX | 3e-05 | 20.2 | 0.09 |
| QA | collisional | DKX | 3e-05 | 27.3 | 0.25 |
| QH | 1/nu | NTX | 7e-05 | 29.1 | 6.39 |
| QH | 1/nu | DKX | 2e-04 | 192.7 | 70.87 |
| QH | plateau | NTX | 8e-09 | 22.1 | 1.80 |
| QH | plateau | DKX | 8e-09 | 35.2 | 2.43 |
| QH | collisional | NTX | 8e-07 | 22.2 | 0.20 |
| QH | collisional | DKX | 8e-07 | 29.3 | 0.43 |
| LHD | 1/nu | NTX | 1e-01 | 36.6 | 13.79 |
| LHD | 1/nu | DKX | 1e-01 | 179.8 | 64.52 |
| LHD | plateau | NTX | 4e-09 | 16.5 | 0.82 |
| LHD | plateau | DKX | 4e-09 | 27.0 | 1.14 |
| LHD | collisional | NTX | 4e-09 | 16.6 | 0.09 |
| LHD | collisional | DKX | 4e-09 | 20.7 | 0.26 |

| Device | nu* | N_xi for 1% | (N_theta, N_zeta) for 1% |
| --- | ---: | ---: | --- |
| W7-X | 0.1 | 16 | ≤ (11, 23) |
| W7-X | 0.01 | 32 | ≤ (11, 23) |
| W7-X | 0.001 | 64 | (15, 31) |
| W7-X | 0.0003 | 128 | (15, 31) |
| W7-X | 0.0001 | 160 | (19, 41) |

NTX warm solve time grows as N_fs^2.21 at N_xi = 64.


Findings:

- **Accuracy.** On the common grids, all three codes are within 0.5% of the
  reference everywhere except LHD at ν* = 3×10⁻⁴. NTX and MONKES agree to about
  10⁻¹¹. DKX agrees with them to rounding at plateau and collisional
  collisionality, and to 0.3% at 1/ν.
- **LHD at 1/ν is not resolved.** D31 changes by 1.9% between (39, 87, 280)
  and (47, 103, 360) (NTX on GPU, the reference for this row), and the common
  1/ν grid is 10% off. LHD needs much finer angular grids there than the
  other three devices.
- **CPU cost.** On this Xeon, MONKES's warm solve is 4–6 times faster than NTX's,
  and NTX is 1–3 times faster than DKX. On an Apple M3 Max (4 threads, a spot
  check outside the record), NTX solved the W7-X 1/ν grid in 5.1 s against
  MONKES's 9–10 s, so the ratio depends on the BLAS and the host.
- **GPU.** On one A4000, NTX's warm 1/ν solve takes 6.4 s, the same as MONKES on
  8 cores. Cold GPU starts are 16–37 s of compilation; the persistent cache
  (on by default) removes most of that on later runs.
- **Memory.** NTX stays near 1.1 GB because the forward sweep keeps only the
  three retained Legendre modes. DKX reaches 23 GB on the 1/ν grid, MONKES 0.2 GB.
- **Resolution for 1% (W7-X, E_r = 0).** N_xi grows from 16 at ν* = 0.1 to 160
  at ν* = 10⁻⁴; the angular grid needed grows from (11, 23) to (19, 41). Warm
  cost grows as N_fs^2.2 at fixed N_xi.

## Reproduce

```bash
python benchmarks/bench_monoenergetic_crosscode.py surfaces --work bench
python benchmarks/bench_monoenergetic_crosscode.py measure --work bench --monkes path/to/main_monkes.x
python benchmarks/bench_monoenergetic_crosscode.py measure --work bench --gpu 0
python benchmarks/bench_monoenergetic_crosscode.py lownu --work bench --devices W7-X
python benchmarks/bench_monoenergetic_crosscode.py cost --work bench
python benchmarks/bench_monoenergetic_crosscode.py          # print the tables
```

`surfaces` downloads the inputs from GitHub (MONKES, simsopt and STELLOPT
repositories). MONKES is built from its sources; only its `ddkes2.data` input
path is used. The record is `benchmarks/results/monoenergetic_crosscode.json`,
and `tests/test_monoenergetic_crosscode_benchmark.py` checks the claims on this
page against it.
