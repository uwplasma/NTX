# NTX Manuscript Tables

## Validation

| Grid `(N_theta, N_zeta, N_xi)` | Max relative error |
| --- | ---: |
| `(13, 17, 17)` | 1.038e+02 |
| `(17, 25, 33)` | 2.965e+01 |
| `(25, 25, 64)` | 1.830e-02 |

## Benchmark Matrix

| Benchmark | Lane | Maturity | Status |
| --- | --- | --- | --- |
| `monoenergetic_validation_summary` | `monoenergetic` | `positive-gate` | `complete` |
| `precise_qs_redl_sfincs` | `bootstrap-current` | `positive-gate` | `complete` |
| `fixed_field_ntx_neopax_closure_stress` | `bootstrap-current` | `stress-gate` | `complete` |
| `w7x_integrated_transfer` | `integrated-workflow` | `positive-gate` | `complete` |
| `prepared_derivative_path` | `autodiff` | `positive-gate` | `complete` |
| `geometry_control_derivative_benchmark` | `autodiff` | `stress-gate` | `complete` |
| `file_backed_geometry_control_derivative_benchmark` | `autodiff` | `stress-gate` | `complete` |
| `boundary_forward_mode_current_derivative_benchmark` | `autodiff` | `stress-gate` | `complete` |
| `implicit_equilibrium_forward_mode_derivative_benchmark` | `autodiff` | `stress-gate` | `complete` |
| `explicit_relaxed_boundary_current_derivative_benchmark` | `autodiff` | `stress-gate` | `complete` |
| `autodiff_inverse_problem` | `autodiff` | `stress-gate` | `complete` |
| `autodiff_profile_uncertainty` | `autodiff` | `stress-gate` | `complete` |
| `robust_bootstrap_current_optimization` | `autodiff` | `stress-gate` | `complete` |
| `profile_force_reconstruction` | `profile-workflow` | `stress-gate` | `complete` |
| `performance_scaling` | `performance` | `software-gate` | `complete` |
| `geometry_breadth_hidden_symmetry` | `geometry-breadth` | `planned-lane` | `planned` |
| `full_monoenergetic_geometry_family` | `monoenergetic` | `planned-lane` | `planned` |
| `large_geometry_control_autodiff` | `autodiff` | `planned-lane` | `planned` |

## Derivatives

| Quantity | Value |
| --- | ---: |
| Grid | `(7, 9, 6)` |
| `nu_hat` | `3.000e-04` |
| `E_r` scan | `1.000e-06` to `3.000e-03` |
| Max relative mismatch | `1.051e-05` |
| Best prepared speedup | `3.861x` |

## Geometry-Control Derivatives

| Quantity | Value |
| --- | ---: |
| Grid | `(7, 9, 6)` |
| Controlled modes | `3` |
| Coefficients | `D11, D31, D33` |
| Max AD/centered-FD mismatch | `1.348e-04` |
| Median AD/centered-FD mismatch | `3.595e-06` |

## Boundary Forward-Mode Current Derivatives

| Quantity | Value |
| --- | ---: |
| Controlled parameters | `rc10, zs10` |
| Max AD/centered-FD mismatch | `7.174e-07` |
| Median AD/centered-FD mismatch | `3.556e-07` |

## Implicit-Equilibrium Forward-Mode Derivatives

| Quantity | Value |
| --- | ---: |
| Controlled parameters | `rc01` |
| Implicit solver | `iter=5, step=1.0, tangent=auto` |
| Max AD/centered-FD mismatch | `6.566e+00` |
| Median AD/centered-FD mismatch | `7.337e-01` |
| Reverse-mode Boozer max mismatch | `unsupported` |
| Reverse-mode Boozer status | `unsupported` |
| Equilibrium-volume mismatch | `9.236e-05` |
| Boozer-scalar mismatch | `7.337e-01` |
| NTX transport mismatch | `6.566e+00` |

## Explicit-Relaxed Boundary Current Derivatives

| Quantity | Value |
| --- | ---: |
| Cases | `qa_lowres, qh_warm_start` |
| Explicit relaxation | `iter=10, step=1.0e-08` |
| Ordinary/explicit volume rel. diff. | `0.000e+00` |
| Max AD/centered-FD mismatch | `2.626e-05` |
| Median AD/centered-FD mismatch | `4.408e-07` |

## File-Backed Geometry-Control Derivatives

| Quantity | Value |
| --- | ---: |
| Cases | `boozmn_sample, vmec_sample` |
| Max AD/centered-FD mismatch | `3.088e-04` |
| Median AD/centered-FD mismatch | `5.158e-08` |

## Bootstrap-Current Optimization

| Quantity | Value |
| --- | ---: |
| Harmonic `(m, n)` | `(0, -1)` |
| Baseline scale | `1.000` |
| Optimized scale | `1.297` |
| Weighted current gain | `1.085x` |
| Serial scan time | `1.451 s` |
| Parallel scan time | `2.405 s` |

## Performance

### CPU heavy-grid scaling

| Cases | Serial [s] | Multiprocess [s] | Speedup |
| ---: | ---: | ---: | ---: |
| 16 | 2.229 | 3.936 | 0.566x |
| 32 | 4.275 | 4.551 | 0.939x |
| 64 | 9.044 | 5.051 | 1.790x |

### GPU heavy-grid scaling

| Cases | Serial [s] | Multiprocess [s] | Speedup | Healthy devices |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 5.177 | 26.306 | 0.197x | 1 |
| 32 | 2.111 | 6.692 | 0.315x | 1 |
| 64 | 2.976 | 7.025 | 0.424x | 1 |

## Reproducibility

| Key | Value |
| --- | --- |
| Commit | `02d1eb3efa9376f7eabeb0e0a0e6d787729e1ebf` |
| Branch | `main` |
| Python | `3.11.14` |
| JAX | `0.9.2` |
| NumPy | `2.4.4` |
| Platform | `macOS-14.4.1-arm64-arm-64bit` |
| Figure bundle | `python examples/make_publication_figures.py --figures main_text,supplement` |
| Main-text figures | `python examples/make_publication_figures.py --figures main_text` |
| Supplement figures | `python examples/make_publication_figures.py --figures supplement` |
| Artifact tables | `python scripts/build_manuscript_artifacts.py` |
| Benchmark matrix | `python scripts/build_benchmark_matrix.py` |
| Validation subset | `python -m pytest -q tests/test_w7x_reference_benchmark.py tests/test_derivative_path_benchmark_example.py tests/test_bootstrap_current_optimization_example.py tests/test_manuscript_artifacts_script.py tests/test_make_publication_figures.py -k "subset_writes_manifest or bootstrap_subset_writes_manifest"` |
