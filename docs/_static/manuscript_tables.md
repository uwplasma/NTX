# NTX Manuscript Tables

## Validation

| Grid `(N_theta, N_zeta, N_xi)` | Max relative error |
| --- | ---: |
| `(13, 17, 17)` | 1.038e+02 |
| `(17, 25, 33)` | 2.965e+01 |
| `(25, 25, 64)` | 1.830e-02 |

## Monoenergetic Validation Summary

| Quantity | Value |
| --- | ---: |
| Grid | `(11, 13, 10)` |
| DKES-style finest plotted `N_xi` error | `1.671e-01` |
| VMEC finest plotted `N_xi` error | `1.969e-01` |
| DKES-style max Onsager residual | `2.785e-06` |
| VMEC monitored max Onsager residual | `1.782e+00` |

## Fixed-Field Precise-QS Benchmark

| Case | Redl/SFINCS interior error | NTX+NEOPAX/SFINCS interior stress |
| --- | ---: | ---: |
| `qa` | `6.857e-02` | `8.305e-02` |
| `qh` | `4.063e-02` | `9.954e-02` |

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
| `bootstrap_current_optimization` | `autodiff` | `stress-gate` | `complete` |
| `robust_bootstrap_current_optimization` | `autodiff` | `stress-gate` | `complete` |
| `profile_force_reconstruction` | `profile-workflow` | `stress-gate` | `complete` |
| `profile_basis_optimization` | `profile-workflow` | `stress-gate` | `complete` |
| `performance_scaling` | `performance` | `software-gate` | `complete` |
| `prepared_geometry_reuse_profile` | `performance` | `software-gate` | `complete` |
| `geometry_family_breadth_summary` | `geometry-breadth` | `stress-gate` | `complete` |
| `geometry_family_transport_convergence` | `geometry-breadth` | `stress-gate` | `complete` |
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
| Best prepared speedup | `3.789x` |

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
| Max AD/centered-FD mismatch | `6.454e+00` |
| Median AD/centered-FD mismatch | `8.544e-01` |
| Reverse-mode Boozer max mismatch | `unsupported` |
| Reverse-mode Boozer status | `unsupported` |
| Equilibrium-volume mismatch | `9.236e-05` |
| Boozer-scalar mismatch | `8.544e-01` |
| NTX transport mismatch | `6.454e+00` |

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

## Geometry-Family Breadth Summary

| Quantity | Value |
| --- | ---: |
| Active artifact-backed cases | `7` |
| Open implicit objectives | `0` |
| Retired implicit diagnostics | `2` |
| Active case ids | `analytic_geometry_control, file_backed_boozmn_sample, file_backed_vmec_sample, boundary_projected_current, explicit_relaxed_qa_lowres, explicit_relaxed_qh_warm_start, implicit_equilibrium_volume` |
| Open case ids | `` |
| Retired implicit ids | `implicit_booz_xform_scalar, implicit_ntx_transport_proxy` |
| Max active AD/centered-FD mismatch | `3.088e-04` |
| Max retired implicit mismatch | `6.454e+00` |

## Geometry-Family Transport Convergence

| Quantity | Value |
| --- | ---: |
| Solved VMEC cases | `13` |
| Below smoke convergence rtol | `5` |
| Max last-step relative change | `1.952e+00` |
| Max relative change to finest grid | `1.022e+01` |
| Solved case ids | `circular_tokamak, shaped_tokamak, precise_qs_qa_reactor, precise_qs_qh_reactor, nfp3_qi, w7x_eim_ejm_standard, nfp4_qh_reference, high_aspect_qs, li383_low_res, n3are_lowres, lhd, hsx_qhs, ncsx` |

## Profile Uncertainty

| Quantity | Value |
| --- | ---: |
| Radial electric-field basis size | `3` |
| Monte Carlo samples | `96` |
| Max linearized/Monte-Carlo std mismatch | `1.000e+00` |
| Max Monte-Carlo mean shift | `5.004e-16` |
| Fisher eigenvalue range | `8.520e-21` to `2.043e-02` |
| Hessian-vector/Fisher probe mismatch | `1.753e-16` |

## Bootstrap-Current Optimization

| Quantity | Value |
| --- | ---: |
| Harmonic `(m, n)` | `(0, -1)` |
| Baseline scale | `1.000` |
| Optimized scale | `1.297` |
| Weighted current gain | `1.085x` |
| Serial scan time | `0.423 s` |
| Parallel scan time | `1.750 s` |

## Performance

### CPU heavy-grid scaling

| Cases | Serial [s] | Multiprocess [s] | Speedup |
| ---: | ---: | ---: | ---: |
| 16 | 0.925 | 4.265 | 0.217x |
| 32 | 1.767 | 4.447 | 0.397x |
| 64 | 2.568 | 4.342 | 0.591x |

### GPU heavy-grid scaling

| Cases | Serial [s] | Multiprocess [s] | Speedup | Healthy devices |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1.987 | 6.756 | 0.294x | 1 |
| 32 | 3.222 | 6.755 | 0.477x | 1 |
| 64 | 2.956 | 7.076 | 0.418x | 1 |

### Prepared-geometry reuse

| Cases | Direct [s] | Prepared total [s] | Compiled steady [s] | Compiled speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.626 | 0.644 | 0.005 | 137.996x |
| 16 | 2.675 | 2.488 | 0.020 | 130.591x |
| 48 | 7.762 | 7.519 | 0.059 | 130.789x |

## Reproducibility

| Key | Value |
| --- | --- |
| Commit | `c98ec8c28bfdcccce56e629d478ea7c1457c5b41` |
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
