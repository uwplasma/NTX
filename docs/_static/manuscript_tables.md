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
| `owned_geometry_neopax_dataset` | `geometry-breadth` | `stress-gate` | `complete` |
| `owned_finite_beta_sfincs_jax_inputs` | `geometry-breadth` | `stress-gate` | `complete` |
| `owned_finite_beta_bootstrap_comparison` | `geometry-breadth` | `stress-gate` | `complete` |
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
| Retired implicit ids | `implicit_booz_xform_scalar, implicit_ntx_transport_response` |
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

## Owned Finite-Beta Bootstrap-Current Stress

| Quantity | Value |
| --- | ---: |
| Case | `finite_beta_qa_pressure_current` |
| Closure configuration | `P=12`, `D33=spitzer`, `nu/v points=17` |
| Boozer psi_p | `1.334630e-02` |
| Max total-current relative difference vs Redl | `3.106e-01` |
| RMS total-current relative difference vs Redl | `1.297e-01` |
| Sign-agreement fraction | `1.000` |
| Inner-gap same-grid coefficient relative difference | `2.055e-02` |
| Inner-gap profile-current relative difference | `3.106e-01` |
| Inner-gap current/coefficient error ratio | `1.512e+01` |
| Stress-radius applied/needed correction | `0.797` |
| Stress-radius residual/needed correction | `0.203` |
| Stress-radius species-correction cancellation amplification | `63.139` |
| Stress-radius residual/species-correction L1 | `4.040e-03` |
| Stress-radius current condition number | `7.688e+01` |
| Required coefficient error for `1e-1` current gate | `1.301e-03` |
| Coefficient precision gap to current gate | `15.798x` |
| Production-grid coefficient precision gap | `15.875x` |
| Tight-harmonic coefficient precision gap | `15.797x` |
| Production radial/collisionality ladder count | `6` |
| Production ladder max coefficient difference | `2.065e-02` |
| Production ladder precision gap | `15.875x` |
| Coefficient-conditioned current-error bound | `1.580e+00` |
| Under-integrated closure current-gate passes | `1` |
| Quadrature-stable closure current-gate passes | `0` |
| Quadrature-stable current gate | `False` |
| Best stress-radius closure setting | `P=14, X=10, error=3.811e-02` |
| Highest-X largest-order stress error | `3.952e-01` |
| Max same-order stress spread over X | `9.495e+00` |
| Field-radius-matched reference stress error | `2.142e-01` |
| Field-radius-matched best apparent setting | `P=18, X=10, error=9.684e-02` |
| Field-radius-matched quadrature-stable pass count | `0` |
| Field-radius-matched highest-X largest-order error | `3.082e-01` |
| Field-radius-matched source-channel reconstruction residual | `1.452e-14` |
| Field-radius-matched source-channel reconstruction gate | `True` |
| Field-radius-matched high-order source-channel stress error | `3.082e-01` |
| Field-radius-matched Redl temperature response multiplier | `7.644e-01` |
| Source-channel reconstruction residual | `1.079e-14` |
| Source-channel reconstruction gate | `True` |
| High-order source-channel stress error | `3.952e-01` |
| Dominant high-order source channel | `effective_temperature_force` |
| High-order temperature/density/parallel fractions | `1.000e+00` / `1.254e-06` / `0.000e+00` |
| Source-channel species-cancellation factor | `8.260e+01` |
| Redl temperature response multiplier at high order | `7.167e-01` |
| Redl temperature-channel relative difference at high order | `3.952e-01` |
| Redl temperature-channel fraction of target current | `1.000e+00` |
| Profile source-response radii | `13` |
| Profile source-response max current stress | `3.952e-01` at `rho=0.143` |
| Profile temperature response multiplier min/median/max | `7.167e-01` / `1.010e+00` / `1.317e+00` |
| Profile temperature response multiplier span | `6.000e-01` |
| Temperature response correlation with log10(nu_e*) | `-1.398e-01` |
| Closure-target best physics driver | `epsilon` (`|r|=9.747e-01`) |
| Closure-target best diagnostic model | `epsilon` (`LOO RMSE=5.267e-02`) |
| Closure-target improvement over constant response | `3.922e+00` |
| Closure-target runtime correction applied | `False` |
| Closure-target matched-radius stress consistency | `same rho=True`, `source gate=True`, `stable gate=False` |
| Closure-target matched-radius apparent pass status | `under-integrated rejected=True`, `stable multiplier=7.644e-01` |
| Stress-radius Pmax error reduction | `3.548x` |
| Sonine-order max/RMS relative differences | `P=2: 1.10e+00/6.40e-01, P=4: 5.03e-01/2.43e-01, P=6: 4.01e-01/1.68e-01, P=8: 3.73e-01/1.52e-01, P=10: 3.46e-01/1.42e-01, P=12: 3.11e-01/1.30e-01` |

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
| Optimized scale | `1.255` |
| Weighted current gain | `1.114x` |
| Serial scan time | `0.981 s` |
| Parallel scan time | `2.570 s` |

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
| Commit | `c4bef899c2236f9bbfa9c032d9cf0fcc8fa8969d` |
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
