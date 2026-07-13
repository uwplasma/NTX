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
| `angular_oversampling_audit` | `geometry-breadth` | `stress-gate` | `complete` |
| `boozmn_same_coordinate_roundtrip` | `geometry-breadth` | `positive-gate` | `complete` |
| `boozmn_finite_beta_wout_roundtrip` | `geometry-breadth` | `positive-gate` | `complete` |
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
| Max relative mismatch | `7.451e-07` |
| Best prepared speedup | `2.556x` |

## Geometry-Control Derivatives

| Quantity | Value |
| --- | ---: |
| Grid | `(7, 9, 6)` |
| Controlled modes | `3` |
| Coefficients | `D11, D31, D33` |
| Max AD/centered-FD mismatch | `1.346e-04` |
| Median AD/centered-FD mismatch | `3.660e-06` |

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
| Solved VMEC cases | `11` |
| Below production convergence rtol | `10` |
| Max last-step relative change | `7.144e-02` |
| Max relative change to finest grid | `1.667e-01` |
| Solved case ids | `nfp2_qa, nfp2_qa_finite_beta, precise_qs_qh_reactor, w7x_eim_ejm_standard, nfp4_qh_reference, high_aspect_qs, li383_low_res, n3are_lowres, lhd, hsx_qhs, ncsx` |

## Owned Finite-Beta Bootstrap-Current Stress

| Quantity | Value |
| --- | ---: |
| Case | `finite_beta_qa_pressure_current` |
| Closure configuration | `P=12`, `D33=spitzer`, `nu/v points=17` |
| Boozer psi_p | `1.334630e-02` |
| Max total-current relative difference vs Redl | `2.193e-01` |
| RMS total-current relative difference vs Redl | `1.443e-01` |
| Sign-agreement fraction | `1.000` |
| Stress-gap same-grid coefficient relative difference | `1.254e-02` |
| Stress-gap profile-current relative difference | `2.193e-01` |
| Stress-gap current/coefficient error ratio | `1.749e+01` |
| Stress-radius applied/needed correction | `2.100` |
| Stress-radius residual/needed correction | `-1.100` |
| Stress-radius species-correction cancellation amplification | `212.898` |
| Stress-radius residual/species-correction L1 | `2.460e-03` |
| Stress-radius current condition number | `8.912e+01` |
| Required coefficient error for `1e-1` current gate | `1.122e-03` |
| Coefficient precision gap to current gate | `29.804x` |
| Production-grid coefficient precision gap | `15.875x` |
| Tight-harmonic coefficient precision gap | `15.797x` |
| Production radial/collisionality ladder count | `6` |
| Production ladder max coefficient difference | `2.065e-02` |
| Production ladder precision gap | `29.948x` |
| Coefficient-conditioned current-error bound | `2.980e+00` |
| Under-integrated closure current-gate passes | `0` |
| Quadrature-stable closure current-gate passes | `0` |
| Quadrature-stable current gate | `False` |
| Best stress-radius closure setting | `P=12, X=18, error=1.162e-01` |
| Highest-X largest-order stress error | `1.271e-01` |
| Max same-order stress spread over X | `2.253e+01` |
| Field-radius-matched reference stress error | `2.273e-01` |
| Field-radius-matched best apparent setting | `P=10, X=18, error=1.240e-01` |
| Field-radius-matched quadrature-stable pass count | `0` |
| Field-radius-matched highest-X largest-order error | `1.441e-01` |
| Field-radius-matched source-channel reconstruction residual | `6.107e-14` |
| Field-radius-matched source-channel reconstruction gate | `True` |
| Field-radius-matched high-order source-channel stress error | `1.441e-01` |
| Field-radius-matched Redl temperature response multiplier | `1.356e+00` |
| Source-channel reconstruction residual | `2.011e-14` |
| Source-channel reconstruction gate | `True` |
| High-order source-channel stress error | `1.271e-01` |
| Dominant high-order source channel | `density_electric_force` |
| High-order temperature/density/parallel fractions | `4.234e-01` / `5.766e-01` / `0.000e+00` |
| Source-channel species-cancellation factor | `1.606e+02` |
| Redl temperature response multiplier at high order | `1.347e+00` |
| Redl temperature-channel relative difference at high order | `2.575e-01` |
| Redl temperature-channel fraction of target current | `4.977e-01` |
| Profile source-response radii | `13` |
| Profile source-response max current stress | `3.077e-01` at `rho=0.143` |
| Profile temperature response multiplier min/median/max | `7.647e-01` / `1.040e+00` / `1.349e+00` |
| Profile temperature response multiplier span | `5.845e-01` |
| Temperature response correlation with log10(nu_e*) | `-1.280e-01` |
| Closure-target best physics driver | `epsilon` (`|r|=9.705e-01`) |
| Closure-target best diagnostic model | `epsilon` (`LOO RMSE=5.576e-02`) |
| Closure-target improvement over constant response | `3.679e+00` |
| Closure-target runtime correction applied | `False` |
| Closure-target matched-radius stress consistency | `same rho=True`, `source gate=True`, `stable gate=False` |
| Closure-target matched-radius pass status | `best-pass rejected=False`, `stable multiplier=1.356e+00` |
| Stress-radius Pmax error reduction | `0.233x` |
| Sonine-order max/RMS relative differences | `P=2: 1.04e+00/6.19e-01, P=4: 4.31e-01/2.09e-01, P=6: 3.25e-01/1.40e-01, P=8: 2.95e-01/1.34e-01, P=10: 2.59e-01/1.35e-01, P=12: 2.19e-01/1.44e-01` |

## Profile Uncertainty

| Quantity | Value |
| --- | ---: |
| Radial electric-field basis size | `3` |
| Monte Carlo samples | `96` |
| Max linearized/Monte-Carlo std mismatch | `6.718e-02` |
| Max Monte-Carlo mean shift | `7.340e-07` |
| Fisher eigenvalue range | `-3.680e-20` to `2.151e-02` |
| Hessian-vector/Fisher probe mismatch | `1.008e-17` |

## Bootstrap-Current Optimization

| Quantity | Value |
| --- | ---: |
| Harmonic `(m, n)` | `(0, -1)` |
| Baseline scale | `1.000` |
| Optimized scale | `1.255` |
| Weighted current gain | `1.114x` |
| Serial scan time | `0.344 s` |
| Parallel scan time | `1.520 s` |

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
| 4 | 0.584 | 0.599 | 0.005 | 119.229x |
| 16 | 2.322 | 2.317 | 0.020 | 117.027x |
| 48 | 7.121 | 7.019 | 0.057 | 124.523x |

## Reproducibility

| Key | Value |
| --- | --- |
| Commit | `8d31e9c1615bbfa02fba04e086145bff70c7722d` |
| Branch | `feat/angular-oversampling-audit` |
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
