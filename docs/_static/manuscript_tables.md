# NTX Manuscript Tables

## Validation

| Grid `(N_theta, N_zeta, N_xi)` | Max relative error |
| --- | ---: |
| `(13, 17, 17)` | 1.038e+02 |
| `(17, 25, 33)` | 2.965e+01 |
| `(25, 25, 64)` | 1.830e-02 |

## Derivatives

| Quantity | Value |
| --- | ---: |
| Grid | `(7, 9, 6)` |
| `nu_hat` | `3.000e-04` |
| `E_r` scan | `1.000e-06` to `3.000e-03` |
| Max relative mismatch | `1.051e-05` |
| Best prepared speedup | `3.709x` |

## Bootstrap-Current Optimization

| Quantity | Value |
| --- | ---: |
| Harmonic `(m, n)` | `(0, -1)` |
| Baseline scale | `1.000` |
| Optimized scale | `1.233` |
| Weighted current gain | `1.109x` |
| Serial scan time | `1.710 s` |
| Parallel scan time | `2.811 s` |

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
| Commit | `1aa316b0c2ecc87dbf011ab3e00bc65409268560` |
| Branch | `main` |
| Python | `3.11.14` |
| JAX | `0.9.2` |
| NumPy | `2.4.4` |
| Platform | `macOS-14.4.1-arm64-arm-64bit` |
| Figure bundle | `python examples/make_publication_figures.py --figures main_text,supplement` |
| Main-text figures | `python examples/make_publication_figures.py --figures main_text` |
| Supplement figures | `python examples/make_publication_figures.py --figures supplement` |
| Artifact tables | `python scripts/build_manuscript_artifacts.py` |
| Validation subset | `python -m pytest -q tests/test_w7x_reference_benchmark.py tests/test_derivative_path_benchmark_example.py tests/test_bootstrap_current_optimization_example.py tests/test_manuscript_artifacts_script.py tests/test_make_publication_figures.py -k "subset_writes_manifest or bootstrap_subset_writes_manifest"` |
