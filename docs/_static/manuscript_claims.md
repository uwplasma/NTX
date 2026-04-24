# NTX Manuscript Claims

These are the current paper-facing technical claims derived directly from the
validated NTX artifacts.

- The monoenergetic validation-summary gate keeps the committed DKES-style and VMEC finest plotted `N_xi` convergence errors at `1.671e-01` and `1.969e-01`; the DKES-style max Onsager residual is `2.785e-06`, while the VMEC Onsager residual is retained as a monitored finite-resolution stress metric at `1.782e+00`.
- The fixed-field precise-QS benchmark keeps the Redl/SFINCS interior maximum relative error at `6.857e-02`; the corresponding `NTX+NEOPAX` current comparison remains a closure stress metric at `1.162e+00`, not a release parity claim.
- W7-X imported-workflow bootstrap-current convergence reaches a maximum relative error of `1.830e-02` on the fine `25 x 25 x 64` grid.
- The prepared implicit-adjoint derivative path matches direct reverse-mode with a maximum relative mismatch of `1.051e-05` on the committed derivative benchmark.
- The prepared derivative path reaches a best observed speedup of `4.033x` on the benchmarked electric-field scan.
- The three-harmonic geometry-control derivative stress benchmark matches centered finite differences with a maximum relative mismatch of `1.348e-04` and a median mismatch of `3.595e-06`.
- The file-backed Boozer and VMEC geometry-control derivative stress benchmark matches centered finite differences with a maximum relative mismatch of `3.088e-04` and a median mismatch of `5.158e-08`.
- The boundary-projected `vmec_jax -> booz_xform_jax -> NTX` and `NTX+NEOPAX` forward-mode stress benchmark matches centered finite differences with a maximum relative mismatch of `7.174e-07` and a median mismatch of `3.556e-07`.
- The implicit fixed-boundary `vmec_jax -> booz_xform_jax -> NTX` diagnostic is closed as non-shipping on the committed QA case: the equilibrium-volume derivative matches centered finite differences with relative mismatch `9.236e-05`, while the Boozer scalar and NTX transport observables fail the surface/transport parity contract at `8.544e-01` and `6.454e+00`.
- The matching reverse-mode Boozer-scalar diagnostic on the non-shipping implicit-equilibrium diagnostic remains unavailable because the current JAX transform rejects the implicit dynamic-loop solve on that path.
- The explicit-relaxed `vmec_jax -> booz_xform_jax -> NTX` and `NTX+NEOPAX` boundary-to-current QA/QH stress benchmark matches centered finite differences with a maximum relative mismatch of `2.626e-05` and a median mismatch of `4.408e-07`, while the ordinary and explicit-relaxed primal volumes agree to `0.000e+00` on the committed QA/QH family cases.
- The artifact-backed geometry-family breadth summary now covers `7` active analytic, file-backed, boundary-projected, explicit-relaxed, and implicit-volume stress cases with maximum active mismatch `3.088e-04`. The implicit Boozer and NTX transport objectives are closed as non-shipping diagnostics with maximum mismatch `6.454e+00` and are excluded from promoted geometry-family claims.
- The geometry-family transport convergence stress diagnostic solves `11` public VMEC-family cases, with `5` below the smoke-grid convergence tolerance and maximum last-step relative D11/D31/D33 change `1.952e+00`. It is a reduced NTX convergence diagnostic, not an independent-code parity claim.
- The differentiable bootstrap-current optimization example improves the weighted current proxy by `1.085x` on the committed W7-X study.
- On the heavy CPU benchmark, multiprocess execution reaches a best observed speedup of `0.591x`.
- On the heavy GPU benchmark, the current multiprocess path reaches a best observed speedup of `0.477x` with `1` healthy parallel GPU device(s), so the current paper should frame GPU multiprocess as a characterized execution mode rather than a throughput win.
- On the prepared-geometry reuse profile, the compiled steady solver reaches a best observed speedup of `150.369x` against direct repeated solves with maximum coefficient mismatch `1.856e-09`.

These claims should be used consistently in the manuscript text, captions, and
response-to-reviewer notes.
