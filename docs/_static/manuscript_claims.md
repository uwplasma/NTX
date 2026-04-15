# NTX Manuscript Claims

These are the current paper-facing technical claims derived directly from the
validated NTX artifacts.

- W7-X imported-workflow bootstrap-current convergence reaches a maximum relative error of `1.830e-02` on the fine `25 x 25 x 64` grid.
- The prepared implicit-adjoint derivative path matches direct reverse-mode with a maximum relative mismatch of `1.051e-05` on the committed derivative benchmark.
- The prepared derivative path reaches a best observed speedup of `3.709x` on the benchmarked electric-field scan.
- The differentiable bootstrap-current optimization example improves the weighted current proxy by `1.109x` on the committed W7-X study.
- On the heavy CPU benchmark, multiprocess execution reaches a best observed speedup of `1.790x`.
- On the heavy GPU benchmark, the current multiprocess path reaches a best observed speedup of `0.424x` with `1` healthy parallel GPU device(s), so the current paper should frame GPU multiprocess as a characterized execution mode rather than a throughput win.

These claims should be used consistently in the manuscript text, captions, and
response-to-reviewer notes.
