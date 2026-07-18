# Algorithm

NTX solves the monoenergetic drift-kinetic equation in the Legendre basis used
in Javier Escoto's PhD thesis,
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

## Core System

The continuous local model is the monoenergetic drift-kinetic equation with
parallel streaming, mirror force, radial-electric-field precession, and Lorentz
pitch-angle scattering. NTX projects that equation onto Legendre polynomials in
pitch angle. For Legendre mode `k`, the runtime system is

```{math}
L_k f^{(k-1)} + D_k f^{(k)} + U_k f^{(k+1)} = s^{(k)}.
```

The implementation uses:

- dense block operators in angular space
- forward Schur recursion over Legendre mode
- back-substitution of the low-order modes needed for transport coefficients
- no explicit matrix inverse

The main implementation files are:

- geometry and angular fields: `src/ntx/geometry.py`
- grids and differentiation matrices: `src/ntx/grids.py`
- operator blocks: `src/ntx/operators.py`
- solve path and scan path: `src/ntx/solver.py`
- transport post-processing: `src/ntx/transport.py`

## Surface Families

NTX currently supports:

- Boozer harmonic surfaces with scalar `psi_p`, `B_theta`, `B_zeta`
- VMEC harmonic surfaces with covariant and contravariant field components

Surface loaders live in:

- `src/ntx/io.py`
- `src/ntx/vmec.py`
- `src/ntx/booz.py`
- `src/ntx/vmex_backend.py`
- `src/ntx/vmex_vmec.py`

## Differentiable Lane

The imported solve path is JAX-native and supports:

- `jit`
- `vmap`
- gradient propagation through scan inputs
- in-memory NEOPAX array mapping

The CLI and file-writing path is optimized for usability, while the imported
path is the one intended for differentiable workflows.
