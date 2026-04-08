# NTX

NTX is a JAX-native prototype solver for monoenergetic neoclassical transport in
stellarator flux surfaces. It solves the Legendre-space block-tridiagonal form of
the monoenergetic drift-kinetic equation described in Escoto's PhD thesis
([arXiv:2510.27513](https://arxiv.org/abs/2510.27513)).

The first implementation focuses on the monoenergetic geometric coefficients
`D11`, `D31`, `D13`, `D33`, and the Spitzer `D33` normalization. It is written as
pure JAX functions so that the same numerical path can run on CPU or GPU.

```bash
python -m pip install -e ".[dev,docs,io]"
pytest -m "not gpu"
ntx solve --example --nu-hat 1e-3 --er-hat 0.0 --n-theta 5 --n-zeta 5 --n-xi 6
ntx solve --dkes /path/to/ddkes2.data --nu-hat 1e-5 --er-hat 0.0 --n-theta 19 --n-zeta 79 --n-xi 180
ntx benchmark --dkes /path/to/ddkes2.data /path/to/reference_executable_Monoenergetic_Database.dat --nu-hat 1e-5 --er-hat 1e-3
```
