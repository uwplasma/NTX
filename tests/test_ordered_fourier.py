"""Fourier arithmetic checks without JAX; optional small GPU/AD check."""
import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'src/ntx/_geometry.py'


def numpy_evaluator():
    names = {'_sum_fourier_modes','evaluate_fourier_series'}
    functions = [node for node in ast.parse(SOURCE.read_text()).body
                 if isinstance(node,ast.FunctionDef) and node.name in names]
    namespace = dict(jnp=np,lax=SimpleNamespace(optimization_barrier=lambda value:value))
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
                              *functions],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(SOURCE),'exec'),namespace)
    return namespace


class OrderedFourierTests(unittest.TestCase):
    def test_zero_single_odd_even_and_162_mode_shapes_and_dtypes(self):
        reduce = numpy_evaluator()['_sum_fourier_modes']
        for width in (0,1,2,3,5,8,81,162):
            for dtype in (np.float32,np.float64):
                values = np.linspace(-.7,1.2,3*width,dtype=dtype).reshape(3,width)
                observed = reduce(values)
                expected = np.asarray([math.fsum(row) for row in values],dtype=dtype)
                np.testing.assert_allclose(observed,expected,rtol=20*np.finfo(dtype).eps,atol=20*np.finfo(dtype).eps)
                self.assertEqual(observed.dtype,values.dtype)
                self.assertEqual(observed.shape,(3,))

    def test_values_and_both_angle_derivatives_against_scalar_fourier_oracle(self):
        evaluate = numpy_evaluator()['evaluate_fourier_series']
        m=np.arange(162)%7
        n=np.arange(162)%9-4
        c=np.sin(np.arange(162)+.2)/200
        s=np.cos(np.arange(162)+.3)/300
        theta,zeta=np.meshgrid(np.linspace(0,2*np.pi,5),np.linspace(0,np.pi,7),indexing='ij')
        for sine in (None,s):
            actual=evaluate(m,n,c,theta,zeta,nfp=2,sin_coeffs=sine)
            expected=np.zeros((3,*theta.shape))
            for index in np.ndindex(theta.shape):
                phase=theta[index]*m+zeta[index]*(2*n)
                terms=c*np.cos(phase)
                dt=-c*m*np.sin(phase)
                dz=-c*(2*n)*np.sin(phase)
                if sine is not None:
                    terms=terms+sine*np.sin(phase)
                    dt=dt+sine*m*np.cos(phase)
                    dz=dz+sine*(2*n)*np.cos(phase)
                for row,values in enumerate((terms,dt,dz)):
                    expected[row][index]=math.fsum(values)
            np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-15)

    def test_constant_and_single_mode_have_the_exact_analytic_behavior(self):
        jax_was_imported='jax' in sys.modules
        evaluate=numpy_evaluator()['evaluate_fourier_series']
        value,dt,dz=evaluate(np.asarray([0,1]),np.asarray([0,0]),np.asarray([2.,.5]),
                             np.asarray([0.,np.pi/2]),np.zeros(2),nfp=2)
        np.testing.assert_allclose(value,[2.5,2.],atol=1e-15)
        np.testing.assert_allclose(dt,[0.,-.5],atol=1e-15)
        np.testing.assert_array_equal(dz,[0.,0.])
        self.assertEqual(jax_was_imported,'jax' in sys.modules)

    @unittest.skipUnless(os.environ.get('NTX_TEST_ORDERED_FOURIER_GPU')=='1','GPU/AD check is opt-in')
    def test_gpu_jit_jvp_and_native_coefficient_transpose(self):
        import jax
        import jax.numpy as jnp
        from ntx.geometry import evaluate_fourier_series
        from ntx._geometry import _sum_fourier_modes
        from ntx._neopax_ad import fourier_series_coefficient_bars_multi_rhs
        self.assertEqual(jax.default_backend(),'gpu')
        self.assertTrue(jax.config.x64_enabled)
        self.assertTrue(callable(_sum_fourier_modes))
        m=jnp.arange(162)%7
        n=jnp.arange(162)%9-4
        c=jnp.sin(jnp.arange(162)+.2)/200
        s=jnp.cos(jnp.arange(162)+.3)/300
        theta,zeta=jnp.meshgrid(jnp.linspace(0,2*jnp.pi,5),jnp.linspace(0,jnp.pi,7),indexing='ij')
        def evaluate(cc,ss):
            return evaluate_fourier_series(m,n,cc,theta,zeta,nfp=2,sin_coeffs=ss)
        compiled=jax.jit(evaluate)
        first=compiled(c,s)
        second=compiled(c,s)
        for a,b in zip(first,second,strict=True):
            np.testing.assert_array_equal(a,b)
        _,dot=jax.jvp(compiled,(c,s),(jnp.ones_like(c),jnp.zeros_like(s)))
        expected=evaluate(jnp.ones_like(c),jnp.zeros_like(s))
        for a,b in zip(dot,expected,strict=True):
            np.testing.assert_allclose(a,b,rtol=3e-13,atol=3e-13)
        bars=tuple(jnp.cos(theta+(i+1)*zeta) for i in range(3))
        _,pullback=jax.vjp(compiled,c,s)
        actual=pullback(bars)
        native=fourier_series_coefficient_bars_multi_rhs(m,n,theta,zeta,nfp=2,
            value_bar=bars[0][None],d_theta_bar=bars[1][None],d_zeta_bar=bars[2][None])
        for a,b in zip(actual,native,strict=True):
            np.testing.assert_allclose(a,b[0],rtol=3e-12,atol=3e-12)


if __name__=='__main__':
    unittest.main()
