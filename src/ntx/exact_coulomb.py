from __future__ import annotations

from math import factorial, gamma

import numpy as np


def l1_lambda_factors(n_order: int) -> np.ndarray:
    """Return lambda_k^{l=1} from the exact Laguerre normalization."""

    return np.asarray(
        [
            gamma(1 + k + 1.5) / (factorial(k) * gamma(1.5))
            for k in range(n_order)
        ],
        dtype=np.float64,
    )


def l1_sigma_factors(n_order: int) -> np.ndarray:
    """Return sigma_{1k} for the exact l=1 moment expansion."""

    sigma_l = factorial(1) * gamma(1.5) / (2.0**1 * gamma(1 + 1.5))
    return sigma_l * l1_lambda_factors(n_order)


def l1_runtime_basis_factors(n_order: int) -> np.ndarray:
    r"""Return the exact map from dimensionless l=1 moments to runtime coefficients.

    For the runtime ansatz

      delta f = f_M (m v_parallel / T) sum_p c_p L_p^{(3/2)}(x^2)

    and the exact moment expansion

      f = sum_k sigma_{1k}^{-1} f_M P_1 L_k^{(3/2)} \hat M_{1k},

    the low-order runtime coefficients satisfy

      c_k = v_th * [1 / (2 sigma_{1k})] * \hat M_{1k}.
    """

    sigma = l1_sigma_factors(n_order)
    return 1.0 / (2.0 * sigma)


def l1_runtime_coefficients_from_exact_moments(
    v_thermal: float,
    mhat_l1: np.ndarray,
) -> np.ndarray:
    mhat = np.asarray(mhat_l1, dtype=np.float64)
    return float(v_thermal) * l1_runtime_basis_factors(mhat.shape[0]) * mhat


def l1_c0_c1_from_flow_heat(
    flow_velocity: float,
    heat_flux: float,
    density: float,
    temperature: float,
) -> tuple[float, float]:
    """Return the first two runtime coefficients from physical flow and heat flux."""

    v = float(flow_velocity)
    q = float(heat_flux)
    n = float(density)
    t = float(temperature)
    c0 = v
    c1 = v - (2.0 * q) / (5.0 * n * t)
    return c0, c1


def appendix_b_l1_a(theta: float, mu: float) -> np.ndarray:
    """Exact l=1 Appendix-B a_{pk} coefficients from Ji & Held (2006)."""

    th = float(theta)
    mu0 = float(mu)
    return np.asarray(
        [
            [
                1.0 / (2.0 * th) + mu0 / (2.0 * th),
                3.0 / (4.0 * th) + 3.0 * mu0 / (4.0 * th),
                15.0 / (16.0 * th) + 15.0 * mu0 / (16.0 * th),
            ],
            [
                1.0
                - 1.0 / (4.0 * th)
                - 5.0 / (2.0 * mu0)
                + 5.0 * th / (2.0 * mu0)
                + 3.0 * mu0 / (4.0 * th),
                5.0 / 2.0
                + 9.0 / (8.0 * th)
                + 15.0 * th / (4.0 * mu0**2)
                - 3.0 / (4.0 * mu0)
                + 13.0 * th / (2.0 * mu0)
                + 13.0 * mu0 / (8.0 * th),
                27.0 / 8.0
                + 57.0 / (32.0 * th)
                + 63.0 * th / (8.0 * mu0**2)
                - 3.0 / (16.0 * mu0)
                + 177.0 * th / (16.0 * mu0)
                + 69.0 * mu0 / (32.0 * th),
            ],
            [
                3.0 / 2.0
                - 9.0 / (16.0 * th)
                - 21.0 / (4.0 * mu0)
                + 21.0 * th / (4.0 * mu0)
                + 15.0 * mu0 / (16.0 * th),
                23.0 / 4.0
                - 19.0 / (32.0 * th)
                - 35.0 * th**2 / (4.0 * mu0**3)
                + 35.0 * th**3 / (4.0 * mu0**3)
                - 21.0 * th / (8.0 * mu0**2)
                + 21.0 * th**2 / (2.0 * mu0**2)
                - 87.0 / (8.0 * mu0)
                + 87.0 * th / (4.0 * mu0)
                + 69.0 * mu0 / (32.0 * th),
                161.0 / 16.0
                + 233.0 / (128.0 * th)
                + 175.0 * th**3 / (16.0 * mu0**4)
                - 7.0 * th**2 / (8.0 * mu0**3)
                + 413.0 * th**3 / (16.0 * mu0**3)
                + 303.0 * th / (16.0 * mu0**2)
                + 95.0 * th**2 / (4.0 * mu0**2)
                - 139.0 / (32.0 * mu0)
                + 1329.0 * th / (32.0 * mu0)
                + 433.0 * mu0 / (128.0 * th),
            ],
        ],
        dtype=np.float64,
    )


def appendix_b_l1_b(theta: float, mu: float) -> np.ndarray:
    """Exact l=1 Appendix-B b_{pk} coefficients from Ji & Held (2006)."""

    th = float(theta)
    mu0 = float(mu)
    return np.asarray(
        [
            [
                1.0 / 2.0 + 1.0 / (2.0 * mu0),
                3.0 / 4.0 + 3.0 / (4.0 * mu0),
                15.0 / 16.0 + 15.0 / (16.0 * mu0),
            ],
            [
                3.0 / 4.0 + 9.0 / (4.0 * mu0) - 3.0 * th / (2.0 * mu0),
                27.0 / 8.0 + 45.0 / (8.0 * mu0) - 9.0 * th / (4.0 * mu0),
                225.0 / 32.0 + 315.0 / (32.0 * mu0) - 45.0 * th / (16.0 * mu0),
            ],
            [
                15.0 / 16.0 + 75.0 / (16.0 * mu0) - 15.0 * th / (4.0 * mu0),
                225.0 / 32.0 + 525.0 / (32.0 * mu0) - 75.0 * th / (8.0 * mu0),
                2625.0 / 128.0 + 4725.0 / (128.0 * mu0) - 525.0 * th / (32.0 * mu0),
            ],
        ],
        dtype=np.float64,
    )


def exact_l1_like_species_runtime_blocks() -> tuple[np.ndarray, np.ndarray]:
    """Return the exact low-order l=1 like-species blocks in runtime basis.

    The common prefactor 4 * nu / sqrt(pi) is omitted. What remains is the
    exact matrix shape in the runtime Sonine coefficient basis.
    """

    a = appendix_b_l1_a(theta=1.0, mu=1.0)
    b = appendix_b_l1_b(theta=1.0, mu=1.0)
    lam = l1_lambda_factors(3)
    basis = np.diag(l1_runtime_basis_factors(3))
    abar = -a / lam[np.newaxis, :]
    bbar = b / lam[np.newaxis, :]
    return basis @ abar @ np.linalg.inv(basis), basis @ bbar @ np.linalg.inv(basis)


def active_p2_like_species_reference_blocks() -> tuple[np.ndarray, np.ndarray]:
    """Return the current low-order runtime collision blocks for theta=mu=1."""

    xab2 = 1.0
    one_plus = 1.0 + xab2
    yab32 = one_plus**1.5
    yab52 = one_plus**2.5
    yab72 = one_plus**3.5
    yab92 = one_plus**4.5

    capm = np.zeros((3, 3), dtype=np.float64)
    capn = np.zeros((3, 3), dtype=np.float64)

    capm[0, 0] = -(1.0 + 1.0) / yab32
    capm[0, 1] = 1.5 * (1.0 + 1.0) / yab52
    capm[1, 0] = capm[0, 1]
    capm[1, 1] = -(13.0 / 4.0 + xab2 * (4.0 + xab2 * 15.0 / 2.0)) / yab52
    capm[0, 2] = -(15.0 / 8.0) * (1.0 + 1.0) / yab72
    capm[1, 2] = -(69.0 / 16.0 + xab2 * (6.0 + xab2 * 63.0 / 4.0)) / yab72
    capm[2, 0] = capm[0, 2]
    capm[2, 1] = capm[1, 2]
    capm[2, 2] = (
        -(
            433.0 / 64.0
            + xab2 * (111.0 / 8.0 + xab2 * (95.0 / 4.0 + xab2 * 175.0 / 16.0))
        )
        / yab92
    )

    capn[0, 0] = -capm[0, 0]
    capn[0, 1] = -xab2 * capm[0, 1]
    capn[1, 0] = -capm[1, 0]
    capn[1, 1] = (27.0 / 4.0) * np.sqrt(1.0) * xab2 / yab52
    capn[0, 2] = -(xab2**2) * capm[0, 2]
    capn[1, 2] = -(225.0 / 16.0) * 1.0 * (xab2**2) / yab72
    capn[2, 0] = -capm[2, 0]
    capn[2, 1] = -(225.0 / 16.0) * (xab2**2) / yab72
    capn[2, 2] = (2625.0 / 64.0) * 1.0 * (xab2**2) / yab92

    sign = np.ones(3, dtype=np.float64)
    sign[1] = -1.0
    transform = np.diag(sign)
    return transform @ capm @ transform, transform @ capn @ transform
