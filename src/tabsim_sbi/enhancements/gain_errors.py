"""E5: stochastic gains beyond the model, and non-closing (baseline-based) errors.

tab-sim 0.1.0 already replaced linear gain drifts with a random-Fourier-feature
GP (``tabsim.dask.interferometry.generate_fourier_gains``). What the SBI
setting needs in addition is a gain component the *compression model cannot
represent*: the model interpolates a few inducing points per antenna, so a GP
residual with a correlation time shorter than the inducing-point spacing is
genuine mismatch. The simulator's gains are therefore

    g_p(t) = exp(a_p(t) + i phi_p(t)) * exp(da_p(t) + i dphi_p(t))

with ``(a, phi)`` from the shared :class:`~tabsim_sbi.gains.GainParameterisation`
(driven by theta) and ``(da, dphi)`` a latent SE-GP residual (std and
correlation time to be set).

Non-closing errors multiply each baseline, ``V_pq -> V_pq (1 + eps_pq)``, with
``eps_pq`` complex Gaussian, constant over the observation, ``eps_qp =
conj(eps_pq)`` so the coherency stays Hermitian.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tabsim_sbi._jax import complex_normal, se_kernel


def gp_residual(key, times, n, sigma, tau, jitter=1e-9):
    """Zero-mean SE-GP draws (n_time, n) with std ``sigma`` and length scale ``tau``."""
    K = se_kernel(times, times, tau) + jitter * jnp.eye(times.shape[0])
    L = jnp.linalg.cholesky(K)
    return sigma * (L @ jax.random.normal(key, (times.shape[0], n)))


def gain_residual(key, times, n_ant, sigma_amp, sigma_phase, tau, ref_ant=-1):
    """Multiplicative residual (n_time, n_ant); the reference antenna keeps zero phase."""
    ka, kp = jax.random.split(key)
    da = gp_residual(ka, times, n_ant, sigma_amp, tau)
    dp = gp_residual(kp, times, n_ant, sigma_phase, tau).at[:, ref_ant].set(0.0)
    return jnp.exp(da + 1j * dp)


def nonclosing_errors(key, n_ant, sigma):
    """Hermitian (n_ant, n_ant) matrix ``1 + eps`` with unit diagonal."""
    e = sigma * complex_normal(key, (n_ant, n_ant))
    e = jnp.triu(e, 1)
    return 1.0 + e + jnp.conj(e.T)
