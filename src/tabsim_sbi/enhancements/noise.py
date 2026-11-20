"""E3: signal-dependent (self-)noise.

For circular Gaussian voltages with coherency matrix ``R`` (autocorrelations
``R_pp = P_p = SEFD + S_ast,p + sum_s |A_ps|^2`` on the diagonal), averaging
``M = chan_width * int_time`` samples gives a visibility estimate whose error
``eta`` satisfies (Kulkarni 1989)::

    <eta_pq conj(eta_rs)> = R_pr conj(R_qs) / M
    <eta_pq eta_rs>       = R_ps R_rq / M

Exact sampler (our construction): with ``R = L L^H`` (Cholesky) and ``G`` a
Hermitian matrix with independent unit-variance entries (GUE: diagonal N(0,1),
off-diagonal circular complex with E|G_pq|^2 = 1),

    eta = L G L^H / sqrt(M)

has exactly these second moments, because E[G_ij conj(G_kl)] = d_ik d_jl and
E[G_ij G_kl] = d_il d_jk. It is the leading (central-limit) term of the
complex-Wishart sample covariance, and costs two n_ant^3 matrix products per
integration. ``tests/test_noise.py`` checks it against the formulas and
against a brute-force voltage simulation.

When ``R`` varies within an integration (fast RFI), the visibility is the mean
of sub-block estimates; drawing each of ``n_blocks`` sub-blocks with
``M / n_blocks`` samples and averaging gives covariance
``(1 / (n_blocks M)) sum_b R_b (x) conj(R_b)``, the exact result for
piecewise-constant ``R``.

With R = SEFD * I the sampler reduces to tab-sim's i.i.d. noise with
per-component std ``SEFD / sqrt(2 M)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tabsim_sbi._jax import complex_normal, herm


def gue(key, shape):
    """Hermitian matrices (..., n, n): diag ~ N(0, 1) real, off-diag E|G|^2 = 1."""
    z = complex_normal(key, shape)
    return (z + herm(z)) / jnp.sqrt(2.0)


def self_noise(key, R, n_samples):
    """Self-noise matrices (..., n, n) for Hermitian positive-definite R (..., n, n)."""
    L = jnp.linalg.cholesky(R)
    G = gue(key, R.shape)
    return L @ G @ herm(L) / jnp.sqrt(n_samples)


def self_noise_blocks(key, R_blocks, n_samples):
    """Average of sub-block self-noise. R_blocks (n_time, n_blk, ..., n, n)."""
    n_blk = R_blocks.shape[1]
    eta = self_noise(key, R_blocks, n_samples / n_blk)
    return eta.mean(axis=1)


def diag_noise_std(P_blocks, a1, a2, n_samples):
    """Per-component std on each baseline for the diagonal approximation.

    P_blocks (n_time, n_blk, n_freq, n_ant) total power per antenna per
    sub-block. Complex variance ``mean_b(P_p P_q) / M``; per component half of it.
    Returns (n_time, n_bl, n_freq).
    """
    prod = P_blocks[..., a1] * P_blocks[..., a2]  # (t, b, f, bl)
    var_c = prod.mean(axis=1) / n_samples  # complex variance (t, f, bl)
    return jnp.swapaxes(jnp.sqrt(0.5 * var_c), -1, -2)


def thermal_noise(key, shape, std):
    """tab-sim's noise: i.i.d. circular complex, ``std`` per component."""
    return complex_normal(key, shape) * jnp.sqrt(2.0) * std


def noise_inflation(P_p, P_q, SEFD):
    """Factor by which the noise std exceeds thermal: sqrt((P_p / SEFD)(P_q / SEFD))."""
    return jnp.sqrt((P_p / SEFD) * (P_q / SEFD))
