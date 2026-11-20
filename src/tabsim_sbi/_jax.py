"""Shared JAX setup and small numerical helpers.

Importing :mod:`tabsim_sbi` enables JAX x64. tab-sim requires it too: RFI
fringe phases are differences of ~2e7 m distances resolved to a fraction of a
wavelength, which float32 cannot represent.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

C_LIGHT = 2.99792458e8  # m/s
OMEGA_EARTH = 7.2921150e-5  # rad/s, sidereal rotation rate
GM_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6.371e6  # m, spherical Earth used only by synthetic geometry


def complex_normal(key, shape):
    """Circular complex normal with E|z|^2 = 1 (variance 1/2 per component)."""
    return jax.random.normal(key, shape, dtype=jnp.complex128)


def herm(x):
    """Conjugate transpose of the last two axes."""
    return jnp.conj(jnp.swapaxes(x, -1, -2))


def unit(x, axis=-1):
    return x / jnp.linalg.norm(x, axis=axis, keepdims=True)


def se_kernel(t1, t2, length_scale):
    """Squared-exponential kernel k(t, t') = exp(-(t - t')^2 / (2 l^2))."""
    d = jnp.asarray(t1)[:, None] - jnp.asarray(t2)[None, :]
    return jnp.exp(-0.5 * (d / length_scale) ** 2)


def gp_basis(t_out, t_ind, length_scale, jitter=1e-8):
    """Matrix P with u(t_out) = P z for inducing values u_ind = L z, z ~ N(0, I).

    The GP conditional mean interpolates inducing values, u(t) = K_ti K_ii^-1 u_i,
    and with u_i = L z (K_ii = L L^T) this is P z with P = K_ti L^-T. Working in
    z makes the prior the identity, which is what the score and the hardening
    assume (tabascal's standardised coordinates).
    """
    t_ind = jnp.atleast_1d(jnp.asarray(t_ind, dtype=jnp.float64))
    t_out = jnp.asarray(t_out, dtype=jnp.float64)
    if t_ind.shape[0] == 1:
        return jnp.ones((t_out.shape[0], 1))
    K_ii = se_kernel(t_ind, t_ind, length_scale) + jitter * jnp.eye(t_ind.shape[0])
    L = jnp.linalg.cholesky(K_ii)
    K_ti = se_kernel(t_out, t_ind, length_scale)
    # P = K_ti L^-T  <=>  P^T = L^-1 K_ti^T
    return jax.scipy.linalg.solve_triangular(L, K_ti.T, lower=True).T


def inducing_times(t_start, t_end, spacing=None, n=None):
    """Evenly spaced inducing points covering [t_start, t_end]."""
    if n is None:
        n = int(max(1, round((t_end - t_start) / spacing) + 1))
    if n == 1:
        return jnp.array([0.5 * (t_start + t_end)])
    return jnp.linspace(t_start, t_end, n)
