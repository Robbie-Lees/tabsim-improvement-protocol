"""Score compression (Alsing & Wandelt 2018) with an optional Student-t score.

All operations use the *real, whitened* form of the model: for complex data
with per-component noise std ``sigma`` (tab-sim's ``noise_std``; tabascal's
likelihood is ``Normal(re/im, sigma)`` with the same ``sigma``),

    m(phi) = [Re mu(phi), Im mu(phi)] / sigma,      y = [Re d, Im d] / sigma,
    log L  = -1/2 ||y - m(phi)||^2 + const,
    t      = grad log L |_phi* = J^T (y - m*),       J = dm/dphi at phi*,
    F      = J^T J = 2 Re[J_c^H C^-1 J_c],  C = 2 sigma^2 the complex variance.

This avoids complex-VJP conventions, and agrees with the protocol's complex
expression ``t = 2 Re[grad mu^H C^-1 (d - mu)]``.

Because ``phi*`` is frozen, the Gaussian score is *linear in the data*:
``t = J^T (y - m*)`` with ``J`` fixed. Compressing a simulation costs one
vector-Jacobian product (or, with the projection precomputed by the hardener,
one small matrix-vector product), far less than simulating it.

Student-t score (protocol Section 3.2): per complex datum the weight
``w = (nu + 2) / (nu + |r|^2 / sigma^2)`` multiplies its whitened residual,
with ``|r|^2 / sigma^2 = rho_re^2 + rho_im^2`` -- the protocol's
``(nu + 2) / (nu + 2 |r|^2 / sigma_n^2)`` with ``sigma_n^2 = 2 sigma^2``. It
tends to the Gaussian score as ``nu -> infinity``. The weights depend on the
data, but they are a fixed function applied identically to data and
simulations, so the inference stays valid.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp


def realify(z):
    """Complex (...,) -> real (2 * size,) as [Re, Im]."""
    z = jnp.ravel(z)
    return jnp.concatenate([jnp.real(z), jnp.imag(z)])


class WhitenedModel:
    """Real, whitened model ``m(phi)`` and data ``y(d)``.

    ``sigma`` is the per-component std: a scalar or an array broadcastable to
    ``mu``'s shape (e.g. the E3 signal-dependent sigma at the fiducial point).
    """

    def __init__(self, mu_fn: Callable, sigma):
        self.mu_fn = mu_fn
        self.sigma = sigma

    def _w(self, z):
        return realify(z / self.sigma)

    def m(self, phi):
        return self._w(self.mu_fn(phi))

    def y(self, d):
        return self._w(d)

    def chi2_per_dof(self, phi, d):
        r = self.y(d) - self.m(phi)
        return jnp.mean(r**2)


class ScoreCompressor:
    """Score of the (Gaussian or Student-t) compression likelihood at frozen phi*."""

    def __init__(self, wm: WhitenedModel, phi_star, nu: Optional[float] = None):
        self.wm = wm
        self.phi_star = phi_star
        self.nu = nu
        self.m_star, self._vjp = jax.vjp(wm.m, phi_star)

    def residual(self, d):
        return self.wm.y(d) - self.m_star

    def weights(self, rho):
        if self.nu is None:
            return jnp.ones_like(rho)
        n = rho.shape[0] // 2
        q = rho[:n] ** 2 + rho[n:] ** 2
        w = (self.nu + 2.0) / (self.nu + q)
        return jnp.concatenate([w, w])

    def score(self, d):
        rho = self.residual(d)
        return self._vjp(self.weights(rho) * rho)[0]

    def batch_score(self, D):
        return jax.vmap(self.score)(D)
