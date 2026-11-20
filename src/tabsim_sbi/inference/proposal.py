"""Simulation proposal around the fiducial, and the prior correction.

Simulations are drawn from ``p~(theta) = N(theta*, s^2 C)`` with ``C`` the
Laplace covariance from the hardener and ``s`` an inflation factor (protocol
Section 3.4: restrict proposals to the neighbourhood of the fit). An NPE
trained on proposal draws estimates ``q(theta | t) ~ p(t | theta) p~(theta)``;
the posterior under the actual prior (standard normal in the standardised
coordinates) follows by importance weighting with ``p(theta) / p~(theta)``,
here by sampling-importance-resampling. The effective sample size is reported:
a low value means the proposal was too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _mvn_logpdf(x, mean, cov):
    L = jnp.linalg.cholesky(cov)
    z = jax.scipy.linalg.solve_triangular(L, (x - mean).T, lower=True).T
    return -0.5 * jnp.sum(z**2, -1) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * mean.shape[0] * jnp.log(2 * jnp.pi)


@dataclass
class GaussianProposal:
    mean: jnp.ndarray
    cov: jnp.ndarray

    @classmethod
    def around_fit(cls, theta_star, laplace_cov, inflate=2.0):
        return cls(jnp.asarray(theta_star), inflate**2 * jnp.asarray(laplace_cov))

    def sample(self, key, n):
        return jax.random.multivariate_normal(key, self.mean, self.cov, (n,))

    def log_prob(self, theta):
        return _mvn_logpdf(theta, self.mean, self.cov)

    def marginal(self, idx):
        return GaussianProposal(self.mean[idx], self.cov[jnp.ix_(idx, idx)])


def standard_normal_logpdf(theta):
    return -0.5 * jnp.sum(theta**2, -1) - 0.5 * theta.shape[-1] * jnp.log(2 * jnp.pi)


def sir_correct(key, samples, log_prior, log_proposal, n_out):
    """Resample proposal-posterior samples to the prior-posterior. Returns (samples, ess)."""
    logw = log_prior - log_proposal
    w = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
    ess = 1.0 / jnp.sum(w**2)
    idx = jax.random.choice(key, samples.shape[0], (n_out,), p=w)
    return samples[idx], ess
