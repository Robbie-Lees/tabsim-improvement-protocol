"""Matrix-free nuisance hardening for all blocks at once.

Alsing & Wandelt (2019) harden the score of interesting parameters theta
against nuisances eta, ``t_bar = t_theta - H_theta,eta H_eta,eta^-1 t_eta``. The
protocol uses the posterior information ``H = J^T J + Sigma_prior^-1`` (identity
prior in standardised coordinates), never formed: products ``H v`` are one JVP
and one VJP, and solves use conjugate gradients.

Block-wise inference hardens each block ``b`` of theta against *everything
else*, ``n = complement(b)``. Doing that block by block needs one CG solve in
the complement space per interesting parameter. We use instead the Schur
complement identity

    [H^-1 t]_b = S_b^-1 (t_b - H_bn H_nn^-1 t_n),    [H^-1]_bb = S_b^-1,

so that  ``t_bar_b = ([H^-1]_bb)^-1 X_b^T t``  with ``X = H^-1 E_theta``: the
same number of CG solves (one per interesting parameter, once per
observation), but they serve every block simultaneously and also give the full
Laplace covariance ``C = [H^-1]_theta,theta``, including the cross-block terms
the protocol needs to assemble per-block posteriors (Section 3.5).

Per simulation the cost is then a product ``X^T t`` -- and since the Gaussian
score is linear in the data, ``X^T t = (J X)^T (y - m*)`` can be precomputed
as a fixed ``(n_theta, n_data)`` projection.

Summaries are reported in parameter units, ``theta_hat_b = theta*_b + X_b^T t``
(one prior-regularised Fisher-scoring step); this is an invertible linear map
of ``t_bar_b`` per block, so it carries the same information and is better
scaled for density estimation.

Properties (tested in ``tests/test_hardening.py``): with zero prior precision
this is exactly Alsing & Wandelt's Fisher hardening and the expected summary
does not depend on eta to first order; with the prior included the residual
sensitivity is ``-H_bn H_nn^-1 P_nn``, i.e. only along nuisance directions the
data do not constrain.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg


class Hardener:
    def __init__(
        self,
        m_fn: Callable,
        phi_star,
        theta_idx: Sequence[int],
        blocks: Optional[Sequence[Sequence[int]]] = None,
        prior_precision: float = 1.0,
        tol: float = 1e-10,
        maxiter: Optional[int] = None,
        batch_size: int = 8,
    ):
        self.phi_star = jnp.asarray(phi_star)
        self.theta_idx = np.asarray(theta_idx, dtype=int)
        self.prior_precision = prior_precision
        dim = self.phi_star.shape[0]

        _, jvp = jax.linearize(m_fn, self.phi_star)
        vjp = jax.linear_transpose(jvp, self.phi_star)
        self._jvp, self._vjp = jvp, vjp

        def H(v):
            return vjp(jvp(v))[0] + prior_precision * v

        self.H = H
        n_t = self.theta_idx.size
        E = jnp.zeros((n_t, dim)).at[jnp.arange(n_t), self.theta_idx].set(1.0)

        def solve(e):
            return cg(H, e, tol=tol, atol=0.0, maxiter=maxiter)[0]

        self.X = jax.lax.map(solve, E, batch_size=batch_size)  # (n_theta, dim), rows = H^-1 e_i
        C = self.X[:, self.theta_idx]
        self.cov = 0.5 * (C + C.T)  # Laplace covariance of theta, [H^-1]_theta,theta

        if blocks is None:
            blocks = [self.theta_idx]
        pos = {int(j): i for i, j in enumerate(self.theta_idx)}
        self.blocks = [np.array([pos[int(j)] for j in b]) for b in blocks]
        self.S = [jnp.linalg.inv(self.cov[np.ix_(b, b)]) for b in self.blocks]

        # Projection of the whitened residual straight to summaries: (J X^T)^T.
        self.K = jax.vmap(jvp)(self.X)  # (n_theta, n_data), rows J x_i

    # ------------------------------------------------------------------
    @property
    def theta_star(self):
        return self.phi_star[self.theta_idx]

    def qmle(self, t):
        """Summaries in parameter units, theta* + X t (n_theta,)."""
        return self.theta_star + self.X @ t

    def qmle_from_residual(self, rho):
        """Same as ``qmle(J^T rho)`` for the Gaussian score, via the precomputed projection."""
        return self.theta_star + self.K @ rho

    def hardened_scores(self, t):
        """Alsing & Wandelt hardened score of each block, ``S_b X_b t``."""
        u = self.X @ t
        return [S @ u[b] for S, b in zip(self.S, self.blocks)]

    def block_summaries(self, s):
        """Split full summaries (..., n_theta) into blocks."""
        return [s[..., b] for b in self.blocks]

    def dense_hessian(self):
        """H as a dense matrix (small problems and tests only)."""
        dim = self.phi_star.shape[0]
        return jax.vmap(self.H)(jnp.eye(dim))
