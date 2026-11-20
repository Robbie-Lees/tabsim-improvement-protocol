"""Fiducial point: a MAP fit of the compression model to the observed data.

Objective (standardised coordinates): ``1/2 ||y - m(phi)||^2 + 1/2 p ||phi||^2``.
A short Adam warm-up (robust far from the optimum) is followed by
Levenberg-Marquardt Gauss-Newton steps, each a matrix-free CG solve with the
same ``J^T J + p I`` operator the hardening uses, which converges in far fewer
iterations than first-order methods on this least-squares problem.

:class:`MapFitter` compiles once per model and takes the data as an argument,
so repeated fits (B0, or one fit per simulation) do not recompile. Building
fresh jitted closures per call recompiled for every simulation and, on CPU,
occasionally hit multi-hour XLA compiles.

The chi^2 gate follows Paper II (chi^2 per real data point below 1.1). A poor
fiducial costs optimality, not validity -- the same compression is applied to
data and simulations -- so the gate decides whether to proceed *with reduced
expectations*, not whether to trust the result (protocol Section 3.4). The
gate is only meaningful with the right noise: under E3 the noise exceeds
thermal by sqrt(P_p P_q) / SEFD, so a thermal-sigma chi^2 rejects good fits
(``chi2_with`` re-evaluates it with another sigma).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from tabsim_sbi.optim import adam_init, adam_update


@dataclass
class FitResult:
    phi: jnp.ndarray
    chi2_per_dof: float
    passed: bool
    history: List[float] = field(default_factory=list)


class MapFitter:
    def __init__(
        self,
        m_fn: Callable,
        *,
        prior_precision: float = 1.0,
        n_adam: int = 300,
        lr: float = 0.02,
        n_gn: int = 30,
        lm_lambda: float = 1e-2,
        cg_maxiter: int = 200,
        gate: float = 1.1,
        rtol: float = 1e-10,
    ):
        self.m_fn = m_fn
        self.p = prior_precision
        self.n_adam, self.n_gn, self.lm_lambda, self.gate, self.rtol = n_adam, n_gn, lm_lambda, gate, rtol
        p = prior_precision

        def loss(phi, y):
            r = y - m_fn(phi)
            return 0.5 * jnp.sum(r**2) + 0.5 * p * jnp.sum(phi**2)

        vg = jax.value_and_grad(loss)

        def adam_run(phi, y):
            def step(carry, _):
                q, s = carry
                val, g = vg(q, y)
                q, s = adam_update(g, s, q, lr)
                return (q, s), val

            (phi, _), vals = jax.lax.scan(step, (phi, adam_init(phi)), None, length=max(n_adam, 1))
            return phi, vals

        def gn_step(phi, lam, y):
            r, jvp = jax.linearize(m_fn, phi)
            vjp = jax.linear_transpose(jvp, phi)
            g = vjp(y - r)[0] - p * phi
            Hop = lambda v: vjp(jvp(v))[0] + (p + lam) * v
            new = phi + cg(Hop, g, maxiter=cg_maxiter)[0]
            return new, loss(new, y)

        self._loss = jax.jit(loss)
        self._adam = jax.jit(adam_run)
        self._gn = jax.jit(gn_step)
        self._chi2 = jax.jit(lambda phi, y: jnp.mean((y - m_fn(phi)) ** 2))

    def fit(self, y, phi0) -> FitResult:
        y = jnp.asarray(y)
        phi = jnp.asarray(phi0)
        history = []
        if self.n_adam:
            phi, losses = self._adam(phi, y)
            history += [float(x) for x in losses[:: max(1, self.n_adam // 20)]]
        cur = float(self._loss(phi, y))
        lam = self.lm_lambda
        for _ in range(self.n_gn):
            cand, cl = self._gn(phi, lam, y)
            cl = float(cl)
            if cl < cur:
                done = (cur - cl) < self.rtol * max(abs(cur), 1.0)
                phi, cur, lam = cand, cl, max(lam / 3.0, 1e-8)
                history.append(cur)
                if done:
                    break
            else:
                lam *= 5.0
                if lam > 1e8:
                    break
        chi2 = float(self._chi2(phi, y))
        return FitResult(phi=phi, chi2_per_dof=chi2, passed=chi2 < self.gate, history=history)


def map_fit(m_fn: Callable, y, phi0, **kwargs) -> FitResult:
    """One-off fit; for repeated fits of the same model build a :class:`MapFitter` once."""
    return MapFitter(m_fn, **kwargs).fit(y, phi0)


def chi2_with(mu_fn: Callable, phi, d, sigma) -> float:
    """chi^2 per real data point of complex data ``d`` with per-component noise ``sigma``."""
    r = (d - mu_fn(phi)) / sigma
    return float(jnp.mean(jnp.real(r) ** 2 + jnp.imag(r) ** 2) / 2.0)
