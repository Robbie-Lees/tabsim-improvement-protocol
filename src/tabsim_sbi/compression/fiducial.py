"""Fiducial point: a MAP fit of the compression model to the observed data.

Objective (standardised coordinates): ``1/2 ||y - m(phi)||^2 + 1/2 p ||phi||^2``.
A short Adam warm-up (robust far from the optimum) is followed by
Levenberg-Marquardt Gauss-Newton steps, each a matrix-free CG solve with the
same ``J^T J + p I`` operator the hardening uses, which converges in far fewer
iterations than first-order methods on this least-squares problem.

The chi^2 gate follows Paper II (chi^2 per real data point below 1.1). A poor
fiducial costs optimality, not validity -- the same compression is applied to
data and simulations -- so the gate decides whether to proceed *with reduced
expectations*, not whether to trust the result (protocol Section 3.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from tabsim_sbi.optim import adam


@dataclass
class FitResult:
    phi: jnp.ndarray
    chi2_per_dof: float
    passed: bool
    history: List[float] = field(default_factory=list)


def map_fit(
    m_fn: Callable,
    y,
    phi0,
    *,
    prior_precision: float = 1.0,
    n_adam: int = 300,
    lr: float = 0.02,
    n_gn: int = 30,
    lm_lambda: float = 1e-2,
    cg_maxiter: int = 200,
    gate: float = 1.1,
    rtol: float = 1e-10,
) -> FitResult:
    y = jnp.asarray(y)
    n = y.shape[0]

    def loss(phi):
        r = y - m_fn(phi)
        return 0.5 * jnp.sum(r**2) + 0.5 * prior_precision * jnp.sum(phi**2)

    history = []
    phi = jnp.asarray(phi0)
    if n_adam:
        phi, losses = adam(jax.value_and_grad(loss), phi, lr=lr, n_steps=n_adam)
        history += [float(x) for x in losses[:: max(1, n_adam // 20)]]

    @jax.jit
    def gn_step(phi, lam):
        r, jvp = jax.linearize(m_fn, phi)
        vjp = jax.linear_transpose(jvp, phi)
        g = vjp(y - r)[0] - prior_precision * phi
        Hop = lambda v: vjp(jvp(v))[0] + (prior_precision + lam) * v
        delta = cg(Hop, g, maxiter=cg_maxiter)[0]
        new = phi + delta
        return new, loss(new)

    cur = float(loss(phi))
    lam = lm_lambda
    for _ in range(n_gn):
        cand, cl = gn_step(phi, lam)
        cl = float(cl)
        if cl < cur:
            done = (cur - cl) < rtol * max(abs(cur), 1.0)
            phi, cur, lam = cand, cl, max(lam / 3.0, 1e-8)
            history.append(cur)
            if done:
                break
        else:
            lam *= 5.0
            if lam > 1e8:
                break

    chi2 = float(jnp.sum((y - m_fn(phi)) ** 2) / n)
    return FitResult(phi=phi, chi2_per_dof=chi2, passed=chi2 < gate, history=history)
