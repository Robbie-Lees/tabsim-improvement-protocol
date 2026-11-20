"""Neural posterior estimation with a Gaussian mixture density network (pure JAX).

``q(theta | s) = sum_k pi_k(s) N(theta; mu_k(s), L_k(s) L_k(s)^T)`` with an MLP
producing the mixture weights, means and Cholesky factors. MDNs are what
pydelfi (Alsing et al.) uses for the 2-6 dimensional problems the per-block
inference produces, and need no dependency beyond JAX. ``sbi``/``ltu-ili``
estimators can be swapped in: the rest of the pipeline only needs
``sample(key, s, n)`` and ``log_prob(theta, s)``.

Inputs and targets are standardised with training statistics; early stopping
on a held-out fraction guards against over-fitting small simulation budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tabsim_sbi.optim import adam_init, adam_update


def _init_mlp(key, sizes):
    params = []
    for k, (a, b) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append((jax.random.normal(k, (a, b)) * jnp.sqrt(1.0 / a), jnp.zeros(b)))
    return params


def _mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def _split(out, K, D):
    logits = out[..., :K]
    mu = out[..., K : K + K * D].reshape(out.shape[:-1] + (K, D))
    raw = out[..., K + K * D :].reshape(out.shape[:-1] + (K, D * (D + 1) // 2))
    il = jnp.tril_indices(D)
    L = jnp.zeros(out.shape[:-1] + (K, D, D)).at[..., il[0], il[1]].set(raw)
    diag = jnp.exp(jnp.clip(jnp.diagonal(L, axis1=-2, axis2=-1), -7.0, 7.0))
    L = L * (1.0 - jnp.eye(D)) + diag[..., None] * jnp.eye(D)
    return jax.nn.log_softmax(logits, axis=-1), mu, L


def _log_prob(params, theta, x, K, D):
    logw, mu, L = _split(_mlp(params, x), K, D)
    diff = theta[..., None, :] - mu  # (..., K, D)
    z = jax.scipy.linalg.solve_triangular(L, diff[..., None], lower=True)[..., 0]
    logdet = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    comp = -0.5 * jnp.sum(z**2, axis=-1) - logdet - 0.5 * D * jnp.log(2 * jnp.pi)
    return jax.scipy.special.logsumexp(logw + comp, axis=-1)


@dataclass
class MDN:
    params: list
    K: int
    D: int
    theta_mean: jnp.ndarray
    theta_std: jnp.ndarray
    x_mean: jnp.ndarray
    x_std: jnp.ndarray
    val_loss: float

    def log_prob(self, theta, x):
        t = (theta - self.theta_mean) / self.theta_std
        xs = (x - self.x_mean) / self.x_std
        return _log_prob(self.params, t, xs, self.K, self.D) - jnp.sum(jnp.log(self.theta_std))

    def sample(self, key, x, n):
        xs = (jnp.asarray(x) - self.x_mean) / self.x_std
        logw, mu, L = _split(_mlp(self.params, xs), self.K, self.D)
        kc, kz = jax.random.split(key)
        comp = jax.random.categorical(kc, logw, shape=(n,))
        z = jax.random.normal(kz, (n, self.D))
        t = mu[comp] + jnp.einsum("nij,nj->ni", L[comp], z)
        return self.theta_mean + self.theta_std * t


def train_mdn(
    key,
    theta,
    x,
    *,
    n_components: int = 4,
    hidden: Sequence[int] = (64, 64),
    lr: float = 1e-3,
    n_steps: int = 4000,
    batch_size: int = 256,
    val_frac: float = 0.1,
    eval_every: int = 100,
) -> MDN:
    theta, x = jnp.asarray(theta), jnp.asarray(x)
    N, D = theta.shape
    K = n_components
    k_perm, k_init, k_train = jax.random.split(key, 3)
    perm = jax.random.permutation(k_perm, N)
    n_val = max(1, int(val_frac * N))
    vi, ti = perm[:n_val], perm[n_val:]

    tm, ts = theta[ti].mean(0), theta[ti].std(0) + 1e-12
    xm, xsd = x[ti].mean(0), x[ti].std(0) + 1e-12
    T = (theta - tm) / ts
    X = (x - xm) / xsd

    out = K + K * D + K * D * (D + 1) // 2
    params = _init_mlp(k_init, [x.shape[1], *hidden, out])
    state = adam_init(params)

    def loss(p, idx):
        return -jnp.mean(_log_prob(p, T[idx], X[idx], K, D))

    @jax.jit
    def chunk(p, s, key):
        def step(carry, k):
            p, s = carry
            idx = ti[jax.random.randint(k, (min(batch_size, ti.size),), 0, ti.size)]
            g = jax.grad(loss)(p, idx)
            p, s = adam_update(g, s, p, lr)
            return (p, s), None

        (p, s), _ = jax.lax.scan(step, (p, s), jax.random.split(key, eval_every))
        return p, s, loss(p, vi)

    best, best_val, bad = params, np.inf, 0
    for k in jax.random.split(k_train, max(1, n_steps // eval_every)):
        params, state, v = chunk(params, state, k)
        v = float(v)
        if v < best_val - 1e-4:
            best, best_val, bad = params, v, 0
        else:
            bad += 1
            if bad >= 10:
                break
    return MDN(best, K, D, tm, ts, xm, xsd, best_val)
