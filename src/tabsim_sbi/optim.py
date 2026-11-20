"""A minimal Adam, to avoid an optimiser dependency."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.zeros((), dtype=jnp.int32)}


def adam_update(grads, state, params, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g**2, state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda m: m / (1 - b1**t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1 - b2**t), v)
    params = jax.tree_util.tree_map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t}


def adam(value_and_grad, params, lr=1e-2, n_steps=100):
    """Run ``n_steps`` of Adam on a function returning (loss, grad). Returns (params, losses)."""

    @jax.jit
    def run(params):
        def step(carry, _):
            p, s = carry
            val, g = value_and_grad(p)
            p, s = adam_update(g, s, p, lr)
            return (p, s), val

        (p, _), vals = jax.lax.scan(step, (params, adam_init(params)), None, length=n_steps)
        return p, vals

    return run(params)
