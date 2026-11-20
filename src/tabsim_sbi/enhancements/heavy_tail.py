"""E8: heavy-tailed catch-all noise (fallback).

A circular complex Student-t as a Gaussian scale mixture: ``eta -> eta * sqrt(nu / u)``
with ``u ~ chi^2_nu`` drawn per visibility and shared by the real and imaginary
parts. ``nu -> infinity`` recovers the Gaussian. ``nu`` itself can be drawn per
simulation from a log-uniform prior. Use only if E1-E6 do not reproduce the
tails of real residuals (validation V3); otherwise it gives the right marginal
tails with the wrong structure across baselines and time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def student_t_scale(key, shape, nu):
    """sqrt(nu / u), u ~ chi^2_nu, per element of ``shape``."""
    u = 2.0 * jax.random.gamma(key, nu / 2.0, shape)
    return jnp.sqrt(nu / u)


def sample_nu(key, log10_range):
    lo, hi = log10_range
    return 10.0 ** jax.random.uniform(key, (), minval=lo, maxval=hi)
