"""V1: calibration of posteriors on held-out simulations.

* Simulation-based calibration (Talts et al. 2018): the rank of the true value
  among posterior samples is uniform for a calibrated posterior.
* Expected coverage (Hermans et al. 2022): for each test simulation, the
  credibility level of the smallest highest-density region containing the
  truth; the empirical CDF of these levels should be the identity. A curve
  below the diagonal means over-confident posteriors.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats


def sbc_ranks(key, sample_fn: Callable, theta_true, summaries, n_post: int = 200):
    """Ranks (N, D) of each true value among ``n_post`` posterior samples."""
    keys = jax.random.split(key, theta_true.shape[0])
    ranks = []
    for k, th, s in zip(keys, theta_true, summaries):
        samp = sample_fn(k, s, n_post)
        ranks.append(np.sum(np.asarray(samp) < np.asarray(th)[None, :], axis=0))
    return np.stack(ranks)


def rank_uniformity_pvalues(ranks, n_post: int, n_bins: int = 10):
    """Chi-square p-value of rank uniformity per dimension."""
    edges = np.linspace(0, n_post + 1, n_bins + 1)
    p = []
    for r in ranks.T:
        counts, _ = np.histogram(r, bins=edges)
        p.append(stats.chisquare(counts).pvalue)
    return np.array(p)


def credibility_levels(key, sample_fn, log_prob_fn, theta_true, summaries, n_post: int = 500):
    """HPD credibility level of the truth for each test case (N,)."""
    keys = jax.random.split(key, theta_true.shape[0])
    levels = []
    for k, th, s in zip(keys, theta_true, summaries):
        samp = sample_fn(k, s, n_post)
        lp = np.asarray(log_prob_fn(samp, jnp.broadcast_to(s, (n_post,) + s.shape)))
        lp_true = float(log_prob_fn(th[None], s[None])[0])
        levels.append(np.mean(lp > lp_true))
    return np.array(levels)


def coverage_curve(levels, grid=None):
    """Empirical coverage at nominal levels ``grid``."""
    grid = np.linspace(0, 1, 21) if grid is None else grid
    return grid, np.array([np.mean(levels <= g) for g in grid])


def gaussian_zscores(theta_true, mean, cov):
    """Per-dimension z-scores of truths under Gaussian (e.g. Laplace) posteriors."""
    sd = np.sqrt(np.diagonal(np.asarray(cov), axis1=-2, axis2=-1))
    return (np.asarray(theta_true) - np.asarray(mean)) / sd
