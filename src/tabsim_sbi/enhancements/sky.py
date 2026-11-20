"""E4: unmodelled sky.

Faint sources drawn from power-law counts ``dN/dS ~ S^-alpha`` between
``S_min`` and ``S_max``, Poisson in number with mean ``rate``, uniform in area
within an annulus ``r_min < r < r_max`` around the phase centre. A fraction
``extended_frac`` are circular Gaussians with FWHM uniform in
``fwhm_arcsec``. Placing the annulus in the primary-beam sidelobes gives the
off-axis case Paper II's fringe-rate prior assumes away; placing it near the
calibrator gives Protocol A's confusion background.

Fixed array sizes (``n_max``) with a mask keep the simulator jit-able.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FaintSkyConfig:
    rate: float = 20.0  # expected number of sources
    n_max: int = 64
    S_min: float = 0.01  # Jy
    S_max: float = 1.0  # Jy
    alpha: float = 1.6  # dN/dS ~ S^-alpha (to be set from source counts)
    r_min_deg: float = 0.0
    r_max_deg: float = 1.0
    extended_frac: float = 0.0
    fwhm_arcsec: tuple = (10.0, 120.0)


def power_law_fluxes(key, shape, S_min, S_max, alpha):
    """Inverse-CDF draw from dN/dS ~ S^-alpha on [S_min, S_max]."""
    u = jax.random.uniform(key, shape)
    if abs(alpha - 1.0) < 1e-9:
        return S_min * (S_max / S_min) ** u
    a = 1.0 - alpha
    return (S_min**a + u * (S_max**a - S_min**a)) ** (1.0 / a)


def sample_faint_sky(key, cfg: FaintSkyConfig, n_freq: int = 1):
    """Returns I (n_max, n_freq) masked to the Poisson count, lmn (n_max, 3), gauss sigma (n_max,) rad."""
    kn, ks, kr, ka, ke, kw = jax.random.split(key, 6)
    n = jax.random.poisson(kn, cfg.rate)
    mask = jnp.arange(cfg.n_max) < n
    S = power_law_fluxes(ks, (cfg.n_max,), cfg.S_min, cfg.S_max, cfg.alpha) * mask
    r0, r1 = jnp.deg2rad(cfg.r_min_deg), jnp.deg2rad(cfg.r_max_deg)
    r = jnp.sqrt(jax.random.uniform(kr, (cfg.n_max,), minval=r0**2, maxval=r1**2))
    az = jax.random.uniform(ka, (cfg.n_max,), minval=0.0, maxval=2 * jnp.pi)
    l, m = jnp.sin(r) * jnp.cos(az), jnp.sin(r) * jnp.sin(az)
    lmn = jnp.stack([l, m, jnp.sqrt(1.0 - l**2 - m**2)], axis=-1)
    ext = jax.random.bernoulli(ke, cfg.extended_frac, (cfg.n_max,))
    fwhm = jnp.deg2rad(jax.random.uniform(kw, (cfg.n_max,), minval=cfg.fwhm_arcsec[0], maxval=cfg.fwhm_arcsec[1]) / 3600.0)
    sigma = jnp.where(ext, fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0))), 0.0)
    return jnp.repeat(S[:, None], n_freq, axis=1), lmn, sigma
