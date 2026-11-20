"""E2: realistic RFI signal.

Replaces tab-sim's constant intrinsic power and identical real beams with

(i)   a positive stochastic power process per emitter: a log-normal
      Ornstein-Uhlenbeck process (correlation time ``tau``, log-std ``sigma``,
      unit mean) optionally multiplied by an on/off telegraph process (duty
      cycle ``duty``, mean on-dwell ``dwell``). Both vary within and between
      integrations, so the fine-sample averaging sees them.
(ii)  per-antenna beam differences: a small pointing error (two tangent-plane
      components) and a dish-size (beam-width) factor per antenna;
(iii) a complex beam: a phase ``kappa_p (theta / theta_0)^2`` growing away from
      the axis, with ``theta_0 = lambda / D`` and ``kappa_p`` per antenna.

The Paper II compression model has a free, complex, time-correlated A_ps(t)
per antenna and satellite, so all three are absorbed by it when they vary more
slowly than its correlation time (24 s in Paper II). Faster variation -- short
duty cycles, sidelobe structure swept quickly -- is genuine model mismatch.
tab-sim already accepts a time-varying ``Pv`` spectrogram in
``Observation.addTLESatelliteRFI``, so (i) can also be fed to tab-sim directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tabsim_sbi._jax import C_LIGHT, unit
from tabsim_sbi.forward import airy_voltage


def ou_log_power(key, n_src, n_tf, dt, tau, sigma_log):
    """Unit-mean log-normal OU multiplier (n_src, n_tf), exact discretisation."""
    rho = jnp.exp(-dt / tau)
    k0, k1 = jax.random.split(key)
    x0 = sigma_log * jax.random.normal(k0, (n_src,))
    xi = jax.random.normal(k1, (n_tf - 1, n_src))

    def step(x, e):
        x = rho * x + jnp.sqrt(1.0 - rho**2) * sigma_log * e
        return x, x

    _, xs = jax.lax.scan(step, x0, xi)
    x = jnp.concatenate([x0[None], xs], axis=0).T
    return jnp.exp(x - 0.5 * sigma_log**2)


def telegraph(key, n_src, n_tf, dt, duty, dwell_on):
    """On/off process (n_src, n_tf) with stationary on-fraction ``duty``.

    Switching rates: off at 1/dwell_on, on at duty / ((1 - duty) dwell_on).
    """
    p_off = 1.0 - jnp.exp(-dt / dwell_on)
    p_on = 1.0 - jnp.exp(-dt * duty / ((1.0 - duty) * dwell_on))
    k0, k1 = jax.random.split(key)
    s0 = jax.random.bernoulli(k0, duty, (n_src,))
    u = jax.random.uniform(k1, (n_tf - 1, n_src))

    def step(s, ui):
        s = jnp.where(s, ui >= p_off, ui < p_on)
        return s, s

    _, ss = jax.lax.scan(step, s0, u)
    return jnp.concatenate([s0[None], ss], axis=0).T.astype(jnp.float64)


def sample_beam_perturbations(key, n_ant, pointing_sigma_rad, dish_scale_sigma, phase_sigma):
    """Per-antenna pointing offsets (n_ant, 2) rad, dish scale (n_ant,), phase coefficient (n_ant,)."""
    k0, k1, k2 = jax.random.split(key, 3)
    offsets = pointing_sigma_rad * jax.random.normal(k0, (n_ant, 2))
    scale = 1.0 + dish_scale_sigma * jax.random.normal(k1, (n_ant,))
    kappa = phase_sigma * jax.random.normal(k2, (n_ant,))
    return offsets, scale, kappa


def antenna_pointings(pointing, offsets):
    """Perturbed pointing unit vectors (n_ant, 3) from tangent-plane offsets (n_ant, 2)."""
    z = jnp.array([0.0, 0.0, 1.0])
    e1 = unit(jnp.cross(z, pointing))
    e2 = jnp.cross(pointing, e1)
    return unit(pointing[None] + offsets[:, :1] * e1[None] + offsets[:, 1:] * e2[None])


def antenna_beams(unit_vecs, pointings, freqs, dish_d, dish_scale=None, kappa=None):
    """Complex voltage beam per source, fine time, antenna, channel.

    unit_vecs (n_src, n_tf, n_ant, 3); pointings (n_ant, 3).
    """
    cos = jnp.einsum("stpi,pi->stp", unit_vecs, pointings)
    theta = jnp.arccos(jnp.clip(cos, -1.0, 1.0))
    d = dish_d if dish_scale is None else dish_d * dish_scale  # (n_ant,) broadcasts on the antenna axis
    beam = airy_voltage(theta, freqs, d).astype(jnp.complex128)
    if kappa is not None:
        theta0 = C_LIGHT / (jnp.asarray(freqs) * dish_d)  # (n_freq,)
        beam = beam * jnp.exp(1j * kappa[None, None, :, None] * (theta[..., None] / theta0) ** 2)
    return beam
