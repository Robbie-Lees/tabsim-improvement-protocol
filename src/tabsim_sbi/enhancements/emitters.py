"""E1: unmodelled emitters and trajectory (TLE) errors.

The inference model (A1 variant) fixes satellite trajectories at the TLE
values and lists a subset of satellites. The simulator breaks both:

* **Trajectory errors.** Each modelled satellite's true position is the TLE
  position plus an offset in its radial / along-track / cross-track (RTN)
  frame, ``delta(t) = delta_0 + delta_dot (t - t_mid)``. TLE errors are
  dominated by the along-track component and grow with TLE age; the default
  prior (40, 180, 50) m has a 3-D rms of ~190 m, the figure Paper II quotes for
  one-day-old TLEs. Drift rates are "to be set" from data.
* **Unmodelled satellites.** Satellites from the catalogue that the model omits
  (the geometry's ``pool``) are switched on at random with intrinsic power from
  a broad log-uniform prior.
* **Ground emitters.** Stationary transmitters at random range and azimuth,
  placed in the inertial frame with the rotation tab-sim itself used
  (``geometry.rot``), entering through the far sidelobes.

All of these are latent variables, drawn afresh in every simulation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tabsim_sbi._jax import R_EARTH, unit


def rtn_basis(sat_xyz, dt):
    """Radial, along-track, cross-track unit vectors (n_sat, n_tf, 3, 3).

    Row 0 radial, row 1 along-track, row 2 cross-track (orbit normal).
    """
    vel = jnp.gradient(sat_xyz, dt, axis=1)
    r = unit(sat_xyz)
    n = unit(jnp.cross(sat_xyz, vel))
    t = jnp.cross(n, r)
    return jnp.stack([r, t, n], axis=-2)


def sample_tle_offsets(key, n_sat, times_fine, sigma_rtn, drift_rtn=None):
    """RTN offsets (n_sat, n_tf, 3) in metres."""
    k0, k1 = jax.random.split(key)
    sigma_rtn = jnp.asarray(sigma_rtn, dtype=jnp.float64)
    delta0 = sigma_rtn * jax.random.normal(k0, (n_sat, 1, 3))
    t_mid = 0.5 * (times_fine[0] + times_fine[-1])
    if drift_rtn is None:
        return jnp.broadcast_to(delta0, (n_sat, times_fine.shape[0], 3))
    rate = jnp.asarray(drift_rtn, dtype=jnp.float64) * jax.random.normal(k1, (n_sat, 1, 3))
    return delta0 + rate * (times_fine - t_mid)[None, :, None]


def perturb_trajectories(sat_xyz, offsets_rtn, dt):
    """True positions = TLE positions + RTN offsets."""
    basis = rtn_basis(sat_xyz, dt)
    return sat_xyz + jnp.einsum("stij,sti->stj", basis, offsets_rtn)


def sample_log_uniform(key, shape, log10_range):
    lo, hi = log10_range
    return 10.0 ** jax.random.uniform(key, shape, minval=lo, maxval=hi)


def sample_unmodelled(key, n_pool, n_freq, prob, nominal_Pv, log10_Pv_range=None):
    """Which pool satellites transmit, and with what power.

    Returns Pv (n_pool, n_freq) with zeros for satellites that are off.
    """
    k_on, k_pv = jax.random.split(key)
    on = jax.random.bernoulli(k_on, prob, (n_pool,))
    if log10_Pv_range is None:
        Pv = nominal_Pv
    else:
        Pv = jnp.broadcast_to(sample_log_uniform(k_pv, (n_pool, 1), log10_Pv_range), (n_pool, n_freq))
    return Pv * on[:, None]


def local_frame(ants_itrf):
    """East, north, up unit vectors at the array centre (spherical Earth)."""
    centre = ants_itrf.mean(axis=0)
    up = unit(centre)
    east = unit(jnp.cross(jnp.array([0.0, 0.0, 1.0]), up))
    north = jnp.cross(up, east)
    return centre, east, north, up


def sample_ground_emitters(key, n, ants_itrf, range_km=(10.0, 80.0), height_m=(5.0, 50.0)):
    """Earth-fixed positions (n, 3) of stationary emitters around the array.

    Range uniform in area between the limits, azimuth uniform, mast height
    uniform; the curvature drop ``r^2 / 2R`` is included.
    """
    k_r, k_a, k_h = jax.random.split(key, 3)
    r0, r1 = (1e3 * x for x in range_km)
    r = jnp.sqrt(jax.random.uniform(k_r, (n,), minval=r0**2, maxval=r1**2))
    az = jax.random.uniform(k_a, (n,), minval=0.0, maxval=2 * jnp.pi)
    h = jax.random.uniform(k_h, (n,), minval=height_m[0], maxval=height_m[1])
    centre, east, north, up = local_frame(ants_itrf)
    drop = r**2 / (2 * R_EARTH)
    return (
        centre
        + (r * jnp.sin(az))[:, None] * east
        + (r * jnp.cos(az))[:, None] * north
        + (h - drop)[:, None] * up
    )


def ground_to_inertial(itrf, rot):
    """Earth-fixed (n, 3) -> inertial trajectories (n, n_tf, 3) with rot (n_tf, 3, 3)."""
    return jnp.einsum("tij,nj->nti", rot, itrf)
