"""Per-observation geometry, computed once and reused by every simulation.

For simulation-based inference the observation is fixed: times, antenna
positions, pointing, the TLE trajectories of the modelled satellites and the
known sky. Only the stochastic layers (gains, RFI power, trajectory errors,
noise, ...) change between simulations. :class:`ObservationGeometry` holds the
fixed part as JAX arrays so that a simulation is a pure, jit- and vmap-able
function of a PRNG key and the parameters.

Two constructors are provided:

* :func:`from_tabsim_zarr` reads the output of a tab-sim run. It only reads the
  store. Satellites not listed in ``modelled`` become the pool of *unmodelled*
  emitters used by enhancement E1.
* :func:`synthetic_geometry` builds a small self-contained observation (circular
  orbits, spherical Earth) for tests and quick experiments without tab-sim's
  orbit and dask machinery.

Conventions follow tab-sim 0.1.0 (``tabsim.jax.interferometry``):

* RFI phase on antenna p is ``-2 pi f / c (d_p + w_p)``, with ``d_p`` the
  source-antenna distance at the fine sample and ``w_p`` the antenna w
  coordinate. A baseline gets ``A_p conj(A_q) exp(i (phi_p - phi_q))``.
* Astronomical visibilities use ``exp(+2 pi i f / c uvw_pq . (lmn - s0))`` at
  the integration centroid, with the Airy power beam applied to the flux.
* ``noise_std`` is the standard deviation of each of the real and imaginary
  parts, ``SEFD / sqrt(2 chan_width int_time)``. The complex noise therefore has
  variance ``SEFD^2 / (chan_width int_time)``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from tabsim_sbi._jax import C_LIGHT, GM_EARTH, OMEGA_EARTH, R_EARTH, unit

_STATIC = ("n_int", "int_time", "chan_width", "dish_d")


@register_pytree_node_class
@dataclass(frozen=True)
class ObservationGeometry:
    """Fixed geometry of one observation (one frequency axis, one polarisation).

    Shapes: ``n_tf = n_time * n_int`` fine samples.
    """

    times: jnp.ndarray  # (n_time,) s, integration centroids
    freqs: jnp.ndarray  # (n_freq,) Hz
    SEFD: jnp.ndarray  # (n_freq,) Jy
    a1: jnp.ndarray  # (n_bl,) int
    a2: jnp.ndarray  # (n_bl,) int
    ants_xyz: jnp.ndarray  # (n_tf, n_ant, 3) m, inertial (GCRS in tab-sim)
    ants_w: jnp.ndarray  # (n_tf, n_ant) m, antenna w coordinate
    ants_uvw: jnp.ndarray  # (n_time, n_ant, 3) m, at centroids
    pointing: jnp.ndarray  # (3,) unit vector of the phase centre, same frame
    rot: jnp.ndarray  # (n_tf, 3, 3) Earth-fixed -> inertial rotation
    ants_itrf: jnp.ndarray  # (n_ant, 3) m, Earth-fixed
    sat_xyz: jnp.ndarray  # (n_sat, n_tf, 3) modelled satellites (TLE positions)
    sat_Pv: jnp.ndarray  # (n_sat, n_freq) W/Hz, nominal intrinsic power
    pool_xyz: jnp.ndarray  # (n_pool, n_tf, 3) unmodelled candidates (E1)
    pool_Pv: jnp.ndarray  # (n_pool, n_freq) W/Hz
    ast_I: jnp.ndarray  # (n_src, n_freq) Jy, known (modelled) sky, intrinsic
    ast_lmn: jnp.ndarray  # (n_src, 3)
    n_int: int
    int_time: float
    chan_width: float
    dish_d: float

    # -- pytree -----------------------------------------------------------
    def tree_flatten(self):
        children = tuple(getattr(self, f.name) for f in fields(self) if f.name not in _STATIC)
        aux = tuple(getattr(self, name) for name in _STATIC)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        names = [f.name for f in fields(cls) if f.name not in _STATIC]
        kwargs = dict(zip(names, children))
        kwargs.update(dict(zip(_STATIC, aux)))
        return cls(**kwargs)

    # -- sizes --------------------------------------------------------------
    @property
    def n_time(self) -> int:
        return self.times.shape[0]

    @property
    def n_time_fine(self) -> int:
        return self.ants_xyz.shape[0]

    @property
    def n_ant(self) -> int:
        return self.ants_xyz.shape[1]

    @property
    def n_bl(self) -> int:
        return self.a1.shape[0]

    @property
    def n_freq(self) -> int:
        return self.freqs.shape[0]

    @property
    def n_sat(self) -> int:
        return self.sat_xyz.shape[0]

    @property
    def n_pool(self) -> int:
        return self.pool_xyz.shape[0]

    # -- derived quantities --------------------------------------------------
    @property
    def times_fine(self) -> jnp.ndarray:
        """Fine sample times, as tabsim.jax.interferometry.int_sample_times."""
        frac = (jnp.arange(self.n_int) + 0.5) / self.n_int - 0.5
        return (self.times[:, None] + frac[None, :] * self.int_time).ravel()

    @property
    def dt_fine(self) -> float:
        return self.int_time / self.n_int

    @property
    def n_samples(self) -> float:
        """Independent complex samples per visibility, M = chan_width * int_time."""
        return self.chan_width * self.int_time

    @property
    def noise_std(self) -> jnp.ndarray:
        """Per-component thermal noise (n_freq,), tab-sim's ``noise_std``."""
        return self.SEFD / jnp.sqrt(2.0 * self.chan_width * self.int_time)

    def bl_lengths(self) -> jnp.ndarray:
        uvw = self.ants_uvw[:, self.a1] - self.ants_uvw[:, self.a2]
        return jnp.linalg.norm(uvw[..., :2], axis=-1).mean(axis=0)

    def replace(self, **kwargs) -> "ObservationGeometry":
        return replace(self, **kwargs)

    def split_satellites(self, modelled: Sequence[int]) -> "ObservationGeometry":
        """Keep satellites ``modelled`` in the inference model; move the rest to the pool."""
        modelled = np.asarray(modelled, dtype=int)
        rest = np.setdiff1d(np.arange(self.n_sat), modelled)
        return self.replace(
            sat_xyz=self.sat_xyz[modelled],
            sat_Pv=self.sat_Pv[modelled],
            pool_xyz=jnp.concatenate([self.pool_xyz, self.sat_xyz[rest]], axis=0),
            pool_Pv=jnp.concatenate([self.pool_Pv, self.sat_Pv[rest]], axis=0),
        )


# ----------------------------------------------------------------------------
# Helpers shared by the constructors
# ----------------------------------------------------------------------------


def rotation_from_antennas(ants_itrf: np.ndarray, ants_xyz: np.ndarray) -> np.ndarray:
    """Earth-fixed -> inertial rotation at each fine time, from the antennas.

    tab-sim stores antenna positions in both frames. Both are geocentric, so the
    map is a pure rotation; it is recovered per time step by the Kabsch method
    on the centred (planar, well conditioned) antenna layout. Ground emitters
    (E1) are then placed with the same rotation tab-sim used, without
    re-implementing precession and nutation.
    """
    ants_itrf = np.asarray(ants_itrf, dtype=np.float64)
    ants_xyz = np.asarray(ants_xyz, dtype=np.float64)
    a = ants_itrf - ants_itrf.mean(axis=0)
    b = ants_xyz - ants_xyz.mean(axis=1, keepdims=True)
    H = np.einsum("pi,tpj->tij", a, b)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tji,tkj->tik", Vt, U)))
    D = np.zeros_like(H)
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    # R = V D U^T maps a (itrf) onto b (inertial)
    return np.einsum("tji,tjk,tlk->til", Vt, D, U)


def _empty(n_tf, n_freq):
    return jnp.zeros((0, n_tf, 3)), jnp.zeros((0, n_freq))


# ----------------------------------------------------------------------------
# From a tab-sim zarr store
# ----------------------------------------------------------------------------


def from_tabsim_zarr(
    path: str,
    modelled: Optional[Sequence[int]] = None,
    time_range: Optional[tuple] = None,
    Pv: Optional[np.ndarray] = None,
) -> ObservationGeometry:
    """Read the fixed geometry of a tab-sim observation from its zarr output.

    Parameters
    ----------
    path:
        A ``.zarr`` written by ``tabsim`` (``Observation.write_to_zarr``).
    modelled:
        Indices of TLE satellites the inference model includes. The others
        become unmodelled emitters for E1. Default: all modelled.
    time_range:
        ``(i0, i1)`` integrations to keep, to bound memory.
    Pv:
        Intrinsic power per satellite ``(n_sat, n_freq)`` in W/Hz. If omitted it
        is recovered from the stored apparent amplitude ``rfi_tle_sat_A``,
        which tab-sim writes as ``sqrt(Pv / (4 pi d^2) 1e26) * airy(ang_sep)``.
        Recovery is exact for constant power.
    """
    import xarray as xr

    from tabsim_sbi.forward import airy_voltage

    ds = xr.open_zarr(path)
    n_int = int(ds.attrs["n_int_samples"])
    n_time = int(ds.attrs["n_time"])
    i0, i1 = time_range if time_range is not None else (0, n_time)
    fine = slice(i0 * n_int, i1 * n_int)

    times = np.asarray(ds.time.values[i0:i1], dtype=np.float64)
    freqs = np.asarray(ds.freq.values, dtype=np.float64)
    ants_xyz = np.asarray(ds.ants_xyz.isel(time_fine=fine).values)
    ants_uvw_f = np.asarray(ds.ants_uvw.isel(time_fine=fine).values)
    time_idx = np.asarray(ds.time_idx.values[i0:i1], dtype=int)
    ants_uvw = np.asarray(ds.ants_uvw.isel(time_fine=time_idx).values)
    ants_itrf = np.asarray(ds.ants_itrf.values)
    ra, dec = np.deg2rad(float(ds.attrs["target_ra"])), np.deg2rad(float(ds.attrs["target_dec"]))
    pointing = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

    if "rfi_tle_sat_xyz" in ds:
        sat_xyz = np.asarray(ds.rfi_tle_sat_xyz.isel(time_fine=fine).values)
    else:
        sat_xyz = np.zeros((0, ants_xyz.shape[0], 3))
    n_sat = sat_xyz.shape[0]

    if Pv is None and n_sat:
        # Recover Pv from a thinned set of fine samples where the beam is not near a null.
        stride = max(1, n_int // 4)
        sub = slice(i0 * n_int, i1 * n_int, stride)
        A = np.abs(np.asarray(ds.rfi_tle_sat_A.isel(time_fine=sub).values))  # (s, t, a, f)
        ang = np.asarray(ds.rfi_tle_sat_ang_sep.isel(time_fine=sub).values)  # (s, t, a) deg
        xyz = sat_xyz[:, ::stride]
        d = np.linalg.norm(xyz[:, :, None, :] - ants_xyz[None, ::stride], axis=-1)
        beam = np.abs(np.asarray(airy_voltage(np.deg2rad(ang), freqs, float(ds.attrs["dish_diameter"]))))
        good = beam > 0.3 * beam.max(axis=(1, 2, 3), keepdims=True)
        est = (A / np.where(good, beam, 1.0)) ** 2 * 4 * np.pi * d[..., None] ** 2 * 1e-26
        Pv = np.stack([np.median(est[s][good[s]].reshape(-1, len(freqs)), axis=0) if good[s].any()
                       else np.zeros(len(freqs)) for s in range(n_sat)])
    elif Pv is None:
        Pv = np.zeros((0, len(freqs)))

    if "ast_p_I" in ds and ds.attrs.get("n_ast_p_src", 0):
        ast_I = np.asarray(ds.ast_p_I.values).mean(axis=1)
        ast_lmn = np.asarray(ds.ast_p_lmn.values)
    else:
        ast_I = np.zeros((0, len(freqs)))
        ast_lmn = np.zeros((0, 3))

    geom = ObservationGeometry(
        times=jnp.asarray(times),
        freqs=jnp.asarray(freqs),
        SEFD=jnp.asarray(np.asarray(ds.SEFD.values, dtype=np.float64)),
        a1=jnp.asarray(ds.antenna1.values, dtype=jnp.int32),
        a2=jnp.asarray(ds.antenna2.values, dtype=jnp.int32),
        ants_xyz=jnp.asarray(ants_xyz),
        ants_w=jnp.asarray(ants_uvw_f[..., 2]),
        ants_uvw=jnp.asarray(ants_uvw),
        pointing=jnp.asarray(pointing),
        rot=jnp.asarray(rotation_from_antennas(ants_itrf, ants_xyz)),
        ants_itrf=jnp.asarray(ants_itrf),
        sat_xyz=jnp.asarray(sat_xyz),
        sat_Pv=jnp.asarray(Pv),
        pool_xyz=jnp.zeros((0, ants_xyz.shape[0], 3)),
        pool_Pv=jnp.zeros((0, len(freqs))),
        ast_I=jnp.asarray(ast_I),
        ast_lmn=jnp.asarray(ast_lmn),
        n_int=n_int,
        int_time=float(ds.attrs["int_time"]),
        chan_width=float(ds.attrs["chan_width"]),
        dish_d=float(ds.attrs["dish_diameter"]),
    )
    if modelled is not None:
        geom = geom.split_satellites(modelled)
    return geom


# ----------------------------------------------------------------------------
# Synthetic observation
# ----------------------------------------------------------------------------


def _rz(angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    z, o = jnp.zeros_like(angle), jnp.ones_like(angle)
    return jnp.stack([jnp.stack([c, -s, z], -1), jnp.stack([s, c, z], -1), jnp.stack([z, z, o], -1)], -2)


def _circular_orbit(times, t_mid, x_centre, pointing, e_u, e_v, offset_deg, azimuth_deg, heading_deg, radius):
    """Circular orbit passing ``offset_deg`` from the pointing at ``t_mid``."""
    th, ps, be = (jnp.deg2rad(x) for x in (offset_deg, azimuth_deg, heading_deg))
    u = jnp.cos(th) * pointing + jnp.sin(th) * (jnp.cos(ps) * e_u + jnp.sin(ps) * e_v)
    xu = jnp.dot(x_centre, u)
    rho = -xu + jnp.sqrt(xu**2 - jnp.dot(x_centre, x_centre) + radius**2)
    p0 = unit(x_centre + rho * u)
    z = jnp.array([0.0, 0.0, 1.0])
    v0 = unit(z - jnp.dot(z, p0) * p0)
    v = jnp.cos(be) * v0 + jnp.sin(be) * jnp.cross(p0, v0)
    n = jnp.sqrt(GM_EARTH / radius**3)
    ang = n * (times - t_mid)
    return radius * (jnp.cos(ang)[:, None] * p0 + jnp.sin(ang)[:, None] * v)


def synthetic_geometry(
    n_ant: int = 6,
    n_time: int = 30,
    n_int: int = 8,
    int_time: float = 2.0,
    freq: float = 1.227e9,
    chan_width: float = 209e3,
    SEFD: float = 420.0,
    dish_d: float = 13.5,
    array_radius: float = 150.0,
    latitude: float = -30.71,
    longitude: float = 21.44,
    sat_offsets_deg: Sequence[float] = (5.0,),
    sat_azimuths_deg: Optional[Sequence[float]] = None,
    sat_Pv: float = 1.5e-6,
    pool_offsets_deg: Sequence[float] = (),
    pool_Pv: float = 1.5e-6,
    orbit_radius: float = 26_560e3,
    ast_I: Sequence[float] = (10.0,),
    ast_lm_deg: Sequence[tuple] = ((0.0, 0.0),),
    seed: int = 0,
) -> ObservationGeometry:
    """A small, self-contained observation for tests and experiments.

    Spherical Earth rotating at the sidereal rate, antennas in a disc of radius
    ``array_radius`` (m), phase centre at the zenith at mid-observation, and
    satellites on circular orbits (GPS radius by default) that pass the given
    angular offsets from the phase centre at mid-observation.
    """
    rng = np.random.default_rng(seed)
    r = array_radius * np.sqrt(rng.uniform(size=n_ant))
    a = rng.uniform(0, 2 * np.pi, size=n_ant)
    enu = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n_ant)], axis=1)

    lat, lon = np.deg2rad(latitude), np.deg2rad(longitude)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(up, east)
    centre = R_EARTH * up
    ants_itrf = centre + enu @ np.stack([east, north, up])

    times = jnp.arange(n_time) * int_time
    frac = (jnp.arange(n_int) + 0.5) / n_int - 0.5
    times_fine = (times[:, None] + frac[None, :] * int_time).ravel()
    t_mid = 0.5 * (times[0] + times[-1])

    rot = _rz(OMEGA_EARTH * times_fine)  # (n_tf, 3, 3)
    ants_xyz = jnp.einsum("tij,pj->tpi", rot, ants_itrf)
    pointing = unit(_rz(jnp.asarray(OMEGA_EARTH * t_mid)) @ up)
    z = jnp.array([0.0, 0.0, 1.0])
    e_u = unit(jnp.cross(z, pointing))
    e_v = jnp.cross(pointing, e_u)

    def uvw_of(xyz):
        rel = xyz - xyz.mean(axis=-2, keepdims=True)
        return jnp.stack([rel @ e_u, rel @ e_v, rel @ pointing], axis=-1)

    ants_uvw_fine = uvw_of(ants_xyz)
    rot_c = _rz(OMEGA_EARTH * times)
    ants_uvw = uvw_of(jnp.einsum("tij,pj->tpi", rot_c, ants_itrf))

    x_centre = (_rz(jnp.asarray(OMEGA_EARTH * t_mid)) @ jnp.asarray(ants_itrf).T).T.mean(axis=0)

    def orbits(offsets, azimuths, seed_off):
        if len(offsets) == 0:
            return jnp.zeros((0, len(times_fine), 3))
        if azimuths is None:
            azimuths = np.random.default_rng(seed + seed_off).uniform(0, 360, size=len(offsets))
        headings = np.random.default_rng(seed + seed_off + 1).uniform(-60, 60, size=len(offsets))
        return jnp.stack([
            _circular_orbit(times_fine, t_mid, x_centre, pointing, e_u, e_v, o, az, hd, orbit_radius)
            for o, az, hd in zip(offsets, azimuths, headings)
        ])

    sat_xyz = orbits(sat_offsets_deg, sat_azimuths_deg, 100)
    pool_xyz = orbits(pool_offsets_deg, None, 200)

    a1, a2 = np.triu_indices(n_ant, 1)
    lm = np.deg2rad(np.asarray(ast_lm_deg, dtype=np.float64).reshape(-1, 2))
    lmn = np.concatenate([lm, np.sqrt(1 - (lm**2).sum(1, keepdims=True))], axis=1)

    return ObservationGeometry(
        times=times,
        freqs=jnp.array([freq]),
        SEFD=jnp.array([SEFD]),
        a1=jnp.asarray(a1, dtype=jnp.int32),
        a2=jnp.asarray(a2, dtype=jnp.int32),
        ants_xyz=ants_xyz,
        ants_w=ants_uvw_fine[..., 2],
        ants_uvw=ants_uvw,
        pointing=pointing,
        rot=rot,
        ants_itrf=jnp.asarray(ants_itrf),
        sat_xyz=sat_xyz,
        sat_Pv=jnp.full((sat_xyz.shape[0], 1), sat_Pv),
        pool_xyz=pool_xyz,
        pool_Pv=jnp.full((pool_xyz.shape[0], 1), pool_Pv),
        ast_I=jnp.asarray(ast_I, dtype=jnp.float64).reshape(-1, 1),
        ast_lmn=jnp.asarray(lmn),
        n_int=n_int,
        int_time=float(int_time),
        chan_width=float(chan_width),
        dish_d=float(dish_d),
    )
