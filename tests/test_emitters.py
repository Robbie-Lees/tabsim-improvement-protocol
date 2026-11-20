"""Step 3 (E1): trajectory errors, unmodelled satellites and ground emitters."""

import jax
import jax.numpy as jnp
import numpy as np

from tabsim_sbi import SimConfig, Simulator
from tabsim_sbi.enhancements import emitters as e1


def test_rtn_basis_is_orthonormal_and_along_velocity(geom):
    B = e1.rtn_basis(geom.sat_xyz, geom.dt_fine)
    np.testing.assert_allclose(jnp.einsum("stij,stkj->stik", B, B), jnp.broadcast_to(jnp.eye(3), B.shape), atol=1e-9)
    vel = jnp.gradient(geom.sat_xyz, geom.dt_fine, axis=1)
    cos = jnp.sum(B[..., 1, :] * vel, -1) / jnp.linalg.norm(vel, axis=-1)
    np.testing.assert_allclose(cos, 1.0, atol=1e-9)  # circular orbit: velocity is along-track


def test_along_track_offset_moves_the_satellite_by_that_much(geom):
    off = jnp.zeros((geom.n_sat, geom.n_time_fine, 3)).at[..., 1].set(190.0)
    moved = e1.perturb_trajectories(geom.sat_xyz, off, geom.dt_fine)
    np.testing.assert_allclose(jnp.linalg.norm(moved - geom.sat_xyz, axis=-1), 190.0, rtol=1e-9)


def test_tle_offset_prior_scale():
    off = e1.sample_tle_offsets(jax.random.PRNGKey(0), 20_000, jnp.linspace(0, 60, 3), (40.0, 180.0, 50.0))
    np.testing.assert_allclose(jnp.std(off[:, 0], axis=0), [40.0, 180.0, 50.0], rtol=0.03)
    rms3d = jnp.sqrt(jnp.mean(jnp.sum(off[:, 0] ** 2, -1)))
    np.testing.assert_allclose(rms3d, 191.3, rtol=0.03)


def test_ground_emitters_at_requested_range(geom):
    itrf = e1.sample_ground_emitters(jax.random.PRNGKey(1), 500, geom.ants_itrf, (10.0, 80.0))
    r = jnp.linalg.norm(itrf - geom.ants_itrf.mean(0), axis=-1)
    assert float(r.min()) > 9.9e3 and float(r.max()) < 80.1e3
    xyz = e1.ground_to_inertial(itrf, geom.rot)
    # stays at the same distance from the (co-rotating) array
    d = jnp.linalg.norm(xyz - geom.ants_xyz.mean(1)[None], axis=-1)
    np.testing.assert_allclose(d, jnp.broadcast_to(r[:, None], d.shape), rtol=1e-9)


def test_unmodelled_switches():
    Pv = jnp.ones((4, 1))
    assert float(e1.sample_unmodelled(jax.random.PRNGKey(0), 4, 1, 0.0, Pv).sum()) == 0.0
    np.testing.assert_allclose(e1.sample_unmodelled(jax.random.PRNGKey(0), 4, 1, 1.0, Pv), Pv)


def test_simulator_with_e1_changes_rfi_only(geom, gp):
    z = jnp.zeros(gp.n_params)
    key = jax.random.PRNGKey(2)
    base = Simulator(geom, gp, SimConfig(noise="none"))
    e1on = Simulator(geom, gp, SimConfig(noise="none", tle_sigma_rtn=(40.0, 180.0, 50.0), tle_drift_rtn=(0.1, 1.0, 0.1),
                                        unmodelled_prob=1.0, n_ground=2, ground_log10_Pv=(-10.0, -9.0)))
    v0, a0 = base.simulate(key, z)
    v1, a1 = jax.jit(e1on.simulate)(key, z)
    assert np.all(np.isfinite(np.asarray(v1)))
    np.testing.assert_allclose(a1["vis_ast"], a0["vis_ast"])
    assert float(jnp.max(jnp.abs(a1["vis_rfi"] - a0["vis_rfi"]))) > 1e-3
