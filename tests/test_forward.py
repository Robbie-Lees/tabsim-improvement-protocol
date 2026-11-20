"""Step 1: geometry cache and Gaussian-limit forward model agree with tab-sim 0.1.0."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabsim_sbi import SimConfig, Simulator, forward as fw
from tabsim_sbi.geometry import rotation_from_antennas

ti = pytest.importorskip("tabsim.jax.interferometry")


def test_times_fine_match_tabsim(geom):
    ref = ti.int_sample_times(geom.times, geom.n_int, geom.int_time)
    np.testing.assert_allclose(geom.times_fine, ref, atol=1e-12)


def test_noise_std_is_tabsim_convention(geom):
    ref = ti.SEFD_to_noise_std(geom.SEFD, geom.chan_width, geom.int_time)
    np.testing.assert_allclose(geom.noise_std, ref)
    # 420 Jy, 209 kHz, 2 s: 0.459 Jy per component, 0.65 Jy complex (Paper I's sigma_n).
    np.testing.assert_allclose(geom.noise_std * np.sqrt(2), 0.6496, rtol=1e-3)


def test_rfi_visibilities_match_tabsim_kernel(geom, gp):
    sim = Simulator(geom, gp, SimConfig(noise="none"))
    _, aux = sim.simulate(jax.random.PRNGKey(0), jnp.zeros(gp.n_params))

    u, d = fw.source_vectors(geom.sat_xyz, geom.ants_xyz)
    ang = jnp.rad2deg(fw.angle_to(u, geom.pointing))
    sh = (geom.n_sat, geom.n_time, geom.n_int)
    Pv = jnp.broadcast_to(geom.sat_Pv[:, None, None, :], sh + (geom.n_freq,))
    ref = ti.rfi_vis_beam(
        Pv, d.reshape(sh + (geom.n_ant,)), geom.ants_w.reshape(sh[1:] + (geom.n_ant,)),
        ang.reshape(sh + (geom.n_ant,)), geom.freqs, geom.dish_d, geom.a1, geom.a2,
        visibility_precision="double",
    )
    assert float(jnp.max(jnp.abs(ref))) > 1.0  # RFI is actually present
    np.testing.assert_allclose(aux["vis_rfi"], ref, rtol=1e-9, atol=1e-9)


def test_sky_visibilities_match_tabsim(gp):
    from tabsim.beam import airy_beam
    from tabsim_sbi import synthetic_geometry

    g = synthetic_geometry(n_ant=4, n_time=6, n_int=2, sat_offsets_deg=(), ast_I=(3.0, 1.5, 0.7),
                           ast_lm_deg=((0.0, 0.0), (0.3, -0.2), (-0.8, 0.5)))
    from tabsim_sbi import GainParameterisation

    gp4 = GainParameterisation(g.times, g.n_ant, n_g=1)
    sim = Simulator(g, gp4, SimConfig(noise="none"))
    _, aux = sim.simulate(jax.random.PRNGKey(0), jnp.zeros(gp4.n_params))

    theta = np.rad2deg(np.arcsin(np.linalg.norm(np.asarray(g.ast_lmn)[:, :2], axis=-1)))
    beam = airy_beam(theta[:, None, None], np.asarray(g.freqs), g.dish_d)[:, 0, 0, :]  # (src, freq)
    I_app = np.asarray(g.ast_I) * beam**2
    I_app = np.broadcast_to(I_app[:, None, :], (3, g.n_time, 1))
    bl_uvw = g.ants_uvw[:, g.a1] - g.ants_uvw[:, g.a2]
    ref = ti.astro_vis(I_app, bl_uvw, g.ast_lmn, g.freqs, visibility_precision="double")
    np.testing.assert_allclose(aux["vis_ast"], ref, rtol=1e-9, atol=1e-10)


def test_gains_match_tabsim(geom, gp):
    sim = Simulator(geom, gp, SimConfig(noise="none"))
    z = jax.random.normal(jax.random.PRNGKey(3), (gp.n_params,))
    vis, aux = sim.simulate(jax.random.PRNGKey(0), z)
    ref = ti.apply_gains(aux["vis_ast"], aux["vis_rfi"], aux["gains"], geom.a1, geom.a2)
    np.testing.assert_allclose(vis, ref, rtol=1e-10, atol=1e-10)
    # reference antenna phase is fixed
    np.testing.assert_allclose(jnp.angle(aux["gains"][:, gp.ref]), 0.0, atol=1e-12)


def test_thermal_noise_level(geom, gp):
    sim = Simulator(geom, gp, SimConfig())
    vis, aux = jax.vmap(sim.simulate, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(1), 50), jnp.zeros(gp.n_params))
    n = aux["noise"]
    np.testing.assert_allclose(jnp.std(n.real), geom.noise_std[0], rtol=0.03)
    np.testing.assert_allclose(jnp.std(n.imag), geom.noise_std[0], rtol=0.03)


def test_rotation_recovered_from_antennas(geom):
    R = rotation_from_antennas(np.asarray(geom.ants_itrf), np.asarray(geom.ants_xyz))
    np.testing.assert_allclose(R, geom.rot, atol=1e-9)
    np.testing.assert_allclose(np.einsum("tij,pj->tpi", R, geom.ants_itrf), geom.ants_xyz, atol=1e-6)


def test_coherency_is_hermitian_with_power_on_diagonal(geom):
    u, d = fw.source_vectors(geom.sat_xyz, geom.ants_xyz)
    beam = fw.airy_voltage(fw.angle_to(u, geom.pointing), geom.freqs, geom.dish_d)
    Pv = jnp.broadcast_to(geom.sat_Pv[:, None, :], (geom.n_sat, geom.n_time_fine, 1))
    A = fw.rfi_amplitude(Pv, d, beam)
    R = fw.rfi_coherency(A * fw.fringe_phasor(d + geom.ants_w[None], geom.freqs), geom.n_int)[:, 0]
    assert fw.is_hermitian(R)
    np.testing.assert_allclose(fw.autos(R)[:, 0], (A[..., 0] ** 2).reshape(geom.n_sat, geom.n_time, geom.n_int, -1).mean(2).sum(0))
