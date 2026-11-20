"""Step 7: E2, E4, E5, E6, E8 behave as specified and compose in the simulator."""

import jax
import jax.numpy as jnp
import numpy as np

from tabsim_sbi import SimConfig, Simulator
from tabsim_sbi.enhancements import gain_errors as e5
from tabsim_sbi.enhancements import heavy_tail as e8
from tabsim_sbi.enhancements import receiver as e6
from tabsim_sbi.enhancements import rfi_signal as e2
from tabsim_sbi.enhancements.sky import FaintSkyConfig, power_law_fluxes, sample_faint_sky


def test_ou_power_has_unit_mean_and_correlation_time():
    m = e2.ou_log_power(jax.random.PRNGKey(0), 400, 2000, 0.1, 5.0, 0.5)
    np.testing.assert_allclose(m.mean(), 1.0, rtol=0.03)
    x = jnp.log(m)
    lag = 50  # 5 s = tau -> autocorrelation exp(-1)
    xc = x - x.mean(1, keepdims=True)
    ac = jnp.mean(xc[:, lag:] * xc[:, :-lag]) / jnp.mean(xc**2)
    np.testing.assert_allclose(ac, np.exp(-1), atol=0.05)


def test_telegraph_duty_cycle():
    s = e2.telegraph(jax.random.PRNGKey(1), 200, 5000, 0.01, 0.3, 0.5)
    np.testing.assert_allclose(s.mean(), 0.3, atol=0.03)


def test_unperturbed_beams_equal_airy(geom):
    from tabsim_sbi import forward as fw

    u, _ = fw.source_vectors(geom.sat_xyz, geom.ants_xyz)
    pts = e2.antenna_pointings(geom.pointing, jnp.zeros((geom.n_ant, 2)))
    b = e2.antenna_beams(u, pts, geom.freqs, geom.dish_d, jnp.ones(geom.n_ant), jnp.zeros(geom.n_ant))
    ref = fw.airy_voltage(fw.angle_to(u, geom.pointing), geom.freqs, geom.dish_d)
    np.testing.assert_allclose(b, ref, rtol=1e-10, atol=1e-14)


def test_power_law_fluxes_in_range_and_slope():
    S = power_law_fluxes(jax.random.PRNGKey(2), (200_000,), 0.01, 1.0, 1.6)
    assert float(S.min()) >= 0.01 and float(S.max()) <= 1.0
    h, e = np.histogram(np.log10(S), bins=10)
    slope = np.polyfit(0.5 * (e[1:] + e[:-1]), np.log10(h), 1)[0]
    np.testing.assert_allclose(slope, 1 - 1.6, atol=0.05)  # dN/dlogS ~ S^(1-alpha)


def test_faint_sky_count_is_poisson_and_masked():
    cfg = FaintSkyConfig(rate=10.0, n_max=40, r_min_deg=1.0, r_max_deg=3.0)
    counts = []
    for k in jax.random.split(jax.random.PRNGKey(3), 300):
        I, lmn, sig = sample_faint_sky(k, cfg)
        counts.append(int(jnp.sum(I[:, 0] > 0)))
        r = jnp.rad2deg(jnp.arcsin(jnp.linalg.norm(lmn[:, :2], axis=-1)))
        assert float(r.min()) >= 0.999 and float(r.max()) <= 3.001
    np.testing.assert_allclose(np.mean(counts), 10.0, rtol=0.1)


def test_nonclosing_errors_hermitian_unit_diagonal():
    E = e5.nonclosing_errors(jax.random.PRNGKey(4), 6, 0.05)
    np.testing.assert_allclose(E, jnp.conj(E.T))
    np.testing.assert_allclose(jnp.diag(E), 1.0)


def test_gain_residual_keeps_reference_phase(geom):
    r = e5.gain_residual(jax.random.PRNGKey(5), geom.times, geom.n_ant, 0.02, 0.1, 10.0, ref_ant=geom.n_ant - 1)
    np.testing.assert_allclose(jnp.angle(r[:, -1]), 0.0, atol=1e-12)


def test_compression_is_unity_without_signal():
    np.testing.assert_allclose(e6.compression_factor(jnp.array([420.0]), 1e4, 420.0), 1.0)
    assert float(e6.compression_factor(jnp.array([2000.0]), 1e4, 420.0)[0]) < 1.0


def test_student_t_scale_variance():
    s = e8.student_t_scale(jax.random.PRNGKey(6), (400_000,), 6.0)
    np.testing.assert_allclose(jnp.mean(s**2), 6.0 / 4.0, rtol=0.03)


def test_everything_on_is_finite_jittable_and_vmappable(geom, gp):
    cfg = SimConfig(
        Pv_log10_sigma=0.2, tle_sigma_rtn=(40.0, 180.0, 50.0), tle_drift_rtn=(0.1, 1.0, 0.1),
        unmodelled_prob=0.5, unmodelled_log10_Pv=(-7.0, -5.0), n_ground=2,
        power_tau=5.0, power_sigma_log=0.3, duty_cycle=0.7, duty_dwell=2.0,
        pointing_sigma_arcmin=1.0, dish_scale_sigma=0.01, beam_phase_sigma=0.2,
        noise="full", noise_blocks=2, faint_sky=FaintSkyConfig(rate=5.0, n_max=16, r_max_deg=2.0, extended_frac=0.3),
        gain_resid_sigma_amp=0.01, gain_resid_sigma_phase=0.05, gain_resid_tau=10.0, nonclosing_sigma=0.01,
        P_sat=5e3, student_log10_nu=(0.5, 1.5),
    )
    sim = Simulator(geom, gp, cfg)
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    zs = jax.random.normal(jax.random.PRNGKey(8), (3, gp.n_params))
    vis, aux = jax.jit(jax.vmap(sim.simulate))(keys, zs)
    assert vis.shape == (3, geom.n_time, geom.n_bl, geom.n_freq)
    assert np.all(np.isfinite(np.asarray(vis)))
    assert not np.allclose(vis[0], vis[1])
