"""Step 8: validation utilities (V1 calibration, V3 realism) and the B0 harness."""

import jax
import jax.numpy as jnp
import numpy as np

from tabsim_sbi import SimConfig, Simulator
from tabsim_sbi.stress_test import run_b0
from tabsim_sbi.validation import calibration as v1
from tabsim_sbi.validation import realism as v3


def test_closure_quantities_are_gain_invariant(geom, gp):
    z = jax.random.normal(jax.random.PRNGKey(0), (gp.n_params,))
    point = geom.replace(sat_xyz=geom.sat_xyz[:0], sat_Pv=geom.sat_Pv[:0])  # point source at phase centre
    vis, _ = Simulator(point, gp, SimConfig(noise="none")).simulate(jax.random.PRNGKey(1), z)
    np.testing.assert_allclose(v3.closure_phases(vis[..., 0], np.asarray(geom.a1), np.asarray(geom.a2)), 0.0, atol=1e-9)
    np.testing.assert_allclose(v3.log_closure_amplitudes(vis[..., 0], np.asarray(geom.a1), np.asarray(geom.a2)), 0.0, atol=1e-9)


def test_shared_antenna_correlation_detects_self_noise(geom, gp):
    strong = geom.replace(sat_Pv=geom.sat_Pv * 3000.0, ast_I=geom.ast_I * 0)
    z = jnp.zeros(gp.n_params)
    a1, a2 = np.asarray(geom.a1), np.asarray(geom.a2)
    out = {}
    for mode in ("thermal", "full"):
        _, aux = Simulator(strong, gp, SimConfig(noise=mode)).simulate(jax.random.PRNGKey(2), z)
        out[mode] = v3.shared_antenna_correlation(aux["noise"][..., 0], a1, a2)
    assert out["full"][0] > out["thermal"][0] + 0.1


def test_sbc_ranks_uniform_for_exact_posterior():
    s = 0.5
    theta = jax.random.normal(jax.random.PRNGKey(3), (400, 1))
    x = theta + s * jax.random.normal(jax.random.PRNGKey(4), (400, 1))
    pv = s**2 / (1 + s**2)
    sample = lambda k, xo, n: xo / (1 + s**2) + np.sqrt(pv) * jax.random.normal(k, (n, 1))
    ranks = v1.sbc_ranks(jax.random.PRNGKey(5), sample, theta, x, n_post=99)
    assert v1.rank_uniformity_pvalues(ranks, 99)[0] > 0.001
    logp = lambda th, xo: -0.5 * jnp.sum((th - xo / (1 + s**2)) ** 2, -1) / pv
    lv = v1.credibility_levels(jax.random.PRNGKey(6), sample, logp, theta, x, n_post=400)
    grid, cov = v1.coverage_curve(lv)
    np.testing.assert_allclose(cov, grid, atol=0.08)


def test_b0_harness_runs(geom, gp):
    base = Simulator(geom, gp, SimConfig())
    out = run_b0(jax.random.PRNGKey(7), base, {"V0": SimConfig(), "E3": SimConfig(noise="diag")}, n_sims=2,
                 model_kwargs={"n_int_model": 4}, fit_kwargs={"n_adam": 100, "n_gn": 15})
    for name in ("V0", "E3"):
        assert 0.0 <= out[name]["convergence_rate"] <= 1.0
        assert np.isfinite(out[name]["phase_err_deg"])
