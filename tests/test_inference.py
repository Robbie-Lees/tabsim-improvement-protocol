"""Step 6: density estimation, prior correction, and Protocol A end to end."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabsim_sbi import GainParameterisation, SimConfig, Simulator, synthetic_geometry
from tabsim_sbi.inference import GaussianProposal, sir_correct, standard_normal_logpdf, train_mdn
from tabsim_sbi.protocols import protocol_a, protocol_b1, protocol_b2, run_protocol


def test_mdn_recovers_conjugate_gaussian_posterior():
    key = jax.random.PRNGKey(0)
    N, s = 4000, 0.5
    theta = jax.random.normal(key, (N, 2))
    x = theta + s * jax.random.normal(jax.random.PRNGKey(1), (N, 2))
    mdn = train_mdn(jax.random.PRNGKey(2), theta, x, n_components=2, n_steps=3000)
    x0 = jnp.array([0.8, -0.4])
    samp = mdn.sample(jax.random.PRNGKey(3), x0, 20_000)
    post_var = s**2 / (1 + s**2)
    np.testing.assert_allclose(samp.mean(0), x0 / (1 + s**2), atol=0.05)
    np.testing.assert_allclose(samp.std(0), np.sqrt(post_var), rtol=0.1)


def test_sir_corrects_proposal_to_prior():
    """Likelihood N(x; theta, s^2); proposal N(m, v); target prior N(0, 1)."""
    s, x, m, v = 0.6, 1.2, 0.5, 2.0**2
    prec_q = 1 / s**2 + 1 / v
    mean_q = (x / s**2 + m / v) / prec_q
    q = mean_q + jax.random.normal(jax.random.PRNGKey(0), (100_000, 1)) / np.sqrt(prec_q)
    prop = GaussianProposal(jnp.array([m]), jnp.array([[v]]))
    out, ess = sir_correct(jax.random.PRNGKey(1), q, standard_normal_logpdf(q), prop.log_prob(q), 50_000)
    prec_p = 1 / s**2 + 1
    np.testing.assert_allclose(out.mean(), (x / s**2) / prec_p, atol=0.02)
    np.testing.assert_allclose(out.std(), 1 / np.sqrt(prec_p), rtol=0.03)
    assert ess > 10_000


@pytest.fixture(scope="module")
def tiny():
    g = synthetic_geometry(n_ant=4, n_time=16, n_int=4, sat_offsets_deg=(4.0,), ast_I=(8.0,), seed=5)
    gp = GainParameterisation(g.times, g.n_ant, n_g=1, sigma_amp=0.1, sigma_phase=0.3)
    return g, gp


@pytest.mark.slow
def test_protocol_a_end_to_end(tiny):
    g, gp = tiny
    sim = Simulator(g, gp, SimConfig(noise="diag", tle_sigma_rtn=(40.0, 180.0, 50.0)))
    spec = protocol_a(sim, n_int_model=4)
    z_true = jax.random.normal(jax.random.PRNGKey(10), (gp.n_params,))
    d = sim.simulate(jax.random.PRNGKey(11), z_true)[0][..., 0]
    res = run_protocol(jax.random.PRNGKey(12), spec, d, n_sims=600, n_post=500,
                       mdn_kwargs={"n_steps": 1500, "n_components": 2})
    assert len(res.blocks) == g.n_ant
    assert res.summaries.shape == (600, spec.theta_idx.size)
    assert np.all(np.isfinite(np.asarray(res.joint_cov)))
    # The truth sits within a generous band of the posterior.
    sd = np.sqrt(np.diag(np.asarray(res.joint_cov)))
    z = (np.asarray(res.joint_mean) - np.asarray(z_true)) / sd
    assert np.all(np.abs(z) < 5.0), z
    for b in res.blocks:
        assert b.ess > 50


def test_protocol_b_specs_build_and_compress(tiny):
    g, gp = tiny
    sim = Simulator(g, gp, SimConfig())
    for spec in (protocol_b1(sim, n_int_model=2),
                 protocol_b2(sim, lmn=g.ast_lmn, I0=g.ast_I[:, 0], n_int_model=2)):
        z = jax.random.normal(jax.random.PRNGKey(0), (spec.theta_idx.size,))
        vis, _ = spec.simulator.simulate(jax.random.PRNGKey(1), **spec.theta_to_sim(jax.random.PRNGKey(2), z))
        mu = spec.model.mu(jnp.zeros(spec.model.dim))
        assert mu.shape == vis[..., 0].shape
        assert np.isfinite(float(jnp.sum(jnp.abs(jax.grad(lambda p: jnp.sum(jnp.abs(spec.model.mu(p)) ** 2))(jnp.zeros(spec.model.dim))))))
