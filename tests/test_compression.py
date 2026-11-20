"""Steps 4-5: score compression, matrix-free hardening and the fiducial fit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tabsim_sbi import GainParameterisation, SimConfig, Simulator, synthetic_geometry
from tabsim_sbi.compression import CompressionModel, Hardener, ScoreCompressor, WhitenedModel, map_fit


# ----------------------------------------------------------------------------
# Linear-Gaussian toy where everything has a closed form
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def linear():
    key = jax.random.PRNGKey(0)
    n_data, dim = 40, 7
    A = jax.random.normal(key, (n_data, dim)) + 1j * jax.random.normal(jax.random.PRNGKey(1), (n_data, dim))
    sigma = 0.7
    mu = lambda phi: A @ phi
    wm = WhitenedModel(mu, sigma)
    phi_star = 0.1 * jax.random.normal(jax.random.PRNGKey(2), (dim,))
    theta_idx = np.array([0, 1, 2, 3])
    blocks = [np.array([0, 1]), np.array([2]), np.array([3])]
    return A, sigma, mu, wm, phi_star, theta_idx, blocks


def test_score_is_gradient_of_log_likelihood(linear):
    A, sigma, mu, wm, phi_star, *_ = linear
    d = mu(jnp.ones(A.shape[1])) + 0.3
    sc = ScoreCompressor(wm, phi_star)
    logL = lambda phi: -0.5 * jnp.sum(jnp.abs(d - mu(phi)) ** 2) / sigma**2
    np.testing.assert_allclose(sc.score(d), jax.grad(logL)(phi_star), rtol=1e-10)
    # the protocol's complex form, t = 2 Re[J^H C^-1 (d - mu)], C = 2 sigma^2
    t_complex = 2 * jnp.real(A.conj().T @ (d - mu(phi_star))) / (2 * sigma**2)
    np.testing.assert_allclose(sc.score(d), t_complex, rtol=1e-10)


def test_student_t_score_tends_to_gaussian(linear):
    A, sigma, mu, wm, phi_star, *_ = linear
    d = mu(jnp.ones(A.shape[1]))
    g = ScoreCompressor(wm, phi_star).score(d)
    t = ScoreCompressor(wm, phi_star, nu=1e9).score(d)
    np.testing.assert_allclose(t, g, rtol=1e-6)
    # and downweights an outlier
    d_out = d.at[0].add(1e3)
    t5 = ScoreCompressor(wm, phi_star, nu=5.0).score(d_out) - ScoreCompressor(wm, phi_star, nu=5.0).score(d)
    g5 = ScoreCompressor(wm, phi_star).score(d_out) - ScoreCompressor(wm, phi_star).score(d)
    assert float(jnp.linalg.norm(t5)) < 0.05 * float(jnp.linalg.norm(g5))


def test_schur_hardening_equals_blockwise_alsing_wandelt(linear):
    A, sigma, mu, wm, phi_star, theta_idx, blocks = linear
    h = Hardener(wm.m, phi_star, theta_idx, blocks)
    H = np.asarray(h.dense_hessian())
    d = mu(jnp.arange(A.shape[1]) * 0.3) + 0.1j
    t = np.asarray(ScoreCompressor(wm, phi_star).score(d))
    for b, tb_hard in zip(blocks, h.hardened_scores(jnp.asarray(t))):
        n = np.setdiff1d(np.arange(H.shape[0]), b)
        expected = t[b] - H[np.ix_(b, n)] @ np.linalg.solve(H[np.ix_(n, n)], t[n])
        np.testing.assert_allclose(tb_hard, expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(h.cov, np.linalg.inv(H)[np.ix_(theta_idx, theta_idx)], rtol=1e-7)


def test_qmle_projection_equals_score_route(linear):
    A, sigma, mu, wm, phi_star, theta_idx, blocks = linear
    h = Hardener(wm.m, phi_star, theta_idx, blocks)
    sc = ScoreCompressor(wm, phi_star)
    d = mu(jnp.ones(A.shape[1]))
    np.testing.assert_allclose(h.qmle(sc.score(d)), h.qmle_from_residual(sc.residual(d)), rtol=1e-9)


def test_hardened_summaries_ignore_nuisances_without_prior(linear):
    """Zero prior precision = Alsing & Wandelt: d E[summary] / d eta = 0."""
    A, sigma, mu, wm, phi_star, theta_idx, blocks = linear
    h = Hardener(wm.m, phi_star, theta_idx, blocks, prior_precision=0.0)
    sc = ScoreCompressor(wm, phi_star)
    summ = lambda phi: h.qmle(sc.score(mu(phi)))  # noise-free = expectation for a linear model
    Jac = jax.jacfwd(summ)(phi_star)
    eta_idx = np.setdiff1d(np.arange(A.shape[1]), theta_idx)
    np.testing.assert_allclose(Jac[:, eta_idx], 0.0, atol=1e-8)
    np.testing.assert_allclose(Jac[:, theta_idx], np.eye(theta_idx.size), atol=1e-8)


def test_prior_leaves_sensitivity_only_through_prior_precision(linear):
    """With the prior, d qmle / d eta = -[H^-1]_theta,eta P_eta (here P = I)."""
    A, sigma, mu, wm, phi_star, theta_idx, blocks = linear
    h = Hardener(wm.m, phi_star, theta_idx, blocks, prior_precision=1.0)
    sc = ScoreCompressor(wm, phi_star)
    Jac = jax.jacfwd(lambda phi: h.qmle(sc.score(mu(phi))))(phi_star)
    Hinv = np.linalg.inv(np.asarray(h.dense_hessian()))
    eta_idx = np.setdiff1d(np.arange(A.shape[1]), theta_idx)
    np.testing.assert_allclose(Jac[:, eta_idx], -Hinv[np.ix_(theta_idx, eta_idx)], atol=1e-8)


# ----------------------------------------------------------------------------
# The Paper II style compression model on a toy calibrator observation
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cal():
    g = synthetic_geometry(n_ant=5, n_time=24, n_int=8, sat_offsets_deg=(4.0,), ast_I=(8.0,), seed=3)
    gp = GainParameterisation(g.times, g.n_ant, n_g=2, length_scale=30.0, sigma_amp=0.1, sigma_phase=0.3)
    return g, gp


def test_model_matches_simulator_without_rfi(cal):
    g, gp = cal
    g0 = g.replace(sat_xyz=g.sat_xyz[:0], sat_Pv=g.sat_Pv[:0])
    model = CompressionModel(g0, gp)
    z = jax.random.normal(jax.random.PRNGKey(4), (gp.n_params,))
    vis, _ = Simulator(g0, gp, SimConfig(noise="none")).simulate(jax.random.PRNGKey(0), z)
    phi = jnp.zeros(model.dim).at[model.index["gain"]].set(z)
    np.testing.assert_allclose(model.mu(phi), vis[..., 0], rtol=1e-10, atol=1e-12)


def test_map_fit_reaches_the_chi2_gate(cal):
    g, gp = cal
    sim = Simulator(g, gp, SimConfig())
    z = jax.random.normal(jax.random.PRNGKey(5), (gp.n_params,))
    d = sim.simulate(jax.random.PRNGKey(6), z)[0][..., 0]
    model = CompressionModel(g, gp, n_int_model=4)
    wm = WhitenedModel(model.mu, model.sigma_c)
    fit = map_fit(wm.m, wm.y(d), model.init_params(jax.random.PRNGKey(7)))
    assert fit.passed, fit.chi2_per_dof
    # gains recovered to within a few Laplace sigma
    h = Hardener(wm.m, fit.phi, model.index["gain"])
    zs = (fit.phi[model.index["gain"]] - z) / jnp.sqrt(jnp.diag(h.cov))
    assert float(jnp.max(jnp.abs(zs))) < 5.0, zs


def test_signal_dependent_sigma_reduces_to_thermal_without_rfi(cal):
    g, gp = cal
    g0 = g.replace(sat_xyz=g.sat_xyz[:0], sat_Pv=g.sat_Pv[:0], ast_I=g.ast_I * 0)
    model = CompressionModel(g0, gp)
    s = model.signal_dependent_sigma(jnp.zeros(model.dim))
    np.testing.assert_allclose(s, g.noise_std[0], rtol=1e-10)
