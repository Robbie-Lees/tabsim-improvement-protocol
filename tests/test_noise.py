"""Step 2 (E3): the self-noise sampler has Kulkarni's second moments."""

import jax
import jax.numpy as jnp
import numpy as np

from tabsim_sbi import SimConfig, Simulator, forward as fw
from tabsim_sbi._jax import complex_normal
from tabsim_sbi.enhancements import noise as e3


def random_coherency(key, n, sefd=1.0, strength=3.0):
    """A Hermitian PD matrix: SEFD on the diagonal plus a strong rank-2 'RFI' part."""
    B = strength * complex_normal(key, (n, 2))
    return B @ B.conj().T + sefd * jnp.eye(n)


def test_gue_moments():
    G = e3.gue(jax.random.PRNGKey(0), (200_000, 3, 3))
    np.testing.assert_allclose(jnp.mean(jnp.abs(G[:, 0, 1]) ** 2), 1.0, rtol=0.02)
    np.testing.assert_allclose(jnp.mean(G[:, 1, 1].real ** 2), 1.0, rtol=0.02)
    np.testing.assert_allclose(jnp.abs(jnp.mean(G[:, 0, 1] ** 2)), 0.0, atol=0.02)
    np.testing.assert_allclose(G, jnp.conj(jnp.swapaxes(G, -1, -2)))


def test_self_noise_matches_kulkarni():
    n, M, N = 4, 1.0, 400_000
    R = random_coherency(jax.random.PRNGKey(1), n)
    eta = e3.self_noise(jax.random.PRNGKey(2), jnp.broadcast_to(R, (N, n, n)), M)
    scale = float(jnp.max(jnp.abs(R))) ** 2
    for (p, q), (r, s) in [((0, 1), (0, 1)), ((0, 1), (0, 2)), ((0, 1), (2, 3)), ((1, 2), (2, 3)), ((0, 2), (1, 3))]:
        cov = jnp.mean(eta[:, p, q] * jnp.conj(eta[:, r, s]))
        pcov = jnp.mean(eta[:, p, q] * eta[:, r, s])
        np.testing.assert_allclose(cov, R[p, r] * jnp.conj(R[q, s]) / M, atol=0.01 * scale)
        np.testing.assert_allclose(pcov, R[p, s] * R[r, q] / M, atol=0.01 * scale)


def test_self_noise_matches_brute_force_voltages():
    """Against an explicit sample covariance of M complex Gaussian voltage samples."""
    n, M, N = 3, 400, 20_000
    R = random_coherency(jax.random.PRNGKey(3), n, strength=2.0)
    L = jnp.linalg.cholesky(R)
    x = jnp.einsum("ij,kmj->kmi", L, complex_normal(jax.random.PRNGKey(4), (N, M, n)))
    eta_bf = jnp.einsum("kmi,kmj->kij", x, jnp.conj(x)) / M - R
    eta = e3.self_noise(jax.random.PRNGKey(5), jnp.broadcast_to(R, (N, n, n)), M)
    for p, q in [(0, 1), (1, 2)]:
        v_bf = jnp.var(eta_bf[:, p, q].real), jnp.var(eta_bf[:, p, q].imag)
        v = jnp.var(eta[:, p, q].real), jnp.var(eta[:, p, q].imag)
        np.testing.assert_allclose(v, v_bf, rtol=0.05)


def test_reduces_to_thermal_when_signal_is_weak():
    n, M = 5, 209e3 * 2
    R = jnp.broadcast_to(420.0 * jnp.eye(n), (50_000, n, n))
    eta = e3.self_noise(jax.random.PRNGKey(6), R, M)
    std = 420.0 / np.sqrt(2 * M)
    np.testing.assert_allclose(jnp.std(eta[:, 0, 1].real), std, rtol=0.02)
    np.testing.assert_allclose(jnp.std(eta[:, 0, 1].imag), std, rtol=0.02)


def test_blocks_average_the_sub_block_covariances():
    n, M, N = 3, 1.0, 200_000
    R1 = random_coherency(jax.random.PRNGKey(7), n)
    R2 = random_coherency(jax.random.PRNGKey(8), n)
    Rb = jnp.broadcast_to(jnp.stack([R1, R2]), (N, 2, n, n))
    eta = e3.self_noise_blocks(jax.random.PRNGKey(9), Rb, M)
    expect = (R1[0, 0] * R1[1, 1] + R2[0, 0] * R2[1, 1]).real / (2 * M)
    np.testing.assert_allclose(jnp.mean(jnp.abs(eta[:, 0, 1]) ** 2), expect, rtol=0.02)


def test_simulator_noise_inflates_with_rfi(geom, gp):
    """Strong RFI: full self-noise std on a baseline ~ thermal * sqrt(P_p P_q) / SEFD."""
    strong = geom.replace(sat_Pv=geom.sat_Pv * 3000.0)
    z = jnp.zeros(gp.n_params)
    keys = jax.random.split(jax.random.PRNGKey(10), 40)
    for mode in ("diag", "full"):
        sim = Simulator(strong, gp, SimConfig(noise=mode, noise_blocks=2))
        _, aux = jax.vmap(sim.simulate, in_axes=(0, None))(keys, z)
        P = aux["total_power"][0, :, 0]  # (t, p), deterministic here
        predicted = geom.noise_std[0] * jnp.sqrt(P[:, geom.a1] * P[:, geom.a2]) / geom.SEFD[0]
        assert float(predicted.mean()) > 2 * float(geom.noise_std[0])  # we are in the strong regime
        measured = jnp.sqrt(jnp.mean(jnp.abs(aux["noise"][..., 0]) ** 2, axis=0) / 2)
        np.testing.assert_allclose(measured.mean(), predicted.mean(), rtol=0.1)
