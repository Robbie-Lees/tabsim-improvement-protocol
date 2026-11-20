import jax
import numpy as np
import pytest

import tabsim_sbi  # noqa: F401  (x64)
from tabsim_sbi import GainParameterisation, SimConfig, Simulator, synthetic_geometry


@pytest.fixture(scope="session")
def geom():
    return synthetic_geometry(n_ant=5, n_time=20, n_int=8, sat_offsets_deg=(4.0,), pool_offsets_deg=(8.0,), seed=1)


@pytest.fixture(scope="session")
def gp(geom):
    return GainParameterisation(geom.times, geom.n_ant, n_g=2, length_scale=30.0, sigma_amp=0.1, sigma_phase=0.3)


@pytest.fixture(scope="session")
def sim(geom, gp):
    return Simulator(geom, gp, SimConfig())


@pytest.fixture(scope="session")
def cal():
    """A toy calibrator scan: 5 antennas, 24 x 2 s, one GPS satellite 4 deg off axis."""
    g = synthetic_geometry(n_ant=5, n_time=24, n_int=8, sat_offsets_deg=(4.0,), ast_I=(8.0,), seed=3)
    gp = GainParameterisation(g.times, g.n_ant, n_g=2, length_scale=30.0, sigma_amp=0.1, sigma_phase=0.3)
    return g, gp


@pytest.fixture(scope="session")
def linear():
    """A complex linear model mu = A phi, where every compression quantity has a closed form."""
    from tabsim_sbi.compression.score import WhitenedModel

    n_data, dim = 40, 7
    A = jax.random.normal(jax.random.PRNGKey(0), (n_data, dim)) + 1j * jax.random.normal(jax.random.PRNGKey(1), (n_data, dim))
    sigma = 0.7
    mu = lambda phi: A @ phi
    wm = WhitenedModel(mu, sigma)
    phi_star = 0.1 * jax.random.normal(jax.random.PRNGKey(2), (dim,))
    theta_idx = np.array([0, 1, 2, 3])
    blocks = [np.array([0, 1]), np.array([2]), np.array([3])]
    return A, sigma, mu, wm, phi_star, theta_idx, blocks
