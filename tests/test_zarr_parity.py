"""Optional: rebuild the geometry of a real tab-sim run and reproduce its stored vis_rfi.

Set TABSIM_SBI_TEST_ZARR to a tab-sim ``.zarr`` output with TLE satellites.
On a 32-antenna, 3-satellite MeerKAT run (1025 fine samples per 2 s) this
agrees to ~3e-9 relative.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PATH = os.environ.get("TABSIM_SBI_TEST_ZARR")
pytestmark = pytest.mark.skipif(not PATH, reason="TABSIM_SBI_TEST_ZARR not set")


def test_vis_rfi_reproduced_from_zarr_geometry():
    xr = pytest.importorskip("xarray")
    from tabsim_sbi import GainParameterisation, SimConfig, Simulator, from_tabsim_zarr

    i0, i1 = 100, 104
    g = from_tabsim_zarr(PATH, time_range=(i0, i1))
    gp = GainParameterisation(g.times, g.n_ant, n_g=1)
    _, aux = Simulator(g, gp, SimConfig(noise="none")).simulate(jax.random.PRNGKey(0), jnp.zeros(gp.n_params))
    ref = xr.open_zarr(PATH).vis_rfi.isel(time=slice(i0, i1)).values
    np.testing.assert_allclose(aux["vis_rfi"], ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())
    np.testing.assert_allclose(np.einsum("tij,pj->tpi", g.rot, g.ants_itrf), g.ants_xyz, atol=1e-3)
