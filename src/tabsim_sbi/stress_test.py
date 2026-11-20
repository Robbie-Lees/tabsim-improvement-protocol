"""B0: stress test of the existing method on enhanced simulations (do first).

Run an *unmodified* fit on simulations with one enhancement switched on at a
time and measure the damage: convergence rate (chi^2 gate), gain errors,
recovered-visibility errors and the coverage of the Laplace uncertainties. It
needs one MAP fit per simulation and no new inference code, and it decides
which effects deserve an inference protocol.

The fitter is pluggable. The default fits this package's Paper II style
compression model (a stand-in for tabascal). To stress-test tabascal itself,
pass a ``fit_fn`` that writes the simulated visibilities to a measurement set
(tab-sim's writer) and runs the tabascal CLI; it must return the fitted
standardised gain parameters and their Laplace covariance.
"""

from __future__ import annotations

import zlib
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tabsim_sbi.compression.fiducial import MapFitter
from tabsim_sbi.compression.model import CompressionModel
from tabsim_sbi.compression.score import WhitenedModel
from tabsim_sbi.simulator import SimConfig, Simulator


def default_fit_fn(model: CompressionModel, fit_kwargs=None):
    """MAP fit + Laplace gain covariance, compiled once per model.

    The Laplace covariance is formed densely (fine for toy sizes); at full
    scale use :class:`~tabsim_sbi.compression.hardening.Hardener` or tabascal.
    """
    wm = WhitenedModel(model.mu, model.sigma_c)
    fitter = MapFitter(wm.m, **(fit_kwargs or {}))
    gi = model.index["gain"]

    @jax.jit
    def laplace_cov(phi):
        J = jax.jacfwd(wm.m)(phi)
        H = J.T @ J + fitter.p * jnp.eye(phi.shape[0])
        return jnp.linalg.inv(H)[jnp.ix_(gi, gi)]

    mu = jax.jit(model.mu)

    def fit(key, d):
        res = fitter.fit(wm.y(d), model.init_params(key))
        return {"gain_z": res.phi[gi], "gain_cov": laplace_cov(res.phi), "chi2": res.chi2_per_dof,
                "passed": res.passed, "vis_model": mu(res.phi)}

    return fit


def gain_errors(gp, z_true, z_fit):
    """rms fractional amplitude error and rms phase error (deg) over time and antennas."""
    a_t, p_t = gp.log_amp_phase(z_true)
    a_f, p_f = gp.log_amp_phase(z_fit)
    amp = jnp.sqrt(jnp.mean((jnp.exp(a_f - a_t) - 1.0) ** 2))
    dph = jnp.angle(jnp.exp(1j * (p_f - p_t)))
    return float(amp), float(jnp.rad2deg(jnp.sqrt(jnp.mean(dph**2))))


def run_b0(
    key,
    base: Simulator,
    configs: Dict[str, SimConfig],
    n_sims: int = 20,
    fit_fn: Optional[Callable] = None,
    model_kwargs: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
):
    """Metrics per configuration name. Gains drawn from the prior in every simulation."""
    model = CompressionModel(base.geom, base.gains, **(model_kwargs or {}))
    fit_fn = fit_fn or default_fit_fn(model, fit_kwargs)
    out = {}
    for name, cfg in configs.items():
        sim = Simulator(base.geom, base.gains, cfg)
        simulate = jax.jit(sim.simulate)
        rows = []
        for k in jax.random.split(jax.random.fold_in(key, zlib.crc32(name.encode()) % 2**31), n_sims):
            k_z, k_s, k_f = jax.random.split(k, 3)
            z = jax.random.normal(k_z, (base.gains.n_params,))
            vis, aux = simulate(k_s, z)
            d = vis[..., 0]
            f = fit_fn(k_f, d)
            amp_err, ph_err = gain_errors(base.gains, z, f["gain_z"])
            zs = (np.asarray(f["gain_z"]) - np.asarray(z)) / np.sqrt(np.diag(np.asarray(f["gain_cov"])))
            clean = aux["vis_ast"][..., 0] + aux["vis_rfi"][..., 0]
            g = aux["gains"][..., 0]
            truth = g[:, base.geom.a1] * clean * jnp.conj(g[:, base.geom.a2])
            rows.append({
                "chi2": f["chi2"],
                "passed": bool(f["passed"]),
                "amp_err": amp_err,
                "phase_err_deg": ph_err,
                "vis_rms_err": float(jnp.sqrt(jnp.mean(jnp.abs(f["vis_model"] - truth) ** 2))),
                "laplace_cov68": float(np.mean(np.abs(zs) < 1.0)),
            })
        out[name] = {
            "convergence_rate": float(np.mean([r["passed"] for r in rows])),
            **{k: float(np.median([r[k] for r in rows])) for k in ("chi2", "amp_err", "phase_err_deg", "vis_rms_err")},
            "laplace_cov68": float(np.mean([r["laplace_cov68"] for r in rows])),
            "rows": rows,
        }
    return out
