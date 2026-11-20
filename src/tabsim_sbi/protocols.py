"""Protocols A, B1 and B2 as specifications, and one runner for all of them.

A :class:`ProtocolSpec` says *what is inferred*: which compression-model
parameters are interesting (``theta_idx``), how they are blocked, and how a
value of theta becomes simulator inputs (``theta_to_sim``). Everything else in
the compression model is a nuisance to harden against; everything else in the
simulator is latent. The two need not coincide -- the compression model's RFI
nuisances (A_ps inducing points) have no counterpart in the simulator, whose RFI
comes from powers, trajectories and beams -- and that is what makes this
simulation-based inference rather than a fit.

:func:`run_protocol` follows the protocol's steps:

1. MAP fit of the compression model to the observed data -> phi* (chi^2 gate).
2. Freeze phi*; build the score compressor and the hardener (CG solves once).
3. Draw theta from a Gaussian proposal around the fit (inflated Laplace) --
   or from the prior if the fit failed the chi^2 gate (``proposal="auto"``) --
   simulate in vmapped batches, compress to per-block summaries.
4. Train one NPE per block on (theta_b, summary_b); evaluate at the observed
   summaries; correct to the prior by importance resampling.
5. Assemble a joint Gaussian approximation: per-block posterior covariances on
   the diagonal, cross-block terms from the Laplace correlation (approximate,
   protocol Section 3.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tabsim_sbi.compression.fiducial import FitResult, chi2_with, map_fit
from tabsim_sbi.compression.hardening import Hardener
from tabsim_sbi.compression.model import CompressionModel
from tabsim_sbi.compression.score import ScoreCompressor, WhitenedModel
from tabsim_sbi.inference.mdn import MDN, train_mdn
from tabsim_sbi.inference.proposal import GaussianProposal, sir_correct, standard_normal_logpdf
from tabsim_sbi.simulator import Simulator


@dataclass
class ProtocolSpec:
    name: str
    model: CompressionModel
    simulator: Simulator
    theta_idx: np.ndarray  # indices into the compression model's phi
    blocks: List[np.ndarray]  # indices into phi, partitioning theta_idx
    theta_to_sim: Callable  # (key, theta) -> dict of Simulator.simulate kwargs


def _gain_theta_to_sim(gp, theta_pos_in_gain):
    """theta fills the given gain parameters; the rest are drawn from the prior."""

    def f(key, theta):
        z = jax.random.normal(key, (gp.n_params,))
        return {"gain_z": z.at[theta_pos_in_gain].set(theta)}

    return f


def protocol_a(simulator: Simulator, **model_kwargs) -> ProtocolSpec:
    """Calibrator: theta = all gain amplitude and phase inducing points, blocks per antenna."""
    model = CompressionModel(simulator.geom, simulator.gains, sky="calibrator", **model_kwargs)
    theta_idx = model.index["gain"]
    return ProtocolSpec("A1", model, simulator, theta_idx, model.theta_blocks(),
                        _gain_theta_to_sim(simulator.gains, np.arange(simulator.gains.n_params)))


def protocol_b1(simulator: Simulator, **model_kwargs) -> ProtocolSpec:
    """Target field: theta = gain-phase inducing points; free-form sky is a nuisance."""
    model = CompressionModel(simulator.geom, simulator.gains, sky="freeform", **model_kwargs)
    gp = simulator.gains
    phase_pos = np.arange(gp.n_amp, gp.n_params)
    theta_idx = model.index["gain"][phase_pos]
    return ProtocolSpec("B1", model, simulator, theta_idx, model.theta_blocks(amplitudes=False),
                        _gain_theta_to_sim(gp, phase_pos))


def protocol_b2(simulator: Simulator, lmn, I0, sigma=None, **model_kwargs) -> ProtocolSpec:
    """Target field: theta = fluxes of a few sources at known positions, one block each.

    The simulator's sky becomes the parametric sources (fluxes from theta) plus
    whatever latent sky its config adds (E4); the geometry's own sky model is
    replaced.
    """
    I0 = jnp.asarray(I0, dtype=jnp.float64).reshape(-1)
    sigma = 0.2 * I0 if sigma is None else jnp.asarray(sigma, dtype=jnp.float64).reshape(-1)
    geom = simulator.geom.replace(ast_lmn=jnp.asarray(lmn), ast_I=I0[:, None])
    sim = Simulator(geom, simulator.gains, simulator.config)
    model = CompressionModel(geom, sim.gains, sky="parametric", parametric_lmn=lmn,
                             parametric_I0=I0, parametric_sigma=sigma, **model_kwargs)
    theta_idx = model.index["sky"]
    gp = sim.gains

    def to_sim(key, theta):
        return {"gain_z": jax.random.normal(key, (gp.n_params,)), "ast_I": (I0 + sigma * theta)[:, None]}

    return ProtocolSpec("B2", model, sim, theta_idx, [np.array([i]) for i in theta_idx], to_sim)


# ----------------------------------------------------------------------------


@dataclass
class BlockPosterior:
    indices: np.ndarray  # into phi
    npe: MDN
    samples: jnp.ndarray  # prior-corrected posterior samples (standardised units)
    ess: float

    @property
    def mean(self):
        return self.samples.mean(0)

    @property
    def cov(self):
        return jnp.atleast_2d(jnp.cov(self.samples.T))


@dataclass
class ProtocolResult:
    spec: ProtocolSpec
    fit: FitResult
    hardener: Hardener
    compressor: ScoreCompressor
    proposal: GaussianProposal
    theta_sims: jnp.ndarray
    summaries: jnp.ndarray
    s_obs: jnp.ndarray
    blocks: List[BlockPosterior] = field(default_factory=list)
    joint_mean: Optional[jnp.ndarray] = None
    joint_cov: Optional[jnp.ndarray] = None


def compress_fn(compressor: ScoreCompressor, hardener: Hardener):
    """Data -> summaries (n_theta,), the same function for data and simulations."""
    if compressor.nu is None:
        return lambda d: hardener.qmle_from_residual(compressor.residual(d))
    return lambda d: hardener.qmle(compressor.score(d))


def simulate_summaries(key, spec: ProtocolSpec, compress, proposal: GaussianProposal, n_sims: int, batch_size: int = 64):
    """Draw theta from the proposal, simulate and compress. Returns (theta, summaries)."""
    k_theta, k_sim = jax.random.split(key)
    theta = proposal.sample(k_theta, n_sims)

    def one(k, th):
        k1, k2 = jax.random.split(k)
        vis, _ = spec.simulator.simulate(k2, **spec.theta_to_sim(k1, th))
        return compress(vis[..., 0])

    run = jax.jit(jax.vmap(one))
    keys = jax.random.split(k_sim, n_sims)
    out = [run(keys[i : i + batch_size], theta[i : i + batch_size]) for i in range(0, n_sims, batch_size)]
    return theta, jnp.concatenate(out)


def run_protocol(
    key,
    spec: ProtocolSpec,
    d_obs,
    *,
    n_sims: int = 2000,
    nu: Optional[float] = None,
    signal_dependent_sigma: bool = False,
    proposal: str = "auto",
    proposal_inflate: float = 2.0,
    n_post: int = 2000,
    fit_kwargs: Optional[dict] = None,
    mdn_kwargs: Optional[dict] = None,
    batch_size: int = 64,
) -> ProtocolResult:
    """Run the protocol on observed visibilities ``d_obs`` (n_time, n_bl)."""
    k_init, k_sims, k_npe = jax.random.split(key, 3)
    model = spec.model

    # 1. Fiducial
    wm0 = WhitenedModel(model.mu, model.sigma_c)
    fit = map_fit(wm0.m, wm0.y(d_obs), model.init_params(k_init), **(fit_kwargs or {}))
    phi_star = fit.phi

    # 2. Frozen compression (optionally with E3-aware weights at phi*). With E3 the
    # noise exceeds thermal, so the chi^2 gate is re-evaluated with the same sigma;
    # a thermal-sigma gate rejects good fits whenever the RFI is a sizeable
    # fraction of the SEFD.
    sigma = model.signal_dependent_sigma(phi_star) if signal_dependent_sigma else model.sigma_c
    if signal_dependent_sigma:
        fit.chi2_per_dof = chi2_with(model.mu, phi_star, d_obs, sigma)
        fit.passed = fit.chi2_per_dof < (fit_kwargs or {}).get("gate", 1.1)
    wm = WhitenedModel(model.mu, sigma)
    compressor = ScoreCompressor(wm, phi_star, nu=nu)
    hardener = Hardener(wm.m, phi_star, spec.theta_idx, spec.blocks)
    compress = compress_fn(compressor, hardener)
    s_obs = compress(d_obs)

    # 3. Simulations. A proposal centred on a fit that failed the gate can exclude the
    # truth (unmodelled emitters bias the fit far beyond its Laplace errors), which
    # breaks validity, not just optimality. "auto" falls back to the prior then.
    if proposal == "prior" or (proposal == "auto" and not fit.passed):
        n_t = spec.theta_idx.size
        prop = GaussianProposal(jnp.zeros(n_t), jnp.eye(n_t))
    elif proposal in ("auto", "laplace"):
        prop = GaussianProposal.around_fit(hardener.theta_star, hardener.cov, proposal_inflate)
    else:
        raise ValueError(f"unknown proposal {proposal!r}")
    proposal = prop
    theta, summ = simulate_summaries(k_sims, spec, compress, proposal, n_sims, batch_size)

    result = ProtocolResult(spec, fit, hardener, compressor, proposal, theta, summ, s_obs)

    # 4. Per-block NPE with prior correction
    for k, (b_idx, b_pos) in zip(jax.random.split(k_npe, len(spec.blocks)), zip(spec.blocks, hardener.blocks)):
        k_train, k_samp, k_sir = jax.random.split(k, 3)
        npe = train_mdn(k_train, theta[:, b_pos], summ[:, b_pos], **(mdn_kwargs or {}))
        raw = npe.sample(k_samp, s_obs[b_pos], 4 * n_post)
        marg = proposal.marginal(jnp.asarray(b_pos))
        samples, ess = sir_correct(k_sir, raw, standard_normal_logpdf(raw), marg.log_prob(raw), n_post)
        result.blocks.append(BlockPosterior(np.asarray(b_idx), npe, samples, float(ess)))

    # 5. Joint Gaussian assembly
    result.joint_mean, result.joint_cov = assemble_joint(result.blocks, hardener)
    return result


def validate_blocks(key, result: ProtocolResult, n_test: int = 200, n_post: int = 200, batch_size: int = 64):
    """V1 on held-out simulations from the same proposal.

    The per-block NPEs are amortised over the proposal region, so fresh
    proposal draws test them directly (before the prior correction, which is
    exact given a calibrated q). Returns per block: SBC rank p-values per
    dimension and the expected-coverage curve.
    """
    from tabsim_sbi.validation import calibration as v1

    spec, h = result.spec, result.hardener
    compress = compress_fn(result.compressor, h)
    k_sim, k_val = jax.random.split(key)
    theta, summ = simulate_summaries(k_sim, spec, compress, result.proposal, n_test, batch_size)
    out = []
    for k, bp, pos in zip(jax.random.split(k_val, len(result.blocks)), result.blocks, h.blocks):
        sample = lambda kk, s, n, npe=bp.npe: npe.sample(kk, s, n)
        k1, k2 = jax.random.split(k)
        ranks = v1.sbc_ranks(k1, sample, theta[:, pos], summ[:, pos], n_post)
        levels = v1.credibility_levels(k2, sample, bp.npe.log_prob, theta[:, pos], summ[:, pos], n_post)
        grid, cov = v1.coverage_curve(levels)
        out.append({"rank_pvalues": v1.rank_uniformity_pvalues(ranks, n_post), "coverage_grid": grid, "coverage": cov})
    return out


def assemble_joint(blocks: List[BlockPosterior], hardener: Hardener):
    """Block posteriors on the diagonal; cross-block terms from the Laplace correlation.

    ``Sigma_bc = D_b^1/2 R^L_bc D_c^1/2`` with ``R^L`` the Laplace correlation
    whitened per block (``R^L_bc = C_bb^-1/2 C_bc C_cc^-1/2``) and ``D_b`` the
    block posterior covariance. Exact when every block posterior equals its
    Laplace approximation; otherwise an approximation (protocol Section 3.5).
    """
    def msqrt(A, inv=False):
        w, V = jnp.linalg.eigh(0.5 * (A + A.T))
        w = jnp.clip(w, 1e-300)
        return (V * (w ** (-0.5 if inv else 0.5))) @ V.T

    n = hardener.theta_idx.size
    mean = jnp.zeros(n)
    cov = jnp.zeros((n, n))
    C = hardener.cov
    for bp, pb in zip(blocks, hardener.blocks):
        mean = mean.at[pb].set(bp.mean)
    for bp, pb in zip(blocks, hardener.blocks):
        Db = msqrt(bp.cov)
        Cb = msqrt(C[jnp.ix_(pb, pb)], inv=True)
        for bq, pq in zip(blocks, hardener.blocks):
            Dq = msqrt(bq.cov)
            Cq = msqrt(C[jnp.ix_(pq, pq)], inv=True)
            cov = cov.at[jnp.ix_(pb, pq)].set(Db @ Cb @ C[jnp.ix_(pb, pq)] @ Cq @ Dq)
    return mean, 0.5 * (cov + cov.T)
