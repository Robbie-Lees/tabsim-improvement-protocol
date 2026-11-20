"""Protocol A (variant A1) end to end on a small synthetic calibrator observation.

    python examples/protocol_a_toy.py --scenario unmodelled [--gpu] [--n-sims 3000]

Scenarios (6 antennas, 30 x 2 s, one modelled GPS satellite, 10 Jy calibrator):

* ``moderate``   ~30 Jy RFI, E1 TLE errors + E3 self-noise on;
* ``strong``     ~180 Jy RFI, same effects, RFI correlation time 12 s in the model;
* ``unmodelled`` as ``moderate`` plus a second, unmodelled satellite 6 deg away.

Fits the Paper II style compression model, hardens per antenna, trains one
MDN per antenna, and compares the SBI posterior with the fit's Laplace
approximation. Then runs V1 on held-out simulations.
"""

import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", choices=["moderate", "strong", "unmodelled"], default="unmodelled")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--n-sims", type=int, default=3000)
parser.add_argument("--n-test", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if not args.gpu:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tabsim_sbi import GainParameterisation, SimConfig, Simulator, synthetic_geometry  # noqa: E402
from tabsim_sbi.protocols import protocol_a, run_protocol, validate_blocks  # noqa: E402

Pv, offset, rfi_ls = {"moderate": (1.5e-6, 4.0, 24.0), "strong": (4e-6, 3.0, 12.0), "unmodelled": (1.5e-6, 4.0, 24.0)}[args.scenario]
geom = synthetic_geometry(n_ant=6, n_time=30, n_int=8, sat_offsets_deg=(offset,), sat_Pv=Pv,
                          pool_offsets_deg=(6.0,), pool_Pv=Pv / 2, ast_I=(10.0,), seed=args.seed)
gp = GainParameterisation(geom.times, geom.n_ant, n_g=2, length_scale=40.0, sigma_amp=0.1, sigma_phase=0.3)
cfg = SimConfig(noise="diag", noise_blocks=2, tle_sigma_rtn=(40.0, 180.0, 50.0), tle_drift_rtn=(0.05, 0.5, 0.05),
                unmodelled_prob=1.0 if args.scenario == "unmodelled" else 0.0)
sim = Simulator(geom, gp, cfg)
spec = protocol_a(sim, rfi_length_scale=rfi_ls)

key = jax.random.PRNGKey(args.seed)
z_true = jax.random.normal(jax.random.fold_in(key, 1), (gp.n_params,))
vis, aux = sim.simulate(jax.random.fold_in(key, 2), z_true)
d_obs = vis[..., 0]
print(f"scenario '{args.scenario}': {geom.n_ant} antennas, {geom.n_time} x {geom.int_time:.0f} s, {geom.n_int} fine samples")
print(f"|V_rfi| mean {float(jnp.abs(aux['vis_rfi']).mean()):.1f} Jy; thermal sigma {float(geom.noise_std[0]):.3f} Jy per component")
print(f"theta: {spec.theta_idx.size} gain parameters in {len(spec.blocks)} blocks;"
      f" phi: {spec.model.dim} parameters ({spec.model.index['rfi'].size} RFI nuisances)")

t0 = time.time()
res = run_protocol(jax.random.fold_in(key, 3), spec, d_obs, n_sims=args.n_sims, signal_dependent_sigma=True)
print(f"\nfit chi2/dof = {res.fit.chi2_per_dof:.3f} (gate passed: {res.fit.passed});"
      f" proposal: {'Laplace around fit' if res.fit.passed else 'prior (gate failed)'}; total {time.time() - t0:.0f} s")

lap_sd = np.sqrt(np.diag(np.asarray(res.hardener.cov)))
sbi_sd = np.sqrt(np.diag(np.asarray(res.joint_cov)))
zt = np.asarray(z_true)
print("\n param   truth    fit +- Laplace       SBI posterior     block ESS")
for b, pos in zip(res.blocks, res.hardener.blocks):
    for j, p in enumerate(pos):
        ess = f"   {b.ess:7.0f}" if j == 0 else ""
        print(f" {int(spec.theta_idx[p]):5d} {zt[p]:+7.3f}   {float(res.hardener.theta_star[p]):+7.3f} +- {lap_sd[p]:.3f}"
              f"   {float(res.joint_mean[p]):+7.3f} +- {sbi_sd[p]:.3f}{ess}")
z_lap = (np.asarray(res.hardener.theta_star) - zt) / lap_sd
z_sbi = (np.asarray(res.joint_mean) - zt) / sbi_sd
print(f"\nrms z-score of the truth (1 if calibrated): Laplace {np.sqrt(np.mean(z_lap**2)):.2f}, SBI {np.sqrt(np.mean(z_sbi**2)):.2f}")
print(f"fraction of truths within 1 sigma (0.68 if calibrated): Laplace {np.mean(np.abs(z_lap) < 1):.2f}, SBI {np.mean(np.abs(z_sbi) < 1):.2f}")
print(f"median width ratio SBI / Laplace: {np.median(sbi_sd / lap_sd):.2f}")

val = validate_blocks(jax.random.fold_in(key, 4), res, n_test=args.n_test)
pmin = min(float(np.min(v["rank_pvalues"])) for v in val)
dev = max(float(np.max(np.abs(v["coverage"] - v["coverage_grid"]))) for v in val)
signed = [float(np.mean(v["coverage"] - v["coverage_grid"])) for v in val]
print(f"\nV1 on {args.n_test} held-out simulations: min SBC rank p-value over {sum(len(v['rank_pvalues']) for v in val)}"
      f" dimensions {pmin:.3f}; max |coverage - nominal| {dev:.3f}")
print("mean (coverage - nominal) per block, >0 conservative, <0 overconfident: "
      + ", ".join(f"{s:+.3f}" for s in signed))
