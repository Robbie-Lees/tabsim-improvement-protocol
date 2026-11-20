"""B0: how much does each enhancement damage an unmodified fit?

    python examples/b0_stress_test.py [--gpu] [--n-sims 20] [--rfi-scale 30]

One MAP fit of the Paper II style model per simulation, one enhancement at a
time, at ~30 Jy mean RFI (use --rfi-scale for more). Prints convergence rate (chi^2 < 1.1),
gain errors and the 68% coverage of the Laplace uncertainties.
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--n-sims", type=int, default=20)
parser.add_argument("--rfi-scale", type=float, default=1.0, help="multiplies Pv; 1 gives ~30 Jy mean RFI")
args = parser.parse_args()
if not args.gpu:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tabsim_sbi import GainParameterisation, SimConfig, Simulator, synthetic_geometry  # noqa: E402
from tabsim_sbi.enhancements.sky import FaintSkyConfig  # noqa: E402
from tabsim_sbi.stress_test import run_b0  # noqa: E402

geom = synthetic_geometry(n_ant=6, n_time=30, n_int=8, sat_offsets_deg=(4.0,), sat_Pv=1.5e-6 * args.rfi_scale,
                          pool_offsets_deg=(6.0,), pool_Pv=0.75e-6 * args.rfi_scale, ast_I=(10.0,), seed=1)
gp = GainParameterisation(geom.times, geom.n_ant, n_g=2, length_scale=40.0, sigma_amp=0.1, sigma_phase=0.3)
base = Simulator(geom, gp, SimConfig())
_, aux = base.simulate(jax.random.PRNGKey(0), jnp.zeros(gp.n_params))
P = aux["total_power"][..., 0]
print(f"mean |V_rfi| {float(jnp.abs(aux['vis_rfi']).mean()):.0f} Jy; total power / SEFD up to {float(P.max() / geom.SEFD[0]):.2f}")

configs = {
    "V0 (tab-sim)": SimConfig(),
    "E3 self-noise": SimConfig(noise="full", noise_blocks=2),
    "E1 TLE errors": SimConfig(tle_sigma_rtn=(40.0, 180.0, 50.0), tle_drift_rtn=(0.05, 0.5, 0.05)),
    "E1 unmodelled sat": SimConfig(unmodelled_prob=1.0),
    "E2 power + beams": SimConfig(power_tau=2.0, power_sigma_log=0.5, pointing_sigma_arcmin=1.0, dish_scale_sigma=0.01),
    "E4 faint sky": SimConfig(faint_sky=FaintSkyConfig(rate=20.0, n_max=40, S_min=0.05, S_max=2.0, r_max_deg=1.0)),
    "E5 gain residual": SimConfig(gain_resid_sigma_amp=0.01, gain_resid_sigma_phase=0.05, gain_resid_tau=10.0, nonclosing_sigma=0.005),
}
out = run_b0(jax.random.PRNGKey(1), base, configs, n_sims=args.n_sims)  # Paper II model settings (24 s RFI GP)
print(f"\n{'configuration':20s} {'conv.':>6s} {'chi2':>6s} {'amp err':>8s} {'ph err':>8s} {'vis err':>8s} {'Lap68':>6s}")
for name, r in out.items():
    print(f"{name:20s} {r['convergence_rate']:6.2f} {r['chi2']:6.3f} {100 * r['amp_err']:7.2f}% {r['phase_err_deg']:7.2f}d"
          f" {r['vis_rms_err']:8.3f} {r['laplace_cov68']:6.2f}")
