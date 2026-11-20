# tabsim-improvement-protocol

Non-Gaussian enhancements to [tab-sim](https://github.com/chrisfinlay/tab-sim) and score-compressed,
nuisance-hardened **simulation-based inference** for RFI-affected calibration. This is an
implementation of the draft note *Protocols for enhancing tab-sim with non-Gaussian effects and
compressing its output for simulation-based inference* (v2: calibrator and target-field settings).

tab-sim is used as a dependency and **never modified**. The package is `tabsim_sbi`.

## Idea in one paragraph

Run tab-sim once per observation to obtain its geometry (SGP4 trajectories, GCRS antenna positions,
uvw, known sky). Then simulate that observation thousands of times with a pure JAX forward model
that reproduces tab-sim exactly in the Gaussian limit and adds the protocol's effects as stochastic
layers. Those effects are unmodelled emitters and TLE errors (E1), realistic RFI signals and beams (E2), signal-dependent
self-noise (E3), unmodelled sky (E4), gain residuals and non-closing errors (E5), receiver
compression (E6) and heavy tails (E8). Compress each simulation to the score of a Paper II style model at a frozen
MAP fit, harden it against all nuisances per block (one set of conjugate-gradient solves serves all
blocks), and learn per-block posteriors with a small mixture density network.

## Status

| | |
|---|---|
| Forward model vs tab-sim 0.1.0 kernels | agree to 1e-9 |
| Forward model vs a real 32-antenna tab-sim run (`vis_rfi`) | agree to 3e-9 |
| E3 self-noise sampler | exact second moments (Kulkarni), checked against brute-force voltages |
| E1, E2, E4, E5, E6, E8 | implemented, tested; priors "to be set" from real data (V3) |
| Score compression, matrix-free hardening | verified against dense formulas and Alsing & Wandelt |
| Protocol A (A1) | toy: calibrated gain posteriors where the conventional fit fails (unmodelled satellite); NPEs slightly overconfident on held-out checks ([docs/09](docs/09_protocol_A.md)) |
| B0 stress test | ranks the effects on a toy (unmodelled emitters and fast RFI variation do the damage); tabascal itself not yet plugged in |
| Protocol B1, B2 | specifications build and compress; untested at scale |

**Short status report and how to run: [REPORT.md](REPORT.md).**

## Step-by-step documentation

| Step | Doc |
|---|---|
| Overview and design hypothesis | [docs/00_overview.md](docs/00_overview.md) |
| 0 Audit of tab-sim 0.1.0 and tabascal; the noise convention | [docs/01_baseline_audit.md](docs/01_baseline_audit.md) |
| 1 Geometry cache and Gaussian-limit forward model | [docs/02_forward_model.md](docs/02_forward_model.md) |
| 2 E3 signal-dependent noise | [docs/03_E3_self_noise.md](docs/03_E3_self_noise.md) |
| 3 E1 unmodelled emitters and TLE errors | [docs/04_E1_emitters.md](docs/04_E1_emitters.md) |
| 4 Score compression | [docs/05_score_compression.md](docs/05_score_compression.md) |
| 5 Matrix-free hardening | [docs/06_hardening.md](docs/06_hardening.md) |
| 6 Fiducial, proposal, block NPE | [docs/07_fiducial_proposal_npe.md](docs/07_fiducial_proposal_npe.md) |
| 7 E2, E4, E5, E6, E8 (E7 deferred) | [docs/08_further_enhancements.md](docs/08_further_enhancements.md) |
| 8 Protocol A | [docs/09_protocol_A.md](docs/09_protocol_A.md) |
| 9 B0 and Protocol B | [docs/10_protocol_B.md](docs/10_protocol_B.md) |
| Validation V0–V4 | [docs/11_validation.md](docs/11_validation.md) |
| Open points | [docs/12_open_points.md](docs/12_open_points.md) |

## Install and test

```bash
pip install -e ".[zarr,test]"          # requires tabsim>=0.1.0 and JAX
pytest -m "not slow"                   # unit tests, ~1-2 min on CPU
pytest                                 # + Protocol A end to end
TABSIM_SBI_TEST_ZARR=/path/to/tabsim_output.zarr pytest tests/test_zarr_parity.py
```

## Minimal use

```python
import jax, jax.numpy as jnp
from tabsim_sbi import GainParameterisation, SimConfig, Simulator, from_tabsim_zarr
from tabsim_sbi.protocols import protocol_a, run_protocol

geom = from_tabsim_zarr("obs.zarr", modelled=[0, 1], time_range=(0, 150))   # 3rd satellite unmodelled
gains = GainParameterisation(geom.times, geom.n_ant, n_g=3)
sim = Simulator(geom, gains, SimConfig(noise="diag", tle_sigma_rtn=(40, 180, 50), unmodelled_prob=1.0))

spec = protocol_a(sim, n_int_model=8)
result = run_protocol(jax.random.PRNGKey(0), spec, d_obs, n_sims=3000, signal_dependent_sigma=True)
result.joint_mean, result.joint_cov          # gain posterior (standardised units)
```

## Layout

```
src/tabsim_sbi/
  geometry.py        ObservationGeometry; from_tabsim_zarr; synthetic_geometry
  forward.py         coherency-matrix kernels (RFI Gram matrix, sky, gains, beams)
  gains.py           gain parameterisation shared by simulator and compression model
  simulator.py       SimConfig + Simulator (E1-E8 composed)
  enhancements/      emitters (E1), rfi_signal (E2), noise (E3), sky (E4),
                     gain_errors (E5), receiver (E6), heavy_tail (E8)
  compression/       model (Paper II style), score, hardening, fiducial
  inference/         mdn (JAX MDN), proposal (+ prior correction)
  protocols.py       Protocol A / B1 / B2 specs, run_protocol, validate_blocks
  stress_test.py     B0 harness
  validation/        calibration (V1), realism (V3)
examples/            protocol_a_toy.py, b0_stress_test.py
tests/
```
