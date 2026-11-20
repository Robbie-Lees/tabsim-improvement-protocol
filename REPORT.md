# Report: tab-sim improvements for simulation-based inference — status and how to run

*20 November 2026. Implements the draft note "Protocols for enhancing tab-sim with non-Gaussian
effects and compressing its output for SBI" (v2). Code: `src/tabsim_sbi`; details in `docs/`.*

## 1. What was built

tab-sim 0.1.0 is used as a dependency and never modified. The new package does three things:

1. **A fast simulator of one observation.** tab-sim is run once to get the geometry (antenna and
   satellite trajectories, uvw, known sky). After that, every simulation is a JAX function that can be
   jit-compiled, vmapped and run on a GPU. With all enhancements off it *is* tab-sim. Seven of the protocol's eight
   physical effects are added as random layers drawn afresh in every simulation:

   | ID | Effect | Implementation |
   |---|---|---|
   | E1 | Unmodelled satellites / ground emitters, TLE position errors | RTN-frame trajectory errors; unmodelled "pool" satellites from tab-sim's own selection; ground emitters placed with tab-sim's rotation |
   | E2 | Realistic RFI signal | power fluctuations (log-normal OU × on/off duty cycle), per-antenna pointing and dish errors, complex beam |
   | E3 | Signal-dependent noise | exact sampler `η = L G Lᴴ/√M` from the full coherency matrix (Kulkarni's covariance) |
   | E4 | Unmodelled sky | power-law faint and extended sources in an annulus |
   | E5 | Gain errors the model cannot follow; non-closing errors | fast GP gain residual; baseline factors (1+ε) |
   | E6 | Receiver compression | total-power saturation factor |
   | E8 | Heavy-tailed noise | complex Student-t |
   | E7 | Ionosphere | deferred, as in the protocol (design note only) |

2. **Compression.** The data are reduced to the score of a Paper II-style model (gains + RFI GP +
   sky) at a frozen MAP fit, then hardened against all nuisances per antenna. One set of
   conjugate-gradient solves hardens every block and also gives the cross-antenna covariance.

3. **Inference.** One small neural density estimator (mixture density network, pure JAX) per
   antenna, a prior correction, and assembly into a joint gain posterior. There are also a stress
   test (B0) and validation tools (calibration V1, realism statistics V3).

## 2. Is it working?

### Verified

* **Simulator = tab-sim in the Gaussian limit.** It matches tab-sim 0.1.0's own kernels to 1e-9.
  It also reproduces the stored RFI visibilities of a real 32-antenna, 3-satellite MeerKAT tab-sim
  run to 3e-9.
* **The protocol's noise question (√2) is settled in code.** tab-sim writes the per-component σ
  (0.459 Jy) to `SIGMA` and tabascal's likelihood uses it per component, so the pipeline is
  consistent. Only the papers' notation differs.
* **E3 is exact.** Its covariance and pseudo-covariance match Kulkarni's formulas and a brute-force
  voltage simulation. Under strong RFI the noise std rises by √(P_p P_q)/SEFD, as predicted.
* **E1, E2, E4, E5, E6 and E8** behave as specified (unit tests) and all compose in one compiled call.
* **Compression maths.** The score equals the gradient of the likelihood. The block-wise hardening
  equals Alsing & Wandelt's formula block by block, and is insensitive to nuisances when the prior is
  switched off. The MAP fit passes Paper II's χ² < 1.1 gate on moderate RFI.
* **The full Protocol A pipeline runs end to end** on a small synthetic calibrator.
* **Test suite:** 48 tests pass (3 min 40 s on a laptop CPU), including the parity test on the
  real 32-antenna run.

### Toy results (6 antennas, 60 s, 3000 simulations, CPU)

**Protocol A** (per-antenna gain posteriors; "within 1σ" should be 0.68):

| Scenario | Fit | Laplace errors: truths within 1σ | SBI posterior: truths within 1σ | SBI width / Laplace width |
|---|---|---|---|---|
| Moderate RFI (~30 Jy), TLE errors + self-noise | χ² 1.13 (just misses gate) | 0.73 | 0.82 | 4.6 |
| Strong RFI (~180 Jy), same effects | χ² 0.90 (passes) | 0.59 | 0.77 | 1.2 |
| **Moderate RFI + an unmodelled satellite** | χ² 42 (fails) | **0.00** (rms z = 11) | **0.77** (rms z = 0.7) | 4.9 |

The last row is the point of the whole exercise. When the model misses an emitter, the
conventional fit is badly biased and its error bars are meaningless. SBI applied to the same data
gives calibrated gain posteriors, at the cost of wider errors.

**B0 stress test** (one unmodified fit per simulation, ~35 Jy RFI, 20 simulations per row; runs in
47 s):

| Effect switched on | Gain amplitude / phase error | Laplace 68% interval covers truth |
|---|---|---|
| none (tab-sim as is) | 0.6% / 0.4° | 60% |
| E3 self-noise | 0.6% / 0.4° (but the χ² gate rejects every fit) | 60% |
| E1 TLE errors | 0.6% / 0.3° | 68% |
| **E1 unmodelled satellite** | **11% / 7°** | **4%** |
| **E2 fast RFI power changes + beam errors** | **10% / 5°** | **3%** |
| E4 faint sky | 2.3% / 1.5° | 30% |
| E5 gain residual + non-closing errors | 2.6% / 2.4° | 13% |

So unmodelled emitters and RFI that varies faster than the model's correlation time are what an
SBI protocol is needed for. TLE errors are absorbed by the RFI model, as Paper II argued.

### Not working yet, or not yet shown

* **The per-antenna posteriors are slightly overconfident.** On 200 held-out simulations, coverage
  sits up to 0.07 below nominal on average in the worst antenna block. Usable, not yet
  publication-grade. More simulations or ensembles of density estimators are the likely fix.
* **The fallback when a fit fails is blunt.** If the fit misses the χ² gate, simulations are drawn
  from the prior so that the truth stays covered. That keeps the result valid but makes it wider than
  necessary when the fit only just misses (moderate case: 4.6× wider than needed).
* **Two problems were found and fixed during testing.** (1) Centring the simulations on a failed
  fit excluded the truth (posterior off by ~40σ), hence the fallback above. (2) The χ² gate must use
  the signal-dependent noise: with thermal σ it rejected every good fit under strong RFI. This
  second point also applies to how tabascal's fits are judged on real data.
* **Only tested at toy scale.** 6 antennas × 60 s for inference; 32 antennas for simulator parity.
  A full MeerKAT scan (64 antennas, 15 min, 1025 samples per 2 s) needs the RFI calculation to be
  chunked in time (~1 GB per satellite), which is not done yet.
* **tabascal itself is not plugged in.** The compression model and the B0 fitter are a Paper
  II-style stand-in. Current tabascal parameterises RFI and gains differently, so an adapter is needed.
* **Not yet compared with real data or published posteriors.** There is no comparison with Paper
  I's HMC posteriors (V0) or with real MeerKAT residuals (V3), so all priors marked "to be set" are
  placeholders.
* **Protocol B1/B2 run on toys only.** Variant A2 (orbit inference) and E7 (ionosphere) are not
  implemented.

## 3. How to run

**Install** (Linux/WSL; needs tab-sim ≥ 0.1.0 and JAX; GPU optional):

```bash
git clone https://github.com/Robbie-Lees/tabsim-improvement-protocol
cd tabsim-improvement-protocol
pip install -e ".[zarr,test]"
pytest -m "not slow"            # quick checks (~2 min CPU)
pytest                          # + end-to-end Protocol A
```

**Check against your own tab-sim run** (any tab-sim `.zarr` with TLE satellites):

```bash
TABSIM_SBI_TEST_ZARR=/path/to/obs.zarr pytest tests/test_zarr_parity.py
```

**Try the protocol on synthetic data:**

```bash
python examples/protocol_a_toy.py --scenario unmodelled   # or moderate / strong; add --gpu
python examples/b0_stress_test.py --n-sims 20             # B0 table
```

**Use it on a tab-sim observation:**

```python
import jax
from tabsim_sbi import GainParameterisation, SimConfig, Simulator, from_tabsim_zarr
from tabsim_sbi.protocols import protocol_a, run_protocol, validate_blocks

# 1. Geometry from one ordinary tab-sim run. Simulate with a generous max_n_sat,
#    then choose which satellites the inference model knows; the rest become E1's pool.
geom = from_tabsim_zarr("obs.zarr", modelled=[0, 1], time_range=(0, 150))

# 2. The simulator: switch enhancements on through SimConfig.
gains = GainParameterisation(geom.times, geom.n_ant, n_g=3)
sim = Simulator(geom, gains, SimConfig(noise="diag", tle_sigma_rtn=(40, 180, 50), unmodelled_prob=1.0))

# 3. Protocol A on the observed visibilities d_obs (n_time, n_bl), single channel.
spec = protocol_a(sim, rfi_length_scale=24.0)
res = run_protocol(jax.random.PRNGKey(0), spec, d_obs, n_sims=3000, signal_dependent_sigma=True)
res.fit.passed, res.joint_mean, res.joint_cov       # gain posterior, standardised units
validate_blocks(jax.random.PRNGKey(1), res)          # V1: SBC + coverage on held-out sims
```

To simulate a full "observed" dataset instead, call `sim.simulate(key, gain_z)`, which returns
`(vis, aux)` with the true RFI, sky, gains and noise in `aux`.

## 4. Next steps

1. Write the tabascal adapter and re-run B0 with tabascal on tab-sim measurement sets.
2. Run V3 on a real MeerKAT GNSS-band observation to set the E1/E2/E5/E6 priors.
3. Tighten NPE calibration (ensembles, more simulations) and replace the prior fallback with a
   prior + Laplace mixture proposal.
4. Chunk the RFI calculation in time and run a full-size Protocol A scan on the GPU.
5. V0 against Paper I's HMC posteriors; then Protocol B1 at scale.

Details for every step: `docs/00_overview.md` onwards.
