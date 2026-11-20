# Step 9 — B0 stress test and Protocol B (target fields)

## B0: stress test of the unmodified method (do first)

[`stress_test.run_b0(key, base_simulator, {name: SimConfig}, n_sims)`](../src/tabsim_sbi/stress_test.py)
switches on one enhancement at a time. Each simulation draws gains from the prior and runs one MAP fit;
the harness reports the convergence rate (χ² < 1.1), median χ², rms gain amplitude and phase errors,
the rms error of the recovered visibilities, and the fraction of gain parameters inside their ±1σ
Laplace interval (0.68 if the uncertainties are calibrated).

The fitter is pluggable. By default it is this repository's Paper II style model at Paper II
settings (24 s RFI correlation time), standing in for tabascal. **To stress-test tabascal itself**,
pass `fit_fn(key, d) -> {"gain_z", "gain_cov", "chi2", "passed", "vis_model"}` that writes `d` to a
measurement set (tab-sim's `write_ms`), runs the tabascal CLI and maps its gains into
`GainParameterisation` coordinates. That adapter is not written yet ([12](12_open_points.md)).

### Result on the toy calibrator

`python examples/b0_stress_test.py --n-sims 20` (6 antennas, 60 s, one GPS satellite 4° off axis,
~35 Jy mean RFI, 10 Jy calibrator, 20 simulations per row, model at Paper II settings):

| Configuration | Converged (χ² < 1.1) | median χ² | gain amp. error | gain phase error | vis. rms error (Jy) | Laplace 68% coverage |
|---|---|---|---|---|---|---|
| V0 (tab-sim) | 1.00 | 1.05 | 0.6% | 0.4° | 0.28 | 0.60 |
| E3 self-noise | 0.00 | 1.24 | 0.6% | 0.4° | 0.29 | 0.60 |
| E1 TLE errors (σ_RTN 40/180/50 m + drift) | 0.75 | 1.05 | 0.6% | 0.3° | 0.28 | 0.68 |
| E1 unmodelled satellite (6° off axis, half power) | 0.00 | 66 | **11.4%** | **7.0°** | 5.2 | **0.04** |
| E2 fast power (τ = 2 s, σ_log 0.5) + 1′ pointing, 1% dish errors | 0.00 | 292 | **10.4%** | **5.2°** | 11.1 | **0.03** |
| E4 faint sky (≈20 sources, 0.05–2 Jy, within 1°) | 0.00 | 1.65 | 2.3% | 1.5° | 0.60 | 0.30 |
| E5 gain residual (1%, 0.05 rad, τ = 10 s) + 0.5% non-closing | 0.00 | 2.75 | 2.6% | 2.4° | 0.89 | 0.13 |

What B0 says, as the protocol intends it to:

* **Benign at this RFI level:** TLE errors are absorbed by the time-correlated `A_ps`, as Paper II
  argues for variant A1. E3 does not hurt the gains (errors unchanged), **but it breaks the χ²
  gate**: a thermal-σ χ² of 1.24 rejects every fit. Any gate on real strong-RFI data must use
  signal-dependent σ ([07](07_fiducial_proposal_npe.md)).
* **Damaging, worth an inference protocol:** unmodelled emitters and RFI variation faster than the
  model's 24 s correlation time bias the gains by ~10% and several degrees, with Laplace errors that
  cover the truth 3–4% of the time instead of 68%. E5 and E4 do moderate damage with overconfident errors.
* V0 itself has Laplace coverage 0.60 rather than 0.68: the Laplace approximation is slightly
  overconfident even when the model is right, at 20 simulations × 22 parameters.

These numbers come from a 6-antenna toy at ~35 Jy mean RFI with a stand-in for tabascal. They rank
the effects; they do not measure tabascal's real-data performance.

## B1: gain phases on target fields

`protocol_b1(simulator)`: compression model with `sky="freeform"` (known sky plus a free complex GP
per baseline in time), θ = gain-phase inducing points in blocks of one antenna, η = gain
amplitudes, RFI and free-form sky. The hardening is unchanged; its cost scales with the number of phases
(`n_θ` CG solves), and the nuisance count enters only through the cost of each JVP. Paper II finds phase
constraints only where RFI SNR ≳ 1, so B1 is informative only in that regime. Status: the
specification builds, simulates and compresses on a toy (`tests/test_inference.py`). It has not been run at
scale, and the free-form sky's GP prior (length scale, amplitude) needs to match Paper II's
fringe-rate prior before results mean anything.

## B2: sky parameters (the protocol's own proposal)

`protocol_b2(simulator, lmn, I0, sigma)`: the free-form sky is replaced by a parametric sky (fluxes
at known positions), θ = those fluxes (one block each), η = gains and RFI. In the simulator the
fluxes come from θ, and E4 supplies the rest of the sky as a latent. Flux posteriors inherit the
gain-amplitude prior, since amplitudes are not calibrated on target fields. Status: builds and compresses;
untested at scale.
