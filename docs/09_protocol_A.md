# Step 8 — Protocol A: calibrator observations (variant A1)

## Specification

`protocol_a(simulator, **model_kwargs)`:

* compression model: known calibrator; gains from the shared `GainParameterisation`; complex SE-GP
  `A_ps` per antenna and satellite on the **TLE** trajectories (A1: no orbit parameters);
* θ = all gain log-amplitude and phase inducing points, **one block per antenna**;
* η = all RFI inducing points (hardened against);
* simulator latents: everything in `SimConfig` (E1–E8), and the RFI is generated physically, not
  from η.

`run_protocol(key, spec, d_obs, n_sims, signal_dependent_sigma=True)` then runs the steps of
[07](07_fiducial_proposal_npe.md). `validate_blocks` runs V1 on held-out simulations.

## Toy results

`python examples/protocol_a_toy.py --scenario {moderate,strong,unmodelled}` with
`signal_dependent_sigma=True` and `proposal="auto"`: 6 antennas, 30 × 2 s,
8 fine samples per integration, a 10 Jy calibrator, one modelled GPS satellite, E1 TLE errors
(σ_RTN = 40/180/50 m plus drift) and E3 self-noise (diagonal, 2 sub-blocks) in every scenario.
θ = 22 gain parameters in 6 blocks; 3000 simulations; held-out V1 on 200 simulations. CPU only.

| Scenario | mean \|V_rfi\| | fit χ² (E3 σ) | gate → proposal | Laplace: rms z / within 1σ | SBI: rms z / within 1σ | SBI / Laplace width | V1: min SBC p (22 dims) | V1: mean coverage − nominal, per block |
|---|---|---|---|---|---|---|---|---|
| moderate | 29 Jy | 1.13 | fail → prior | 0.92 / 0.73 | 0.60 / 0.82 | 4.6 | 0.004 | −0.069 … −0.004 |
| strong (RFI GP 12 s) | 181 Jy | 0.90 | pass → Laplace | 1.01 / 0.59 | 0.88 / 0.77 | 1.2 | 0.005 | −0.072 … +0.016 |
| **unmodelled** satellite | 29 Jy + unmodelled | 41.6 | fail → prior | **11.3 / 0.00** | **0.71 / 0.77** | 4.9 | 0.030 | −0.071 … +0.006 |

"rms z / within 1σ" is computed over the 22 gain parameters of the one simulated observation, so
it is noisy (calibrated: 1 and 0.68, ±~0.15). V1 uses 200 fresh simulations per scenario.
Each scenario takes ~40 s for fit + 3000 simulations + 6 MDNs on a laptop CPU, plus a few minutes
for V1.

## What the runs taught us

1. **The central claim works on the toy.** With an unmodelled satellite, the compression model is
   wrong, the fit is biased by ~11 Laplace σ and its error bars cover none of the 22 true gain
   parameters. SBI on the same data, with the same frozen compression, covers 17 of 22 (rms z 0.71).
   The price is width: ~5× the (wrong) Laplace errors, which is the "optimality, not validity" cost
   the protocol predicts. The posteriors are still well inside the prior: widths of 0.1–0.7 in
   prior-standardised units.
2. **When the model is adequate, SBI ≈ Laplace** (strong case: 1.2× the width, both calibrated).
3. **The χ² gate needs E3.** Against thermal σ the strong-case fit has χ² = 2.18. With the E3 σ
   at φ\* it has 0.90 and passes.
4. **The prior fallback is blunt.** The moderate fit misses the gate at χ² = 1.13 (TLE errors plus the
   24 s RFI correlation time), so the prior proposal is used and the posterior is 4.6× wider than
   needed, although Laplace was fine there. A defensive mixture proposal (prior + Laplace) or a
   second sequential round would recover most of that.
5. **The RFI correlation time matters at high SNR.** At ~180 Jy the Paper II value of 24 s left
   χ² = 3.6 (gains biased by ~5σ); 12 s gave χ² = 1.07. It should be chosen per observation (or
   fitted), not fixed.
6. **The NPEs are slightly overconfident.** On held-out simulations, mean coverage is up to 0.07 below
   nominal in the worst block (max pointwise deviation 0.13–0.15). The smallest SBC p-value, 0.004
   over 22 dimensions, is not significant after correcting for 22 tests. Ensembles of MDNs, more
   simulations or a calibration-aware loss are the next steps. In the strong case one block's SIR
   ESS was 29 of 8000. A larger `proposal_inflate` for that block would fix it.

## Variant A2 (not implemented)

Orbit parameters in θ as a further block. With the geometry cache this means making `sat_xyz` in
the compression model a differentiable function of orbital elements, e.g. via tab-sim's JAX
`kepler_orbit`. Expect strongly non-linear summaries and fringe-wrap aliasing (Heavens et al.;
Alsing & Wandelt): use Fisher-scoring iteration of the fiducial and summaries at several expansion points.
