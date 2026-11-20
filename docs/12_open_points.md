# Open points (protocol §8), with what this work settled

| Protocol open point | Status |
|---|---|
| The noise convention | **Settled at code level.** tab-sim writes per-component σ to `SIGMA`; tabascal's likelihood uses it per component. Consistent; only the papers' notation differs ([01](01_baseline_audit.md)). Still to decide: SEFD referred to sky or correlator (`noise_frame`) |
| Whether block-wise inference is adequate; how to carry cross-block correlations | The Schur construction gives exact per-block hardening against everything else and the full Laplace cross-covariance from one set of solves ([06](06_hardening.md)). Adequacy of block-conditioning (each NPE sees only its own summary) is **untested** beyond the toy. Test: compare with a joint NPE on a 4-antenna array where θ is small enough |
| Emitter populations and power priors (E1); correlation times (E2) | Open. Mechanisms implemented with explicit prior parameters; values need V3 |
| Which tractable effects go into tabascal's likelihood | E3's diagonal form is already usable in the score (`signal_dependent_sigma`). Adding it to tabascal's likelihood means replacing the fixed `noise` by `σ_pq(φ)` from the model's autos (`CompressionModel.total_power_fine` shows the computation). E5's non-closing errors and E8 are also tractable |
| B2: choice of parametric sky; sensitivity to omitted sources | Open. `protocol_b2` accepts any positions; E4 supplies the omitted-source latent for the sensitivity study |

## New open points from the implementation

1. **Compression cost** is negligible for the Gaussian score (linear projection). The bottleneck
   is now the simulation itself at full fine-sample resolution (≈1 GB of field per satellite for
   64 antennas × 15 min × 1025 samples). Time-chunking the Gram product, or justified thinning of the
   fine grid, is needed before full-size Protocol A on a GPU.
2. **Hardening solves** scale as `n_θ × n_CG`. Preconditioning and block-CG are untested.
3. **tabascal adapter.** The machinery takes any differentiable `mu(φ)` with an identity prior;
   current tabascal (FFT-GP RFI, gauge-fixed ConstGains) differs from Paper II's parameterisation. An
   adapter via `ravel_pytree` over tabascal's `forward` is the next piece of glue.
4. **Variant A2** (orbit parameters in θ) is not implemented. With the geometry cache it means
   making `sat_xyz` a differentiable function of orbital elements in the compression model, e.g.
   with tab-sim's JAX `kepler_orbit`. Fringe-wrapping aliasing (Heavens et al.; Alsing & Wandelt)
   suggests summaries at several expansion points.
5. **NPE calibration.** Held-out V1 shows the per-block MDNs slightly overconfident (mean coverage
   deficit up to 0.07 in the worst block, [09](09_protocol_A.md)). Try ensembles, more simulations,
   or a calibration-aware loss.
6. **Proposal after a failed gate.** The prior fallback is valid but wastes information when the fit
   only just misses (χ² 1.13 → 4.6× wider than Laplace on a case where Laplace was fine). Try a
   defensive mixture proposal or a sequential second round.
7. **RFI correlation time.** At high SNR the Paper II 24 s was too slow on the toy (12 s needed).
   Choose it per observation, or fit it.
8. **Fiducial far from truth.** On real data the χ² gate may fail. Iterating the fiducial
   (Fisher scoring, then re-freeze) changes only optimality, but changes the summaries: data and
   all simulations must be re-compressed.
