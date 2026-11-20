# Step 4 — Score compression

## What the protocol asks

Compress the data to the score of an approximate log-likelihood at a fiducial point,
`t = ∇_φ L |_{φ*}`, which equals `2 Re[∇μ*ᴴ C*⁻¹ (d − μ*)]` for circular complex Gaussian noise. Optionally use a Student-t
score to limit the leverage of outliers. Nuisances are latent; the density estimator sees only `(θ, t)`.

## Implementation

**The compression model** ([`compression/model.py`](../src/tabsim_sbi/compression/model.py)) is
Paper II's parameterisation in standardised coordinates (prior N(0, I)):

* gains: the shared `GainParameterisation`;
* RFI: complex `A_ps(t)` per satellite and antenna, an SE-GP (length scale 24 s, Paper II) on
  inducing points, times the fixed fringe phasor `K_ps(t)` from the **TLE** trajectories (A1). The
  prior scale per satellite defaults to the rms nominal apparent amplitude;
* sky: `"calibrator"` (known), `"freeform"` (known part + a free complex GP per baseline, B1) or
  `"parametric"` (a few fluxes at known positions, B2);
* optional thinning of the fine grid (`n_int_model`).

**The score** ([`compression/score.py`](../src/tabsim_sbi/compression/score.py)) uses the *real,
whitened* form: `m(φ) = [Re μ, Im μ]/σ`, `y = [Re d, Im d]/σ`, `t = Jᵀ(y − m*)`, `F = JᵀJ`. This
removes any doubt about complex VJP conventions. It equals the protocol's complex expression with
`C = 2σ²`, the complex variance; the test checks both, and against `jax.grad` of the log-likelihood.

**Student-t score**: per complex datum, the weight `w = (ν+2)/(ν + ρ_re² + ρ_im²)` multiplies the
whitened residual. That is the protocol's `(ν+2)/(ν + 2|r|²/σ_n²)` with `σ_n² = 2σ²`, and it comes from a
bivariate Student-t with per-component scale σ.

## Result: compression is linear in the data

With φ\* frozen, `t = J*ᵀ(y − m*)` is a fixed affine map of the data. Combined with the hardening
projection ([06](06_hardening.md)), the per-block summaries are `θ* + K ρ` with a precomputed
`K = (J X)` of size `n_θ × 2 n_data`. For Protocol A on 64 antennas (≈400 θ, 10⁶ real data) that is
4×10⁸ multiply-adds per simulation: milliseconds on a GPU, far below the simulation itself. The
protocol's estimate that "compression costs about as much as the simulation" (§3.5) is therefore
pessimistic for the Gaussian score. The Student-t score is nonlinear (the weights depend on the
residual), but it still needs only one VJP at the fixed point, with the linearisation reused.

## Verified (`tests/test_compression.py`)

* Score = `jax.grad` of the log-likelihood = the protocol's complex formula (rtol 1e-10).
* Student-t → Gaussian as ν → ∞; with ν = 5 a 10³σ outlier moves the score by <5% of what it
  moves the Gaussian score.
* The compression model reproduces the simulator exactly when there is no RFI (shared gains).
* A MAP fit of the compression model to simulator data passes Paper II's χ² gate, even though the
  simulator's RFI (powers, beams, trajectories) is not in the model's family, and recovers the gains within
  Laplace errors.

## Open

* The quadratic (variance) term of the Gaussian score with signal-dependent `C` is omitted, following
  the protocol's estimate that it carries ~1/(ΔνΔt) of the information for signal parameters. A
  numerical check on the toy would settle it: compare the Fisher information with and without the term.
* Adapter to tabascal's current forward pass (FFT-GP RFI, ConstGains), see [01](01_baseline_audit.md) §5.
