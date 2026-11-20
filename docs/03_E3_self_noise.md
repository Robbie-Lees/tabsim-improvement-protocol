# Step 2 — E3: signal-dependent noise

## What the protocol asks

Replace i.i.d. thermal noise by the self-noise of Kulkarni (1989):

```
⟨η_pq η*_rs⟩ = R_pr R*_qs / M,     ⟨η_pq η_rs⟩ = R_ps R_rq / M,     M = Δν Δt
```

with `R` the noise-free coherency including the total powers `R_pp = P_p = SEFD + S_ast,p + Σ_s |A_ps|²`.
It asks for a diagonal approximation when RFI ≪ SEFD and the full covariance when RFI ≳ SEFD.

## Hypothesis: an exact, cheap sampler

Drawing from the full `n_bl × n_bl` covariance directly is expensive. The structure suggests
something better. Let `R = L Lᴴ` (Cholesky) and let `G` be a Hermitian matrix with independent entries:
real N(0,1) on the diagonal and circular complex with E|G_pq|² = 1 off it (a GUE matrix). Then

```
η = L G Lᴴ / √M
```

has exactly Kulkarni's second moments. The proof uses `E[G_ij G*_kl] = δ_ik δ_jl` and
`E[G_ij G_kl] = δ_il δ_jk`, which give `E[η_pq η*_rs] = (LLᴴ)_pr (LLᴴ)*_qs / M` and
`E[η_pq η_rs] = (LLᴴ)_ps (LLᴴ)_rq / M`. This is the central-limit term of the complex-Wishart
sample covariance, so it is the right object for M ≈ 4×10⁵. The cost is one Cholesky and two
`n_ant³` products per integration, about 0.5 MFLOP for 64 antennas. Covariance between baselines that share an antenna, and
unequal real and imaginary variances, come out automatically.

**Variation within an integration.** Fast RFI changes `R` within the 2 s. Splitting the
integration into `n_blocks` sub-blocks, drawing each with `M / n_blocks` samples and averaging gives
covariance `(1/(n_blocks M)) Σ_b R_b ⊗ R*_b`. That is exact for piecewise-constant `R`, and the diagonal
mode uses the same blocks (`σ²_pq = mean_b(P_p P_q)/M`, the protocol's `⟨P_p P_q⟩`).

## Implementation

[`enhancements/noise.py`](../src/tabsim_sbi/enhancements/noise.py) (`gue`, `self_noise`,
`self_noise_blocks`, `diag_noise_std`). In the simulator use
`SimConfig(noise="thermal" | "diag" | "full", noise_blocks=k, noise_frame="data" | "sky")`. The RFI
coherency is computed per sub-block when `noise_blocks > 1`. The total power fed to E3 includes the
SEFD, the apparent sky and the RFI autos, all in the data frame.

## How it serves SBI

* The simulator: E3 is the first effect to switch on for strong RFI. At 1.5 kJy apparent RFI the
  noise std rises by ~4.6× (`noise_inflation`), so i.i.d. noise is wrong by a factor of a few.
* The compression: E3 is tractable, so it can also enter the compression likelihood.
  `CompressionModel.signal_dependent_sigma(φ*)` gives the diagonal `σ_pq` at the fiducial point, and
  `run_protocol(..., signal_dependent_sigma=True)` uses it as the score's weights (protocol §3.2,
  "using the signal-dependent C* still matters for optimality"). Following the protocol, the
  quadratic variance term of the score is not used.

## Verified (`tests/test_noise.py`)

* GUE moments; Hermitian.
* Sampler vs Kulkarni's covariance **and** pseudo-covariance for five index patterns: shared
  antenna, disjoint, etc. (4×10⁵ draws, random rank-2 RFI + SEFD).
* Sampler vs a **brute-force voltage simulation** (M = 400 complex samples, explicit sample
  covariance): real and imaginary variances agree to 5%.
* Weak signal: reduces to tab-sim's per-component `SEFD/√(2M)`.
* Sub-blocks: average of the block covariances.
* In the simulator with strong RFI, the measured noise std on each baseline matches
  `σ·√(P_p P_q)/SEFD` to 10% in both `diag` and `full` modes.
* Realism statistic (V3): residual correlation between baselines sharing an antenna rises when E3
  is on (`tests/test_validation.py`).

## To be set / open

* `noise_frame` (see [01](01_baseline_audit.md) §2).
* `noise_blocks`: enough to resolve RFI amplitude changes within 2 s. The fringe phase itself does not
  enter `P_p`, so a few blocks usually suffice. Check with the toy at Paper II's strongest RFI.
* Kulkarni (1989) is cited from memory in the protocol. The formulas implemented here are the standard
  complex-Gaussian fourth-moment (Isserlis) results and are verified numerically above,
  independently of the citation.
