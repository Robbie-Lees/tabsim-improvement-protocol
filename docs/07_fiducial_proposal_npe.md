# Step 6 — Fiducial, proposal, block-wise NPE and assembly

## Fiducial (protocol §3.4)

[`compression/fiducial.py`](../src/tabsim_sbi/compression/fiducial.py) minimises
`½‖y − m(φ)‖² + ½‖φ‖²` with a short Adam warm-up, then Levenberg–Marquardt Gauss–Newton steps whose
linear solves are matrix-free CG with the same `JᵀJ + (1+λ)I` operator the hardening uses. On these
least-squares problems GN converges in tens of iterations, where Paper II's first-order fits take
~2000 steps.

Two practical points the protocol does not mention:

* **The RFI term is bilinear in `A`** (`A_p A*_q`), so `A = 0` is a saddle point with zero
  gradient. `CompressionModel.init_params` starts the RFI parameters at small random values.
* **The χ² gate** (Paper II: χ² per real data point < 1.1) needs the right noise. Under E3 the noise
  exceeds thermal by `√(P_p P_q)/SEFD`. At ~30 Jy of RFI a perfect fit already has χ² ≈ 1.2–1.4
  against thermal σ, and at ~180 Jy χ² ≈ 2.2, so a thermal-σ gate rejects good fits. With
  `signal_dependent_sigma=True`, `run_protocol` re-evaluates the gate with the E3 σ at φ\*. This
  applies to tabascal too: a gate computed with the MS `SIGMA` column will reject fits on strong-RFI data
  that are in fact fine (B0, [10](10_protocol_B.md)).
* `MapFitter` compiles once per model and takes the data as an argument. Building new jitted
  closures per call recompiled for every simulation, and on CPU one such compile took ~2 h.

## Proposal and prior correction (protocol §3.4 item 3)

[`inference/proposal.py`](../src/tabsim_sbi/inference/proposal.py): θ is drawn from
`N(θ*, s² C)`, with `C` the Laplace covariance from the hardener and `s = 2` by default. An NPE trained on
proposal draws learns `q(θ|t) ∝ p(t|θ) p̃(θ)`. The posterior under the actual prior (standard
normal in standardised coordinates) follows by importance weighting `p(θ)/p̃(θ)`, done here by
sampling-importance-resampling per block, with the effective sample size reported. A low ESS means
the proposal was too narrow. Verified on a conjugate Gaussian (`tests/test_inference.py`).

**When the fit fails the gate, the proposal must not be centred on it.** The protocol argues that a
poor fiducial "costs optimality, not validity". That holds for the *compression*, but not for a
proposal built around the fit. With an unmodelled satellite, the toy fit is biased by ~11 Laplace σ.
A proposal `N(θ*, 4C)` then contains no simulations near the truth: in the first toy run the SIR
ESS collapsed to 1–5 and the posterior missed the truth by ~40σ. `run_protocol(proposal="auto")`
therefore uses the Laplace proposal only when the gate passes and falls back to the prior
otherwise. `"laplace"` and `"prior"` force either choice. A defensive mixture of the two, or a
sequential round from the first posterior, would recover efficiency, but neither is implemented.

## Block-wise NPE (protocol §3.5)

[`inference/mdn.py`](../src/tabsim_sbi/inference/mdn.py): a Gaussian mixture density network in pure
JAX, trained by maximum likelihood with early stopping. Per-block problems are 2–6 dimensional, the
regime where Alsing & Wandelt report convergence in 10²–10³ simulations, and MDNs are what pydelfi
uses there. No torch dependency. Any estimator exposing `sample(key, s, n)` and `log_prob(θ, s)`
(e.g. from `sbi` or `ltu-ili`) can replace it. Verified to recover a conjugate Gaussian posterior.

Each block's NPE is conditioned on that block's hardened summary only; all θ are drawn jointly from
the proposal, so every NPE marginalises over the other blocks.

## Assembly

[`protocols.assemble_joint`](../src/tabsim_sbi/protocols.py): per-block posterior covariances `D_b`
on the diagonal, and cross-block terms from the Laplace correlation whitened per block,
`Σ_bc = D_b^½ C_bb^-½ C_bc C_cc^-½ D_c^½`. This is exact when every block posterior equals its
Laplace approximation and approximate otherwise. The joint posterior then goes to the GP gain
interpolation for the target scans (protocol Step A4), which is outside this repository.

## The runner

[`protocols.run_protocol(key, spec, d_obs, n_sims=..., nu=..., signal_dependent_sigma=...)`](../src/tabsim_sbi/protocols.py)
does fit → frozen compressor and hardener → proposal → vmapped simulate+compress in batches →
per-block NPE + SIR → assembly. `validate_blocks` runs V1 on held-out proposal simulations.
