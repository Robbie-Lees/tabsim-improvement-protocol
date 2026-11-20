# Step 5 — Matrix-free nuisance hardening, all blocks at once

## What the protocol asks

`t̄_θ = t_θ − H_θη H_ηη⁻¹ t_η` (Alsing & Wandelt 2019), with the posterior information
`H = JᵀC⁻¹J + Σ_Π⁻¹` instead of the Fisher matrix, so that the global phase of each `A_ps` and the
over-parameterised target-field nuisances stay invertible. `H` must never be formed: products by JVP, `H_ηη⁻¹` by
conjugate gradients, `W = H_ηη⁻¹ H_ηθ` once per observation. Then infer block by block (e.g. one
antenna's gains), hardened against everything else including the other blocks, with cross-block
correlations "approximated from H⁻¹".

## Hypothesis: one set of solves for every block

Doing this literally needs, for each block *b*, `|b|` CG solves in the complement space `n = ¬b`,
with a different operator per block. The Schur-complement identity gives a better route:

```
[H⁻¹ t]_b = S_b⁻¹ (t_b − H_bn H_nn⁻¹ t_n),     [H⁻¹]_bb = S_b⁻¹,     S_b = H_bb − H_bn H_nn⁻¹ H_nb
⇒   t̄_b = ([H⁻¹]_bb)⁻¹ X_bᵀ t,     X = H⁻¹ E_θ.
```

`X` costs one full-space CG solve per interesting parameter, the same count as the literal route,
but the result serves **every block simultaneously**. It also yields `C = [H⁻¹]_θθ`, the
Laplace covariance including the **cross-block** terms the protocol needs for assembly, at no extra cost.

Summaries are reported in parameter units, `θ̂_b = θ*_b + X_bᵀ t`: one prior-regularised
Fisher-scoring step from the fiducial. This is an invertible linear map of `t̄_b` per block (the
same information), and better scaled for density estimation.

## Implementation

[`compression/hardening.py`](../src/tabsim_sbi/compression/hardening.py): `jax.linearize` once at φ\*,
`jax.linear_transpose` for the VJP, `H v = Jᵀ(Jv) + p v`, `jax.scipy.sparse.linalg.cg` mapped over the
unit vectors in batches (`lax.map(batch_size=...)`). It stores `X`, `cov`, the per-block `S_b`, and
`K = J X` for the linear projection ([05](05_score_compression.md)).

## What the prior does to hardening

With zero prior precision this is exactly Alsing & Wandelt: the expected summary has **zero**
first-order sensitivity to the nuisances. With the protocol's prior term, the sensitivity becomes
`∂θ̂/∂η = −[H⁻¹]_θη P_η` (tested). It is non-zero only along nuisance directions the prior, not the data,
constrains: the global `A_ps` phases and, on target fields, the directions where nuisances
outnumber data. Including the prior is therefore safe in the sense that matters: any residual
dependence is on directions the data cannot inform, and the NPE marginalises it with the rest of
the latent variables.

## Verified (`tests/test_compression.py`)

* Schur route = explicit per-block `t_b − H_bn H_nn⁻¹ t_n` on a random complex linear model, for blocks
  of size 1 and 2 (rtol 1e-7); `cov` = dense `H⁻¹` block.
* Linear projection `K ρ` = `X J ᵀρ`.
* Zero prior precision: `∂E[θ̂]/∂η = 0`, `∂E[θ̂]/∂θ = I`.
* With prior: `∂θ̂/∂η = −[H⁻¹]_θη`.

## Cost and scale

Per observation: `n_θ` CG solves, each `n_iter` × (1 JVP + 1 VJP). For Protocol A at 64 antennas
(~400 θ) with ~100 iterations, that is ~8×10⁴ JVP/VJP pairs. At Paper II's 10–100 ms per gradient, this is 15 min
to 2 h, once per observation. Preconditioning (Jacobi from the diagonal of `JᵀJ`, or the prior) and
block-CG over several right-hand sides are the obvious accelerations; neither is implemented yet.
For Protocol B1 (10⁵–10⁶ nuisances), only `n_θ` (≈ 3 × 64 phases) matters for the number of
solves; the nuisance count enters only through the cost of each JVP.
