# Overview: from the protocol to the code

This repository implements *Protocols for enhancing tab-sim with non-Gaussian effects and
compressing its output for simulation-based inference* (draft note v2, 20 Nov 2026), on top of
[tab-sim](https://github.com/chrisfinlay/tab-sim) 0.1.0. tab-sim is imported and never modified.

## The design hypothesis

SBI for one observation needs 10³–10⁴ simulations **of that observation**. Re-running tab-sim's
full pipeline (dask graph, SGP4 propagation, zarr/MS writing) for each one would waste almost all of
the time on work that does not change. What is fixed per observation — times, antenna positions,
pointing, uvw, the TLE trajectories of the modelled satellites, the known sky — is extracted once
into an `ObservationGeometry`. A simulation is then a pure JAX function of a PRNG key and the
parameters. It can be jit-compiled, vmapped and run on a GPU, and it adds the protocol's enhancements as stochastic
layers:

```
tab-sim run (once) ──► ObservationGeometry ──► Simulator.simulate(key, θ)    ×10³–10⁴, vmapped
                         (geometry.py)            ├─ E1 trajectories, unmodelled emitters
                                                  ├─ E2 power process, per-antenna complex beams
                                                  ├─ RFI + sky coherency (forward.py)
                                                  ├─ gains from θ (gains.py) × E5 residual
                                                  ├─ E3 self-noise / E8 heavy tails
                                                  └─ E6 compression
```

With every enhancement off, the simulator reproduces tab-sim. It agrees with tab-sim's own JAX kernels to 1e-9
relative, and it reproduces the stored `vis_rfi` of a real 32-antenna, 3-satellite tab-sim run to
3e-9 ([02](02_forward_model.md)). That is validation V0 at the simulator level.

Internally the forward model works on full **coherency matrices** `R[t, f, p, q]` rather than on
baseline lists. This makes the RFI a Gram matrix (one batched GEMM per integration), puts the per-antenna total power on the
diagonal, and lets E3 draw self-noise with the right
correlations at the cost of two `n_ant³` products per integration.

The compression side follows the protocol: score of an approximate likelihood at a frozen MAP
fiducial, nuisance hardening with the posterior information `H = JᵀJ + I`, block-wise NPE. Two
implementation results simplify it:

* **The Gaussian score is linear in the data** once φ\* is frozen, so compressing a simulation is
  one fixed `(n_θ × n_data)` matrix–vector product. It is not "about as expensive as the simulation" (protocol §3.5);
  it is negligible.
* **One set of CG solves hardens every block.** With `X = H⁻¹E_θ` (one CG solve per interesting
  parameter, once per observation), the hardened summary of block *b* against *everything else* is
  `t̄_b = ([H⁻¹]_bb)⁻¹ X_bᵀ t` (Schur complement identity). The same solves give the full Laplace
  covariance, including the cross-block terms the protocol needs to assemble per-block posteriors
  ([06](06_hardening.md)).

## Map of the protocol to this repository

| Protocol | What | Code | Doc | Status |
|---|---|---|---|---|
| §1 noise convention | √2 between Paper I and Paper II | `geometry.noise_std` | [01](01_baseline_audit.md) | **Resolved in code**: tab-sim and tabascal agree; only the papers' notation differs |
| baseline | tab-sim model in the Gaussian limit | `geometry.py`, `forward.py`, `simulator.py` | [02](02_forward_model.md) | Verified against tab-sim 0.1.0 and a real tab-sim zarr |
| E3 | signal-dependent self-noise | `enhancements/noise.py` | [03](03_E3_self_noise.md) | Exact sampler; verified against Kulkarni moments and brute-force voltages |
| E1 | unmodelled emitters, TLE errors | `enhancements/emitters.py` | [04](04_E1_emitters.md) | Implemented and tested; priors to be set |
| §3.2 | score compression (Gaussian, Student-t) | `compression/score.py`, `compression/model.py` | [05](05_score_compression.md) | Verified |
| §3.3 | matrix-free hardening | `compression/hardening.py` | [06](06_hardening.md) | Verified (Schur = per-block AW19; nuisance insensitivity) |
| §3.4–3.5 | fiducial, proposal, block NPE | `compression/fiducial.py`, `inference/`, `protocols.py` | [07](07_fiducial_proposal_npe.md) | Runs end to end; prior correction verified |
| E2, E4, E5, E6, E8 (E7 deferred) | further effects | `enhancements/*.py` | [08](08_further_enhancements.md) | Implemented and tested; priors to be set |
| §4 Protocol A | calibrator, A1 | `protocols.protocol_a` | [09](09_protocol_A.md) | Toy: calibrated SBI posterior where the Laplace errors fail (unmodelled satellite); held-out V1 roughly calibrated |
| §5 B0, B1, B2 | target fields | `stress_test.py`, `protocols.protocol_b1/b2` | [10](10_protocol_B.md) | B0 ranks the effects (unmodelled emitters and fast RFI variation are the damaging ones); B1/B2 build and compress, untested at scale |
| §6 V0–V4 | validation | `validation/` | [11](11_validation.md) | V0, V1, V3 tools in place; V3 needs real data |
| §8 open points | | | [12](12_open_points.md) | Partly answered |

## Build order followed

The protocol's build order is E3 + E1 → B0 → machinery (V0 on Protocol A) → A1 with E1–E5 → B1 →
B2, E6–E8. The commits follow the same order. B0's harness is written after the machinery because
it reuses the compression model's MAP fit as a stand-in for tabascal (see [10](10_protocol_B.md) for
how to plug tabascal itself in).

## Running

```bash
pip install -e ".[zarr,test]"      # needs tabsim>=0.1.0 (and its JAX)
pytest -m "not slow"               # ~1 min on CPU
pytest                             # + end-to-end Protocol A (~1 min more)
TABSIM_SBI_TEST_ZARR=/path/to/obs.zarr pytest tests/test_zarr_parity.py
python examples/protocol_a_toy.py  # Protocol A on a synthetic calibrator scan
python examples/b0_stress_test.py  # B0 table
```
