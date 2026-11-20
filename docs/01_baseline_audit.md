# Step 0 — Baseline audit of tab-sim 0.1.0 (and the tabascal likelihood)

The draft protocol describes tab-sim "as used in Paper II". Upstream has moved since, so before any
enhancement was designed the code was read at fixed commits:

* tab-sim: [`chrisfinlay/tab-sim@e569fb7`](https://github.com/chrisfinlay/tab-sim/tree/e569fb77f94af2b44cda5bafae3991a62a39446b) (Release 0.1.0)
* tabascal: [`epfl-radio-astro/tabascal@4d06faa`](https://github.com/epfl-radio-astro/tabascal/tree/4d06faa) (the old
  `chrisfinlay/tabascal` is archived with a notice pointing there)

Links below are to those commits.

## 1. The noise convention (protocol §1, "to check") — resolved

| Where | What it does |
|---|---|
| tab-sim [`SEFD_to_noise_std`](https://github.com/chrisfinlay/tab-sim/blob/e569fb77f94af2b44cda5bafae3991a62a39446b/tabsim/jax/interferometry.py#L716) | `noise_std = SEFD / sqrt(2 Δν Δt)`, documented as *per real/imaginary component* |
| tab-sim [`add_noise`](https://github.com/chrisfinlay/tab-sim/blob/e569fb77f94af2b44cda5bafae3991a62a39446b/tabsim/jax/interferometry.py#L690) | each component gets `noise_std` |
| tab-sim [`write.py` L626](https://github.com/chrisfinlay/tab-sim/blob/e569fb77f94af2b44cda5bafae3991a62a39446b/tabsim/write.py#L626) | MS `SIGMA` column = `noise_std` |
| tabascal [`components/likelihood.py`](https://github.com/epfl-radio-astro/tabascal/blob/4d06faa/tabascal/components/likelihood.py) | `Normal([Re, Im], noise)` with `noise` read from `SIGMA` |

With SEFD = 420 Jy, Δν = 209 kHz, Δt = 2 s: per-component σ = 0.459 Jy, complex std
σ√2 = 0.650 Jy. That is Paper I's σ_n = 0.65 Jy as the *complex* standard deviation. **The code
pipeline is self-consistent.** tab-sim's data and tabascal's likelihood use the same per-component
σ. The √2 discrepancy exists only in how Paper II *writes* the likelihood. This repository uses
per-component σ everywhere (`ObservationGeometry.noise_std`, `WhitenedModel`), and E3's complex
variance is `2σ² = SEFD²/(Δν Δt)`, consistent with the radiometer equation for one polarisation
product (`tests/test_forward.py::test_noise_std_is_tabsim_convention`).

## 2. Where noise and gains enter

[`Observation.calculate_vis`](https://github.com/chrisfinlay/tab-sim/blob/e569fb77f94af2b44cda5bafae3991a62a39446b/tabsim/dask/observation.py#L1050):
`vis_obs = G (vis_ast + vis_rfi) Gᴴ + noise`. **Noise is added after the gains, with a fixed σ.**
In other words, tab-sim refers the SEFD to the correlator output, not to the sky. The protocol asks to "fix whether noise is
added before or after the gains, consistently with the definition of the SEFD". This repository
offers both (`SimConfig.noise_frame`):

* `"data"` (default, matches tab-sim): self-noise from `G R_sky Gᴴ + SEFD·I`;
* `"sky"`: self-noise from `R_sky + SEFD·I`, then scaled by `g_p g_q*`, so that thermal noise
  scales with |g|².

With gains near unity the two differ by O(gain amplitude error). The choice is "to be set" against how
MeerKAT's SEFD is quoted.

## 3. What upstream already provides (and the protocol assumed missing)

* **Stochastic gains.** `addGains` now calls
  [`generate_fourier_gains`](https://github.com/chrisfinlay/tab-sim/blob/e569fb77f94af2b44cda5bafae3991a62a39446b/tabsim/dask/interferometry.py#L723):
  1000 random Fourier features with Gaussian-distributed frequencies, i.e. an SE-kernel GP with
  correlation times `Gt_corr_amp/phase` (hours). Linear drifts survive only in the `generate_gains`
  functions, which `addGains` no longer calls. Part of E5 is therefore already done; what SBI still needs is a gain component
  the *compression model* cannot represent ([08](08_further_enhancements.md)).
* **Time-varying intrinsic power.**
  [`addTLESatelliteRFI`](https://github.com/chrisfinlay/tab-sim/blob/e569fb77f94af2b44cda5bafae3991a62a39446b/tabsim/dask/observation.py#L855)
  accepts `Pv` of shape `(n_src, n_time_fine, n_freq)`. E2(i) can be fed to tab-sim directly, without changing it.
* **Fast RFI kernel.** `ri_kernels` (optional) with a pure-JAX fallback; float64 phases. This repository does
  not use `ri_kernels` because it needs gradients through the RFI model; its own kernel is a Gram
  matrix (see [02](02_forward_model.md)).

## 4. What is missing, and the hook each enhancement uses

| Gap | Hook (no change to tab-sim) |
|---|---|
| Identical real Airy beam; [`airy_beam`](https://github.com/chrisfinlay/tab-sim/blob/e569fb77f94af2b44cda5bafae3991a62a39446b/tabsim/jax/beam.py) takes a scalar dish diameter | Own per-antenna complex beam on tab-sim's float64 `j1` (E2) |
| True trajectory = TLE trajectory | Perturb the stored `rfi_tle_sat_xyz` in the RTN frame (E1) |
| Every visible satellite (up to `max_n_sat`, [config.py](https://github.com/chrisfinlay/tab-sim/blob/e569fb77f94af2b44cda5bafae3991a62a39446b/tabsim/config.py#L730)) is modelled | Simulate with a large `max_n_sat`; `ObservationGeometry.split_satellites` moves the unmodelled ones to a pool (E1) |
| Ground emitters need ITRF→GCRS | Recover tab-sim's own rotation from `ants_itrf`/`ants_xyz` per time step by Kabsch (12 µm residual on a real run) |
| i.i.d. noise | E3/E8 on the coherency matrix |
| Linear receiver | E6 on total power |

## 5. tabascal today vs. the protocol's compression model

The protocol's compression model ("gains on inducing points, complex A_ps on inducing points with
an SE prior") is Paper II's. Current tabascal is a component-based NumPyro model: RFI via an
FFT-domain GP (`rfi_signal.BaseGPRFI`, power-law spectrum), gains as a gauge-fixed constant per
antenna (`ConstGains`, zero-sum log-amplitudes and a pinned reference phase). This repository ships
a self-contained Paper II style model (`compression/model.py`), so the machinery can be developed
and tested without tabascal. The machinery only needs a differentiable `mu(φ)` with an identity
prior, so tabascal's `forward(params, state, constants)` (flattened with `ravel_pytree`) can be
substituted when it is installed. That adapter is not written yet.
