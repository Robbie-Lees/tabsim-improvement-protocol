# Step 7 — E2, E4, E5, E6, E8 (and E7, deferred)

Each is a set of pure JAX functions in [`src/tabsim_sbi/enhancements/`](../src/tabsim_sbi/enhancements/),
switched on through `SimConfig`. All draws are latent and fresh per simulation. All compose in one
jit/vmap-able call (`tests/test_enhancements.py::test_everything_on_is_finite_jittable_and_vmappable`).
Values marked *to be set* should come from V3 on real data.

## E2 — Realistic RFI signal ([`rfi_signal.py`](../src/tabsim_sbi/enhancements/rfi_signal.py))

| Part | Model | Switches |
|---|---|---|
| (i) power variation | unit-mean log-normal Ornstein–Uhlenbeck (exact AR(1) on the fine grid) × optional on/off telegraph process with stationary duty cycle | `power_tau`, `power_sigma_log`, `duty_cycle`, `duty_dwell` |
| (ii) beam differences | per-antenna pointing offset (two tangent-plane components) and dish-size factor | `pointing_sigma_arcmin`, `dish_scale_sigma` |
| (iii) complex beam | phase `κ_p (θ/(λ/D))²` growing off-axis | `beam_phase_sigma` |

The beam is built on tab-sim's float64 Cephes `j1`, because tab-sim's `airy_beam` takes one dish
diameter. With zero perturbations it equals tab-sim's beam to 1e-10. Hypothesis for SBI: the Paper II model
absorbs anything slower than its 24 s RFI correlation time, so the informative settings are
`power_tau` and `duty_dwell` *below* 24 s, and fast sidelobe sweeps. For a direct tab-sim run,
(i) can also be passed as a `Pv` spectrogram to `addTLESatelliteRFI`.

## E4 — Unmodelled sky ([`sky.py`](../src/tabsim_sbi/enhancements/sky.py))

Poisson number of sources with power-law counts `dN/dS ∝ S^-α` on `[S_min, S_max]`, uniform in area in
an annulus `[r_min, r_max]`. A fraction are circular Gaussians, whose visibility taper depends on the baseline
and is accumulated per source with `lax.scan`. A fixed `n_max` with a mask keeps shapes static. Near the
calibrator this is Protocol A's confusion background; in the sidelobes it is the off-axis case Paper
II's fringe-rate prior assumes away; in B2 it is the "rest of the sky" latent.

## E5 — Gains beyond the model, non-closing errors ([`gain_errors.py`](../src/tabsim_sbi/enhancements/gain_errors.py))

tab-sim 0.1.0 already draws gains as a GP ([01](01_baseline_audit.md)). What SBI needs is the part the
compression model *cannot* represent: the simulator multiplies the θ-driven gains by
`exp(δa + iδφ)`, an SE-GP residual whose correlation time (`gain_resid_tau`) is shorter than
the inducing-point spacing. The reference antenna's phase is kept at zero. Non-closing errors
`V_pq → V_pq(1+ε_pq)`, constant over the observation and Hermitian, act on the signal (not the noise).

## E6 — Receiver compression ([`receiver.py`](../src/tabsim_sbi/enhancements/receiver.py))

Voltage factor `c_p = sqrt((1 + SEFD/P_sat)/(1 + P_p/P_sat))` from each antenna's integration-mean
total power, normalised to 1 without signal so that only the RFI-dependent part is new, applied to
signal and noise. Instantaneous compression of fast RFI is understated. A fine-sample version would
apply `c_p(t_k)` to the field before the Gram matrix, which is straightforward if V3 calls for it.

## E8 — Heavy-tailed catch-all ([`heavy_tail.py`](../src/tabsim_sbi/enhancements/heavy_tail.py))

Circular complex Student-t as a scale mixture `η·sqrt(ν/u)`, `u ~ χ²_ν` per visibility, with `ν` fixed
or log-uniform per simulation. As the protocol warns, it fixes the marginal tails but not the
structure; use it only if V3 leaves unexplained tails after E1–E6. The matching compression is the
Student-t score ([05](05_score_compression.md)).

## E7 — Ionosphere and troposphere (deferred, as in the protocol)

Design, if V3 calls for it: a frozen-flow Kolmogorov phase screen at height `h` (FFT-generated once
per simulation, advected with velocity `v`). The phase for antenna *p* towards source *s* is the screen at the
pierce point `x_p + h·tan(z_s)·ŝ_⊥`, which differs between the calibrator and each satellite: a
direction-dependent term the direction-independent gains cannot absorb. In this code it would multiply
the RFI field `B_ps` and the sky's `e_ps` before the Gram products. Expected to be small at 1.2 GHz on
≤8 km baselines.
