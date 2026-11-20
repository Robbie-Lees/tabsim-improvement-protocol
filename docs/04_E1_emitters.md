# Step 3 — E1: unmodelled emitters and trajectory errors

## What the protocol asks

Create a mismatch between simulator and inference model. Simulate (i) emitters the model omits:
catalogue satellites left out of the model list, optionally stationary ground emitters, with
powers from a broad prior. Also simulate (ii) a position error of each modelled satellite relative to its TLE (~190 m for
one-day-old TLEs, Paper II). All of these are latent and drawn afresh in every simulation.

## Hypotheses

**Trajectory errors** belong in the satellite's own radial / along-track / cross-track (RTN) frame,
where TLE errors are known to be anisotropic. They are dominated by along-track timing error and grow
with TLE age. The model is `δ(t) = δ₀ + δ̇ (t − t_mid)`. The default prior σ = (40, 180, 50) m has a 3-D rms of
191 m, Paper II's figure. Perturbing the stored trajectory directly, rather than the TLE elements
(4-decimal-degree fields, ~46 m quantisation at GPS altitude), keeps tab-sim's SGP4 output as the
reference. The alternative of editing TLE lines and re-running tab-sim works only for TLE records,
not OMM records.

**Unmodelled satellites** come for free from tab-sim's own selection. Run tab-sim once with a
generous `max_n_sat` and `max_ang_sep`, then choose which satellites the model knows
(`ObservationGeometry.split_satellites(modelled)`). The rest form the `pool`. Each pool satellite
transmits with probability `unmodelled_prob`, with a power from a log-uniform prior.

**Ground emitters** need positions in tab-sim's inertial (GCRS) frame. Instead of re-implementing
precession and nutation, the Earth-fixed→inertial rotation is recovered at every fine time step
from the antenna positions tab-sim stores in both frames: Kabsch on the centred, planar,
well-conditioned layout (`geometry.rotation_from_antennas`). The residual is 12 µm on a real run. Emitters
are placed at random range (uniform in area), azimuth and mast height, with Earth curvature included.

## Implementation

[`enhancements/emitters.py`](../src/tabsim_sbi/enhancements/emitters.py); simulator switches
`tle_sigma_rtn`, `tle_drift_rtn`, `unmodelled_prob`, `unmodelled_log10_Pv`, `n_ground`,
`ground_log10_Pv`, `ground_range_km`. The perturbed trajectories feed the same distance, beam and fringe
computation as the modelled ones. The compression model keeps the TLE trajectories (variant A1),
so `A_ps` must absorb the error.

## How it serves SBI

E1 is where the compression model is *wrong by construction*. Score compression under a wrong
model loses optimality but not validity, provided the same compression is applied to data and
simulations (Alsing & Wandelt 2018/19), and that is exactly what this enhancement tests. The B0
stress test ([10](10_protocol_B.md)) measures how much an unmodified fit suffers.

A position error δ changes the fringe phase on baseline *b* by about `2π (b/λ)(δ_⊥/d)`. For
δ = 190 m, d = 2×10⁷ m and b = 8 km at λ = 24.4 cm (1.227 GHz), that is ~2 rad. It is a
per-antenna phase, linear in antenna position, so the per-antenna complex `A_ps` absorbs it while
it changes slowly. That is Paper II's argument for variant A1. What `A_ps` cannot absorb is change
faster than its correlation time, e.g. from drift or from fast changes in projection near
closest approach.

## Verified (`tests/test_emitters.py`)

RTN basis orthonormal with the along-track axis on the velocity; an along-track offset moves the
satellite by exactly that distance; prior scales and 3-D rms; ground emitters at the requested range
that co-rotate with the array; pool switches; the simulator with all E1 options is finite and
jit-able, and leaves the sky untouched.

## To be set / open

* Drift rates, and whether along-track error should grow quadratically with TLE age (drag).
* Emitter populations and power priors (protocol §8): which satellites in L2 at 1.227 GHz
  (GPS, GLONASS, Galileo, BeiDou), duty, and realistic ground-transmitter powers. V3 should set these.
* Variant A2 (orbit parameters inferred) is not implemented; see [09](09_protocol_A.md).
