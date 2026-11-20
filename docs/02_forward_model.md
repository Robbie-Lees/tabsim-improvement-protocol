# Step 1 — Geometry cache and the Gaussian-limit forward model

## Hypothesis

Everything the protocol adds is either (a) a change to *what is simulated on fixed geometry* (powers,
beams, gains, noise, extra sources) or (b) a perturbation of that geometry (trajectory errors,
ground emitters). So split tab-sim's job in two: run it once per observation for the geometry, and
re-implement only the cheap, stochastic part as a pure JAX function. The re-implementation is only
trustworthy if, with every enhancement off, it **is** tab-sim. That is tested first.

## Implementation

* [`geometry.py`](../src/tabsim_sbi/geometry.py): `ObservationGeometry`, a JAX pytree with the fine
  sample grid (`n_tf = n_time × n_int`), `ants_xyz`, `ants_w`, `ants_uvw` at centroids, pointing,
  TLE trajectories `sat_xyz` and nominal powers `sat_Pv`, an unmodelled `pool`, the known sky, and
  the Earth-fixed→inertial rotation `rot`.
  * `from_tabsim_zarr(path, modelled=..., time_range=...)` reads a tab-sim output store (read-only).
    `Pv` is recovered from the stored apparent amplitude `rfi_tle_sat_A = sqrt(Pv/(4πd²)·1e26)·airy`.
    This is exact for constant power and needs no access to tab-sim's spectral-model files.
  * `synthetic_geometry(...)`: spherical rotating Earth, disc array, zenith pointing, circular orbits
    passing given offsets from the phase centre. Used by the tests and examples.
* [`forward.py`](../src/tabsim_sbi/forward.py): differentiable kernels on coherency matrices.
  * RFI: field `B_ps(t_k) = A_ps(t_k) exp(-2πi f/c (d_ps + w_p))`, coherency
    `R_pq = (1/n_int) Σ_{k,s} B_kps B*_kqs`. This is an einsum XLA lowers to a batched matrix product.
    The autos `R_pp = Σ_s ⟨|A_ps|²⟩` come out with it.
  * Sky: `R_pq = Σ_s I_s^app e_ps e*_qs (·taper)` with `e_ps = exp(2πi f/c uvw_p·(lmn_s − s0))`.
  * `apply_gains`, `to_baselines` (tab-sim's `(n_time, n_bl, n_freq)` layout), `autos`.
* [`gains.py`](../src/tabsim_sbi/gains.py): `GainParameterisation`, shared by the simulator and the
  compression model, so θ means the same thing on both sides (log-amplitude and phase on `n_g`
  inducing points per antenna, SE-GP conditional-mean interpolation, standardised `z ~ N(0, I)`,
  reference-antenna phase fixed to 0 as in tab-sim).
* [`simulator.py`](../src/tabsim_sbi/simulator.py): `Simulator(geom, gains, SimConfig()).simulate(key, gain_z)`.

## How it serves SBI

A simulation is `jit(vmap(simulate))(keys, θ)`. On the 6-antenna toy, 3000 simulations plus
compression take well under a minute on a laptop CPU. The same code runs on a GPU without changes. The
geometry for a real observation comes from one ordinary tab-sim run, so the SBI simulations inherit
tab-sim's SGP4 trajectories, GCRS antenna positions and phase-tracking conventions exactly.

## Verified

| Test | Result |
|---|---|
| RFI vs `tabsim.jax.interferometry.rfi_vis_beam` (synthetic) | agree to rtol 1e-9 |
| Sky vs `tabsim.jax.interferometry.astro_vis` + host Airy beam | rtol 1e-9 |
| Gains vs `tabsim.jax.interferometry.apply_gains` | rtol 1e-10 |
| `times_fine` vs `int_sample_times`; `noise_std` vs `SEFD_to_noise_std` | exact |
| **Real tab-sim run** (MeerKAT 32 antennas, 3 GPS satellites, 1025 fine samples / 2 s, integrations 100–103): recomputed `vis_rfi` vs the stored one | max error 3.2e-9 of max\|V\| (≈20 Jy); Pv recovered as 9.02e-6 W/Hz for all three satellites |
| Kabsch rotation on the same run: `rot · ants_itrf` vs `ants_xyz` | 12 µm |

The real-run check is `tests/test_zarr_parity.py`, enabled with `TABSIM_SBI_TEST_ZARR=/path/to/obs.zarr`.

## Limits

* Memory: the fine-sample field is `n_src × n_tf × n_ant` complex. For 64 antennas, 15 minutes and
  1025 samples per 2 s, that is about 1 GB per satellite, too large to vmap many simulations at once.
  Use `time_range` to simulate scans in pieces, or thin the fine grid where fringe rates allow
  (the compression model already supports `n_int_model < n_int`). Chunking the einsum over time with
  `lax.map` would remove the limit and is not done yet.
* One polarisation product, as in the protocol.
