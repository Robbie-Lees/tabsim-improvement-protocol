# Validation (protocol §6)

| | What | Where | State |
|---|---|---|---|
| V0 | Gaussian limit reproduces existing results | simulator parity with tab-sim ([02](02_forward_model.md)); Protocol A with `SimConfig()` | Simulator side done. Comparison with **Paper I's HMC posteriors** needs those chains and is not done. For B, run HMC (e.g. NumPyro NUTS on `CompressionModel`) on ~8 antennas, a few minutes |
| V1 | SBC and coverage on held-out simulations | [`validation/calibration.py`](../src/tabsim_sbi/validation/calibration.py), `protocols.validate_blocks` | Implemented; used in the Protocol A example ([09](09_protocol_A.md)) |
| V2 | Compression loss: Gaussian vs Student-t score vs learned compressor | `run_protocol(nu=...)` gives the first two | Learned compressor not implemented. Compare posterior widths on a reduced array |
| V3 | Realism vs real MeerKAT GNSS-band residuals | [`validation/realism.py`](../src/tabsim_sbi/validation/realism.py) | Statistics implemented and tested on simulations; needs real data |
| V4 | Posterior predictive checks | simulator + V3 statistics with posterior draws | Straightforward once real data are in |

## V1 in practice

The per-block NPEs are amortised over the proposal region, so fresh proposal draws test them
directly: SBC rank histograms (χ² uniformity p-values per dimension) and expected-coverage curves
(Hermans et al. 2022, HPD credibility level of the truth). The SIR prior correction is exact given
a calibrated `q`, so V1 is run on `q`. After each enhancement is switched on (protocol build order
step 3), re-run `validate_blocks`.

## V3 statistics

All take `(n_time, n_bl)` arrays and baseline metadata, so they run identically on a simulation
and on a measurement set read with dask-ms:

* amplitude histograms per baseline-length bin;
* closure phases and log closure amplitudes. Gain-invariant (tested to 1e-9), so they isolate
  non-closing structure: RFI, E5's baseline errors;
* temporal autocorrelation;
* mean |correlation| of residuals between baselines sharing an antenna vs not. E3 predicts a clear
  excess (tested);
* `compare(sim, real)`: KS and Wasserstein-1 distances.

The output of V3 should fix the "to be set" priors (E1 powers and populations, E2 correlation times,
E5 residual, E6 `P_sat`) and decide E6–E8.
