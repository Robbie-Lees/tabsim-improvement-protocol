"""The enhanced simulator: fixed geometry + stochastic layers E1-E8.

``Simulator.simulate(key, gain_z, ast_I=None)`` is a pure function: jit it,
vmap it over keys and parameters, run it on a GPU. With every enhancement off
(the default :class:`SimConfig`) it reproduces tab-sim's model,

    V_pq = g_p (V_ast + V_rfi)_pq conj(g_q) + eta_pq,   eta i.i.d., std SEFD / sqrt(2 M),

which is validation V0 at the simulator level (``tests/test_forward.py``).

Order of operations (data frame):

1. RFI sources: modelled satellites (E1 trajectory errors), unmodelled pool
   satellites and ground emitters (E1), each with a power process (E2) and
   per-antenna complex beams (E2); fields ``A_ps(t) exp(i phi_ps(t))``.
2. RFI coherency per integration (and per sub-block if E3 needs it).
3. Sky coherency: known sky (or ``ast_I`` override, Protocol B2) + faint sky (E4).
4. Gains from theta (shared parameterisation) times a latent residual (E5).
5. Non-closing errors on the signal (E5).
6. Noise: thermal, or self-noise from the data-frame coherency (E3),
   optionally heavy-tailed (E8).
7. Receiver compression on signal + noise (E6).

The parameters theta enter only through ``gain_z`` (and ``ast_I`` for B2).
Everything else -- RFI powers, trajectory errors, unmodelled emitters, beams,
sky, gain residuals, noise -- is latent and drawn afresh in every call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from tabsim_sbi import forward as fw
from tabsim_sbi.enhancements import emitters as e1
from tabsim_sbi.enhancements import gain_errors as e5
from tabsim_sbi.enhancements import heavy_tail as e8
from tabsim_sbi.enhancements import noise as e3
from tabsim_sbi.enhancements import receiver as e6
from tabsim_sbi.enhancements import rfi_signal as e2
from tabsim_sbi.enhancements.sky import FaintSkyConfig, sample_faint_sky
from tabsim_sbi.gains import GainParameterisation
from tabsim_sbi.geometry import ObservationGeometry

ARCMIN = jnp.pi / (180.0 * 60.0)


@dataclass(frozen=True)
class SimConfig:
    """Switches and priors for each enhancement. Defaults = tab-sim (V0)."""

    # Base RFI power prior: Pv = nominal * 10^(Pv_log10_sigma * N(0,1))
    Pv_log10_sigma: float = 0.0
    # E1 trajectory errors (m, m/s) in radial / along-track / cross-track
    tle_sigma_rtn: Optional[Tuple[float, float, float]] = None
    tle_drift_rtn: Optional[Tuple[float, float, float]] = None
    # E1 unmodelled pool satellites
    unmodelled_prob: float = 0.0
    unmodelled_log10_Pv: Optional[Tuple[float, float]] = None
    # E1 ground emitters
    n_ground: int = 0
    ground_log10_Pv: Tuple[float, float] = (-11.0, -9.0)
    ground_range_km: Tuple[float, float] = (10.0, 80.0)
    # E2 power process and beams
    power_tau: Optional[float] = None  # s
    power_sigma_log: float = 0.0
    duty_cycle: Optional[float] = None
    duty_dwell: float = 1.0  # s
    pointing_sigma_arcmin: float = 0.0
    dish_scale_sigma: float = 0.0
    beam_phase_sigma: float = 0.0  # rad at theta = lambda / D
    # E3 noise: "none" | "thermal" | "diag" | "full"
    noise: str = "thermal"
    noise_blocks: int = 1
    noise_frame: str = "data"  # "data": SEFD at the correlator (tab-sim); "sky": noise scaled by gains
    # E4 unmodelled sky
    faint_sky: Optional[FaintSkyConfig] = None
    # E5 gain residual beyond the model and non-closing errors
    gain_resid_sigma_amp: float = 0.0
    gain_resid_sigma_phase: float = 0.0
    gain_resid_tau: float = 60.0
    nonclosing_sigma: float = 0.0
    # E6 receiver compression, saturation power in Jy
    P_sat: Optional[float] = None
    # E8 Student-t; fixed nu, or log10 range to draw it from
    student_nu: Optional[float] = None
    student_log10_nu: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class Simulator:
    geom: ObservationGeometry
    gains: GainParameterisation
    config: SimConfig = field(default_factory=SimConfig)

    # ------------------------------------------------------------------
    def rfi_sources(self, key):
        """True trajectories (n_src, n_tf, 3) and power time series (n_src, n_tf, n_freq)."""
        g, c = self.geom, self.config
        k_tle, k_pv, k_pool, k_gpos, k_gpv, k_ou, k_tel = jax.random.split(key, 7)
        tf = g.times_fine

        sat_xyz = g.sat_xyz
        if c.tle_sigma_rtn is not None and g.n_sat:
            offsets = e1.sample_tle_offsets(k_tle, g.n_sat, tf, c.tle_sigma_rtn, c.tle_drift_rtn)
            sat_xyz = e1.perturb_trajectories(sat_xyz, offsets, g.dt_fine)
        Pv = g.sat_Pv * 10.0 ** (c.Pv_log10_sigma * jax.random.normal(k_pv, (g.n_sat, 1)))
        xyz, Pvs = [sat_xyz], [Pv]

        if g.n_pool and c.unmodelled_prob > 0:
            xyz.append(g.pool_xyz)
            Pvs.append(e1.sample_unmodelled(k_pool, g.n_pool, g.n_freq, c.unmodelled_prob, g.pool_Pv, c.unmodelled_log10_Pv))

        if c.n_ground:
            itrf = e1.sample_ground_emitters(k_gpos, c.n_ground, g.ants_itrf, c.ground_range_km)
            xyz.append(e1.ground_to_inertial(itrf, g.rot))
            Pvs.append(jnp.broadcast_to(e1.sample_log_uniform(k_gpv, (c.n_ground, 1), c.ground_log10_Pv), (c.n_ground, g.n_freq)))

        xyz = jnp.concatenate(xyz, axis=0)
        Pv = jnp.concatenate(Pvs, axis=0)
        n_src = xyz.shape[0]
        mult = jnp.ones((n_src, g.n_time_fine))
        if c.power_tau is not None and c.power_sigma_log > 0:
            mult = mult * e2.ou_log_power(k_ou, n_src, g.n_time_fine, g.dt_fine, c.power_tau, c.power_sigma_log)
        if c.duty_cycle is not None:
            mult = mult * e2.telegraph(k_tel, n_src, g.n_time_fine, g.dt_fine, c.duty_cycle, c.duty_dwell)
        return xyz, Pv[:, None, :] * mult[..., None]

    def rfi_field(self, key, xyz, Pv_t):
        """Per-antenna RFI field (n_src, n_tf, n_ant, n_freq)."""
        g, c = self.geom, self.config
        u, d = fw.source_vectors(xyz, g.ants_xyz)
        perturbed = c.pointing_sigma_arcmin > 0 or c.dish_scale_sigma > 0 or c.beam_phase_sigma > 0
        if perturbed:
            offs, scale, kappa = e2.sample_beam_perturbations(
                key, g.n_ant, c.pointing_sigma_arcmin * ARCMIN, c.dish_scale_sigma, c.beam_phase_sigma
            )
            beam = e2.antenna_beams(u, e2.antenna_pointings(g.pointing, offs), g.freqs, g.dish_d, scale, kappa)
        else:
            beam = fw.airy_voltage(fw.angle_to(u, g.pointing), g.freqs, g.dish_d)
        A = fw.rfi_amplitude(Pv_t, d, beam)
        return A * fw.fringe_phasor(d + g.ants_w[None], g.freqs)

    def sky(self, key, ast_I=None):
        g = self.geom
        I = g.ast_I if ast_I is None else ast_I
        lmn = g.ast_lmn
        sigma = jnp.zeros(lmn.shape[0])
        if self.config.faint_sky is not None:
            If, lmnf, sf = sample_faint_sky(key, self.config.faint_sky, g.n_freq)
            I, lmn, sigma = jnp.concatenate([I, If]), jnp.concatenate([lmn, lmnf]), jnp.concatenate([sigma, sf])
        I_app = fw.apparent_flux(I, lmn, g.freqs, g.dish_d)
        ext = self.config.faint_sky is not None and self.config.faint_sky.extended_frac > 0
        return fw.sky_coherency(I_app, lmn, g.ants_uvw, g.freqs, sigma if ext else None)

    # ------------------------------------------------------------------
    def simulate(self, key, gain_z, ast_I=None):
        """One simulation. Returns ``(vis_obs (n_time, n_bl, n_freq), aux)``."""
        g, c = self.geom, self.config
        keys = jax.random.split(key, 10)

        n_blk = c.noise_blocks if c.noise in ("diag", "full") else 1
        if g.n_sat or g.n_pool or c.n_ground:
            xyz, Pv_t = self.rfi_sources(keys[0])
            R_rfi = fw.rfi_coherency(self.rfi_field(keys[1], xyz, Pv_t), g.n_int, n_blk)
        else:
            R_rfi = jnp.zeros((g.n_time, n_blk, g.n_freq, g.n_ant, g.n_ant), dtype=jnp.complex128)
        R_sky = self.sky(keys[2], ast_I)  # (t, f, p, q)

        gains = self.gains.gains(gain_z, g.n_freq)
        if c.gain_resid_sigma_amp > 0 or c.gain_resid_sigma_phase > 0:
            res = e5.gain_residual(keys[3], g.times, g.n_ant, c.gain_resid_sigma_amp,
                                   c.gain_resid_sigma_phase, c.gain_resid_tau, self.gains.ref)
            gains = gains * res[..., None]

        R_true_blk = R_sky[:, None] + R_rfi  # (t, b, f, p, q), sky frame
        R_sig_blk = fw.apply_gains(R_true_blk, gains)  # data frame
        R_sig = R_sig_blk.mean(axis=1)
        if c.nonclosing_sigma > 0:
            R_sig = R_sig * e5.nonclosing_errors(keys[4], g.n_ant, c.nonclosing_sigma)
        signal = fw.to_baselines(R_sig, g.a1, g.a2)

        eta = self._noise(keys[5], R_true_blk, R_sig_blk, gains)
        if c.student_nu is not None or c.student_log10_nu is not None:
            nu = c.student_nu if c.student_nu is not None else e8.sample_nu(keys[6], c.student_log10_nu)
            eta = eta * e8.student_t_scale(keys[7], eta.shape[:2], nu)[..., None]
        vis = signal + eta

        P = fw.autos(R_sig_blk).mean(axis=1) + g.SEFD[None, :, None]  # (t, f, p) total power, Jy
        if c.P_sat is not None:
            cfac = e6.compression_factor(P, c.P_sat, g.SEFD[None, :, None])
            vis = e6.apply_compression(vis, cfac, g.a1, g.a2)

        aux = {
            "vis_rfi": fw.to_baselines(R_rfi.mean(axis=1), g.a1, g.a2),
            "vis_ast": fw.to_baselines(R_sky, g.a1, g.a2),
            "gains": gains,
            "noise": eta,
            "total_power": P,
        }
        return vis, aux

    def _noise(self, key, R_true_blk, R_sig_blk, gains):
        g, c = self.geom, self.config
        shape = (g.n_time, g.n_bl, g.n_freq)
        if c.noise == "none":
            return jnp.zeros(shape, dtype=jnp.complex128)
        if c.noise == "thermal":
            return e3.thermal_noise(key, shape, g.noise_std[None, None, :])

        eye = jnp.eye(g.n_ant)
        sefd = g.SEFD[None, None, :, None, None] * eye
        if c.noise_frame == "data":
            R_noise = R_sig_blk + sefd
        elif c.noise_frame == "sky":
            R_noise = R_true_blk + sefd
        else:
            raise ValueError(f"unknown noise_frame {c.noise_frame!r}")
        # The self-noise formula needs the Hermitian part only; enforce it numerically.
        R_noise = 0.5 * (R_noise + jnp.conj(jnp.swapaxes(R_noise, -1, -2)))

        if c.noise == "diag":
            std = e3.diag_noise_std(fw.autos(R_noise), g.a1, g.a2, g.n_samples)
            eta = e3.thermal_noise(key, shape, std)
        elif c.noise == "full":
            eta = fw.to_baselines(e3.self_noise_blocks(key, R_noise, g.n_samples), g.a1, g.a2)
        else:
            raise ValueError(f"unknown noise model {c.noise!r}")
        if c.noise_frame == "sky":
            gb = gains[:, g.a1] * jnp.conj(gains[:, g.a2])  # (t, bl, f)
            eta = eta * gb
        return eta
