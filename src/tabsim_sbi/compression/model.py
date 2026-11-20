"""Compression model: an approximate, differentiable model of the data.

A Paper II style parameterisation in standardised coordinates (prior N(0, I)):

* gains: :class:`~tabsim_sbi.gains.GainParameterisation` (shared with the simulator);
* RFI: a complex signal ``A_ps(t)`` per satellite and antenna, a GP with a
  squared-exponential prior (length scale ``rfi_length_scale``, Paper II uses
  24 s) on inducing points, multiplied by the fixed near-field fringe phasor
  ``K_ps(t) = exp(-2 pi i f / c (d_ps + w_p))`` computed from the *TLE*
  trajectories (variant A1: no orbit parameters, A absorbs position errors);
* sky, one of
    - ``"calibrator"``: known visibilities from the geometry's sky model (Protocol A);
    - ``"freeform"``: known part plus a free complex GP per baseline (Protocol B1),
      the nuisance that makes the target-field problem larger than the data;
    - ``"parametric"``: fluxes of a few sources at known positions (Protocol B2).

``mu(phi)`` returns the model visibilities (n_time, n_bl) for the single
channel the protocols use. Only gradients of ``mu`` are ever needed, so any
differentiable model -- including tabascal's own forward pass -- can replace
this class in :mod:`tabsim_sbi.compression.score` and ``.hardening``.

The current tabascal (epfl-radio-astro/tabascal) parameterises RFI with an
FFT-based GP and gains with a gauge-fixed constant per antenna; this class
follows the protocol's description (inducing points, SE prior) instead, and is
self-contained so the machinery can be tested without tabascal installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from tabsim_sbi import forward as fw
from tabsim_sbi._jax import gp_basis, inducing_times
from tabsim_sbi.gains import GainParameterisation
from tabsim_sbi.geometry import ObservationGeometry


class CompressionModel:
    def __init__(
        self,
        geom: ObservationGeometry,
        gains: GainParameterisation,
        *,
        rfi_length_scale: float = 24.0,
        rfi_spacing: Optional[float] = None,
        rfi_prior_scale: Optional[np.ndarray] = None,
        n_int_model: Optional[int] = None,
        sky: str = "calibrator",
        freeform_length_scale: float = 120.0,
        freeform_spacing: Optional[float] = None,
        freeform_scale: float = 1.0,
        parametric_lmn: Optional[np.ndarray] = None,
        parametric_I0: Optional[np.ndarray] = None,
        parametric_sigma: Optional[np.ndarray] = None,
        sigma_c=None,
    ):
        if geom.n_freq != 1:
            raise ValueError("The compression model uses one channel, as the protocols do.")
        self.geom, self.gp = geom, gains
        self.sky_mode = sky
        g = geom
        f = g.freqs

        # Fine samples kept by the model: n_int_model evenly spaced per integration.
        n_int_model = n_int_model or g.n_int
        if g.n_int % n_int_model:
            raise ValueError("n_int_model must divide the geometry's n_int")
        stride = g.n_int // n_int_model
        fine_idx = (np.arange(g.n_time)[:, None] * g.n_int + np.arange(stride // 2, g.n_int, stride)[None, :]).ravel()
        self.n_int_model = n_int_model
        t_fine = g.times_fine[fine_idx]

        # Fixed fringe phasors from the TLE trajectories (A1): (n_sat, n_tf_model, n_ant)
        if g.n_sat:
            u, d = fw.source_vectors(g.sat_xyz[:, fine_idx], g.ants_xyz[fine_idx])
            self.K = fw.fringe_phasor(d + g.ants_w[fine_idx][None], f)[..., 0]
            if rfi_prior_scale is None:
                beam = fw.airy_voltage(fw.angle_to(u, g.pointing), f, g.dish_d)
                amp = jnp.abs(fw.rfi_amplitude(jnp.broadcast_to(g.sat_Pv[:, None, :], d.shape[:2] + (1,)), d, beam))
                rfi_prior_scale = jnp.sqrt(jnp.mean(amp[..., 0] ** 2, axis=(1, 2)))
            self.rfi_scale = jnp.asarray(rfi_prior_scale, dtype=jnp.float64).reshape(g.n_sat)
        else:
            self.K = jnp.zeros((0, t_fine.shape[0], g.n_ant), dtype=jnp.complex128)
            self.rfi_scale = jnp.zeros(0)
        t_rfi = inducing_times(float(g.times[0] - g.int_time / 2), float(g.times[-1] + g.int_time / 2),
                               spacing=rfi_spacing or rfi_length_scale)
        self.P_rfi = gp_basis(t_fine, t_rfi, rfi_length_scale)  # (n_tf_model, n_r)
        self.n_r = self.P_rfi.shape[1]

        # Known sky
        I_app = fw.apparent_flux(g.ast_I, g.ast_lmn, f, g.dish_d)
        self.R_known = fw.sky_coherency(I_app, g.ast_lmn, g.ants_uvw, f)[:, 0]  # (t, p, q)

        # Sky nuisance / parameters
        self.n_sky = 0
        if sky == "freeform":
            t_ast = inducing_times(float(g.times[0]), float(g.times[-1]), spacing=freeform_spacing or freeform_length_scale)
            self.P_ast = gp_basis(g.times, t_ast, freeform_length_scale)  # (n_time, n_a)
            self.freeform_scale = freeform_scale
            self.n_sky = 2 * g.n_bl * self.P_ast.shape[1]
        elif sky == "parametric":
            lmn = jnp.asarray(parametric_lmn)
            self.par_lmn = lmn
            self.par_I0 = jnp.asarray(parametric_I0, dtype=jnp.float64)
            self.par_sigma = jnp.asarray(parametric_sigma if parametric_sigma is not None else 0.2 * self.par_I0)
            beam2 = fw.apparent_flux(jnp.ones((lmn.shape[0], 1)), lmn, f, g.dish_d)[:, 0]
            k = 2 * jnp.pi * f[0] / fw.C_LIGHT
            proj = jnp.einsum("tpi,si->tps", g.ants_uvw, lmn - jnp.array([0.0, 0.0, 1.0]))
            self.par_E = jnp.exp(1j * k * proj) * jnp.sqrt(beam2)[None, None, :]  # (t, p, s)
            self.n_sky = lmn.shape[0]
        elif sky != "calibrator":
            raise ValueError(f"unknown sky mode {sky!r}")

        # Parameter layout
        n_gain = gains.n_params
        n_rfi = 2 * g.n_sat * g.n_ant * self.n_r
        self.index = {
            "gain": np.arange(n_gain),
            "rfi": n_gain + np.arange(n_rfi),
            "sky": n_gain + n_rfi + np.arange(self.n_sky),
        }
        self.dim = n_gain + n_rfi + self.n_sky
        self.sigma_c = geom.noise_std[0] if sigma_c is None else sigma_c

    # ------------------------------------------------------------------
    def unpack(self, phi):
        i = self.index
        return phi[i["gain"]], phi[i["rfi"]], phi[i["sky"]]

    def rfi_signal(self, z_rfi):
        """Complex A (n_sat, n_tf_model, n_ant) from standardised parameters."""
        g = self.geom
        z = z_rfi.reshape(2, g.n_sat, g.n_ant, self.n_r)
        zc = z[0] + 1j * z[1]
        return self.rfi_scale[:, None, None] * jnp.einsum("tr,spr->stp", self.P_rfi, zc) / jnp.sqrt(2.0)

    def rfi_coherency(self, z_rfi):
        g = self.geom
        if not g.n_sat:
            return jnp.zeros((g.n_time, g.n_ant, g.n_ant), dtype=jnp.complex128)
        B = (self.rfi_signal(z_rfi) * self.K)[..., None]  # (s, tf, p, 1)
        return fw.rfi_coherency(B, self.n_int_model)[:, 0, 0]  # (t, p, q)

    def sky_coherency(self, z_sky):
        g = self.geom
        if self.sky_mode == "calibrator":
            return self.R_known
        if self.sky_mode == "freeform":
            z = z_sky.reshape(2, g.n_bl, -1)
            v = self.freeform_scale * (self.P_ast @ (z[0] + 1j * z[1]).T) / jnp.sqrt(2.0)  # (t, bl)
            dR = fw.from_baselines(v[..., None], g.a1, g.a2, g.n_ant)[:, 0]
            return self.R_known + dR
        I = self.par_I0.reshape(-1) + self.par_sigma.reshape(-1) * z_sky
        return jnp.einsum("tps,s,tqs->tpq", self.par_E, I.astype(jnp.complex128), jnp.conj(self.par_E))

    def gains(self, z_gain):
        return self.gp.gains(z_gain, 1)[..., 0]  # (t, p)

    def coherency(self, phi):
        z_gain, z_rfi, z_sky = self.unpack(phi)
        R = self.sky_coherency(z_sky) + self.rfi_coherency(z_rfi)
        gg = self.gains(z_gain)
        return gg[:, :, None] * R * jnp.conj(gg[:, None, :])

    def mu(self, phi):
        """Model visibilities (n_time, n_bl)."""
        R = self.coherency(phi)
        return R[:, self.geom.a1, self.geom.a2]

    def total_power_fine(self, phi):
        """Data-frame total power per antenna (n_time, n_int_model, n_ant), for E3-aware weights."""
        g = self.geom
        z_gain, z_rfi, z_sky = self.unpack(phi)
        gg2 = jnp.abs(self.gains(z_gain)) ** 2  # (t, p)
        sky = jnp.real(jnp.diagonal(self.sky_coherency(z_sky), axis1=-2, axis2=-1))  # (t, p)
        if g.n_sat:
            A2 = jnp.sum(jnp.abs(self.rfi_signal(z_rfi)) ** 2, axis=0)  # (tf, p)
            A2 = A2.reshape(g.n_time, self.n_int_model, g.n_ant)
        else:
            A2 = jnp.zeros((g.n_time, self.n_int_model, g.n_ant))
        return gg2[:, None, :] * (sky[:, None, :] + A2) + g.SEFD[0]

    def signal_dependent_sigma(self, phi):
        """Per-component noise std (n_time, n_bl) at phi from E3's diagonal approximation."""
        g = self.geom
        P = self.total_power_fine(phi)
        prod = (P[..., g.a1] * P[..., g.a2]).mean(axis=1)
        return jnp.sqrt(0.5 * prod / g.n_samples)

    # ------------------------------------------------------------------
    def theta_blocks(self, amplitudes=True, phases=True):
        """Per-antenna gain blocks (indices into phi)."""
        return self.gp.antenna_blocks(int(self.index["gain"][0]) if self.index["gain"].size else 0, amplitudes, phases)

    def init_params(self, key, d=None, rfi_init_scale=0.3):
        """A starting point for the MAP fit.

        The RFI term is bilinear in A, so A = 0 is a saddle point of the loss;
        start the RFI parameters at small random values instead.
        """
        phi = jnp.zeros(self.dim)
        n = self.index["rfi"].size
        if n:
            phi = phi.at[self.index["rfi"]].set(rfi_init_scale * jax.random.normal(key, (n,)))
        return phi
