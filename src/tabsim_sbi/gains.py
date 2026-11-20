"""Gain parameterisation shared by the simulator and the compression model.

The interesting parameters of Protocols A and B1 are gain inducing points. For
the posterior to mean anything, the simulator must generate gains from exactly
the parameters the compression model uses, so both import this class.

Gains are ``g_p(t) = exp(a_p(t) + i phi_p(t))`` with log-amplitude ``a`` and
phase ``phi`` interpolated from ``n_g`` inducing points per antenna by the
conditional mean of a squared-exponential GP (Paper II style). Parameters are
standardised: ``z ~ N(0, I)`` a priori, physical values ``a = mu_a + sigma_a P z``.

Gauge: the phase of ``ref_ant`` is fixed to zero (tab-sim makes the last
antenna's gain real; tabascal pins a reference antenna). The overall amplitude
is kept: on a calibrator of known flux it is identifiable (Paper I).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from tabsim_sbi._jax import gp_basis, inducing_times


@dataclass(frozen=True)
class GainParameterisation:
    times: jnp.ndarray  # (n_time,) s
    n_ant: int
    n_g: int = 3  # inducing points per antenna (Paper II: 3 per 15 min for phases)
    length_scale: float = 300.0  # s
    sigma_amp: float = 0.1  # prior std of log|g|
    sigma_phase: float = 0.5  # rad
    mu_amp: float = 0.0  # prior mean of log|g|
    ref_ant: int = -1
    P: jnp.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        t = jnp.asarray(self.times)
        t_ind = inducing_times(float(t[0]), float(t[-1]), n=self.n_g)
        object.__setattr__(self, "P", gp_basis(t, t_ind, self.length_scale))

    # -- layout ------------------------------------------------------------
    @property
    def ref(self) -> int:
        return self.ref_ant % self.n_ant

    @property
    def phase_ants(self) -> np.ndarray:
        return np.array([p for p in range(self.n_ant) if p != self.ref])

    @property
    def n_amp(self) -> int:
        return self.n_ant * self.n_g

    @property
    def n_phase(self) -> int:
        return (self.n_ant - 1) * self.n_g

    @property
    def n_params(self) -> int:
        return self.n_amp + self.n_phase

    def amp_index(self, p: int, offset: int = 0) -> np.ndarray:
        return offset + p * self.n_g + np.arange(self.n_g)

    def phase_index(self, p: int, offset: int = 0) -> np.ndarray:
        if p == self.ref:
            return np.zeros(0, dtype=int)
        k = int(np.searchsorted(self.phase_ants, p))
        return offset + self.n_amp + k * self.n_g + np.arange(self.n_g)

    def antenna_blocks(self, offset: int = 0, amplitudes: bool = True, phases: bool = True):
        """One block of parameter indices per antenna (the protocol's default blocking)."""
        blocks = []
        for p in range(self.n_ant):
            idx = []
            if amplitudes:
                idx.append(self.amp_index(p, offset))
            if phases:
                idx.append(self.phase_index(p, offset))
            idx = np.concatenate(idx)
            if idx.size:
                blocks.append(idx)
        return blocks

    # -- maps --------------------------------------------------------------
    def log_amp_phase(self, z):
        """z (n_params,) -> log-amplitude and phase, each (n_time, n_ant)."""
        z_amp = z[: self.n_amp].reshape(self.n_ant, self.n_g)
        z_ph = z[self.n_amp :].reshape(self.n_ant - 1, self.n_g)
        a = self.mu_amp + self.sigma_amp * (self.P @ z_amp.T)
        ph = jnp.zeros((self.P.shape[0], self.n_ant)).at[:, self.phase_ants].set(
            self.sigma_phase * (self.P @ z_ph.T)
        )
        return a, ph

    def gains(self, z, n_freq: int = 1):
        """Complex gains (n_time, n_ant, n_freq), frequency independent."""
        a, ph = self.log_amp_phase(z)
        g = jnp.exp(a + 1j * ph)
        return jnp.repeat(g[..., None], n_freq, axis=-1)
