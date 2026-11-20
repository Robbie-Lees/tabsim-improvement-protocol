"""E6: receiver compression (crude stand-in for non-linearity).

Each antenna's voltage is scaled by ``c_p = sqrt((1 + SEFD / P_sat) / (1 + P_p / P_sat))``
with ``P_p`` its total power (SEFD + sky + RFI, in Jy) and ``P_sat`` a saturation
power per antenna. The normalisation makes ``c_p = 1`` with no signal, so the
constant part is a gain that calibration absorbs and only the RFI-dependent part
is new. The factor multiplies signal and noise alike, ``V_pq -> c_p c_q V_pq``.

Applied per integration using the integration-mean power; true compression acts
on the instantaneous power, so this understates the effect of fast RFI
variation. Both papers assume a linear response; the kJy regime of Paper II is
where this matters.
"""

from __future__ import annotations

import jax.numpy as jnp


def compression_factor(P, P_sat, SEFD):
    """P (..., n_ant) total power; returns c (..., n_ant)."""
    return jnp.sqrt((1.0 + SEFD / P_sat) / (1.0 + P / P_sat))


def apply_compression(vis, c, a1, a2):
    """vis (n_time, n_bl, n_freq); c (n_time, n_freq, n_ant)."""
    cc = c[..., a1] * c[..., a2]  # (t, f, bl)
    return vis * jnp.swapaxes(cc, -1, -2)
