"""Differentiable forward-model kernels on full coherency matrices.

Everything here works on the coherency matrix ``R[t, f, p, q]`` of one
integration rather than on the baseline list. Three reasons:

* RFI visibilities become a Gram matrix: with the per-antenna field
  ``B[s, k, p] = A_ps(t_k) exp(i phi_ps(t_k))`` the time-averaged RFI coherency
  is ``R_pq = mean_k sum_s B_kps conj(B_kqs)``, one batched matrix product per
  integration (GEMM friendly) instead of an outer product per baseline.
* The autocorrelations ``R_pp`` (total power per antenna) come for free; the
  signal-dependent noise (E3) and receiver compression (E6) need them.
* E3 needs the full matrix to draw self-noise with the right correlations.

The kernels reproduce tab-sim 0.1.0 (``tabsim.jax.interferometry.rfi_vis_beam``
and ``astro_vis``) in the Gaussian limit; ``tests/test_forward.py`` checks this
against tab-sim itself.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

from tabsim_sbi._jax import C_LIGHT, herm

try:  # tab-sim >= 0.1.0 ships an accurate float64 j1 (Cephes) for the device beam
    from tabsim.jax.beam import j1 as _j1
except Exception:  # pragma: no cover - fallback when tab-sim is older or absent
    warnings.warn(
        "tabsim.jax.beam.j1 not available (tab-sim < 0.1.0?); using "
        "jax.scipy.special.bessel_jn, which loses accuracy for x > ~40."
    )

    def _j1(x):
        return jax.scipy.special.bessel_jn(x, v=1)[1]


# ----------------------------------------------------------------------------
# Beams and source geometry
# ----------------------------------------------------------------------------


def airy_voltage(theta, freqs, dish_d):
    """Airy voltage beam ``2 J1(x) / x``, ``x = pi f D sin(theta) / c``.

    ``theta`` in radians with shape (..., ); ``dish_d`` scalar or broadcastable
    to ``theta`` (per-antenna dishes for E2). Returns (..., n_freq). Signed
    sidelobes, as tab-sim.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)[..., None]
    dish_d = jnp.asarray(dish_d, dtype=jnp.float64)
    if dish_d.ndim:
        dish_d = dish_d[..., None]
    x = jnp.pi * jnp.asarray(freqs, dtype=jnp.float64) * dish_d * jnp.sin(theta) / C_LIGHT
    x = jnp.where(jnp.abs(x) < 1e-12, 1e-12, x)
    return 2.0 * _j1(x) / x


def source_vectors(src_xyz, ants_xyz):
    """Unit vectors and distances from each antenna to each source.

    src_xyz (n_src, n_tf, 3), ants_xyz (n_tf, n_ant, 3) ->
    unit (n_src, n_tf, n_ant, 3), distance (n_src, n_tf, n_ant).
    """
    rel = src_xyz[:, :, None, :] - ants_xyz[None, :, :, :]
    d = jnp.linalg.norm(rel, axis=-1)
    return rel / d[..., None], d


def angle_to(unit_vecs, axis):
    """Angle (rad) between unit vectors (..., 3) and ``axis`` (3,) or (..., 3)."""
    cos = jnp.sum(unit_vecs * axis, axis=-1)
    return jnp.arccos(jnp.clip(cos, -1.0, 1.0))


def rfi_amplitude(Pv_t, distance, beam):
    """Apparent amplitude in sqrt(Jy): ``sqrt(Pv / (4 pi d^2) 1e26) * beam``.

    Pv_t (n_src, n_tf, n_freq) W/Hz, distance (n_src, n_tf, n_ant),
    beam (n_src, n_tf, n_ant, n_freq) real or complex.
    """
    Sv = Pv_t[:, :, None, :] / (4.0 * jnp.pi * distance[..., None] ** 2) * 1e26
    return jnp.sqrt(Sv) * beam


def fringe_phasor(c_distance, freqs):
    """``exp(-2 pi i f / c (d_p + w_p))`` per source, fine time, antenna, channel.

    The antenna mean is removed first (in float64) as in tab-sim's kernel: only
    differences between antennas reach a baseline.
    """
    c_distance = c_distance - jnp.mean(c_distance, axis=-1, keepdims=True)
    phase = -2.0 * jnp.pi * jnp.asarray(freqs)[None, None, None, :] / C_LIGHT * c_distance[..., None]
    return jnp.exp(1j * phase)


# ----------------------------------------------------------------------------
# Coherency matrices
# ----------------------------------------------------------------------------


def rfi_coherency(field, n_int, n_blocks=1):
    """Time-averaged RFI coherency from the per-antenna field.

    field (n_src, n_tf, n_ant, n_freq), complex, ``A * exp(i phi)``.
    Returns (n_time, n_blocks, n_freq, n_ant, n_ant): the average over each of
    ``n_blocks`` equal sub-blocks of every integration. ``n_blocks=1`` gives the
    integration average; more blocks keep within-integration variation for E3.
    """
    n_src, n_tf, n_ant, n_freq = field.shape
    n_time = n_tf // n_int
    if n_int % n_blocks:
        raise ValueError(f"n_blocks={n_blocks} must divide n_int={n_int}")
    B = field.reshape(n_src, n_time, n_blocks, n_int // n_blocks, n_ant, n_freq)
    return jnp.einsum("stbkpf,stbkqf->tbfpq", B, jnp.conj(B)) / (n_int // n_blocks)


def sky_coherency(I_app, lmn, ants_uvw, freqs, gauss_sigma=None):
    """Sky coherency at integration centroids.

    I_app (n_src, n_freq) apparent flux (beam applied), lmn (n_src, 3),
    ants_uvw (n_time, n_ant, 3), optional circular Gaussian widths (n_src,) in
    radians for extended sources (E4). Returns (n_time, n_freq, n_ant, n_ant)
    with ``R_pq = sum_s I_s exp(2 pi i f/c (uvw_p - uvw_q).(lmn_s - s0)) T_s,pq``.
    """
    n_src = lmn.shape[0]
    n_time, n_ant, _ = ants_uvw.shape
    n_freq = freqs.shape[0]
    if n_src == 0:
        return jnp.zeros((n_time, n_freq, n_ant, n_ant), dtype=jnp.complex128)
    s0 = jnp.array([0.0, 0.0, 1.0])
    k = 2.0 * jnp.pi * jnp.asarray(freqs) / C_LIGHT  # (n_freq,)
    proj = jnp.einsum("tpi,si->tps", ants_uvw, lmn - s0)  # (n_time, n_ant, n_src)
    E = jnp.exp(1j * proj[:, None, :, :] * k[None, :, None, None])  # (t, f, p, s)
    if gauss_sigma is None:
        return jnp.einsum("tfps,sf,tfqs->tfpq", E, I_app.astype(jnp.complex128), jnp.conj(E))

    # Extended sources: the taper depends on the baseline, so accumulate per source.
    uv = ants_uvw[:, :, None, :2] - ants_uvw[:, None, :, :2]  # (t, p, q, 2)
    uv2 = jnp.sum(uv**2, axis=-1)  # (t, p, q)

    def add(R, s):
        taper = jnp.exp(-2.0 * jnp.pi**2 * gauss_sigma[s] ** 2 * uv2[:, None] * (k[None, :, None, None] / (2 * jnp.pi)) ** 2)
        Es = E[..., s]
        R = R + I_app[s][None, :, None, None] * Es[..., :, None] * jnp.conj(Es[..., None, :]) * taper
        return R, None

    R0 = jnp.zeros((n_time, n_freq, n_ant, n_ant), dtype=jnp.complex128)
    return jax.lax.scan(add, R0, jnp.arange(n_src))[0]


def apparent_flux(I, lmn, freqs, dish_d):
    """Apply the Airy power beam to intrinsic fluxes, as tab-sim's ``addAstro``."""
    theta = jnp.arcsin(jnp.clip(jnp.linalg.norm(lmn[:, :2], axis=-1), 0.0, 1.0))
    return I * airy_voltage(theta, freqs, dish_d) ** 2


def apply_gains(R, gains):
    """``g_p R_pq conj(g_q)``; R (..., n_freq, n_ant, n_ant), gains (n_time, n_ant, n_freq).

    R may carry a sub-block axis after time: (n_time, n_blk, n_freq, n_ant, n_ant).
    """
    g = jnp.swapaxes(gains, -1, -2)  # (t, f, p)
    if R.ndim == 5:
        g = g[:, None]
    return g[..., :, None] * R * jnp.conj(g[..., None, :])


def to_baselines(R, a1, a2):
    """(n_time, n_freq, n_ant, n_ant) -> (n_time, n_bl, n_freq), tab-sim layout."""
    return jnp.swapaxes(R[..., a1, a2], -1, -2)


def autos(R):
    """Real diagonal (..., n_ant) of coherency matrices (..., n_ant, n_ant)."""
    return jnp.real(jnp.diagonal(R, axis1=-2, axis2=-1))


def from_baselines(vis, a1, a2, n_ant, diag=None):
    """Hermitian matrices (n_time, n_freq, n_ant, n_ant) from baselines (n_time, n_bl, n_freq)."""
    n_time, _, n_freq = vis.shape
    R = jnp.zeros((n_time, n_freq, n_ant, n_ant), dtype=jnp.complex128)
    v = jnp.swapaxes(vis, -1, -2)
    R = R.at[..., a1, a2].set(v).at[..., a2, a1].set(jnp.conj(v))
    if diag is not None:
        R = R + jnp.eye(n_ant) * diag[..., None, :]
    return R


def is_hermitian(R, atol=1e-10):
    return jnp.allclose(R, herm(R), atol=atol)
