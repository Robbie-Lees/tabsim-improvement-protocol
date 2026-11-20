"""V3: realism statistics for comparing simulated and real residuals.

Each function takes visibilities or residuals ``(n_time, n_bl)`` (one channel)
plus baseline metadata and returns plain numpy arrays, so the same code runs
on a simulation and on a MeerKAT measurement set read with dask-ms. The
comparison sets the priors marked "to be set" and decides whether E6-E8 are
needed.

* amplitude distributions per baseline-length bin (RFI fringe-washing makes
  them depend strongly on baseline length);
* closure phases and log closure amplitudes (insensitive to antenna gains, so
  they isolate non-closing structure: RFI, E5's baseline errors);
* temporal autocorrelation per baseline;
* correlation of residuals across baselines sharing an antenna (E3 predicts it).
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def amplitude_hist_by_length(vis, bl_length, length_bins, amp_bins):
    """Normalised amplitude histograms (n_length_bins, n_amp_bins)."""
    amp = np.abs(np.asarray(vis))
    which = np.digitize(bl_length, length_bins) - 1
    out = np.zeros((len(length_bins) - 1, len(amp_bins) - 1))
    for i in range(len(length_bins) - 1):
        sel = amp[:, which == i].ravel()
        if sel.size:
            out[i], _ = np.histogram(sel, bins=amp_bins, density=True)
    return out


def triangles(n_ant):
    return list(itertools.combinations(range(n_ant), 3))


def _bl_lookup(a1, a2):
    return {(int(p), int(q)): i for i, (p, q) in enumerate(zip(a1, a2))}


def _vis_pq(vis, lut, p, q):
    if (p, q) in lut:
        return vis[:, lut[(p, q)]]
    return np.conj(vis[:, lut[(q, p)]])


def closure_phases(vis, a1, a2, tri=None):
    """Closure phases (n_time, n_tri) arg(V_pq V_qr V_rp)."""
    vis = np.asarray(vis)
    lut = _bl_lookup(a1, a2)
    tri = tri or triangles(int(max(a1.max(), a2.max())) + 1)
    return np.stack([np.angle(_vis_pq(vis, lut, p, q) * _vis_pq(vis, lut, q, r) * _vis_pq(vis, lut, r, p))
                     for p, q, r in tri], axis=1)


def log_closure_amplitudes(vis, a1, a2, quads=None):
    """log |V_pq V_rs / (V_pr V_qs)| for antenna quadruples (n_time, n_quad)."""
    vis = np.asarray(vis)
    lut = _bl_lookup(a1, a2)
    n_ant = int(max(a1.max(), a2.max())) + 1
    quads = quads or list(itertools.combinations(range(n_ant), 4))
    out = []
    for p, q, r, s in quads:
        num = np.abs(_vis_pq(vis, lut, p, q) * _vis_pq(vis, lut, r, s))
        den = np.abs(_vis_pq(vis, lut, p, r) * _vis_pq(vis, lut, q, s))
        out.append(np.log(num / den))
    return np.stack(out, axis=1)


def temporal_autocorr(res, max_lag):
    """Mean normalised autocorrelation over baselines, lags 0..max_lag."""
    res = np.asarray(res) - np.mean(res, axis=0, keepdims=True)
    var = np.mean(np.abs(res) ** 2, axis=0)
    ac = [np.mean(np.real(np.mean(res[l:] * np.conj(res[: res.shape[0] - l]), axis=0)) / var) for l in range(max_lag + 1)]
    return np.array(ac)


def shared_antenna_correlation(res, a1, a2):
    """Mean |corr| of residuals for baseline pairs sharing an antenna vs not."""
    res = np.asarray(res) - np.mean(res, axis=0, keepdims=True)
    res = res / np.sqrt(np.mean(np.abs(res) ** 2, axis=0, keepdims=True))
    C = np.abs(res.conj().T @ res) / res.shape[0]
    share = (a1[:, None] == a1[None]) | (a1[:, None] == a2[None]) | (a2[:, None] == a1[None]) | (a2[:, None] == a2[None])
    off = ~np.eye(len(a1), dtype=bool)
    return float(C[share & off].mean()), float(C[~share].mean()) if (~share).any() else np.nan


def compare(sim, real):
    """Two-sample KS distance and Wasserstein-1 between flattened statistics."""
    sim, real = np.ravel(sim), np.ravel(real)
    return {"ks": stats.ks_2samp(sim, real).statistic, "w1": stats.wasserstein_distance(sim, real)}
