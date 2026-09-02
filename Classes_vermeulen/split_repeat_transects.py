from __future__ import annotations

from typing import Any

import numpy as np

from .EnsembleFilter import EnsembleFilter


"""
splits data in a cross-section to different repeat transects :

   ef_out = split_repeat_transects(v,xs) returns a vector of
   ensemblefilter object. Each object indicates the ensembles in the
   VMADCP object v that belong a specific repeat transect on the 
   cross-section xs. The algorithm detects peaks in the n-coordinate of 
   the track, when the peak is spaced at least the scale of the 
   cross-section (xs.scale) from the previous peak.

   split_repeat_transects(..., scale) optionally specify scale for peak
   detection algorithm.

   split_repeat_transect(..., ef) optionally specify an ensemble filter
   object to exclude part of the data

"""


def _peakdet(v: np.ndarray, delta: float):
    """
    MATLAB-like peak detection.

    Returns
    -------
    maxidx, minidx : np.ndarray
        Arrays of shape (k, 2) with [index, value] for maxima and minima.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a positive finite scalar")

    maxtab = []
    mintab = []

    mn = np.inf
    mx = -np.inf
    mnpos = np.nan
    mxpos = np.nan

    look_for_max = True

    for ii, val in enumerate(v):
        if val > mx:
            mx = val
            mxpos = ii
        if val < mn:
            mn = val
            mnpos = ii

        if look_for_max:
            if val < mx - delta:
                maxtab.append((mxpos, mx))
                mn = val
                mnpos = ii
                look_for_max = False
        else:
            if val > mn + delta:
                mintab.append((mnpos, mn))
                mx = val
                mxpos = ii
                look_for_max = True

    maxidx = np.asarray(maxtab, dtype=float).reshape(-1, 2) if maxtab else np.empty((0, 2), dtype=float)
    minidx = np.asarray(mintab, dtype=float).reshape(-1, 2) if mintab else np.empty((0, 2), dtype=float)
    return maxidx, minidx


def split_repeat_transects(v: Any, xs: Any, *args, scale: float | None = None, ef: EnsembleFilter | None = None):
    """
    Split a VMADCP transect into repeat-transect ensemble filters.

    Parameters
    ----------
    v:
        VMADCP-like object.
    xs:
        XSection-like object with xy2sn() and scale.
    scale:
        Peak detection threshold. Defaults to xs.scale.
    ef:
        Optional EnsembleFilter excluding data before repeat detection.

    Returns
    -------
    list[EnsembleFilter]
        One filter per detected repeat transect.
    """
    if not hasattr(xs, "xy2sn"):
        raise TypeError("xs must provide xy2sn()")
    if not hasattr(v, "horizontal_position"):
        raise TypeError("v must provide horizontal_position")
    if scale is None:
        scale = float(getattr(xs, "scale", 0.0))
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite scalar")

    n_ens = int(getattr(v, "nensembles", getattr(v, "n_ensembles", 0)))
    if n_ens <= 0:
        hp = np.asarray(v.horizontal_position, dtype=float)
        if hp.ndim == 2:
            n_ens = hp.shape[1]

    if ef is None:
        ef = EnsembleFilter(np.zeros((n_ens,), dtype=bool))
    elif not hasattr(ef, "bad_ensembles"):
        raise TypeError("ef must be an EnsembleFilter-like object with bad_ensembles")

    bad = np.asarray(ef.bad_ensembles, dtype=bool).reshape(-1)
    if bad.size < n_ens:
        padded = np.zeros((n_ens,), dtype=bool)
        padded[: bad.size] = bad
        bad = padded
    elif bad.size > n_ens:
        bad = bad[:n_ens]

    fgood = ~bad

    hpos = np.asarray(v.horizontal_position, dtype=float)
    if hpos.ndim != 2 or hpos.shape[0] < 2:
        raise ValueError("v.horizontal_position must have shape (2, n_ensembles)")

    xg = hpos[0, fgood]
    yg = hpos[1, fgood]
    _, n = xs.xy2sn(xg, yg)
    n = np.asarray(n, dtype=float).reshape(-1)

    finite = np.isfinite(n)
    if np.count_nonzero(finite) < 2:
        return []

    n = n[finite]
    good_idx = np.flatnonzero(fgood)[finite]

    maxidx, minidx = _peakdet(n, scale)

    if maxidx.size == 0 and minidx.size == 0:
        return []

    idx = np.sort(
        np.concatenate(
            [
                maxidx[:, 0].astype(int) if maxidx.size else np.empty((0,), dtype=int),
                minidx[:, 0].astype(int) if minidx.size else np.empty((0,), dtype=int),
            ]
        )
    )

    if idx.size < 2:
        return []

    ef_out = []
    nrp = idx.size - 1

    for cc in range(nrp):
        cfilt = np.ones((n_ens,), dtype=bool)
        start = int(good_idx[idx[cc]])
        stop = int(good_idx[idx[cc + 1]])
        if stop < start:
            start, stop = stop, start
        cfilt[start : stop + 1] = False
        cfilt |= bad
        ef_out.append(EnsembleFilter(cfilt))

    return ef_out