from __future__ import annotations

from typing import Any

import numpy as np

### Code get_gpsvel.py ###
"""
 Calcule la vitesse GPS en UTM, comme getGPSvel.m, à partir d'une trajectoire x,y,t 
 ou d'un objet qui expose boat_vel.compute_boat_track(...).
 """


def _to_seconds(time: Any) -> np.ndarray:
    """
    Convert a time vector to seconds starting at 0.
    Accepts datetime64, numeric arrays, or None.
    """
    if time is None:
        return np.empty((0,), dtype=float)

    arr = np.asarray(time)
    if arr.size == 0:
        return np.empty((0,), dtype=float)

    if np.issubdtype(arr.dtype, np.datetime64):
        arr_ns = arr.astype("datetime64[ns]").astype("int64").astype(float)
        arr_s = arr_ns / 1e9
        return arr_s - arr_s[0]

    try:
        arr_s = arr.astype(float).reshape(-1)
        return arr_s - arr_s[0]
    except Exception:
        return np.arange(arr.size, dtype=float)


def _fill_nan_linear(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """
    Fill NaNs by linear interpolation in time.
    If fewer than 2 finite points exist, return the input unchanged.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    time_s = np.asarray(time_s, dtype=float).reshape(-1)

    if values.size != time_s.size:
        raise ValueError("values and time_s must have the same length")

    good = np.isfinite(values) & np.isfinite(time_s)
    if np.count_nonzero(good) < 2:
        return values

    out = values.copy()
    bad = ~good
    if np.any(bad):
        out[bad] = np.interp(time_s[bad], time_s[good], values[good])
    return out


def _extract_track(source: Any, nav_ref: str = "bt_vel") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract an UTM track (x, y, t) from:
    - a transect with boat_vel.compute_boat_track(...)
    - a VMADCP-like object exposing horizontal_position and time
    - a tuple/list (x, y, t)
    """
    if isinstance(source, (tuple, list)) and len(source) >= 3:
        x_track = np.asarray(source[0], dtype=float).reshape(-1)
        y_track = np.asarray(source[1], dtype=float).reshape(-1)
        time_s = _to_seconds(source[2])
        return x_track, y_track, time_s

    if hasattr(source, "boat_vel") and hasattr(source.boat_vel, "compute_boat_track"):
        ship_data = source.boat_vel.compute_boat_track(source, ref=nav_ref)
        x_track = np.asarray(ship_data["track_x_m"], dtype=float).reshape(-1)
        y_track = np.asarray(ship_data["track_y_m"], dtype=float).reshape(-1)

        time_source = (
            ship_data.get("time_s", None)
            or ship_data.get("time", None)
            or getattr(source, "time", None)
        )
        time_s = _to_seconds(time_source)
        if time_s.size == 0:
            time_s = np.arange(x_track.size, dtype=float)
        return x_track, y_track, time_s

    if hasattr(source, "horizontal_position"):
        hp = np.asarray(source.horizontal_position, dtype=float)
        if hp.ndim != 2 or hp.shape[0] < 2:
            raise ValueError("horizontal_position must be shaped (2, N)")

        x_track = hp[0].reshape(-1)
        y_track = hp[1].reshape(-1)
        time_s = _to_seconds(getattr(source, "time", None))
        if time_s.size == 0:
            time_s = np.arange(x_track.size, dtype=float)
        return x_track, y_track, time_s

    raise ValueError(
        "Unsupported source: expected a transect with boat_vel, a VMADCP-like object, "
        "or a (x, y, t) tuple."
    )


def get_gpsvel_from_track(
    track_x_m: Any,
    track_y_m: Any,
    time_s: Any,
    *,
    fill_missing: bool = True,
) -> np.ndarray:
    """
    Faithful Python analogue of MATLAB getGPSvel.m for Earth-frame velocity.

    Returns
    -------
    gpsvel : ndarray, shape (N, 3)
        Columns are [vx, vy, vz] in m/s.
        vx, vy are the negative time derivatives of the UTM track.
        vz is set to 0.

    Notes
    -----
    MATLAB getGPSvel.m computes:
        dxdt = gradient(X)./gradient(time)
        dydt = gradient(Y)./gradient(time)
        gpsvel(:,[1 2]) = -[dxdt, dydt]
    """
    x_track = np.asarray(track_x_m, dtype=float).reshape(-1)
    y_track = np.asarray(track_y_m, dtype=float).reshape(-1)
    time_s = _to_seconds(time_s)

    if x_track.size != y_track.size:
        raise ValueError("track_x_m and track_y_m must have the same length")

    if time_s.size != x_track.size:
        raise ValueError("time_s must have the same length as the track")

    if fill_missing:
        x_track = _fill_nan_linear(x_track, time_s)
        y_track = _fill_nan_linear(y_track, time_s)

    dxdt = np.gradient(x_track, time_s)
    dydt = np.gradient(y_track, time_s)

    gpsvel = np.zeros((x_track.size, 3), dtype=float)
    gpsvel[:, 0] = -dxdt
    gpsvel[:, 1] = -dydt
    return gpsvel


def get_gpsvel(
    source: Any,
    *,
    nav_ref: str = "bt_vel",
    fill_missing: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper that extracts the track from a transect or VMADCP-like
    object, then computes Earth-frame GPS velocity.
    """
    x_track, y_track, time_s = _extract_track(source, nav_ref=nav_ref)
    return get_gpsvel_from_track(
        x_track,
        y_track,
        time_s,
        fill_missing=fill_missing,
    )