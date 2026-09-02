from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .NMEA import NMEAStream
from qrevint_21_03.Classes.GPSData import GPSData


def _is_string_like(value: Any) -> bool:
    """Return True for text lines, False for nested groups."""
    return isinstance(value, (str, bytes))


def _normalize_ensemble_groups(
    nmea_input: Iterable[Any],
) -> list[list[str]]:
    """
    Normalize the input into a list of ensemble groups.

    Accepted forms:
    - list[str]
      One single ensemble group containing all lines.
    - list[list[str]]
      Several ensemble groups, one list per ensemble.
    - any iterable that yields strings or iterables of strings.

    The helper does NOT try to infer ensemble boundaries from a flat stream.
    If the source is a raw uninterrupted stream, split it upstream first.
    """
    data = list(nmea_input)
    if not data:
        return []

    # Flat list of NMEA lines -> treat as a single ensemble group.
    if all(_is_string_like(item) for item in data):
        return [[str(item) for item in data]]

    # Nested structure -> convert each group to a list of strings.
    groups: list[list[str]] = []
    for group in data:
        if group is None:
            groups.append([])
            continue

        if _is_string_like(group):
            groups.append([str(group)])
        else:
            groups.append([str(line) for line in group])

    return groups


def _pad_numeric_rows(rows: list[list[float]], fill_value: float = np.nan) -> np.ndarray:
    """
    Convert ragged numeric rows into a 2D float array.

    Output shape:
        [n_ensembles, max_length_within_ensemble]
    """
    if not rows:
        return np.empty((0, 0), dtype=float)

    max_len = max((len(row) for row in rows), default=0)
    out = np.full((len(rows), max_len), fill_value, dtype=float)

    for ii, row in enumerate(rows):
        arr = np.asarray(row, dtype=float).reshape(-1)
        out[ii, : arr.size] = arr

    return out


def _pad_object_rows(rows: list[list[Any]], fill_value: Any = "") -> np.ndarray:
    """
    Convert ragged object rows into a 2D object array.

    Useful for VTG mode indicators, which are character values.
    """
    if not rows:
        return np.empty((0, 0), dtype=object)

    max_len = max((len(row) for row in rows), default=0)
    out = np.full((len(rows), max_len), fill_value, dtype=object)

    for ii, row in enumerate(rows):
        arr = list(row)
        out[ii, : len(arr)] = arr

    return out


def _finite_values(values: list[float]) -> list[float]:
    """Return only finite values."""
    out = []
    for value in values:
        try:
            if np.isfinite(value):
                out.append(float(value))
        except Exception:
            continue
    return out


def _mean_or_nan(values: list[float]) -> float:
    """Compute a finite mean, or NaN if nothing usable exists."""
    finite = _finite_values(values)
    if not finite:
        return np.nan
    return float(np.nanmean(np.asarray(finite, dtype=float)))


def _last_or_nan(values: list[float]) -> float:
    """Return the last finite value, or NaN if nothing usable exists."""
    for value in reversed(values):
        try:
            if np.isfinite(value):
                return float(value)
        except Exception:
            continue
    return np.nan


def _first_or_nan(values: list[float]) -> float:
    """Return the first finite value, or NaN if nothing usable exists."""
    for value in values:
        try:
            if np.isfinite(value):
                return float(value)
        except Exception:
            continue
    return np.nan


def _select_scalar(values: list[float], method: str = "last") -> float:
    """
    Pick one representative value from a list of values.

    This is used for the external / ensemble-level values expected by GPSData.
    """
    method = (method or "last").lower()
    if method == "mean":
        return _mean_or_nan(values)
    if method == "first":
        return _first_or_nan(values)
    return _last_or_nan(values)


def _build_ensemble_payload(
    parsed: list[dict[str, Any]],
    gga_method: str = "last",
    vtg_method: str = "last",
    use_rmc_as_fallback: bool = True,
) -> dict[str, Any]:
    """
    Extract raw matrices and ensemble-level values from one parsed ensemble.

    This is the bridge between:
    - sentence-level NMEA parsing
    - GPSData.populate_data(...), which expects numeric arrays.
    """
    gga = [p for p in parsed if p.get("type") == "GGA"]
    vtg = [p for p in parsed if p.get("type") == "VTG"]
    rmc = [p for p in parsed if p.get("type") == "RMC"]

    # Raw GGA arrays: one row per ensemble, one column per GGA sentence in the group.
    raw_gga_utc = [float(p.get("utc_seconds", np.nan)) for p in gga]
    raw_gga_lat = [float(p.get("lat_deg", np.nan)) for p in gga]
    raw_gga_lon = [float(p.get("lon_deg", np.nan)) for p in gga]
    raw_gga_alt = [float(p.get("altitude_m", np.nan)) for p in gga]
    raw_gga_diff = [float(p.get("fix_quality", np.nan)) for p in gga]
    raw_gga_hdop = [float(p.get("hdop", np.nan)) for p in gga]
    raw_gga_num_sats = [float(p.get("num_sats", np.nan)) for p in gga]

    # Raw VTG arrays.
    raw_vtg_course = [float(p.get("course_true_deg", np.nan)) for p in vtg]
    raw_vtg_speed = [float(p.get("speed_mps", np.nan)) for p in vtg]

    # RMC fallback: useful when GGA or VTG is missing in a group.
    # RMC does not provide HDOP, altitude, or satellite count, so those stay NaN.
    rmc_lat = [float(p.get("lat_deg", np.nan)) for p in rmc]
    rmc_lon = [float(p.get("lon_deg", np.nan)) for p in rmc]
    rmc_time = [float(p.get("utc_seconds", np.nan)) for p in rmc]
    rmc_course = [float(p.get("course_deg", np.nan)) for p in rmc]
    rmc_speed = [float(p.get("speed_mps", np.nan)) for p in rmc]

    # If this ensemble has no GGA but has RMC, use RMC as a continuity fallback.
    # That lets the downstream GPSData object still receive usable positions.
    if use_rmc_as_fallback and not raw_gga_lat and rmc_lat:
        raw_gga_utc = rmc_time[:1]
        raw_gga_lat = rmc_lat[:1]
        raw_gga_lon = rmc_lon[:1]
        raw_gga_alt = [np.nan]
        raw_gga_diff = [1.0]
        raw_gga_hdop = [np.nan]
        raw_gga_num_sats = [1.0]

    # If this ensemble has no VTG but has RMC, use RMC as a speed/course fallback.
    if use_rmc_as_fallback and not raw_vtg_speed and rmc_speed:
        raw_vtg_course = rmc_course[:1]
        raw_vtg_speed = rmc_speed[:1]

    # Ensemble-level values for GPSData external fields.
    # The default convention here is "last valid value in the ensemble".
    ext_gga_utc = _select_scalar(raw_gga_utc, method=gga_method)
    ext_gga_lat = _select_scalar(raw_gga_lat, method=gga_method)
    ext_gga_lon = _select_scalar(raw_gga_lon, method=gga_method)
    ext_gga_alt = _select_scalar(raw_gga_alt, method=gga_method)
    ext_gga_diff = _select_scalar(raw_gga_diff, method=gga_method)
    ext_gga_hdop = _select_scalar(raw_gga_hdop, method=gga_method)
    ext_gga_num_sats = _select_scalar(raw_gga_num_sats, method=gga_method)

    ext_vtg_course = _select_scalar(raw_vtg_course, method=vtg_method)
    ext_vtg_speed = _select_scalar(raw_vtg_speed, method=vtg_method)

    return {
        "raw_gga_utc": raw_gga_utc,
        "raw_gga_lat": raw_gga_lat,
        "raw_gga_lon": raw_gga_lon,
        "raw_gga_alt": raw_gga_alt,
        "raw_gga_diff": raw_gga_diff,
        "raw_gga_hdop": raw_gga_hdop,
        "raw_gga_num_sats": raw_gga_num_sats,
        "raw_vtg_course": raw_vtg_course,
        "raw_vtg_speed": raw_vtg_speed,
        "raw_gga_rmc_fallback_used": bool(use_rmc_as_fallback and not gga and bool(rmc_lat)),
        "raw_vtg_rmc_fallback_used": bool(use_rmc_as_fallback and not vtg and bool(rmc_speed)),
        "ext_gga_utc": ext_gga_utc,
        "ext_gga_lat": ext_gga_lat,
        "ext_gga_lon": ext_gga_lon,
        "ext_gga_alt": ext_gga_alt,
        "ext_gga_diff": ext_gga_diff,
        "ext_gga_hdop": ext_gga_hdop,
        "ext_gga_num_sats": ext_gga_num_sats,
        "ext_vtg_course": ext_vtg_course,
        "ext_vtg_speed": ext_vtg_speed,
        "sentence_count": len(parsed),
        "gga_count": len(gga),
        "vtg_count": len(vtg),
        "rmc_count": len(rmc),
    }


def build_gpsdata_from_nmea_groups(
    nmea_groups: Iterable[Any],
    gga_position_method: str = "End",
    gga_velocity_method: str = "Average",
    vtg_velocity_method: str = "Average",
    use_rmc_as_fallback: bool = True,
) -> tuple[GPSData, dict[str, Any]]:
    """
    Build a GPSData object from NMEA groups.

    Parameters
    ----------
    nmea_groups:
        Iterable of ensembles, each ensemble being an iterable of NMEA lines.
        Example:
            [
                ["$GPGGA,...", "$GPVTG,..."],
                ["$GPGGA,...", "$GPRMC,..."],
            ]
    gga_position_method:
        Ensemble choice for GGA external values ('End', 'First', 'Mean').
    gga_velocity_method:
        Same idea for the GGA values used at ensemble level.
    vtg_velocity_method:
        Ensemble choice for VTG external values ('End', 'First', 'Mean').
    use_rmc_as_fallback:
        If True, RMC can fill missing GGA/VTG values in an ensemble.

    Returns
    -------
    gps:
        Filled GPSData object.
    meta:
        Small metadata dictionary useful for debugging and validation.
    """
    groups = _normalize_ensemble_groups(nmea_groups)
    stream = NMEAStream()

    # Each row corresponds to one ensemble.
    raw_gga_utc_rows: list[list[float]] = []
    raw_gga_lat_rows: list[list[float]] = []
    raw_gga_lon_rows: list[list[float]] = []
    raw_gga_alt_rows: list[list[float]] = []
    raw_gga_diff_rows: list[list[float]] = []
    raw_gga_hdop_rows: list[list[float]] = []
    raw_gga_num_sats_rows: list[list[float]] = []
    raw_vtg_course_rows: list[list[float]] = []
    raw_vtg_speed_rows: list[list[float]] = []
    raw_vtg_mode_rows: list[list[Any]] = []

    # Ensemble-level values expected by GPSData.populate_data(...).
    ext_gga_utc: list[float] = []
    ext_gga_lat: list[float] = []
    ext_gga_lon: list[float] = []
    ext_gga_alt: list[float] = []
    ext_gga_diff: list[float] = []
    ext_gga_hdop: list[float] = []
    ext_gga_num_sats: list[float] = []
    ext_vtg_course: list[float] = []
    ext_vtg_speed: list[float] = []

    group_meta: list[dict[str, Any]] = []

    for group in groups:
        parsed = stream.extend(group)
        payload = _build_ensemble_payload(
            parsed=parsed,
            gga_method=gga_position_method,
            vtg_method=vtg_velocity_method,
            use_rmc_as_fallback=use_rmc_as_fallback,
        )

        raw_gga_utc_rows.append(payload["raw_gga_utc"])
        raw_gga_lat_rows.append(payload["raw_gga_lat"])
        raw_gga_lon_rows.append(payload["raw_gga_lon"])
        raw_gga_alt_rows.append(payload["raw_gga_alt"])
        raw_gga_diff_rows.append(payload["raw_gga_diff"])
        raw_gga_hdop_rows.append(payload["raw_gga_hdop"])
        raw_gga_num_sats_rows.append(payload["raw_gga_num_sats"])
        raw_vtg_course_rows.append(payload["raw_vtg_course"])
        raw_vtg_speed_rows.append(payload["raw_vtg_speed"])

        # VTG mode indicators are optional in many datasets.
        # Keep the shape compatible with raw_vtg_* if the dataset contains them.
        vtg_modes = [p.get("mode", "") for p in parsed if p.get("type") == "VTG"]
        raw_vtg_mode_rows.append(vtg_modes)

        ext_gga_utc.append(payload["ext_gga_utc"])
        ext_gga_lat.append(payload["ext_gga_lat"])
        ext_gga_lon.append(payload["ext_gga_lon"])
        ext_gga_alt.append(payload["ext_gga_alt"])
        ext_gga_diff.append(payload["ext_gga_diff"])
        ext_gga_hdop.append(payload["ext_gga_hdop"])
        ext_gga_num_sats.append(payload["ext_gga_num_sats"])
        ext_vtg_course.append(payload["ext_vtg_course"])
        ext_vtg_speed.append(payload["ext_vtg_speed"])

        group_meta.append(
            {
                "sentence_count": payload["sentence_count"],
                "gga_count": payload["gga_count"],
                "vtg_count": payload["vtg_count"],
                "rmc_count": payload["rmc_count"],
                "gga_fallback_used": payload["raw_gga_rmc_fallback_used"],
                "vtg_fallback_used": payload["raw_vtg_rmc_fallback_used"],
            }
        )

    # Convert ragged per-ensemble rows into matrices [ensemble, n].
    raw_gga_utc_arr = _pad_numeric_rows(raw_gga_utc_rows)
    raw_gga_lat_arr = _pad_numeric_rows(raw_gga_lat_rows)
    raw_gga_lon_arr = _pad_numeric_rows(raw_gga_lon_rows)
    raw_gga_alt_arr = _pad_numeric_rows(raw_gga_alt_rows)
    raw_gga_diff_arr = _pad_numeric_rows(raw_gga_diff_rows)
    raw_gga_hdop_arr = _pad_numeric_rows(raw_gga_hdop_rows)
    raw_gga_num_sats_arr = _pad_numeric_rows(raw_gga_num_sats_rows)
    raw_vtg_course_arr = _pad_numeric_rows(raw_vtg_course_rows)
    raw_vtg_speed_arr = _pad_numeric_rows(raw_vtg_speed_rows)

    # Empty/missing delta-time arrays are kept as NaN matrices for shape compatibility.
    raw_gga_delta_time_arr = np.full_like(raw_gga_lat_arr, np.nan, dtype=float)
    raw_vtg_delta_time_arr = np.full_like(raw_vtg_speed_arr, np.nan, dtype=float)

    # VTG mode indicator is character data.
    raw_vtg_mode_arr = _pad_object_rows(raw_vtg_mode_rows, fill_value="")

    gps = GPSData()
    gps.populate_data(
        raw_gga_utc=raw_gga_utc_arr,
        raw_gga_lat=raw_gga_lat_arr,
        raw_gga_lon=raw_gga_lon_arr,
        raw_gga_alt=raw_gga_alt_arr,
        raw_gga_diff=raw_gga_diff_arr,
        raw_gga_hdop=raw_gga_hdop_arr,
        raw_gga_num_sats=raw_gga_num_sats_arr,
        raw_gga_delta_time=raw_gga_delta_time_arr,
        raw_vtg_course=raw_vtg_course_arr,
        raw_vtg_speed=raw_vtg_speed_arr,
        raw_vtg_delta_time=raw_vtg_delta_time_arr,
        raw_vtg_mode_indicator=raw_vtg_mode_arr,
        ext_gga_utc=np.asarray(ext_gga_utc, dtype=float),
        ext_gga_lat=np.asarray(ext_gga_lat, dtype=float),
        ext_gga_lon=np.asarray(ext_gga_lon, dtype=float),
        ext_gga_alt=np.asarray(ext_gga_alt, dtype=float),
        ext_gga_diff=np.asarray(ext_gga_diff, dtype=float),
        ext_gga_hdop=np.asarray(ext_gga_hdop, dtype=float),
        ext_gga_num_sats=np.asarray(ext_gga_num_sats, dtype=float),
        ext_vtg_course=np.asarray(ext_vtg_course, dtype=float),
        ext_vtg_speed=np.asarray(ext_vtg_speed, dtype=float),
        gga_p_method=gga_position_method,
        gga_v_method=gga_velocity_method,
        vtg_method=vtg_velocity_method,
    )

    meta = {
        "n_ensembles": len(groups),
        "ensembles": group_meta,
        "gga_position_method": gga_position_method,
        "gga_velocity_method": gga_velocity_method,
        "vtg_velocity_method": vtg_velocity_method,
        "use_rmc_as_fallback": use_rmc_as_fallback,
    }

    return gps, meta


def build_gpsdata_from_single_group(
    nmea_lines: Iterable[str],
    gga_position_method: str = "End",
    gga_velocity_method: str = "Average",
    vtg_velocity_method: str = "Average",
    use_rmc_as_fallback: bool = True,
) -> tuple[GPSData, dict[str, Any]]:
    """
    Convenience wrapper for a single ensemble group.
    """
    return build_gpsdata_from_nmea_groups(
        nmea_groups=[list(nmea_lines)],
        gga_position_method=gga_position_method,
        gga_velocity_method=gga_velocity_method,
        vtg_velocity_method=vtg_velocity_method,
        use_rmc_as_fallback=use_rmc_as_fallback,
    )