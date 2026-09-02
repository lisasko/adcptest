from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PistonTransducerSimple:
    """
    Minimal equivalent of acoustics.PistonTransducer used by svADCP.m.
    """
    attenuation: float = 0.0

    def near_field_correction(self, r: np.ndarray) -> np.ndarray:
        # Same default behavior as a neutral correction.
        return np.ones_like(np.asarray(r, dtype=float), dtype=float)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


def _as_array(value: Any, dtype=float) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _beam_angle_deg(inadcp: Any) -> np.ndarray:
    # MATLAB uses get_beam_angle(inadcp)
    b = _get(inadcp, "beam_angle_deg", None)
    if b is None:
        b = _get(inadcp, "beam_angle", None)
    if b is None:
        b = 20.0
    arr = np.asarray(b, dtype=float).reshape(-1)
    if arr.size == 0:
        arr = np.array([20.0], dtype=float)
    return arr


def _sysconf_code(inadcp: Any) -> str:
    sc = _get(inadcp, "sysconf", "")
    if sc is None:
        return ""
    if isinstance(sc, bytes):
        sc = sc.decode(errors="ignore")
    if isinstance(sc, str):
        return sc[:3]

    arr = np.asarray(sc)
    if arr.size == 0:
        return ""
    first = arr.reshape(-1)[0]
    if isinstance(first, (bytes, np.bytes_)):
        first = first.decode(errors="ignore")
    return str(first)[:3]


def _coerce_ens_range(ens_range: Any, nens_total: int) -> np.ndarray:
    if ens_range is None:
        return np.arange(nens_total, dtype=int)

    arr = np.asarray(ens_range, dtype=int).reshape(-1)
    if arr.size != 2:
        raise ValueError("ens_range must be 2 elements: [start, end] (1-based, inclusive)")
    if arr[0] == 0 and arr[1] == 0:
        return np.arange(nens_total, dtype=int)

    # MATLAB is 1-based inclusive.
    start = max(int(arr[0]) - 1, 0)
    stop = min(int(arr[1]) - 1, nens_total - 1)
    if stop < start:
        return np.array([], dtype=int)
    return np.arange(start, stop + 1, dtype=int)


def _select_echo(ECHO: np.ndarray, ens_idx: np.ndarray, nens_total: int) -> np.ndarray:
    # Expected MATLAB shape: (nbins, nens, nbeams)
    arr = np.asarray(ECHO, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("ECHO must be 2D or 3D")

    if arr.shape[1] == nens_total:
        return arr[:, ens_idx, :]

    # Common alternate storage: (nens, nbins, nbeams)
    if arr.shape[0] == nens_total:
        arr2 = np.transpose(arr, (1, 0, 2))
        return arr2[:, ens_idx, :]

    raise ValueError("Cannot align ECHO array with ensemble dimension")


def _select_adc(ADC: np.ndarray, ens_idx: np.ndarray, nens_total: int) -> np.ndarray:
    arr = np.asarray(ADC, dtype=float)
    if arr.ndim != 2:
        raise ValueError("ADC must be 2D")
    if arr.shape[0] == nens_total:
        return arr[ens_idx, :]
    if arr.shape[1] == nens_total:
        return arr[:, ens_idx].T
    raise ValueError("Cannot align ADC with ensemble dimension")


def _take_file_indexed(series: Any, file_number: Any, ens_idx: np.ndarray, nens_total: int) -> np.ndarray:
    vals = np.asarray(series, dtype=float).reshape(-1)
    if vals.size == 0:
        return np.full((ens_idx.size,), np.nan, dtype=float)

    fn = np.asarray(file_number).reshape(-1)
    if fn.size == nens_total:
        sel = fn[ens_idx].astype(int)
        # MATLAB FileNumber is typically 1-based.
        sel0 = np.clip(sel - 1, 0, vals.size - 1)
        return vals[sel0]

    # fallback: already per ensemble
    if vals.size == nens_total:
        return vals[ens_idx]

    # scalar fallback
    if vals.size == 1:
        return np.full((ens_idx.size,), float(vals[0]), dtype=float)

    # pad/truncate fallback
    out = np.full((ens_idx.size,), np.nan, dtype=float)
    ncopy = min(ens_idx.size, vals.size)
    out[:ncopy] = vals[:ncopy]
    return out


def svADCP(
    inadcp: Any,
    pt: Any,
    ens_range: Any = None,
    *,
    Kc: float = 0.43,
    Er: float = 40.0,
    T_Offset: float = -0.35,
    C: float | None = None,
):
    """
    Python translation of MATLAB svADCP.m

    Returns
    -------
    SV, Kc, C
    SV shape is typically (nbins, n_ens_selected, nbeams)
    """
    if inadcp is None:
        raise ValueError("inadcp is required")
    if pt is None:
        raise ValueError("pt is required and must expose attenuation + near_field_correction")

    ECHO = _get(inadcp, "ECHO", None)
    if ECHO is None:
        raise ValueError("inadcp.ECHO is required")
    ADC = _get(inadcp, "ADC", None)
    if ADC is None:
        raise ValueError("inadcp.ADC is required")

    ensnum = _get(inadcp, "ensnum", None)
    if ensnum is not None:
        nens_total = int(np.asarray(ensnum).size)
    else:
        e = np.asarray(ECHO)
        # assume MATLAB orientation first: (nbins, nens, nbeams)
        nens_total = int(e.shape[1] if e.ndim >= 2 else 0)

    if nens_total <= 0:
        raise ValueError("Unable to determine number of ensembles")

    ens_idx = _coerce_ens_range(ens_range, nens_total)
    if ens_idx.size == 0:
        return np.empty((0, 0, 0), dtype=float), float(Kc), (np.nan if C is None else float(C))

    nbins = int(np.nanmax(np.asarray(_get(inadcp, "nbins", np.array([np.asarray(ECHO).shape[0]])), dtype=float)))
    if nbins <= 0:
        nbins = int(np.asarray(ECHO).shape[0])

    # Step 1: instrument constants by sysconf code
    code = _sysconf_code(inadcp)
    current_fact = 11451.0 / 1e6
    if code == "000":
        volt_fact = 2092719.0 / 1e6
        current_fact = 43838.0 / 1e6
        Ctmp = -159.1
    elif code == "100":
        volt_fact = 592157.0 / 1e6
        Ctmp = -153.0
    elif code == "010":
        volt_fact = 592157.0 / 1e6
        Ctmp = -143.0
    elif code == "110":
        volt_fact = 380667.0 / 1e6
        Ctmp = -139.3
    elif code == "001":
        volt_fact = 253765.0 / 1e6
        Ctmp = -129.1
    elif code == "101":
        volt_fact = 253765.0 / 1e6
        Ctmp = np.nan
    else:
        volt_fact = 592157.0 / 1e6
        Ctmp = np.nan

    if C is None:
        C = Ctmp
    C = float(C)

    file_number = _get(inadcp, "FileNumber", np.arange(1, nens_total + 1))

    # Step 3: selected ADCP parameters
    blnk = _get(inadcp, "blnk", np.nan)
    lng = _get(inadcp, "lngthtranspulse", np.nan)
    binsize = _get(inadcp, "binsize", np.nan)

    B = _take_file_indexed(blnk, file_number, ens_idx, nens_total) / 100.0
    L = _take_file_indexed(lng, file_number, ens_idx, nens_total) / 100.0
    D = _take_file_indexed(binsize, file_number, ens_idx, nens_total) / 100.0

    adc_sel = _select_adc(np.asarray(ADC, dtype=float), ens_idx, nens_total)
    if adc_sel.shape[1] < 6:
        raise ValueError("inadcp.ADC must have at least 6 columns")

    curr = current_fact * adc_sel[:, 0]
    volt = volt_fact * adc_sel[:, 1]
    t_cnts = adc_sel[:, 5] * 256.0

    DC_COEF = 9.82697464e1
    FIRST_COEF = -5.86074151382e-3
    SECOND_COEF = 1.60433886495e-7
    THIRD_COEF = -2.32924716883e-12

    Tx = T_Offset + ((THIRD_COEF * t_cnts + SECOND_COEF) * t_cnts + FIRST_COEF) * t_cnts + DC_COEF

    # Step 5: transmit power
    power = volt * curr
    with np.errstate(invalid="ignore", divide="ignore"):
        PDBW = 10.0 * np.log10(power)
        LDBM = 10.0 * np.log10(L)

    # Step 6: sonar equation geometric terms
    ba = _beam_angle_deg(inadcp)
    if ba.size == 1:
        ba = np.full((ens_idx.size,), float(ba[0]), dtype=float)
    elif ba.size >= nens_total:
        ba = ba[ens_idx]
    else:
        ba = np.pad(ba, (0, max(0, ens_idx.size - ba.size)), constant_values=np.nan)[: ens_idx.size]

    cos_b = np.cos(np.deg2rad(ba))
    alpha = float(getattr(pt, "attenuation", 0.0))

    k = np.arange(0, nbins, dtype=float)[:, None]      # MATLAB (1:nbins)-1
    kk = np.arange(1, nbins + 1, dtype=float)[:, None] # MATLAB (1:nbins)

    B2 = B[None, :]
    L2 = L[None, :]
    D2 = D[None, :]
    cos2 = cos_b[None, :]

    R = (B2 + (D2 + L2) / 2.0 + k * D2 + D2 / 4.0) / cos2

    alpha_n = 2.0 * alpha * D / cos_b
    two_alpha_R = 2.0 * alpha * B2 / cos2 + kk * alpha_n[None, :]

    nfc = np.asarray(pt.near_field_correction(R), dtype=float)
    if nfc.shape != R.shape:
        nfc = np.broadcast_to(nfc, R.shape)

    # Step 7: backscatter
    echo_sel = _select_echo(np.asarray(ECHO, dtype=float), ens_idx, nens_total)

    geom = (
        C
        + 10.0 * np.log10((Tx[None, :] + 273.16) * (R ** 2) * (nfc ** 2))
        - LDBM[None, :]
        - PDBW[None, :]
        + two_alpha_R
    )

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        inner = np.power(10.0, Kc * (echo_sel - Er) / 10.0) - 1.0
        inner = np.where(inner > 0.0, inner, np.nan)
        sv_term = 10.0 * np.log10(inner)

    SV = geom[:, :, None] + sv_term
    return SV, float(Kc), float(C)


# Optional snake_case alias
sv_adcp = svADCP