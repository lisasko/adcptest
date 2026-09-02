from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
import re

import numpy as np

from .NMEA import NMEAStream
from .nmea_to_gpsdata import build_gpsdata_from_nmea_groups


_TEXT_EXTENSIONS = {".txt", ".nmea", ".nme", ".log", ".dat"}


def _ensure_paths(filenames: Any) -> list[Path]:
    if filenames is None:
        return []
    if isinstance(filenames, (str, Path)):
        return [Path(filenames)]
    return [Path(item) for item in filenames]


def _read_text_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def _as_groups(source: Any) -> list[list[str]]:
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        return [_read_text_lines(Path(source))]

    items = list(source)
    if not items:
        return []

    if all(isinstance(item, (str, bytes, Path)) for item in items):
        return [_read_text_lines(Path(item)) for item in items]

    return [[str(line) for line in group] for group in items]


def _split_fields(line: str) -> list[str]:
    return [field for field in re.split(r"[,\t; ]+", line.strip()) if field]


def _maybe_float(token: str) -> Any:
    if token in ("", None):
        return np.nan
    try:
        return float(token)
    except Exception:
        return token


def _read_generic_table_files(filenames: Any) -> dict[str, np.ndarray]:
    """
    Read one or more delimited text files with a header row.

    The parser is intentionally tolerant: it accepts spaces, commas, tabs,
    or semicolons as separators and keeps unknown columns as-is.
    """
    tables: list[dict[str, np.ndarray]] = []

    for path in _ensure_paths(filenames):
        lines = _read_text_lines(path)
        if not lines:
            continue

        header = _split_fields(lines[0].lstrip("#"))
        if not header:
            continue

        rows = [_split_fields(line) for line in lines[1:] if _split_fields(line)]
        if not rows:
            continue

        columns: dict[str, list[Any]] = {name: [] for name in header}
        for row in rows:
            for idx, name in enumerate(header):
                token = row[idx] if idx < len(row) else ""
                columns[name].append(_maybe_float(token))

        tables.append(
            {name: np.asarray(values) for name, values in columns.items()}
        )

    if not tables:
        return {}

    if len(tables) == 1:
        return tables[0]

    merged: dict[str, list[np.ndarray]] = {}
    for table in tables:
        for key, value in table.items():
            merged.setdefault(key, []).append(np.asarray(value).reshape(-1))

    return {key: np.concatenate(chunks) for key, chunks in merged.items()}


def readNMEA(filenames: Any) -> list[dict[str, Any]]:
    """
    Read one or more NMEA text files and parse them with NMEAStream.
    Returns one dict per file.
    """
    out: list[dict[str, Any]] = []
    for path in _ensure_paths(filenames):
        lines = _read_text_lines(path)
        stream = NMEAStream()
        stream.extend(lines)
        out.append(
            {
                "file": str(path),
                "lines": lines,
                "sentences": stream.sentences,
                "parsed": stream.parsed,
            }
        )
    return out


def readNMEAADCP(
    inadcp: Any,
    nmeafilename: Any,
    gga_position_method: str = "End",
    gga_velocity_method: str = "Average",
    vtg_velocity_method: str = "Average",
    use_rmc_as_fallback: bool = True,
):
    """
    Build GPSData from NMEA groups.
    This is the Python equivalent of the MATLAB NMEA->ADCP alignment layer,
    but it expects the NMEA source to already be grouped by ensemble/file.
    """
    groups = _as_groups(nmeafilename)
    gps, meta = build_gpsdata_from_nmea_groups(
        nmea_groups=groups,
        gga_position_method=gga_position_method,
        gga_velocity_method=gga_velocity_method,
        vtg_velocity_method=vtg_velocity_method,
        use_rmc_as_fallback=use_rmc_as_fallback,
    )
    return gps, meta


def readViseaExtern(adcp: Any, filenames: Any, rfiles: Any = None, **kwargs):
    """
    Read VISEA extern text files into a dictionary of arrays.
    This is a pragmatic text parser; it returns keys usable by LatLonVisea
    and ProjectedCoordinatesFromViseaExtern when those fields are present.
    """
    tables: list[dict[str, Any]] = []
    for path in _ensure_paths(filenames):
        lines = _read_text_lines(path)
        if not lines:
            continue

        header = _split_fields(lines[0].lstrip("#"))
        if not header:
            continue

        rows = [_split_fields(line) for line in lines[1:] if _split_fields(line)]
        if not rows:
            continue

        columns: dict[str, list[Any]] = {name: [] for name in header}
        for row in rows:
            for idx, name in enumerate(header):
                token = row[idx] if idx < len(row) else ""
                columns[name].append(_maybe_float(token))

        tables.append(
            {
                "file": str(path),
                "columns": {name: np.asarray(values) for name, values in columns.items()},
                "raw_lines": lines,
            }
        )

    if not tables:
        return {}

    if len(tables) == 1:
        return tables[0]["columns"]

    merged: dict[str, list[np.ndarray]] = {}
    for table in tables:
        for key, value in table["columns"].items():
            merged.setdefault(key, []).append(np.asarray(value).reshape(-1))

    return {key: np.concatenate(chunks) for key, chunks in merged.items()}


def readTfiles(adcp: Any, filenames: Any, rfiles: Any = None, **kwargs):
    """
    Read SonTek Tfiles-style exports and return a dictionary under the tFiles key.

    The returned structure is compatible with LatLonTfiles:
    adcp._raw["tFiles"]["lat"] and adcp._raw["tFiles"]["long"].
    """
    data = _read_generic_table_files(filenames)
    if not data:
        return {}

    normalized = {}
    for key, value in data.items():
        lower = key.strip().lower()
        if lower in ("latitude", "lat", "latitudeseconds"):
            normalized["lat"] = value
        elif lower in ("longitude", "lon", "long", "longitudeseconds"):
            normalized["long"] = value
        else:
            normalized[key] = value

    return {"tFiles": normalized}



def readRDENS(adcp: Any, filenames: Any, rfiles: Any = None, **kwargs):
    """
    Read RDENS-style exports and return a dictionary under the RDENS key.

    The exact column names can vary between instruments and exports, so the
    function keeps the parsed columns and leaves interpretation to the caller.
    """
    data = _read_generic_table_files(filenames)
    return {"RDENS": data} if data else {}




def readViseaLogFiles(inadcp: Any, fname: Any):
    """
    Read VISEA log files and parse NMEA content with NMEAStream.
    """
    out: dict[str, Any] = {}
    for path in _ensure_paths(fname):
        lines = _read_text_lines(path)
        stream = NMEAStream()
        parsed = stream.extend(lines)
        out[str(path)] = {
            "lines": lines,
            "sentences": stream.sentences,
            "parsed": parsed,
        }
    return out


def readDeployment(DepName: str, path: str | Path = ""):
    """
    Text-oriented deployment wrapper.
    This covers NMEA text files and VISEA extern/log files.
    Raw PD0/ADCP binary reading still needs a Python readADCP equivalent.
    """
    base = Path(path) if path else Path(".")
    candidates = sorted(base.glob(f"{DepName}*"))

    text_files = [
        item for item in candidates
        if item.is_file() and item.suffix.lower() in _TEXT_EXTENSIONS
    ]

    out: dict[str, Any] = {
        "deployment": DepName,
        "path": str(base),
        "files": [str(item) for item in text_files],
    }

    nmea_files = [
        item for item in text_files
        if item.suffix.lower() in {".txt", ".nmea", ".nme", ".log"}
    ]
    if nmea_files:
        out["NMEA"] = readNMEA(nmea_files)
        gpsdata, meta = readNMEAADCP(None, nmea_files)
        out["GPSData"] = gpsdata
        out["NMEA_meta"] = meta

    visea_extern_files = [item for item in text_files if "extern" in item.stem.lower()]
    if visea_extern_files:
        out["VISEA_Extern"] = readViseaExtern(None, visea_extern_files)

    visea_log_files = [item for item in text_files if item.suffix.lower() == ".log"]
    if visea_log_files:
        out["VISEA_log"] = readViseaLogFiles(None, visea_log_files)

    tfiles = [
        item for item in text_files
        if "tfile" in item.stem.lower() or "tfiles" in item.stem.lower()
    ]
    if tfiles:
        out["tFiles"] = readTfiles(None, tfiles)

    rdens_files = [
        item for item in text_files
        if "rdens" in item.stem.lower()
    ]
    if rdens_files:
        out["RDENS"] = readRDENS(None, rdens_files)

    return out


read_nmea_files = readNMEA
read_nmea_adcp = readNMEAADCP
read_visea_extern = readViseaExtern
read_visea_log_files = readViseaLogFiles
read_deployment = readDeployment