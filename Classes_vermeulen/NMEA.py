from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import math
import re
import numpy as np

##
""" 
NMEA = National Marine Electronics Association.
Défini des standards de communication pour les instruments marins (transmission GPS, capn profondeurn vitesse etc.)
"""
##

_NMEA_RE = re.compile(
    r"^(?P<start>[$!])(?P<body>[^*]+)(?:\*(?P<checksum>[0-9A-Fa-f]{2}))?$"
)


@dataclass
class NMEASentence:
    raw: str
    talker: str
    sentence_type: str
    fields: list[str]
    checksum: Optional[str] = None
    checksum_ok: bool = True
    parsed: dict[str, Any] = field(default_factory=dict)


def nmea_checksum(text: str) -> int:
    value = 0
    for char in text:
        value ^= ord(char)
    return value


def validate_nmea_line(line: str) -> tuple[bool, str, Optional[str]]:
    line = line.strip()
    match = _NMEA_RE.match(line)
    if not match:
        return False, "", None

    body = match.group("body")
    expected = match.group("checksum")
    computed = f"{nmea_checksum(body):02X}"
    return expected is None or expected.upper() == computed, body, expected


def parse_lat_lon(value: str, hemisphere: str) -> float:
    if value in ("", None) or hemisphere in ("", None):
        return np.nan
    try:
        raw = float(value)
    except Exception:
        return np.nan

    degrees = int(raw // 100)
    minutes = raw - degrees * 100
    decimal = degrees + minutes / 60.0
    hemisphere = hemisphere.upper()
    if hemisphere in ("S", "W"):
        decimal = -decimal
    return float(decimal)


def parse_hhmmss(value: str) -> float:
    if value in ("", None):
        return np.nan
    try:
        raw = float(value)
    except Exception:
        return np.nan

    hours = int(raw // 10000)
    minutes = int((raw - hours * 10000) // 100)
    seconds = raw - hours * 10000 - minutes * 100
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_ddmmyy(value: str) -> tuple[int, int, int]:
    if value in ("", None) or len(value) != 6:
        return -1, -1, -1
    try:
        day = int(value[0:2])
        month = int(value[2:4])
        year = int(value[4:6])
        year = 2000 + year if year < 80 else 1900 + year
        return year, month, day
    except Exception:
        return -1, -1, -1


def parse_nmea_line(line: str) -> Optional[NMEASentence]:
    ok, body, checksum = validate_nmea_line(line)
    if not body:
        return None

    parts = body.split(",")
    if not parts or len(parts[0]) < 3:
        return None

    sentence_id = parts[0].upper()
    talker = sentence_id[:2]
    sentence_type = sentence_id[2:]
    fields = parts[1:]

    return NMEASentence(
        raw=line.strip(),
        talker=talker,
        sentence_type=sentence_type,
        fields=fields,
        checksum=checksum,
        checksum_ok=ok,
    )


def parse_gga(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields + [""] * max(0, 14 - len(sentence.fields))
    utc = parse_hhmmss(f[0])
    lat = parse_lat_lon(f[1], f[2])
    lon = parse_lat_lon(f[3], f[4])

    def _float(idx: int) -> float:
        try:
            return float(f[idx]) if f[idx] not in ("", None) else np.nan
        except Exception:
            return np.nan

    def _int(idx: int) -> int:
        try:
            return int(float(f[idx])) if f[idx] not in ("", None) else -1
        except Exception:
            return -1

    return {
        "type": "GGA",
        "utc_seconds": utc,
        "lat_deg": lat,
        "lon_deg": lon,
        "fix_quality": _int(5),
        "num_sats": _int(6),
        "hdop": _float(7),
        "altitude_m": _float(8),
        "geoid_sep_m": _float(10),
        "age_diff_sec": _float(12),
        "station_id": f[13] if len(f) > 13 else "",
    }

def parse_gll(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields + [""] * max(0, 7 - len(sentence.fields))
    lat = parse_lat_lon(f[0], f[1])
    lon = parse_lat_lon(f[2], f[3])
    utc = parse_hhmmss(f[4])

    return {
        "type": "GLL",
        "lat_deg": lat,
        "lon_deg": lon,
        "utc_seconds": utc,
        "status": f[5] if len(f) > 5 else "",
        "mode": f[6] if len(f) > 6 else "",
    }


def parse_gbs(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields + [""] * max(0, 8 - len(sentence.fields))

    def _float(idx: int) -> float:
        try:
            return float(f[idx]) if f[idx] not in ("", None) else np.nan
        except Exception:
            return np.nan

    def _int(idx: int) -> int:
        try:
            return int(float(f[idx])) if f[idx] not in ("", None) else -1
        except Exception:
            return -1

    return {
        "type": "GBS",
        "utc_seconds": parse_hhmmss(f[0]),
        "errlat": _float(1),
        "errlong": _float(2),
        "erralt": _float(3),
        "failID": _int(4),
        "pHPRfail": _float(5),
        "rbias": _float(6),
        "biasstd": _float(7),
        "flag": _int(8),
    }


def parse_rmc(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields + [""] * max(0, 12 - len(sentence.fields))
    utc = parse_hhmmss(f[0])
    lat = parse_lat_lon(f[2], f[3])
    lon = parse_lat_lon(f[4], f[5])

    speed_knots = np.nan
    course_deg = np.nan
    try:
        speed_knots = float(f[6]) if f[6] not in ("", None) else np.nan
    except Exception:
        pass
    try:
        course_deg = float(f[7]) if f[7] not in ("", None) else np.nan
    except Exception:
        pass

    year, month, day = parse_ddmmyy(f[8])

    return {
        "type": "RMC",
        "utc_seconds": utc,
        "status": f[1] if len(f) > 1 else "",
        "lat_deg": lat,
        "lon_deg": lon,
        "speed_knots": speed_knots,
        "speed_mps": speed_knots * 0.514444 if np.isfinite(speed_knots) else np.nan,
        "course_deg": course_deg,
        "date_ymd": (year, month, day),
        "mode": f[11] if len(f) > 11 else "",
    }


def parse_vtg(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields + [""] * max(0, 8 - len(sentence.fields))
    course_true = np.nan
    speed_knots = np.nan
    speed_kph = np.nan
    try:
        course_true = float(f[0]) if f[0] not in ("", None) else np.nan
    except Exception:
        pass
    try:
        speed_knots = float(f[4]) if f[4] not in ("", None) else np.nan
    except Exception:
        pass
    try:
        speed_kph = float(f[6]) if f[6] not in ("", None) else np.nan
    except Exception:
        pass

    return {
        "type": "VTG",
        "course_true_deg": course_true,
        "speed_knots": speed_knots,
        "speed_mps": speed_knots * 0.514444 if np.isfinite(speed_knots) else np.nan,
        "speed_kph": speed_kph,
        "mode": f[7] if len(f) > 7 else "",
    }

def parse_gsa(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields + [""] * max(0, 17 - len(sentence.fields))

    def _float(idx: int) -> float:
        try:
            return float(f[idx]) if f[idx] not in ("", None) else np.nan
        except Exception:
            return np.nan

    def _int(idx: int) -> int:
        try:
            return int(float(f[idx])) if f[idx] not in ("", None) else -1
        except Exception:
            return -1

    prn = np.array([_int(i) for i in range(2, 14)], dtype=float)

    return {
        "type": "GSA",
        "modesel": (f[0] or "").upper() == "A",
        "mode": _int(1),
        "prn": prn,
        "pdop": _float(14),
        "hdop": _float(15),
        "vdop": _float(16),
    }


def parse_hdt(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields + [""] * max(0, 2 - len(sentence.fields))
    try:
        heading = float(f[0]) if f[0] not in ("", None) else np.nan
    except Exception:
        heading = np.nan
    return {
        "type": "HDT",
        "heading_true_deg": heading,
    }



def parse_dbt(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields + [""] * max(0, 3 - len(sentence.fields))

    def _float(idx: int) -> float:
        try:
            return float(f[idx]) if f[idx] not in ("", None) else np.nan
        except Exception:
            return np.nan

    return {
        "type": "DBT",
        "depthf": _float(0),
        "depthM": _float(1),
        "depthF": _float(2),
    }


parse_dbs = parse_dbt


def parse_hpr(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields
    values = []
    for idx in range(min(3, len(f))):
        try:
            values.append(float(f[idx]) if f[idx] not in ("", None) else np.nan)
        except Exception:
            values.append(np.nan)
    while len(values) < 3:
        values.append(np.nan)
    return {
        "type": "HPR",
        "heading_deg": values[0],
        "pitch_deg": values[1],
        "roll_deg": values[2],
    }


def parse_zda(sentence: NMEASentence) -> dict[str, Any]:
    f = sentence.fields + [""] * max(0, 5 - len(sentence.fields))
    utc = parse_hhmmss(f[0])
    try:
        day = int(f[1]) if f[1] not in ("", None) else -1
        month = int(f[2]) if f[2] not in ("", None) else -1
        year = int(f[3]) if f[3] not in ("", None) else -1
    except Exception:
        day, month, year = -1, -1, -1
    try:
        hour_offset = int(f[4]) if f[4] not in ("", None) else 0
    except Exception:
        hour_offset = 0
    try:
        minute_offset = int(f[5]) if f[5] not in ("", None) else 0
    except Exception:
        minute_offset = 0
    return {
        "type": "ZDA",
        "utc_seconds": utc,
        "date_ymd": (year, month, day),
        "tz_offset_h": hour_offset,
        "tz_offset_m": minute_offset,
    }




class NMEAStream:
    def __init__(self) -> None:
        self.sentences: list[NMEASentence] = []
        self.parsed: list[dict[str, Any]] = []

    def append(self, line: str) -> Optional[dict[str, Any]]:
        sentence = parse_nmea_line(line)
        if sentence is None:
            return None

        self.sentences.append(sentence)

        if sentence.sentence_type == "GGA":
            parsed = parse_gga(sentence)
        elif sentence.sentence_type == "GLL":
            parsed = parse_gll(sentence)
        elif sentence.sentence_type == "GBS":
            parsed = parse_gbs(sentence)
        elif sentence.sentence_type == "GSA":
            parsed = parse_gsa(sentence)
        elif sentence.sentence_type == "RMC":
            parsed = parse_rmc(sentence)
        elif sentence.sentence_type == "VTG":
            parsed = parse_vtg(sentence)
        elif sentence.sentence_type == "HDT":
            parsed = parse_hdt(sentence)
        elif sentence.sentence_type in ("DBT", "DBS"):
            parsed = parse_dbt(sentence)
        elif sentence.sentence_type == "HPR":
            parsed = parse_hpr(sentence)
        elif sentence.sentence_type == "ZDA":
            parsed = parse_zda(sentence)
        else:
            parsed = {
                "type": sentence.sentence_type,
                "fields": sentence.fields,
            }

        sentence.parsed = parsed
        self.parsed.append(parsed)
        return parsed

    def extend(self, lines: Iterable[str]) -> list[dict[str, Any]]:
        out = []
        for line in lines:
            parsed = self.append(line)
            if parsed is not None:
                out.append(parsed)
        return out

    def by_type(self, sentence_type: str) -> list[dict[str, Any]]:
        sentence_type = sentence_type.upper()
        return [item for item in self.parsed if item.get("type", "").upper() == sentence_type]

    def raw_by_type(self, sentence_type: str) -> list[NMEASentence]:
        sentence_type = sentence_type.upper()
        return [item for item in self.sentences if item.sentence_type.upper() == sentence_type]

    def latest(self, sentence_type: str) -> Optional[dict[str, Any]]:
        items = self.by_type(sentence_type)
        return items[-1] if items else None
    



#