from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from Classes.Measurement import Measurement as _BaseMeasurement

from Classes_vermeulen.nmea_file_io import (
    readDeployment as _readDeployment,
    readNMEAADCP as _readNMEAADCP,
    readViseaExtern as _readViseaExtern,
    readViseaLogFiles as _readViseaLogFiles,
    readTfiles as _readTfiles,
    readRDENS as _readRDENS,
)

"""
Measurement_new is a new wrapper around the original QRev/QRevInt Measurement class.
It adds support for automatically finding and attaching optional side data files (VISEA extern/log, Tfiles, RDENS) and provides static methods for reading these files in a MATLAB-style manner.
The original QRev/QRevInt behavior is preserved, and the new features are additive.

Measurement is the class that holds all measurement details.
"""


def _ensure_paths(source: Any) -> list[Path]:
    """ Normalise une entrée fichier en liste de chemins Path."""

    if source is None:
        return []
    if isinstance(source, (str, Path)):
        return [Path(source)]
    return [Path(item) for item in source]


def _read_lines(path: Path) -> list[str]:
    """ Lit un fichier texte en ignorant les lignes vides et les erreurs d'encodage."""

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def _split_fields(line: str) -> list[str]:
    """Découpe une ligne de texte en champs, accepte plusieurs séparateurs."""

    return [field for field in line.replace(";", " ").replace("\t", " ").replace(",", " ").split() if field]


def _normalize_key(key: str) -> str:
    """ Harmonise les noms de colonnes vers des clés internes communes. """

    lowered = key.strip().lower()
    aliases = {
        "latitude": "lat",
        "lat": "lat",
        "latitudeseconds": "lat",
        "longitude": "long",
        "lon": "long",
        "long": "long",
        "longitudeseconds": "long",
        "validity": "valid",
        "valid": "valid",
        "status": "valid",
        "quality": "quality",
        "q": "quality",
    }
    return aliases.get(lowered, lowered)




class Measurement(_BaseMeasurement):
    """
    Wrapper around the qrev Measurement class.

    This class keeps the original QRev/QRevInt behavior and adds:
    - automatic attachment of optional VISEA / Tfiles / RDENS side data
    - MATLAB-style helper readers exposed from one place
    - raw-block storage under self._raw for the Vermeulen providers
    """

    def __init__(
        self,
        in_file,
        source,
        proc_type="QRev",
        checked=False,
        run_oursin=False,
        use_weighted=False,
        auto_attach_support_files=True,
    ):
        super().__init__(
            in_file=in_file,
            source=source,
            proc_type=proc_type,
            checked=checked,
            run_oursin=run_oursin,
            use_weighted=use_weighted,
        )

        self._raw: dict[str, Any] = {}
        self.support_files: dict[str, Any] = {}
        self._support_loaded = False

        if auto_attach_support_files:
            self.attach_support_files(in_file)

    @staticmethod
    def readViseaExtern(adcp: Any, filenames: Any, rfiles: Any = None, **kwargs):
        return _readViseaExtern(adcp, filenames, rfiles=rfiles, **kwargs)

    @staticmethod
    def readViseaLogFiles(inadcp: Any, fname: Any):
        return _readViseaLogFiles(inadcp, fname)

    @staticmethod
    def readNMEAADCP(
        inadcp: Any,
        nmeafilename: Any,
        gga_position_method: str = "End",
        gga_velocity_method: str = "Average",
        vtg_velocity_method: str = "Average",
        use_rmc_as_fallback: bool = True,
    ):
        return _readNMEAADCP(
            inadcp,
            nmeafilename,
            gga_position_method=gga_position_method,
            gga_velocity_method=gga_velocity_method,
            vtg_velocity_method=vtg_velocity_method,
            use_rmc_as_fallback=use_rmc_as_fallback,
        )

    @staticmethod
    def readDeployment(DepName: str, path: str | Path = ""):
        return _readDeployment(DepName, path)
    
        
    @staticmethod
    def readTfiles(adcp: Any, filenames: Any, rfiles: Any = None, **kwargs):
        return _readTfiles(adcp, filenames, rfiles=rfiles, **kwargs)

    @staticmethod
    def readRDENS(adcp: Any, filenames: Any, rfiles: Any = None, **kwargs):
        return _readRDENS(adcp, filenames, rfiles=rfiles, **kwargs)


    def attach_support_files(self, in_file: Any) -> None:
        """ Rattache les fichiers externes sans modifier le cœur Measurement de qrev."""

        roots = self._candidate_roots(in_file)
        self.support_files = {
            "VISEA_Extern": [],
            "VISEA_log": [],
            "tFiles": [],
            "RDENS": [],
        }

        for root in roots:
            if not root.exists() or not root.is_dir():
                continue

            extern_files = sorted(
                item for item in root.rglob("*") if item.is_file() and "extern" in item.stem.lower()
            )
            if extern_files:
                self.support_files["VISEA_Extern"].extend(str(item) for item in extern_files)
                self._raw["VISEA_Extern"] = self.readViseaExtern(None, extern_files)

            log_files = sorted(
                item for item in root.rglob("*") if item.is_file() and item.suffix.lower() == ".log"
            )
            if log_files:
                self.support_files["VISEA_log"].extend(str(item) for item in log_files)
                self._raw["VISEA_log"] = self.readViseaLogFiles(None, log_files)

            tfile_files = sorted(
                item
                for item in root.rglob("*")
                if item.is_file() and ("tfile" in item.stem.lower() or "tfiles" in item.stem.lower())
            )
            if tfile_files:
                self.support_files["tFiles"].extend(str(item) for item in tfile_files)
                tfiles = self.readTfiles(None, tfile_files)
                if "tFiles" in tfiles:
                    self._raw["tFiles"] = tfiles["tFiles"]

            rdens_files = sorted(
                item for item in root.rglob("*") if item.is_file() and "rdens" in item.stem.lower()
            )
            if rdens_files:
                self.support_files["RDENS"].extend(str(item) for item in rdens_files)
                rdens = self.readRDENS(None, rdens_files)
                if "RDENS" in rdens:
                    self._raw["RDENS"] = rdens["RDENS"]

        self._support_loaded = True

    @staticmethod
    def _candidate_roots(in_file: Any) -> list[Path]:
        roots: list[Path] = []
        for item in _ensure_paths(in_file):
            if item.is_dir():
                roots.append(item)
            else:
                roots.append(item.parent)
        unique_roots: list[Path] = []
        for root in roots:
            if root not in unique_roots:
                unique_roots.append(root)
        return unique_roots


Measurement_new = Measurement