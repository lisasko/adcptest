from __future__ import annotations
from typing import Any, Iterable, Sequence, TYPE_CHECKING, List, Union, Optional
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
import utm
from dataclasses import dataclass
from datetime import timezone as _timezone
from enum import IntEnum
from abc import ABC, abstractmethod
from scipy.fft import dst
from pyproj import Transformer

if TYPE_CHECKING:
    from .ADCPHorizontalPosition import ADCPHorizontalPosition, ADCPFixedHorizontalPosition
    from .ADCPVerticalPosition import ADCPVerticalPosition, ADCPFixedVerticalPosition, ADCPVerticalPositionFromWaterLevel
    from .ShipVelocity import ShipVelocityFromBT, ShipVelocityFromGPS, ShipVelocityProvider

from .ADCP_Type import ADCPType
from .CoordinateSystem import CoordinateSystem  
from .WaterLevel import ConstantWaterLevel, VaryingWaterLevel, WaterLevel
from .Filter import Filter
from .EnsembleFilter import EnsembleFilter
from .LatLon import LatLonToUTM, ProjectedCoordinatesFromViseaExtern
from qrevint_21_03.Classes.TransformationMatrix import TransformationMatrix
from .InstrumentMatrix import (
    InstrumentMatrixFromCalibration,
    InstrumentMatrixFromBAngle,
    InstrumentMatrixFromPS3,
    InstrumentMatrixProvider,
);

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from get_utm_zone import get_utm_zone


##### A DEPLACER 
@dataclass
# Classe stockant les propriétés de l'eau 
class WaterState:
    temperature_c: float | np.ndarray = np.nan
    salinity_psu: float | np.ndarray = np.nan
    density_kgm3: float = 1000.0

# Classe stockant les paramètres du transducteur 
class SimpleTransducer:
    def __init__(self) -> None:
        self.frequency_hz = np.nan
        self.attenuation = 0.0
        self.depth = 0.0

    def near_field_correction(self, r: np.ndarray) -> np.ndarray:
        return np.ones_like(r, dtype=float)


""" 
    Wrapper class for adcp structures.

        This class provides an interface for working with ADCP (Acoustic Doppler Current Profiler) data.
    It includes methods for accessing raw data, computing velocities, and visualizing results.

    Attributes:
        raw: Raw ADCP data (dict or struct-like object).
        filters: List of Filter objects for filtering data.
        timezone: Timezone of the data (str).
        type: Type of ADCP (ADCPType enum).
        transformation_matrix_source: List of InstrumentMatrixProvider objects.
        temperature_offset: Temperature offset for Workhorse ADCPs (float).
        noise_level: Background noise level (float).
        transducer: Transducer object (SimpleTransducer).
        water: Water properties (WaterState).
        horizontal_position_provider: Provider for horizontal position.
        vertical_position_provider: Provider for vertical position.
        heading_provider: Provider for heading data.
        _horizontal_position: Internal horizontal position (2, nensembles).
        _vertical_position: Internal vertical position (1, nensembles).
        depth_cell_position: Depth cell positions (ncells, nensembles, nbeams, 3).
        water_velocity: Water velocity data (ncells, nensembles, nbeams).
        xform: Transformation matrix (1, nensembles, 4, 4).
        bed_position: Bed position data (1, nensembles, 1, 3).
        beam_angle_deg: Beam angle in degrees (float).
        water_level: Water level data (nensembles,).
        n_ensembles: Number of ensembles (int).

"""

class ADCP:

    def __init__(self, *args, src: Any = None) -> None:
        self._raw = None
        self.filters: List[Filter] = [Filter()]
        self.timezone: str = ""
        self.transformation_matrix_source: list[Any] = []
        self.type: ADCPType = ADCPType.Unknown
        self.temperature_offset: float = -0.35 ## Variable chosie manuellement 
        self.noise_level: float = -40.0 ## Variable chosie manuellement
        self.transducer: SimpleTransducer = SimpleTransducer()
        self.water: WaterState = WaterState()

        from .ADCPHorizontalPosition import ADCPFixedHorizontalPosition
        from .ADCPVerticalPosition import ADCPFixedVerticalPosition
        from .ADCPVerticalPosition import ADCPVerticalPositionFromWaterLevel

        self.horizontal_position_provider = ADCPFixedHorizontalPosition()
        self.vertical_position_provider = ADCPFixedVerticalPosition()
        self.heading_provider: Any = None

        self._horizontal_position = np.empty((2, 0), dtype=float)
        self._vertical_position = np.empty((1, 0), dtype=float)
        self.depth_cell_position = np.empty((0, 0, 0, 3), dtype=float)
        self.water_velocity = np.empty((0, 0, 0), dtype=float)
        self.xform = None
        self.bed_position = np.empty((1, 0, 1, 3), dtype=float)
        self.beam_angle_deg = 20.0 ## Variable chosie manuellement : paramètre ADCP
        self.water_level = np.zeros((0,), dtype=float)
        self.water_level_object = ConstantWaterLevel(0.0)
        self.n_ensembles = 0

        for arg in args:
            if isinstance(arg, ADCPType):
                self.type = arg
            elif isinstance(arg, Filter):
                self.filters.append(arg)
            elif isinstance(arg, WaterState):
                self.water = arg
            elif isinstance(arg, SimpleTransducer):
                self.transducer = arg
            elif isinstance(arg, ADCPFixedHorizontalPosition):
                self.horizontal_position_provider = arg
            elif isinstance(arg, ADCPFixedVerticalPosition):
                self.vertical_position_provider = arg
            elif isinstance(arg, dict) or hasattr(arg, "__dict__"):
                self.raw = arg

        if src is not None:
            self._copy_from_adcp_like(src)


    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, value):
        self._raw = value
        self.reset_water()
        self.reset_transducer()


    def _copy_from_adcp_like(self, src: Any) -> None:
        self.depth_cell_position = np.asarray(
            getattr(src, "depth_cell_position", np.empty((0, 0, 0, 3))),
            dtype=float,
        )
        self.water_velocity = np.asarray(
            getattr(src, "water_velocity", np.empty((0, 0, 0))),
            dtype=float,
        )
        xform_src = getattr(src, "xform", None)
        self.xform = None if xform_src is None else np.asarray(xform_src, dtype=float)

        self._horizontal_position = np.asarray(
            getattr(src, "horizontal_position", np.empty((2, 0))),
            dtype=float,
        )
        self._vertical_position = np.asarray(
            getattr(src, "vertical_position", np.empty((1, 0))),
            dtype=float,
        )

        self.bed_position = np.asarray(
            getattr(src, "bed_position", np.empty((1, 0, 1, 3))),
            dtype=float,
        )
        self.beam_angle_deg = float(getattr(src, "beam_angle_deg", 20.0))
        self.water_level = np.asarray(
            getattr(src, "water_level", np.zeros((self._horizontal_position.shape[1],))),
            dtype=float,
        )
        self.n_ensembles = int(self._horizontal_position.shape[1]) if self._horizontal_position.ndim == 2 else 0

        self.raw = getattr(src, "raw", None)
    
    def apply_instrument_matrix_providers(self, providers: Sequence[InstrumentMatrixProvider] | None = None) -> None:

        if providers is None:
            providers = [
                InstrumentMatrixFromCalibration(),
                InstrumentMatrixFromPS3(),
                InstrumentMatrixFromBAngle(),
            ]
        chosen = None
        chosen_mats = None
        for p in providers:
            try:
                if p.get_has_data(self):
                    chosen = p
                    chosen_mats = p.get_b2i_matrix(self)  # shape (1, n, 4,4)
                    break
            except Exception:
                continue

        self.transformation_matrix_source = providers
        if chosen is None:
            return

        tm = TransformationMatrix()
        mats = np.asarray(chosen_mats)
        if mats.ndim == 4 and mats.shape[0] == 1:
            mats = mats[0]
        if mats.shape[0] == 1:
            tm.matrix = mats[0]
        else:
            tm.matrix = mats
        tm.source = chosen.__class__.__name__
        self.t_matrix = tm

    def _raw_get(self, *names: str, default=np.nan):
        if self._raw is None:
            return default
        for n in names:
            if hasattr(self._raw, n):
                return getattr(self._raw, n)
            if isinstance(self._raw, dict) and n in self._raw:
                return self._raw[n]
        return default

    @property
    def fileid(self) -> np.ndarray:
        v = self._raw_get("FileNumber", default=None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.arange(self.nensembles, dtype=int)
        arr = np.asarray(v).reshape(-1)
        if arr.size == 0:
            return np.arange(self.nensembles, dtype=int)
        return arr.astype(int)

    
    @property
    def nensembles(self) -> int:
        if self._horizontal_position.ndim == 2 and self._horizontal_position.shape[1] > 0:
            return int(self._horizontal_position.shape[1])
        if self.water_velocity.ndim >= 2 and self.water_velocity.shape[1] > 0:
            return int(self.water_velocity.shape[1])
        return int(self.n_ensembles)

    @property
    def ncells(self) -> int:
        if self.water_velocity.ndim >= 1:
            return int(self.water_velocity.shape[0])
        return 0

    @property
    def nbeams(self) -> int:
        if self.water_velocity.ndim >= 3:
            return int(self.water_velocity.shape[2])
        return 4

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return CoordinateSystem.Beam

    @property
    def beam_angle(self) -> np.ndarray:
        return np.full((self.nensembles,), float(self.beam_angle_deg), dtype=float)

    @property
    def roll(self) -> np.ndarray:
        r = self._raw_get("roll", default=np.nan)
        a = np.asarray(r, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        if a.size > 0 and not np.all(np.isnan(a)) and np.nanmax(np.abs(a)) > 90: # likely centidegrees
            a = a / 100.0
        return a[: self.nensembles] if a.size >= self.nensembles else np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)

    @property
    def pitch(self) -> np.ndarray:
        p = self._raw_get("pitch", default=np.nan)
        a = np.asarray(p, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        if a.size > 0 and not np.all(np.isnan(a)) and np.nanmax(np.abs(a)) > 90:
            a = a / 100.0
        if a.size < self.nensembles:
            a = np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)
        a = a[: self.nensembles]
        rr = self.roll
        return np.degrees(np.arctan(np.tan(np.radians(a)) * np.cos(np.radians(rr))))

    @property
    def heading(self) -> np.ndarray:
        if self.heading_provider is not None and hasattr(self.heading_provider, "heading"):
            return np.asarray(self.heading_provider.heading(self), dtype=float)
        h = self._raw_get("heading", default=np.nan)
        a = np.asarray(h, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        if np.nanmax(np.abs(a)) > 360:
            a = a / 100.0
        if a.size < self.nensembles:
            a = np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)
        return a[: self.nensembles]

    @property
    def headalign(self) -> np.ndarray:
        ha = self._raw_get("headalign", default=0.0)
        a = np.asarray(ha, dtype=float).reshape(-1)
        if a.size == 0:
            return np.zeros((self.nensembles,), dtype=float)
        if np.nanmax(np.abs(a)) > 360:
            a = a / 100.0
        if a.size < self.nensembles:
            a = np.pad(a, (0, self.nensembles - a.size), constant_values=0.0)
        return a[: self.nensembles]


    @property
    def time(self):

        if hasattr(self, "_time_override"):
            arr = np.asarray(getattr(self, "_time_override"))
            if arr.size == 0:
                return np.array([], dtype=float)

            if np.issubdtype(arr.dtype, np.number):
                return arr.astype(float).reshape(-1)

            if np.issubdtype(arr.dtype, np.datetime64):
                return arr.reshape(-1)

            try:
                return np.asarray(arr, dtype="datetime64[ns]").reshape(-1)
            except Exception:
                try:
                    return np.asarray(arr, dtype=float).reshape(-1)
                except Exception:
                    return np.array([], dtype=float)

        tv = self._raw_get("timeV", default=None)
        if tv is None or (isinstance(tv, float) and np.isnan(tv)):
            return np.array([], dtype=float)

        arr = np.asarray(tv)
        if arr.size == 0:
            return np.array([], dtype=float)

        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(float).reshape(-1)

        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.reshape(-1)

        try:
            return np.asarray(arr, dtype="datetime64[ns]").reshape(-1)
        except Exception:
            try:
                return np.asarray(arr, dtype=float).reshape(-1)
            except Exception:
                return np.array([], dtype=float)

    @property
    def blanking(self) -> np.ndarray:
        v = self._raw_get("blnk", default=np.nan)
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        if np.nanmax(np.abs(a)) > 10:
            a = a / 100.0
        return a[: self.nensembles] if a.size >= self.nensembles else np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)

    @property
    def lengthxmitpulse(self) -> np.ndarray:
        sp = self._raw_get("sp_transmit_length", default=None)
        if sp is not None and not (isinstance(sp, float) and np.isnan(sp)):
            return np.asarray(sp, dtype=float).reshape(-1) / 1000.0
        v = self._raw_get("lngthtranspulse", default=np.nan)
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        if np.nanmax(np.abs(a)) > 10:
            a = a / 100.0
        return a[: self.nensembles] if a.size >= self.nensembles else np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)

    @property
    def cellsize(self) -> np.ndarray:
        sp = self._raw_get("sp_bin_space", default=None)
        if sp is not None and not (isinstance(sp, float) and np.isnan(sp)):
            return np.asarray(sp, dtype=float).reshape(-1) / 10000.0
        v = self._raw_get("binsize", default=np.nan)
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        if np.nanmax(np.abs(a)) > 1:
            a = a / 100.0
        return a[: self.nensembles] if a.size >= self.nensembles else np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)

    @property
    def distmidfirstcell(self) -> np.ndarray:
        sp = self._raw_get("sp_mid_bin1", default=None)
        if sp is not None and not (isinstance(sp, float) and np.isnan(sp)):
            return np.asarray(sp, dtype=float).reshape(-1) / 10000.0
        v = self._raw_get("distmidbin1", default=np.nan)
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        if np.nanmax(np.abs(a)) > 1:
            a = a / 100.0
        return a[: self.nensembles] if a.size >= self.nensembles else np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)

    @property
    def depth_cell_slant_range(self) -> np.ndarray:
        if self.ncells == 0 or self.nensembles == 0:
            return np.empty((0, 0), dtype=float)
        k = np.arange(self.ncells, dtype=float)[:, None]
        d0 = np.asarray(self.distmidfirstcell, dtype=float)[None, :]
        cs = np.asarray(self.cellsize, dtype=float)[None, :]
        ba = np.asarray(self.beam_angle, dtype=float)[None, :]
        return (d0 + k * cs) / np.cos(np.radians(ba))

    @property
    def temperature(self) -> np.ndarray:
        v = self._raw_get("temperature", default=np.nan)
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        if np.nanmax(np.abs(a)) > 80:
            a = a / 100.0
        return a[: self.nensembles] if a.size >= self.nensembles else np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)

    @property
    def salinity(self) -> np.ndarray:
        v = self._raw_get("salinity", default=np.nan)
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        return a[: self.nensembles] if a.size >= self.nensembles else np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)

    @property
    def pressure(self) -> np.ndarray:
        v = self._raw_get("pressure", default=np.nan)
        a = np.asarray(v, dtype=float).reshape(-1)
        if a.size == 0:
            return np.full((self.nensembles,), np.nan)
        underflow = a > 3e9
        if np.any(underflow):
            a = a.copy()
            a[underflow] = a[underflow] - np.iinfo(np.uint32).max
        a = a * 10.0
        return a[: self.nensembles] if a.size >= self.nensembles else np.pad(a, (0, self.nensembles - a.size), constant_values=np.nan)
    

    @property
    def horizontal_position(self):
        if self.horizontal_position_provider is None:
            return self._horizontal_position
        return self.horizontal_position_provider.horizontal_position(self)

    @property
    def vertical_position(self):
        if self.vertical_position_provider is None:
            return self._vertical_position
        return self.vertical_position_provider.get_vertical_position(self)

    @property
    def convexity(self) -> np.ndarray:
        if self._raw is None:
            return np.array([], dtype=int)
        sysconf = self._raw_get("sysconf", default=np.array([], dtype=str))
        if sysconf.size == 0:
            return np.array([], dtype=int)
        conv = np.array([int(conf[3]) if len(conf) >= 4 else 0 for conf in sysconf], dtype=int)
        conv = np.where(conv == 0, -1, 1)
        return conv[:self.nensembles]

    @property
    def is_upward(self) -> np.ndarray:

        if self._raw is None:
            return np.array([], dtype=bool)
        sysconf = self._raw_get("sysconf", default=np.array([], dtype=str))
        if sysconf.size == 0:
            return np.array([], dtype=bool)
        is_up = np.array([int(conf[7]) == 1 if len(conf) >= 8 else False for conf in sysconf], dtype=bool)
        return is_up[:self.nensembles]
    

    @property
    def tilts_used_in_transform(self) -> np.ndarray:

        if self._raw is None:
            return np.array([], dtype=bool)
        corinfo = self._raw_get("corinfo", default=np.array([], dtype=int))
        if corinfo.size == 0:
            return np.array([], dtype=bool)
        tilts_used = np.array([(int(conf[2]) == 1) if len(conf) >= 3 else False for conf in corinfo], dtype=bool)
        return tilts_used[:self.nensembles]
    
    @property
    def three_beam_solutions_used(self) -> np.ndarray:

        if self._raw is None:
            return np.array([], dtype=bool)
        corinfo = self._raw_get("corinfo", default=np.array([], dtype=int))
        if corinfo.size == 0:
            return np.array([], dtype=bool)
        three_beam_used = np.array([(int(conf[1]) == 1) if len(conf) >= 2 else False for conf in corinfo], dtype=bool)
        return three_beam_used[:self.nensembles]
    

    @property
    def bandwidth(self) -> np.ndarray:
        """Bandwidth used (0=wide, 1=narrow)."""
        if self._raw is None:
            return np.array([], dtype=int)
        bw = self._raw_get("bandwidth", default=np.array([], dtype=int))
        if bw is None or (isinstance(bw, float) and np.isnan(bw)):
            return np.array([], dtype=int)
        bw = np.asarray(bw, dtype=int).reshape(-1)
        return bw[:self.nensembles]

    @property
    def current_factor(self) -> float:

        if not self.is_workhorse:
            warnings.warn("Assuming ADCP is a Workhorse")
        freq = self.transducer.frequency_hz
        if freq == 76.8e3:
            return 43838
        elif freq in [153.6e3, 307.2e3, 614.4e3, 1228.8e3, 2457.6e3]:
            return 11451
        else:
            warnings.warn("Do not know current factor for given ADCP type")
            return np.nan

    @property
    def current(self) -> np.ndarray:

        adc = self._raw_get("ADC", default=np.empty((0, 8)))
        if adc.size == 0:
            return np.array([], dtype=float)
        adc = np.asarray(adc, dtype=int)
        return (adc[:, 0] * self.current_factor / 1e6).reshape(1, -1)


    @property
    def bin_mapping_used(self) -> np.ndarray:

        if self._raw is None:
            return np.array([], dtype=bool)
        corinfo = self._raw_get("corinfo", default=np.array([], dtype=int))
        if corinfo.size == 0:
            return np.array([], dtype=bool)
        bin_mapping_used = np.array([(int(conf[0]) == 1) if len(conf) >= 1 else False for conf in corinfo], dtype=bool)
        return bin_mapping_used[:self.nensembles]
    
    @property
    def voltage_factor(self) -> float:

        if self._raw_get("sp_mid_bin1", default=None) is not None:
            return 1e5
        if not self.is_workhorse:
            warnings.warn("Assuming ADCP is a Workhorse")
        freq = self.transducer.frequency_hz
        if freq == 76.8e3:
            return 2092719
        elif freq in [153.6e3, 307.2e3]:
            return 592157
        elif freq == 614.4e3:
            return 380667
        elif freq in [1228.8e3, 2457.6e3]:
            return 253765
        else:
            warnings.warn("Unknown voltage factor")
            return np.nan

    @property
    def voltage(self) -> np.ndarray:

        adc = self._raw_get("ADC", default=np.empty((0, 8)))
        if adc.size == 0:
            return np.array([], dtype=float)
        adc = np.asarray(adc, dtype=int)
        return (adc[:, 1] * self.voltage_factor / 1e6).reshape(1, -1)

    @property
    def power(self) -> np.ndarray:
        return self.voltage * self.current

    @property
    def attitude_temperature(self) -> np.ndarray:

        if not self.is_workhorse:
            warnings.warn("Assuming ADCP is a Workhorse")
        DC_COEF = 9.82697464e1
        FIRST_COEF = -5.86074151382e-3
        SECOND_COEF = 1.60433886495e-7
        THIRD_COEF = -2.32924716883e-12
        adc_data = self._raw_get("ADC", default=np.empty((0, 8)))
        if adc_data.size == 0:
            return np.array([], dtype=float)
        t_cnts = (np.asarray(adc_data, dtype=int)[:, 5] * 256).reshape(1, -1)
        return (
            self.temperature_offset +
            ((THIRD_COEF * t_cnts + SECOND_COEF) * t_cnts + FIRST_COEF) * t_cnts +
            DC_COEF
        )

    @property
    def intensity_scale(self) -> np.ndarray:
        return 127.3 / (self.attitude_temperature + 273)

    @property
    def echo(self) -> np.ndarray:
        """Received raw echo intensity (dB)."""
        echo_raw = self._raw_get("ECHO", default=np.empty((0, 0, 0)))
        return np.asarray(echo_raw, dtype=float) * self.intensity_scale

    @property
    def backscatter_constant(self) -> float:

        if not self.is_workhorse:
            warnings.warn("Assuming ADCP is a Workhorse")
        freq = self.transducer.frequency_hz
        if freq == 76.8e3:
            return -159.1
        elif freq == 307.2e3:
            if self.type == ADCPType.Sentinel:
                return -143.5
            elif self.type == ADCPType.Monitor:
                return -143
            else:
                warnings.warn("Assuming ADCP is a Monitor")
                return -143
        elif freq == 614.4e3:
            if self.type != ADCPType.RioGrande:
                warnings.warn("Assuming ADCP is a RioGrande")
            return -139.3
        elif freq == 1228.8e3:
            return -129.1
        else:
            warnings.warn("Unknown backscatter constant for current ADCP Type")
            return np.nan

    @property
    def backscatter(self) -> np.ndarray:

        pt = self.transducer
        R = self.depth_cell_slant_range + self.cellsize / 2 / np.cos(np.radians(self.beam_angle))
        two_alpha_R = 2 * pt.attenuation * R
        LDBM = 10 * np.log10(self.lengthxmitpulse)
        PDBW = 10 * np.log10(self.power)
        val = (
            self.backscatter_constant +
            10 * np.log10((self.attitude_temperature + 273.16) * R**2 * pt.near_field_correction(R)**2) -
            LDBM - PDBW + two_alpha_R +
            10 * np.log10(10**((self.echo - self.noise_level) / 10) - 1)
        )
        val[self.bad()] = np.nan
        return val   


    def reset_water(self):
        if self._raw is None:
            return
        t = self.temperature
        s = self.salinity
        self.water.temperature_c = t
        self.water.salinity_psu = s * 1000

    def reset_transducer(self):

        if self._raw is None:
            return
        
        freq = self._raw_get("frequency_hz", "frequency_khz", default=np.nan)
        fv = np.asarray(freq).reshape(-1)
        if fv.size > 0 and np.isfinite(fv[0]):
            val = float(fv[0])
            if val < 1e4:
                val *= 1e3  # khz -> hz if needed
            self.transducer.frequency_hz = val
        p = self.pressure
        if p.size > 0 and np.isfinite(np.nanmedian(p)):
            self.transducer.depth = max(float(np.nanmedian(p)) / 9.81 / self.water.density_kgm3, 0.0)

        # if self.type == ADCPType.RiverRay:
        #     from Classes.acoustics import PhasedArrayTransducer
        #     if not isinstance(self.transducer, PhasedArrayTransducer):
        #         self.transducer = PhasedArrayTransducer()
        #     self.transducer.frequency_hz = 614.4e3
        #     self.transducer.radius = 0.076 / 2
        #     return

        sysid = self._raw_get("sysconf", default=np.array([], dtype=str))
        if sysid.size == 0:
            return
        sysid = sysid[0][:3] if isinstance(sysid, np.ndarray) and sysid.size > 0 else ""

        if sysid == "000":  # Long Ranger
            self.transducer.radius = 0.203 / 2
            self.transducer.frequency_hz = 76.8e3
        elif sysid == "100":  # QuarterMaster
            self.transducer.frequency_hz = 153.6e3
            if self.type in [ADCPType.QuarterMaster1500, ADCPType.QuarterMaster3000]:
                self.transducer.radius = 0.178 / 2
            elif self.type == ADCPType.QuarterMaster1500ModBeams:
                self.transducer.radius = 0.1854 / 2
            elif self.type == ADCPType.QuarterMaster6000:
                self.transducer.radius = 0.184 / 2
            else:
                self.transducer.radius = 0.1854 / 2  # Valeur par défaut
        elif sysid == "010":
            self.transducer.frequency_hz = 307.2e3
            if self.type in [ADCPType.SentinelV, ADCPType.MonitorV]:
                self.transducer.radius = np.nan
            elif self.type in [ADCPType.Monitor, ADCPType.Sentinel]:
                self.transducer.radius = 0.0984 / 2
            elif self.type == ADCPType.Mariner:
                self.transducer.radius = 0.0895 / 2
            else:
                self.transducer.radius = 0.0984 / 2
        elif sysid == "110":
            if self.type == ADCPType.SentinelV:
                self.transducer.frequency_hz = 491.52e3
                self.transducer.radius = np.nan
            elif self.type in [ADCPType.Monitor, ADCPType.Sentinel]:
                self.transducer.frequency_hz = 614.4e3
                self.transducer.radius = 0.0984 / 2
            elif self.type in [ADCPType.Mariner, ADCPType.RioGrande]:
                self.transducer.frequency_hz = 614.4e3
                self.transducer.radius = 0.0895 / 2
            else:
                self.transducer.frequency_hz = 614.4e3
                self.transducer.radius = 0.0984 / 2
        elif sysid == "001":
            if self.type in [ADCPType.SentinelV, ADCPType.MonitorV]:
                self.transducer.frequency_hz = 983.04e3
                self.transducer.radius = np.nan
            elif self.type in [ADCPType.Monitor, ADCPType.Sentinel, ADCPType.Mariner, ADCPType.RioGrande]:
                self.transducer.frequency_hz = 1228.8e3
                self.transducer.radius = 0.0699 / 2
            else:
                self.transducer.frequency_hz = 1228.8e3
                self.transducer.radius = 0.0699 / 2
        elif sysid == "101":
            self.transducer.radius = np.nan
            if self.type == ADCPType.StreamPro:
                self.transducer.frequency_hz = 2e6
            else:
                self.transducer.frequency_hz = 2e6

    def is_workhorse(self) -> bool:

        workhorse_types = [
            ADCPType.LongRanger1500, ADCPType.LongRanger3000,
            ADCPType.QuarterMaster1500, ADCPType.QuarterMaster1500ModBeams,
            ADCPType.QuarterMaster3000, ADCPType.QuarterMaster6000,
            ADCPType.Sentinel, ADCPType.Mariner, ADCPType.Monitor
        ]
        return self.type in workhorse_types


    def bad(self, filters=None):
        if self.water_velocity is None or self.water_velocity.size == 0:
            return np.empty_like(self.water_velocity, dtype=bool)

        use_filters = self.filters if filters is None else filters
        if use_filters is None or len(use_filters) == 0:
            return np.zeros_like(self.water_velocity, dtype=bool)

        bad_mask = np.zeros_like(self.water_velocity, dtype=bool)
        for f in use_filters:
            if hasattr(f, "bad"):
                fb = np.asarray(f.bad(self), dtype=bool)
                if fb.shape == bad_mask.shape:
                    bad_mask |= fb
        return bad_mask

    def velocity(self, dst: str | CoordinateSystem | None = None, filters=None):

        if dst is None:
            dst = CoordinateSystem.Beam
        if isinstance(dst, str):
            d = dst.lower()
            if d == "beam":
                dst = CoordinateSystem.Beam
            elif d == "earth":
                dst = CoordinateSystem.Earth
            else:
                raise ValueError("dst must be Beam or Earth")

        vel_beam = np.asarray(self.water_velocity, dtype=float)

        if dst == CoordinateSystem.Beam:
            out = vel_beam.copy()
        elif dst == CoordinateSystem.Earth:
            if self.xform is None:
                raise RuntimeError("xform not available to convert Beam -> Earth")
            xf = np.asarray(self.xform, dtype=float)
            if vel_beam.ndim != 3 or xf.ndim != 4:
                raise ValueError("Invalid shape for water_velocity or xform")
            out = np.nansum(xf * vel_beam[..., None], axis=2)
        else:
            raise NotImplementedError("Only Beam and Earth are currently implemented")

        bm = self.bad(filters=filters)
        if out.ndim == 3 and bm.shape == out.shape:
            out = out.copy()
            out[bm] = np.nan
        elif out.ndim == 3 and bm.ndim == 3 and out.shape[:2] == bm.shape[:2] and out.shape[2] == 3:
            m2 = np.any(bm, axis=2)
            out = out.copy()
            out[m2, :] = np.nan
        return out

    def depth_cell_offset(self, dst: str | CoordinateSystem = CoordinateSystem.Earth):
        if isinstance(dst, str):
            dst = CoordinateSystem.Earth if dst.lower() == "earth" else CoordinateSystem.Beam

        rng = self.depth_cell_slant_range  # (cells, ens)
        if rng.size == 0:
            return np.empty((0, 0, 0, 3), dtype=float)

        if self.xform is None:
            n_cells, n_ens = rng.shape
            n_beams = self.nbeams
            out = np.zeros((n_cells, n_ens, n_beams, 3), dtype=float)
            out[..., 2] = -rng[:, :, None]
            return out

        tm = -np.asarray(self.xform, dtype=float)  
        return tm * rng[:, :, None, None]

    def depth_cell_position_xyz(self):
        off = self.depth_cell_offset(dst=CoordinateSystem.Earth)
        if off.size == 0:
            return off
        
        hp = self.horizontal_position
        vp = self.vertical_position

        base = np.zeros((1, off.shape[1], 1, 3), dtype=float)
        base[0, :, 0, 0] = hp[0, :]  # x
        base[0, :, 0, 1] = hp[1, :]  # y
        base[0, :, 0, 2] = vp[0, :]  # z

        return off + base
    
    def xform(
        self,
        dst: Union[CoordinateSystem, str],
        src: Union[CoordinateSystem, str, None] = None,
        use_tilts: bool = False
    ) -> np.ndarray:

        if isinstance(dst, str):
            dst = CoordinateSystem[dst]
        if src is None:
            src = self.coordinate_system
        if isinstance(src, str):
            src = CoordinateSystem[src]

        tm = np.tile(np.eye(4)[None, :, :, :], (1, self.nensembles, 1, 1))

        I = CoordinateSystem.Instrument
        S = CoordinateSystem.Ship
        E = CoordinateSystem.Earth
        B = CoordinateSystem.Beam

        croll = self.roll.copy()
        croll[self.is_upward] += 180  #
        ha = self.headalign
        cpitch = self.pitch
        head = self.heading

        if dst >= I and src < I:
            tmptm = self.transformation_matrix_source[0].b2i_matrix(self)
            tm = tmptm

        if dst >= S and src < S:
            tilt_matrix = self.head_tilt(ha, cpitch, croll)
            tm = np.matmul(tilt_matrix[:, None, :, :], tm)

        if dst >= E and src < E:
            heading_matrix = self.head_tilt(head)
            tm = np.matmul(heading_matrix[:, None, :, :], tm)

        if dst <= S and src > S:
            heading_matrix = self.head_tilt(head)
            tm = np.linalg.inv(heading_matrix[:, None, :, :]) @ tm

        if dst <= I and src > I:
            if not use_tilts:
                cpitch[~self.tilts_used_in_transform] = 0
                croll[~self.tilts_used_in_transform] = 0
            tilt_matrix = self.head_tilt(ha, cpitch, croll)
            tm = np.linalg.inv(tilt_matrix[:, None, :, :]) @ tm

        if dst == B and src > B:
            tmptm = self.transformation_matrix_source[0].i2b_matrix(self)
            tm = np.matmul(tm, tmptm[:, None, :, :])

        return tm

    def plot_orientations(self):

        fig, axs = plt.subplots(4, 1, figsize=(10, 8))

        # is_upward
        axs[0].plot(self.is_upward)
        axs[0].set_ylabel("Is upward")

        # pitch
        axs[1].plot(self.pitch)
        axs[1].set_ylabel("Pitch (deg)")

        # roll
        axs[2].plot(self.roll)
        axs[2].set_ylabel("Roll (deg)")

        # heading
        axs[3].plot(self.heading)
        axs[3].set_ylabel("Heading (deg)")

        plt.tight_layout()
        return fig

    def plot_velocity(self, vel: Optional[np.ndarray] = None):

        vel_pos = self.depth_cell_position
        vel_pos = np.nanmean(vel_pos[:, :, :, 2], axis=2) 

        if vel is None:
            vel = self.velocity(CoordinateSystem.Earth)

        t = self.time
        if t.size == 0:
            return None
        t_seconds = np.array([(t_i - t[0]).total_seconds() for t_i in t])

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        im1 = axs[0].pcolormesh(t_seconds, vel_pos, vel[:, :, 0], shading="flat")
        fig.colorbar(im1, ax=axs[0], label="V_x (m/s)")
        im2 = axs[1].pcolormesh(t_seconds, vel_pos, vel[:, :, 1], shading="flat")
        fig.colorbar(im2, ax=axs[1], label="V_y (m/s)")
        im3 = axs[2].pcolormesh(t_seconds, vel_pos, vel[:, :, 2], shading="flat")
        fig.colorbar(im3, ax=axs[2], label="V_z (m/s)")
        axs[2].set_xlabel("Time (s)")
        for ax in axs:
            ax.set_ylabel("Vertical position (m)")
        plt.tight_layout()
        return fig

    def plot_backscatter(self):

        sv_pos = self.depth_cell_offset
        sv_pos = np.nanmean(sv_pos[:, :, :, 2], axis=2)  # Moyenne sur les faisceaux
        sv = self.backscatter
        t = self.time
        if t.size == 0:
            return None
        t_seconds = np.array([(t_i - t[0]).total_seconds() for t_i in t])
        nb = sv.shape[2] if sv.ndim >= 3 else 1

        fig, axs = plt.subplots(nb, 1, figsize=(10, 4 * nb))
        for cb in range(nb):
            if sv.ndim == 3:
                im = axs[cb].pcolormesh(t_seconds, sv_pos, sv[:, :, cb], shading="flat")
            else:
                im = axs[cb].pcolormesh(t_seconds, sv_pos, sv, shading="flat")
            clim = np.nanmean(sv[:, :, cb]) + np.array([-2, 2]) * np.nanstd(sv[:, :, cb])
            fig.colorbar(im, ax=axs[cb], label="Backscatter (dB)")
            axs[cb].set_clim(clim)
            axs[cb].set_title(f"Beam {cb + 1}")
        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        return fig

    def plot_filters(self):

        if hasattr(self.filters, "plot"):
            self.filters.plot(self)

    def plot_all(self):

        fig1 = self.plot_orientations()
        fig2 = self.plot_filters()
        fig3 = self.plot_velocity()
        fig4 = self.plot_backscatter()
        return fig1, fig2, fig3, fig4


    @staticmethod
    def int16_to_double(i):
        arr = np.asarray(i)
        d = arr.astype(float)
        fbad = arr == np.iinfo(np.int16).min
        d[fbad] = np.nan
        return d

    @staticmethod
    def head_tilt(heading_deg, pitch_deg=0.0, roll_deg=0.0):
        h = np.radians(np.asarray(heading_deg, dtype=float))
        p = np.radians(np.asarray(pitch_deg, dtype=float))
        r = np.radians(np.asarray(roll_deg, dtype=float))

        ch, sh = np.cos(h), np.sin(h)
        cp, sp = np.cos(p), np.sin(p)
        cr, sr = np.cos(r), np.sin(r)

        shp = np.broadcast(ch, sh, cp, sp, cr, sr).shape
        out = np.zeros(shp + (4, 4), dtype=float)

        out[..., 0, 0] = ch * cr + sh * sp * sr
        out[..., 0, 1] = sh * cp
        out[..., 0, 2] = ch * sr - sh * sp * cr
        out[..., 0, 3] = 0.0


        out[..., 1, 0] = -sh * cr + ch * sp * sr
        out[..., 1, 1] = ch * cp
        out[..., 1, 2] = -sh * sr - ch * sp * cr
        out[..., 1, 3] = 0.0

        out[..., 2, 0] = -cp * sr
        out[..., 2, 1] = sp
        out[..., 2, 2] = cp * cr
        out[..., 2, 3] = 0.0

        out[..., 3, 0] = 0.0
        out[..., 3, 1] = 0.0
        out[..., 3, 2] = 0.0
        out[..., 3, 3] = 1.0

        return out

    @staticmethod
    def invert_xform(tm: np.ndarray) -> np.ndarray:
        arr = np.asarray(tm, dtype=float)
        if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
            raise ValueError("tm must be (..., n, n)")
        inv = np.empty_like(arr)
        it = np.ndindex(arr.shape[:-2])
        for idx in it:
            inv[idx] = np.linalg.pinv(arr[idx])
        return inv


"""
    Vessel mounted ADCP wrapper class for adcp structures read with readADCP.
    This class extends ADCP with vessel-specific functionality, such as bottom tracking
    and ship velocity estimation.

    Attributes:
        shipvel_provider: List of ShipVelocityProvider objects to estimate ship velocity.
        water_level_object: WaterLevel object to compute water level.

    VMADCP methods
       bed_offset - xyz offset to the bed
       bed_position - xyz positions of the bed
       water_velocity - get velocity profile data corrected for ship motions
       btvel - ship velocity detected with  bottom tracking
       plot_bed_position - plot position of detected bed

"""

class VMADCP(ADCP):

    def __init__(
        self,
        *args,
        source,
        transect_idx: int | None = None,
        nav_ref: str = "bt_vel",
        use_raw_bt_beam_bathy: bool = True,
    ) -> None:
        

        from .ADCPVerticalPosition import ADCPVerticalPositionFromWaterLevel

        if hasattr(source, "depth_cell_position") and hasattr(source, "water_velocity") and hasattr(source, "xform"):
            super().__init__(source)
            self._time_override = getattr(source, "time", None)
            self.water_level_object = getattr(source, "water_level_object", ConstantWaterLevel(0.0))
            self.shipvel_provider = getattr(source, "shipvel_provider", [ShipVelocityFromBT(), ShipVelocityFromGPS()])
            self._ensure_default_filters()
            return
        else:
            super().__init__(*args)


        if not hasattr(source, "transects"):
            raise ValueError("VMADCP expects a qrevint Measurement or an adcp-like source")

        self.vertical_position_provider = ADCPVerticalPositionFromWaterLevel(self.water_level_object, self)
        self.source = source

        self.horizontal_position_provider = LatLonToUTM(zone=31)
        print(f"Provider utilisé : {self.horizontal_position_provider.__class__.__name__}") 

        for arg in args:
            if isinstance(arg, ShipVelocityProvider):
                self.shipvel_provider = [arg]

        if source is not None:
            if not hasattr(source, "transects"):
                raise ValueError("VMADCP expects a qrevint Measurement or an adcp-like source")


        if hasattr(source, "_raw") and "VISEA_Extern" in source._raw:
            easting = np.asarray(source._raw["VISEA_Extern"]["Easting"], dtype=float)
            northing = np.asarray(source._raw["VISEA_Extern"]["Northing"], dtype=float)

        meas = source

        all_track_x = []
        all_track_y = []

        all_latitude = []
        all_longitude = []

        all_depth_ens = []
        all_cell_depth = []
        all_raw_beam = []
        all_heading = []
        all_pitch = []
        all_roll = []
        all_time = []
        all_valid = []

        lon1 = np.asarray(meas.transects[0].gps.gga_lon_ens_deg, dtype=float)
        mean_lon = np.nanmean(lon1)
        zone_meas = get_utm_zone(mean_lon)
        
        print(f"Zone UTM déterminée pour la mesure : {zone_meas} (Longitude moyenne : {mean_lon})")

        ## 30/07
        all_depth_beams = []
        ##

        for transect in meas.transects:
            bt = transect.depths.bt_depths

            # # Boolean mask filter from QRevInt :
            # valid = np.asarray(bt.valid_data, dtype=bool)
            # No filter (all ping are valid) :
            valid = np.ones_like(bt.valid_data, dtype=bool)

            is_right_start = getattr(transect, "orig_start_edge", "Left") == "Right"
            if is_right_start:
                valid = valid[::-1]

            track_x, track_y = self._get_track(transect, nav_ref=nav_ref, zone_meas=zone_meas)
            depth_ens = np.asarray(bt.depth_processed_m, dtype=float)

            depth_beams = None

            if use_raw_bt_beam_bathy and hasattr(bt, "depth_beams_m"):
                depth_beams = np.asarray(bt.depth_beams_m, dtype=float)
                if depth_beams.ndim == 1:
                    depth_beams = depth_beams.reshape(1, -1)

                valid_beams = np.asarray(getattr(bt, "valid_beams", np.ones_like(depth_beams, dtype=bool)), dtype=bool)
                if valid_beams.ndim == 1:
                    valid_beams = valid_beams.reshape(1, -1)
                beam_depth_for_median = np.array(depth_beams, dtype=float, copy=True)
                if valid_beams.shape == beam_depth_for_median.shape:
                    beam_depth_for_median[~valid_beams] = np.nan
                depth_raw = np.nanmedian(beam_depth_for_median, axis=0)
                if np.any(np.isfinite(depth_raw)):
                    depth_ens = depth_raw

            cell_depth = np.asarray(bt.depth_cell_depth_m, dtype=float)
            raw_beam = np.asarray(transect.w_vel.raw_vel_mps, dtype=float)

            if raw_beam.ndim != 3 or raw_beam.shape[0] < 4:
                raise ValueError("raw_vel_mps must have shape (4, n_cells, n_ensembles)")
            
            if is_right_start:
                track_x = track_x[::-1]
                track_y = track_y[::-1]
                depth_ens = depth_ens[::-1]
                cell_depth = cell_depth[:, ::-1]
                raw_beam = raw_beam[:, :, ::-1]
                if depth_beams is not None:
                    depth_beams = depth_beams[:, ::-1]  
            
            all_track_x.append(track_x[valid])
            all_track_y.append(track_y[valid])
            all_depth_ens.append(depth_ens[valid])
            all_cell_depth.append(cell_depth[:, valid])
            all_raw_beam.append(raw_beam[:, :, valid])

            if depth_beams is None:
                depth_beams_for_bed = np.tile(depth_ens[np.newaxis, :], (4, 1))
            else:
                depth_beams_for_bed = depth_beams.copy()

            depth_beams_for_bed = np.where(depth_beams_for_bed >= 0.01, depth_beams_for_bed, np.nan)
            all_depth_beams.append(depth_beams_for_bed[:, valid])

            if hasattr(transect, "gps") and transect.gps is not None:
                lat = np.asarray(transect.gps.gga_lat_ens_deg, dtype=float)
                lon = np.asarray(transect.gps.gga_lon_ens_deg, dtype=float)

                ## 04/08
                lat, lon = VMADCP._fill_missing_positions_from_raw(transect, lat, lon)

                t_ens_rel = VMADCP._compute_ensemble_times(transect)
                if t_ens_rel.size == lat.size:
                    lat, lon = VMADCP._interpolate_missing_positions(lat, lon, t_ens_rel)
                ##

                # n_nan_ens = int(np.sum(np.isnan(lat) | np.isnan(lon)))
                # n_qualind_reject = int(np.sum(transect.gps.raw_gga_differential < 1)) if hasattr(transect.gps, "raw_gga_differential") else -1
                # n_raw_nan = int(np.sum(np.all(np.isnan(transect.gps.raw_gga_lat_deg), axis=1))) if hasattr(transect.gps, "raw_gga_lat_deg") else -1

        
            else:
                lat = np.full(len(valid), np.nan)
                lon = np.full(len(valid), np.nan)

            if is_right_start:
                lat = lat[::-1]
                lon = lon[::-1]

            all_latitude.append(lat[valid])
            all_longitude.append(lon[valid])

            sensors = transect.sensors
            heading = np.asarray(getattr(sensors.heading_deg, sensors.heading_deg.selected).data, dtype=float)
            pitch = np.asarray(getattr(sensors.pitch_deg, sensors.pitch_deg.selected).data, dtype=float)
            roll = np.asarray(getattr(sensors.roll_deg, sensors.roll_deg.selected).data, dtype=float)

            if is_right_start:
                heading = heading[::-1]
                pitch = pitch[::-1]
                roll = roll[::-1]

            all_heading.append(heading[valid])
            all_pitch.append(pitch[valid])
            all_roll.append(roll[valid])
            all_valid.append(valid)
            
            # dt_obj = transect.date_time
            # if hasattr(dt_obj, 'start_serial_time') and hasattr(dt_obj, 'ens_duration_sec'):
            #     start_time = dt_obj.start_serial_time
            #     ens_durations = dt_obj.ens_duration_sec

            #     timestamps = [start_time]
            #     for duration in ens_durations[:-1]:
            #         timestamps.append(timestamps[-1] + duration)

            #     time_array = np.array(timestamps)[valid]
            #     all_time.append(time_array)
            # else:
            #     all_time.append(np.arange(len(valid))[valid])

            ## 04/08
            dt_obj = transect.date_time
            t_ens_rel = VMADCP._compute_ensemble_times(transect)  

            if t_ens_rel.size == len(valid) and hasattr(dt_obj, 'start_serial_time'):
                time_full = float(dt_obj.start_serial_time) + t_ens_rel  
                if is_right_start:
                    time_full = time_full[::-1] 
                time_array = time_full[valid]
                all_time.append(time_array)

                # if time_array.size >= 2:
                #     sens = "décroissant (transect inversé)" if time_array[-1] < time_array[0] else "croissant"
                    # print(f"DEBUG transect start_edge={'Right' if is_right_start else 'Left'} : "
                    #       f"time {sens}, range=[{time_array.min():.2f}, {time_array.max():.2f}] "
                    #       f"(n={time_array.size})")
            else:
                all_time.append(np.arange(len(valid))[valid])
            #        


        # Concatenate all transects data: 

        self._horizontal_position = np.hstack((np.concatenate(all_track_x), np.concatenate(all_track_y))).reshape(2, -1)

        track_x_global = np.concatenate(all_track_x)
        track_y_global = np.concatenate(all_track_y)
        self._horizontal_position = np.vstack((track_x_global, track_y_global)) 
        self.horizontal_position_provider = None

        lat_global = np.concatenate(all_latitude)
        lon_global = np.concatenate(all_longitude)

        self._vertical_position = np.zeros((1, self._horizontal_position.shape[1]), dtype=float)
        self.n_ensembles = self._horizontal_position.shape[1]

        self._latitude = np.concatenate(all_latitude)
        self._longitude = np.concatenate(all_longitude)
        self.latitude = self._latitude
        self.longitude = self._longitude

        # bed_xyz = np.full((1, self.n_ensembles, 1, 3), np.nan, dtype=float)
        # bed_xyz[0, :, 0, 0] = self._horizontal_position[0, :]
        # bed_xyz[0, :, 0, 1] = self._horizontal_position[1, :]
        # bed_xyz[0, :, 0, 2] = -np.concatenate(all_depth_ens)
        # self.bed_position = bed_xyz

        depth_cell_position_list = []
        start_idx = 0
        for i, cell_depth in enumerate(all_cell_depth):
            n_cells = cell_depth.shape[0]  
            n_ens = cell_depth.shape[1]    

            dcp_block = np.full((n_cells, n_ens, 4, 3), np.nan, dtype=float)
            dcp_block[:, :, :, 0] = self._horizontal_position[0, start_idx:start_idx + n_ens][np.newaxis, :, np.newaxis]
            dcp_block[:, :, :, 1] = self._horizontal_position[1, start_idx:start_idx + n_ens][np.newaxis, :, np.newaxis]
            dcp_block[:, :, :, 2] = -cell_depth[:, :, np.newaxis]

            depth_cell_position_list.append(dcp_block)
            start_idx += n_ens

        # self.depth_cell_position = np.concatenate(depth_cell_position_list, axis=1)

        ## Ajout de Nan pour régler erreur sur les mesures Sontek au niveau des profondeur de cellules 

        max_depth_cells = max(dcp_block.shape[0] for dcp_block in depth_cell_position_list)
        depth_cell_position_padded = []

        for dcp_block in depth_cell_position_list:
            if dcp_block.shape[0] < max_depth_cells:
                padding_shape = (
                    max_depth_cells - dcp_block.shape[0], 
                    *dcp_block.shape[1:]  
                )   
                padding = np.full(padding_shape, np.nan, dtype=float)
                dcp_block_padded = np.vstack((dcp_block, padding))
                depth_cell_position_padded.append(dcp_block_padded)
            else:
                depth_cell_position_padded.append(dcp_block)

        self.depth_cell_position = np.concatenate(depth_cell_position_padded, axis=1)

        ##

        ## Ajout de Nan pour régler erreur sur les mesures Sontek au niveau des vitesse d'eau

        max_n_cells = max(raw_beam.shape[1] for raw_beam in all_raw_beam)
        # max_n_ens = max(raw_beam.shape[2] for raw_beam in all_raw_beam)

        all_raw_beam_padded = []

        for raw_beam in all_raw_beam:
            if raw_beam.shape[1] < max_n_cells:
                padding_cells = np.full((4, max_n_cells - raw_beam.shape[1], raw_beam.shape[2]), np.nan, dtype=float)
                raw_beam = np.concatenate((raw_beam, padding_cells), axis=1)

            # if raw_beam.shape[2] < max_n_ens:
            #     padding_ens = np.full((4, raw_beam.shape[1], max_n_ens - raw_beam.shape[2]), np.nan, dtype=float)
            #     raw_beam = np.concatenate((raw_beam, padding_ens), axis=2)

            all_raw_beam_padded.append(raw_beam)

        self.water_velocity = np.concatenate(all_raw_beam_padded, axis=2).transpose(1, 2, 0)

        # Transformation matrix for all ensembles:

        heading = np.concatenate(all_heading)
        pitch = np.concatenate(all_pitch)
        roll = np.concatenate(all_roll)

        ## Ajout de Nan pour régler erreur sur les mesures Sontek au niveau du xform

        # max_n_cells = max(cell_depth.shape[0] for cell_depth in all_cell_depth)
        # xform = np.full((max_n_cells, self.n_ensembles, 4, 3), np.nan, dtype=float)
        max_n_cells = max(raw_beam.shape[1] for raw_beam in all_raw_beam)
        max_n_ens = max(raw_beam.shape[2] for raw_beam in all_raw_beam)

        self.beam_angle_deg = float(getattr(meas.transects[0].adcp, "beam_angle_deg", 20.0))

        depth_beams_concat = np.concatenate(all_depth_beams, axis=1)

        if depth_beams_concat.shape[1] != self.n_ensembles:
            raise ValueError(
                f"depth_beams_concat has {depth_beams_concat.shape[1]} ensembles, "
                f"expected {self.n_ensembles} (mismatch with horizontal_position). "
                "Vérifiez que all_valid est aligné transect par transect avec "
                "all_track_x / all_heading."
            )

        for i, transect in enumerate(meas.transects):
            bt = transect.depths.bt_depths
            db = np.asarray(bt.depth_beams_m, dtype=float)
            vb = getattr(bt, "valid_beams", None)
            n_total = db.size
            n_below_threshold = int(np.sum(db < 0.01))
            n_invalid_by_mask = 0 if vb is None else int(np.sum(~np.asarray(vb, dtype=bool)))

        self.bed_position = self._compute_bed_position_4beams(
            depth_beams_m=depth_beams_concat,
            heading_deg=heading,
            pitch_deg=pitch,
            roll_deg=roll,
            beam_angle_deg=self.beam_angle_deg,
            horizontal_position=self._horizontal_position,
            vertical_position=self._vertical_position,
        )

        max_n_cells = max(cell_depth.shape[0] for cell_depth in all_cell_depth)
        xform = np.full((max_n_cells, self.n_ensembles, 4, 3), np.nan, dtype=float)

        print(f"DEBUG bed_position: {np.sum(np.all(np.isfinite(self.bed_position), axis=-1))} points valides (4 faisceaux)")
        print(f"DEBUG z range: [{np.nanmin(self.bed_position[...,2]):.2f}, {np.nanmax(self.bed_position[...,2]):.2f}]")


        t_mat_raw = np.asarray(meas.transects[0].adcp.t_matrix.matrix, dtype=float)

        if t_mat_raw.ndim == 3:
            t_mat = t_mat_raw[:, :, 0]  
        else:
            t_mat = t_mat_raw  

        # if t_mat.shape == (4, 4):
        #     t_mat_3x4 = t_mat[:3, :]  
        # elif t_mat.shape == (3, 4):
        #     t_mat_3x4 = t_mat  
        # elif t_mat.shape == (4, 3):
        #     t_mat_3x4 = t_mat.T  
        # else:
        #     raise ValueError(f"Forme inattendue pour t_mat : {t_mat.shape}")

        # start_idx = 0
        # for i, cell_depth in enumerate(all_cell_depth):
        #     n_cells = cell_depth.shape[0]
        #     n_ens = cell_depth.shape[1]

        #     for jj in range(n_ens):
        #         global_jj = start_idx + jj
        #         hpr = self._build_hpr_matrix(heading[global_jj], pitch[global_jj], roll[global_jj])
               
        #         hpr_3x4 = hpr[:3, :]

        #         rows_3x3 = hpr_3x4 @ t_mat_3x4.T
        #         rows_4x3 = np.vstack([rows_3x3, np.zeros((1, 3))])

        #         xform[:n_cells, global_jj, :, :] = rows_4x3[np.newaxis, :, :]

        #     start_idx += n_ens

        # self.xform = xform

        ## 06/08
        if t_mat.shape != (4, 4):
            raise ValueError(
                f"t_matrix doit être la matrice de calibration complète 4x4 "
                f"(beam -> instrument : lignes x,y,z,erreur ; colonnes = 4 faisceaux). "
                f"Forme reçue : {t_mat.shape}"
            )

        try:
            i2b = np.linalg.inv(t_mat)  # (4,4)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "La matrice de calibration t_matrix n'est pas inversible : "
                "impossible de calculer la transformation instrument -> faisceau."
            ) from exc

        i2b_xyz = i2b[:, :3] 

        start_idx = 0
        for i, cell_depth in enumerate(all_cell_depth):
            n_cells = cell_depth.shape[0]
            n_ens = cell_depth.shape[1]

            for jj in range(n_ens):
                global_jj = start_idx + jj
                hpr = self._build_hpr_matrix(heading[global_jj], pitch[global_jj], roll[global_jj])

                R = hpr[:3, :3]

                xform_ens = i2b_xyz @ R.T  

                xform[:n_cells, global_jj, :, :] = xform_ens[np.newaxis, :, :]

            start_idx += n_ens

        self.xform = xform
        ##
        
        norms = np.linalg.norm(xform[0, 0, :, :], axis=-1)  
        print(f"**DEBUG normes des vecteurs de faisceau (devrait être proche de 1.0) : {norms}")

        ##

        # self.beam_angle_deg = float(getattr(meas.transects[0].adcp, "beam_angle_deg", 20.0))
        self.water_level_object = ConstantWaterLevel(0.0)
        self.water_level = np.zeros((self.n_ensembles,), dtype=float)

        # self._time = np.concatenate(all_time)
        self._time_override = np.concatenate(all_time)

        self._ensure_default_filters()

    ##
    def _compute_bed_position_4beams(
        self,
        depth_beams_m: np.ndarray,      # (4, n_ens) : portée mesurée par chaque faisceau
        heading_deg: np.ndarray,        # (n_ens,)
        pitch_deg: np.ndarray,          # (n_ens,)
        roll_deg: np.ndarray,           # (n_ens,)
        beam_angle_deg: float,
        horizontal_position: np.ndarray,  # (2, n_ens)
        vertical_position: np.ndarray,    # (1, n_ens) ou (n_ens,)
    ) -> np.ndarray:
        """
        Calcule la position Terre (Earth) du fond détecté par CHACUN des 4
        faisceaux, équivalent Python de bed_offset()+bed_position() dans
        VMADCP.m.

        Retourne un tableau de forme (1, n_ens, 4, 3) : x, y, z pour chaque
        faisceau et chaque ensemble.
        """
        n_ens = depth_beams_m.shape[1]

        bangle = math.radians(float(beam_angle_deg))
        tbangle = math.tan(bangle)
        vecmagn = math.sqrt(tbangle * tbangle + 1.0)

        zz = -np.ones((4, n_ens), dtype=float)
        xx = np.zeros((4, n_ens), dtype=float)
        yy = np.zeros((4, n_ens), dtype=float)
        xx[0, :] = zz[0, :] * tbangle
        xx[1, :] = -zz[1, :] * tbangle
        yy[2, :] = -zz[2, :] * tbangle
        yy[3, :] = zz[3, :] * tbangle

        # Matrice de rotation cap/tangage/roulis (Instrument -> Terre) :
        heading = np.radians(np.asarray(heading_deg, dtype=float))
        pitch = np.radians(np.asarray(pitch_deg, dtype=float))
        roll = np.radians(np.asarray(roll_deg, dtype=float))

        ch, sh = np.cos(heading), np.sin(heading)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        m11 = ch * cr + sh * sp * sr
        m12 = sh * cp
        m13 = ch * sr - sh * sp * cr
        m21 = -sh * cr + ch * sp * sr
        m22 = ch * cp
        m23 = -sh * sr - ch * sp * cr
        m31 = -cp * sr
        m32 = sp
        m33 = cp * cr

        xxt = xx * m11 + yy * m12 + zz * m13
        yyt = xx * m21 + yy * m22 + zz * m23
        zzt = xx * m31 + yy * m32 + zz * m33

        D = np.asarray(depth_beams_m, dtype=float) / math.cos(bangle)
        xxt = D / vecmagn * xxt
        yyt = D / vecmagn * yyt
        zzt = D / vecmagn * zzt

        hp = np.asarray(horizontal_position, dtype=float)
        vp = np.asarray(vertical_position, dtype=float).reshape(-1)

        x_earth = xxt + hp[0, :][np.newaxis, :]   # (4, n_ens)
        y_earth = yyt + hp[1, :][np.newaxis, :]
        z_earth = zzt + vp[np.newaxis, :]

        # Réarrangement en (1, n_ens, 4, 3) :
        bed_xyz = np.stack([x_earth, y_earth, z_earth], axis=-1)  # (4, n_ens, 3)
        bed_xyz = np.moveaxis(bed_xyz, 0, 1)                      # (n_ens, 4, 3)
        bed_xyz = bed_xyz[np.newaxis, :, :, :]                    # (1, n_ens, 4, 3)

        return bed_xyz

    ##

    
    @property
    def bt_vertical_range(self) -> np.ndarray:
        if self._raw is None:
            return np.array([], dtype=float)
        btrange = self._raw_get("btrange", default=np.array([], dtype=int))
        if btrange.size == 0:
            return np.array([], dtype=float)
        r = np.asarray(btrange, dtype=float).reshape(-1) / 100.0
        r[r == 0] = np.nan
        return r[:self.nensembles]

    @property
    def slant_range_to_bed(self) -> np.ndarray:
        bangle = self.beam_angle  
        return self.bt_vertical_range / np.cos(np.radians(bangle))
    
    @property
    def horizontal_position(self):
        return self._horizontal_position
    
    @property
    def water_level(self) -> np.ndarray:
        
        if not hasattr(self, 'time') or self.time is None or getattr(self.time, 'size', 0) == 0:
            return np.zeros((0,), dtype=float)
        
        return self.water_level_object.get_water_level(self.time)
    
    @water_level.setter
    def water_level(self, value: np.ndarray):
        if not hasattr(self, 'time') or self.time is None or getattr(self.time, 'size', 0) == 0:
            self._water_level_fallback = value
            return

        if hasattr(self, "water_level_object") and self.water_level_object is not None:
            if hasattr(self.water_level_object, 'set_water_level'):
                self.water_level_object.set_water_level(value, self.time)
            else:
                self._water_level_fallback = value
        else:
            self._water_level_fallback = value


    def ship_velocity(self, dst: Union[CoordinateSystem, str, None] = None):

        if dst is None:
            dst = self.coordinate_system
        if isinstance(dst, str):
            dst = CoordinateSystem[dst]
        return self.shipvel_provider[0].ship_velocity(self, dst)
    
    def water_velocity(
        self,
        dst: Union[CoordinateSystem, str, None] = None,
        filters=None
    ) -> np.ndarray:
        
        if dst is None:
            dst = self.coordinate_system
        if isinstance(dst, str):
            dst = CoordinateSystem[dst]
        return super().velocity(dst, filters) + self.ship_velocity(dst)
    

    def btvel(
        self,
        dst: Union[CoordinateSystem, str, None] = None
    ) -> np.ndarray:

        if dst is None:
            dst = self.coordinate_system
        if isinstance(dst, str):
            dst = CoordinateSystem[dst]

        btvel_raw = self._raw_get("btvel", default=np.empty((0, 0, 0)))
        if btvel_raw.size == 0:
            return np.empty((0, 0, 0), dtype=float)

        btvel = self.int16_to_double(btvel_raw) / 1000.0  
        btvel = np.moveaxis(btvel, 0, -1)  

        btvel = np.asarray(btvel_raw, dtype=np.int16)
        btvel = self.int16_to_double(btvel) / 1000.0
        btvel = np.moveaxis(btvel, 0, -1)  

        if dst != self.coordinate_system:
            tm = self.xform(dst)
            btvel = np.matmul(tm, btvel[..., None]).squeeze(-1)

        return btvel
    
    def bed_offset(self, dst: Union[CoordinateSystem, str] = CoordinateSystem.Earth) -> np.ndarray:

        if isinstance(dst, str):
            dst = CoordinateSystem[dst]

        tm = -self.xform(CoordinateSystem.Beam, dst, use_tilts=True)
        tm = tm[..., :3, :3] 
        return tm * self.slant_range_to_bed[None, :, None, None]     


    def bed_position(self) -> np.ndarray:

        pos = self.bed_offset(CoordinateSystem.Earth)
        pos[:, :, :, 0] += self.horizontal_position[0:1, :]  
        pos[:, :, :, 1] += self.horizontal_position[1:2, :]  
        pos[:, :, :, 2] += self.vertical_position[0:1, :]    
        return pos

    def depth_cell_position(self) -> np.ndarray:

        pos = self.depth_cell_offset(dst=CoordinateSystem.Earth)
        pos[:, :, :, 0] += self.horizontal_position[0:1, :]  
        pos[:, :, :, 1] += self.horizontal_position[1:2, :]  
        pos[:, :, :, 2] += self.vertical_position[0:1, :]   
        return pos
    
    def plot_track(self, *args, **kwargs):

        ensfilt = EnsembleFilter(self)
        for arg in args:
            if isinstance(arg, EnsembleFilter):
                ensfilt = arg

        fig, ax = plt.subplots()
        for ce, filt in enumerate([ensfilt]):
            valid = ~filt.all_cells_bad(self)
            ax.plot(
                self.horizontal_position[0, valid],
                self.horizontal_position[1, valid],
                **kwargs
            )

        ax.set_aspect("equal")
        ax.set_xlabel(f"{self.horizontal_position_provider.__class__.__name__} x (m)")
        ax.set_ylabel(f"{self.horizontal_position_provider.__class__.__name__} y (m)")
        return ax
    

    def plot_bed_position(self, *args, **kwargs):

        pos = self.bed_position()
        pos = pos.reshape(-1, 3)  

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=pos[:, 2],
            s=10,
            **kwargs
        )
        fig.colorbar(sc, label="Bed elevation (m)")
        ax.set_xlabel(f"{self.horizontal_position_provider.__class__.__name__} x (m)")
        ax.set_ylabel(f"{self.horizontal_position_provider.__class__.__name__} y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=0, azim=90) 
        return ax, sc    

    def plot_velocity(self, vel=None, *args, **kwargs):

        if vel is None:
            vel = self.water_velocity(CoordinateSystem.Earth)
        fig = super().plot_velocity(vel, *args, **kwargs)
        self.add_bed_and_surface(fig)
        return fig

    def plot_backscatter(self, *args, **kwargs):

        fig = super().plot_backscatter(*args, **kwargs)
        self.add_bed_and_surface(fig, av_beams=False)
        return fig

    def plot_all(self):

        super().plot_all()
        fig1 = self.plot_track()
        fig2 = self.plot_bed_position()
        return fig1, fig2

    def add_bed_and_surface(self, hf, av_beams=True):

        if hf is None:
            hf = plt.gcf()

        bed_pos = self.bed_position
        bed_pos_z = np.nanmean(bed_pos[:, :, :, 2], axis=2)  # Moyenne sur les faisceaux

        if av_beams:
            bed_pos_z = np.tile(np.nanmean(bed_pos_z), (1, bed_pos_z.shape[1]))

        t = self.time
        if t.size == 0:
            return
        t_seconds = np.array([(t_i - t[0]).total_seconds() for t_i in t])

        for ax in hf.get_axes():
            ax.plot(t_seconds, bed_pos_z, "k", linewidth=2)
            ax.set_ylim([np.nanmin(bed_pos_z), 0])


    def _copy_from_adcp_like(self, src: Any) -> None:

        super()._copy_from_adcp_like(src)

        self.depth_cell_position = np.asarray(
            getattr(src, "depth_cell_position", np.empty((0, 0, 0, 3))),
            dtype=float,
        )
        self.water_velocity = np.asarray(
            getattr(src, "water_velocity", np.empty((0, 0, 0))),
            dtype=float,
        )

        xform_src = getattr(src, "xform", None)
        self.xform = None if xform_src is None else np.asarray(xform_src, dtype=float)

        self._horizontal_position = np.asarray(
            getattr(src, "horizontal_position", np.empty((2, 0))),
            dtype=float,
        )
        self._vertical_position = np.asarray(
            getattr(src, "vertical_position", np.empty((1, 0))),
            dtype=float,
        )

        self.bed_position = np.asarray(
            getattr(src, "bed_position", np.empty((1, 0, 1, 3))),
            dtype=float,
        )
        self.beam_angle_deg = float(getattr(src, "beam_angle_deg", 20.0))
        self.water_level = np.asarray(
            getattr(src, "water_level", np.zeros((self._horizontal_position.shape[1],))),
            dtype=float,
        )
        self.n_ensembles = int(self._horizontal_position.shape[1]) if self._horizontal_position.ndim == 2 else 0

        self.raw = getattr(src, "raw", None)
        self.water_level_object = getattr(src, "water_level_object", ConstantWaterLevel(0.0))
        self.shipvel_provider = getattr(src, "shipvel_provider", [ShipVelocityFromBT(), ShipVelocityFromGPS()])
        self.filters = list(getattr(src, "filters", []))

        try:
            if not (hasattr(self, "t_matrix") and getattr(self.t_matrix, "matrix", None) is not None):
                self.apply_instrument_matrix_providers()
        except Exception:
            pass

    def _ensure_default_filters(self) -> None:
        try:
            from .SideLobeFilter import SideLobeFilter
        except Exception:
            return
        if not any(f.__class__.__name__ == "SideLobeFilter" for f in self.filters):
            self.filters.append(SideLobeFilter())


    ## 03/08
    @staticmethod
    def _fill_missing_positions_from_raw(transect, lat_ens: np.ndarray, lon_ens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        gps = getattr(transect, "gps", None)
        if gps is None or not hasattr(gps, "raw_gga_lat_deg_unfiltered"):
            return lat_ens, lon_ens

        lat_ens = np.asarray(lat_ens, dtype=float).copy()
        lon_ens = np.asarray(lon_ens, dtype=float).copy()

        missing = np.isnan(lat_ens) | np.isnan(lon_ens)
        if not np.any(missing):
            return lat_ens, lon_ens

        raw_lat = np.asarray(gps.raw_gga_lat_deg_unfiltered, dtype=float)
        raw_lon = np.asarray(gps.raw_gga_lon_deg_unfiltered, dtype=float)

        n_recovered = 0
        for i in np.flatnonzero(missing):
            row_lat = raw_lat[i, :]
            row_lon = raw_lon[i, :]
            finite = np.isfinite(row_lat) & np.isfinite(row_lon) & (row_lat != 0) & (row_lon != 0)
            if np.any(finite):
                idx = np.flatnonzero(finite)[-1]
                lat_ens[i] = row_lat[idx]
                lon_ens[i] = row_lon[idx]
                n_recovered += 1

        if n_recovered > 0:
            print(f"DEBUG position fallback: {n_recovered} ensemble(s) récupéré(s) via GGA brut non filtré")

        return lat_ens, lon_ens
    

    @staticmethod
    def _compute_ensemble_times(transect) -> np.ndarray:

        dt_obj = transect.date_time
        if not hasattr(dt_obj, 'ens_duration_sec'):
            return np.array([], dtype=float)

        durations = np.asarray(dt_obj.ens_duration_sec, dtype=float)

        n_nan = int(np.sum(np.isnan(durations)))
        if n_nan > 0:
            med = np.nanmedian(durations)
            if not np.isfinite(med):
                med = 0.0
                
            durations = np.where(np.isnan(durations), med, durations)

        return np.concatenate(([0.0], np.cumsum(durations[:-1])))
    

    @staticmethod
    def _interpolate_missing_positions(lat: np.ndarray, lon: np.ndarray, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        lat = np.asarray(lat, dtype=float).copy()
        lon = np.asarray(lon, dtype=float).copy()
        time_s = np.asarray(time_s, dtype=float)

        valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(time_s)
        if np.count_nonzero(valid) < 2:
            return lat, lon

        missing = ~valid
        n_missing = int(np.sum(missing))
        if n_missing == 0:
            return lat, lon

        lat[missing] = np.interp(time_s[missing], time_s[valid], lat[valid])
        lon[missing] = np.interp(time_s[missing], time_s[valid], lon[valid])

        print(f"DEBUG interpolation position: {n_missing} ensemble(s) comblé(s) par interpolation temporelle")
        return lat, lon
    ##
    
    
    @staticmethod
    def _get_track(transect, nav_ref: str = "bt_vel", zone_meas: int = None):

        if hasattr(transect, "gps") and transect.gps is not None:
            lat = np.asarray(transect.gps.gga_lat_ens_deg, dtype=float)
            lon = np.asarray(transect.gps.gga_lon_ens_deg, dtype=float)  

            lat, lon = VMADCP._fill_missing_positions_from_raw(transect, lat, lon)

            ## 04/08
            t_ens_rel = VMADCP._compute_ensemble_times(transect)
            if t_ens_rel.size == lat.size:
                lat, lon = VMADCP._interpolate_missing_positions(lat, lon, t_ens_rel)
            ##

            transformer = Transformer.from_crs(
                "+proj=longlat +ellps=intl +towgs84=-87,-98,-121,0,0,0,0 +no_defs", 
                f"+proj=utm +zone={zone_meas} +ellps=intl +towgs84=-87,-98,-121,0,0,0,0 +units=m +no_defs",  
            )

            # transformer = Transformer.from_crs(
            #     "EPSG:4326",
            #     f"+proj=utm +zone={zone_meas} +datum=WGS84 +units=m +no_defs",
            #     always_xy=True,
            # )
        
            easting, northing = transformer.transform(lon, lat)
            return easting, northing
        
        else : 
            ship_data = transect.boat_vel.compute_boat_track(transect, ref=nav_ref)
            x_track = np.asarray(ship_data["track_x_m"], dtype=float)
            y_track = np.asarray(ship_data["track_y_m"], dtype=float)

            return x_track, y_track
    

    @staticmethod
    def _build_hpr_matrix(heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        h = math.radians(float(heading_deg))
        p = math.radians(float(pitch_deg))
        r = math.radians(float(roll_deg))

        ch, sh = math.cos(h), math.sin(h)
        cp, sp = math.cos(p), math.sin(p)
        cr, sr = math.cos(r), math.sin(r)

        return np.array(
            [
                [ch * cr + sh * sp * sr, sh * cp, ch * sr - sh * sp * cr, 0.0],
                [-sh * cr + ch * sp * sr, ch * cp, -sh * sr - ch * sp * cr, 0.0],
                [-cp * sr, sp, cp * cr, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
