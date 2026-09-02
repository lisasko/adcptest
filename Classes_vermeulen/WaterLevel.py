from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

"""
Defines ADCP vertical position based on the water level
"""

class WaterLevel(ABC):

    @abstractmethod # Subclasses should implement the get_water_level method
    def get_water_level(self, time=None):
        raise NotImplementedError

    # Compute the depth for a given time and elevation
    def get_depth(self, z : np.ndarray, time=None) -> np.ndarray:
        wl = self.get_water_level(time)
        return np.asarray(wl, dtype=float) - np.asarray(z, dtype=float)


"""
Defines a constant water level
"""

class ConstantWaterLevel(WaterLevel):

    def __init__(self, level_m: float = 0.0) -> None: 
        self.level_m = float(level_m)

    def get_water_level(self, time=None):
        if time is None:
            return self.level_m
        
        time_arr = np.asarray(time)
        return np.full(time_arr.shape, self.level_m, dtype=float)
    

"""
Implements a water level varying in time
"""

class VaryingWaterLevel(WaterLevel):

    def __init__(self, time, level_m) -> None:
        t = np.asarray(time, dtype=float).reshape(-1)
        wl = np.asarray(level_m, dtype=float).reshape(-1)
        if t.size != wl.size:
            raise ValueError("time and level_m must have the same length")
        if t.size < 1:
            raise ValueError("time and level_m must contain at least one sample")
        finite = np.isfinite(t) & np.isfinite(wl)
        t = t[finite]
        wl = wl[finite]
        if t.size < 1:
            raise ValueError("time and level_m must contain finite samples")

        order = np.argsort(t)
        self.time = t[order]
        self.level_m = wl[order]

    def get_water_level(self, time=None):
        if time is None:
            return float(np.nanmean(self.level_m))

        query = np.asarray(time, dtype=float)
        if self.time.size == 1:
            return np.full(query.shape, self.level_m[0], dtype=float)

        out = np.interp(
            query.ravel(),
            self.time,
            self.level_m,
            left=self.level_m[0],
            right=self.level_m[-1],
        )
        return out.reshape(query.shape)