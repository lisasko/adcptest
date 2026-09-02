from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .VMADCP import VMADCP
from .WaterLevel import WaterLevel, ConstantWaterLevel, VaryingWaterLevel

"""
Abstract class defining the vertical position of the ADCP
"""

class ADCPVerticalPosition(ABC):
    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @staticmethod
    def has_data(self, adcp) -> np.ndarray:
        from .VMADCP import VMADCP  # Import local
        if not isinstance(adcp, VMADCP):
            raise TypeError("adcp must be a VMADCP instance")

        if isinstance(self, (list, np.ndarray)):
            tf = np.zeros(len(self), dtype=bool)
            for i, obj in enumerate(self):
                tf[i] = obj._get_has_data(adcp)
            return tf
        else:
            return np.array([self._get_has_data(adcp)], dtype=bool)

    @staticmethod
    def vertical_position(self, adcp):
        from .VMADCP import VMADCP  # Import local
        fprovider = np.where(ADCPVerticalPosition.has_data(self, adcp))[0]
        if len(fprovider) == 0:
            pos = np.full((1, adcp.nensembles), np.nan)
        else:
            pos = self[fprovider].get_vertical_position(adcp)
        return pos

    @abstractmethod
    def get_vertical_position(self, adcp):
        raise NotImplementedError

"""
Defines the ADCP position as a fixed position (moored deployment)
"""
class ADCPFixedVerticalPosition(ADCPVerticalPosition):
    def __init__(self, pos=0.0):
        self.description = "Position verticale fixe par défaut"
        self.position = np.asarray(pos, dtype=float).reshape(1, 1)

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str):
        self._description = value

    def get_vertical_position(self, adcp):
        return np.repeat(self.position, adcp.nensembles, axis=1)


"""
Defines ADCP vertical position based on the water level
"""


class ADCPVerticalPositionFromWaterLevel(ADCPVerticalPosition):
    def __init__(self, water_level : WaterLevel, depth_transducer : float = 0.07):
        super().__init__()

        from .VMADCP import VMADCP  

        self.description = "Position verticale basée sur le niveau d'eau"

        self.water_level = water_level
        self.depth_transducer = depth_transducer

    def get_vertical_position(self, adcp):
        val = self.water_level.get_water_level(adcp.time) - self.depth_transducer
        return val

    @property
    def description(self) -> str:
        return self._description
    
    @description.setter
    def description(self, value: str):
        self._description = value

    def set_wl(self, src):
        self.water_level = src.water_level_object 