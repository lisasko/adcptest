from abc import ABC, abstractmethod
import numpy as np 
from typing import TYPE_CHECKING


"""
Abstract class defining the horizontal position of the ADCP

"""

class ADCPHorizontalPosition(ABC):

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @staticmethod
    def has_data(self, adcp) -> np.ndarray:

        from .VMADCP import VMADCP
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
    def horizontal_position(self, adcp):

        from .VMADCP import VMADCP
        fprovider = np.where(ADCPHorizontalPosition.has_data(self, adcp))[0]
        if len(fprovider) == 0:
            pos = np.full((2, adcp.nensembles), np.nan)
        else:
            pos = self[fprovider].get_horizontal_position(adcp)
        return pos

    @abstractmethod
    def get_horizontal_position(self, adcp):
        raise NotImplementedError
    

"""
Defines the ADCP position as a fixed position (moored deployment)

"""

class ADCPFixedHorizontalPosition(ADCPHorizontalPosition):
    
    def __init__(self, pos=(0.0, 0.0)):
        self.description = "Position horizontale fixe par défaut"
        self.position = np.asarray(pos, dtype=float).reshape(2, 1)

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str):
        self._description = value


    def get_horizontal_position(self, adcp):
        return np.repeat(self.position, adcp.nensembles, axis=1)