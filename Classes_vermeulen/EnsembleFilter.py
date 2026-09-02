from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .VMADCP import ADCP

from .Filter import Filter


"""
    Class to filter out ensembles.

    Attributes:
        bad_ensembles (np.ndarray): Boolean array marking bad ensembles (1D array of shape (n_ensembles,)).

"""

class EnsembleFilter(Filter):

    def __init__(self, *args) -> None:

        super().__init__() 
        self._description = "Ensemble filter"
        self.bad_ensembles: np.ndarray = np.array([], dtype=bool)

        for arg in args:
            if isinstance(arg, (np.ndarray, list)):
                self.bad_ensembles = np.asarray(arg, dtype=bool).reshape(-1)
            elif hasattr(arg, "nensembles"):  
                self.bad_ensembles = np.zeros((arg.nensembles,), dtype=bool)
    
    def _bad_int(self, adcp: "ADCP") -> np.ndarray:

        if self.bad_ensembles.size == 0:
            self.bad_ensembles = np.zeros((adcp.nensembles,), dtype=bool)
        
        if self.bad_ensembles.size != adcp.nensembles:
            raise ValueError(
                "The number of elements in bad_ensembles must match the number of ensembles in the ADCP object."
            )
        return np.tile(
            self.bad_ensembles.reshape(1, -1, 1),
            (adcp.ncells, 1, adcp.nbeams)
        )

    # def all_cells_bad(self, vmadcp: Union["ADCP", object]) -> np.ndarray:

    #     n_ens = getattr(vmadcp, "nensembles", 0)
    #     if n_ens <= 0 and hasattr(vmadcp, "horizontal_position"):
    #         hp = np.asarray(vmadcp.horizontal_position)
    #         if hp.ndim == 2:
    #             n_ens = hp.shape[1]

    #     if self.bad_ensembles.size == 0:
    #         return np.zeros((max(n_ens, 0),), dtype=bool)

    #     if n_ens <= 0:
    #         return self.bad_ensembles.copy()

    #     out = np.zeros((n_ens,), dtype=bool)
    #     n_copy = min(n_ens, self.bad_ensembles.size)
    #     out[:n_copy] = self.bad_ensembles[:n_copy]
    #     return out


    def all_cells_bad(self, vmadcp: Union["ADCP", object]) -> np.ndarray:
        n_ens = getattr(vmadcp, "nensembles", 0)
        if n_ens <= 0 and hasattr(vmadcp, "horizontal_position"):
            hp = np.asarray(vmadcp.horizontal_position)
            if hp.ndim == 2:
                n_ens = hp.shape[1]

        if self.bad_ensembles.size == 0:
            self.bad_ensembles = np.zeros((n_ens,), dtype=bool)

        if n_ens <= 0:
            return self.bad_ensembles.copy()

        return self.bad_ensembles.copy()

        
    @classmethod
    def from_qrevint_transect(cls, transect: object) -> "EnsembleFilter":
        try:
            valid = np.asarray(transect.depths.bt_depths.valid_data, dtype=bool).reshape(-1)
            return cls(bad_ensembles=~valid)
        except Exception:
            return cls()


