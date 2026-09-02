from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Callable
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .VMADCP import ADCP  

"""
    Generic class to implement filters for ADCP objects .

    Methods:
        bad(adcp): Returns a boolean mask marking bad cells (ncells x nensembles x nbeams).
        all_cells_bad(adcp): Returns a boolean mask marking ensembles where ALL cells are bad.
        any_cells_bad(adcp): Returns a boolean mask marking ensembles where ANY cell is bad.
        plot(adcp): Plots the bad cells for each filter and the combined result.

"""

class Filter :

    def __init__(self) -> None:
        self._description: str = "Dummy filter"

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        self._description = value

    def bad(self, adcp: "ADCP") -> np.ndarray:
        """ Returns a boolean mask marking bad cells for the given ADCP object."""

        return self._expand_bad(lambda obj, adcp: obj._bad_int(adcp), adcp)
    
    def all_cells_bad(self, adcp: "ADCP") -> np.ndarray:
        """ Returns a boolean mask marking ensembles where ALL cells are bad."""

        return self._expand_bad(
            lambda obj, adcp: np.all(np.all(obj._bad_int(adcp), axis=0), axis=1),
            adcp
        )
    
    def any_cells_bad(self, adcp: "ADCP") -> np.ndarray:
        """ Returns a boolean mask marking ensembles where ANY cell is bad."""

        return self._expand_bad(
            lambda obj, adcp: np.any(np.any(obj._bad_int(adcp), axis=0), axis=1),
            adcp
        )
    
    def plot(self, adcp: "ADCP") -> None:
        """ Plots the bad cells for each filter and the combined result."""

        filters = [self] if not isinstance(self, list) else self
        n_filters = len(filters)
        n_beams = adcp.nbeams

        fig, axs = plt.subplots(
            n_filters + 1,
            n_beams,
            figsize=(5 * n_beams, 5 * (n_filters + 1)),
            squeeze=False
        )

        for co, filt in enumerate(filters):
            bad_mask = filt._bad_int(adcp) 
            for cb in range(n_beams):
                ax = axs[co, cb]
                im = ax.imshow(bad_mask[:, :, cb].T, aspect="auto", origin="lower")
                ax.set_title(filt.description)
                plt.colorbar(im, ax=ax)
        
        combined_bad = self.bad(adcp)
        for cb in range(n_beams):
            ax = axs[n_filters, cb]
            im = ax.imshow(combined_bad[:, :, cb].T, aspect="auto", origin="lower")
            ax.set_title("All filters")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

    
    def _bad_int(self, adcp: "ADCP") -> np.ndarray:
        """ Internal method to compute the bad cells mask for a single filter."""

        return np.zeros((adcp.ncells, adcp.nensembles, adcp.nbeams), dtype=bool)
    
    def _expand_bad(
        self,
        func: Callable[[Filter, "ADCP"], np.ndarray],
        adcp: "ADCP"
    ) -> np.ndarray:
        """ Combines filter functions on object arrays (private method)."""

        filters = [self] if not isinstance(self, list) else self

        if not filters:
            return np.array(False)

        if len(filters) == 1:
            return func(filters[0], adcp)

        combined_bad = func(filters[0], adcp)
        for filt in filters[1:]:
            combined_bad = combined_bad | func(filt, adcp)

        return combined_bad
    
    @classmethod
    def combine(cls, *filters: "Filter") -> List["Filter"]:
        return list(filters)

    def __and__(self, other: "Filter") -> List["Filter"]:
        return self.combine(self, other)

    def __or__(self, other: "Filter") -> List["Filter"]:
        return self.combine(self, other)