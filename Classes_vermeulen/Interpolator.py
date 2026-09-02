from __future__ import annotations
from abc import ABC

import numpy as np


class Interpolator(ABC):
    """Base interpolator class.
    Subclasses should define the interpolate function."""

    def __init__(self, known: np.ndarray | None = None, density_reduction: int = 1) -> None:
        self._known = np.empty((3, 0), dtype=float)
        self.density_reduction = density_reduction
        if known is not None:
            self.known = known

    @property
    def density_reduction(self) -> int:
        return self._density_reduction

    @density_reduction.setter
    def density_reduction(self, value: int) -> None:
        val = int(value)
        if val <= 0:
            raise ValueError("density_reduction must be a strictly positive integer")
        self._density_reduction = val

    @property
    def known(self) -> np.ndarray:
        if self._known.size == 0:
            return self._known
        step = int(self.density_reduction)
        return self._known[:, ::step]

    @known.setter
    def known(self, value: np.ndarray) -> None:
        known = np.asarray(value, dtype=float)
        if known.ndim != 2 or known.shape[0] < 2:
            raise ValueError("known must be a 2D array with D+1 rows")
        if not np.all(np.isfinite(known)):
            raise ValueError("known must contain only finite values")
        self._known = known
        self.on_known_updated()

    # Returns the number of dimensions of the input points :
    @property
    def n_dims(self) -> int:
        return self.known.shape[0] - 1

    def on_known_updated(self) -> None:
        """Hook for subclasses to invalidate cached interpolants."""
        return None

    def interpolate(self, query_position: np.ndarray) -> np.ndarray:
        query = np.asarray(query_position, dtype=float)
        if query.ndim != 2:
            raise ValueError("query_position must be a 2D real array")
        if query.shape[0] != self.n_dims:
            raise AssertionError(f"query_position must have {self.n_dims} dimensions")
        raise NotImplementedError
