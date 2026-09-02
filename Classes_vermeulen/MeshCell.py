from __future__ import annotations

import numpy as np
from matplotlib.path import Path


class MeshCell:
    """Polygonal mesh cell compatible with MATLAB MeshCell semantics."""

    def __init__(self, coordinates: np.ndarray | None = None) -> None:
        self._coordinates = np.empty((2, 0), dtype=float)
        if coordinates is not None:
            self.coordinates = coordinates

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @coordinates.setter
    def coordinates(self, val: np.ndarray) -> None:
        coordinates = np.asarray(val, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] != 2:
            raise ValueError("coordinates must be shaped (2, N)")
        if coordinates.shape[1] >= 1 and not np.allclose(coordinates[:, 0], coordinates[:, -1]):
            coordinates = np.column_stack((coordinates, coordinates[:, 0]))
        self._coordinates = coordinates

    @property
    def n_coordinates(self) -> int:
        return self._coordinates.shape[1]

    def is_in_cell(self, qcoord: np.ndarray, dim: int = 1):
        qcoord = np.asarray(qcoord, dtype=float)
        if self.n_coordinates == 0:
            return np.array([], dtype=bool)

        if qcoord.ndim == 1:
            if qcoord.size != 2:
                raise ValueError("qcoord must contain 2 values")
            return Path(self._coordinates.T).contains_point(qcoord)

        if qcoord.shape[dim - 1] != 2 and qcoord.shape[dim] != 2:
            raise ValueError("qcoord must have size 2 along the selected dimension")

        if qcoord.shape[-1] == 2:
            pts = qcoord.reshape(-1, 2)
            inside = Path(self._coordinates.T).contains_points(pts)
            return inside.reshape(qcoord.shape[:-1])

        pts = np.moveaxis(qcoord, dim - 1, 0)
        pts = pts.reshape(2, -1).T
        inside = Path(self._coordinates.T).contains_points(pts)
        return inside.reshape(np.moveaxis(qcoord, dim - 1, 0).shape[1:])

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self._coordinates[0], self._coordinates[1], **kwargs)
        return ax