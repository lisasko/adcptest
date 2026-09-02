from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.spatial import cKDTree

from .Interpolator import Interpolator
from .loess_nd import loess_nd


class LoessInterpolator(Interpolator):
    """Loess interpolator class."""

    def __init__(
        self,
        known: np.ndarray | None = None,
        span: float = 0.01,
        robust_iterations: int = 0,
        n_threads: int | None = None,
        order: int = 1,
        density_reduction: int = 1,
    ) -> None:
        
        self._span = 0.01
        self._robust_iterations = 0
        self._n_threads = None
        self._order = 1

        # self._interp_linear = None
        # self._interp_nearest = None

        super().__init__(known=known, density_reduction=density_reduction)

        self.span = span
        self.robust_iterations = robust_iterations
        self.n_threads = n_threads
        self.order = order

    @property
    def span(self) -> float:
        return self._span

    @span.setter
    def span(self, value: float) -> None:
        value = float(value)
        if value <= 0 or not np.isfinite(value):
            raise ValueError("span must be positive and finite")
        self._span = value
        self.on_known_updated()

    @property
    def robust_iterations(self) -> int:
        return self._robust_iterations

    @robust_iterations.setter
    def robust_iterations(self, value: int) -> None:
        value = int(value)
        if value < 0:
            raise ValueError("robust_iterations must be nonnegative")
        self._robust_iterations = value
        self.on_known_updated()

    @property
    def n_threads(self) -> int | None:
        return self._n_threads

    @n_threads.setter
    def n_threads(self, value: int | None) -> None:
        if value is None:
            self._n_threads = None
        else:
            value = int(value)
            if value <= 0:
                raise ValueError("n_threads must be positive when provided")
            self._n_threads = value
        self.on_known_updated()

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, value: int) -> None:
        value = int(value)
        if value not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self._order = value
        self.on_known_updated()

    def on_known_updated(self) -> None:
        return None

    def _build_interpolant(self) -> None:
        return None

    def interpolate(self, query_position: np.ndarray) -> np.ndarray:
        query_position = np.asarray(query_position, dtype=float)

        if query_position.ndim != 2:
            raise ValueError("query_position must be a 2D real array")
        if query_position.shape[0] != self.n_dims:
            raise AssertionError(f"query_position must have {self.n_dims} dimensions")

        known = self.known
        if known.size == 0:
            return np.full(query_position.shape[1], np.nan, dtype=float)

        x_known = known[:-1, :].T  
        y_known = known[-1, :]    
        xi = query_position.T

        values = loess_nd(
            x=x_known,
            v=y_known,
            xi=xi,
            span=self.span,
            niter=self.robust_iterations,
            order=self.order,
            nthreads=self.n_threads,
        )
        values = np.asarray(values, dtype=float)

        nan_mask = ~np.isfinite(values)
        if np.any(nan_mask):
            tree = cKDTree(x_known)
            _, nearest_idx = tree.query(xi[nan_mask])
            values[nan_mask] = y_known[nearest_idx]

        return values 