from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d

from .Interpolator import Interpolator
from .loess_nd import loess_nd


class LoessInterpolator(Interpolator):
    """Loess interpolator class (MATLAB-like options)."""

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

        self._interp_linear = None
        self._interp_nearest = None

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
        self._interp_linear = None
        self._interp_nearest = None

    def _build_interpolant(self) -> None:
        known = self.known
        if known.size == 0:
            self._interp_linear = None
            self._interp_nearest = None
            return

        x = known[:-1, :].T  # (N,D)
        y = known[-1, :]     # (N,)

        # smooth at known locations (as LoessNN MATLAB workflow)
        smooth_y = loess_nd(
            x=x,
            v=y,
            xi=x,
            span=self.span,
            niter=self.robust_iterations,
            order=self.order,
            nthreads=self.n_threads,
        )

        if known.shape[0] == 2:
            # 1D
            x_1d = known[0, :]
            order = np.argsort(x_1d)
            x_sorted = x_1d[order]
            y_sorted = np.asarray(smooth_y, dtype=float)[order]

            self._interp_linear = interp1d(
                x_sorted,
                y_sorted,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            self._interp_nearest = interp1d(
                x_sorted,
                y_sorted,
                kind="nearest",
                bounds_error=False,
                fill_value=(y_sorted[0], y_sorted[-1]),
                assume_sorted=True,
            )
        else:
            # N-D (linear inside hull, nearest outside)
            self._interp_linear = LinearNDInterpolator(x, smooth_y, fill_value=np.nan)
            self._interp_nearest = NearestNDInterpolator(x, smooth_y)

    def interpolate(self, query_position: np.ndarray) -> np.ndarray:
        query_position = np.asarray(query_position, dtype=float)
        if query_position.ndim != 2:
            raise ValueError("query_position must be a 2D real array")
        if query_position.shape[0] != self.n_dims:
            raise AssertionError(f"query_position must have {self.n_dims} dimensions")

        if self._interp_linear is None or self._interp_nearest is None:
            self._build_interpolant()

        if self._interp_linear is None:
            return np.full(query_position.shape[1], np.nan, dtype=float)

        pts = query_position.T
        values = self._interp_linear(pts)
        nan_mask = ~np.isfinite(values)
        if np.any(nan_mask):
            values[nan_mask] = self._interp_nearest(pts[nan_mask])

        return np.asarray(values, dtype=float)