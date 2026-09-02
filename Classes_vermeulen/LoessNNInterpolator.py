from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d

from .LoessInterpolator import LoessInterpolator
from .loess_nd import loess_nd


class LoessNNInterpolator(LoessInterpolator):
    """Loess and nearest-neighbor interpolator."""

    def __init__(
        self,
        known: np.ndarray | None = None,
        span: float = 0.01,
        robust_iterations: int = 0,
        n_threads: int | None = None,
        order: int = 1,
        density_reduction: int = 1,
    ) -> None:

        ## 30/07
        self._interp_linear = None
        self._interp_nearest = None
        ##
        
        super().__init__(
            known=known,
            span=span,
            robust_iterations=robust_iterations,
            n_threads=n_threads,
            order=order,
            density_reduction=density_reduction,
        )
        self.update_int = True

    def on_known_updated(self) -> None:
        self.update_int = True
        # super().on_known_updated()
        self._interp_linear = None
        self._interp_nearest = None

    def reset_interpolant(self, *args, **kwargs) -> None:
        self.update_int = True


    def make_interpolant(self) -> None:

        known = self.known
        if known.size == 0:
            self._interp_linear = None
            self._interp_nearest = None
            self.update_int = False
            return

        x = known[:-1, :].T   # (N, D)
        y = known[-1, :]      # (N,)

        smooth_y = loess_nd(
            x=x, v=y, xi=x,
            span=self.span,
            niter=self.robust_iterations,
            order=self.order,
            nthreads=self.n_threads,
        )

        ## 19/08
        smooth_y = np.asarray(smooth_y, dtype=float)

        nan_mask_smooth = ~np.isfinite(smooth_y)
        n_nan_smooth = int(np.sum(nan_mask_smooth))
        if n_nan_smooth > 0:
            print(
                f"DEBUG LoessNNInterpolator : {n_nan_smooth}/{smooth_y.size} points "
                f"connus ont un pré-lissage local en échec (< 3 voisins pondérés "
                f"valides -- nuage trop peu dense pour span={self.span}). "
                f"Repli sur la valeur brute (non lissée) pour ces points."
            )
            smooth_y[nan_mask_smooth] = y[nan_mask_smooth]

        ##

        if known.shape[0] == 2:
            # cas 1D
            x_1d = known[0, :]
            order_idx = np.argsort(x_1d)
            x_sorted = x_1d[order_idx]
            y_sorted = np.asarray(smooth_y, dtype=float)[order_idx]
            self._interp_linear = interp1d(
                x_sorted, y_sorted, kind="linear",
                bounds_error=False, fill_value=np.nan, assume_sorted=True,
            )
            self._interp_nearest = interp1d(
                x_sorted, y_sorted, kind="nearest",
                bounds_error=False,
                fill_value=(y_sorted[0], y_sorted[-1]),
                assume_sorted=True,
            )
        else:
            self._interp_linear = LinearNDInterpolator(x, smooth_y, fill_value=np.nan)
            self._interp_nearest = NearestNDInterpolator(x, smooth_y)

        self.update_int = False

    def interpolate(self, query_position: np.ndarray) -> np.ndarray:
        # if self.update_int:
        #     self.make_interpolant()
        # return super().interpolate(query_position)

        query_position = np.asarray(query_position, dtype=float)
        if query_position.ndim != 2:
            raise ValueError("query_position must be a 2D real array")
        if query_position.shape[0] != self.n_dims:
            raise AssertionError(f"query_position must have {self.n_dims} dimensions")

        if self.update_int:
            self.make_interpolant()

        if self._interp_linear is None:
            return np.full(query_position.shape[1], np.nan, dtype=float)

        pts = query_position.T
        values = self._interp_linear(pts)
        nan_mask = ~np.isfinite(values)
        if np.any(nan_mask):
            # ÉTAPE 3 : extrapolation par plus proche voisin hors enveloppe
            values = np.asarray(values, dtype=float)
            values[nan_mask] = self._interp_nearest(pts[nan_mask])

        return np.asarray(values, dtype=float)
