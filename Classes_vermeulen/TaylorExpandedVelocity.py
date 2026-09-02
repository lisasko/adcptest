from __future__ import annotations

import numpy as np

from .VelocityModel import VelocityModel


class TaylorExpandedVelocity(VelocityModel):
    """Velocity model based on low-order Taylor expansions."""

    def __init__(
        self,
        s_order=(0, 0, 0),
        n_order=(0, 0, 0),
        z_order=(0, 0, 0),
        sigma_order=(0, 0, 0),
        time_order=(0, 0, 0),
    ) -> None:
        self.s_order = np.asarray(s_order, dtype=int).reshape(3)
        self.n_order = np.asarray(n_order, dtype=int).reshape(3)
        self.z_order = np.asarray(z_order, dtype=int).reshape(3)
        self.sigma_order = np.asarray(sigma_order, dtype=int).reshape(3)
        self.time_order = np.asarray(time_order, dtype=int).reshape(3)

    def get_npars(self) -> np.ndarray:
        return (
            np.ones((3,), dtype=int)
            + self.s_order
            + self.n_order
            + self.z_order
            + self.sigma_order
            + self.time_order
        )

    @staticmethod
    def taylor_expand(var: np.ndarray, order: int) -> np.ndarray:
        if order <= 0:
            return np.empty((np.asarray(var).size, 0), dtype=float)
        vec = np.asarray(var, dtype=float).reshape(-1, 1)
        num = np.cumprod(np.repeat(vec, order, axis=1), axis=1)
        den = np.cumprod(np.cumsum(np.ones_like(num), axis=1), axis=1)
        return num / den

    def _combine(self, dim: int, d_time, d_s, d_n, d_z, d_sigma) -> np.ndarray:
        return np.column_stack(
            [
                np.ones((np.asarray(d_time).size, 1), dtype=float),
                self.taylor_expand(d_time, int(self.time_order[dim])),
                self.taylor_expand(d_s, int(self.s_order[dim])),
                self.taylor_expand(d_n, int(self.n_order[dim])),
                self.taylor_expand(d_z, int(self.z_order[dim])),
                self.taylor_expand(d_sigma, int(self.sigma_order[dim])),
            ]
        )

    def get_model(self, d_time, d_s, d_n, d_z, d_sigma):
        return (
            self._combine(0, d_time, d_s, d_n, d_z, d_sigma),
            self._combine(1, d_time, d_s, d_n, d_z, d_sigma),
            self._combine(2, d_time, d_s, d_n, d_z, d_sigma),
        )
