from __future__ import annotations

from typing import Any

import numpy as np

from .VelocitySolver import VelocitySolver


class TimeBasedVelocitySolver(VelocitySolver):
    """Velocity solver that follows the MATLAB time-based solver structure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_solver_input(self):
        if self.adcp is None:
            raise ValueError("TimeBasedVelocitySolver requires an adcp object")

        vpos = np.asarray(self.adcp.depth_cell_position)
        if vpos.ndim == 4 and vpos.shape[2] > 1:
            vpos = np.nanmean(vpos, axis=2, keepdims=True)
            vpos = np.repeat(vpos, 4, axis=2)

        vdat = np.asarray(self.adcp.water_velocity)
        xform = np.asarray(self.adcp.xform)
        if xform.ndim == 4 and xform.shape[3] >= 4:
            xform = xform[..., :3]
        return vpos, vdat, xform

    def solve(self, n_obs: np.ndarray, sigma_obs: np.ndarray,
              beam_obs: np.ndarray, xform_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.solve_cellwise(n_obs, sigma_obs, beam_obs, xform_obs)
