from __future__ import annotations
from typing import Any
import warnings

import numpy as np

from .VMADCP import CoordinateSystem
from .VelocitySolver import VelocitySolver

def _ship_velocity_earth(adcp) -> np.ndarray:
    """
    Vitesse du bateau en repère Terre (m/s), par dérivée centrée de la
    trajectoire GPS. 
    """
    hp = np.asarray(adcp.horizontal_position, dtype=float) 
    t = VelocitySolver._to_seconds_like(np.asarray(adcp.time)) 

    n_ens = hp.shape[1]
    if n_ens < 3 or t.size < 3:
        return np.zeros((n_ens, 3), dtype=float)

    dt = t[2:] - t[:-2]
    dx = hp[0, 2:] - hp[0, :-2]
    dy = hp[1, 2:] - hp[1, :-2]

    with np.errstate(invalid="ignore", divide="ignore"):
        vx = dx / dt
        vy = dy / dt

    vx = np.concatenate(([vx[0]], vx, [vx[-1]]))
    vy = np.concatenate(([vy[0]], vy, [vy[-1]]))

    return np.stack([vx, vy, np.zeros_like(vx)], axis=-1) 


class LocationBasedVelocitySolver(VelocitySolver):
    """
    Location-based velocity solver.

    This mirrors the MATLAB class:
    - constructor delegates to VelocitySolver
    - get_solver_input returns position, beam velocity, and beam->earth transform
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        for cur_arg in args:
            if isinstance(cur_arg, (str, bytes)):
                warnings.warn(
                    f"Unhandled input of type: {type(cur_arg).__name__} on construction of "
                    "LocationBasedVelocitySolver object",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # def _get_water_velocity_beam(self):
    #     vel = getattr(self.adcp, "water_velocity", None)
    #     if vel is None:
    #         raise ValueError("LocationBasedVelocitySolver requires an adcp object with water_velocity")

    #     # MATLAB: obj.adcp.water_velocity(CoordinateSystem.Beam)
    #     # Python VMADCP currently exposes an array-like property, so we accept both.
    #     if callable(vel):
    #         return np.asarray(vel(CoordinateSystem.Beam), dtype=float)
    #     return np.asarray(vel, dtype=float)

    def _get_water_velocity_beam(self):

        adcp = self.adcp

        raw_beam = getattr(adcp, "water_velocity", None)
        if raw_beam is None:
            raise ValueError("adcp.water_velocity (données brutes) indisponible")
        raw_beam = np.asarray(raw_beam, dtype=float)

        ship_vel_earth = _ship_velocity_earth(adcp)

        # ## 04/08
        # t_dbg = VelocitySolver._to_seconds_like(np.asarray(adcp.time))
        # dt_dbg = t_dbg[2:] - t_dbg[:-2]
        # n_dt_zero = int(np.sum(np.abs(dt_dbg) < 1e-6))
        # n_dt_huge = int(np.sum(np.abs(dt_dbg) > 60.0))  # sauts > 1 min : probable frontière de transect
        # print(f"\nDEBUG dt central-diff : min={np.nanmin(dt_dbg):.4f}s, max={np.nanmax(dt_dbg):.4f}s, "
        #     f"dt≈0 (division par 0)={n_dt_zero}, dt>60s (frontière transect probable)={n_dt_huge}")
        # ##

        # ## 04/08
        # print(f"\nDEBUG ship_vel_earth: shape={ship_vel_earth.shape}, "
        #     f"norme moyenne={np.nanmean(np.linalg.norm(ship_vel_earth, axis=1)):.3f} m/s, "
        #     f"NaN count={int(np.sum(np.isnan(ship_vel_earth)))}")

        # xform = np.asarray(adcp.xform, dtype=float)
        # if xform.ndim != 4 or xform.shape[-1] < 3:
        #     raise ValueError("adcp.xform doit être de forme (ncells, nens, nbeams, 3)")

        # M = xform[0, :, :, :3]                     
        # M_T = np.transpose(M, (0, 2, 1))            
        # M_T_pinv = np.linalg.pinv(M_T)

        # ship_vel_beam = np.einsum("eij,ej->ei", M_T_pinv, ship_vel_earth)  # (nens, 4)

        # ## 04/08
        # print(f"DEBUG ship_vel_beam: norme moyenne={np.nanmean(np.linalg.norm(ship_vel_beam, axis=1)):.3f} m/s "
        #     f"(comparer à raw_beam std={np.nanstd(raw_beam):.3f} m/s), "
        #     f"NaN count={int(np.sum(np.isnan(ship_vel_beam)))}")

        ## 05/08
        xform = np.asarray(adcp.xform, dtype=float)
        if xform.ndim != 4 or xform.shape[-1] < 3:
            raise ValueError("adcp.xform doit être de forme (ncells, nens, nbeams, 3)")

        M = xform[0, :, :, :3] 

        ship_vel_beam = np.einsum("ebd,ed->eb", M, ship_vel_earth)  

        print(f"\nDEBUG ship_vel_beam (corrigé): norme moyenne="
            f"{np.nanmean(np.abs(ship_vel_beam)):.3f} m/s, "
            f"NaN count={int(np.sum(np.isnan(ship_vel_beam)))}")
        ##

        return raw_beam + ship_vel_beam[np.newaxis, :, :]


    def _get_xform_beam_to_earth(self):
        xform = getattr(self.adcp, "xform", None)
        if xform is None:
            raise ValueError("LocationBasedVelocitySolver requires an adcp object with xform")

        if callable(xform):
            try:
                xform = xform(CoordinateSystem.Beam, CoordinateSystem.Earth)
            except TypeError:
                xform = xform(CoordinateSystem.Beam)

        xform = np.asarray(xform, dtype=float)

        if xform.ndim == 4 and xform.shape[-1] >= 4:
            xform = xform[..., :3]

        return xform

    def get_solver_input(self):
        if self.adcp is None:
            raise ValueError("LocationBasedVelocitySolver requires an adcp object")

        vpos = np.asarray(getattr(self.adcp, "depth_cell_position"), dtype=float)
        vdat = self._get_water_velocity_beam()
        xform = self._get_xform_beam_to_earth()

        return vpos, vdat, xform