## Python class trying to reproduct ShipVelocityProvider.m, ShipVelocityFromBT.m and ShipVelocityFromGPS.m actions.

"""
Le comportement de base est déjà couvert dans les codes de QRevInt:

- TransectData.py compare déjà BT et GPS.
- BoatStructure.py calcule les tracks.

Dans la pipeline Vermeulen, on aura plus qu'à choisir entre bt_vel et gga_vel 
et injecter ship_velocity. 

"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .VMADCP import CoordinateSystem
from .get_gpsvel import get_gpsvel


class ShipVelocityProvider(ABC):
    """
    Compatibility layer for the ShipVelocity hierarchy.

    The qrev logic already lives in BoatStructure, GPSData and get_gpsvel.
    This module only exposes a clean, MATLAB-like API for callers that want
    ship-velocity providers as objects.
    """

    def ship_velocity(self, adcp, dst=CoordinateSystem.Earth):
        """
        Public MATLAB-like entry point.

        Parameters
        ----------
        adcp:
            Transect-like or VMADCP-like object.
        dst:
            Destination coordinate system. The current Python pipeline keeps
            boat velocities aligned through the existing qrevint processing,
            so this wrapper mainly preserves the API shape.
        """
        return self.get_ship_velocity(adcp, dst)

    @abstractmethod
    def get_ship_velocity(self, adcp, dst=CoordinateSystem.Earth):
        raise NotImplementedError


class ShipVelocityFromBT(ShipVelocityProvider):
    """
    Ship velocity based on bottom track data.

    In the Python port, the BT velocity is already stored inside transect.boat_vel.bt_vel.
    This wrapper exposes it with the same class-based API as MATLAB.
    """

    def get_ship_velocity(self, adcp, dst=CoordinateSystem.Earth):
        boat_vel = getattr(adcp, "boat_vel", None)
        bt_vel = getattr(boat_vel, "bt_vel", None) if boat_vel is not None else None

        if bt_vel is None:
            n_ensembles = int(getattr(adcp, "n_ensembles", 0))
            return np.full((n_ensembles, 3), np.nan, dtype=float)

        u = np.asarray(getattr(bt_vel, "u_processed_mps", np.empty((0,))), dtype=float).reshape(-1)
        v = np.asarray(getattr(bt_vel, "v_processed_mps", np.empty((0,))), dtype=float).reshape(-1)

        n_ensembles = min(u.size, v.size)
        if n_ensembles == 0:
            return np.empty((0, 3), dtype=float)

        vel = np.zeros((n_ensembles, 3), dtype=float)
        vel[:, 0] = u[:n_ensembles]
        vel[:, 1] = v[:n_ensembles]

        # MATLAB ShipVelocityFromBT returns the BT velocity in the requested frame.
        # In the current Python pipeline, BoatStructure already manages the active
        # navigation reference and coordinate system upstream.
        return vel


class ShipVelocityFromGPS(ShipVelocityProvider):
    """
    Ship velocity derived from GPS positions.

    This wrapper delegates to get_gpsvel, which already reproduces the
    central-difference GPS-track logic used by the qrev/MATLAB workflow.
    """

    def __init__(self, nav_ref: str = "gga_vel", fill_missing: bool = True) -> None:
        self.nav_ref = nav_ref
        self.fill_missing = fill_missing

    def get_ship_velocity(self, adcp, dst=CoordinateSystem.Earth):
        # get_gpsvel already extracts the ship track from the transect and
        # computes Earth-frame velocity from the track derivative.
        vel = get_gpsvel(adcp, nav_ref=self.nav_ref, fill_missing=self.fill_missing)

        # If you later need a non-Earth destination frame, the safest place to
        # add that conversion is here, once the exact VMADCP xform usage is fixed
        # for your dataset and reference frame.
        return vel