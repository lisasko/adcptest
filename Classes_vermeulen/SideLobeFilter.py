from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np

from .VMADCP import CoordinateSystem


class Filter(ABC):
    description = "Dummy filter"

    def bad(self, adcp: Any):
        return self.bad_int(adcp)

    def all_cells_bad(self, adcp: Any):
        bad = np.asarray(self.bad(adcp), dtype=bool)
        if bad.ndim != 3:
            raise ValueError("bad mask must have shape (ncells, nensembles, nbeams)")
        return np.all(np.all(bad, axis=0), axis=-1)

    def any_cells_bad(self, adcp: Any):
        bad = np.asarray(self.bad(adcp), dtype=bool)
        if bad.ndim != 3:
            raise ValueError("bad mask must have shape (ncells, nensembles, nbeams)")
        return np.any(np.any(bad, axis=0), axis=-1)

    def bad_int(self, adcp: Any):
        raise NotImplementedError


class SideLobeFilter(Filter):
    """
    Vermeulen-style side lobe filter.

    This follows the geometry-based logic from SideLobeFilter.m:
    - take the highest detected bed level per ensemble,
    - remove half a cell size,
    - use the vertical component of the beam orientation,
    - compare against the cell vertical positions.
    """

    def __init__(self) -> None:
        self.description = "Side lobe filter"

    def bad_int(self, adcp: Any):
        water_velocity = np.asarray(getattr(adcp, "water_velocity", np.empty((0, 0, 0))), dtype=float)
        if water_velocity.size == 0:
            return np.zeros_like(water_velocity, dtype=bool)

        if not hasattr(adcp, "bed_position"):
            raise AttributeError("SideLobeFilter requires adcp.bed_position")
        if not hasattr(adcp, "cellsize"):
            raise AttributeError("SideLobeFilter requires adcp.cellsize")
        if not hasattr(adcp, "xform"):
            raise AttributeError("SideLobeFilter requires adcp.xform")
        if not hasattr(adcp, "depth_cell_offset"):
            raise AttributeError("SideLobeFilter requires adcp.depth_cell_offset")

        bed_position = np.asarray(adcp.bed_position, dtype=float)
        if bed_position.size == 0:
            return np.zeros_like(water_velocity, dtype=bool)

        bed_z = self._bed_z_per_ensemble(bed_position)

        cellsize = np.asarray(adcp.cellsize, dtype=float).reshape(-1)
        if cellsize.size == 1 and bed_z.size > 1:
            cellsize = np.full(bed_z.shape, float(cellsize[0]), dtype=float)
        if cellsize.size != bed_z.size:
            raise ValueError(
                f"cellsize has length {cellsize.size}, but bed_position implies {bed_z.size} ensembles"
            )

        # MATLAB equivalent:
        # bed_range = -permute(max(adcp.bed_offset,[],3),[1 2 4 3]);
        # bed_range = bed_range(:,:,3) - adcp.cellsize/2;
        # Here bed_position z is negative downward in this codebase, so -bed_z gives a positive range.
        bed_range = (-bed_z) - (cellsize / 2.0)

        xform = np.asarray(adcp.xform, dtype=float)
        if xform.ndim != 4 or xform.shape[-1] != 3:
            raise ValueError("adcp.xform must have shape (ncells, nensembles, nbeams, 3)")

        # MATLAB: tm = -adcp.xform(...); tm(:,:,:,[1 2 4]) = []; keep vertical component only.
        # In this Python port, xform[..., 2] is the vertical component, and the MATLAB sign is preserved by the minus.
        tm_vertical = -xform[..., 2]

        vel_pos = np.asarray(adcp.depth_cell_offset(CoordinateSystem.Earth), dtype=float)
        if vel_pos.ndim != 4 or vel_pos.shape[-1] != 3:
            raise ValueError("depth_cell_offset(CoordinateSystem.Earth) must return shape (ncells, nensembles, nbeams, 3)")

        vel_pos_z = vel_pos[..., 2]

        # Broadcast bed_range across cells and beams.
        min_lev = bed_range.reshape(1, -1, 1) * tm_vertical
        bad = vel_pos_z < min_lev
        return bad

    @staticmethod
    def _bed_z_per_ensemble(bed_position: np.ndarray) -> np.ndarray:
        if bed_position.ndim == 4 and bed_position.shape[-1] == 3:
            # Expected adapter shape: (1, n_ens, n_beams, 3)
            return np.nanmax(bed_position[..., 2], axis=2).reshape(-1)

        if bed_position.ndim == 3 and bed_position.shape[-1] == 3:
            return np.nanmax(bed_position[..., 2], axis=1).reshape(-1)

        if bed_position.ndim == 2 and bed_position.shape[1] == 3:
            return bed_position[:, 2].reshape(-1)

        raise ValueError(
            "Unsupported bed_position shape. Expected (..., 3) with ensemble axis present."
        )