from __future__ import annotations
from abc import ABC, abstractmethod
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from typing import List


from .WaterLevel import WaterLevel, ConstantWaterLevel, VaryingWaterLevel
from .LoessNNInterpolator import LoessNNInterpolator



"""
    Abstract class defining how bathymetry is provided to ADCPtools.

    Subclasses need to implement the 'get_bed_elev' method.

    Attributes:
        water_level (WaterLevel): Defines the water level. Needed to compute depth given position and time.

    Methods:
        get_bed_elev(x, y): Returns the bed elevation at the locations given in x, y.
        get_depth(x, y, time): Computes the depth given the position x, y and time.
        plot(*args, **kwargs): Plots the bathymetry. Subclasses should override this method.

"""

class Bathymetry(ABC):

    def __init__(self, *args) -> None:
        self.water_level: WaterLevel = ConstantWaterLevel() # Default value if no WaterLevel is provided
        for arg in args:
            if isinstance(arg, WaterLevel):
                self.water_level = arg

    @abstractmethod 
    def get_bed_elev(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")

        raise NotImplementedError
    
    def get_depth(self, x: np.ndarray, y: np.ndarray, time=None) -> np.ndarray:
        """Compute water depth from bathymetry and water level."""
        bed = self.get_bed_elev(x, y)
        return self.water_level.get_depth(bed, time=time)

    @abstractmethod 
    def plot(self, *args, **kwargs):
        """Subclasses should implement plotting."""
        raise NotImplementedError("Subclasses should implement plot().")



"""
    Creates bathymetry from scattered input points

    The bathymetry is obtained by interpolating the bathymetry from scattered input points
    defined in the property 'known' using the Interpolator object defined in the property 'interpolator'.

    Attributes:
        known (np.ndarray): Known scattered input points (3 x N array of x, y, z coordinates).
        interpolator (LoessNNInterpolator): Interpolator object performing the interpolation.

"""


class BathymetryScatteredPoints(Bathymetry):

    def __init__(self, *args, water_level: WaterLevel | None = None, known: np.ndarray | None = None) -> None:

        super().__init__(*args)
        self._known = np.empty((3, 0), dtype=float)
        self._interpolator = LoessNNInterpolator()

        vmadcp = None
        filter_objs: List = []
        interpolator = None

        water_level_explicit = False
        
        for arg in args:
            if isinstance(arg, np.ndarray):
                self.set_known(arg)
            elif hasattr(arg, "interpolate"):
                interpolator = arg
            elif hasattr(arg, "get_water_level"):
                self.water_level = arg
                water_level_explicit = True
            elif hasattr(arg, "bed_position"):
                vmadcp = arg
            elif hasattr(arg, "all_cells_bad"):
                filter_objs.append(arg)
            else:
                warnings.warn(f"Unhandled input of type: {type(arg)}")       

        if interpolator is not None:
            self.interpolator = interpolator

        if water_level is not None:
            self.water_level = water_level
            water_level_explicit = True

        if vmadcp is not None:
            filter_obj = filter_objs[0] if filter_objs else None
            self.known_from_vmadcp(vmadcp, filter_obj=filter_obj)

            if not water_level_explicit and hasattr(vmadcp, "water_level_object"):
                self.water_level = vmadcp.water_level_object
    
    @property
    def known(self) -> np.ndarray:
        return self._known

    @known.setter
    def known(self, value: np.ndarray) -> None:
        self.set_known(value)

    @property
    def interpolator(self) -> LoessNNInterpolator:
        return self._interpolator

    @interpolator.setter
    def interpolator(self, value: LoessNNInterpolator) -> None:
        self._interpolator = value
        self._set_interpolator_known()


    def set_known(self, known: np.ndarray) -> None:

        known = np.asarray(known, dtype=float)
        if known.ndim != 2 or known.shape[0] != 3:
            raise ValueError("known must be shaped (3, N)")
        finite = np.all(np.isfinite(known), axis=0)
        self._known = known[:, finite]
        self._set_interpolator_known()
    
    def _set_interpolator_known(self) -> None:

        self._interpolator.known = self._known
        if hasattr(self._interpolator, "reset_interpolant"):
            self._interpolator.reset_interpolant()


    def known_from_vmadcp(self, vmadcp, filter_obj=None) -> None:
        """Populate known points from a VMADCP-like object."""

        if not hasattr(vmadcp, "bed_position"):
            raise AttributeError("vmadcp must provide bed_position")

        tpos = np.asarray(vmadcp.bed_position, dtype=float)
        if tpos.ndim < 3:
            raise ValueError("bed_position must be at least 3D")

        if filter_obj is not None and hasattr(filter_obj, "all_cells_bad"):

            bad = np.asarray(filter_obj.all_cells_bad(vmadcp), dtype=bool).reshape(-1)

            if bad.size > 0:
                if bad.size == tpos.shape[1]:
                    tpos[:, bad, ...] = np.nan
                else:
                    warnings.warn(
                        f"Taille du masque de filtre ({bad.size}) incompatible "
                        f"avec le nombre d'ensembles de bed_position ({tpos.shape[1]}); "
                        "aucun filtrage appliqué."
                    )

        if tpos.shape[-1] < 3:
            raise ValueError("bed_position last axis must contain x, y, z")

        xpos = tpos[..., 0]
        ypos = tpos[..., 1]
        zpos = tpos[..., 2]

        isfin = np.isfinite(xpos) & np.isfinite(ypos) & np.isfinite(zpos)
        self.set_known(np.vstack((xpos[isfin], ypos[isfin], zpos[isfin])))

    def get_bed_elev(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Returns bed elevation at the given query points (x, y)."""

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have same shape")
        
        if self.known.shape[1] == 0:
            return np.full(x_arr.shape, np.nan, dtype=float)

        # z = self.interpolator.interpolate(np.vstack((x_arr.ravel(), y_arr.ravel())))
        # z = np.asarray(z, dtype=float).reshape(x_arr.shape)

        ## 03/08
        origin_shape = x_arr.shape
        x_flat = x_arr.ravel()
        y_flat = y_arr.ravel()

        valid_query = np.isfinite(x_flat) & np.isfinite(y_flat)

        z_flat = np.full(x_flat.shape, np.nan, dtype=float)

        if np.any(valid_query):
            z_valid = self.interpolator.interpolate(
                np.vstack((x_flat[valid_query], y_flat[valid_query]))
            )
            z_flat[valid_query] = np.asarray(z_valid, dtype=float)

        z = z_flat.reshape(origin_shape)

        z_min = float(np.min(self.known[2, :]))
        z_max = float(np.max(self.known[2, :]))
        finite_z = np.isfinite(z)
        z[finite_z] = np.clip(z[finite_z], z_min, z_max)
        #


        return z

    def plot_residuals(self, ax=None):
        """Plot interpolation residuals at the known input points."""

        if self.known.shape[1] == 0:
            return None

        z_interp = self.get_bed_elev(self.known[0], self.known[1])
        residuals = z_interp - self.known[2]

        if ax is None:
            fig, ax = plt.subplots()

        sc = ax.scatter(self.known[0], self.known[1], c=residuals, s=5, cmap="coolwarm")
        mean_res = np.nanmean(residuals)
        std_res = np.nanstd(residuals)
        sc.set_clim(mean_res - 2 * std_res, mean_res + 2 * std_res)

        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Residuals")
        ax.set_aspect("equal", adjustable="box")

        return ax

    def plot(self, *args, **kwargs):
        """Plot known points and the interpolated bathymetry surface."""

        ax = kwargs.get("ax")
        return_handles = kwargs.get("return_handles", False)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        hp = None
        ht = None

        if self.known.shape[1] > 0:
            hp = ax.scatter(
                self.known[0], 
                self.known[1], 
                self.known[2], 
                s=2, 
                c="k",
                label="Known points"
            )

            if self.known.shape[1] >= 3:
                tri = Triangulation(self.known[0], self.known[1])
                z_interp = self.get_bed_elev(self.known[0], self.known[1])
                ht = ax.plot_trisurf(
                    tri, 
                    z_interp, 
                    cmap="viridis", 
                    linewidth=0.0, 
                    alpha=0.7,
                    label="Interpolated surface")

                if hasattr(ht, "_facecolor3d"):
                    ht._facecolors2d = ht._facecolor3d
                if hasattr(ht, "_edgecolor3d"):
                    ht._edgecolors2d = ht._edgecolor3d

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z bed")
        ax.set_title("Bathymetry from Scattered Points")

        if return_handles:
            return hp, ht
        return ax
