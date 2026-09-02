from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .Mesh import Mesh


"""
% Defines a sigma-z mesh
%   
%   The SigmaZetaMesh should be generated with a SigmaZetaMeshGenerator.
%
%   The mesh consists of verticals. Each vertical has a certain number of
%   cells. In each vertical the cells follow the bed. The name Sigma-Zeta
%   refers to the fact that this mesh is a hyrbid between a z-mesh and a
%   sigma-mesh. It tries to combine the best of both worlds: it follows
%   nicely the bed just like a sigma mesh would, but the vertical spacing
%   of the cells is constant, just like a z-mesh would. This means that
%   each cell will hold approximately the same number of adcp velocity
%   estimates.
%
%   Each mesh cell consists of six edges, called:
%           *         ---> top-middle
%         /   \
%       /       \
%     /          *    ---> top-right
%   *             |   ---> top-left 
%   |       *     |   ---> bottom-middle
%   |     /   \   |
%   |   /       \ |
%   | /           *   ---> bottom-right
%   *                 ---> bottom-left
%
%   Note that the left, middle and right edges are always vertically
%   stacked and share the same n-coordinate.
%
%   Data can be stored either in a vector with a value for each cell or in
%   a matrix which has a toplogy similar to the mesh. The indexing
%   properties (see below) help map between these formats.
"""

@dataclass
class SigmaZetaMesh(Mesh):

    n_left: np.ndarray
    n_middle: np.ndarray
    n_right: np.ndarray
    sig_bottom_left: np.ndarray
    sig_top_left: np.ndarray
    sig_bottom_mid: np.ndarray
    sig_top_mid: np.ndarray
    sig_bottom_right: np.ndarray
    sig_top_right: np.ndarray
    row_to_cell: np.ndarray
    col_to_cell: np.ndarray

    # Attributes populated by SigmaZetaMeshFromVMADCP.
    water_level: float = 0.0
    z_bottom_left: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    z_top_left: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    z_bottom_mid: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    z_top_mid: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    z_bottom_right: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    z_top_right: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    zb_left: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    zb_middle: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    zb_right: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    zb_all: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    nb_all: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=float))
    nw: np.ndarray = field(default_factory=lambda: np.empty((2, 0), dtype=float))
    col_to_mat: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    row_to_mat: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    mat_to_cell: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=bool))
    cell_to_mat: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    def __post_init__(self) -> None:
        self.n_left = np.asarray(self.n_left, dtype=float).reshape(-1)
        self.n_middle = np.asarray(self.n_middle, dtype=float).reshape(-1)
        self.n_right = np.asarray(self.n_right, dtype=float).reshape(-1)
        self.sig_bottom_left = np.asarray(self.sig_bottom_left, dtype=float).reshape(-1)
        self.sig_top_left = np.asarray(self.sig_top_left, dtype=float).reshape(-1)
        self.sig_bottom_mid = np.asarray(self.sig_bottom_mid, dtype=float).reshape(-1)
        self.sig_top_mid = np.asarray(self.sig_top_mid, dtype=float).reshape(-1)
        self.sig_bottom_right = np.asarray(self.sig_bottom_right, dtype=float).reshape(-1)
        self.sig_top_right = np.asarray(self.sig_top_right, dtype=float).reshape(-1)
        self.row_to_cell = np.asarray(self.row_to_cell, dtype=int).reshape(-1)
        self.col_to_cell = np.asarray(self.col_to_cell, dtype=int).reshape(-1)

    # @staticmethod
    # def fit_sig(n: np.ndarray, n0: float, n1: float, sig0: float, sig1: float) -> np.ndarray:
    #     den = (n1 - n0)
    #     if abs(den) < 1e-12:
    #         return np.zeros_like(np.asarray(n, dtype=float)) + 0.5 * (sig0 + sig1)
    #     return ((sig1 - sig0) / den) * (np.asarray(n, dtype=float) - n0) + sig0

    # Matlab ne protège pas le code n1 == n0 (il laisse la division produire inf ou nan).
    @staticmethod
    def fit_sig(n: np.ndarray, n0: float, n1: float, sig0: float, sig1: float) -> np.ndarray:
        n_arr = np.asarray(n, dtype=float)
        return ((sig1 - sig0) / (n1 - n0)) * (n_arr - n0) + sig0


    @property
    def ncells(self) -> int:
        return len(self.row_to_cell)

    def get_ncells(self) -> int:
        return self.ncells

    def _sigma_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return sigma edges, preferring MATLAB-like z->sigma when z geometry exists."""
        has_z = (
            self.z_bottom_left.size == self.ncells
            and self.z_top_left.size == self.ncells
            and self.z_bottom_mid.size == self.ncells
            and self.z_top_mid.size == self.ncells
            and self.z_bottom_right.size == self.ncells
            and self.z_top_right.size == self.ncells
        )
        has_zb = (
            self.zb_left.size > np.max(self.col_to_cell, initial=-1)
            and self.zb_middle.size > np.max(self.col_to_cell, initial=-1)
            and self.zb_right.size > np.max(self.col_to_cell, initial=-1)
        )

        if has_z and has_zb:
            zb_l = self.zb_left[self.col_to_cell]
            zb_m = self.zb_middle[self.col_to_cell]
            zb_r = self.zb_right[self.col_to_cell]
            sbl = self.z_to_sigma(self.z_bottom_left, zb_l)
            stl = self.z_to_sigma(self.z_top_left, zb_l)
            sbm = self.z_to_sigma(self.z_bottom_mid, zb_m)
            stm = self.z_to_sigma(self.z_top_mid, zb_m)
            sbr = self.z_to_sigma(self.z_bottom_right, zb_r)
            str_ = self.z_to_sigma(self.z_top_right, zb_r)
            return sbl, stl, sbm, stm, sbr, str_

        return (
            self.sig_bottom_left,
            self.sig_top_left,
            self.sig_bottom_mid,
            self.sig_top_mid,
            self.sig_bottom_right,
            self.sig_top_right,
        )

    @property
    def z_center(self) -> np.ndarray:
        if self.z_bottom_mid.size == self.ncells and self.z_top_mid.size == self.ncells:
            return 0.5 * (self.z_bottom_mid + self.z_top_mid)
        return self.sig_center

    @property
    def n_center(self) -> np.ndarray:
        if self.n_middle.size == 0 or self.col_to_cell.size == 0:
            return np.empty((0,), dtype=float)
        return self.n_middle[self.col_to_cell]

    def _sn2xy(self, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "xs") or self.xs is None:
            nan = np.full_like(np.asarray(n, dtype=float), np.nan, dtype=float)
            return nan, nan
        s = np.zeros_like(np.asarray(n, dtype=float), dtype=float)
        return self.xs.sn2xy(s, np.asarray(n, dtype=float))

    @property
    def x_left(self) -> np.ndarray:
        x, _ = self._sn2xy(self.n_left)
        return np.asarray(x, dtype=float)

    @property
    def x_middle(self) -> np.ndarray:
        x, _ = self._sn2xy(self.n_middle)
        return np.asarray(x, dtype=float)

    @property
    def x_right(self) -> np.ndarray:
        x, _ = self._sn2xy(self.n_right)
        return np.asarray(x, dtype=float)

    @property
    def y_left(self) -> np.ndarray:
        _, y = self._sn2xy(self.n_left)
        return np.asarray(y, dtype=float)

    @property
    def y_middle(self) -> np.ndarray:
        _, y = self._sn2xy(self.n_middle)
        return np.asarray(y, dtype=float)

    @property
    def y_right(self) -> np.ndarray:
        _, y = self._sn2xy(self.n_right)
        return np.asarray(y, dtype=float)

    @property
    def xb_all(self) -> np.ndarray:
        x, _ = self._sn2xy(self.nb_all)
        return np.asarray(x, dtype=float)

    @property
    def yb_all(self) -> np.ndarray:
        _, y = self._sn2xy(self.nb_all)
        return np.asarray(y, dtype=float)

    @property
    def xw(self) -> np.ndarray:
        x, _ = self._sn2xy(self.nw)
        return np.asarray(x, dtype=float)

    @property
    def yw(self) -> np.ndarray:
        _, y = self._sn2xy(self.nw)
        return np.asarray(y, dtype=float)

    @property
    def x_patch(self) -> np.ndarray:
        x, _ = self._sn2xy(self.n_patch)
        return np.asarray(x, dtype=float)

    @property
    def y_patch(self) -> np.ndarray:
        _, y = self._sn2xy(self.n_patch)
        return np.asarray(y, dtype=float)

    @property
    def sig_center(self) -> np.ndarray:
        _, _, sbm, stm, _, _ = self._sigma_edges()
        return 0.5 * (sbm + stm)

    @property
    def n_patch(self) -> np.ndarray:
        if self.ncells == 0:
            return np.empty((7, 0), dtype=float)
        cols = self.col_to_cell
        return np.vstack(
            (
                self.n_left[cols],
                self.n_middle[cols],
                self.n_right[cols],
                self.n_right[cols],
                self.n_middle[cols],
                self.n_left[cols],
                self.n_left[cols],
            )
        )

    @property
    def z_patch(self) -> np.ndarray:
        if self.z_bottom_left.size == self.ncells:
            return np.vstack(
                (
                    self.z_bottom_left,
                    self.z_bottom_mid,
                    self.z_bottom_right,
                    self.z_top_right,
                    self.z_top_mid,
                    self.z_top_left,
                    self.z_bottom_left,
                )
            )
        # Fallback in sigma coordinates when z geometry is not available.
        sbl, stl, sbm, stm, sbr, str_ = self._sigma_edges()
        return np.vstack(
            (
                sbl,
                sbm,
                sbr,
                str_,
                stm,
                stl,
                sbl,
            )
        )

    
    # Reproduciton du comportement Matlan où z_to_sigma produit des NaN lorsque zb_l == zb_r (ce qui peut arriver dans les cas extrêmes de bathymétries très raides ou de maillages très grossiers).
    # Pas de robustesse opérationelle python pour le moment, à voir si ça soulève trop d'erreur ou pas. 
    def z_to_sigma(self, z: np.ndarray, zb: np.ndarray, water_level: float | None = None) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        zb_arr = np.asarray(zb, dtype=float)
        wl = float(self.water_level if water_level is None else water_level)
        return (z_arr - zb_arr) / (wl - zb_arr)

    def mesh_at_water_level(self, target_wl, constant_z: bool = False):
        wl_values = np.asarray(target_wl, dtype=float).reshape(-1)
        meshes = [copy.deepcopy(self) for _ in wl_values]
        if constant_z:
            return meshes

        zb_l = self.zb_left[self.col_to_cell] if self.zb_left.size > 0 else np.zeros((self.ncells,), dtype=float)
        zb_m = self.zb_middle[self.col_to_cell] if self.zb_middle.size > 0 else np.zeros((self.ncells,), dtype=float)
        zb_r = self.zb_right[self.col_to_cell] if self.zb_right.size > 0 else np.zeros((self.ncells,), dtype=float)
        sbl, stl, sbm, stm, sbr, str_ = self._sigma_edges()

        for mm, wl in zip(meshes, wl_values):
            mm.z_bottom_left = self.sigma_to_z(sbl, zb_l, wl)
            mm.z_top_left = self.sigma_to_z(stl, zb_l, wl)
            mm.z_bottom_mid = self.sigma_to_z(sbm, zb_m, wl)
            mm.z_top_mid = self.sigma_to_z(stm, zb_m, wl)
            mm.z_bottom_right = self.sigma_to_z(sbr, zb_r, wl)
            mm.z_top_right = self.sigma_to_z(str_, zb_r, wl)
            mm.water_level = float(wl)

        return meshes

    def index(self, n: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Indices of mesh cells for given positions.

        idx = index(obj, n, sigma) returns the indices of the mesh cells that
        hold the points given in (n,sigma) coordinates
        """

        n = np.asarray(n, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        idx = np.full(sigma.shape, np.nan)
        sbl_all, stl_all, sbm_all, stm_all, sbr_all, str_all = self._sigma_edges()

        for cc in range(self.ncells):
            nl = self.n_left[self.col_to_cell[cc]]
            nm = self.n_middle[self.col_to_cell[cc]]
            nr = self.n_right[self.col_to_cell[cc]]

            sbl = sbl_all[cc]
            sbm = sbm_all[cc]
            sbr = sbr_all[cc]
            stl = stl_all[cc]
            stm = stm_all[cc]
            str_ = str_all[cc]

            fleft = (n >= nl) & (n < nm)
            fright = (n >= nm) & (n < nr)

            sig_bot_left = self.fit_sig(n, nl, nm, sbl, sbm)
            sig_top_left = self.fit_sig(n, nl, nm, stl, stm)
            sig_bot_right = self.fit_sig(n, nm, nr, sbm, sbr)
            sig_top_right = self.fit_sig(n, nm, nr, stm, str_)

            fsigleft = (
                (sigma > sig_bot_left)
                & (sigma <= sig_top_left)
                & fleft
            )
            fsigright = (
                (sigma > sig_bot_right)
                & (sigma <= sig_top_right)
                & fright
            )

            in_cell = fsigleft | fsigright
            idx[in_cell] = cc

        return idx
    

### Partie Plot ###

    def _validate_cell_values(self, values, label: str = "values") -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != self.ncells:
            raise ValueError(f"{label} must have the same number of elements as cells in the mesh")
        return arr


    def _plot_bed_and_water(self, ax) -> None:
        """ 
        Tracé commun du lit et de la surface de l'eau pour les différentes méthodes de tracé. 
        """
        if self.nb_all.size > 0 and self.zb_all.size == self.nb_all.size:
            ax.plot(self.nb_all, self.zb_all, "k", linewidth=2)
        if self.nw.size > 0:
            ax.plot(self.nw, np.zeros_like(self.nw) + float(self.water_level), "b", linewidth=2)


    def _add_patch_collection(self, ax, values=None):
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        patches = []
        for cc in range(self.ncells):
            verts = np.c_[self.n_patch[:, cc], self.z_patch[:, cc]]
            patches.append(Polygon(verts, closed=True))

        collection = PatchCollection(patches, cmap="viridis", alpha=0.8)
        if values is not None:
            collection.set_array(self._validate_cell_values(values))
        ax.add_collection(collection)
        return collection


    def _finalize_2d_axes(self, ax) -> None:
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("n")
        ax.set_ylabel("z / sigma")


    def plot(self, vel=None, type=None, emplacement=None, method=None):
        """
        Plot the mesh optionnally colored with a variable. 

        - plot() : bed + water surface only
        - plot(var) : patch colored by a scalar per cell
        - plot(vel) : if vel has shape (ncells, 3), color by vel[:, 0] and draw arrows from vel[:, 1:3]
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_bed_and_water(ax)

        if vel is None:
            self._finalize_2d_axes(ax)
            return ax

        vel_arr = np.asarray(vel, dtype=float)

        # One value per cell: patch coloring only.
        if vel_arr.ndim == 1 or (vel_arr.ndim == 2 and vel_arr.shape[1] == 1):
            plot_var = self._validate_cell_values(vel_arr, "vel")
            self._add_patch_collection(ax, plot_var)
            self._finalize_2d_axes(ax)
            return ax

        # MATLAB-compatible velocity mode: vel(:,1) colors the cells, vel(:,2:3) are arrows.
        if vel_arr.ndim != 2 or vel_arr.shape[0] != self.ncells or vel_arr.shape[1] < 3:
            raise ValueError("vel must have shape (ncells, 3) for the MATLAB-like velocity plot")

        plot_var = self._validate_cell_values(vel_arr[:, 0], "vel[:, 0]")
        self._add_patch_collection(ax, plot_var)

        vel_sec = vel_arr[:, 1:3].copy()

        # Keep the same outlier suppression idea as MATLAB.
        vec_norm = np.linalg.norm(vel_sec, axis=1)
        vec_norm[np.isnan(vec_norm)] = 0.0
        mean_norm = float(np.mean(vec_norm)) if vec_norm.size > 0 else 0.0
        for ii in range(vel_sec.shape[0]):
            if np.linalg.norm(vel_sec[ii]) > 4.0 * mean_norm:
                vel_sec[ii] = 0.0

        ax.quiver(
            self.n_center,
            self.z_center,
            vel_sec[:, 0],
            vel_sec[:, 1],
            color="k",
            linewidth=1.2,
            angles="xy",
            scale_units="xy",
            scale=1.0,
        )
        ax.text(-5, -3.5, "Arrow scale: 1m/s")

        self._finalize_2d_axes(ax)
        return ax


    def plot_info(self, nb_vel, emplacement=None, methode=None):
        """
        Plot the mesh optionnaly colored with a variable. 
        (very similar to plot() but with a specific variable and without velocity arrows)
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_bed_and_water(ax)
        self._add_patch_collection(ax, nb_vel)
        self._finalize_2d_axes(ax)
        return ax

    def plot_r2(self, r2, emplacement=None, methode=None):
        """
        Plot the mesh optionnaly colored with a variable. 
        (very similar to plot() but with a specific variable and without velocity arrows)
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_bed_and_water(ax)
        self._add_patch_collection(ax, r2)
        self._finalize_2d_axes(ax)
        return ax

    def plot_sigma(self, sigma, emplacement=None, methode=None):
        """
        Plot the mesh optionnaly colored with a variable. 
        (very similar to plot() but with a specific variable and without velocity arrows)
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_bed_and_water(ax)
        self._add_patch_collection(ax, sigma)
        self._finalize_2d_axes(ax)
        return ax

    def plot3(self, var=None):
        """
        3D plot of the bed and water surface.

        The MATLAB version validates var but does not actually color the patch in 3D,
        so this Python version keeps the same spirit and ignores coloring for now.
        """
        import matplotlib.pyplot as plt

        if var is not None:
            self._validate_cell_values(var, "var")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if self.nb_all.size > 0 and self.zb_all.size == self.nb_all.size:
            ax.plot(self.xb_all, self.yb_all, self.zb_all, "k", linewidth=2)

        if self.nw.size > 0:
            # ax.plot(self.xw, self.yw, np.zeros_like(self.xw) + float(self.water_level), "b", linewidth=2)
            ##
            nw_flat = np.asarray(self.nw, dtype=float).reshape(-1)
            xw_flat, yw_flat = self._sn2xy(nw_flat)
            zw_flat = np.full_like(nw_flat, float(self.water_level))
            ax.plot(xw_flat, yw_flat, zs=zw_flat, color="b", linewidth=2)
            ##

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        return ax


    