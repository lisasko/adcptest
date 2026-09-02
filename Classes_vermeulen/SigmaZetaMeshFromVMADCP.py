from __future__ import annotations

import numpy as np
import importlib 
import warnings
import alphashape

from .Bathymetry import BathymetryScatteredPoints
from .EnsembleFilter import EnsembleFilter
from .SigmaZetaMeshGenerator import SigmaZetaMeshGenerator
from .SigmaZetaMesh import SigmaZetaMesh
from .XSection import XSection

class SigmaZetaMeshFromVMADCP(SigmaZetaMeshGenerator):
    """Generate a Sigma-Zeta mesh from VMADCP-like data objects."""

    def __init__(
        self,
        vmadcp,
        bathymetry=None,
        xs: XSection | None = None,
        filter_obj=None,
        deltan: float = 5.0,
        deltaz: float = 1.0,
        time=None,
    ) -> None:
        if vmadcp is None:
            raise ValueError("vmadcp is required")

        self.vmadcp = vmadcp
        self.filter = filter_obj if filter_obj is not None else EnsembleFilter()
        self.bathymetry = bathymetry if bathymetry is not None else BathymetryScatteredPoints(vmadcp, self.filter)
        self.xs = xs if xs is not None else XSection(vmadcp)
        self.deltan = float(deltan)
        self.deltaz = float(deltaz)
        self.time = self._default_time() if time is None else time


    def _default_time(self):
        """MATLAB-like default: mean valid VMADCP time when available."""
        t = getattr(self.vmadcp, "time", None)
        if t is None:
            return None

        arr = np.asarray(t)
        if arr.ndim == 0:
            return arr.item()

        bad = np.asarray(self.filter.all_cells_bad(self.vmadcp), dtype=bool).reshape(-1)
        if bad.size == arr.shape[0]:
            arr = arr[~bad]
        if arr.size == 0:
            return None

        if np.issubdtype(arr.dtype, np.datetime64):
            arr_ns = arr.astype("datetime64[ns]")
            nat = np.datetime64("NaT", "ns")
            valid = arr_ns != nat
            if not np.any(valid):
                return None
            ints = arr_ns[valid].astype("int64")
            return np.datetime64(int(np.nanmean(ints)), "ns")

        try:
            return float(np.nanmean(arr.astype(float)))
        except Exception:
            return arr.reshape(-1)[arr.size // 2]

    @property
    def water_level(self) -> float:
        if not (hasattr(self.vmadcp, "water_level_object") and hasattr(self.vmadcp.water_level_object, "get_water_level")):
            raise AttributeError(
                "vmadcp must provide water_level_object.get_water_level(time) for MATLAB parity"
            )

        wl = self.vmadcp.water_level_object.get_water_level(self.time)
        arr = np.asarray(wl, dtype=float)
        return float(np.nanmean(arr))


    @staticmethod
    def _get_intersections(vec: np.ndarray, lev: float) -> np.ndarray:

        vec = np.asarray(vec, dtype=float).reshape(-1)
        if vec.size == 0:
            return np.empty((2, 0), dtype=int)

        above = vec > lev
        starts = []
        ends = []

        if vec[0] < lev:
            starts.append(0)
        starts.extend(np.where(np.diff(above.astype(int)) == -1)[0].tolist())

        if len(starts) == 0:
            return np.empty((2, 0), dtype=int)

        ends.extend(np.where(np.diff(above.astype(int)) == 1)[0].tolist())
        if vec[-1] < lev:
            ends.append(vec.size - 1)

        n = min(len(starts), len(ends))
        if n == 0:
            return np.empty((2, 0), dtype=int)
        return np.vstack((np.asarray(starts[:n], dtype=int), np.asarray(ends[:n], dtype=int)))

    ##
    @staticmethod
    def _fill_small_holes(shape, hole_threshold: float, shapely_geom):
        """Reproduit l'option 'HoleThreshold' de alphaShape MATLAB"""

        if hole_threshold is None or hole_threshold <= 0:
            return shape

        def _fill_polygon(poly):
            kept_holes = [
                ring for ring in poly.interiors
                if shapely_geom.Polygon(ring).area >= hole_threshold
            ]
            return shapely_geom.Polygon(poly.exterior, kept_holes)

        try:
            if shape.geom_type == "Polygon":
                return _fill_polygon(shape)
            if shape.geom_type == "MultiPolygon":
                return shapely_geom.MultiPolygon([_fill_polygon(p) for p in shape.geoms])
        except Exception:
            pass
        return shape
    ##


    @staticmethod
    def _points_inside_alpha_like(points_xy: np.ndarray, query_xy: np.ndarray, hole_threshold: float) -> np.ndarray:
        
        points_xy = np.asarray(points_xy, dtype=float)
        query_xy = np.asarray(query_xy, dtype=float)

        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError(f"points_xy must be a (N, 2) array, got shape {points_xy.shape}")

        valid_pts = np.all(np.isfinite(points_xy), axis=1)
        points_xy = points_xy[valid_pts]
        if points_xy.shape[0] < 3:
            return np.ones((query_xy.shape[0],), dtype=bool) 

        valid_q = np.all(np.isfinite(query_xy), axis=1)
        inside = np.zeros((query_xy.shape[0],), dtype=bool)

        try:
            alphashape = importlib.import_module("alphashape")
            shapely_geom = importlib.import_module("shapely.geometry")
        except ImportError as e:
            warnings.warn(
                f"alphashape/shapely non insponible ({e}). Tous les points seront considérés comme valides."
            )
            return np.ones((query_xy.shape[0],), dtype=bool)  

        # shape = None
        # try:
        #     alpha = alphashape.optimizealpha(points_xy)
        #     if alpha is not None and np.isfinite(alpha) and alpha > 0:
        #         shape = alphashape.alphashape(points_xy, alpha)
        # except Exception:
        #     shape = None

        ## 05/08
        shape = SigmaZetaMeshFromVMADCP._find_critical_alpha(points_xy, shapely_geom)

        if shape is None or shape.is_empty:
            warnings.warn(
                "Impossible de trouver une alpha-shape en une seule région ; "
                "tous les points sont considérés comme valides pour l'étendue horizontale du maillage."
            )
            return np.ones((query_xy.shape[0],), dtype=bool)

        print(f"DEBUG alphashape: n_points={points_xy.shape[0]}, "
            f"aire de la forme={shape.area:.1f} m², bounds={shape.bounds}")
        ##

        if shape is None or shape.is_empty:
            print("\n*DEBUG ALPHASHAPE: optimizealpha a échoué -> fallback alpha=0.1 (forme probablement trop lâche)")
            try:
                shape = alphashape.alphashape(points_xy, alpha=0.1)
            except Exception as exc:
                warnings.warn(
                    f"Echec de alphashape ({exc}) ; tous les points sont "
                    "considérés comme valides pour l'étendue horizontale du maillage."
                )
                return np.ones((query_xy.shape[0],), dtype=bool)

        if shape is None or shape.is_empty or not hasattr(shape, "contains"):
            return np.ones((query_xy.shape[0],), dtype=bool)

        shape = SigmaZetaMeshFromVMADCP._fill_small_holes(shape, hole_threshold, shapely_geom)

        for ii, q in enumerate(query_xy):
            if valid_q[ii]:
                point = shapely_geom.Point(q)
                inside[ii] = bool(shape.contains(point) or shape.touches(point))

        return inside

    @staticmethod
    def _find_critical_alpha(points_xy, shapely_geom, alpha_max=None, n_iter=25):

        if alpha_max is None:
            tree_dist = np.median(np.linalg.norm(
                points_xy[1:] - points_xy[:-1], axis=1
            ))
            tree_dist = tree_dist if np.isfinite(tree_dist) and tree_dist > 0 else 1.0
            alpha_max = 5.0 / tree_dist 

        lo, hi = 0.0, alpha_max
        best_shape = None

        def _is_one_region(a):
            try:
                s = alphashape.alphashape(points_xy, a)
            except Exception:
                return False, None
            if s is None or s.is_empty:
                return False, None
            if s.geom_type == "Polygon":
                return True, s
            return False, None

        ok, s = _is_one_region(lo)
        if ok:
            best_shape = s

        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            ok, s = _is_one_region(mid)
            if ok:
                best_shape = s
                lo = mid  
            else:
                hi = mid 

        return best_shape


    def _beam_angle_deg(self, reduce: str = "mean") -> float:

        if hasattr(self.vmadcp, "beam_angle_deg"):
            values = np.asarray(self.vmadcp.beam_angle_deg, dtype=float)
        elif hasattr(self.vmadcp, "beam_angle"):
            values = np.asarray(self.vmadcp.beam_angle, dtype=float)
        else:
            return 20.0

        if reduce == "max":
            return float(np.nanmax(values))
        return float(np.nanmean(values))

    ##

    def _get_bed_position(self) -> np.ndarray:
        bed = getattr(self.vmadcp, "bed_position", None)
        if callable(bed):
            bed = bed()
        return np.asarray(bed, dtype=float).copy()

    def _get_depth_cell_position(self) -> np.ndarray:
        return np.asarray(getattr(self.vmadcp, "depth_cell_position"), dtype=float).copy()

    def get_mesh(self):
        """ Function to generate the Sigma-Zeta mesh.""" 

        mesh = SigmaZetaMesh(
            n_left=np.empty((0,), dtype=float),
            n_middle=np.empty((0,), dtype=float),
            n_right=np.empty((0,), dtype=float),
            sig_bottom_left=np.empty((0,), dtype=float),
            sig_top_left=np.empty((0,), dtype=float),
            sig_bottom_mid=np.empty((0,), dtype=float),
            sig_top_mid=np.empty((0,), dtype=float),
            sig_bottom_right=np.empty((0,), dtype=float),
            sig_top_right=np.empty((0,), dtype=float),
            row_to_cell=np.empty((0,), dtype=int),
            col_to_cell=np.empty((0,), dtype=int),
        )

        mesh.xs = self.xs
        mesh.time = self.time

        vpos = self._get_depth_cell_position()
        n_ensembles_vpos = vpos.shape[1]

        if self.filter.bad_ensembles.size != n_ensembles_vpos:
            self.filter.bad_ensembles = np.zeros((n_ensembles_vpos,), dtype=bool)
        bad = np.asarray(self.filter.all_cells_bad(self.vmadcp), dtype=bool).reshape(-1)

        if bad.size not in (0, vpos.shape[1]):
            raise ValueError("EnsembleFilter size does not match depth_cell_position ensembles")
        if bad.size == vpos.shape[1]:
            vpos[:, bad, :, :] = np.nan

        # bpos = self.bathymetry.known.T.reshape(1, -1, 1, 3)  
        # if bpos.ndim != 4:
        #     bpos = bpos.reshape(1, -1, 1, 3)

        # known_x = self.bathymetry.known[0, :]
        # known_y = self.bathymetry.known[1, :]
        # _, n = self.xs.xy2sn(known_x, known_y)  
        # xn, yn = self.xs.sn2xy(n * 0, n)

        # print(f"DEBUG: n range = [{np.nanmin(n):.2f}, {np.nanmax(n):.2f}]")
        # print(f"DEBUG: known_x range = [{np.nanmin(known_x):.2f}, {np.nanmax(known_x):.2f}]")
        # print(f"DEBUG: known_y range = [{np.nanmin(known_y):.2f}, {np.nanmax(known_y):.2f}]")

        # fgood = np.isfinite(n)
        # n_valid = n[fgood]
        # xn = xn[fgood]
        # yn = yn[fgood]

        # bpos_2d = bpos.reshape(-1, 3)
        # valid_bpos = np.all(np.isfinite(bpos_2d[:, :2]), axis=1)
        # pts_xy = bpos_2d[valid_bpos, :2]

        # inside = self._points_inside_alpha_like(
        #     pts_xy,
        #     np.c_[xn.reshape(-1), yn.reshape(-1)],
        #     hole_threshold=float(self.xs.scale) ** 2,
        # )
        # n_valid = n_valid.reshape(-1)[inside]

        ##
        bpos = self._get_bed_position()
        bad_bpos_size = bpos.shape[1] if bpos.ndim >= 2 else 0
        if bad.size == bad_bpos_size:
            bpos[:, bad, :, :] = np.nan

        bx = bpos[..., 0]
        by = bpos[..., 1]
        _, n = self.xs.xy2sn(bx, by)
        xn, yn = self.xs.sn2xy(n * 0, n)

        bpos_flat = bpos.reshape(-1, bpos.shape[-1])
        valid_bpos = np.all(np.isfinite(bpos_flat[:, :2]), axis=1)
        pts_xy = bpos_flat[valid_bpos, :2]

        fgood = np.isfinite(n).reshape(-1)
        n_flat = n.reshape(-1)[fgood]
        xn_flat = xn.reshape(-1)[fgood]
        yn_flat = yn.reshape(-1)[fgood]

        inside = self._points_inside_alpha_like(
            pts_xy,
            np.c_[xn_flat, yn_flat],
            hole_threshold=float(self.xs.scale) ** 2,
        )
        n_valid = n_flat[inside]

        ##

        if n_valid.size < 2:
            raise ValueError("Not enough valid bed detections to construct Sigma-Zeta mesh")

        # Etendue spatiale n et interpolation de z

        nmin = float(np.nanmin(n_valid)) 
        nmax = float(np.nanmax(n_valid)) 

        nvec = np.arange(nmin, nmax, self.deltan)
        nvec = np.append(nvec, nmax)

        # Interpolation de l'élévation du lit sur la grille n 

        xvec, yvec = self.xs.sn2xy(nvec * 0, nvec)
        zvec = np.asarray(self.bathymetry.get_bed_elev(xvec, yvec), dtype=float)

        print(f"DEBUG zvec: min={np.nanmin(zvec):.2f}, max={np.nanmax(zvec):.2f}, NaN={np.isnan(zvec).sum()}")

        print(
            f"\nDEBUG bathymetry.known: n_points={self.bathymetry.known.shape[1]}, "
            f"span interpolateur={getattr(self.bathymetry.interpolator, 'span', 'N/A')}"
        )
        if np.nanmax(zvec) >= self.water_level:
            n_above = int(np.sum(zvec >= self.water_level))
            print(
                f"DEBUG ALERTE : {n_above}/{zvec.size} points du fond interpolé sont "
                f"au-dessus (ou au niveau) du niveau d'eau ({self.water_level:.2f} m). "
                f"Probable extrapolation instable de la bathymétrie (nuage de points "
                f"trop clairsemé pour ce span)."
            )



        mesh.nb_all = nvec.copy()
        mesh.zb_all = zvec.copy()

        minsigma = 1.0 - np.cos(np.deg2rad(self._beam_angle_deg(reduce="max")))


        pitch = np.asarray(getattr(self.vmadcp, "pitch", np.zeros((vpos.shape[1],))), dtype=float).reshape(-1)
        roll = np.asarray(getattr(self.vmadcp, "roll", np.zeros((vpos.shape[1],))), dtype=float).reshape(-1)

        if pitch.size == vpos.shape[1] and roll.size == vpos.shape[1]:
            ftilt = (pitch ** 2 < 25.0) & (roll ** 2 < 25.0)
        else:
            ftilt = np.ones((vpos.shape[1],), dtype=bool)
        if not np.any(ftilt):
            ftilt = np.ones((vpos.shape[1],), dtype=bool)

        maxz = float(np.nanmax(vpos[:, ftilt, :, 2]))

        # Construction du plan d'eau

        # wl = 0.0
        # print(f"Water level forcé à {wl:.2f} m (surface de l'eau).")

        wl = self.water_level
        mesh.water_level = wl

        fwl = self._get_intersections(zvec, wl)
        if fwl.size == 0:
            raise ValueError("Cannot create mesh since the water level is lower than the bed")
        

        nw = np.vstack((nvec[fwl[0]], nvec[fwl[1]])).astype(float)
        for rr in range(2):
            for cc in range(fwl.shape[1]):
                idx = int(fwl[rr, cc])
                cond = (idx != 0 and idx != (nvec.size - 1)) or (idx == 0 and zvec[idx] > wl)
                if not cond or idx >= nvec.size - 1:
                    continue
                den = zvec[idx + 1] - zvec[idx]
                if abs(den) < 1e-12:
                    continue
                nw[rr, cc] = nvec[idx] + (nvec[idx + 1] - nvec[idx]) * (wl - zvec[idx]) / den
        mesh.nw = nw

        z_bot_mesh = zvec * (1.0 - minsigma) + minsigma * wl
        fbnds = self._get_intersections(z_bot_mesh, maxz)
        if fbnds.size == 0:
            raise ValueError("Cannot create mesh since the maximum mesh level is lower than the minimum measurement level")

        n_left = nvec[:-1].copy()
        n_right = nvec[1:].copy()

        lf = fbnds[0, (fbnds[0] != 0) | ((fbnds[0] == 0) & (z_bot_mesh[fbnds[0]] > maxz))]
        for s in lf:
            s = int(s)
            if s >= n_left.size:
                continue
            den = z_bot_mesh[s + 1] - z_bot_mesh[s] if s + 1 < z_bot_mesh.size else np.nan
            if not np.isfinite(den) or abs(den) < 1e-12:
                continue
            n_left[s] = n_left[s] + (n_left[s + 1] - n_left[s]) * (maxz - z_bot_mesh[s]) / den

        right_size = n_right.size
        right_in = fbnds[1] <= (right_size - 1)
        right_at_end = (fbnds[0] == (right_size - 1)) & (z_bot_mesh[fbnds[1]] > maxz)
        rf = fbnds[1, right_in | right_at_end]

        for e in rf:
            e = int(e)
            if e <= 0 or e >= n_right.size:
                continue
            den = z_bot_mesh[e] - z_bot_mesh[e - 1]
            if abs(den) < 1e-12:
                continue
            n_right[e] = n_right[e - 1] + (n_right[e] - n_right[e - 1]) * (maxz - z_bot_mesh[e - 1]) / den

        zb_left = zvec[:-1].copy()
        zb_right = zvec[1:].copy()

        # zb_left = np.where(np.isnan(zb_left), zvec[0] - 1.0, zb_left)
        # zb_right = np.where(np.isnan(zb_right), zvec[-1] - 1.0, zb_right)
        
        z_inter = (maxz - wl * minsigma) / max(1.0 - minsigma, 1e-12)
        for s in lf:
            s = int(s)
            if s < zb_left.size:
                zb_left[s] = z_inter
        for e in rf:
            e = int(e)
            if 0 <= e < zb_right.size:
                zb_right[e] = z_inter

        frem = (z_bot_mesh[:-1] > maxz) & (z_bot_mesh[1:] > maxz)
        if np.any(frem):
            keep = ~frem
            n_left = n_left[keep]
            n_right = n_right[keep]
            zb_left = zb_left[keep]
            zb_right = zb_right[keep]

        nmid = 0.5 * (n_left + n_right)
        xmid, ymid = self.xs.sn2xy(np.zeros_like(nmid), nmid)
        zmid = np.asarray(self.bathymetry.get_bed_elev(xmid, ymid), dtype=float)
        # zmid = np.where(np.isnan(zmid), zvec[0], zmid)
        minz_mid = zmid * (1.0 - minsigma) + minsigma * wl

        # sort_order = np.argsort(nvec)
        # mesh.nb_all = nvec[sort_order]
        # mesh.zb_all = zvec[sort_order]

        ##

        nb_all_concat = np.concatenate((mesh.nb_all, mesh.nw.reshape(-1), n_left, n_right, nmid))
        zb_all_concat = np.concatenate(
            (mesh.zb_all, np.full(mesh.nw.size, wl, dtype=float), zb_left, zb_right, zmid)
        )
        nb_all_unique, idx_unq = np.unique(nb_all_concat, return_index=True)
        mesh.nb_all = nb_all_unique
        mesh.zb_all = zb_all_concat[idx_unq]

        ##

        nz = np.maximum(1, np.ceil((maxz - minz_mid) / max(self.deltaz, 1e-6)).astype(int))
        max_num = int(np.nanmax(nz))
        n_cols = int(nz.size)

        col_to_mat = np.tile(np.arange(n_cols, dtype=int), (max_num, 1))
        row_to_mat = np.tile(np.arange(1, max_num + 1, dtype=int).reshape(-1, 1), (1, n_cols))
        mat_to_cell = (row_to_mat <= nz[np.newaxis, :]).reshape(-1, order="F")

        row_to_cell = row_to_mat.reshape(-1, order="F")[mat_to_cell]
        col_to_cell = col_to_mat.reshape(-1, order="F")[mat_to_cell]

        mesh.col_to_mat = col_to_mat
        mesh.row_to_mat = row_to_mat
        mesh.mat_to_cell = mat_to_cell
        mesh.row_to_cell = row_to_cell
        mesh.col_to_cell = col_to_cell
        mesh.cell_to_mat = np.arange(mat_to_cell.size, dtype=int)[mat_to_cell]

        mesh.n_middle = nmid.reshape(-1)
        mesh.n_left = n_left.reshape(-1)
        mesh.n_right = n_right.reshape(-1)
        mesh.zb_middle = zmid.reshape(-1)
        mesh.zb_left = zb_left.reshape(-1)
        mesh.zb_right = zb_right.reshape(-1)

        minz_left = zb_left * (1.0 - minsigma) + minsigma * wl
        minz_right = zb_right * (1.0 - minsigma) + minsigma * wl
        deltaz_mid = (maxz - minz_mid) / nz
        deltaz_left = (maxz - minz_left) / nz
        deltaz_right = (maxz - minz_right) / nz

        z_bottom_left_all = (maxz - row_to_mat * deltaz_left[np.newaxis, :]).reshape(-1, order="F")
        z_top_left_all = (maxz - (row_to_mat - 1) * deltaz_left[np.newaxis, :]).reshape(-1, order="F")
        z_bottom_mid_all = (maxz - row_to_mat * deltaz_mid[np.newaxis, :]).reshape(-1, order="F")
        z_top_mid_all = (maxz - (row_to_mat - 1) * deltaz_mid[np.newaxis, :]).reshape(-1, order="F")
        z_bottom_right_all = (maxz - row_to_mat * deltaz_right[np.newaxis, :]).reshape(-1, order="F")
        z_top_right_all = (maxz - (row_to_mat - 1) * deltaz_right[np.newaxis, :]).reshape(-1,order="F")

        mesh.z_bottom_left = z_bottom_left_all[mat_to_cell]
        mesh.z_top_left = z_top_left_all[mat_to_cell]
        mesh.z_bottom_mid = z_bottom_mid_all[mat_to_cell]
        mesh.z_top_mid = z_top_mid_all[mat_to_cell]
        mesh.z_bottom_right = z_bottom_right_all[mat_to_cell]
        mesh.z_top_right = z_top_right_all[mat_to_cell]

        zb_l = mesh.zb_left[mesh.col_to_cell]
        zb_m = mesh.zb_middle[mesh.col_to_cell]
        zb_r = mesh.zb_right[mesh.col_to_cell]
        den_l = np.maximum(wl - zb_l, 1e-9)
        den_m = np.maximum(wl - zb_m, 1e-9)
        den_r = np.maximum(wl - zb_r, 1e-9)
        mesh.sig_bottom_left = (mesh.z_bottom_left - zb_l) / den_l
        mesh.sig_top_left = (mesh.z_top_left - zb_l) / den_l
        mesh.sig_bottom_mid = (mesh.z_bottom_mid - zb_m) / den_m
        mesh.sig_top_mid = (mesh.z_top_mid - zb_m) / den_m
        mesh.sig_bottom_right = (mesh.z_bottom_right - zb_r) / den_r
        mesh.sig_top_right = (mesh.z_top_right - zb_r) / den_r

        return mesh

    #     nb_all = np.concatenate((mesh.nb_all, mesh.nw.reshape(-1), n_left, n_right, nmid))
    #     zb_all = np.concatenate((mesh.zb_all, np.full(mesh.nw.size, wl, dtype=float), zb_left, zb_right, zmid))

    #     ##
    #     valid_zb = np.isfinite(zb_all)
    #     if np.any(valid_zb):
    #         uniq_nb, uniq_idx = np.unique(nb_all, return_index=True)
    #         mesh.nb_all = uniq_nb
    #         mesh.zb_all = zb_all[uniq_idx]
    #     else:
    #         # Si tout est NaN, garder les valeurs initiales de mesh.nb_all et mesh.zb_all
    #         print("Warning: zb_all contient uniquement des NaN. Conservation des valeurs initiales.")
    # ##

    #     uniq_nb, uniq_idx = np.unique(nb_all, return_index=True)
    #     mesh.nb_all = uniq_nb
    #     mesh.zb_all = zb_all[uniq_idx]

    #     nz = np.maximum(1, np.ceil((maxz - minz_mid) / max(self.deltaz, 1e-6)).astype(int))
    #     max_num = int(np.nanmax(nz))
    #     n_cols = int(nz.size)
    #     col_to_mat = np.tile(np.arange(n_cols, dtype=int), (max_num, 1))
    #     row_to_mat = np.tile(np.arange(1, max_num + 1, dtype=int).reshape(-1, 1), (1, n_cols))
    #     mat_to_cell = (row_to_mat <= nz[np.newaxis, :]).reshape(-1)

    #     row_to_cell = row_to_mat.reshape(-1)[mat_to_cell]
    #     col_to_cell = col_to_mat.reshape(-1)[mat_to_cell]

    #     mesh.col_to_mat = col_to_mat
    #     mesh.row_to_mat = row_to_mat
    #     mesh.mat_to_cell = mat_to_cell
    #     mesh.row_to_cell = row_to_cell
    #     mesh.col_to_cell = col_to_cell
    #     mesh.cell_to_mat = np.arange(mat_to_cell.size, dtype=int)[mat_to_cell]

    #     mesh.n_middle = nmid.reshape(-1)
    #     mesh.n_left = n_left.reshape(-1)
    #     mesh.n_right = n_right.reshape(-1)
    #     mesh.zb_middle = zmid.reshape(-1)
    #     mesh.zb_left = zb_left.reshape(-1)
    #     mesh.zb_right = zb_right.reshape(-1)

    #     minz_left = zb_left * (1.0 - minsigma) + minsigma * wl
    #     minz_right = zb_right * (1.0 - minsigma) + minsigma * wl
    #     deltaz_mid = (maxz - minz_mid) / nz
    #     deltaz_left = (maxz - minz_left) / nz
    #     deltaz_right = (maxz - minz_right) / nz

    #     z_bottom_left_all = (maxz - row_to_mat * deltaz_left[np.newaxis, :]).reshape(-1)
    #     z_top_left_all = (maxz - (row_to_mat - 1) * deltaz_left[np.newaxis, :]).reshape(-1)
    #     z_bottom_mid_all = (maxz - row_to_mat * deltaz_mid[np.newaxis, :]).reshape(-1)
    #     z_top_mid_all = (maxz - (row_to_mat - 1) * deltaz_mid[np.newaxis, :]).reshape(-1)
    #     z_bottom_right_all = (maxz - row_to_mat * deltaz_right[np.newaxis, :]).reshape(-1)
    #     z_top_right_all = (maxz - (row_to_mat - 1) * deltaz_right[np.newaxis, :]).reshape(-1)

    #     mesh.z_bottom_left = z_bottom_left_all[mat_to_cell]
    #     mesh.z_top_left = z_top_left_all[mat_to_cell]
    #     mesh.z_bottom_mid = z_bottom_mid_all[mat_to_cell]
    #     mesh.z_top_mid = z_top_mid_all[mat_to_cell]
    #     mesh.z_bottom_right = z_bottom_right_all[mat_to_cell]
    #     mesh.z_top_right = z_top_right_all[mat_to_cell]

    #     zb_l = mesh.zb_left[mesh.col_to_cell]
    #     zb_m = mesh.zb_middle[mesh.col_to_cell]
    #     zb_r = mesh.zb_right[mesh.col_to_cell]
    #     den_l = np.maximum(wl - zb_l, 1e-9)
    #     den_m = np.maximum(wl - zb_m, 1e-9)
    #     den_r = np.maximum(wl - zb_r, 1e-9)
    #     mesh.sig_bottom_left = (mesh.z_bottom_left - zb_l) / den_l
    #     mesh.sig_top_left = (mesh.z_top_left - zb_l) / den_l
    #     mesh.sig_bottom_mid = (mesh.z_bottom_mid - zb_m) / den_m
    #     mesh.sig_top_mid = (mesh.z_top_mid - zb_m) / den_m
    #     mesh.sig_bottom_right = (mesh.z_bottom_right - zb_r) / den_r
    #     mesh.sig_top_right = (mesh.z_top_right - zb_r) / den_r

        # print(f"DEBUG: nvec range = [{nmin:.2f}, {nmax:.2f}]")
        # print(f"DEBUG: known_n range = [{np.nanmin(known_n):.2f}, {np.nanmax(known_n):.2f}]")
        # print(f"DEBUG: zvec range = [{np.nanmin(zvec):.2f}, {np.nanmax(zvec):.2f}]")
        # print(f"DEBUG: water_level = {wl:.2f} m")

        # return mesh
