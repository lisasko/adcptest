from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .Mesh import Mesh
from .VelocityModel import VelocityModel


class VelocitySolver(ABC):
    """
    Abstract base class to solve ADCP repeat-transect velocity data.

    Subclasses must implement:
        get_solver_input() -> (vpos, vdat, xform)

    This is a MATLAB-oriented port of VelocitySolver.m, focused on the
    location-based workflow only.
    """

    def __init__(self, *args: Any):
        self.adcp = None
        self.mesh = None
        self.bathy = None
        self.xs = None
        self.ensemble_filter = None
        self.velocity_model = None

        for cur_arg in args:
            if self.adcp is None and hasattr(cur_arg, "depth_cell_position"):
                self.adcp = cur_arg
            elif self.mesh is None and isinstance(cur_arg, Mesh):
                self.mesh = cur_arg
            elif self.bathy is None and hasattr(cur_arg, "get_bed_elev"):
                self.bathy = cur_arg
            elif self.xs is None and hasattr(cur_arg, "xy2sn"):
                self.xs = cur_arg
            elif self.ensemble_filter is None and hasattr(cur_arg, "bad_ensembles"):
                self.ensemble_filter = cur_arg
            elif self.velocity_model is None and hasattr(cur_arg, "get_velocity"):
                self.velocity_model = cur_arg

        if self.mesh is None:
            raise ValueError("You must provide a Mesh object upon construction of a VelocitySolver object")
        if self.adcp is None:
            raise ValueError("You must provide a VMADCP object upon construction of a VelocitySolver object")

        if self.bathy is None:
            try:
                from .Bathymetry import BathymetryScatteredPoints
                self.bathy = BathymetryScatteredPoints(self.adcp)
            except Exception:
                self.bathy = None

        if self.xs is None:
            try:
                from .XSection import XSection
                self.xs = XSection(self.adcp)
            except Exception:
                self.xs = None

        if self.velocity_model is None:
            self.velocity_model = VelocityModel()

    @abstractmethod
    def get_solver_input(self):
        """
        Return:
            vpos:  (cells, ensembles, beams, 3)
            vdat:  (cells, ensembles, beams)
            xform: (cells, ensembles, beams, 3)
        """
        raise NotImplementedError

    def make_indices(self):
        """
        MATLAB-like index helper for repeat transects.

        Returns
        -------
        idx_mesh, idx_ef
            Integer arrays used to pair mesh/filter entries.
        """
        n_mesh = len(self.mesh) if isinstance(self.mesh, (list, tuple)) else 1
        n_ef = len(self.ensemble_filter) if isinstance(self.ensemble_filter, (list, tuple)) else 1
        nrp = max(n_mesh, n_ef)

        idx_mesh = np.zeros((nrp,), dtype=int)
        idx_ef = np.zeros((nrp,), dtype=int)

        if n_mesh > 1:
            idx_mesh = np.arange(nrp, dtype=int)
        if n_ef > 1:
            idx_ef = np.arange(nrp, dtype=int)

        return idx_mesh, idx_ef

    @staticmethod
    def _to_seconds_like(time_array):
        time_array = np.asarray(time_array)
        if time_array.size == 0:
            return np.array([], dtype=float)

        if np.issubdtype(time_array.dtype, np.datetime64):
            return time_array.astype("datetime64[ns]").astype("int64").astype(float) / 1e9

        if np.issubdtype(time_array.dtype, np.timedelta64):
            return time_array.astype("timedelta64[ns]").astype("int64").astype(float) / 1e9

        try:
            return time_array.astype(float)
        except Exception:
            return np.asarray(time_array, dtype=float)

    @staticmethod
    def fit_model(weight, vel, model_mat):
        """
        Weighted least squares fit, close to the MATLAB helper behavior.

        Returns
        -------
        pars, n_dat, r_carre, cov_pars
        """
        vel = np.asarray(vel, dtype=float).reshape(-1)
        model_mat = np.asarray(model_mat, dtype=float)
        weight = np.asarray(weight, dtype=float).reshape(-1)

        if model_mat.ndim != 2:
            raise ValueError("model_mat must be 2D")
        if model_mat.shape[0] != vel.size:
            raise ValueError("model_mat and vel must have the same number of rows")

        n_dat = vel.size
        n_pars = model_mat.shape[1]

        if n_dat == 0:
            return (
                np.full((n_pars,), np.nan, dtype=float),
                0,
                np.nan,
                np.full((n_pars, n_pars), np.nan, dtype=float),
            )

        if weight.size == 1:
            weight = np.full((n_dat,), float(weight[0]), dtype=float)

        if weight.size != n_dat:
            raise ValueError("weight and vel must have the same length")

        weight = np.clip(weight, 1e-12, np.inf)
        sw = np.sqrt(weight)

        Aw = model_mat * sw[:, None]
        bw = vel * sw

        if np.linalg.matrix_rank(Aw) < n_pars:
            return (
                np.full((n_pars,), np.nan, dtype=float),
                n_dat,
                np.nan,
                np.full((n_pars, n_pars), np.nan, dtype=float),
            )

        pars, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        resid = bw - Aw @ pars
        dof = max(n_dat - n_pars, 1)
        r_carre = float(np.sum(resid**2) / dof)

        ata = Aw.T @ Aw
        try:
            cov_pars = r_carre * np.linalg.inv(ata)
        except np.linalg.LinAlgError:
            cov_pars = r_carre * np.linalg.pinv(ata)

        return pars, n_dat, r_carre, cov_pars

    @staticmethod
    def _expand_model_matrix(Mu, Mv, Mw, xform):
        """
        Build the full matrix [Mu.*x1, Mv.*x2, Mw.*x3].
        """
        Mu = np.asarray(Mu, dtype=float)
        Mv = np.asarray(Mv, dtype=float)
        Mw = np.asarray(Mw, dtype=float)
        xform = np.asarray(xform, dtype=float)

        if Mu.ndim == 1:
            Mu = Mu.reshape(-1, 1)
        if Mv.ndim == 1:
            Mv = Mv.reshape(-1, 1)
        if Mw.ndim == 1:
            Mw = Mw.reshape(-1, 1)

        if xform.ndim != 2 or xform.shape[1] < 3:
            raise ValueError("xform must be shaped (N, 3)")

        if Mu.shape[0] != xform.shape[0] or Mv.shape[0] != xform.shape[0] or Mw.shape[0] != xform.shape[0]:
            raise ValueError("model matrices and xform must have the same number of rows")

        return np.column_stack(
            (
                Mu * xform[:, [0]],
                Mv * xform[:, [1]],
                Mw * xform[:, [2]],
            )
        )

    def get_parameters(self, f_vitesse_z=10.0, f_direction_fixe=0.97, f_direction_pond=0.0002, pond_vitesses=1.0):

        if self.xs is None:
            raise ValueError("VelocitySolver requires an XSection object to compute n/s coordinates")

        vpos, vel_data, xform = self.get_solver_input()
        vpos = np.asarray(vpos, dtype=float)
        vel_data = np.asarray(vel_data, dtype=float)
        xform = np.asarray(xform, dtype=float)

        if vpos.ndim != 4 or vpos.shape[-1] < 3:
            raise ValueError("vpos must be shaped (cells, ensembles, beams, 3)")
        if vel_data.ndim != 3:
            raise ValueError("vel_data must be shaped (cells, ensembles, beams)")
        if xform.ndim != 4 or xform.shape[-1] < 3:
            raise ValueError("xform must be shaped (cells, ensembles, beams, 3)")

        idx_mesh, idx_ef = self.make_indices()

        self.pdop_cell = []
        self.cond_cell = []
        self.sphericity_cell = []

        s_pos, n_pos = self.xs.xy2sn(vpos[..., 0], vpos[..., 1])
        zb_pos = self.bathy.get_bed_elev(vpos[..., 0], vpos[..., 1])

        water_level = np.asarray(getattr(self.adcp, "water_level", np.nan), dtype=float)
        # if water_level.ndim == 0:
        #     water_level = np.full(vpos[..., 2].shape, float(water_level), dtype=float)
        # elif water_level.size == 1:
        #     water_level = np.full(vpos[..., 2].shape, float(water_level.reshape(-1)[0]), dtype=float)
        # else:
        #     water_level = np.broadcast_to(water_level.reshape(1, -1, 1), vpos[..., 2].shape)

        water_level = np.asarray(getattr(self.adcp, "water_level", np.nan), dtype=float)
        if water_level.size == 0:
            water_level = np.full(vpos[..., 2].shape, 0.0, dtype=float)
        elif water_level.ndim == 0:
            water_level = np.full(vpos[..., 2].shape, float(water_level), dtype=float)
        elif water_level.size == 1:
            water_level = np.full(vpos[..., 2].shape, float(water_level.reshape(-1)[0]), dtype=float)
        else:
            try:
                water_level = np.broadcast_to(water_level.reshape(1, -1, 1), vpos[..., 2].shape)
            except ValueError:
                water_level = np.full(vpos[..., 2].shape, 0.0, dtype=float)

        sig_pos = (vpos[..., 2] - zb_pos) / (water_level - zb_pos)

        time = np.asarray(getattr(self.adcp, "time", np.array([])))
        if time.size == 0:
            time = np.zeros((vpos.shape[1],), dtype=float)
        time = self._to_seconds_like(time)
        if time.size == 1:
            time = np.full((vpos.shape[1],), float(time.reshape(-1)[0]), dtype=float)

        pars = []
        cov_pars = []
        n_vels = []
        r_carre = []
        r_sigma = []

        for crp in range(len(idx_ef)):
            mesh_idx = idx_mesh[min(crp, len(idx_mesh) - 1)]
            ef_idx = idx_ef[min(crp, len(idx_ef) - 1)]

            cmesh = self.mesh[mesh_idx] if isinstance(self.mesh, (list, tuple)) else self.mesh
            ef = self.ensemble_filter[ef_idx] if isinstance(self.ensemble_filter, (list, tuple)) else self.ensemble_filter

            if ef is None:
                ens_filt = np.ones((vpos.shape[1],), dtype=bool)
            else:
                ens_filt = ~np.asarray(ef.bad_ensembles, dtype=bool).reshape(-1)

            cur_n = n_pos[:, ens_filt, :]
            cur_s = s_pos[:, ens_filt, :]
            cur_sig = sig_pos[:, ens_filt, :]
            cur_vel = vel_data[:, ens_filt, :]
            cur_z = vpos[:, ens_filt, :, 2]
            cur_xform = xform[:, ens_filt, :, :]
            cur_t = np.broadcast_to(time.reshape(1, -1, 1), cur_vel.shape)

            cur_n = cur_n.reshape(-1)
            cur_s = cur_s.reshape(-1)
            cur_sig = cur_sig.reshape(-1)
            cur_vel = cur_vel.reshape(-1)
            cur_z = cur_z.reshape(-1)
            cur_t = cur_t.reshape(-1)
            cur_xform = cur_xform.reshape(-1, 3)

            finite = (
                np.isfinite(cur_n)
                & np.isfinite(cur_sig)
                & np.isfinite(cur_vel)
                & np.all(np.isfinite(cur_xform), axis=1)
            )

            cur_n = cur_n[finite]
            cur_s = cur_s[finite]
            cur_sig = cur_sig[finite]
            cur_vel = cur_vel[finite]
            cur_z = cur_z[finite]
            cur_t = cur_t[finite]
            cur_xform = cur_xform[finite]

            cell_idx = np.asarray(cmesh.index(cur_n, cur_sig), dtype=float)
            good = np.isfinite(cell_idx)

            cur_n = cur_n[good]
            cur_s = cur_s[good]
            cur_sig = cur_sig[good]
            cur_vel = cur_vel[good]
            cur_z = cur_z[good]
            cur_t = cur_t[good]
            cur_xform = cur_xform[good]
            cell_idx = cell_idx[good].astype(int)

            mesh_time = getattr(cmesh, "time", None)
            if mesh_time is not None:
                cur_t = cur_t - float(mesh_time)

            Mu, Mv, Mw = self.velocity_model.get_model(cur_t, cur_s, cur_n, cur_z, cur_sig)
            cur_xform = self._expand_model_matrix(Mu, Mv, Mw, cur_xform)

            ndat = cur_xform.shape[1] + 1

            gather_dat = [[], [], [], []]
            gather_s = [[]]

            ncells = int(getattr(cmesh, "ncells"))
            pars_cell = [np.full((0,), np.nan, dtype=float) for _ in range(ncells)]
            cov_cell = [np.full((0, 0), np.nan, dtype=float) for _ in range(ncells)]
            n_vels_cell = [np.nan for _ in range(ncells)]
            r_carre_cell = [np.nan for _ in range(ncells)]
            r_sigma_cell = [np.nan for _ in range(ncells)]

            cell_data = [[] for _ in range(ncells)]
            cell_s = [[] for _ in range(ncells)]

            pdop_cur = []
            cond_cur = []
            sphericity_cur = []

            for ii, cc in enumerate(cell_idx):
                row = np.concatenate(([cur_vel[ii]], cur_xform[ii]))
                cell_data[cc].append(row)
                cell_s[cc].append(cur_s[ii])

            pars_cur = []
            cov_cur = []
            n_cur = []
            r2_cur = []
            rs_cur = []

            for cc in range(ncells):
                if len(cell_data[cc]) == 0:
                    pars_cur.append(np.full((ndat - 1,), np.nan, dtype=float))
                    cov_cur.append(np.full((ndat - 1, ndat - 1), np.nan, dtype=float))
                    n_cur.append(np.nan)
                    r2_cur.append(np.nan)
                    rs_cur.append(np.nan)
                    pdop_cur.append(np.nan)   
                    cond_cur.append(np.nan)
                    sphericity_cur.append(np.nan)
                    continue

                dat = np.asarray(cell_data[cc], dtype=float)
                svals = np.asarray(cell_s[cc], dtype=float)

                for col in range(dat.shape[1]):
                    m = np.nanmean(dat[:, 0])
                    sig = np.nanstd(dat[:, 0])
                    if np.isfinite(sig) and sig > 0:
                        keep = (dat[:, 0] >= m - f_vitesse_z * sig) & (dat[:, 0] <= m + f_vitesse_z * sig)
                        dat = dat[keep]
                        svals = svals[keep]

                if dat.shape[0] == 0:
                    pars_cur.append(np.full((ndat - 1,), np.nan, dtype=float))
                    cov_cur.append(np.full((ndat - 1, ndat - 1), np.nan, dtype=float))
                    n_cur.append(np.nan)
                    r2_cur.append(np.nan)
                    rs_cur.append(np.nan)
                    pdop_cur.append(np.nan)  
                    cond_cur.append(np.nan)
                    sphericity_cur.append(np.nan)
                    continue

                mx = np.nanmean(dat[:, 1])
                my = np.nanmean(dat[:, 2])
                mz = np.nanmean(dat[:, 3]) if dat.shape[1] > 3 else np.nan
                vdir = np.array([mx, my, mz], dtype=float)
                nv = np.linalg.norm(vdir)

                if np.isfinite(nv) and nv > 0 and dat.shape[1] > 3:
                    vdir = vdir / nv
                    seuil = max(f_direction_fixe - f_direction_pond * dat.shape[0], 0.95)
                    if vdir[2] < seuil and vdir[2] > 0.9:
                        pars_cur.append(np.full((ndat - 1,), np.nan, dtype=float))
                        cov_cur.append(np.full((ndat - 1, ndat - 1), np.nan, dtype=float))
                        n_cur.append(np.nan)
                        r2_cur.append(np.nan)
                        rs_cur.append(np.nan)
                        pdop_cur.append(np.nan)   
                        cond_cur.append(np.nan)
                        sphericity_cur.append(np.nan)
                        continue

                max_s = 1.01 * np.nanmax(np.abs(cur_s)) if np.any(np.isfinite(cur_s)) else 1.0
                if not np.isfinite(max_s) or max_s == 0:
                    max_s = 1.0

                sweight = -(svals / max_s) ** pond_vitesses + 1.0
                sweight = np.asarray(sweight, dtype=float).reshape(-1)
                sweight = np.clip(sweight, 1e-6, 1.0)

                y = dat[:, 0]
                X = dat[:, 1:]

                pars_cell_i, n_dat, r_carre_i, cov_i = self.fit_model(sweight, y, X)

                sw_i = np.sqrt(np.clip(sweight, 1e-12, np.inf))
                Aw_i = X * sw_i[:, None]
                try:
                    ata_i = Aw_i.T @ Aw_i
                    ata_inv_i = np.linalg.inv(ata_i)
                    pdop_i = float(np.sqrt(np.trace(ata_inv_i)))
                    cond_i = float(np.linalg.cond(Aw_i))
                except np.linalg.LinAlgError:
                    pdop_i = np.inf
                    cond_i = np.inf

                # ## 27/08
                
                # _cond_debug_list = []
                # if cc == 0:
                #     _cond_debug_list = []
                # if np.isfinite(cond_i):
                #     _cond_debug_list.append(cond_i)
                # if cc == ncells - 1:
                #     arr_cond = np.asarray(_cond_debug_list)
                #     print(
                #         f"DEBUG distribution cond_number (repeat transect {crp}) : "
                #         f"min={np.min(arr_cond):.1f}, p50={np.percentile(arr_cond,50):.1f}, "
                #         f"p90={np.percentile(arr_cond,90):.1f}, p99={np.percentile(arr_cond,99):.1f}, "
                #         f"max={np.max(arr_cond):.1f}"
                #     )


                # MAX_ACCEPTABLE_COND = 1e5  # seuil empirique, à ajuster si besoin
                # if not np.isfinite(cond_i) or cond_i > MAX_ACCEPTABLE_COND:
                #     pars_cell_i = np.full_like(pars_cell_i, np.nan)
                #     cov_i = np.full_like(cov_i, np.nan)
                # ##

                ## 02/09
                dir_cols = min(3, X.shape[1])
                Xdir = X[:, :dir_cols]
                try:
                    G = Xdir.T @ Xdir
                    eigvals = np.sort(np.linalg.eigvalsh(G))[::-1]  # décroissant : lambda1 >= lambda2 >= lambda3
                    sphericity_i = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 1e-12 else 0.0
                except np.linalg.LinAlgError:
                    sphericity_i = np.nan
                sphericity_cur.append(sphericity_i)
                ##


                residuals = y - X @ pars_cell_i
                r_sigma_i = float(np.nanstd(residuals))

                pars_cur.append(pars_cell_i)
                cov_cur.append(cov_i)
                n_cur.append(float(n_dat))
                r2_cur.append(float(r_carre_i))
                rs_cur.append(float(r_sigma_i))
                pdop_cur.append(pdop_i)
                cond_cur.append(cond_i)

                

            pars.append(pars_cur)
            cov_pars.append(cov_cur)
            n_vels.append(n_cur)
            r_carre.append(r2_cur)
            r_sigma.append(rs_cur)
            self.pdop_cell.append(pdop_cur)  
            self.cond_cell.append(cond_cur)
            self.sphericity_cell.append(sphericity_cur)

            p_array = np.asarray(pars_cur, dtype=float)
            print(f"--- Debug repeat transect {crp} --- pars shape: {p_array.shape}, "
                f"NaN count: {np.sum(np.isnan(p_array))}")
    

        return pars, r_sigma, r_carre, cov_pars, n_vels
            

    def get_velocity(self, f_vitesse=10.0, f_direction_fixe=0.97, f_direction_pond=0.0002, pond_vitesses=1.0):
        """
        Return Cartesian velocity and covariance on the mesh.
        """
        pars, r_sigma, r_carre, cov_pars, n_vels = self.get_parameters(
            f_vitesse_z=f_vitesse,
            f_direction_fixe=f_direction_fixe,
            f_direction_pond=f_direction_pond,
            pond_vitesses=pond_vitesses,
        )

        vel = []
        cov_vel = []
        for p, c in zip(pars, cov_pars):
            vel_i, cov_i = self.velocity_model.get_velocity(np.asarray(p, dtype=float), np.asarray(c, dtype=float))
            vel.append(vel_i)
            cov_vel.append(cov_i)

        return vel, cov_vel, n_vels, r_carre, r_sigma

    def rotate_to_xs(self, orig_vel, orig_cov=None):
        """
        Rotate Cartesian velocities to cross-section coordinates.
        """
        if self.xs is None:
            raise ValueError("rotate_to_xs requires an XSection object")

        if isinstance(orig_vel, list):
            vel_out = []
            cov_out = [] if orig_cov is not None else None
            for ii, vel_i in enumerate(orig_vel):
                cov_i = orig_cov[ii] if orig_cov is not None else None
                v, c = self.rotate_to_xs(vel_i, cov_i)
                vel_out.append(v)
                if cov_out is not None:
                    cov_out.append(c)
            return vel_out, cov_out

        vel = np.asarray(orig_vel, dtype=float)
        if vel.ndim != 2 or vel.shape[1] < 3:
            raise ValueError("orig_vel must be shaped (N, 3)")

        vels, veln = self.xs.xy2sn_vel(vel[:, 0], vel[:, 1])
        out = np.column_stack((vels, veln, vel[:, 2]))

        if orig_cov is None:
            return out, None

        cov = np.asarray(orig_cov, dtype=float)
        cov_out = np.full_like(cov, np.nan)

        d = np.asarray(self.xs.direction, dtype=float)
        do = np.asarray(self.xs.direction_orthogonal, dtype=float)
        T = np.array([[d[0], d[1]], [do[0], do[1]]], dtype=float)

        for ii in range(min(cov.shape[0], out.shape[0])):
            cxy = cov[ii, :2, :2]
            if np.all(np.isfinite(cxy)):
                cov_out[ii, :2, :2] = T @ cxy @ T.T
            if cov.shape[1] > 2 and cov.shape[2] > 2:
                cov_out[ii, 2, 2] = cov[ii, 2, 2]

        return out, cov_out

    def compute_rozovskii(self, orig_vel):
        """
        Project velocity using Rozovskii's formula.
        """
        if isinstance(orig_vel, list):
            return [self.compute_rozovskii(v) for v in orig_vel]

        vel = np.asarray(orig_vel, dtype=float)
        if vel.ndim != 2 or vel.shape[1] < 3:
            raise ValueError("orig_vel must be shaped (N, 3)")

        uv = vel[:, :2]
        finite = np.all(np.isfinite(uv), axis=1)
        if not np.any(finite):
            return np.full_like(vel, np.nan)

        mean_uv = np.nanmean(uv[finite], axis=0)
        nrm = np.linalg.norm(mean_uv)
        if nrm <= 0:
            return np.full_like(vel, np.nan)

        # vecs = -mean_uv / nrm
        vecs = mean_uv / nrm
        vecn = np.array([-vecs[1], vecs[0]], dtype=float)

        s = uv @ vecs
        n = uv @ vecn
        return np.column_stack((s, n, vel[:, 2]))