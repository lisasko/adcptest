from __future__ import annotations

import numpy as np


class VelocityModel:
    """Cartesian velocity model compatible with MATLAB VelocityModel."""

    @property
    def npars(self) -> np.ndarray:
        return self.get_npars()

    def get_npars(self) -> np.ndarray:
        return np.array([1, 1, 1], dtype=int)

    def get_model(
        self,
        d_time: np.ndarray,
        d_s: np.ndarray,
        d_n: np.ndarray,
        d_z: np.ndarray,
        d_sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = np.asarray(d_time).size
        ones = np.ones((n, 1), dtype=float)
        return ones, ones, ones

    def get_velocity(
        self,
        pars: np.ndarray,
        cov_pars: np.ndarray | None = None,
        d_time: np.ndarray | float | None = None,
        d_s: np.ndarray | float | None = None,
        d_n: np.ndarray | float | None = None,
        d_z: np.ndarray | float | None = None,
        d_sigma: np.ndarray | float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pars = np.asarray(pars, dtype=float)
        if pars.ndim == 1:
            pars = pars.reshape(1, -1)

        nin = pars.shape[0]

        def _as_vec(value):
            if value is None:
                return np.zeros((nin,), dtype=float)
            arr = np.asarray(value, dtype=float)
            if arr.size == 1:
                return np.full((nin,), float(arr), dtype=float)
            return arr.reshape(-1)

        d_time_v = _as_vec(d_time)
        d_s_v = _as_vec(d_s)
        d_n_v = _as_vec(d_n)
        d_z_v = _as_vec(d_z)
        d_sigma_v = _as_vec(d_sigma)

        Mu, Mv, Mw = self.get_model(d_time_v, d_s_v, d_n_v, d_z_v, d_sigma_v)
        Mu = np.asarray(Mu, dtype=float)
        Mv = np.asarray(Mv, dtype=float)
        Mw = np.asarray(Mw, dtype=float)

        np_u, np_v, np_w = [int(v) for v in self.npars]
        total_pars = np_u + np_v + np_w

        if pars.shape[1] < total_pars:
            raise ValueError("pars has fewer columns than required by model npars")

        pu = pars[:, :np_u]
        pv = pars[:, np_u : np_u + np_v]
        pw = pars[:, np_u + np_v : total_pars]

        u = np.sum(Mu * pu, axis=1)
        v = np.sum(Mv * pv, axis=1)
        w = np.sum(Mw * pw, axis=1)
        vel = np.column_stack((u, v, w))

        M = np.zeros((nin, 3, total_pars), dtype=float)
        M[:, 0, :np_u] = Mu
        M[:, 1, np_u : np_u + np_v] = Mv
        M[:, 2, np_u + np_v : total_pars] = Mw

        if cov_pars is None:
            cov_vel = np.full((nin, 3, 3), np.nan, dtype=float)
            return vel, cov_vel

        cov_pars = np.asarray(cov_pars, dtype=float)

        if cov_pars.ndim == 2:
            cov_vel = np.einsum("nip,pq,njq->nij", M, cov_pars, M)
        elif cov_pars.ndim == 3:
            if cov_pars.shape[0] == 1:
                cov_vel = np.einsum("nip,pq,njq->nij", M, cov_pars[0], M)
            elif cov_pars.shape[0] == nin:
                cov_vel = np.einsum("nip,npq,njq->nij", M, cov_pars, M)
            else:
                raise ValueError("cov_pars has incompatible leading dimension")
        else:
            raise ValueError("cov_pars must be a 2D or 3D array")

        return vel, cov_vel