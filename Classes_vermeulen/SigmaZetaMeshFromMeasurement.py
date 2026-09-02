from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .SigmaZetaMesh import SigmaZetaMesh


@dataclass
class SigmaZetaMeshFromMeasurement:
    """Builds a Sigma-Zeta mesh from 1D section profiles, MATLAB-style."""

    deltan: float
    deltaz: float
    beam_angle_deg: float

    def build(self, n_profile: np.ndarray, depth_profile: np.ndarray) -> SigmaZetaMesh:
        n_profile = np.asarray(n_profile, dtype=float)
        depth_profile = np.asarray(depth_profile, dtype=float)
        finite = np.isfinite(n_profile) & np.isfinite(depth_profile) & (depth_profile > 0)
        n_profile = n_profile[finite]
        depth_profile = depth_profile[finite]
        if n_profile.size < 2:
            raise ValueError("Not enough valid profile points to build mesh")

        order = np.argsort(n_profile)
        n_profile = n_profile[order]
        depth_profile = depth_profile[order]

        nmin = float(np.nanmin(n_profile))
        nmax = float(np.nanmax(n_profile))
        n_edges = np.arange(nmin, nmax + self.deltan, self.deltan)
        if n_edges[-1] < nmax:
            n_edges = np.append(n_edges, nmax)

        n_left = n_edges[:-1]
        n_right = n_edges[1:]
        n_middle = 0.5 * (n_left + n_right)

        d_left = np.interp(n_left, n_profile, depth_profile)
        d_mid = np.interp(n_middle, n_profile, depth_profile)
        d_right = np.interp(n_right, n_profile, depth_profile)

        sigma_cap = float(np.clip(np.cos(np.deg2rad(self.beam_angle_deg)), 0.05, 1.0))
        maxz = float(np.nanmax(depth_profile * sigma_cap))

        sig_left_max = np.clip(maxz / np.maximum(d_left, 1e-6), 0.05, sigma_cap)
        sig_mid_max = np.clip(maxz / np.maximum(d_mid, 1e-6), 0.05, sigma_cap)
        sig_right_max = np.clip(maxz / np.maximum(d_right, 1e-6), 0.05, sigma_cap)

        nz = np.clip(np.ceil((sig_mid_max * d_mid) / max(self.deltaz, 1e-6)).astype(int), 1, None)
        max_num = int(np.nanmax(nz))

        col_to_mat = np.tile(np.arange(len(n_middle)), (max_num, 1))
        row_to_mat = np.tile(np.arange(1, max_num + 1).reshape(-1, 1), (1, len(n_middle)))
        mat_to_cell = (row_to_mat <= nz)

        row_to_cell = row_to_mat[mat_to_cell].astype(int)
        col_to_cell = col_to_mat[mat_to_cell].astype(int)

        sig_bottom_left = (row_to_cell / nz[col_to_cell]) * sig_left_max[col_to_cell]
        sig_top_left = ((row_to_cell - 1) / nz[col_to_cell]) * sig_left_max[col_to_cell]
        sig_bottom_mid = (row_to_cell / nz[col_to_cell]) * sig_mid_max[col_to_cell]
        sig_top_mid = ((row_to_cell - 1) / nz[col_to_cell]) * sig_mid_max[col_to_cell]
        sig_bottom_right = (row_to_cell / nz[col_to_cell]) * sig_right_max[col_to_cell]
        sig_top_right = ((row_to_cell - 1) / nz[col_to_cell]) * sig_right_max[col_to_cell]

        return SigmaZetaMesh(
            n_left=n_left,
            n_middle=n_middle,
            n_right=n_right,
            sig_bottom_left=sig_bottom_left,
            sig_top_left=sig_top_left,
            sig_bottom_mid=sig_bottom_mid,
            sig_top_mid=sig_top_mid,
            sig_bottom_right=sig_bottom_right,
            sig_top_right=sig_top_right,
            row_to_cell=row_to_cell,
            col_to_cell=col_to_cell,
        )
