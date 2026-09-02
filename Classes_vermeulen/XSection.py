from __future__ import annotations

import numpy as np

"""
    Define a cross section 

    XSection properties:
       origin - origin coordinates of the cross-section
       direction - tangential direction of the cross-section
    
    XSection properties (read only):
       direction_orthogonal - orthogonal direction of the cross-section
    
    XSection methods:
       xy2sn - transform xy coordinates of points and vectors to sn coordinates
       sn2xy - transform sn coordinates of points and vectors to xy coordinates
       xy2sn_vel - rotate xy velocity to sn velocity
       sn2xy_vel - rotate sn velocity to xy velocity
       xy2sn_tens -rotate xy tensor to sn tensor
       sn2xy_tens - rotate sn tensor to xy tensor
       plot - plots the tangential and orthogonal vectors at the origin
       set_from_vmadcp - sets the origin and direction based on the vmadcp track
       revert - revert the direction of the cross-section

"""

class XSection:

    def __init__(self, *args, origin=None, direction=None, scale: float = 10.0) -> None:

        # Default values :
        self.origin = np.array([0.0, 0.0], dtype=float) if origin is None else np.asarray(origin, dtype=float).reshape(2)
        self.direction = np.array([1.0, 0.0], dtype=float) if direction is None else np.asarray(direction, dtype=float).reshape(2)
        self.scale = float(scale)

        filter_obj = None
        construct_from_vmadcp = False
        vmadcp_obj = None

        for arg in args:
            if hasattr(arg, "horizontal_position"):
                vmadcp_obj = arg
            elif arg.__class__.__name__ == "EnsembleFilter":
                filter_obj = arg

        if vmadcp_obj is not None:
            # print(f"DEBUG: VMADCP détecté, calcul de la section à partir de track.shape = {vmadcp_obj.horizontal_position.shape}")
            self.set_from_vmadcp(vmadcp_obj, filter_obj)       
        
               



    @property
    def direction_orthogonal(self) -> np.ndarray:
        """Unit vector orthogonal to the tangential direction """
        return np.array([self._direction[1], -self._direction[0]], dtype=float)

    @direction_orthogonal.setter
    def direction_orthogonal(self, val) -> None:

        vec_orth = np.asarray(val, dtype=float).reshape(2)
        new_dir = np.array([-vec_orth[1], vec_orth[0]], dtype=float)
        self.direction = new_dir

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @direction.setter
    def direction(self, val) -> None:
        vec = np.asarray(val, dtype=float).reshape(2)
        norm = np.linalg.norm(vec)
        if norm <= 0:
            raise ValueError("direction cannot be zero")
        self._direction = vec / norm

    def xy2sn(self, x, y, u=None, v=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x and y must have same shape")

        d_orth = self.direction_orthogonal
        s = (x - self.origin[0]) * d_orth[0] + (y - self.origin[1]) * d_orth[1]
        n = (x - self.origin[0]) * self.direction[0] + (y - self.origin[1]) * self.direction[1]

        if u is None or v is None:
            return s, n
        us, un = self.xy2sn_vel(u, v)
        return s, n, us, un

    def sn2xy(self, s, n, us=None, un=None):
        s = np.asarray(s, dtype=float)
        n = np.asarray(n, dtype=float)
        if s.shape != n.shape:
            raise ValueError("s and n must have same shape")

        d_orth = self.direction_orthogonal
        x = self.origin[0] + self.direction[0] * n + d_orth[0] * s
        y = self.origin[1] + self.direction[1] * n + d_orth[1] * s

        if us is None or un is None:
            return x, y
        u, v = self.sn2xy_vel(us, un)
        return x, y, u, v

    def xy2sn_vel(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if u.shape != v.shape:
            raise ValueError("u and v must have same shape")
        d_orth = self.direction_orthogonal
        us = u * d_orth[0] + v * d_orth[1]
        un = u * self.direction[0] + v * self.direction[1]
        return us, un

    def sn2xy_vel(self, us, un):
        us = np.asarray(us, dtype=float)
        un = np.asarray(un, dtype=float)
        if us.shape != un.shape:
            raise ValueError("us and un must have same shape")
        d_orth = self.direction_orthogonal
        u = self.direction[0] * un + d_orth[0] * us
        v = self.direction[1] * un + d_orth[1] * us
        return u, v
    

    def xy2sn_tens(self, T):
        """Transform tensors (2x2 matrices) from xy to sn coordinates."""
        T = np.asarray(T, dtype=float)
        assert T.shape[-2:] == (2, 2), "Trailing dimensions of T must be (2, 2)"
        
        # Matrice de passage M
        M = np.vstack([self.direction_orthogonal, self.direction])
        
        # Calcul de M @ T @ M.T sur toutes les dimensions (Einsum gère le multi-dimensionnel nativement)
        return np.einsum('...ij,jk,kl->...il', M, T, M.T)

    def sn2xy_tens(self, T_sn):
        """Transform tensors (2x2 matrices) from sn to xy coordinates."""
        T_sn = np.asarray(T_sn, dtype=float)
        assert T_sn.shape[-2:] == (2, 2), "Trailing dimensions of T_sn must be (2, 2)"
        
        M = np.column_stack([self.direction_orthogonal, self.direction])
        return np.einsum('...ij,jk,kl->...il', M, T_sn, M.T)


    def set_from_vmadcp(self, vmadcp, filter_obj=None):

        hp = np.asarray(vmadcp.horizontal_position, dtype=float)
        if hp.ndim != 2 or hp.shape[0] < 2:
            raise ValueError("vmadcp.horizontal_position must be shaped (2, N)")

        x = hp[0, :]  # easting
        y = hp[1, :]  # northing

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size < 2:
            print("Pas assez de points valides pour calculer la section.")
            return

        xy = np.column_stack((x, y))
        cov = np.cov(xy, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        principal = eigvec[:, np.argmax(eigval)]
        self.direction = principal 

        self.origin = np.array([np.nanmean(x), np.nanmean(y)], dtype=float)

        _, n = self.xy2sn(x, y)
        n0 = 0.5 * (np.nanmax(n) + np.nanmin(n))
        self.scale = float(np.nanstd(n)) if np.isfinite(np.nanstd(n)) else self.scale
        ox, oy = self.sn2xy(0.0, n0)
        self.origin = np.array([float(ox), float(oy)], dtype=float)


    def revert(self):
        self.direction = -self.direction

    def plot(self, ax=None, scale: float | None = None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        vec_scale = self.scale if scale is None else float(scale)
        d_orth = self.direction_orthogonal

        h1 = ax.quiver(self.origin[0], self.origin[1], self.direction[0] * vec_scale, self.direction[1] * vec_scale, color="r", angles="xy", scale_units="xy", scale=1)
        h2 = ax.quiver(self.origin[0], self.origin[1], d_orth[0] * vec_scale, d_orth[1] * vec_scale, color="g", angles="xy", scale_units="xy", scale=1)
        ax.set_aspect("equal", adjustable="box")
        return h1, h2
