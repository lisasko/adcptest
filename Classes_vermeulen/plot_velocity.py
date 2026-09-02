from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .plot_mesh_bathy import _rg_left_orientation


def plot_velocity_cross_section(
    mesh,
    vel_sn,
    vmadcp=None,
    component: str = "streamwise",
    clim: tuple[float, float] | None = (-0.5, 0.5),
    ax=None,
):
    """
    Paramètres : 

    mesh : SigmaZetaMesh
        Maillage déjà résolu (mêmes n_left/n_right/... que pour la bathy).
    vel_sn : np.ndarray (ncells, 3)
        Vitesse par cellule dans le repère section (s = travers-section,
        n = long-section streamwise, z = verticale). C'est la sortie de
        VelocitySolver.rotate_to_xs(vel, cov_vel)[0].
    vmadcp : VMADCP, optionnel
        Utilisé uniquement pour orienter automatiquement l'axe RG/RD comme
        pour la bathymétrie (cohérence visuelle entre les deux graphiques).
    component : str
        'streamwise' (vel_sn[:,0], vitesse longitudinale = défaut) ou
        'secondary' (norme du courant secondaire vel_sn[:,1:3]).
    clim : tuple ou None
        Bornes fixes de la colorbar (m/s). None = auto.
    """

    vel_sn = np.asarray(vel_sn, dtype=float)
    if vel_sn.ndim != 2 or vel_sn.shape[0] != mesh.ncells or vel_sn.shape[1] < 3:
        raise ValueError("vel_sn doit être de forme (mesh.ncells, 3)")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    wl = float(mesh.water_level)

    # Ligne de fond (identique à la bathymétrie):
    if mesh.nb_all.size > 0:
        ax.plot(mesh.nb_all, mesh.zb_all, "k-", linewidth=2.0, label="Lit (maillage)", zorder=4)

    # Cellules colorées par la vitesse :
    if component == "streamwise":
        values = vel_sn[:, 0]
        label = "Vitesse streamwise (m/s)"
    elif component == "secondary":
        values = np.linalg.norm(vel_sn[:, 1:3], axis=1)
        label = "Vitesse secondaire (m/s)"
    else:
        raise ValueError("component doit être 'streamwise' ou 'secondary'")

    clim_used = clim

    if mesh.ncells > 0:
        polygons = [
            np.c_[mesh.n_patch[:, cc], mesh.z_patch[:, cc]]
            for cc in range(mesh.ncells)
        ]
        coll = PolyCollection(polygons, cmap="jet", edgecolors="#444444", linewidths=0.3, zorder=2)
        coll.set_array(values)

        if clim is not None:
            coll.set_clim(*clim)
        else :
            vmin = float(np.nanmin(values)) if np.any(np.isfinite(values)) else 0.0
            vmax = float(np.nanmax(values)) if np.any(np.isfinite(values)) else 1.0
            coll.set_clim(vmin, vmax)
            clim_used = (vmin, vmax)

        ax.add_collection(coll)
        cb = fig.colorbar(coll, ax=ax)
        cb.set_label(label)

        # Flèches de courant secondaire (transversal, vertical):
        vel_sec = vel_sn[:, 1:3].copy()
        vec_norm = np.linalg.norm(vel_sec, axis=1)
        vec_norm[np.isnan(vec_norm)] = 0.0
        mean_norm = float(np.mean(vec_norm)) if vec_norm.size > 0 else 0.0

        if mean_norm > 0:
            outliers = vec_norm > 4.0 * mean_norm
            vel_sec[outliers] = 0.0

            ax.quiver(
                mesh.n_center, mesh.z_center,
                vel_sec[:, 0], vel_sec[:, 1],
                color="k", linewidth=1.0, angles="xy",
                scale_units="xy", scale=1.0, zorder=5,
            )

            x0, y0 = 0.06, 0.04
            arrow_len = 0.05  
            ax.annotate(
                "", xy=(x0 + arrow_len, y0), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5),
            )
            ax.text(
                x0 + arrow_len + 0.01, y0, "1 m/s", transform=ax.transAxes,
                fontsize=8, va="center", ha="left",
            )
           

    # Niveau d'eau :
    if mesh.nw.size > 0:
        nw_flat = mesh.nw.reshape(-1)
        ax.plot(nw_flat, np.full_like(nw_flat, wl), "b-", linewidth=2.5, label="Niveau d'eau", zorder=6)
    else:
        ax.axhline(y=wl, color="b", linestyle="-", linewidth=2.5, label="Niveau d'eau", zorder=6)

    ax.set_xlabel("Distance le long de la section (n) [m]")
    ax.set_ylabel("Élévation [m]")
    ax.set_title(f"Profil des vitesses ({component})")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Orientation RG/RD (même logique que la bathymétrie) :
    start_is_left = None
    if vmadcp is not None and hasattr(mesh, "xs"):
        start_is_left = _rg_left_orientation(mesh.xs, vmadcp)

    if mesh.nb_all.size > 0:
        n_min, n_max = float(np.min(mesh.nb_all)), float(np.max(mesh.nb_all))
        margin = 0.05 * max(n_max - n_min, 1e-6)
        if start_is_left is None or start_is_left:
            ax.set_xlim(left=n_min - margin, right=n_max + margin)
        else:
            ax.set_xlim(left=n_max + margin, right=n_min - margin)

    ax.text(0.02, 0.95, "RG", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
    ax.text(0.98, 0.95, "RD", transform=ax.transAxes, ha="right", va="top",
            fontweight="bold", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    ax.legend(loc="lower right")
    return ax, clim_used