# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def _unwrap_repeat_transect(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(f"Forme inattendue : {arr.shape} (attendu (ncells,) ou (1,ncells))")


def plot_sphericity_map(mesh, sphericity_values, vmadcp=None, ax=None, cmap="RdYlGn"):

    from Classes_vermeulen.plot_mesh_bathy import _rg_left_orientation

    sphericity_values = _unwrap_repeat_transect(sphericity_values)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    if mesh.nb_all.size > 0:
        ax.plot(mesh.nb_all, mesh.zb_all, "k-", linewidth=2.0, label="Lit (maillage)", zorder=4)

    if mesh.ncells > 0:
        polygons = [np.c_[mesh.n_patch[:, cc], mesh.z_patch[:, cc]] for cc in range(mesh.ncells)]
        coll = PolyCollection(polygons, cmap=cmap, edgecolors="#444444", linewidths=0.3, zorder=2)
        coll.set_array(sphericity_values)
        coll.set_clim(0.0, 1.0) 
        ax.add_collection(coll)
        cb = fig.colorbar(coll, ax=ax)
        cb.set_label("Indice de sphéricité (0 = orientations colinéaires, 1 = isotrope)")

    wl = float(mesh.water_level)
    if mesh.nw.size > 0:
        nw_flat = mesh.nw.reshape(-1)
        ax.plot(nw_flat, np.full_like(nw_flat, wl), "b-", linewidth=2.5, label="Niveau d'eau", zorder=6)

    ax.set_xlabel("Distance le long de la section (n) [m]")
    ax.set_ylabel("Élévation [m]")
    ax.set_title("Diversité angulaire des faisceaux par cellule (sphéricité)")
    ax.grid(True, linestyle=":", alpha=0.6)

    if vmadcp is not None and hasattr(mesh, "xs") and mesh.nb_all.size > 0:
        start_is_left = _rg_left_orientation(mesh.xs, vmadcp)
        n_min, n_max = float(np.min(mesh.nb_all)), float(np.max(mesh.nb_all))
        margin = 0.05 * max(n_max - n_min, 1e-6)
        if start_is_left:
            ax.set_xlim(left=n_min - margin, right=n_max + margin)
        else:
            ax.set_xlim(left=n_max + margin, right=n_min - margin)
        ax.text(0.02, 0.95, "RG", transform=ax.transAxes, ha="left", va="top", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
        ax.text(0.98, 0.95, "RD", transform=ax.transAxes, ha="right", va="top", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

    ax.legend(loc="lower right")
    print(
        f"DEBUG sphéricité : min={np.nanmin(sphericity_values):.3f}, "
        f"médiane={np.nanmedian(sphericity_values):.3f}, "
        f"{int(np.sum(sphericity_values < 0.1))} cellule(s) < 0.1 (très fragile)"
    )
    return ax


def plot_sphericity_vs_count_scatter(nb_vel, sphericity_values, ax=None, threshold=0.1):
    
    nb_vel = _unwrap_repeat_transect(nb_vel)
    sphericity_values = _unwrap_repeat_transect(sphericity_values)
    valid = np.isfinite(nb_vel) & np.isfinite(sphericity_values)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    sc = ax.scatter(nb_vel[valid], sphericity_values[valid], c=sphericity_values[valid],
                    cmap="RdYlGn", vmin=0, vmax=1, s=18, edgecolors="#444444", linewidths=0.3)
    ax.axhline(y=threshold, color="r", linestyle="--", label=f"Seuil d'alerte = {threshold}")
    ax.set_xlabel("Nombre de mesures brutes par cellule")
    ax.set_ylabel("Indice de sphéricité")
    ax.set_title("Nombre de mesures vs diversité angulaire, par cellule")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.colorbar(sc, ax=ax, label="Sphéricité")
    return ax


def run_directional_sensitivity_analysis(mesh, vsolver, nb_vel, vmadcp=None):
    
    if not hasattr(vsolver, "sphericity_cell") or len(vsolver.sphericity_cell) == 0:
        raise AttributeError(
            "vsolver.sphericity_cell est vide -- as-tu bien appliqué le patch dans "
            "VelocitySolver.get_parameters ?"
        )
    sphericity0 = _unwrap_repeat_transect(vsolver.sphericity_cell)

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plot_sphericity_map(mesh, sphericity0, vmadcp=vmadcp, ax=ax1)
    plt.show(block=False); plt.pause(0.1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_sphericity_vs_count_scatter(nb_vel, sphericity0, ax=ax2)
    plt.show(block=False); plt.pause(0.1)

    return fig1, fig2