# -*- coding: utf-8 -*-
"""
PDOP = Position Dilution Of Precision, indicateur de sensibilité géométrique des mesures de vitesse par cellule.

- PDOP- = sqrt(trace((A^T.A)^-1))
On veut un PDOP faible (en dessous ou proche de 1)

- Nombre de conditionnement = sigma_max(A) / sigma_min(A)
  Indicateur de mauvais conditionnement numérique. Une valeur
  très élevée (typiquement > 100-1000) signale une direction de l'espace
  quasiment non contrainte par les données disponibles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from Classes_vermeulen.plot_mesh_bathy import _rg_left_orientation  



def _unwrap_repeat_transect(x):

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(
        f"Forme inattendue pour un tableau par-cellule : {arr.shape}. "
        "Attendu (ncells,) ou (1, ncells) -- un seul repeat transect."
    )


def plot_pdop_map(mesh, pdop_values, indicator="pdop", vmadcp=None, clim=None,
                  cmap="RdYlGn_r", ax=None, log_scale=True):
    
    pdop_values = np.asarray(pdop_values, dtype=float)
    values_plot = np.log10(np.clip(pdop_values, 1e-6, np.inf)) if log_scale else pdop_values
    label = ("log10(PDOP)" if indicator == "pdop" else "log10(nb. conditionnement)") if log_scale \
        else ("PDOP" if indicator == "pdop" else "Nb. conditionnement")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    if mesh.nb_all.size > 0:
        ax.plot(mesh.nb_all, mesh.zb_all, "k-", linewidth=2.0, label="Lit (maillage)", zorder=4)

    if mesh.ncells > 0:
        polygons = [np.c_[mesh.n_patch[:, cc], mesh.z_patch[:, cc]] for cc in range(mesh.ncells)]
        coll = PolyCollection(polygons, cmap=cmap, edgecolors="#444444", linewidths=0.3, zorder=2)
        coll.set_array(values_plot)

        if clim is not None:
            coll.set_clim(*clim)
        else:
            finite_vals = values_plot[np.isfinite(values_plot)]
            if finite_vals.size > 0:
                vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
                coll.set_clim(vmin, vmax)

        ax.add_collection(coll)
        cb = fig.colorbar(coll, ax=ax)
        cb.set_label(label)

    wl = float(mesh.water_level)
    if mesh.nw.size > 0:
        nw_flat = mesh.nw.reshape(-1)
        ax.plot(nw_flat, np.full_like(nw_flat, wl), "b-", linewidth=2.5, label="Niveau d'eau", zorder=6)

    ax.set_xlabel("Distance le long de la section (n) [m]")
    ax.set_ylabel("Élévation [m]")
    ax.set_title(f"Sensibilité géométrique par cellule ({indicator.upper()})")
    ax.grid(True, linestyle=":", alpha=0.6)

    if vmadcp is not None and hasattr(mesh, "xs") and mesh.nb_all.size > 0:
        start_is_left = _rg_left_orientation(mesh.xs, vmadcp)
        n_min, n_max = float(np.min(mesh.nb_all)), float(np.max(mesh.nb_all))
        margin = 0.05 * max(n_max - n_min, 1e-6)
        if start_is_left:
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
    return ax


def plot_sensitivity_scatter(nb_vel, pdop_values, ax=None, pdop_warning_threshold=10.0):

    nb_vel = _unwrap_repeat_transect(nb_vel)
    pdop_values = _unwrap_repeat_transect(pdop_values)

    valid = np.isfinite(nb_vel) & np.isfinite(pdop_values) & (pdop_values > 0)
    nb_vel_v, pdop_v = nb_vel[valid], pdop_values[valid]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    sc = ax.scatter(nb_vel_v, pdop_v, c=pdop_v, cmap="RdYlGn_r", s=18,
                    edgecolors="#444444", linewidths=0.3, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Nombre de mesures brutes par cellule (nb_vel)")
    ax.set_ylabel("PDOP (échelle log)")
    ax.set_title("Sensibilité géométrique : nb. mesures vs PDOP, par cellule")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax.axhline(y=pdop_warning_threshold, color="r", linestyle="--", linewidth=1.0,
              label=f"Seuil d'alerte PDOP = {pdop_warning_threshold:.0f}")

    n_fragile = int(np.sum((pdop_v > pdop_warning_threshold)))
    n_fragile_high_count = int(np.sum((pdop_v > pdop_warning_threshold) & (nb_vel_v > np.nanmedian(nb_vel_v))))
    ax.text(
        0.02, 0.02,
        f"{n_fragile} cellules au-dessus du seuil\n"
        f"dont {n_fragile_high_count} avec nb_vel > médiane\n"
        f"(= beaucoup de mesures, mal réparties)",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85),
    )

    fig.colorbar(sc, ax=ax, label="PDOP")
    ax.legend(loc="upper right")

    print(
        f"DEBUG sensibilité géométrique : {n_fragile}/{nb_vel_v.size} cellules "
        f"au-dessus du seuil PDOP={pdop_warning_threshold:.0f}, dont "
        f"{n_fragile_high_count} avec nb_vel > médiane ({np.nanmedian(nb_vel_v):.0f}) "
        f"-- ce sont les cellules 'beaucoup de mesures mais mal réparties'."
    )

    return ax


def run_sensitivity_analysis(mesh, vsolver, nb_vel, vmadcp=None, pdop_warning_threshold=10.0):

    if not hasattr(vsolver, "pdop_cell") or len(vsolver.pdop_cell) == 0:
        raise AttributeError(
            "vsolver.pdop_cell est vide -- as-tu bien appliqué le patch dans "
            "VelocitySolver.get_parameters (calcul de pdop_cell/cond_cell) "
            "et appelé get_velocity()/get_parameters() avant cet appel ?"
        )

    pdop_cell0 = _unwrap_repeat_transect(vsolver.pdop_cell)
    nb_vel_arr = _unwrap_repeat_transect(nb_vel)

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plot_pdop_map(mesh, pdop_cell0, indicator="pdop", vmadcp=vmadcp, ax=ax1)
    plt.show(block=False)
    plt.pause(0.1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_sensitivity_scatter(nb_vel_arr, pdop_cell0, ax=ax2, pdop_warning_threshold=pdop_warning_threshold)
    plt.show(block=False)
    plt.pause(0.1)

    return fig1, fig2