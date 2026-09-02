from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def _rg_left_orientation(xs, vmadcp, n_ensembles_edge: int = 5) -> bool:
    """
    Détermine si le transect démarre en Rive Gauche (RG), en comparant la
    coordonnée n projetée du début du transect (moyenne des premiers
    ensembles) à celle de la fin (moyenne des derniers ensembles).

    Retourne True si le transect démarre au n le plus petit (RG affichée
    à gauche du graphique), False sinon.
    """
    hp = np.asarray(getattr(vmadcp, "horizontal_position", np.empty((2, 0))), dtype=float)
    if hp.ndim != 2 or hp.shape[1] == 0:
        return True

    x = hp[0, :]
    y = hp[1, :]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        return True

    edge = min(n_ensembles_edge, max(1, x.size // 2))

    _, n_start = xs.xy2sn(np.array([np.mean(x[:edge])]), np.array([np.mean(y[:edge])]))
    _, n_end = xs.xy2sn(np.array([np.mean(x[-edge:])]), np.array([np.mean(y[-edge:])]))

    return bool(n_start[0] < n_end[0])


def plot_bathymetry_3d(bathy, mesh=None, ax=None):
    """
    Vue 3D de la surface de bathymétrie interpolée (colorée par
    élévation du lit), des points connus bruts, et, si un maillage est
    fourni, de la ligne de fond du maillage et du niveau d'eau
    superposés.

    Transposition de la séquence MATLAB dans demo.m :
        Bathy(cs).plot;
        hc = colorbar; ylabel(hc, 'bed elevation (m)')
        hold on
        Mesh(cs).plot3;
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    hp, ht = bathy.plot(ax=ax, return_handles=True)

    if ht is not None:
        cb = fig.colorbar(ht, ax=ax, shrink=0.7)
        cb.set_label("Bed elevation (m)")

    if mesh is not None:

        if mesh.nb_all.size > 0 and mesh.zb_all.size == mesh.nb_all.size:
            # ax.plot(mesh.xb_all, mesh.yb_all, mesh.zb_all, "k", linewidth=2, label="Ligne de fond (maillage)")
            ##
            xb = np.asarray(mesh.xb_all, dtype=float).reshape(-1)
            yb = np.asarray(mesh.yb_all, dtype=float).reshape(-1)
            zb = np.asarray(mesh.zb_all, dtype=float).reshape(-1)
            ax.plot(xb, yb, zs=zb, color="k", linewidth=2, label="Ligne de fond (maillage)")
            ##

        if mesh.nw.size > 0:
            # ax.plot(
            #     mesh.xw, mesh.yw,
            #     np.full_like(mesh.xw, float(mesh.water_level)),
            #     "b", linewidth=2, label="Niveau d'eau",
            # )

            nw_flat = np.asarray(mesh.nw, dtype=float).reshape(-1)
            xw_flat, yw_flat = mesh.xs.sn2xy(np.zeros_like(nw_flat), nw_flat)
            zw_flat = np.full_like(nw_flat, float(mesh.water_level))
            ax.plot(xw_flat, yw_flat, zs=zw_flat, color="b", linewidth=2, label="Niveau d'eau")

        ax.legend(loc="upper left")

    ax.set_xlabel("UTM x (m)")
    ax.set_ylabel("UTM y (m)")
    ax.set_zlabel("Elévation (m)")
    ax.set_title("Bathymétrie et maillage")

    return ax


def plot_cross_section_2d(mesh, bathy, xs, vmadcp=None, color_by=None, color_label="Valeur", ax=None):
    """
    Profil en travers 2D : bathymétrie moyenne, mailles du maillage
    Sigma-Zeta, niveau d'eau, orientation RG/RD, et boîte d'annotation
    interactive (n, z, X_UTM, Y_UTM) au clic.

    Paramètres
    ----------
    mesh : SigmaZetaMesh
        Maillage déjà construit (SigmaZetaMeshFromVMADCP.get_mesh()).
    bathy : BathymetryScatteredPoints
        Bathymétrie utilisée pour construire le maillage (pour l'affichage
        des points de mesure bruts, à titre indicatif seulement).
    xs : XSection
        Section transversale, utilisée pour projeter les clics de souris
        en coordonnées UTM.
    vmadcp : VMADCP, optionnel
        Utilisé pour déterminer automatiquement l'orientation RG/RD
        (sens du transect). Si absent, l'axe n'est pas inversé.
    color_by : array-like, optionnel
        Une valeur par cellule du maillage (mesh.ncells éléments), par
        exemple une vitesse moyenne par cellule, pour colorer les mailles.
        Si None, les mailles sont tracées avec un remplissage transparent
        (juste les contours), pour visualiser la structure du maillage.
    color_label : str
        Légende de la barre de couleur si color_by est fourni.
    ax : matplotlib.axes.Axes, optionnel
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    wl = float(mesh.water_level)

    # Ligne de fond : 
    if mesh.nb_all.size > 0:
        ax.plot(mesh.nb_all, mesh.zb_all, "k-", linewidth=2.0, label="Lit (maillage)", zorder=4)

    # Points de bathymétrie bruts, projetés sur la section : 
    if bathy is not None and bathy.known.shape[1] > 0:
        _, known_n = xs.xy2sn(bathy.known[0, :], bathy.known[1, :])
        finite = np.isfinite(known_n)
        ax.scatter(
            known_n[finite], bathy.known[2, finite],
            c="0.4", s=8, alpha=0.5, label="Points de bathymétrie mesurés", zorder=3
        )

    # Mailles du maillage, tracées comme des polygones. n_patch / z_patch :
    if mesh.ncells > 0:
        polygons = [
            np.c_[mesh.n_patch[:, cc], mesh.z_patch[:, cc]]
            for cc in range(mesh.ncells)
        ]
        if color_by is not None:
            values = np.asarray(color_by, dtype=float).reshape(-1)
            if values.size != mesh.ncells:
                raise ValueError("color_by must have exactly one value per mesh cell")
            coll = PolyCollection(polygons, cmap="viridis", edgecolors="#444444", linewidths=0.4, zorder=2)
            coll.set_array(values)
            ax.add_collection(coll)
            cb = fig.colorbar(coll, ax=ax)
            cb.set_label(color_label)
        else:
            coll = PolyCollection(polygons, facecolors="none", edgecolors="#444444", linewidths=0.4, zorder=2)
            ax.add_collection(coll)

    # Niveau d'eau : 
    if mesh.nw.size > 0:
        nw_flat = mesh.nw.reshape(-1)
        ax.plot(nw_flat, np.full_like(nw_flat, wl), "b-", linewidth=2.5, label="Niveau d'eau", zorder=5)
    else:
        ax.axhline(y=wl, color="b", linestyle="-", linewidth=2.5, label="Niveau d'eau", zorder=5)

    ax.set_xlabel("Distance le long de la section (n) [m]")
    ax.set_ylabel("Elévation [m]")
    ax.set_title("Profil en travers : bathymétrie et maillage Sigma-Zeta")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Orientation RG/RD :
    start_is_left = None
    if vmadcp is not None:
        start_is_left = _rg_left_orientation(xs, vmadcp)

    if mesh.nb_all.size > 0:
        n_min, n_max = float(np.min(mesh.nb_all)), float(np.max(mesh.nb_all))
        margin = 0.05 * max(n_max - n_min, 1e-6)
        if start_is_left is None or start_is_left:
            ax.set_xlim(left=n_min - margin, right=n_max + margin)
        else:
            ax.set_xlim(left=n_max + margin, right=n_min - margin)

    ax.text(
        0.02, 0.95, "RG", transform=ax.transAxes, ha="left", va="top",
        fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )
    ax.text(
        0.98, 0.95, "RD", transform=ax.transAxes, ha="right", va="top",
        fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )

    ax.legend(loc="lower right")

    # Boîte d'annotation interactive : 
    coord_box = ax.annotate(
        "Cliquez sur le graphique\nn = -- m | z = -- m\nX = -- | Y = --",
        xy=(0.5, 0.02), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffd1", edgecolor="#cccccc", alpha=0.95),
        zorder=10,
    )

    def _on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        n_click = event.xdata
        z_click = event.ydata
        try:
            x_utm, y_utm = xs.sn2xy(np.array([0.0]), np.array([n_click]))
            x_str = f"{x_utm[0]:.2f}"
            y_str = f"{y_utm[0]:.2f}"
        except Exception:
            x_str, y_str = "--", "--"
        coord_box.set_text(
            f"n = {n_click:.2f} m | z = {z_click:.2f} m\nX UTM = {x_str} m | Y UTM = {y_str} m"
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)

    return ax


def plot_bathymetry_and_mesh(bathy, mesh, xs, vmadcp=None, color_by=None):
    """
    Fonction globale de visualisation pour une mesure donnée.
    """
    ax3d = plot_bathymetry_3d(bathy, mesh=mesh)
    ax2d = plot_cross_section_2d(mesh, bathy, xs, vmadcp=vmadcp, color_by=color_by)
    return ax3d, ax2d