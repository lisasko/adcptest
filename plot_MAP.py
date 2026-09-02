# -*- coding: utf-8 -*-

import os
import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
qrevint_path = os.path.join(parent_dir, 'qrevint_21_03')
if qrevint_path not in sys.path:
    sys.path.insert(0, qrevint_path)

from MAP_class_streamwise import MAP_streamwise
from MAP_class_roz import MAP_roz
from open_functions import new_settings
from common_functions import round_it
from Classes_vermeulen.plot_velocity import plot_velocity_cross_section
from Classes_vermeulen.plot_mesh_bathy import _rg_left_orientation


# Alignement du référentiel de navigation 
def _align_navigation_reference(meas, navigation_reference_user):

    nav_ref_avant = meas.current_settings()['NavRef']
    print(f"DEBUG MAP : référence de navigation AVANT alignement = {nav_ref_avant}")

    if navigation_reference_user == 'GPS':
        checked_idx0 = meas.checked_transect_idx[0]
        if not meas.transects[checked_idx0].gps:
            print(
                "ATTENTION MAP : GPS demandé pour l'alignement mais indisponible " # Impossible en pratique car la méthode Vermeulen ne fonctionne que sur des mesures avec GPS
            )
            navigation_reference_user = 'BT'

    meas, checked_transect_idx, navigation_reference = new_settings(
        meas,
        navigation_reference_user=navigation_reference_user,
        checked_transect_user=None,
        extrap_velocity=False,
    )

    nav_ref_apres = meas.current_settings()['NavRef']
    print(f"DEBUG MAP : référence de navigation APRES alignement = {nav_ref_apres}")

    if navigation_reference_user == 'GPS' and nav_ref_apres != 'gga_vel':
        raise RuntimeError(
            f"Incohérence de référence de navigation : GPS demandé mais "
            f"MAP est resté sur '{nav_ref_apres}'. La comparaison de débit "
            f"Vermeulen/MAP ne serait pas fiable."
        )
    
    return meas


# Reconstruction de la géométrie et du maillage 

def _extract_map_geometry(average_profile):
    """"
    Returns
        dict avec les clés :
            list_poly_vertices, vect_vel, quiver_x, quiver_z, quiver_v, quiver_w,
            x_axis, elevation_bed, quiver_scale
    """
    self = average_profile

    # Détection du type de vitesse voulu (streamwise ou Rozovskii)
    if hasattr(self, "left_streamwise_velocity"):
        is_roz = False
    elif hasattr(self, "left_primary_velocity"):
        is_roz = True
    else:
        raise AttributeError(
            f"Objet MAP de type {type(self).__name__} non reconnu : ni "
            "'left_streamwise_velocity' (MAP_streamwise) ni 'left_primary_velocity' "
            "(MAP_roz) ne sont présents."
        )

    def _vel(streamwise_name, roz_name):
        """Lit l'attribut sous son nom natif selon le type d'objet détecté."""
        return getattr(self, roz_name if is_roz else streamwise_name)

    # Attributs géométriques 
    required_common = [
        "left_distance", "left_coef", "right_coef", "borders_ens",
        "depths", "depth_cells_border", "left_borders", "right_borders",
        "nodes_depth_raw", "left_mid_cells_x", "right_mid_cells_x",
        "left_mid_cells_y", "right_mid_cells_y",
    ]
    missing = [name for name in required_common if getattr(self, name, None) is None]
    if missing:
        raise AttributeError(
            "Attributs MAP manquants pour reconstruire la géométrie : "
            f"{missing}.\nAs-tu bien appliqué les patchs dans populate_data "
            "(MAP_class_streamwise.py ET MAP_class_roz.py) qui stockent "
            "nodes_depth_raw / left_mid_cells_x / etc. en tant que self.xxx ?"
        )

    left_distance = self.left_distance
    left_coef = self.left_coef
    right_coef = self.right_coef
    borders_ens = self.borders_ens
    nodes_depth_raw = self.nodes_depth_raw

    left_streamwise_velocity = _vel("left_streamwise_velocity", "left_primary_velocity")
    extrap_streamwise_velocity = _vel("extrap_streamwise_velocity", "extrap_primary_velocity")
    right_streamwise_velocity = _vel("right_streamwise_velocity", "right_primary_velocity")
    left_transverse_velocity = _vel("left_transverse_velocity", "left_secondary_velocity")
    extrap_transverse_velocity = _vel("extrap_transverse_velocity", "extrap_secondary_velocity")
    right_transverse_velocity = _vel("right_transverse_velocity", "right_secondary_velocity")

    left_vertical_velocity = self.left_vertical_velocity
    extrap_vertical_velocity = self.extrap_vertical_velocity
    right_vertical_velocity = self.right_vertical_velocity

    if any(v is None for v in [
        left_streamwise_velocity, extrap_streamwise_velocity, right_streamwise_velocity,
        left_transverse_velocity, extrap_transverse_velocity, right_transverse_velocity,
        left_vertical_velocity, extrap_vertical_velocity, right_vertical_velocity,
    ]):
        raise AttributeError(
            f"Les vitesses ne sont pas encore calculées sur cet objet {type(self).__name__} "
            "(populate_data() a-t-elle bien été appelée et a-t-elle réussi ?)."
        )

    plot_data = np.c_[left_streamwise_velocity, extrap_streamwise_velocity, right_streamwise_velocity]
    distance = left_distance + (borders_ens[1:] + borders_ens[:-1]) / 2
    x_axis = np.copy(distance)
    vertical_nodes = nodes_depth_raw

    extrap_active = bool(getattr(self, "extrap_option", True))

    depths_plt = np.copy(self.depths)
    if extrap_active and left_coef == 0.3535:
        depths_plt = np.insert(depths_plt, 0, [0, self.depths[0]])
        x_axis = np.insert(x_axis, 0, self.left_borders[[0, -1]])
    elif extrap_active and left_coef == 0.91:
        depths_plt = np.insert(depths_plt, 0, [0, self.depths[0], self.depths[0]])
        x_axis = np.insert(x_axis, 0, self.left_borders[[0, 0, -1]])
    else:
        depths_plt = np.insert(depths_plt, 0, [self.depths[0]])
        x_axis = np.insert(x_axis, 0, 0)

    if extrap_active and right_coef == 0.3535:
        depths_plt = np.append(depths_plt, [self.depths[-1], 0])
        x_axis = np.append(x_axis, left_distance + borders_ens[-1] + self.right_borders[[0, -1]])
    elif extrap_active and right_coef == 0.91:
        depths_plt = np.append(depths_plt, [self.depths[-1], self.depths[-1], 0])
        x_axis = np.append(x_axis, left_distance + borders_ens[-1] + self.right_borders[[0, -1, -1]])
    else:
        depths_plt = np.append(depths_plt, self.depths[-1])
        x_axis = np.append(x_axis, distance[-1] + (distance[-1] - distance[-2]) / 2)

    v = np.c_[left_transverse_velocity, np.c_[extrap_transverse_velocity, right_transverse_velocity]]
    w = np.c_[left_vertical_velocity, np.c_[extrap_vertical_velocity, right_vertical_velocity]]

    mid_dist = np.append(
        np.insert(borders_ens[1:] + left_distance, 0, self.left_borders),
        self.right_borders[1:] + borders_ens[-1] + left_distance,
    )

    x_plt = np.tile(np.nan, (2 * plot_data.shape[0], 2 * plot_data.shape[1]))
    x_pand = np.array([val for val in mid_dist for _ in (0, 1)][1:-1])
    for n in range(len(x_pand)):
        x_plt[:, n] = x_pand[n]

    cell_plt = np.tile(np.nan, (2 * plot_data.shape[0], 2 * plot_data.shape[1]))
    cell_pand = np.array([val for val in vertical_nodes for _ in (0, 1)][1:-1])
    for p in range(cell_pand.shape[0]):
        cell_plt[p, :] = cell_pand[p]

    speed_xpand = np.tile(np.nan, (plot_data.shape[0], 2 * plot_data.shape[1]))
    for j in range(plot_data.shape[1]):
        speed_xpand[:, 2 * j] = plot_data[:, j]
        speed_xpand[:, 2 * j + 1] = plot_data[:, j]
    speed_plt = np.repeat(speed_xpand, 2, axis=0)

    # Construction des polygones
    list_poly_vertices = []
    vect_vel = []
    for i in range(int(cell_plt.shape[0] / 2)):
        for j in range(int(cell_plt.shape[1] / 2)):
            list_poly_vertices.append([
                [x_plt[2 * i, 2 * j],     cell_plt[2 * i, 2 * j]],
                [x_plt[2 * i, 2 * j + 1], cell_plt[2 * i, 2 * j]],
                [x_plt[2 * i, 2 * j + 1], cell_plt[2 * i + 1, 2 * j]],
                [x_plt[2 * i, 2 * j],     cell_plt[2 * i + 1, 2 * j]],
            ])
            vect_vel.append(speed_plt[2 * i, 2 * j])
    vect_vel = np.asarray(vect_vel, dtype=float)

    middle_poly = np.array([np.mean(poly, axis=0) for poly in list_poly_vertices])
    quiver_x = middle_poly[:, 0]
    quiver_z = middle_poly[:, 1]

    quiver_v = v.flatten()
    quiver_w = w.flatten()

    quiver_scale = np.nanmax([0.05, np.nanquantile(np.sqrt(v ** 2 + w ** 2), 0.95)])
    quiver_scale_val = round_it(quiver_scale * 5 / np.nanmax(depths_plt), 2)

    # Conversion profondeur/élévation 
    list_poly_vertices_elev = [
        [[pt[0], -pt[1]] for pt in poly] for poly in list_poly_vertices
    ]
    quiver_z_elev = -quiver_z
    quiver_w_elev = -quiver_w
    elevation_bed = -np.asarray(depths_plt, dtype=float)

    return {
        "list_poly_vertices": list_poly_vertices_elev,
        "vect_vel": vect_vel,
        "quiver_x": quiver_x,
        "quiver_z": quiver_z_elev,
        "quiver_v": quiver_v,
        "quiver_w": quiver_w_elev,
        "x_axis": x_axis,
        "elevation_bed": elevation_bed,
        "quiver_scale": quiver_scale_val,
    }


def map_profile_has_result(profile):
    """
    Vérifie qu'un profil MAP (streamwise OU roz) a bien été calculé, sans
    dépendre du nom d'attribut spécifique à la classe (streamwise_velocity
    pour MAP_streamwise, primary_velocity pour MAP_roz).
    """
    if profile is None:
        return False
    return (
        getattr(profile, "streamwise_velocity", None) is not None
        or getattr(profile, "primary_velocity", None) is not None
    )

# Sortie propre comparable avec Vermeulen
def plot_map_streamwise(average_profile, ax=None, clim=None, cmap="jet",
                        title="Profil des vitesses - Méthode classique (MAP)", x_offset=0.0, x_direction=1.0):
    """
    Parameters
        average_profile: MAP_streamwise ou MAP_roz
            Objet déjà rempli via populate_data(..., plot=False).
        ax: matplotlib.axes.Axes, optionnel
            Axe existant (pour composer une figure côte à côte). None = nouvelle figure.
        clim: tuple(float, float) ou None
            Bornes fixes de la colorbar (m/s). None = auto à partir des données.
        cmap: str
            Colormap (utilise la même que le Vermeulen pour comparer, ex 'viridis').
    """
    geo = _extract_map_geometry(average_profile)

    geo["x_axis"] = x_offset + x_direction * geo["x_axis"]
    geo["quiver_x"] = x_offset + x_direction * geo["quiver_x"]
    geo["list_poly_vertices"] = [
        [[x_offset + x_direction * pt[0], pt[1]] for pt in poly]
        for poly in geo["list_poly_vertices"]
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # Cellules colorées par vitesse streamwise
    coll = PolyCollection(geo["list_poly_vertices"], cmap=cmap,
                          edgecolors="#444444", linewidths=0.2, zorder=2)
    coll.set_array(geo["vect_vel"])
    if clim is not None:
        coll.set_clim(*clim)
    ax.add_collection(coll)
    cb = fig.colorbar(coll, ax=ax)
    cb.set_label("Vitesse streamwise / Méthode classique (m/s)")

    # Flèches de vitesses secondaires (transversale + verticale)
    
    ax.quiver(
        geo["quiver_x"], geo["quiver_z"],
        geo["quiver_v"], geo["quiver_w"],
        color="k", angles="xy", scale_units="xy", scale=1.0, zorder=5,
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

    # Lit et niveau d'eau
    ax.plot(geo["x_axis"], geo["elevation_bed"], color="k", linewidth=1.5,
           label="Lit (MAP)", zorder=4)
    ax.axhline(y=0, color="b", linewidth=2.0, label="Niveau d'eau", zorder=6)

    ax.autoscale()
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Élévation (m)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    return ax, geo

def plot_comparison(mesh, vel_sn_vermeulen, vmadcp_obj, map_profile, clim=None, cmap="jet",
                    method_label="streamwise", x_alignment=None):
    """
    Parameters :
        mesh: SigmaZetaMesh
            Maillage Vermeulen déjà résolu.
        vel_sn_vermeulen: np.ndarray (ncells, 3)
            vel_sn[0] du Vermeulen (repère section).
        vmadcp_obj: VMADCP
            Pour l'orientation RG/RD du sous-graphe Vermeulen.
        map_profile: MAP_streamwise (ou MAP_roz)
            Résultat déjà calculé (populate_data(..., plot=False)).
        clim: tuple(float, float) ou None
            Bornes de couleur communes. None = calculées à partir du max des deux
            jeux de données.
        cmap: str
            Colormap partagée (par défaut 'jet').
        method_label: str
            Étiquette de la méthode (ex 'streamwise' ou 'roz'), pour les titres.
        x_alignment: tuple(float, float) ou None
            (x_offset, x_direction) déjà calculé par un appel précédent de
            plot_comparison(). 
    """
    print(
        f"\nDEBUG plot_comparison({method_label}) : "
        f"id(map_profile)={id(map_profile)}, "
        f"map_profile.total_discharge={getattr(map_profile, 'total_discharge', None)}"
    )

    # x_offset, x_direction = _compute_map_x_alignment(mesh, vmadcp_obj, map_profile)

    if x_alignment is not None:
        x_offset, x_direction = x_alignment
        print(f"DEBUG plot_comparison({method_label}) : alignement RÉUTILISÉ "
             f"(calculé par un appel précédent) : offset={x_offset:.2f} m, "
             f"direction={x_direction:+.0f}")
    else:
        x_offset, x_direction = _compute_map_x_alignment(mesh, vmadcp_obj, map_profile)

    fig, (ax_verm, ax_map) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    # Visualisation nouvelle méthode à gauche

    ax_verm, clim_used = plot_velocity_cross_section(
        mesh, vel_sn_vermeulen, vmadcp=vmadcp_obj,
        component="streamwise", clim=clim, ax=ax_verm,
    )
    ax_verm.set_title(f"Vermeulen - vitesse {method_label}")

    clim_map = clim if clim is not None else clim_used
    print(f"DEBUG comparaison MAP/Vermeulen : échelle de couleur commune (calée sur Vermeulen) = {clim_map}")

    # Visualisation méthode classique à droite

    ax_map, geo_map = plot_map_streamwise(
        map_profile, ax=ax_map, clim=clim_map, cmap=cmap,
        title=f"MAP classique - vitesse {method_label}",
        x_offset=x_offset, x_direction=x_direction,
    )

    n_min_verm, n_max_verm = float(np.min(mesh.nb_all)), float(np.max(mesh.nb_all))

    x_map_arr = np.asarray(geo_map["x_axis"], dtype=float)
    x_map_valid = x_map_arr[np.isfinite(x_map_arr)]
    if x_map_valid.size > 0:
        n_min_map, n_max_map = float(np.min(x_map_valid)), float(np.max(x_map_valid))
    else:
        n_min_map, n_max_map = n_min_verm, n_max_verm

    n_min = min(n_min_verm, n_min_map)
    n_max = max(n_max_verm, n_max_map)
    margin = 0.05 * max(n_max - n_min, 1e-6)

    # Orientation RD/RG
    start_is_left = _rg_left_orientation(mesh.xs, vmadcp_obj)

    if start_is_left:
        ax_verm.set_xlim(left=n_min - margin, right=n_max + margin)
        ax_map.set_xlim(left=n_min - margin, right=n_max + margin)
    else:
        ax_verm.set_xlim(left=n_max + margin, right=n_min - margin)
        ax_map.set_xlim(left=n_max + margin, right=n_min - margin)

    print(
        f"\nDEBUG bornes horizontales communes (union) : "
        f"n_min={n_min:.2f} m (Vermeulen={n_min_verm:.2f}, MAP={n_min_map:.2f}), "
        f"n_max={n_max:.2f} m (Vermeulen={n_max_verm:.2f}, MAP={n_max_map:.2f})"
    )

    print(
        f"\nDEBUG orientation RG/RD (commune aux deux panneaux) : "
        f"start_is_left={start_is_left} -> RG {'à gauche' if start_is_left else 'à droite'}"
    )

    fig.suptitle(
        f"Comparaison Vermeulen / Méthode classique (MAP) - vitesse {method_label}\n"
        f"[axe n : {n_min:.1f} à {n_max:.1f} m]", fontsize=13
    )
    fig.tight_layout()

    return fig, (ax_verm, ax_map), (x_offset, x_direction)


def run_map_comparison(
    meas,
    navigation_reference_user='GPS',
    nbr_cell_hor=80,
    nbr_cell_vert=30,
    node_horizontal_user=None,
    node_vertical_user=None,
    edge_constant=True,
    extrap_option=False,
    interp_option=True,
    track_section=True,
    methods=('streamwise',),
):
    """
    Returns
        dict : {'streamwise': MAP_streamwise|None, 'roz': MAP_roz|None}
    """
    meas = _align_navigation_reference(meas, navigation_reference_user)
    results = {'streamwise': None, 'roz': None}

    if 'streamwise' in methods:
        print("\nCalcul de la méthode classique MAP (streamwise)...")
        try:
            average_profile = MAP_streamwise()
            average_profile.populate_data(
                meas, nb_max=1.0, 
                nbr_cell_hor=nbr_cell_hor, nbr_cell_vert=nbr_cell_vert,
                node_horizontal_user=node_horizontal_user, node_vertical_user=node_vertical_user,
                edge_constant=edge_constant, extrap_option=extrap_option,
                interp_option=interp_option, track_section=track_section,
                plot=False, name_meas='', path_results='',
            )
            results['streamwise'] = average_profile
            if average_profile.streamwise_velocity is not None:
                sv = np.asarray(average_profile.streamwise_velocity, dtype=float)
                print(
                    f"DEBUG MAP streamwise : min={np.nanmin(sv):.3f} m/s, "
                    f"max={np.nanmax(sv):.3f} m/s, mean={np.nanmean(sv):.3f} m/s"
                )
        except Exception as e:
            print(f"Erreur lors du calcul MAP streamwise : {e}")
            traceback.print_exc()

    if 'roz' in methods:
        print("\nCalcul de la méthode classique MAP (Rozovskii)...")
        try:
            average_profile_roz = MAP_roz()
            average_profile_roz.populate_data(
                meas, nb_max=1.0,
                nbr_cell_hor=nbr_cell_hor, nbr_cell_vert=nbr_cell_vert,
                node_horizontal_user=node_horizontal_user, node_vertical_user=node_vertical_user,
                edge_constant=edge_constant, extrap_option=extrap_option,
                interp_option=interp_option, track_section=track_section,
                plot=False, name_meas='', path_results='',
            )
            results['roz'] = average_profile_roz
        except Exception as e:
            print(f"Erreur lors du calcul MAP Rozovskii : {e}")
            traceback.print_exc()

    return results


## Travail sur le nombre d'informations par cellules


def _extract_map_count_geometry(average_profile):
    """
    Returns
        dict avec les clés :
            list_poly_vertices : polygones [[x, z_elevation], ...] (4 sommets)
            vect_count          : nombre de mesures brutes par polygone (même ordre)
            x_bed, elevation_bed : ligne de fond simplifiée (repère de niveau)
    """
    self = average_profile

    required = ["borders_ens", "depth_cells_border", "info_cell", "depths"]
    missing = [name for name in required if getattr(self, name, None) is None]
    if missing:
        raise AttributeError(f"Attributs MAP manquants pour le comptage : {missing}")

    borders_ens = np.asarray(self.borders_ens, dtype=float)          # (n_node+1,)
    depth_cells_border = np.asarray(self.depth_cells_border, dtype=float)  # (n_depth+1, n_node)
    info_cell = np.asarray(self.info_cell, dtype=float)              # (n_depth, n_node)

    n_depth = depth_cells_border.shape[0] - 1
    n_node = len(borders_ens) - 1

    list_poly_vertices = []
    vect_count = []

    for i in range(n_depth):
        for j in range(n_node):
            x0, x1 = borders_ens[j], borders_ens[j + 1]
            z0, z1 = depth_cells_border[i, j], depth_cells_border[i + 1, j]
            if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(z0) and np.isfinite(z1)):
                continue  
            list_poly_vertices.append([[x0, -z0], [x1, -z0], [x1, -z1], [x0, -z1]])
            vect_count.append(info_cell[i, j])

    x_bed = (borders_ens[1:] + borders_ens[:-1]) / 2
    elevation_bed = -np.asarray(self.depths, dtype=float)

    return {
        "list_poly_vertices": list_poly_vertices,
        "vect_count": np.asarray(vect_count, dtype=float),
        "x_bed": x_bed,
        "elevation_bed": elevation_bed,
    }


def plot_map_info_count(average_profile, ax=None, clim=None, cmap="viridis",
                        title="Nombre de mesures brutes / cellule - MAP classique"):

    geo = _extract_map_count_geometry(average_profile)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    coll = PolyCollection(geo["list_poly_vertices"], cmap=cmap,
                          edgecolors="#444444", linewidths=0.2, zorder=2)


    ax.add_collection(coll)
    _apply_low_count_coloring(coll, geo["vect_count"], cmap_name="Blues", clim=clim, ax=ax, fig=fig)

    ax.plot(geo["x_bed"], geo["elevation_bed"], color="k", linewidth=1.5,
           label="Lit (MAP)", zorder=4)
    ax.axhline(y=0, color="b", linewidth=2.0, label="Niveau d'eau", zorder=6)

    ax.autoscale()
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Élévation (m)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    fig.text(
        0.5, 0.02,
        "\u25A0: moins de 3 valeurs brutes"
        "(vitesse 3D non calculable)",
        ha="center", va="bottom", fontsize=9, color="red",
    )
    fig.subplots_adjust(bottom=0.12)

    return ax, geo


def plot_mesh_info_count(mesh, nb_vel_cell, ax=None, clim=None, cmap="viridis",
                         vmadcp=None, title="Nombre de mesures brutes / cellule - Vermeulen"):
    """
    Parameters
        mesh : SigmaZetaMesh
        nb_vel_cell : array-like, longueur mesh.ncells
            nb_vel[0] (3e retour de vsolver.get_velocity) : nombre de mesures
            brutes retenues par cellule après filtrage des outliers.
    """
    from Classes_vermeulen.plot_mesh_bathy import _rg_left_orientation

    nb_vel_cell = np.asarray(nb_vel_cell, dtype=float)
    if nb_vel_cell.shape[0] != mesh.ncells:
        raise ValueError("nb_vel_cell doit avoir mesh.ncells éléments")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    if mesh.nb_all.size > 0:
        ax.plot(mesh.nb_all, mesh.zb_all, "k-", linewidth=2.0, label="Lit (maillage)", zorder=4)

    if mesh.ncells > 0:
        polygons = [np.c_[mesh.n_patch[:, cc], mesh.z_patch[:, cc]] for cc in range(mesh.ncells)]
        coll = PolyCollection(polygons, cmap=cmap, edgecolors="#444444", linewidths=0.3, zorder=2)

        # coll.set_array(nb_vel_cell)
        # if clim is not None:
        #     coll.set_clim(*clim)
        
        # ax.add_collection(coll)
        # cb = fig.colorbar(coll, ax=ax)
        # cb.set_label("Nombre de mesures brutes / cellule")

        ## 27/08
        ax.add_collection(coll)
        _apply_low_count_coloring(coll, nb_vel_cell, cmap_name="Blues", clim=clim, ax=ax, fig=fig)
        ##

    wl = float(mesh.water_level)
    if mesh.nw.size > 0:
        nw_flat = mesh.nw.reshape(-1)
        ax.plot(nw_flat, np.full_like(nw_flat, wl), "b-", linewidth=2.5, label="Niveau d'eau", zorder=6)
    else:
        ax.axhline(y=wl, color="b", linewidth=2.5, label="Niveau d'eau", zorder=6)

    ax.set_xlabel("Distance le long de la section (n) [m]")
    ax.set_ylabel("Élévation [m]")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)

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

    ax.legend(loc="lower right")
    return ax


def plot_comparison_info_count(mesh, nb_vel_cell, vmadcp_obj, map_profile, clim=None, cmap="Blues"):

    if clim is None:
        nb_vel_arr = np.asarray(nb_vel_cell, dtype=float)
        vermeulen_max = np.nanmax(nb_vel_arr) if nb_vel_arr.size else 0.0

        geo_probe = _extract_map_count_geometry(map_profile)
        map_max = np.nanmax(geo_probe["vect_count"]) if geo_probe["vect_count"].size else 0.0

        clim = (0.0, max(vermeulen_max, map_max, 1.0))
        print(f"DEBUG comparaison comptage MAP/Vermeulen : échelle commune = {clim}")

    fig, (ax_verm, ax_map) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    plot_mesh_info_count(mesh, nb_vel_cell, ax=ax_verm, clim=clim, cmap=cmap, vmadcp=vmadcp_obj,
                         title="Vermeulen - nb. mesures brutes / cellule")
    plot_map_info_count(map_profile, ax=ax_map, clim=clim, cmap=cmap,
                        title="MAP classique - nb. mesures brutes / cellule")

    fig.suptitle("Comparaison Vermeulen / Méthode classique (MAP) - Nombre de mesures brutes par cellule",
                 fontsize=14)
    fig.tight_layout()
    return fig, (ax_verm, ax_map)



def _compute_map_x_alignment(mesh, vmadcp, map_profile, n_ensembles_edge=5,
                             search_margin_m=None, n_search_points=201,
                             min_overlap_fraction=0.6):
    """
    Transformation de l'axe local MAP vers le repère n du Vermeulen.

    Returns :
        offset: float, direction_sign: float (+1.0 ou -1.0)
    """

    if search_margin_m is None:
        span_verm_guess = float(np.max(mesh.nb_all) - np.min(mesh.nb_all))
        search_margin_m = max(5.0, 0.5 * span_verm_guess)
        print(f"DEBUG alignement MAP/Vermeulen : marge de recherche adaptative = {search_margin_m:.1f} m "
              f"(section ~{span_verm_guess:.0f} m)")

    hp = np.asarray(getattr(vmadcp, "horizontal_position", np.empty((2, 0))), dtype=float)
    offset_guess = 0.0
    if hp.ndim == 2 and hp.shape[1] > 0:
        x, y = hp[0, :], hp[1, :]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if x.size > 0:
            edge = min(n_ensembles_edge, max(1, x.size // 2))
            _, n_start = mesh.xs.xy2sn(np.array([np.mean(x[:edge])]), np.array([np.mean(y[:edge])]))
            offset_guess = float(n_start[0])
    print(f"DEBUG alignement MAP/Vermeulen : estimation GPS initiale = {offset_guess:.2f} m")

    # Profils de fond
    n_verm = np.asarray(mesh.nb_all, dtype=float)
    z_verm = np.asarray(mesh.zb_all, dtype=float)
    order = np.argsort(n_verm)
    n_verm, z_verm = n_verm[order], z_verm[order]
    n_verm_min, n_verm_max = float(n_verm.min()), float(n_verm.max())
    span_verm = n_verm_max - n_verm_min

    geo_raw = _extract_map_geometry(map_profile)
    x_map = np.asarray(geo_raw["x_axis"], dtype=float)
    z_map = np.asarray(geo_raw["elevation_bed"], dtype=float)
    valid_map = np.isfinite(x_map) & np.isfinite(z_map)
    x_map, z_map = x_map[valid_map], z_map[valid_map]
    x_map_min, x_map_max = float(x_map.min()), float(x_map.max())
    span_map = x_map_max - x_map_min

    def _bed_mismatch_and_overlap(offset, direction_sign):
        """
        Retourne (score, overlap_fraction) pour un candidat (offset, sens).
        overlap_fraction = fraction du plus petit des deux profils qui se
        recouvre réellement une fois MAP transformé -- c'est ce garde-fou
        qui manquait dans la version précédente.
        """
        n_map_test = offset + direction_sign * x_map
        n_map_test_min, n_map_test_max = float(n_map_test.min()), float(n_map_test.max())

        overlap_lo = max(n_verm_min, n_map_test_min)
        overlap_hi = min(n_verm_max, n_map_test_max)
        overlap_len = max(0.0, overlap_hi - overlap_lo)
        overlap_fraction = overlap_len / max(min(span_verm, span_map), 1e-6)

        if overlap_fraction < min_overlap_fraction:
            return np.inf, overlap_fraction

        z_verm_interp = np.interp(n_map_test, n_verm, z_verm, left=np.nan, right=np.nan)
        valid = np.isfinite(z_verm_interp) & np.isfinite(z_map)
        if np.sum(valid) < 10:
            return np.inf, overlap_fraction

        score = float(np.mean((z_verm_interp[valid] - z_map[valid]) ** 2))
        return score, overlap_fraction

    best_score = np.inf
    best_offset, best_direction, best_overlap = offset_guess, 1.0, 0.0

    for direction_sign in (1.0, -1.0):
        offsets_to_test = np.linspace(
            offset_guess - search_margin_m, offset_guess + search_margin_m, n_search_points
        )
        local_best_score, local_best_offset, local_best_overlap = np.inf, offset_guess, 0.0
        for o in offsets_to_test:
            score, overlap_fraction = _bed_mismatch_and_overlap(o, direction_sign)
            if score < local_best_score:
                local_best_score = score
                local_best_offset = o
                local_best_overlap = overlap_fraction

        print(
            f"DEBUG alignement MAP/Vermeulen : direction_sign={direction_sign:+.0f} -> "
            f"meilleur offset={local_best_offset:.2f} m, erreur={local_best_score:.4f} m2, "
            f"recouvrement={local_best_overlap*100:.0f}%"
        )

        if local_best_score < best_score:
            best_score = local_best_score
            best_offset = local_best_offset
            best_direction = direction_sign
            best_overlap = local_best_overlap

    if not np.isfinite(best_score):
        print(
            "ATTENTION : aucun candidat avec un recouvrement suffisant n'a été "
            f"trouvé dans la plage de recherche (±{search_margin_m} m autour de "
            f"l'estimation GPS). Repli sur l'estimation GPS brute (offset="
            f"{offset_guess:.2f} m, direction_sign=+1). Essaie d'augmenter "
            "search_margin_m si le vrai décalage est plus grand que ça."
        )
        return offset_guess, 1.0

    print(
        f"DEBUG alignement MAP/Vermeulen RETENU : offset={best_offset:.2f} m, "
        f"direction_sign={best_direction:+.0f}, erreur quadratique={best_score:.4f} m2, "
        f"recouvrement={best_overlap*100:.0f}%"
    )

    left_edge_map = getattr(map_profile, "left_distance", None)
    right_edge_map = getattr(map_profile, "right_distance", None)
    print(
        f"\nDEBUG bords MAP : left_distance={left_edge_map}, right_distance={right_edge_map} "
        f"(distances de bord mesurées sur le terrain, indépendantes de la bathymétrie ADCP)"
    )


    return best_offset, best_direction


# def _build_facecolors_with_low_count_highlight(values, cmap_name, clim, low_count_threshold=3,
#                                                 low_count_color="red", nan_color="white"):

#     values = np.asarray(values, dtype=float)
#     cmap = cm.get_cmap(cmap_name)
#     norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])

#     colors = np.zeros((values.size, 4))
#     is_nan = ~np.isfinite(values)
#     is_low = np.isfinite(values) & (values < low_count_threshold)
#     is_normal = np.isfinite(values) & (values >= low_count_threshold)

#     colors[is_normal] = cmap(norm(values[is_normal]))
#     colors[is_low] = mcolors.to_rgba(low_count_color)
#     colors[is_nan] = mcolors.to_rgba(nan_color)

#     n_low = int(np.sum(is_low))
#     n_nan = int(np.sum(is_nan))
#     print(
#         f"DEBUG cellules à faible comptage : {n_low} cellule(s) avec < "
#         f"{low_count_threshold} mesures (colorées en {low_count_color}), "
#         f"{n_nan} cellule(s) sans aucune mesure (colorées en {nan_color})."
#     )

#     return colors

def _apply_low_count_coloring(coll, values, cmap_name, clim, ax, fig,
                              low_count_threshold=3, low_count_color="red",
                              nan_color="white", colorbar_label="Nombre de mesures brutes / cellule"):

    values = np.asarray(values, dtype=float)
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])

    colors = np.zeros((values.size, 4))
    is_nan = ~np.isfinite(values)
    is_low = np.isfinite(values) & (values < low_count_threshold)  
    is_normal = np.isfinite(values) & (values >= low_count_threshold)

    colors[is_normal] = cmap(norm(values[is_normal]))
    colors[is_low] = mcolors.to_rgba(low_count_color)
    colors[is_nan] = mcolors.to_rgba(nan_color)
    coll.set_facecolor(colors)

    n_low, n_nan = int(np.sum(is_low)), int(np.sum(is_nan))
    print(
        f"DEBUG cellules à faible comptage : {n_low} cellule(s) avec < {low_count_threshold} "
        f"mesures (rouge), {n_nan} cellule(s) sans aucune mesure (blanc)."
    )

    sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(colorbar_label)

    # legend_patch = Patch(facecolor=low_count_color, edgecolor="#444444",
    #                      label=f"< {low_count_threshold} mesures\n(vitesse 3D non calculable)")
    # handles, labels = ax.get_legend_handles_labels()
    # handles.append(legend_patch)
    # labels.append(legend_patch.get_label())
    # ax.legend(handles=handles, labels=labels, loc="lower right", fontsize=8)