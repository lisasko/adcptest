# -*- coding: utf-8 -*-
"""
Created on Tue April 21 2026
@author: lisa
"""

### Traduction Python de la logique Matlab de Location Based Velocity Solver 
"""
Portage Python de LocationBasedVelocitySolver.m / VelocitySolver.m
(B. Vermeulen et al., adcptools)

Principe :
    Contrairement à MAP_streamwise (méthode temporelle), cette classe
    utilise les vitesses brutes par faisceau (raw_vel_mps, coord_sys='Beam')
    ainsi que les matrices de transformation heading-pitch-roll pour chaque
    ping.  Pour chaque cellule du maillage, toutes les mesures radiales
    issues de tous les transects qui se trouvent physiquement dans cette
    cellule sont rassemblées et un vecteur vitesse [u, v, w] est estimé par
    moindres carrés pondérés.
    (lscov Matlab → np.linalg.lstsq pondéré)
 
"""

"""
Qu'est-ce qui change par raport à MAP_streamwise ?
-> Pour chaque cellule du maillage, au lieu de faire np.nanmean(vitesses_déjà_en_Earth), on fait :

- Récupère les raw_vel_mps (faisceau 1..4) bruts
- Construit la matrice HPR (heading-pitch-roll) ping par ping via _build_hpr_matrix — même formule que WaterData.change_coord_sys dans QRevInt
- Assemble le système A·[u,v,w] = b (une ligne par mesure radiale valide)
- Applique le filtre sigma, le filtre directionnel et la pondération spatiale (paramètres f_vitesse_z, f_direction_fixe, f_direction_pond, pond_vitesses)
- Résout par np.linalg.lstsq pondéré — équivalent de lscov Matlab
- Stocke aussi nb_vel, r2, r_sig par cellule

"""
## Entrées disponibles dans QRevInt : 

"""
# Vitesses radiales brutes faisceau (si orig_coord_sys == 'Beam')
b_raw = transect.w_vel.raw_vel_mps  # (4, n_cells, n_ens)

# Matrices HPR par ping — construites comme dans change_coord_sys
h = getattr(sensors.heading_deg, sensors.heading_deg.selected).data  # (n_ens,)
p = getattr(sensors.pitch_deg,   sensors.pitch_deg.selected).data
r = getattr(sensors.roll_deg,    sensors.roll_deg.selected).data
t_matrix = transect.adcp.t_matrix.matrix  # matrice instrument (4×4)

# Position de chaque cellule
cell_depth = transect.depths.bt_depths.depth_cell_depth_m  # (n_cells, n_ens)

"""

## Algorithme : 

"""
Pour chaque cellule du maillage (i_hor, i_vert) :
  1. Trouver tous les pings k dont la cellule physique tombe dans cette case
     (même logique que compute_nodes_velocity existant)
  
  2. Pour chaque ping k retenu :
     - Récupérer les 4 vitesses faisceau brutes : b_k = raw_vel_mps[:, cell_idx, k]
     - Construire hpr_matrix_k à partir de h[k], p[k], r[k]
     - Construire xform_k = hpr_matrix_k @ t_matrix  (Beam → Earth)
     - Chaque faisceau j donne une équation : b_k[j] = xform_k[j,:3] · [u, v, w]
  
  3. Assembler le système pour toutes les mesures de la cellule :
     A (N×3) où chaque ligne = xform_k[j, :3]
     b (N,)  où chaque élément = b_k[j]
  
  4. Filtre sigma : retirer les b_k[j] avec |b_k[j] - mean(b)| > f_vitesse_z * std(b)
  
  5. Filtre directionnel : mean(A, axis=0) normalisé → composante Z > seuil adaptatif
  
  6. Pondération : s_k = distance du ping k à la section transversale
     w_k = 1 - (|s_k| / max|s|)^pond_vitesses
     → Multiplier chaque ligne de A et b par sqrt(w_k)
  
  7. Résolution : [u, v, w] = lstsq(A_pond, b_pond)
     → Aussi : nb_vel = N, r2 = résidu², r_sig = std(résidu)

"""

# ============================================================
# External imports
# ============================================================
import copy
import math
import os
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from math import isnan


# ============================================================
# Internal imports
# ============================================================
from MiscLibs.common_functions import cart2pol, pol2cart, nan_greater
from UI.MplCanvas import MplCanvas
from common_functions import show_figure, interpolation, round_it
from Classes_vermeulen import (
    BathymetryScatteredPoints,
    VMADCP,
    EnsembleFilter,
    SigmaZetaMeshFromVMADCP,
    XSection,
)


# ============================================================
# Classe principale
# ============================================================
 
class MAP_vermeulen:
    """Multitransect Averaged Profile (MAP) — méthode dite Vermeulen, adaptation Location-Based).
 
    Résout le vecteur vitesse [u, v, w] par moindres carrés pondérés à
    partir des vitesses radiales brutes par faisceau (coord_sys='Beam'),
    en tenant compte du roulis/tangage réel de l'ADCP à chaque ping.
 
    Attributes
    ----------
    streamwise_velocity : np.ndarray (n_vert × n_hor)
        Vitesse dans le sens de l'écoulement (m/s)
    transverse_velocity : np.ndarray (n_vert × n_hor)
        Vitesse transversale (m/s)
    vertical_velocity : np.ndarray (n_vert × n_hor)
        Vitesse verticale (m/s)
    depths : np.ndarray (n_hor,)
        Profondeur moyenne par verticale (m)
    depth_cells_border : np.ndarray
        Bornes verticales des cellules
    borders_ens : np.ndarray
        Bornes horizontales des verticales
    nb_vel : np.ndarray (n_vert × n_hor)
        Nombre de mesures radiales utilisées par cellule
    r2 : np.ndarray (n_vert × n_hor)
        Erreur quadratique moyenne par cellule (résidu lstsq)
    r_sig : np.ndarray (n_vert × n_hor)
        Écart-type des résidus par cellule
    vmin, vmax : float
        Bornes de l'échelle de couleur
    quiver_scale : float
        Échelle des flèches de vitesses secondaires
    """
 
    def __init__(self):
        """Initialize class and instances variables."""
 
        # Vitesses MAP (même nommage que MAP_streamwise)
        self.streamwise_velocity = None  # MAPstreamwise velocity of each middle cell without extrapolation
        self.transverse_velocity = None  # MAP transverse velocity of each middle cell without extrapolation
        self.vertical_velocity = None  # MAPstreamwise velocity of each middle cell without extrapolation
        self.depths = None  # MAP depths
        self.extrap_streamwise_velocity = None  # MAPstreamwise velocity with extrpolation on bottom/top part
        self.extrap_transverse_velocity = None  # MAP transverse velocity with extrpolation on bottom/top part
        self.extrap_vertical_velocity = None  # MAP vertical velocity with extrpolation on bottom/top part
        self.depth_cells_border = None  # Depth borders of each MAP cell, last one of each vertical equal to vertical depth
        self.depth_cells_center = None  # Depth center of each MAP cell
        self.borders_ens = None  # Borders of each MAP vertical
 
        # Bords / rives
        self.left_distance = None  # MAP edge distance from edge computation
        self.left_borders = None  # Borders of each MAP vertical from edge computation
        self.left_coef = None  # Shape coefficient of MAP edge
        self.left_streamwise_velocity = None  # MAPstreamwise velocity for edge cells
        self.left_transverse_velocity = None  # MAP transverse velocity for edge cells
        self.left_vertical_velocity = None  # MAP vertical velocity for edge cells
        self.left_cells_discharge = None  # MAP edge cells discharge
        self.left_discharge = None  # MAP edge total discharge
        self.right_distance = None  # MAP edge distance from edge computation
        self.right_borders = None  # Borders of each MAP vertical from edge computation
        self.right_coef = None  # Shape coefficient of MAP edge
        self.right_streamwise_velocity = None  # MAPstreamwise velocity for edge cells
        self.right_transverse_velocity = None  # MAP transverse velocity for edge cells
        self.right_vertical_velocity = None  # MAP vertical velocity for edge cells
        self.right_cells_discharge = None  # MAP edge cells discharge
        self.right_discharge = None  # MAP edge total discharge
 
        # Débit
        self.total_discharge = None  # MAP total discharge with current parameters
        self.middle_cells_discharge = None  # MAP discharge of each middle cell (with top/bottom extrapolation if selected)
        self.middle_discharge = None  # MAP middle discharge with top/bottom extrapolation (if selected)
 
        # Affichage
        self.vmin = None # Vitesse minimale de l'échelle de couleur
        self.vmax = None # Vitesse maximale de l'échelle de couleur
        self.quiver_scale = None # Echelle pour les vitesses secondaires
 
        # Attributs spécifiques à la méthode Vermeulen
        self.nb_vel = None   # nb de mesures radiales utilisées par cellule
        self.r2 = None       # erreur quadratique (résidu² moyen)
        self.r_sig = None    # écart-type des résidus
        self.orig_start_edge = None  # original start edge per selected transect


# ------------------------------------------------------------------ #
#  POINT D'ENTRÉE PRINCIPAL                                            #
# ------------------------------------------------------------------ #
 
    def populate_data(self, meas, nbr_cell_hor, nbr_cell_vert,
                      node_horizontal_user=None, node_vertical_user=None,
                      f_vitesse_z=10, f_direction_fixe=0.97,
                      f_direction_pond=0.0002, pond_vitesses=1,
                      edge_constant=True, extrap_option=False,
                      interp_option=True, track_section=True,
                      use_translated_bathy=False,
                      use_raw_bt_beam_bathy=False,
                      use_scattered_bathy_interpolation=False,
                      bathy_span=0.005,
                      use_sigma_zeta_mesh_builder=True,
                      plot_sigma_zeta_mesh=True,
                      quiver_scale_mode='classic95',
                      quiver_scale_fixed=None,
                      print_velocity_diagnostics=False,
                      plot=False, name_meas='unknown', path_results=''):
        
        """Lance le calcul MAP complet avec la méthode Vermeulen.
 
        Parameters
        ----------
        meas : Measurement
            Objet Measurement de QRevInt.
        nbr_cell_hor : int
            Nombre approximatif de cellules horizontales.
        nbr_cell_vert : int
            Nombre approximatif de cellules verticales.
        node_horizontal_user : float or None
            Taille de maille horizontale imposée (m). None = automatique.
        node_vertical_user : float or None
            Taille de maille verticale imposée (m). None = automatique.
        f_vitesse_z : float
            Seuil de rejet sigma (filtre vitesses radiales aberrantes).
            Mesures écartées si |b - mean| > f_vitesse_z × std. (défaut 10)
        f_direction_fixe : float
            Seuil fixe de la composante Z du vecteur directionnel moyen
            normalisé. Cellule rejetée si Z < seuil adaptatif. (défaut 0.97)
        f_direction_pond : float
            Décrement du seuil directionnel par mesure supplémentaire.
            (défaut 0.0002)
        pond_vitesses : float
            Exposant de pondération spatiale : w = 1-(|s|/max_s)^pond.
            1 = décroissance linéaire, >1 plus lente, <1 plus rapide.
            (défaut 1)
        edge_constant : bool
            Taille uniforme des cellules de berge. (défaut True)
        extrap_option : bool
            Active l'extrapolation haut/bas/berges. (défaut False)
        interp_option : bool
            Utilise les vitesses interpolées QRevInt. (défaut True)
        track_section : bool
            Section moyenne sur la trajectoire bateau (True) ou sur la
            direction de l'écoulement (False). (défaut True)
        use_translated_bathy : bool
            Si True, utilise la translation inter-transects (comme méthode
            classique), ce qui tend à lisser le fond. Si False, conserve la
            géométrie brute projetée pour mieux préserver les irrégularités
            du fond côté Vermeulen. (défaut False)
        use_raw_bt_beam_bathy : bool
            Si True, reconstruit la profondeur de fond par ensemble à partir
            de la médiane des faisceaux BT (depth_beams_m), au lieu de
            depth_processed_m (déjà filtré/interpolé par QRev). Cela donne
            un fond généralement plus irrégulier, plus proche de MATLAB
            Vermeulen. (défaut False)
        use_scattered_bathy_interpolation : bool
            Si True, reconstruit une bathymétrie interpolée à partir des
            points (x,y,z) de tous les transects, puis réévalue la
            profondeur sur les positions ADCP (logique proche du demo.m
            MATLAB avec BathymetryScatteredPoints). (défaut True)
        bathy_span : float
            Span Loess du modèle de bathymétrie interpolée. Les valeurs
            faibles (ex: 0.005 comme dans demo.m) préservent plus les
            irrégularités locales du fond.
        use_sigma_zeta_mesh_builder : bool
            Mode expérimental. Si True, construit les bornes de maillage via
            SigmaZetaMeshFromVMADCP (logique MATLAB) puis les projette dans
            le repère MAP utilisé par ce solveur. Ce solveur restant basé
            sur une discrétisation MAP historique, ce mode peut créer des
            zones peu résolues. Par défaut False pour préserver la densité
            des cellules résolues.
        plot_sigma_zeta_mesh : bool
            Si True et si le mesh builder est actif, sauvegarde un graphe
            dédié du maillage Sigma-Zeta utilisé.
        quiver_scale_mode : str
            Mode d'échelle des flèches secondaires pour la figure
            streamwise. 'classic95' reproduit la logique de MAP classique
            (q95 des vitesses secondaires, normalisée par profondeur).
            'mean' utilise la moyenne des normes sqrt(v^2+w^2).
        quiver_scale_fixed : float or None
            Si défini, force une échelle de quiver fixe (m/s).
        print_velocity_diagnostics : bool
            Si True, imprime des statistiques de cohérence sur les
            composantes streamwise/transverse/vertical et leur norme
            secondaire.
        plot : bool
            Génère et sauvegarde les figures. (défaut False)
        name_meas : str
            Nom de la mesure pour les figures.
        path_results : str
            Chemin de sauvegarde des figures.
        """

        # Get meas current parameters
        settings = meas.current_settings()
        navigation_reference = settings['NavRef']
        checked_transect_idx = meas.checked_transect_idx
 
        if all(deg == 0 for deg in
               meas.transects[checked_transect_idx[0]].sensors.heading_deg.internal.data):
            print("MAP_vermeulen failed : No compass available")
            return
 
        # Get main data from selected transects
        x_raw, y_raw, w_vel_x, w_vel_y, w_vel_z, \
        depth_data, q_cells, orig_start_edge, invalid_data, \
        cell_depth, left_geometry, right_geometry, dmg = \
            self.collect_data(meas, navigation_reference,
                              checked_transect_idx, interp_option)
        self.orig_start_edge = orig_start_edge

        if use_raw_bt_beam_bathy:
            depth_data = self._rebuild_depth_profiles_from_bt_beams(
                meas,
                checked_transect_idx,
            )

        # Compute coefficients of the average cross-section
        alpha, beta, direction_meas, q_ens = \
            self.compute_coef(x_raw, y_raw, w_vel_x,
                              w_vel_y, q_cells, track_section)
 
        # Project raw coordinates on average cross-section
        x_proj, y_proj, acs_distance_raw = \
            self.project_transect(alpha, beta, x_raw, y_raw)

        if use_scattered_bathy_interpolation:
            # MATLAB-like intent: build a global scattered bathymetry then
            # sample depths on the average cross-section geometry.
            depth_data = self._rebuild_depth_profiles_from_scattered_bathy(
                x_raw,
                y_raw,
                depth_data,
                x_query=x_proj,
                y_query=y_proj,
                bathy_span=bathy_span,
            )
 
        # Vermeulen: keep raw projected geometry by default to preserve
        # bottom irregularities. Translation can be re-enabled for strict
        # side-by-side geometry with classic MAP.
        if use_translated_bathy:
            acs_distance = self.translated_transects(
                acs_distance_raw, depth_data)
        else:
            acs_distance = acs_distance_raw
 
        # Define horizontal and vertical mesh
        transect = meas.transects[checked_transect_idx[0]]
        in_transect_idx = transect.in_transect_idx
        valid_depth = transect.w_vel.valid_data[0, :, in_transect_idx].T
        top_cell = (np.nanmin(
            transect.depths.bt_depths.depth_cell_depth_m[valid_depth])
                    - transect.depths.bt_depths.depth_cell_size_m[0, 0] / 2)
 
        sigma_mesh = None
        if use_sigma_zeta_mesh_builder:
            try:
                borders_ens_raw, nodes_depth_raw, sigma_mesh = \
                    self.compute_node_size_from_sigma_zeta_builder(
                        meas=meas,
                        navigation_reference=navigation_reference,
                        checked_transect_idx=checked_transect_idx,
                        acs_distance=acs_distance,
                        nbr_cell_hor=nbr_cell_hor,
                        nbr_cell_vert=nbr_cell_vert,
                        node_vertical_user=node_vertical_user,
                        bathy_span=bathy_span,
                        use_raw_bt_beam_bathy=use_raw_bt_beam_bathy,
                    )
                print("[Vermeulen] Mesh source: SigmaZetaMeshFromVMADCP")
            except Exception as exc:
                print(f"[Vermeulen] SigmaZeta mesh builder failed, fallback compute_node_size: {exc}")
                borders_ens_raw, nodes_depth_raw = \
                    self.compute_node_size(
                        cell_depth, acs_distance, depth_data,
                        node_horizontal_user, node_vertical_user,
                        nbr_cell_hor, nbr_cell_vert, top_cell)
        else:
            borders_ens_raw, nodes_depth_raw = \
                self.compute_node_size(
                    cell_depth, acs_distance, depth_data,
                    node_horizontal_user, node_vertical_user,
                    nbr_cell_hor, nbr_cell_vert, top_cell)
 
        # RÉSOLUTION VERMEULEN 
        transects_node_x_vel, transects_node_y_vel, \
        transects_node_z_vel, transects_node_depth, \
        transects_nodes, transects_node_nz, sigma_cap, \
        info_cell, r2_cell, r_sig_cell = \
            MAP_vermeulen.compute_nodes_velocity_vermeulen(
                meas, navigation_reference, checked_transect_idx,
                cell_depth, depth_data, acs_distance,
                borders_ens_raw, nodes_depth_raw, orig_start_edge,
                f_vitesse_z, f_direction_fixe,
                f_direction_pond, pond_vitesses)

        # Compute mesh mean value of selected transects
        MAP_x_vel, MAP_y_vel = self.compute_mean(
            transects_nodes, transects_node_x_vel,
            transects_node_y_vel, transects_node_z_vel,
            transects_node_depth, transects_node_nz, sigma_cap,
            borders_ens_raw,
            nodes_depth_raw, info_cell, r2_cell, r_sig_cell)
 
        # Compute streamwise and transverse velocity
        self.compute_projection(MAP_x_vel, MAP_y_vel, direction_meas)
 
        # Compute edges parameters
        if extrap_option:
            self.left_distance, self.left_coef = list(left_geometry)
            self.right_distance, self.right_coef = list(right_geometry)
        else:
            self.left_distance = 0
            self.right_distance = 0
 
        # Compute top/bottom extrapolation according QRevInt velocity exponent
        self.compute_extrap_velocity(nodes_depth_raw, settings, extrap_option)
 
        # Compute edge extrapolation
        left_dir, right_dir, left_area, right_area, \
        left_mid_x, right_mid_x, left_mid_y, right_mid_y = \
            self.compute_edges(borders_ens_raw, direction_meas,
                               settings, edge_constant)
 
        # Compute discharge
        self.compute_discharge(extrap_option, left_dir, right_dir,
                               left_area, right_area)
 
        # Plot results
        if plot:
            self.plot_profile(borders_ens_raw, nodes_depth_raw,
                              left_mid_x, right_mid_x,
                              left_mid_y, right_mid_y,
                              path_results, name_meas,
                              quiver_scale_mode=quiver_scale_mode,
                              quiver_scale_fixed=quiver_scale_fixed,
                              print_velocity_diagnostics=print_velocity_diagnostics)

            if use_sigma_zeta_mesh_builder and plot_sigma_zeta_mesh and sigma_mesh is not None:
                self.plot_sigma_zeta_mesh(sigma_mesh, path_results, name_meas,
                                          self.orig_start_edge)

    @staticmethod
    def _rebuild_depth_profiles_from_bt_beams(meas, checked_transect_idx):
        """Build depth profiles from raw BT beams (median over valid beams)."""
        depth_data = []

        for id_transect in checked_transect_idx:
            transect = meas.transects[id_transect]
            bt = transect.depths.bt_depths

            depth_beams = np.asarray(bt.depth_beams_m, dtype=float)
            valid_beams = np.asarray(bt.valid_beams, dtype=bool)
            if depth_beams.ndim == 1:
                depth_beams = depth_beams.reshape(1, -1)
            if valid_beams.ndim == 1:
                valid_beams = valid_beams.reshape(1, -1)

            beam_depth = np.array(depth_beams, dtype=float, copy=True)
            if valid_beams.shape == beam_depth.shape:
                beam_depth[~valid_beams] = np.nan
            depth_transect = np.nanmedian(beam_depth, axis=0)

            # Robust fallback when no valid beam remains.
            if not np.any(np.isfinite(depth_transect)):
                depth_transect = np.asarray(bt.depth_processed_m, dtype=float)

            if transect.orig_start_edge == 'Right':
                valid = np.asarray(bt.valid_data[::-1], dtype=bool)
                depth_transect = depth_transect[::-1]
            else:
                valid = np.asarray(bt.valid_data, dtype=bool)

            depth_data.append(depth_transect[valid])

        return depth_data

    @staticmethod
    def _rebuild_depth_profiles_from_scattered_bathy(
            x_raw,
            y_raw,
            depth_data,
            x_query=None,
            y_query=None,
            bathy_span=0.005):
        """Rebuild depth profiles from scattered (x,y,z) bathymetry interpolation."""
        from scipy.spatial import Delaunay

        try:
            x_all = np.concatenate([np.asarray(x, dtype=float).ravel() for x in x_raw])
            y_all = np.concatenate([np.asarray(y, dtype=float).ravel() for y in y_raw])
            d_all = np.concatenate([np.asarray(d, dtype=float).ravel() for d in depth_data])
        except Exception:
            return depth_data

        finite = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(d_all) & (d_all > 0)
        if np.count_nonzero(finite) < 10:
            return depth_data

        known = np.vstack((x_all[finite], y_all[finite], -d_all[finite]))
        bathy = BathymetryScatteredPoints(known)

        tri = None
        try:
            if known.shape[1] >= 3:
                tri = Delaunay(known[:2, :].T)
        except Exception:
            tri = None

        try:
            bathy.interpolator.span = float(bathy_span)
            if hasattr(bathy.interpolator, "reset_interpolant"):
                bathy.interpolator.reset_interpolant()
        except Exception:
            pass

        if x_query is None or y_query is None:
            x_query = x_raw
            y_query = y_raw

        new_depth_data = []
        for x_q, y_q, d_t in zip(x_query, y_query, depth_data):
            x_arr = np.asarray(x_q, dtype=float)
            y_arr = np.asarray(y_q, dtype=float)
            d_arr = np.asarray(d_t, dtype=float)

            # Align fallback depths if query geometry differs from original track.
            if d_arr.shape != x_arr.shape:
                if d_arr.size >= 2 and x_arr.size >= 1:
                    src_idx = np.linspace(0.0, 1.0, d_arr.size)
                    tgt_idx = np.linspace(0.0, 1.0, x_arr.size)
                    d_base = np.interp(tgt_idx, src_idx, d_arr)
                elif d_arr.size == 1:
                    d_base = np.full(x_arr.shape, float(d_arr[0]), dtype=float)
                else:
                    d_base = np.full(x_arr.shape, np.nan, dtype=float)
            else:
                d_base = d_arr

            if x_arr.shape != y_arr.shape:
                new_depth_data.append(np.asarray(d_base, dtype=float))
                continue

            try:
                z_bed = bathy.get_bed_elev(x_arr, y_arr)
                d_new = -np.asarray(z_bed, dtype=float)

                # Avoid unstable extrapolation beyond the sampled domain.
                if tri is not None:
                    pts = np.c_[x_arr.ravel(), y_arr.ravel()]
                    inside = tri.find_simplex(pts) >= 0
                    inside = inside.reshape(x_arr.shape)
                else:
                    inside = np.ones_like(d_new, dtype=bool)

                d_new[~inside] = d_base[~inside]
                invalid = (~np.isfinite(d_new)) | (d_new <= 0)
                d_new[invalid] = d_base[invalid]
                new_depth_data.append(d_new)
            except Exception:
                new_depth_data.append(np.asarray(d_base, dtype=float))

        return new_depth_data

    @staticmethod
    def collect_data(meas, navigation_reference, checked_transect_idx, interp_option):
        """Collect all transect data required by Vermeulen processing."""
        depth_data = []
        x_raw_coordinates = []
        y_raw_coordinates = []
        w_vel_x = []
        w_vel_y = []
        w_vel_z = []
        q_cells = []
        dmg = []
        orig_start_edge = []
        invalid_data = []
        cell_depth = []

        left_param = np.tile([np.nan], (len(checked_transect_idx), 2))
        right_param = np.tile([np.nan], (len(checked_transect_idx), 2))

        for id_transect in checked_transect_idx:
            transect = meas.transects[id_transect]
            index_transect = checked_transect_idx.index(id_transect)
            in_transect_idx = transect.in_transect_idx

            if navigation_reference == 'bt_vel':
                ship_data = transect.boat_vel.compute_boat_track(transect, ref='bt_vel')
            else:
                ship_data = transect.boat_vel.compute_boat_track(transect, ref='gga_vel')

            if transect.orig_start_edge == 'Right':
                valid = transect.depths.bt_depths.valid_data[::-1]
                dmg_ind = np.where(abs(ship_data['dmg_m']) == max(abs(ship_data['dmg_m'])))[0][0]
                x_track = ship_data['track_x_m'] - ship_data['track_x_m'][dmg_ind]
                y_track = ship_data['track_y_m'] - ship_data['track_y_m'][dmg_ind]
                x_transect = x_track[::-1]
                y_transect = y_track[::-1]
                depth_transect = transect.depths.bt_depths.depth_processed_m[::-1]
                dmg_transect = ship_data['dmg_m'][::-1]
                q_cell_transect = meas.discharge[id_transect].middle_cells[:, ::-1]
            else:
                valid = transect.depths.bt_depths.valid_data
                x_transect = ship_data['track_x_m']
                y_transect = ship_data['track_y_m']
                depth_transect = transect.depths.bt_depths.depth_processed_m
                dmg_transect = ship_data['dmg_m']
                q_cell_transect = meas.discharge[id_transect].middle_cells

            q_cells.append(q_cell_transect[:, valid])
            x_transect = x_transect[valid]
            y_transect = y_transect[valid]
            depth_data.append(depth_transect[valid])
            x_raw_coordinates.append(x_transect)
            y_raw_coordinates.append(y_transect)
            dmg.append(dmg_transect[valid])

            left_param[index_transect, :] = [
                meas.transects[id_transect].edges.left.distance_m,
                meas.discharge[id_transect].edge_coef('left', meas.transects[id_transect]),
            ]
            right_param[index_transect, :] = [
                meas.transects[id_transect].edges.right.distance_m,
                meas.discharge[id_transect].edge_coef('right', meas.transects[id_transect]),
            ]
            left_geometry = np.nanmedian(left_param, axis=0)
            right_geometry = np.nanmedian(right_param, axis=0)

            invalid = np.logical_not(transect.w_vel.valid_data[0, :, in_transect_idx]).T
            vel_x = np.copy(transect.w_vel.u_processed_mps)
            vel_y = np.copy(transect.w_vel.v_processed_mps)
            vel_z = np.copy(transect.w_vel.w_mps[:, in_transect_idx])

            if not interp_option:
                vel_x[invalid] = np.nan
                vel_y[invalid] = np.nan
                vel_z[invalid] = np.nan

            w_vel_x.append(vel_x[:, valid])
            w_vel_y.append(vel_y[:, valid])
            w_vel_z.append(vel_z[:, valid])
            invalid_data.append(invalid[:, valid])

            orig_start_edge.append(transect.orig_start_edge)
            cell_depth.append(transect.depths.bt_depths.depth_cell_depth_m[:, valid])

        return x_raw_coordinates, y_raw_coordinates, w_vel_x, w_vel_y, w_vel_z, \
            depth_data, q_cells, orig_start_edge, invalid_data, cell_depth, left_geometry, \
            right_geometry, dmg

    @staticmethod
    def compute_coef(x_raw_coordinates, y_raw_coordinates, w_vel_x, w_vel_y, q_cells, track_section):
        """Compute average cross-section coefficients and direction."""
        func = {'velocity': lambda x, b: x + b,
                'track': lambda x, a, b: a * x + b}
        q_ens = [abs(np.nansum(l, axis=0) / np.nanmean(l)) for l in q_cells]

        if track_section:
            coef_meas = np.tile(np.nan, (len(x_raw_coordinates), 2))
            for i in range(len(x_raw_coordinates)):
                wls = LinearRegression()
                wls.fit(
                    x_raw_coordinates[i].reshape(-1, 1),
                    y_raw_coordinates[i].reshape(-1, 1),
                    sample_weight=np.abs(q_ens[i]),
                )
                coef_meas[i, :] = [wls.coef_, wls.intercept_]

            alpha = np.nanmedian(coef_meas[:, 0])
            beta = np.nanmedian(coef_meas[:, 1])
            direction_meas = np.arctan2(-1, alpha)
        else:
            mean_x = []
            mean_y = []
            for i in range(len(w_vel_x)):
                mean_x.append(np.nansum(w_vel_x[i] * q_cells[i]) / np.nansum(q_cells[i]))
                mean_y.append(np.nansum(w_vel_y[i] * q_cells[i]) / np.nansum(q_cells[i]))
            v_x = np.nanmedian(mean_x)
            v_y = np.nanmedian(mean_y)
            direction_meas, _ = cart2pol(v_x, v_y)
            alpha = -1 / np.tan(direction_meas)
            coef_meas = []
            for i in range(len(w_vel_x)):
                coef_meas.append(curve_fit(func['velocity'], alpha * x_raw_coordinates[i], y_raw_coordinates[i])[0])
            beta = np.nanmean(coef_meas)

        return alpha, beta, direction_meas, q_ens

    @staticmethod
    def project_transect(alpha, beta, x_raw_coordinates, y_raw_coordinates):
        """Project transects on the average cross-section."""
        x_projected = []
        y_projected = []
        x_left = []
        y_left = []
        x_boundaries = [np.nan, np.nan]
        y_boundaries = [np.nan, np.nan]

        for i in range(len(x_raw_coordinates)):
            x_projected.append((x_raw_coordinates[i] - alpha * beta + alpha * y_raw_coordinates[i]) / (alpha ** 2 + 1))
            y_projected.append((beta + alpha * x_raw_coordinates[i] + alpha ** 2 * y_raw_coordinates[i]) / (alpha ** 2 + 1))

            x_left.append(x_projected[i][0])
            y_left.append(y_projected[i][0])

            x_boundaries = [min([min(l) for l in x_projected]), max([max(l) for l in x_projected])]
            y_boundaries = [min([min(l) for l in y_projected]), max([max(l) for l in y_projected])]

            x_start = min(x_boundaries, key=lambda x: abs(x - np.nanmedian(x_left)))
            y_start = min(y_boundaries, key=lambda x: abs(x - np.nanmedian(y_left)))

        acs_distance = []
        x_origin = min(x_boundaries, key=abs)
        y_origin = min(y_boundaries, key=abs)
        for i in range(len(x_projected)):
            x_distance = x_projected[i] - x_origin
            y_distance = y_projected[i] - y_origin
            acs_distance.append(np.sqrt(x_distance ** 2 + y_distance ** 2))

        _ = (x_start, y_start)
        return x_projected, y_projected, acs_distance

    @staticmethod
    def translated_transects(acs_distance_raw, depth_data):
        """Translate transects to align bathymetry on the average section."""
        acs_translated = []
        acs_translated.append(copy.deepcopy(acs_distance_raw[0]))
        max_acs = max([max(l) for l in acs_distance_raw])
        min_acs = min([min(l) for l in acs_distance_raw])
        grid_acs = np.arange(min_acs, max_acs, (max_acs - min_acs) / 100)
        acs_depth_ref = sc.interpolate.griddata(acs_distance_raw[0], depth_data[0], grid_acs)

        for i in range(1, len(acs_distance_raw)):
            acs_depth = sc.interpolate.griddata(acs_distance_raw[i], depth_data[i], grid_acs)
            if not np.all(np.isnan(acs_depth)):
                first_valid_acs = int(np.argwhere(~np.isnan(acs_depth))[0])
                last_valid_acs = int(np.argwhere(~np.isnan(acs_depth))[-1])
                valid_acs_depth = last_valid_acs - first_valid_acs + 1
                acs_depth_valid = acs_depth[first_valid_acs:last_valid_acs + 1]

                default_acs_ssd = np.nanmean((acs_depth_ref[:valid_acs_depth] - acs_depth_valid) ** 2)
                lag_acs_idx = 0
                for j in range(1, len(acs_depth_ref) - valid_acs_depth + 1):
                    acs_ssd = np.nanmean((acs_depth_ref[j:j + valid_acs_depth] - acs_depth_valid) ** 2)
                    if (acs_ssd < default_acs_ssd or np.isnan(default_acs_ssd)) and \
                            np.count_nonzero(np.isnan(acs_depth_ref[j:j + valid_acs_depth])) < 0.5 * valid_acs_depth:
                        default_acs_ssd = acs_ssd
                        lag_acs_idx = j

                acs_corrected = acs_distance_raw[i] - (grid_acs[first_valid_acs] - grid_acs[lag_acs_idx])
                acs_translated.append(acs_corrected)
            else:
                acs_translated.append(acs_distance_raw[i])

        min_dist = min([min(l) for l in acs_translated])
        for i in range(len(acs_translated)):
            acs_translated[i] -= min_dist

        return acs_translated

    @staticmethod
    def compute_node_size_from_sigma_zeta_builder(
            meas,
            navigation_reference,
            checked_transect_idx,
            acs_distance,
            nbr_cell_hor,
            nbr_cell_vert,
            node_vertical_user,
            bathy_span,
            use_raw_bt_beam_bathy=False):
        """Build mesh limits from SigmaZetaMeshFromVMADCP and map to MAP axis."""
        class _ScatteredBathyFallback:
            """Simple scattered bathymetry fallback (linear + nearest)."""

            def __init__(self, known_xyz):
                xyz = np.asarray(known_xyz, dtype=float)
                if xyz.ndim != 2 or xyz.shape[0] != 3:
                    raise ValueError("known bathymetry must be shaped (3, N)")
                finite = np.all(np.isfinite(xyz), axis=0)
                self.known = xyz[:, finite]
                if self.known.shape[1] < 3:
                    raise ValueError("not enough known bathymetry points")

                pts = self.known[:2, :].T
                vals = self.known[2, :]
                self._lin = sc.interpolate.LinearNDInterpolator(pts, vals)
                self._nn = sc.interpolate.NearestNDInterpolator(pts, vals)

            def get_bed_elev(self, x, y):
                x_arr = np.asarray(x, dtype=float)
                y_arr = np.asarray(y, dtype=float)
                if x_arr.shape != y_arr.shape:
                    raise ValueError("x and y must have same shape")
                pts = np.c_[x_arr.ravel(), y_arr.ravel()]
                z_lin = np.asarray(self._lin(pts), dtype=float)
                z_nn = np.asarray(self._nn(pts), dtype=float)
                z = np.where(np.isfinite(z_lin), z_lin, z_nn)
                return z.reshape(x_arr.shape)

        id_ref = int(checked_transect_idx[0])
        transect = meas.transects[id_ref]

        vmadcp = VMADCP(
            meas,
            transect_idx=id_ref,
            nav_ref=navigation_reference,
            use_raw_bt_beam_bathy=use_raw_bt_beam_bathy,
        )
        ef = EnsembleFilter.from_qrevint_transect(transect)

        # Build scattered bathymetry robustly for mesh generation.
        # We try LoessNN first, then fallback to linear+nearest if loess runtime
        # is unavailable in the current environment.
        bpos = np.asarray(getattr(vmadcp, "bed_position"), dtype=float)
        if bpos.ndim < 4 or bpos.shape[-1] < 3:
            raise ValueError("vmadcp.bed_position has unexpected shape")
        bx = bpos[..., 0].reshape(-1)
        by = bpos[..., 1].reshape(-1)
        bz = bpos[..., 2].reshape(-1)
        finite_b = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bz)
        known_xyz = np.vstack((bx[finite_b], by[finite_b], bz[finite_b]))

        try:
            bathy = BathymetryScatteredPoints(known_xyz)
            bathy.interpolator.span = float(bathy_span)
            if hasattr(bathy.interpolator, "reset_interpolant"):
                bathy.interpolator.reset_interpolant()
            # Probe once to ensure backend is callable.
            _ = bathy.get_bed_elev(np.array([known_xyz[0, 0]]), np.array([known_xyz[1, 0]]))
        except Exception:
            bathy = _ScatteredBathyFallback(known_xyz)
            print("[Vermeulen] Loess bathy unavailable for mesh builder, using linear+nearest fallback")

        all_acs = np.concatenate([np.asarray(a, dtype=float).ravel() for a in acs_distance])
        all_acs = all_acs[np.isfinite(all_acs)]
        if all_acs.size < 2:
            raise ValueError("Not enough finite acs_distance values")
        acs_min = float(np.nanmin(all_acs))
        acs_max = float(np.nanmax(all_acs))
        if not np.isfinite(acs_max - acs_min) or (acs_max - acs_min) <= 1e-9:
            raise ValueError("Degenerate acs_distance range")

        n_hp = np.asarray(vmadcp.horizontal_position, dtype=float)
        _, n_track = XSection(vmadcp).xy2sn(n_hp[0], n_hp[1])
        n_track = np.asarray(n_track, dtype=float)
        finite_track = np.isfinite(n_track)
        if np.count_nonzero(finite_track) < 2:
            raise ValueError("Not enough valid projected track points")
        n_span = float(np.nanmax(n_track[finite_track]) - np.nanmin(n_track[finite_track]))
        if n_span <= 1e-9:
            raise ValueError("Degenerate projected n-range")

        n_cells = max(int(nbr_cell_hor) - 1, 1)
        deltan = max(n_span / n_cells, 1e-6)

        if node_vertical_user is not None:
            deltaz = max(float(node_vertical_user), 1e-6)
        else:
            dcp = np.asarray(vmadcp.depth_cell_position[..., 2], dtype=float)
            zmin = float(np.nanmin(dcp))
            zmax = float(np.nanmax(dcp))
            depth_guess = max(abs(zmin - zmax), 0.1)
            deltaz = max(depth_guess / max(int(nbr_cell_vert), 1), 1e-6)

        builder = SigmaZetaMeshFromVMADCP(
            vmadcp=vmadcp,
            bathymetry=bathy,
            filter_obj=ef,
            deltan=deltan,
            deltaz=deltaz,
        )
        mesh = builder.get_mesh()

        borders_n = np.concatenate((mesh.n_left[:1], mesh.n_right))
        borders_n = borders_n[np.isfinite(borders_n)]
        if borders_n.size < 2:
            raise ValueError("SigmaZeta builder returned insufficient borders")

        nmin = float(np.nanmin(borders_n))
        nmax = float(np.nanmax(borders_n))
        if nmax - nmin <= 1e-12:
            raise ValueError("SigmaZeta borders have zero span")

        borders_ens = acs_min + (borders_n - nmin) / (nmax - nmin) * (acs_max - acs_min)

        n_vert = int(np.nanmax(mesh.row_to_cell)) if mesh.row_to_cell.size > 0 else int(nbr_cell_vert)
        n_vert = max(n_vert, 1)
        depth_max = float(np.nanmax(builder.water_level - np.asarray(mesh.zb_all, dtype=float)))
        depth_max = depth_max if np.isfinite(depth_max) and depth_max > 0 else 1.0
        nodes_depth = np.linspace(0.0, depth_max, n_vert + 1)

        return np.asarray(borders_ens, dtype=float), np.asarray(nodes_depth, dtype=float), mesh

    @staticmethod
    def plot_sigma_zeta_mesh(mesh, path_results, name_meas, start_edge=None):
        """Save a dedicated figure of the Sigma-Zeta mesh used by Vermeulen."""
        os.makedirs(path_results, exist_ok=True)
        wl = float(getattr(mesh, "water_level", 0.0))

        fig, ax = plt.subplots(figsize=(11, 5.5))
        for cc in range(mesh.ncells):
            col = int(mesh.col_to_cell[cc])
            verts = np.array([
                [mesh.n_left[col], wl - mesh.z_bottom_left[cc]],
                [mesh.n_middle[col], wl - mesh.z_bottom_mid[cc]],
                [mesh.n_right[col], wl - mesh.z_bottom_right[cc]],
                [mesh.n_right[col], wl - mesh.z_top_right[cc]],
                [mesh.n_middle[col], wl - mesh.z_top_mid[cc]],
                [mesh.n_left[col], wl - mesh.z_top_left[cc]],
            ])
            poly = Polygon(verts, closed=True, fill=False, edgecolor='k', linewidth=0.35)
            ax.add_patch(poly)

        if hasattr(mesh, "nb_all") and hasattr(mesh, "zb_all"):
            nbed = np.asarray(mesh.nb_all, dtype=float)
            zbed = np.asarray(mesh.zb_all, dtype=float)
            ok = np.isfinite(nbed) & np.isfinite(zbed)
            if np.any(ok):
                ax.plot(nbed[ok], wl - zbed[ok], color='tab:blue', linewidth=1.4, label='Fond')
        ax.axhline(0.0, color='tab:cyan', linewidth=1.2, label='Surface')

        ax.set_title(f'Maillage Sigma-Zeta Vermeulen - {name_meas}')
        ax.set_xlabel('Distance section (m)')
        ax.set_ylabel('Profondeur (m)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)
        if MAP_vermeulen._is_right_start_edge(start_edge):
            ax.invert_xaxis()
        MAP_vermeulen._add_bank_labels(ax)
        ax.legend(loc='best')
        ax.autoscale()

        out_path = os.path.join(path_results, f"SigmaZeta_Mesh_{name_meas}.png")
        fig.savefig(out_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        print(f"[Vermeulen] Sigma-Zeta mesh plot saved: {out_path}")

    @staticmethod
    def compute_node_size(cell_depth, acs_distance, depth_data, node_horizontal_user,
                          node_vertical_user, nbr_cell_hor, nbr_cell_vert, top_cell):
        """Define horizontal and vertical mesh for Vermeulen solution."""
        borders_ens = np.linspace(min([min(l) for l in acs_distance]),
                                  max([max(l) for l in acs_distance]) + 10 ** -5,
                                  int(nbr_cell_hor))

        all_depth = np.array([item for subarray in depth_data for item in subarray])
        if node_vertical_user:
            nodes_depth = np.round(np.arange(top_cell,
                                             np.nanmax(all_depth) + 2 * node_vertical_user,
                                             node_vertical_user).tolist(), 3)
        else:
            lag = np.nanmax([0.1, 2 * (cell_depth[0][1, 0] - cell_depth[0][0, 0])])
            nodes_depth = np.arange(top_cell,
                                    np.nanmax(all_depth) + lag,
                                    (np.nanmax(all_depth) + lag) / (nbr_cell_vert * 1.4))

        print("Profondeur premier noeud = ", nodes_depth[0])
        _ = node_horizontal_user
        return borders_ens, nodes_depth
 

# ------------------------------------------------------------------ #
#  RÉSOLUTION PAR MOINDRES CARRÉS                                    #
# ------------------------------------------------------------------ #
 
    @staticmethod
    def _build_hpr_matrix(heading_deg, pitch_deg, roll_deg):
        """Construit la matrice de rotation Heading-Pitch-Roll (3×3).
 
        Reproduit exactement le calcul de WaterData.change_coord_sys dans
        QRevInt (et hpr_matrix dans VelocitySolver.m).
 
        Parameters
        ----------
        heading_deg, pitch_deg, roll_deg : float
            Angles en degrés pour un ping donné.
 
        Returns
        -------
        hpr : np.ndarray (3, 3)
        """
        h = math.radians(heading_deg)
        p = math.radians(pitch_deg)
        r = math.radians(roll_deg)
 
        ch, sh = math.cos(h), math.sin(h)
        cp, sp = math.cos(p), math.sin(p)
        cr, sr = math.cos(r), math.sin(r)
 
        # Convention QRevInt / Teledyne RDI :
        # même formule que WaterData.change_coord_sys lignes 552-558
        hpr = np.array([
            [(ch * cr) + (sh * sp * sr),  sh * cp, (ch * sr) - (sh * sp * cr)],
            [-(sh * cr) + (ch * sp * sr), ch * cp, -(sh * sr) - (ch * sp * cr)],
            [-cp * sr,                    sp,       cp * cr                    ]
        ])
        return hpr
 
    @staticmethod
    def compute_nodes_velocity_vermeulen(
            meas, navigation_reference, checked_transect_idx,
            cell_depth, depth_data, acs_distance,
            borders_ens_raw, nodes_depth_raw, orig_start_edge,
            f_vitesse_z, f_direction_fixe, f_direction_pond,
            pond_vitesses):
        """Résout [u, v, w] par moindres carrés pondérés — méthode Vermeulen.
 
        Portage Python de VelocitySolver.get_parameters() (Matlab).
 
        Pour chaque cellule du maillage :
          1. Collecte de tous les scalaires faisceau (raw_vel_mps)
             dont la position physique tombe dans la cellule.
          2. Filtrage sigma sur les vitesses radiales brutes.
          3. Filtrage directionnel (vecteurs de transformation).
          4. Pondération spatiale (distance à la section).
          5. Résolution : A·[u,v,w] = b  (moindres carrés pondérés).
 
        Parameters
        ----------
        meas : Measurement
        navigation_reference : str  ('bt_vel' ou 'gga_vel')
        checked_transect_idx : list[int]
        cell_depth : list[np.ndarray]
            Profondeur des cellules ADCP (n_cells × n_ens) par transect.
        depth_data : list[np.ndarray]
            Profondeur bathymétrique (n_ens,) par transect.
        acs_distance : list[np.ndarray]
            Distance sur la section (n_ens,) par transect.
        borders_ens_raw : np.ndarray
            Bornes horizontales du maillage.
        nodes_depth_raw : np.ndarray
            Bornes verticales du maillage.
        orig_start_edge : list[str]
            Rive de départ de chaque transect.
        f_vitesse_z, f_direction_fixe, f_direction_pond, pond_vitesses : float
            Paramètres de la méthode (voir populate_data).
 
        Returns
        -------
        transects_node_x_vel : np.ndarray (n_tr × n_vert × n_hor)
        transects_node_y_vel : np.ndarray
        transects_node_z_vel : np.ndarray
        transects_node_depth : np.ndarray (n_tr × n_hor)
        transects_nodes : list[list[int]]
        transects_node_nz : np.ndarray (n_tr × n_hor)
            Nombre de couches verticales actives par nœud horizontal.
        sigma_cap_ref : float
            Sigma maximum de référence (lié à l'angle des faisceaux).
        info_cell : np.ndarray  (n_tr × n_vert × n_hor)
            Nb de mesures radiales par cellule et transect.
        r2_cell : np.ndarray   (n_tr × n_vert × n_hor)
            Erreur quadratique résiduelle par cellule.
        r_sig_cell : np.ndarray (n_tr × n_vert × n_hor)
            Écart-type des résidus par cellule.
        """
        n_tr = len(checked_transect_idx)
        n_vert = len(nodes_depth_raw) - 1
        n_hor = len(borders_ens_raw) - 1
        dz_target = np.nanmedian(np.diff(nodes_depth_raw))
        if not np.isfinite(dz_target) or dz_target <= 0:
            dz_target = 1.0

        def fit_sig(n_val, n0, n1, sig0, sig1):
            """Linear sigma interpolation used by SigmaZetaMesh.fit_sig."""
            if not np.isfinite(n_val) or not np.isfinite(n0) or not np.isfinite(n1):
                return np.nan
            if abs(n1 - n0) < 1e-12:
                return 0.5 * (sig0 + sig1)
            return ((sig1 - sig0) / (n1 - n0)) * (n_val - n0) + sig0
 
        transects_nodes = []
        transects_node_x_vel = np.full((n_tr, n_vert, n_hor), np.nan)
        transects_node_y_vel = np.full((n_tr, n_vert, n_hor), np.nan)
        transects_node_z_vel = np.full((n_tr, n_vert, n_hor), np.nan)
        transects_node_depth = np.full((n_tr, n_hor), np.nan)
        transects_node_nz = np.full((n_tr, n_hor), np.nan)
        info_cell   = np.full((n_tr, n_vert, n_hor), np.nan)
        r2_cell     = np.full((n_tr, n_vert, n_hor), np.nan)
        r_sig_cell  = np.full((n_tr, n_vert, n_hor), np.nan)
        sigma_cap_ref = np.nan
 
        for id_transect in checked_transect_idx:
            idx_tr = checked_transect_idx.index(id_transect)
            print(f'===== Transect {idx_tr} of {n_tr - 1} =====')

            transect = meas.transects[id_transect]

            # Sigma maximum utilisable (minsigma = 1-cos(angle), donc sigma_max = cos(angle)).
            try:
                beam_angle = float(transect.adcp.beam_angle_deg)
                sigma_cap = float(np.clip(np.cos(np.deg2rad(beam_angle)), 0.05, 1.0))
            except Exception:
                sigma_cap = 0.94
            if not np.isfinite(sigma_cap_ref):
                sigma_cap_ref = sigma_cap

            # ── Matrice instrument (beam↔instrument) ─────────────────
            t_mat = np.asarray(transect.adcp.t_matrix.matrix)

            # ── Profondeurs des cellules (déjà filtrées par collect_data) ──
            # cell_depth[idx_tr]  : (n_cells, n_ens_valides)
            # depth_data[idx_tr]  : (n_ens_valides,)
            # acs_distance[idx_tr]: (n_ens_valides,)
            # Ces arrays n'ont que les ensembles in_transect valides.
            cell_depth_tr = cell_depth[idx_tr]    # (n_cells, n_ens_valides)
            depth_ens_tr  = depth_data[idx_tr]    # (n_ens_valides,)
            acs_dist_tr   = acs_distance[idx_tr]  # (n_ens_valides,)
            n_ens_valides = len(acs_dist_tr)

            # ── Masque valid + réordonnancement (comme collect_data) ──
            valid_mask = np.asarray(transect.depths.bt_depths.valid_data)

            # ── Vitesses brutes faisceau (4 × n_cells × n_ens_total) ──
            raw_vel_full = np.asarray(transect.w_vel.raw_vel_mps)

            # ── Capteurs (n_ens_total,) ───────────────────────────────
            sensors = transect.sensors
            heading_full = getattr(sensors.heading_deg,
                                   sensors.heading_deg.selected).data
            pitch_full   = getattr(sensors.pitch_deg,
                                   sensors.pitch_deg.selected).data
            roll_full    = getattr(sensors.roll_deg,
                                   sensors.roll_deg.selected).data

            if orig_start_edge[idx_tr] == 'Right':
                valid_mask = valid_mask[::-1]
                raw_vel_full = np.flip(raw_vel_full, axis=2)
                heading_full = heading_full[::-1]
                pitch_full = pitch_full[::-1]
                roll_full = roll_full[::-1]

            # Filtrage sur les ensembles valides → (4, n_cells, n_ens_valides)
            raw_vel = raw_vel_full[:, :, valid_mask]
            heading = heading_full[valid_mask]
            pitch   = pitch_full[valid_mask]
            roll    = roll_full[valid_mask]

            # ── Distance longitudinale pour la pondération ───────────
            s_ens = acs_dist_tr
            max_s = 1.01 * np.nanmax(np.abs(s_ens)) if np.any(np.isfinite(s_ens)) else 1.0

            # cmesh.index utilise n >= n_left & n < n_right. side='right' reproduit ce choix aux frontières.
            node_indices = np.searchsorted(borders_ens_raw, acs_dist_tr, side='right') - 1
            node_indices = np.where(
                np.isfinite(acs_dist_tr) & (acs_dist_tr >= borders_ens_raw[0])
                & (acs_dist_tr < borders_ens_raw[-1]),
                node_indices,
                -1)
            node_indices = np.clip(node_indices, -1, n_hor - 1)

            id_proj = [int(v) for v in np.unique(node_indices) if v >= 0]
            transects_nodes.append(id_proj)

            # Prépare une interpolation 1D n -> profondeur (bathy locale)
            finite_depth = np.isfinite(acs_dist_tr) & np.isfinite(depth_ens_tr)
            if np.any(finite_depth):
                n_sorted_idx = np.argsort(acs_dist_tr[finite_depth])
                n_sorted = acs_dist_tr[finite_depth][n_sorted_idx]
                d_sorted = depth_ens_tr[finite_depth][n_sorted_idx]
                n_unique, idx_unique = np.unique(n_sorted, return_index=True)
                d_unique = d_sorted[idx_unique]
            else:
                n_unique = np.array([])
                d_unique = np.array([])

            def interp_depth_at_n(nq):
                if n_unique.size == 0:
                    return np.nan
                if n_unique.size == 1:
                    return float(d_unique[0])
                return float(np.interp(nq, n_unique, d_unique,
                                       left=d_unique[0], right=d_unique[-1]))

            maxz_tr = np.nanmax(cell_depth_tr)
            if not np.isfinite(maxz_tr) or maxz_tr <= 0:
                maxz_tr = np.nanmax(depth_ens_tr)

            # ── Boucle sur les nœuds horizontaux ─────────────────────
            for node in id_proj:
                ens_in_node = np.where(node_indices == node)[0]
                if len(ens_in_node) == 0:
                    continue

                node_depth = np.nanmedian(depth_ens_tr[ens_in_node])
                transects_node_depth[idx_tr, node] = node_depth
                if not np.isfinite(node_depth) or node_depth <= 0:
                    continue

                # Géométrie horizontale de la cellule Sigma-Zeta (gauche/milieu/droite)
                n_left = borders_ens_raw[node]
                n_right = borders_ens_raw[node + 1]
                n_mid = 0.5 * (n_left + n_right)

                depth_left = interp_depth_at_n(n_left)
                depth_mid = interp_depth_at_n(n_mid)
                depth_right = interp_depth_at_n(n_right)
                if not np.all(np.isfinite([depth_left, depth_mid, depth_right])):
                    continue
                if depth_mid <= 0:
                    continue

                sig_left_max = min(maxz_tr / max(depth_left, 1e-6), sigma_cap)
                sig_mid_max = min(maxz_tr / max(depth_mid, 1e-6), sigma_cap)
                sig_right_max = min(maxz_tr / max(depth_right, 1e-6), sigma_cap)
                sig_left_max = float(np.clip(sig_left_max, 0.05, sigma_cap))
                sig_mid_max = float(np.clip(sig_mid_max, 0.05, sigma_cap))
                sig_right_max = float(np.clip(sig_right_max, 0.05, sigma_cap))

                # nz comme SigmaZetaMeshFromVMADCP: ceil((maxz - minz_mid)/deltaz), minz_mid <=> sigma=0.
                usable_mid = sig_mid_max * depth_mid
                nz_node = int(np.ceil(usable_mid / dz_target))
                nz_node = int(np.clip(nz_node, 1, n_vert))
                transects_node_nz[idx_tr, node] = nz_node

                row_top_left = (np.arange(nz_node) / nz_node) * sig_left_max
                row_top_mid = (np.arange(nz_node) / nz_node) * sig_mid_max
                row_top_right = (np.arange(nz_node) / nz_node) * sig_right_max
                row_bottom_left = ((np.arange(nz_node) + 1) / nz_node) * sig_left_max
                row_bottom_mid = ((np.arange(nz_node) + 1) / nz_node) * sig_mid_max
                row_bottom_right = ((np.arange(nz_node) + 1) / nz_node) * sig_right_max

                # Accumulation des équations par cellule verticale
                b_lists = [[] for _ in range(nz_node)]
                A_lists = [[] for _ in range(nz_node)]
                s_lists = [[] for _ in range(nz_node)]

                for ens in ens_in_node:
                    ens_depth = depth_ens_tr[ens]
                    if not np.isfinite(ens_depth) or ens_depth <= 0:
                        continue

                    n_ens = acs_dist_tr[ens]
                    if n_ens < n_left or n_ens > n_right:
                        continue

                    cell_z = cell_depth_tr[:, ens]
                    cell_sigma = cell_z / ens_depth
                    valid_rows = np.where(
                        np.isfinite(cell_sigma)
                        & (cell_sigma >= 0.0)
                        & (cell_sigma <= sigma_cap)
                    )[0]
                    if len(valid_rows) == 0:
                        continue

                    # Matrice HPR pour cet ensemble
                    try:
                        hpr = MAP_vermeulen._build_hpr_matrix(
                            heading[ens], pitch[ens], roll[ens])
                    except Exception:
                        continue

                    hpr4 = np.eye(4)
                    hpr4[:3, :3] = hpr

                    if t_mat.ndim == 2:
                        tm_ens = t_mat
                    elif t_mat.ndim == 3 and t_mat.shape[2] == len(heading):
                        tm_ens = t_mat[:, :, ens]
                    elif t_mat.ndim == 3:
                        tm_ens = t_mat[:, :, 0]
                    else:
                        continue

                    try:
                        beam_to_earth = hpr4 @ tm_ens
                        earth_to_beam = np.linalg.pinv(beam_to_earth)
                    except Exception:
                        continue

                    for cell_row in valid_rows:
                        sig_obs = float(cell_sigma[cell_row])

                        i_vert = -1
                        if (n_ens >= n_left) and (n_ens < n_mid):
                            for ridx in range(nz_node):
                                sig_bottom = fit_sig(
                                    n_ens, n_left, n_mid,
                                    row_top_left[ridx], row_top_mid[ridx])
                                sig_top = fit_sig(
                                    n_ens, n_left, n_mid,
                                    row_bottom_left[ridx], row_bottom_mid[ridx])
                                if sig_obs > sig_bottom and sig_obs <= sig_top:
                                    i_vert = ridx
                                    break
                        elif (n_ens >= n_mid) and (n_ens < n_right):
                            for ridx in range(nz_node):
                                sig_bottom = fit_sig(
                                    n_ens, n_mid, n_right,
                                    row_top_mid[ridx], row_top_right[ridx])
                                sig_top = fit_sig(
                                    n_ens, n_mid, n_right,
                                    row_bottom_mid[ridx], row_bottom_right[ridx])
                                if sig_obs > sig_bottom and sig_obs <= sig_top:
                                    i_vert = ridx
                                    break

                        if i_vert < 0:
                            continue

                        beams_raw = raw_vel[:, cell_row, ens]
                        for beam_idx in range(4):
                            b_val = beams_raw[beam_idx]
                            if not np.isfinite(b_val):
                                continue
                            xform_row = earth_to_beam[beam_idx, :3]
                            if not np.all(np.isfinite(xform_row)):
                                continue
                            b_lists[i_vert].append(b_val)
                            A_lists[i_vert].append(xform_row)
                            s_lists[i_vert].append(s_ens[ens])

                # ── Résolution cellule par cellule ────────────────────
                for id_vert in range(nz_node):
                    b_list = b_lists[id_vert]
                    A_list = A_lists[id_vert]
                    s_list = s_lists[id_vert]

                    if len(b_list) < 3:
                        continue

                    b_arr = np.array(b_list)
                    A_arr = np.array(A_list)
                    s_arr = np.array(s_list)

                    mean_b = np.mean(b_arr)
                    std_b = np.std(b_arr)
                    if std_b > 0:
                        keep = np.abs(b_arr - mean_b) <= f_vitesse_z * std_b
                        b_arr = b_arr[keep]
                        A_arr = A_arr[keep]
                        s_arr = s_arr[keep]

                    if len(b_arr) < 3:
                        continue

                    mean_A = np.mean(A_arr, axis=0)
                    norm_A = np.linalg.norm(mean_A)
                    if norm_A > 0:
                        v_norm = mean_A / norm_A
                        z_comp = v_norm[2]
                        seuil = max(
                            f_direction_fixe - f_direction_pond * len(b_arr),
                            0.95)
                        if z_comp < seuil and z_comp > 0.9:
                            continue

                    if len(b_arr) < 3:
                        continue

                    w_arr = 1.0 - (np.abs(s_arr) / max_s) ** pond_vitesses
                    w_arr = np.clip(w_arr, 1e-6, 1.0)
                    sqrt_w = np.sqrt(w_arr)
                    Aw = A_arr * sqrt_w[:, np.newaxis]
                    bw = b_arr * sqrt_w

                    if np.linalg.matrix_rank(Aw) < 3:
                        continue

                    result = np.linalg.lstsq(Aw, bw, rcond=None)
                    uvw = result[0]

                    residuals = b_arr - A_arr @ uvw
                    residuals_w = bw - Aw @ uvw
                    dof = max(len(b_arr) - 3, 1)
                    r2_val = np.sum(residuals_w ** 2) / dof
                    rsig_val = np.std(residuals)

                    transects_node_x_vel[idx_tr, id_vert, node] = uvw[0]
                    transects_node_y_vel[idx_tr, id_vert, node] = uvw[1]
                    transects_node_z_vel[idx_tr, id_vert, node] = uvw[2]
                    info_cell [idx_tr, id_vert, node] = len(b_arr)
                    r2_cell   [idx_tr, id_vert, node] = r2_val
                    r_sig_cell[idx_tr, id_vert, node] = rsig_val
 
        return (transects_node_x_vel, transects_node_y_vel,
            transects_node_z_vel, transects_node_depth,
            transects_nodes, transects_node_nz, sigma_cap_ref,
            info_cell, r2_cell, r_sig_cell)

# ------------------------------------------------------------------ #
#  MOYENNE MULTI-TRANSECTS                                             #
# ------------------------------------------------------------------ #
 
    def compute_mean(self, transects_nodes, transects_node_x_vel,
                     transects_node_y_vel, transects_node_z_vel,
                     transects_node_depth, transects_node_nz, sigma_cap,
                     borders_ens_raw,
                     nodes_depth_raw, info_cell, r2_cell, r_sig_cell):
        """Moyenne des cellules sur tous les transects.
 
        Version Vermeulen autonome, avec propagation de nb_vel, r2 et r_sig.
        """
        unique_node  = list(np.unique([x for l in transects_nodes for x in l]))
        node_selected = copy.deepcopy(unique_node)
        valid_cell    = max(int(len(transects_nodes) / 3), 1)
        for node in unique_node:
            if sum(x.count(node) for x in transects_nodes) < valid_cell or \
                    np.isnan(transects_node_depth[:, node]).all():
                node_selected.remove(node)
 
        node_min   = np.nanmin(node_selected)
        node_max   = np.nanmax(node_selected)
        node_range = list(range(node_min, node_max + 1))
        node_range_border = list(range(node_min, node_max + 2))
 
        borders_ens = borders_ens_raw[node_range_border]
        borders_ens -= min(borders_ens)
 
        n_vert = len(nodes_depth_raw) - 1
        n_hor  = len(node_range)
        if not np.isfinite(sigma_cap):
            sigma_cap = 0.94
 
        MAP_depth_cells_border = np.full((len(nodes_depth_raw), n_hor), np.nan)
 
        MAP_x_vel     = np.full((n_vert, n_hor), np.nan)
        MAP_y_vel     = np.full((n_vert, n_hor), np.nan)
        MAP_z_vel     = np.full((n_vert, n_hor), np.nan)
        MAP_depth     = np.full(n_hor, np.nan)
        MAP_info_cell = np.full((n_vert, n_hor), np.nan)
        MAP_r2        = np.full((n_vert, n_hor), np.nan)
        MAP_r_sig     = np.full((n_vert, n_hor), np.nan)
 
        for node in node_selected:
            idx_node = node_range.index(node)
            row = np.array([j for (j, sub) in enumerate(transects_nodes)
                            if node in sub])
 
            x_cell     = transects_node_x_vel[row, :, node]
            y_cell     = transects_node_y_vel[row, :, node]
            z_cell     = transects_node_z_vel[row, :, node]
            depth_cell = transects_node_depth[row, node]
            ic_cell    = info_cell  [row, :, node]
            r2c_cell   = r2_cell    [row, :, node]
            rsc_cell   = r_sig_cell [row, :, node]
 
            MAP_depth[idx_node] = np.nanmedian(depth_cell)

            nz_candidates = transects_node_nz[row, node]
            if np.any(np.isfinite(nz_candidates)):
                nz_node = int(np.round(np.nanmedian(nz_candidates)))
            else:
                nz_node = n_vert
            nz_node = int(np.clip(nz_node, 1, n_vert))
 
            MAP_x_vel[:, idx_node]     = np.nanmean(x_cell, axis=0)
            MAP_y_vel[:, idx_node]     = np.nanmean(y_cell, axis=0)
            MAP_z_vel[:, idx_node]     = np.nanmean(z_cell, axis=0)
            MAP_info_cell[:, idx_node] = np.nansum(ic_cell, axis=0)
            MAP_r2[:, idx_node]        = np.nanmean(r2c_cell, axis=0)
            MAP_r_sig[:, idx_node]     = np.nanmean(rsc_cell, axis=0)

            MAP_x_vel[nz_node:, idx_node] = np.nan
            MAP_y_vel[nz_node:, idx_node] = np.nan
            MAP_z_vel[nz_node:, idx_node] = np.nan
            MAP_info_cell[nz_node:, idx_node] = np.nan
            MAP_r2[nz_node:, idx_node] = np.nan
            MAP_r_sig[nz_node:, idx_node] = np.nan

            if np.isfinite(MAP_depth[idx_node]) and MAP_depth[idx_node] > 0:
                sigma_edges_node = np.linspace(0.0, sigma_cap, nz_node + 1)
                MAP_depth_cells_border[:nz_node + 1, idx_node] = \
                    sigma_edges_node * MAP_depth[idx_node]
 
        self.vertical_velocity = MAP_z_vel
        self.depths            = MAP_depth
        self.depth_cells_border = MAP_depth_cells_border
        self.borders_ens       = borders_ens
        self.nb_vel            = MAP_info_cell
        self.r2                = MAP_r2
        self.r_sig             = MAP_r_sig
 
        return MAP_x_vel, MAP_y_vel
 
# ------------------------------------------------------------------ #
#  MÉTHODES COMMUNES DU PIPELINE VERMEULEN                           #
# ------------------------------------------------------------------ #
 
    def compute_projection(self, x_velocity, y_velocity, direction_meas):
        """Projette [u_east, v_north] → [streamwise, transverse]."""
        unit_vec_1, unit_vec_2 = pol2cart(direction_meas, 1)
        unit_vec = np.vstack([unit_vec_1, unit_vec_2])
        w_vel_st = np.full(x_velocity.shape, np.nan)
        w_vel_tr = np.full(x_velocity.shape, np.nan)
        for i in range(x_velocity.shape[0]):
            for j in range(x_velocity.shape[1]):
                w_vel_st[i, j] = np.sum(
                    np.vstack([x_velocity[i, j], y_velocity[i, j]]) * unit_vec, 0)
                w_vel_tr[i, j] = (unit_vec_2 * x_velocity[i, j]
                                  - unit_vec_1 * y_velocity[i, j])
        self.streamwise_velocity = -w_vel_st
        self.transverse_velocity = -w_vel_tr
 
    def compute_extrap_velocity(self, nodes_depth_raw, settings, extrap_option):
        """Extrapolation haut/bas appliquée au profil Vermeulen."""
        units = 1 if extrap_option else np.nan
 
        depths            = self.depths
        depth_cells_border = self.depth_cells_border
        w_vel_st  = np.copy(self.streamwise_velocity)
        w_vel_tr  = np.copy(self.transverse_velocity)
        w_vel_z   = np.copy(self.vertical_velocity)
 
        MAP_dcb = np.full(depth_cells_border.shape, np.nan)
        for i in range(depth_cells_border.shape[1]):
            MAP_dcb[:, i] = nodes_depth_raw
        MAP_dcc = (MAP_dcb[1:, :] + MAP_dcb[:-1, :]) / 2
 
        blanking = depths * 0.9
        for i in range(MAP_dcc.shape[1]):
            inv = MAP_dcc[:, i] > blanking[i]
            w_vel_st[inv, i] = np.nan
            w_vel_tr[inv, i] = np.nan
            w_vel_z [inv, i] = np.nan
 
        valid      = ~np.isnan(w_vel_st)
        n_ens      = valid.shape[1]
        idx_bot    = np.full(n_ens, -1, dtype=int)
        idx_top    = np.full(n_ens, -1, dtype=int)
        for n in range(n_ens):
            tmp = np.where(~np.isnan(w_vel_st[:, n]))[0]
            if len(tmp) > 0:
                idx_top[n] = tmp[0]
                idx_bot[n] = tmp[-1]
            else:
                idx_top[n] = 0
 
        bot_method = settings['extrapBot']
        exponent   = settings['extrapExp']
        dcc = (depth_cells_border[1:] + depth_cells_border[:-1]) / 2
        mid_bed    = depths - dcc
 
        if bot_method == 'Power':
            coef_bot = np.nanmean(w_vel_st, 0)
        else:  # No Slip
            cutoff = 0.8 * depths
            depth_ok = nan_greater(dcc, np.tile(cutoff, (dcc.shape[0], 1)))
            use_ns   = depth_ok & ~np.isnan(w_vel_st)
            for j in range(len(idx_bot)):
                if idx_bot[j] >= 0:
                    use_ns[idx_bot[j], j] = True
            comp_ns = np.copy(w_vel_st)
            comp_ns[~use_ns] = np.nan
            coef_bot = np.nanmean(comp_ns, 0)
 
        idx_bed = copy.deepcopy(idx_bot)
        for n in range(len(idx_bed)):
            if idx_bed[n] > -1:
                while (idx_bed[n] < len(dcc[:, n]) and
                       depth_cells_border[idx_bed[n] + 1, n] <= depths[n]):
                    idx_bed[n] += 1
                bot_depth = mid_bed[idx_bot[n] + 1: idx_bed[n], n]
                val = coef_bot[n] * ((1 + 1 / exponent) / (1 / exponent)) * \
                      (bot_depth / depths[n]) ** exponent
                w_vel_st[(idx_bot[n] + 1): idx_bed[n], n] = val * units
                w_vel_tr[(idx_bot[n] + 1): idx_bed[n], n] = \
                    w_vel_tr[idx_bot[n], n] * units
                w_vel_z [(idx_bot[n] + 1): idx_bed[n], n] = np.nan
 
        top_method = settings['extrapTop']
        if top_method in ('Power', 'Constant'):
            coef_top = np.nanmean(w_vel_st, 0)
            for n in range(len(idx_top)):
                if idx_top[n] > 0:
                    if top_method == 'Power':
                        td = mid_bed[:idx_top[n], n]
                        val = coef_top[n] * ((1 + 1/exponent)/(1/exponent)) \
                              * (td / depths[n]) ** exponent
                        w_vel_st[:idx_top[n], n] = val * units
                    else:
                        w_vel_st[:idx_top[n], n] = w_vel_st[idx_top[n], n] * units
                    w_vel_tr[:idx_top[n], n] = w_vel_tr[idx_top[n], n] * units
 
        self.depth_cells_center       = dcc
        self.extrap_streamwise_velocity = w_vel_st
        self.extrap_transverse_velocity = w_vel_tr
        self.extrap_vertical_velocity   = w_vel_z
 
    def compute_edges(self, borders_ens_raw, mid_direction, settings, edge_constant):
        """Compute edge extrapolation for Vermeulen results."""
        extrap_exp = 1 / settings['extrapExp']
        self.left_streamwise_velocity, self.left_transverse_velocity, self.left_vertical_velocity, \
        self.left_borders, left_direction, left_mid_cells_x, left_mid_cells_y, \
        left_area = self.edge_velocity(self.left_distance, self.left_coef, 'left', borders_ens_raw, mid_direction,
                                       extrap_exp, edge_constant)

        self.right_streamwise_velocity, self.right_transverse_velocity, self.right_vertical_velocity, \
        self.right_borders, right_direction, right_mid_cells_x, right_mid_cells_y, \
        right_area = self.edge_velocity(self.right_distance, self.right_coef, 'right', borders_ens_raw, mid_direction,
                                        extrap_exp, edge_constant)

        return left_direction, right_direction, left_area, right_area, left_mid_cells_x, \
            right_mid_cells_x, left_mid_cells_y, right_mid_cells_y

    def edge_velocity(self, edge_distance, edge_coef, edge, borders_ens_raw, direction_meas,
                      extrap_exp, edge_constant):
        """Compute left/right edge velocity cells and geometry."""
        if edge == 'left':
            id_edge = 0
            node_size = abs(borders_ens_raw[1] - borders_ens_raw[0])
        elif edge == 'right':
            id_edge = -1
            node_size = abs(borders_ens_raw[-1] - borders_ens_raw[-2])

        if edge_constant:
            nb_nodes = int(np.round(0.5 + edge_distance / node_size))
            nodes = np.linspace(0, edge_distance, nb_nodes + 1)
        else:
            nodes = edge_distance - np.arange(0, edge_distance, node_size)[::-1]
            nodes = np.insert(nodes, 0, 0)
            nb_nodes = len(nodes) - 1

        nodes_mid = (nodes[1:] + nodes[:-1]) / 2
        edge_size_raw = self.depth_cells_border[:, id_edge]

        if edge_coef == 0.3535 and edge_distance > 0:
            border_depths = np.multiply(nodes, self.depths[id_edge] / edge_distance)
            cells_borders_depths_1 = np.transpose([edge_size_raw] * (len(border_depths) - 1))
            cells_borders_depths_2 = np.transpose([edge_size_raw] * len(border_depths))

            for i in range(len(border_depths) - 1):
                sub_index = next(x[0] for x in enumerate(cells_borders_depths_1[:, i])
                                 if x[1] >= int(1000 * border_depths[i + 1]) / 1000)
                cells_borders_depths_1[sub_index, i] = border_depths[i + 1]
                cells_borders_depths_1[sub_index + 1:, i] = np.nan
                cells_borders_depths_2[sub_index - 1, i + 1] = border_depths[i + 1]
                cells_borders_depths_2[sub_index:, i] = np.nan

            cut_x = edge_distance * edge_size_raw[edge_size_raw <= self.depths[id_edge]] / self.depths[id_edge]
            x_left = np.tile(nodes, (cells_borders_depths_1.shape[0], 1))

            for j in range(np.count_nonzero(~np.isnan(cut_x)) - 1):
                col, _ = next(x for x in enumerate(nodes) if x[1] > cut_x[j])
                row = np.where(edge_size_raw == edge_size_raw[j])[0][0]
                x_left[row, col - 1] = cut_x[j]
                x_left[row + 1:, col - 1] = nodes[col]

            area_rec2 = (x_left[:-1, 1:] - x_left[1:, :-1]) * (cells_borders_depths_1[1:, :] - cells_borders_depths_1[:-1, :])
            area_rec1 = (x_left[1:, :-1] - x_left[:-1, :-1]) * (cells_borders_depths_2[:-1, :-1] - cells_borders_depths_1[:-1, :])
            area_tri1 = (x_left[1:, :-1] - x_left[:-1, :-1]) * (cells_borders_depths_1[1:, :] - cells_borders_depths_2[:-1, :-1]) / 2
            area_tra1 = area_rec1 + area_tri1
            area = area_tra1 + area_rec2

            mid_rec2_x = (x_left[:-1, 1:] + x_left[1:, :-1]) / 2
            mid_rec2_y = (cells_borders_depths_1[:-1, :] + cells_borders_depths_1[1:, :]) / 2
            mid_rec1_x = (x_left[:-1, :-1] + x_left[1:, :-1]) / 2
            mid_rec1_y = (cells_borders_depths_1[:-1, :] + cells_borders_depths_2[:-1, :-1]) / 2
            mid_tri1_x = (x_left[:-1, :-1] + 2 * x_left[1:, :-1]) / 3
            mid_tri1_y = (2 * cells_borders_depths_2[:-1, :-1] + cells_borders_depths_1[1:, :]) / 3

            mid_tra1_x = (area_rec1 * mid_rec1_x + area_tri1 * mid_tri1_x) / area_tra1
            mid_tra1_y = (area_rec1 * mid_rec1_y + area_tri1 * mid_tri1_y) / area_tra1
            mid_tra1_x[area_tra1 == 0] = 0
            mid_tra1_y[area_tra1 == 0] = 0

            mid_cells_x = (area_rec2 * mid_rec2_x + area_tra1 * mid_tra1_x) / area
            mid_cells_y = (area_rec2 * mid_rec2_y + area_tra1 * mid_tra1_y) / area
            edge_exp = 2.41

            bed_distance = mid_cells_y * edge_distance / self.depths[id_edge]
            vertical_depth = mid_cells_x * self.depths[id_edge] / edge_distance
            is_edge = True

        elif edge_coef == 0.91 and edge_distance > 0:
            mid_cells_x = np.tile([nodes_mid], (len(edge_size_raw) - 1, 1))
            mid_cells_y = np.transpose([self.depth_cells_center[:, id_edge]] * nb_nodes)
            size_x = np.tile([nodes[1:] - nodes[:-1]], (len(edge_size_raw) - 1, 1))
            size_y = np.transpose([edge_size_raw[1:] - edge_size_raw[:-1]] * nb_nodes)
            area = size_x * size_y
            bed_distance = np.tile(0, area.shape)
            vertical_depth = np.tile([self.depths[id_edge]], area.shape)
            edge_exp = 10
            is_edge = True

        else:
            mid_cells_x = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            mid_cells_y = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            area = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            bed_distance = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            vertical_depth = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            edge_exp = np.nan

            edge_streamwise_velocity = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            edge_transverse_velocity = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            edge_vertical_velocity = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            is_edge = False

        if np.all(np.isnan(self.streamwise_velocity[:, id_edge])) or not is_edge:
            edge_streamwise_velocity = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            edge_transverse_velocity = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            edge_vertical_velocity = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
        else:
            streamwise_mean_valid = np.nanmean(self.extrap_streamwise_velocity[:, id_edge])
            vp_mean = streamwise_mean_valid * (mid_cells_x / edge_distance) ** (1 / edge_exp)
            edge_streamwise_velocity = vp_mean * ((extrap_exp + 1) / extrap_exp) * (
                (vertical_depth - mid_cells_y) / vertical_depth) ** (1 / extrap_exp)

            vertical_vel_first = np.insert(self.extrap_vertical_velocity[:, id_edge], 0, 0)
            vertical_vel_first = np.append(vertical_vel_first, 0)
            norm_depth_first = np.insert(self.depth_cells_center[:, id_edge] / self.depths[id_edge], 0, 0)
            norm_depth_first = np.append(norm_depth_first, 1)
            norm_depth_edge = mid_cells_y / vertical_depth
            edge_vertical_velocity = sc.interpolate.griddata(norm_depth_first, vertical_vel_first, norm_depth_edge)

            transverse_vel_first = np.insert(self.extrap_transverse_velocity[:, id_edge], 0,
                                             self.transverse_velocity[0, id_edge])
            transverse_vel_first = np.append(
                transverse_vel_first,
                transverse_vel_first[np.where(~np.isnan(transverse_vel_first))[-1][-1]],
            )
            depth_first = np.insert(self.depth_cells_center[:, id_edge], 0, 0)
            depth_first = np.append(depth_first, self.depths[id_edge])
            edge_transverse_vel_interp = sc.interpolate.griddata(depth_first, transverse_vel_first, mid_cells_y)

            edge_transverse_velocity = np.tile(np.nan, edge_streamwise_velocity.shape)
            for j in range(edge_streamwise_velocity.shape[1]):
                for i in range(np.count_nonzero(~np.isnan(edge_transverse_vel_interp[:, j]))):
                    edge_transverse_velocity[i, j] = interpolation(
                        np.array([bed_distance[i, j], edge_distance]),
                        np.array([0, edge_transverse_vel_interp[i, j]]),
                        mid_cells_x[i, j],
                    )

        if edge == 'right':
            edge_streamwise_velocity = edge_streamwise_velocity[:, ::-1]
            edge_transverse_velocity = edge_transverse_velocity[:, ::-1]
            edge_vertical_velocity = edge_vertical_velocity[:, ::-1]
            mid_cells_x = abs(edge_distance - mid_cells_x[:, ::-1])
            mid_cells_y = mid_cells_y[:, ::-1]
            area = area[:, ::-1]
            nodes = abs(edge_distance - nodes[::-1])

        edge_direction = np.tile(direction_meas, edge_streamwise_velocity.shape[1])
        return edge_streamwise_velocity, edge_transverse_velocity, edge_vertical_velocity, nodes, \
            edge_direction, mid_cells_x, mid_cells_y, area
 
    def compute_discharge(self, extrap_option, left_direction=0, right_direction=0,
                          left_area=0, right_area=0):
        """Compute discharge from middle cells and optional edge extrapolation."""
        distance = (self.borders_ens[1:] - self.borders_ens[:-1])
        depth = self.depth_cells_border[1:, :] - self.depth_cells_border[:-1, :]
        mid_area = distance * depth

        if extrap_option:
            middle_cells_discharge = mid_area * self.extrap_streamwise_velocity
            middle_discharge = np.nansum(middle_cells_discharge)

            if middle_discharge < 0:
                unit = -1
            else:
                unit = 1

            self.middle_cells_discharge = middle_cells_discharge * unit
            self.middle_discharge = middle_discharge * unit

            self.left_cells_discharge = self.left_streamwise_velocity * left_area * unit
            self.right_cells_discharge = self.right_streamwise_velocity * right_area * unit

            self.left_discharge = np.nansum(self.left_cells_discharge)
            self.right_discharge = np.nansum(self.right_cells_discharge)
            self.total_discharge = self.left_discharge + self.middle_discharge + self.right_discharge
        else:
            middle_cells_discharge = mid_area * self.streamwise_velocity
            middle_discharge = np.nansum(middle_cells_discharge)

            if middle_discharge < 0:
                unit = -1
            else:
                unit = 1

            self.middle_cells_discharge = middle_cells_discharge * unit
            self.middle_discharge = middle_discharge * unit
            self.total_discharge = middle_discharge * unit

    @staticmethod
    def _add_bank_labels(ax, left_label='RG', right_label='RD'):
        ax.text(0.02, -0.08, left_label,
                transform=ax.transAxes, ha='left', va='top', fontweight='bold', clip_on=False)
        ax.text(0.98, -0.08, right_label,
                transform=ax.transAxes, ha='right', va='top', fontweight='bold', clip_on=False)

    @staticmethod
    def _is_right_start_edge(start_edge):
        if not start_edge:
            return False
        if isinstance(start_edge, (list, tuple, np.ndarray)):
            if len(start_edge) == 0:
                return False
            start_edge = start_edge[0]
        return str(start_edge).lower() == 'right'

    def _start_edge_is_right(self):
        return MAP_vermeulen._is_right_start_edge(self.orig_start_edge)
 
    # ------------------------------------------------------------------ #
    #  VISUALISATION                                                        #
    # ------------------------------------------------------------------ #
 
    def plot_profile(self, borders_ens_raw, nodes_depth_raw,
                     left_mid_x, right_mid_x,
                     left_mid_y, right_mid_y,
                     path_results, name_meas,
                     quiver_scale_mode='classic95',
                     quiver_scale_fixed=None,
                     print_velocity_diagnostics=False):
        """Génère les figures de la méthode Vermeulen.
 
        Produit :
          - Classic_Streamwise_verm.png   (vitesse streamwise)
          - Info_cell_verm.png            (nb mesures par cellule)
          - r2_verm.png                   (erreur quadratique)
          - r_sig_verm.png                (écart-type résidu)
        """
        # ── Construction des polygones ─────────────────────────────────
        node_mid   = (borders_ens_raw[1:] + borders_ens_raw[:-1]) / 2
        mid_dist   = (self.borders_ens[1:] + self.borders_ens[:-1]) / 2
        depth_cells = self.depth_cells_border
 
        patches            = []
        list_poly_vertices = []
        vect_vel_st        = []
        vect_vel_tr        = []
        vect_vel_w         = []
        vect_nb_vel        = []
        vect_r2            = []
        vect_rsig          = []
 
        n_vert, n_hor = self.streamwise_velocity.shape
 
        for j in range(n_hor):
            for i in range(n_vert):
                if np.isnan(depth_cells[i + 1, j]):
                    continue
                x0 = self.borders_ens[j]
                x1 = self.borders_ens[j + 1]

                # Maillage souple: les faces haute/basse suivent aussi la
                # verticale voisine pour respecter la forme locale de section.
                y0_l = depth_cells[i, j]
                y1_l = depth_cells[i + 1, j]
                if j < n_hor - 1:
                    y0_r = depth_cells[i, j + 1]
                    y1_r = depth_cells[i + 1, j + 1]
                else:
                    y0_r = y0_l
                    y1_r = y1_l

                if any(np.isnan([x0, x1, y0_l, y1_l, y0_r, y1_r])):
                    continue
                verts = [[x0, y0_l], [x1, y0_r], [x1, y1_r], [x0, y1_l]]
                list_poly_vertices.append(verts)
                patches.append(Polygon(verts, closed=True))
                vect_vel_st.append(self.streamwise_velocity[i, j])
                vect_vel_tr.append(self.transverse_velocity[i, j])
                vect_vel_w.append(self.vertical_velocity[i, j])
                vect_nb_vel.append(
                    self.nb_vel[i, j] if self.nb_vel is not None else np.nan)
                vect_r2.append(
                    self.r2[i, j]    if self.r2    is not None else np.nan)
                vect_rsig.append(
                    self.r_sig[i, j] if self.r_sig is not None else np.nan)
 
        depths_plt = [self.depths[j] if not np.isnan(self.depths[j]) else 0
                      for j in range(n_hor)]
        x_axis = mid_dist
        x_fill = np.concatenate(([mid_dist[0] - 1], mid_dist,
                                  [mid_dist[-1] + 1]))
        depth_fill = np.concatenate(([0], depths_plt, [0]))
 
        def _make_figure(vect_data, vmin_val, vmax_val, title, cbar_label,
                         fname, v_dir=None, w_dir=None):
            """Sous-routine : génère et sauve une figure."""
            norm_data = [(v - vmin_val) / (vmax_val - vmin_val)
                         if not np.isnan(v) else np.nan
                         for v in vect_data]
            colors = [self.jet(v) for v in norm_data]
            pc = PatchCollection(patches, facecolors='white')
            pc.set_color(colors)
 
            canvas = MplCanvas(width=15, height=9, dpi=240)
            fig = canvas.fig
            fig.ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(left=0.08, bottom=0.2, right=1,
                                top=0.97, wspace=0.1, hspace=0)
            fig.ax.add_collection(pc)
            fig.ax.autoscale()
            fig.ax.invert_yaxis()
            fig.ax.fill_between(x_fill, np.nanmax(depths_plt) + 2,
                                depth_fill, color='w')
            fig.ax.plot(x_axis, depths_plt, color='k', linewidth=1.5)
            fig.ax.plot(x_axis, [0]*len(depths_plt), color='b', linewidth=2)
 
            if v_dir is not None and w_dir is not None:
                mid_x = [np.sum(v, 0)[0] / len(v) for v in list_poly_vertices]
                mid_y = [np.sum(v, 0)[1] / len(v) for v in list_poly_vertices]
                q = fig.ax.quiver(mid_x, mid_y, v_dir, w_dir, units='xy',
                                  scale=self.quiver_scale,
                                  width=0.015 * np.nanmax(depths_plt))
                fig.ax.quiverkey(
                    q, X=1, Y=-0.03, U=np.round(self.quiver_scale, 2),
                    label=f'{self.quiver_scale:.2f} m/s',
                    labelpos='S', coordinates='axes',
                    fontproperties={'size': 16})
 
            X_sc = list(range(len(vect_data)))
            Y_sc = [v + 30 for v in X_sc]
            cb = fig.colorbar(
                fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_data,
                               cmap='jet', vmin=vmin_val, vmax=vmax_val))
            cb.ax.set_ylabel(canvas.tr(cbar_label))
            cb.ax.yaxis.label.set_fontsize(16)
            cb.ax.tick_params(labelsize=16)
            fig.ax.set_xlabel(canvas.tr('Distance (m)'))
            fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
            fig.ax.xaxis.label.set_fontsize(18)
            fig.ax.yaxis.label.set_fontsize(18)
            fig.ax.tick_params(axis='both', direction='in',
                               bottom=True, top=True, left=True, right=True)
            fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(depths_plt) * 1.02)
            fig.ax.set_xlim(left=mid_dist[0] - 0.5,
                             right=mid_dist[-1] + 0.5)
            if self._start_edge_is_right():
                fig.ax.invert_xaxis()
            MAP_vermeulen._add_bank_labels(fig.ax)
            canvas.draw()
            show_figure(fig)
            plt.title(title)
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
            plt.close('all')
 
        # ── Calcul vmin/vmax (z-score comme MAP_streamwise) ────────────
        v0 = [0 if isnan(v) else v for v in vect_vel_st]
        zs = np.abs(sc.stats.zscore(v0))
        vmax = vmin = np.mean(v0)
        for i, v in enumerate(vect_vel_st):
            if zs[i] < 3:
                if v > vmax:
                    vmax = v
                elif v < vmin:
                    vmin = v
        self.vmin = vmin * (1 - 0.1 * np.sign(vmin))
        self.vmax = vmax * (1 + 0.1 * np.sign(vmax))
 
        # Échelle quiver (diagnostic/cohérence)
        v_sec = np.asarray([
            self.transverse_velocity[i, j]
            for j in range(n_hor) for i in range(n_vert)
            if not np.isnan(self.streamwise_velocity[i, j])
        ], dtype=float)
        w_sec = np.asarray([
            self.vertical_velocity[i, j]
            for j in range(n_hor) for i in range(n_vert)
            if not np.isnan(self.streamwise_velocity[i, j])
        ], dtype=float)

        finite_sec = np.isfinite(v_sec) & np.isfinite(w_sec)
        sec_mag = np.sqrt(v_sec[finite_sec] ** 2 + w_sec[finite_sec] ** 2)
        depth_max = float(np.nanmax(depths_plt)) if np.any(np.isfinite(depths_plt)) else 1.0
        depth_max = max(depth_max, 1e-6)

        if quiver_scale_fixed is not None:
            self.quiver_scale = max(float(quiver_scale_fixed), 1e-6)
        elif sec_mag.size == 0:
            self.quiver_scale = 0.1
        elif quiver_scale_mode == 'mean':
            self.quiver_scale = round_it(float(np.nanmean(sec_mag)), 1)
        elif quiver_scale_mode == 'classic95':
            q95 = float(np.nanquantile(sec_mag, 0.95))
            q95 = max(0.05, q95)
            self.quiver_scale = round_it((q95 * 5.0) / depth_max, 2)
        else:
            raise ValueError("quiver_scale_mode must be 'classic95' or 'mean'")

        if print_velocity_diagnostics:
            def _stats(label, arr):
                arr = np.asarray(arr, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    print(f"[Vermeulen Diagnostic] {label}: no finite data")
                    return
                print(
                    f"[Vermeulen Diagnostic] {label}: "
                    f"min={np.nanmin(arr):+.3f}, "
                    f"q50={np.nanmedian(arr):+.3f}, "
                    f"q95={np.nanquantile(arr, 0.95):+.3f}, "
                    f"max={np.nanmax(arr):+.3f}, "
                    f"mean={np.nanmean(arr):+.3f}"
                )

            _stats("streamwise (m/s)", self.streamwise_velocity)
            _stats("transverse (m/s)", self.transverse_velocity)
            _stats("vertical (m/s)", self.vertical_velocity)
            _stats("secondary norm sqrt(v^2+w^2) (m/s)", sec_mag)
            print(
                f"[Vermeulen Diagnostic] quiver mode={quiver_scale_mode}, "
                f"fixed={quiver_scale_fixed}, scale={self.quiver_scale:.3f}, "
                f"depth_max={depth_max:.3f}"
            )
 
        # ── Figure 1 : vitesse streamwise ──────────────────────────────
        _make_figure(
            vect_vel_st, self.vmin, self.vmax,
            "Méthode Vermeulen — Streamwise velocity",
            "Streamwise velocity / Méthode Vermeulen (m/s)",
            "Images_ppt/Classic_Streamwise_verm.png",
            v_dir=vect_vel_tr,
            w_dir=vect_vel_w)
 
        # ── Figure 2 : nombre d'informations ───────────────────────────
        nb_max = np.nanmax(self.nb_vel) if self.nb_vel is not None else 1
        _make_figure(
            vect_nb_vel, 0, nb_max,
            "Nombre d'informations par cellule — Vermeulen",
            "Nombre d'informations par cellule",
            "Images_ppt/Info_cell_verm.png")
 
        # ── Figure 3 : erreur quadratique ──────────────────────────────
        r2_max = np.nanmax(self.r2) if self.r2 is not None else 1
        _make_figure(
            vect_r2, 0, r2_max,
            "Erreur quadratique — Vermeulen",
            "Erreur quadratique R²",
            "Images_ppt/r2_verm.png")
 
        # ── Figure 4 : écart-type résidu ───────────────────────────────
        rsig_max = np.nanmax(self.r_sig) if self.r_sig is not None else 1
        _make_figure(
            vect_rsig, 0, rsig_max,
            "Écart-type résidu — Vermeulen",
            "Écart-type σ (m/s)",
            "Images_ppt/r_sig_verm.png")
 
    @staticmethod
    def jet(x):
        """Convertit un scalaire [0,1] en couleur RGB selon la colormap jet."""
        if x < 0:
            return [0, 0, 0.5]
        elif x < 0.125:
            return [0, 0, 0.5 + 4 * x]
        elif x < 0.375:
            return [0, 4 * (x - 0.125), 1]
        elif x < 0.625:
            return [4 * (x - 0.375), 1, -4 * (x - 0.625)]
        elif x < 0.875:
            return [1, -4 * (x - 0.875), 0]
        elif isnan(x):
            return [1, 1, 1]
        elif x < 1:
            return [-4 * (x - 1.125), 0, 0]
        else:
            return [0.5, 0, 0]