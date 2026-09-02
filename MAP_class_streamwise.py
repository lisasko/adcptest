# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:12:55 2022

@author: blais
"""

# ========================================
# External imports
# ========================================
import copy
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from math import isnan
import os # inutilisé ? 

# ========================================
# Internal imports
# ========================================

from MiscLibs.common_functions import cart2pol, pol2cart, nan_greater
from UI.MplCanvas import MplCanvas
from common_functions import show_figure, interpolation, round_it


# ============================================================
# Classe principale
# ============================================================
 
class MAP_streamwise:
    """Multitransect Averaged Profile (MAP) generates an average profile of selected transects.

    Attributes
    ----------
    streamwise_velocity: np.array(float)
        MAPstreamwise velocity of each middle cell without extrapolation
    transverse_velocity: np.array(float)
        MAP transverse velocity of each middle cell without extrapolation
    vertical_velocity: np.array(float)
        MAPstreamwise velocity of each middle cell without extrapolation
    depths: np.array(float) 1D
        MAP depths
    extrap_streamwise_velocity: np.array(float)
        MAPstreamwise velocity with extrpolation on bottom/top part
    extrap_transverse_velocity: np.array(float)
        MAP transverse velocity with extrpolation on bottom/top part
    extrap_vertical_velocity: np.array(float)
        MAP vertical velocity with extrpolation on bottom/top part
    depth_cells_border: np.array(float)
        Depth borders of each MAP cell, last one of each vertical equal to vertical depth
    depth_cells_center: np.array(float)
        Depth center of each MAP cell
    borders_ens: np.array(float) 1D
        Borders of each MAP vertical
       
    left_distance/right_distance: float
        MAP edge distance from edge computation
    left_borders/right_borders: np.array(float) 1D
        Borders of each MAP vertical from edge computation
    left_coef/right_coef: float
        Shape coefficient of MAP edge
    left_streamwise_velocity/right_streamwise_velocity: np.array(float)
        MAPstreamwise velocity for edge cells
    left_transverse_velocity/right_transverse_velocity: np.array(float)
        MAP transverse velocity for edge cells
    left_vertical_velocity/right_vertical_velocity: np.array(float)
        MAP vertical velocity for edge cells
    left_cells_discharge/right_cells_discharge: np.array(float)
        MAP edge cells discharge
    left_discharge/right_discharge: float
        MAP edge total discharge
        
    total_discharge: float
        MAP total discharge with current parameters
    middle_cells_discharge: np.array(float)
        MAP discharge of each middle cell (with top/bottom extrapolation if selected)
    middle_discharge: float
        MAP middle discharge with top/bottom extrapolation (if selected)
    """

    def __init__(self):
        """Initialize class and instance variables."""

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

        self.total_discharge = None  # MAP total discharge with current parameters
        self.middle_cells_discharge = None  # MAP discharge of each middle cell (with top/bottom extrapolation if selected)
        self.middle_discharge = None  # MAP middle discharge with top/bottom extrapolation (if selected)
        
        self.vmin = None # Vitesse minimale de l'échelle de couleur
        self.vmax = None # Vitesse maximale de l'échelle de couleur
        self.quiver_scale = None # Echelle pour les vitesses secondaires
        
        self.info_cell = None # Nombre d'informations permettant de recomposer la vitesse pour une cellule
        self.nb_max = None # Nombre d'info maximal x3 pour l'échelle de couleur
        self.orig_start_edge = None  # original start edge per selected transect

    def populate_data(self, meas, nb_max, nbr_cell_hor, nbr_cell_vert, node_horizontal_user, node_vertical_user,
                      edge_constant=False, extrap_option=True, interp_option=True,
                      track_section=True, plot=False, name_meas='unknown', path_results=''):
        """Parameters
        ----------
        meas: Measurement
            Object of Measurement class
        node_horizontal_user: float
            Length of MAP cells, if None automatical value is selected
        node_vertical_user: float
            Depth of MAP cells, if None automatical value is selected
        edge_constant: bool
            Indicates if edge cells should all be the same size
        extrap_option: bool
            Indicates if top/bottom/edges extrapolation should be applied
        interp_option: bool
            Indicates if velocities interpolation should be applied
        track_section: bool
            Indicates if average cross-section should be computed on boat track (True) or 
            on mean velocities direction (False)
        plot: bool
            Indicates if graphics should be plotted and saved
        name_meas: str
            Indicates graphics name for save
        path_results: str
            Indicates the path to save graphics
            
        nb_max : int
            limite haute du nombre d'information par cellule pour l'affichage
        nbr_cell_hor : float
            nombre approximatif maximal de cellules valides sur une ligne de la section
        nbr_cell_vert : float
            nombre approximatif maximal de cellules valides sur une colonne de la section
        """
        # Get meas current parameters
        settings = meas.current_settings()
        navigation_reference = settings['NavRef']
        checked_transect_idx = meas.checked_transect_idx
        if all(deg == 0 for deg in meas.transects[checked_transect_idx[0]].sensors.heading_deg.internal.data):
            print("MAP failed : No compass available")
        else:
            # Get main data from selected transects
            x_raw_coordinates, y_raw_coordinates, w_vel_x, w_vel_y, w_vel_z, \
            depth_data, q_cells, orig_start_edge, invalid_data, cell_depth, \
            left_geometry, right_geometry, dmg = MAP_streamwise.collect_data(meas, navigation_reference, checked_transect_idx,
                                                                  interp_option)
            self.orig_start_edge = orig_start_edge

            # Compute coefficients of the average cross-section
            alpha, beta, direction_meas, q_ens = MAP_streamwise.compute_coef(x_raw_coordinates, y_raw_coordinates,
                                                                  w_vel_x, w_vel_y, q_cells, track_section)

            # Project raw coordinates on average cross-section
            x_projected, y_projected, acs_distance_raw = MAP_streamwise.project_transect(alpha, beta, x_raw_coordinates,
                                                                              y_raw_coordinates)

            # Compare bathymetry and translate transects on average cross-section if needed
            acs_distance = MAP_streamwise.translated_transects(acs_distance_raw, depth_data)

            # Define horizontal and vertical mesh
            transect = meas.transects[checked_transect_idx[0]]
            in_transect_idx = transect.in_transect_idx
            valid_depth = transect.w_vel.valid_data[0, :, in_transect_idx].T
                #definition du sommet de la cellule pour respecter le blanking de surface
            top_cell = np.nanmin(transect.depths.bt_depths.depth_cell_depth_m[valid_depth]) - transect.depths.bt_depths.depth_cell_size_m[0, 0] / 2

            borders_ens_raw, nodes_depth_raw = MAP_streamwise.compute_node_size(cell_depth, acs_distance,
                                                                     depth_data, node_horizontal_user,
                                                                     node_vertical_user, nbr_cell_hor, nbr_cell_vert,top_cell)
            # Compute transect median velocity on each mesh (North, East and vertical velocities) and depth on each vertical
            transects_node_x_velocity, transects_node_y_velocity, transects_node_vertical_velocity, \
            transects_node_depth, transects_nodes,info_cell = MAP_streamwise.compute_nodes_velocity(meas,navigation_reference,
                                                                               checked_transect_idx, w_vel_x, w_vel_y,
                                                                               w_vel_z,
                                                                               cell_depth, depth_data, acs_distance,
                                                                               borders_ens_raw, nodes_depth_raw,
                                                                               orig_start_edge)

            self.borders_ens_raw = borders_ens_raw
            self.nodes_depth_raw = nodes_depth_raw
        

            # Compute mesh mean value of selected transects
            MAP_x_velocity, MAP_y_velocity = self.compute_mean(transects_nodes, transects_node_x_velocity,
                                                               transects_node_y_velocity,
                                                               transects_node_vertical_velocity,
                                                               transects_node_depth, borders_ens_raw, nodes_depth_raw,info_cell)

            # Compute streamwise and transverse velocity
            self.compute_projection(MAP_x_velocity, MAP_y_velocity,direction_meas)

            # Compute edges parameters
            # if extrap_option:
            #     self.left_distance, self.left_coef = [x for x in left_geometry]
            #     self.right_distance, self.right_coef = [x for x in right_geometry]
            # else:
            #     self.left_distance = 0
            #     self.right_distance = 0

            self.left_distance, self.left_coef = [x for x in left_geometry]
            self.right_distance, self.right_coef = [x for x in right_geometry]

            if not extrap_option:
                self.left_distance = 0
                self.right_distance = 0

            self.extrap_option = extrap_option

            # Compute top/bottom extrapolation according QRevInt velocity exponent
            self.compute_extrap_velocity(nodes_depth_raw, settings, extrap_option)

            # Compute edge extrapolation
            left_direction, right_direction, left_area, right_area, left_mid_cells_x, right_mid_cells_x, \
            left_mid_cells_y, right_mid_cells_y = self.compute_edges(borders_ens_raw, direction_meas, settings,
                                                                     edge_constant)

            self.left_mid_cells_x = left_mid_cells_x
            self.right_mid_cells_x = right_mid_cells_x
            self.left_mid_cells_y = left_mid_cells_y
            self.right_mid_cells_y = right_mid_cells_y

            # Compute discharge
            self.compute_discharge(extrap_option, left_direction, right_direction,
                                   left_area, right_area)
            if plot:
                self.plot_projected_data(alpha, beta, x_raw_coordinates, y_raw_coordinates, x_projected, y_projected,
                                         path_results, name_meas, q_ens)
                self.plot_profile(borders_ens_raw, nodes_depth_raw, left_mid_cells_x, right_mid_cells_x,
                                  left_mid_cells_y, right_mid_cells_y, path_results, name_meas, nb_max)

    @staticmethod
    def collect_data(meas, navigation_reference, checked_transect_idx, interp_option):
        """ Collect data of valid position and depth for each selected transect
        
        Parameters
        ----------
        meas: Measurement
            Object of Measurement class
        navigation_reference: str
            Indicated selected navigation reference 'bt_vel' (Bottom trakc) or 'gga_vel' (GGA)
        checked_transect_idx: list
            List of selected transects
        interp_option: bool
            Indicates if interpolated velocities should be used
            
        Returns
        -------
        x_raw_coordinates: list(np.array(float))
            List of 1D arrays of East boat coordinates from selected transects
        y_raw_coordinates: list(np.array(float))
            List of 1D arrays of North boat coordinates from selected transects
        w_vel_x: list(np.array(float))
            List of arrays of cells East velocity from selected transects
        w_vel_y: list(np.array(float))
            List of arrays of cells North velocity from selected transects
        w_vel_z: list(np.array(float))
            List of arrays of cells vertical velocity from selected transects
        depth_data: list(np.array(float))
            List of 1D arrays of depth from selected transects
        q_cells: list(np.array(float))
            List of arrays of cells discharge from selected transects
        orig_start_edge: list(str)
            List of original starting edge of transect looking downstream (Left or Right) from selected transects
        invalid_data: list(np.array(bool))
            List of arrays of invalid cells
        cell_depth: list(np.array(float))
            Depth to centerline of depth cells in meters
        left_geometry/right_geometry: np.array(float)
            Array returning values of edge distance and shape coefficient
        """
        # Create empty lists to iterate
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
                # Reverse transects in ordred to start at 0 on left edge
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

            # else:  
            #     valid = ~np.isinf(transect.gps.gga_lon_ens_deg)*~np.isnan(transect.gps.gga_lon_ens_deg)
            #     x_raw_coordinates.append(transect.gps.gga_lon_ens_deg[valid])
            #     y_raw_coordinates.append(transect.gps.gga_lat_ens_deg[valid])
            #     depth_data.append(transect.depths.bt_depths.depth_processed_m[valid])
            #     ship_data = transect.boat_vel.compute_boat_track(transect, ref='bt_vel')
            #     dmg.append(ship_data['dmg_m'][valid])
            #     q_cells.append(meas.discharge[id_transect].middle_cells[:,valid])

            # Edges parameters 
            left_param[index_transect, :] = [meas.transects[id_transect].edges.left.distance_m,
                                             meas.discharge[id_transect].edge_coef('left', meas.transects[id_transect])]
            right_param[index_transect, :] = [meas.transects[id_transect].edges.right.distance_m,
                                              meas.discharge[id_transect].edge_coef('right',
                                                                                    meas.transects[id_transect])]
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
        """ Compute average cross-section
        
        Parameters
        ----------
        x_raw_coordinates: list(np.array(float))
            List of 1D arrays of East boat coordinates from selected transects
        y_raw_coordinates: list(np.array(float))
            List of 1D arrays of North boat coordinates from selected transects
        w_vel_x: list(np.array(float))
            List of arrays of cells East velocity from selected transects
        w_vel_y: list(np.array(float))
            List of arrays of cells North velocity from selected transects
        q_cells: list(np.array(float))
            List of arrays of cells discharge from selected transects
        track_section: bool
            Indicates if average cross-section should be computed on boat track (True) or mean velocity (False)
        
        Returns
        -------
        alpha: float
            Slope of the line which define the average cross_section
        beta: float
            y-intercept of the line which define the average cross_section
        direction_meas: float
            Direction normal to the average cross-section
        """

        func = {'velocity': lambda x, b: x + b,
                'track': lambda x, a, b: a * x + b}
        q_ens = [abs(np.nansum(l, axis=0) / np.nanmean(l)) for l in q_cells]
        # Compute average cross-section on boat track
        if track_section:
            coef_meas = np.tile(np.nan, (len(x_raw_coordinates), 2))
            for i in range(len(x_raw_coordinates)):
                WLS = LinearRegression()
                WLS.fit(x_raw_coordinates[i].reshape(-1, 1), y_raw_coordinates[i].reshape(-1, 1),
                        sample_weight=np.abs(q_ens[i]))
                # coef_meas[i, :] = [WLS.coef_, WLS.intercept_]

                coef_meas[i, 0] = float(np.ravel(WLS.coef_)[0])
                coef_meas[i, 1] = float(np.ravel(WLS.intercept_)[0])

            alpha = np.nanmedian(coef_meas[:, 0])
            beta = np.nanmedian(coef_meas[:, 1])

            direction_meas = np.arctan2(-1, alpha)

        # Compute average cross-section on mean velocity direction
        else:
            mean_x = list()
            mean_y = list()
            for i in range(len(w_vel_x)):
                mean_x.append(np.nansum(w_vel_x[i] * q_cells[i]) / np.nansum(q_cells[i]))
                mean_y.append(np.nansum(w_vel_y[i] * q_cells[i]) / np.nansum(q_cells[i]))
            v_x = np.nanmedian(mean_x)
            v_y = np.nanmedian(mean_y)
            direction_meas, _ = cart2pol(v_x, v_y)
            alpha = -1 / np.tan(direction_meas)
            coef_meas = list()
            for i in range(len(w_vel_x)):
                coef_meas.append(curve_fit(func['velocity'], alpha * x_raw_coordinates[i], y_raw_coordinates[i])[0])
            beta = np.nanmean(coef_meas)

        return alpha, beta, direction_meas, q_ens

    @staticmethod
    def project_transect(alpha, beta, x_raw_coordinates, y_raw_coordinates):
        """ Project transects on the average cross-section
        
        Parameters :
            alpha: float
                Slope of the line which define the average cross_section
            beta: float
                y-intercept of the line which define the average cross_section
            meas: Measurement
                Object of Measurement class
            x_raw_coordinates: list(np.array(float))
                List of 1D arrays of East boat coordinates from selected transects
            y_raw_coordinates: list(np.array(float))
                List of 1D arrays of North boat coordinates from selected transects
            navigation_reference: str
                Indicated selected navigation reference 'bt_vel' (Bottom trakc) or 'gga_vel' (GGA)
            
        Returns :
            x_projected: list(np.array(float))
                List of 1D arrays of the projection on the average cross-section of the East coordinates
            y_projected: list(np.array(float))
                List of 1D arrays of the projection on the average cross-section of the North coordinates
            acs_distance: list(np.array(float))
                List of 1D arrays to the left_most point distance on the average cross-section
        """
        x_projected = list()
        y_projected = list()
        distance_transect = list()
        x_left = list()
        y_left = list()

        # Project transect
        for i in range(len(x_raw_coordinates)):
            # Project x and y coordinates on the cross-section
            x_projected.append((x_raw_coordinates[i] - alpha * beta + alpha * y_raw_coordinates[i]) / (alpha ** 2 + 1))
            y_projected.append(
                (beta + alpha * x_raw_coordinates[i] + alpha ** 2 * y_raw_coordinates[i]) / (alpha ** 2 + 1))

            x_left.append(x_projected[i][0])
            y_left.append(y_projected[i][0])

            # Boundaries on x and y coordinates for the selected transects
            x_boundaries = [min([min(l) for l in x_projected]), max([max(l) for l in x_projected])]
            y_boundaries = [min([min(l) for l in y_projected]), max([max(l) for l in y_projected])]

            x_start = min(x_boundaries, key=lambda x: abs(x - np.nanmedian(x_left)))
            y_start = min(y_boundaries, key=lambda x: abs(x - np.nanmedian(y_left)))

        # Compute distance on the average cross-section
        acs_distance = list()

        # x_distance = x_projected - min(x_boundaries, key=abs)
        # y_distance = y_projected - min(y_boundaries, key=abs)
        # for i in range(len(x_projected)):
        #     acs_distance.append(np.sqrt(x_distance[i] ** 2 + y_distance[i] ** 2))

        x_ref = min(x_boundaries, key=abs)
        y_ref = min(y_boundaries, key=abs)
        x_distance = [xp - x_ref for xp in x_projected]
        y_distance = [yp - y_ref for yp in y_projected]
        for i in range(len(x_projected)):
            acs_distance.append(np.sqrt(x_distance[i] ** 2 + y_distance[i] ** 2))

        return x_projected, y_projected, acs_distance

    @staticmethod
    def translated_transects(acs_distance_raw, depth_data):
        """ Compare bathymetry and translate transects on average cross-section if needed
        
        Parameters
        ----------
        acs_distance_raw: list(np.array(float))
            List of 1D arrays of the distance on the average cross-section
        depth_data: list(np.array(float))
            List of 1D arrays of bathymetry from selected transects
       
        Returns
        -------
        acs_translated: list(np.array(float))
            List of 1D arrays of the corrected distance depending on the bathymetry
        """
        acs_translated = list()
        # Use the first transect as reference and define an homogeneous x grid across measurement
        acs_translated.append(copy.deepcopy(acs_distance_raw[0]))
        max_acs = max([max(l) for l in acs_distance_raw])
        min_acs = min([min(l) for l in acs_distance_raw])
        grid_acs = np.arange(min_acs, max_acs, (max_acs - min_acs) / 100)
        acs_depth_ref = sc.interpolate.griddata(acs_distance_raw[0], depth_data[0], grid_acs)

        for i in range(1, len(acs_distance_raw)):
            # Interpolate depth on x grid
            acs_depth = sc.interpolate.griddata(acs_distance_raw[i], depth_data[i], grid_acs)
            if not np.all(np.isnan(acs_depth)):
                first_valid_acs = int(np.argwhere(~np.isnan(acs_depth))[0])
                last_valid_acs = int(np.argwhere(~np.isnan(acs_depth))[-1])
                valid_acs_depth = last_valid_acs - first_valid_acs + 1
                acs_depth_valid = acs_depth[first_valid_acs:last_valid_acs + 1]

                # Find the best position to minimize median of sum square
                default_acs_ssd = np.nanmean((acs_depth_ref[:valid_acs_depth] - acs_depth_valid) ** 2)
                lag_acs_idx = 0
                for j in range(1, len(acs_depth_ref) - valid_acs_depth + 1):
                    acs_ssd = np.nanmean((acs_depth_ref[j:j + valid_acs_depth] - acs_depth_valid) ** 2)
                    if (acs_ssd < default_acs_ssd or np.isnan(default_acs_ssd)) and \
                            np.count_nonzero(np.isnan(acs_depth_ref[j:j + valid_acs_depth])) < 0.5 * valid_acs_depth:
                        default_acs_ssd = acs_ssd
                        lag_acs_idx = j
                # Correct average cross-section distance by translation
                acs_corrected = acs_distance_raw[i] - (grid_acs[first_valid_acs] - grid_acs[lag_acs_idx])
                acs_translated.append(acs_corrected)
            else:
                acs_translated.append(acs_distance_raw[i])

        # Start from 0
        min_dist = min([min(l) for l in acs_translated])
        for i in range(len(acs_translated)):
            acs_translated[i] -= min_dist

        return acs_translated

    @staticmethod
    def compute_node_size(cell_depth, acs_distance, depth_data, node_horizontal_user,
                          node_vertical_user, nbr_cell_hor, nbr_cell_vert,top_cell):
        """ Define horizontal and vertical mesh
        
        Parameters
        ----------
        cell_depth: list(np.array(float))
            Depth to centerline of depth cells in meters
        acs_distance: list(np.array(float))
            List of 1D arrays of the corrected distance on the average cross-section
        depth_data: list(np.array(float))
            List of 1D arrays of bathymetry from selected transects
        node_horizontal_user: float
            Horizontal size of the mesh define by the user
        node_vertical_user: float
            Vertical size of the mesh define by the user
        top_cell : float
            profondeur de la première cellule
        nbr_cell_hor : float
        nbr_cell_vert : float
            
        
            
        Returns
        -------
        borders_ens: list(float)
            Horizontal grid length on the average cross-section
        nodes_depth: list(float)
            Vertical grid depth from the free-surface
        """
        # Define number of meshs on the average cross-section
        # acs_total = max([max(l) for l in acs_distance]) - min([min(l) for l in acs_distance])
        # if node_horizontal_user == None:
        #     # all_acs_distance = np.array([item for subarray in acs_distance for item in subarray])
        #     # all_acs_distance.sort()
        #     # acs_diff = abs(all_acs_distance[1:] - all_acs_distance[:-1])
            
        #     node_horz = np.nanmedian([np.quantile(l[1:] - l[:-1], 0.95) for l in acs_distance])
        # else:
        #     node_horz = node_horizontal_user

        # Divise total length in same size meshs
        # TODO see logspace
        
        borders_ens = np.linspace(min([min(l) for l in acs_distance]),
                                  max([max(l) for l in acs_distance]) + 10 ** -5,
                                  int(nbr_cell_hor))   # Pour avoir environ les mêmes tailles de cellules que Matlab 80

        # Define mesh vertical size
        all_depth = np.array([item for subarray in depth_data for item in subarray])
        if node_vertical_user:
            nodes_depth = np.round(np.arange(top_cell, np.nanmax(all_depth) + 2 * node_vertical_user,node_vertical_user).tolist(), 3)
        else:
            lag = np.nanmax([0.1, 2 * (cell_depth[0][1, 0] - cell_depth[0][0, 0])])
            nodes_depth = np.arange(top_cell, np.nanmax(all_depth) + lag, (np.nanmax(all_depth) + lag)/(nbr_cell_vert*1.4)) #30
            
        print("Profondeur premier noeud = ", nodes_depth[0])

        return borders_ens, nodes_depth

    @staticmethod
    def compute_nodes_velocity(meas,navigation_reference, checked_transect_idx, w_vel_x, w_vel_y, w_vel_z,
                               cell_depth, depth_data, acs_distance, borders_ens_raw,
                               nodes_depth_raw, orig_start_edge):
        """ Compute transect median velocity on each mesh (North, East and vertical velocities) 
        and depth on each vertical
        
        Parameters
        ----------
        meas: Measurement
            Object of Measurement class
        navigation_reference: str
            Object TransectData
        checked_transect_idx: list(int)
            List of selected transects
        w_vel_x: list(np.array(float))
            List of arrays of cells East velocity from selected transects
        w_vel_y: list(np.array(float))
            List of arrays of cells North velocity from selected transects
        w_vel_z: list(np.array(float))
            List of arrays of cells vertical velocity from selected transects
        cell_depth:
            
        depth_data: list(np.array(float))
            List of 1D arrays of bathymetry from selected transects
        acs_distance: list(np.array(float))
            List of 1D arrays of the corrected distance on the average cross-section
        borders_ens_raw: list(float)
            Horizontal grid length on the average cross-section
        nodes_depth_raw: list(float)
            Vertical grid depth from the free-surface
        orig_start_edge: list(str)
            List of original starting edge of transect looking downstream (Left or Right) from 
            selected transects
            
        Returns
        -------
        borders_ens: list(float)
            Horizontal grid length on the average cross-section
        nodes_depth: list(float)
            Vertical grid depth from the free-surface
        """

        node_mid = (borders_ens_raw[1:] + borders_ens_raw[:-1]) / 2
        # Create list to save transects interpolated on mesh grid
        transects_nodes = list()
        transects_node_x_velocity = np.tile(np.nan,
                                            (len(checked_transect_idx), len(nodes_depth_raw) - 1, len(node_mid)))
        transects_node_y_velocity = np.tile(np.nan,
                                            (len(checked_transect_idx), len(nodes_depth_raw) - 1, len(node_mid)))
        transects_node_vertical_velocity = np.tile(np.nan,
                                                   (len(checked_transect_idx), len(nodes_depth_raw) - 1, len(node_mid)))
        transects_node_depth = np.tile(np.nan, (len(checked_transect_idx), len(node_mid)))
        
        info_cell = np.tile(np.nan,(len(checked_transect_idx), len(nodes_depth_raw) - 1, len(node_mid)))

        for id_transect in checked_transect_idx:
            index_transect = checked_transect_idx.index(id_transect)
            print('===== Transect ' + str(index_transect) + ' of ' + str(len(checked_transect_idx) - 1) + ' =====')
            w_vel_x_tr = w_vel_x[index_transect]
            w_vel_y_tr = w_vel_y[index_transect]
            w_vel_z_tr = w_vel_z[index_transect]
            cell_depth_tr = cell_depth[index_transect]
            depth_ens_tr = depth_data[index_transect]
            if orig_start_edge[index_transect] == 'Right':
                cell_depth_tr = np.flip(cell_depth_tr, axis=1)
                w_vel_x_tr = np.flip(w_vel_x_tr, axis=1)
                w_vel_y_tr = np.flip(w_vel_y_tr, axis=1)
                w_vel_z_tr = np.flip(w_vel_z_tr, axis=1)


            # Nombre d'informations par cellule de l'ADCP (ping)
            info_ping = meas.transects[index_transect].w_vel.valid_vel_sum[:,:]
            
            
            # Find the representative mesh of each transect's vertical
            lag_distance = borders_ens_raw[1] - borders_ens_raw[0]

            data = {'Index_ensemble': np.arange(len(acs_distance[index_transect])),
                    'Index_Node': [int(i) for i in acs_distance[index_transect] / lag_distance],
                    'Distance_ensemble': acs_distance[index_transect]}
            df = pd.DataFrame(data)
            # for i in range(len(acs_distance[index_transect])):
            #         df.loc[[i], ['Index_Node_orig', 'Distance_node']] = next(x for x in enumerate(borders_ens_raw[1:]) 
            #                                                             if x[1] >= df.loc[i]['Distance_ensemble'])   

            # Transect's nodes
            id_proj = list(np.unique(df['Index_Node']))
            transects_nodes.append(id_proj)

            # Run through nodes to determine each parameters
            for node in id_proj:
                index_node = np.array(df[df['Index_Node'] == node]['Index_ensemble'])
                w_vel_x_node = w_vel_x_tr[:, index_node]
                w_vel_y_node = w_vel_y_tr[:, index_node]
                w_vel_z_node = w_vel_z_tr[:, index_node]
                mid_cell_node = cell_depth_tr[:, index_node]
                depth_node = depth_ens_tr[index_node]
                info_ping_node = info_ping[:,index_node]
                

                transects_node_depth[index_transect, node] = np.nanmedian(depth_node)
                # Determine every transect's cells in the mesh
                for id_vert in range(len(nodes_depth_raw) - 1):
                    (id_x, id_y) = np.where(np.logical_and(mid_cell_node >= nodes_depth_raw[id_vert], \
                                                           mid_cell_node < nodes_depth_raw[id_vert + 1]))
                    w_vel_x_loc = w_vel_x_node[id_x, id_y]
                    w_vel_y_loc = w_vel_y_node[id_x, id_y]
                    w_vel_z_loc = w_vel_z_node[id_x, id_y]
                    info_ping_loc = info_ping_node[id_x, id_y]

                    transects_node_x_velocity[index_transect, id_vert, node] = np.nanmean(w_vel_x_loc)
                    transects_node_y_velocity[index_transect, id_vert, node] = np.nanmean(w_vel_y_loc)
                    transects_node_vertical_velocity[index_transect, id_vert, node] = np.nanmedian(w_vel_z_loc)
                    info_cell[index_transect, id_vert, node] = np.nansum(info_ping_loc)

        return transects_node_x_velocity, transects_node_y_velocity, transects_node_vertical_velocity, \
               transects_node_depth, transects_nodes,info_cell

    def compute_mean(self, transects_nodes, transects_node_x_velocity, transects_node_y_velocity,
                     transects_node_vertical_velocity, transects_node_depth, borders_ens_raw,
                     nodes_depth_raw,info_cell):
        """ Compute mesh mean value of selected transects
        
        Parameters
        ----------
        transects_nodes: np.array(np.array(flot))
            Array of 1D arrays of nodes detected by each transect
        transects_node_x_velocity: np.array(np.array(flot))
            Median East velocity on each mesh of each transect
        transects_node_y_velocity: np.array(np.array(flot))
            Median North velocity on each mesh of each transect
        transects_node_vertical_velocity: np.array(np.array(flot))
            Median vertical velocity on each mesh of each transect
       transects_node_depth: np.array(np.array(float))
            Array of 1D arrays of depth values on each vertical for each transects
        borders_ens_raw: list(float)
            Horizontal grid length on the average cross-section
        nodes_depth_raw: list(float)
            Vertical grid depth from the free-surface
            
        Returns
        -------
        MAP_x_velocity: np.array
            East velocity on each cell of the MAP section
        MAP_y_velocity: np.array
            North velocity on each cell of the MAP section
        """
        # Define meshs detected by enough transects
        unique_node = list(np.unique([x for l in transects_nodes for x in l]))
        node_selected = copy.deepcopy(unique_node)
        valid_cell = max(int(len(transects_nodes) / 3), 1)
        for node in unique_node:
            if sum(x.count(node) for x in transects_nodes) < valid_cell or \
                    np.isnan(transects_node_depth[:, node]).all():
                node_selected.remove(node)


        node_min = np.nanmin(node_selected)
        node_max = np.nanmax(node_selected)
        node_range = list(range(node_min, node_max + 1))
        node_range_border = list(range(node_min, node_max + 2))            
            
        borders_ens = borders_ens_raw[node_range_border]
        borders_ens -= min(borders_ens)

        MAP_depth_cells_border = np.tile(np.nan, (len(nodes_depth_raw), len(node_range)))
        for i in range(len(node_range)):
            MAP_depth_cells_border[:, i] = nodes_depth_raw

        MAP_x_velocity = np.tile(np.nan, (len(nodes_depth_raw) - 1, len(node_range)))
        MAP_y_velocity = np.tile(np.nan, (len(nodes_depth_raw) - 1, len(node_range)))
        MAP_vertical_velocity = np.tile(np.nan, (len(nodes_depth_raw) - 1, len(node_range)))
        MAP_depth = np.tile(np.nan, len(node_range))
        MAP_info_cell = np.tile(np.nan, (len(nodes_depth_raw) - 1, len(node_range)))

        for node in node_selected:
            index_node = node_range.index(node)
            row = np.array([j for (j, sub) in enumerate(transects_nodes) if node in sub])
            
            
            x_map_cell = transects_node_x_velocity[row, :, node]
            y_map_cell = transects_node_y_velocity[row, :, node]
            vertical_map_cell = transects_node_vertical_velocity[row, :, node]
            depth_map_cell = transects_node_depth[row, node]
            info_map_cell = info_cell[row, :, node]
            
            

            MAP_depth[index_node] = np.nanmean(depth_map_cell)

            # Cut values under streambed    
            if np.isnan(MAP_depth[index_node]):
                depth_limit = 0
            else:
                depth_limit = next(x[0] for x in enumerate(nodes_depth_raw) if x[1] > MAP_depth[index_node])

            x_map_cell[:, depth_limit:] = np.nan
            y_map_cell[:, depth_limit:] = np.nan
            vertical_map_cell[:, depth_limit:] = np.nan
            info_map_cell[:, depth_limit:] = np.nan

            # Cut value if not detected by enough transects
            # MAP_x_velocity[np.count_nonzero(~np.isnan(x_map_cell), axis = 0) > valid_cell,index_node] = np.nanmedian(x_map_cell[:, 
            #                 np.count_nonzero(~np.isnan(x_map_cell), axis = 0) > valid_cell], axis = 0)
            # MAP_y_velocity[np.count_nonzero(~np.isnan(y_map_cell), axis = 0) > valid_cell,index_node] = np.nanmedian(y_map_cell[:,
            #                 np.count_nonzero(~np.isnan(y_map_cell), axis = 0) > valid_cell], axis = 0)
            # MAP_vertical_velocity[np.count_nonzero(~np.isnan(vertical_map_cell), axis = 0) > valid_cell,index_node] = np.nanmedian(vertical_map_cell[:, 
            #                 np.count_nonzero(~np.isnan(vertical_map_cell), axis = 0) > valid_cell], axis = 0)

            MAP_x_velocity[:, index_node] = np.nanmean(x_map_cell, axis=0)
            MAP_y_velocity[:, index_node] = np.nanmean(y_map_cell, axis=0)
            MAP_vertical_velocity[:, index_node] = np.nanmean(vertical_map_cell, axis=0)
            MAP_depth_cells_border[depth_limit, index_node] = MAP_depth[index_node]
            MAP_depth_cells_border[depth_limit + 1:, index_node] = np.nan
            MAP_info_cell[:, index_node] = np.nansum(info_map_cell, axis=0)

        self.vertical_velocity = MAP_vertical_velocity
        self.depths = MAP_depth
        self.depth_cells_border = MAP_depth_cells_border
        self.borders_ens = borders_ens
        self.info_cell = MAP_info_cell

        return MAP_x_velocity, MAP_y_velocity

    
    
    def compute_projection(self, x_velocity, y_velocity,direction_meas):
        unit_vec_1, unit_vec_2 = pol2cart(direction_meas, 1)
        unit_vec = np.vstack([unit_vec_1, unit_vec_2])
        w_vel_st = np.tile([np.nan], x_velocity.shape)
        w_vel_tr = np.tile([np.nan], x_velocity.shape)
        for i in range(x_velocity.shape[0]):
            for j in range(x_velocity.shape[1]):
                w_vel_st[i, j] = np.sum(np.vstack([x_velocity[i,j], y_velocity[i,j]]) * unit_vec, 0)
                w_vel_tr[i, j] = unit_vec_2 * x_velocity[i,j] - unit_vec_1 * y_velocity[i,j]

        self.streamwise_velocity = -w_vel_st
        self.transverse_velocity = -w_vel_tr
        

    def compute_extrap_velocity(self, nodes_depth_raw, settings, extrap_option):
        """ Compute top/bottom extrapolation according QRevInt velocity exponent
        
        Parameters
        ----------
        nodes_depth_raw: list(float)
            Vertical grid depth from the free-surface
        settings: dict
            Measurement current settings
        extrap_option: bool
            Option to define if extrapolation should be applied
        Returns
        -------
        idx_bot: np.array(int)
            Index to the bottom most valid depth cell in each ensemble
        idx_top: np.array(int)
            Index to the top most valid depth cell in each ensemble
        """
        # Parameter to return nan in extrap value if extrapolation is not selected
        if extrap_option:
            units = 1
        else:
            units = np.nan

        depths = self.depths
        depth_cells_border = self.depth_cells_border
        w_vel_prim_extrap = np.copy(self.streamwise_velocity)
        w_vel_sec_extrap = np.copy(self.transverse_velocity)
        w_vel_z_extrap = np.copy(self.vertical_velocity)

        MAP_depth_cells_border = np.tile(np.nan, depth_cells_border.shape)
        for i in range(depth_cells_border.shape[1]):
            MAP_depth_cells_border[:, i] = nodes_depth_raw

        MAP_depth_cells_center = (MAP_depth_cells_border[1:, :] + MAP_depth_cells_border[:-1, :]) / 2

        # Blanking on the bottom
        blanking_depth = depths * 0.9
        for i in range(MAP_depth_cells_center.shape[1]):
            invalid = MAP_depth_cells_center[:, i] > blanking_depth[i]
            w_vel_prim_extrap[invalid, i] = np.nan
            w_vel_sec_extrap[invalid, i] = np.nan
            w_vel_z_extrap[invalid, i] = np.nan

        # Identify valid data      
        valid_data = np.logical_not(np.isnan(w_vel_prim_extrap))
        # Preallocate variables
        n_ensembles = valid_data.shape[1]
        idx_bot = np.tile(-1, (valid_data.shape[1])).astype(int)
        idx_top = np.tile(-1, valid_data.shape[1]).astype(int)

        for n in range(n_ensembles):
            # Identifying bottom most valid cell
            idx_temp = np.where(np.logical_not(np.isnan(w_vel_prim_extrap[:, n])))[0]
            if len(idx_temp) > 0:
                idx_top[n] = idx_temp[0]
                idx_bot[n] = idx_temp[-1]
            else:
                idx_top[n] = 0

        # QRevInt extrapolation method
        idx_bed = copy.deepcopy(idx_bot)
        bot_method = settings['extrapBot']
        top_method = settings['extrapTop']
        exponent = settings['extrapExp']

        depth_cells_center = (depth_cells_border[1:] + depth_cells_border[:-1]) / 2
        mid_bed_cells = depths - depth_cells_center

        if bot_method == 'Power':
            coef_streamwise_bot = np.nanmean(w_vel_prim_extrap, 0)
        elif bot_method == 'No Slip':
            cutoff_depth = 0.8 * depths
            depth_ok = (nan_greater(depth_cells_center, np.tile(cutoff_depth, (depth_cells_center.shape[0], 1))))
            component_ok = np.logical_not(np.isnan(w_vel_prim_extrap))
            use_ns = depth_ok * component_ok
            for j in range(len(idx_bot)):
                if idx_bot[j] >= 0:
                    use_ns[idx_bot[j], j] = 1
            component_ns = np.copy(w_vel_prim_extrap)
            component_ns[np.logical_not(use_ns)] = np.nan

            coef_streamwise_bot = np.nanmean(component_ns, 0)

        # Extrapolation Bot velocity
        for n in range(len(idx_bed)):
            if idx_bed[n] > -1:
                while idx_bed[n] < len(depth_cells_center[:, n]) and depth_cells_border[idx_bed[n] + 1, n] <= depths[n]:
                    idx_bed[n] += 1
                # Shape of bottom cells
                bot_depth = mid_bed_cells[idx_bot[n] + 1:(idx_bed[n]), n]

                # Extrapolation forstreamwise velocity
                bot_prim_value = coef_streamwise_bot[n] * ((1 + 1 / exponent) / (1 / exponent)) * (
                            (bot_depth / depths[n]) ** exponent)
                w_vel_prim_extrap[(idx_bot[n] + 1):(idx_bed[n]), n] = bot_prim_value * units

                # Constant extrapolation for transverse velocity
                w_vel_sec_extrap[(idx_bot[n] + 1):(idx_bed[n]), n] = w_vel_sec_extrap[idx_bot[n], n] * units
                # Linear extrapolation to 0 at streambed for Vertical velocity
                try:
                    w_vel_z_extrap[(idx_bot[n] + 1):(idx_bed[n]), n] = units * sc.interpolate.griddata(
                        np.append(mid_bed_cells[idx_bot[n], n], 0), np.append(self.vertical_velocity[idx_bot[n], n], 0),
                        mid_bed_cells[(idx_bot[n] + 1):(idx_bed[n]), n])
                except Exception:
                    pass

        # Top power extrapolation (streamwise)
        if top_method == 'Power':
            coef_streamwise_top = np.nanmean(w_vel_prim_extrap, 0)
            for n in range(len(idx_top)):
                top_depth = mid_bed_cells[:idx_top[n], n]
                top_prim_value = coef_streamwise_top[n] * ((1 + 1 / exponent) / (1 / exponent)) * (
                            (top_depth / depths[n]) ** exponent)
                w_vel_prim_extrap[:idx_top[n], n] = top_prim_value * units

        # Top constant extrapolation (streamwise)
        elif top_method == 'Constant':
            n_ensembles = len(idx_top)
            top_prim_value = np.tile([np.nan], n_ensembles)
            for n in range(n_ensembles):
                if idx_top[n] >= 0:
                    w_vel_prim_extrap[:idx_top[n], n] = w_vel_prim_extrap[idx_top[n], n] * units

        # Extrap top for second and vertical velocities
        for n in range(len(idx_top)):
            if idx_top[n] >= 0:
                w_vel_sec_extrap[:idx_top[n], n] = w_vel_sec_extrap[idx_top[n], n] * units
                try:
                    top_z_value = sc.interpolate.griddata(np.append(mid_bed_cells[idx_top[n], n], self.depths[n]),
                                                          np.append(w_vel_z_extrap[idx_top[n], n], 0),
                                                          mid_bed_cells[:idx_top[n], n])
                    w_vel_z_extrap[:idx_top[n], n] = top_z_value * units
                except Exception:
                    pass

        self.depth_cells_center = depth_cells_center
        self.extrap_streamwise_velocity = w_vel_prim_extrap
        self.extrap_transverse_velocity = w_vel_sec_extrap
        self.extrap_vertical_velocity = w_vel_z_extrap


    def compute_edges(self, borders_ens_raw, mid_direction, settings, edge_constant):
        """ Compute edge extrapolation
        
        Parameters
        ----------
        borders_ens_raw: list(float)
            Horizontal grid length on the average cross-section
        mid_direction: np.array
            1D array of mean velocity direction of each MAP vertical
        settings: dict
            Measurement current settings
        edge_constant: bool
            Option to define if edge's meshs should share the exact same length or if they should be
            the same length as those in the middle (except the last vertical which is shorter)
        
        Returns
        -------
        left_direction/right_direction: np.array
            Direction of the first/last ensemble applied to edge
        left_area/right_area: np.array
            Area of edge's cells
        left_mid_cells_x/right_mid_cells_x: np.array
            Longitudinal position of the middle of each cell
        left_mid_cells_y/right_mid_cells_y: np.array
            Depth position of the middle of each cell
        """
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
        """ Compute edge extrapolation
        
        Parameters
        ----------
        edge_distance: float
            Edge distance
        edge_coef: float
            Shape coefficient of the edge
        edge: str
            'left' or 'right'
        borders_ens_raw: list(float)
            Horizontal grid length on the average cross-section
        mid_direction: np.array
            1D array of mean velocity direction of each MAP vertical
        extrap_exp: float
            QRevInt exponent power law value
        edge_constant: bool
            Option to define if edge's meshs should share the exact same length or if they should be
            the same length as those in the middle (except the last vertical which is shorter)
        
        Returns
        -------
        edge_streamwise_velocity: np.array
           streamwise velocity of cells in edge
        edge_transverse_velocity: np.array
            transverse velocity of cells in edge
        edge_vertical_velocity: np.array
            Vertical velocity of cells in edge
        nodes: np.array
            Length borders posiotion of ensemble
        edge_direction: np.array
            Direction of the first/last ensemble applied to edge
        mid_cells_x: np.array
            Longitudinal position of the middle of each cell
        mid_cells_y: np.array
            Depth position of the middle of each cell
        area: np.array
            Area of edge's cells
        """
        if edge == 'left':
            id_edge = 0
            node_size = abs(borders_ens_raw[1] - borders_ens_raw[0])
        elif edge == 'right':
            id_edge = -1
            node_size = abs(borders_ens_raw[-1] - borders_ens_raw[-2])

        if edge_constant:
            # Découpage de la berge
            nb_nodes = int(np.round(0.5 + edge_distance / node_size))
            nodes = np.linspace(0, edge_distance, nb_nodes + 1)  # Découpage x (bords)
        else:
            nodes = edge_distance - np.arange(0, edge_distance, node_size)[::-1]
            nodes = np.insert(nodes, 0, 0)
            nb_nodes = len(nodes) - 1

        nodes_mid = (nodes[1:] + nodes[:-1]) / 2  # Decoupage x
        edge_size_raw = self.depth_cells_border[:, id_edge]  # Decoupage y

        if edge_coef == 0.3535 and edge_distance > 0:
            # Depth arrays 
            border_depths = np.multiply(nodes, self.depths[
                id_edge] / edge_distance)  # Profondeur sur les bords de chaque verticale
            cells_borders_depths_1 = np.transpose(
                [edge_size_raw] * (len(border_depths) - 1))  # Position en profondeur des bords de chaque cellule
            cells_borders_depths_2 = np.transpose([edge_size_raw] * (len(border_depths)))

            for i in range(len(border_depths) - 1):
                sub_index = next(x[0] for x in enumerate(cells_borders_depths_1[:, i]) if
                                 x[1] >= int(1000 * border_depths[i + 1]) / 1000)
                cells_borders_depths_1[sub_index, i] = border_depths[i + 1]
                cells_borders_depths_1[sub_index + 1:, i] = np.nan
                cells_borders_depths_2[sub_index - 1, i + 1] = border_depths[i + 1]
                cells_borders_depths_2[sub_index:, i] = np.nan

            # Distance arrays
            cut_x = edge_distance * edge_size_raw[edge_size_raw <= self.depths[id_edge]] / self.depths[
                id_edge]  # Position couche coupe fond
            x_left = np.tile(nodes, (cells_borders_depths_1.shape[0], 1))

            for j in range(np.count_nonzero(~np.isnan(cut_x)) - 1):
                col, _ = next(x for x in enumerate(nodes) if x[1] > cut_x[j])
                row = np.where(edge_size_raw == edge_size_raw[j])[0][0]
                x_left[row, col - 1] = cut_x[j]
                x_left[row + 1:, col - 1] = nodes[col]

            # Cells serparate in 2 rectangles and 1 triangle 
            area_rec2 = (x_left[:-1, 1:] - x_left[1:, :-1]) * (
                        cells_borders_depths_1[1:, :] - cells_borders_depths_1[:-1, :])
            area_rec1 = (x_left[1:, :-1] - x_left[:-1, :-1]) * (
                        cells_borders_depths_2[:-1, :-1] - cells_borders_depths_1[:-1, :])
            area_tri1 = (x_left[1:, :-1] - x_left[:-1, :-1]) * (
                        cells_borders_depths_1[1:, :] - cells_borders_depths_2[:-1, :-1]) / 2
            area_tra1 = area_rec1 + area_tri1
            area = area_tra1 + area_rec2

            # Compute mid of every shape
            mid_rec2_x = (x_left[:-1, 1:] + x_left[1:, :-1]) / 2
            mid_rec2_y = (cells_borders_depths_1[:-1, :] + cells_borders_depths_1[1:, :]) / 2

            mid_rec1_x = (x_left[:-1, :-1] + x_left[1:, :-1]) / 2
            mid_rec1_y = (cells_borders_depths_1[:-1, :] + cells_borders_depths_2[:-1, :-1]) / 2

            mid_tri1_x = (x_left[:-1, :-1] + 2 * x_left[1:, :-1]) / 3
            mid_tri1_y = (2 * cells_borders_depths_2[:-1, :-1] + cells_borders_depths_1[1:, :]) / 3

            mid_tra1_x = (area_rec1 * mid_rec1_x + area_tri1 * mid_tri1_x) / (area_tra1)
            mid_tra1_y = (area_rec1 * mid_rec1_y + area_tri1 * mid_tri1_y) / (area_tra1)
            mid_tra1_x[area_tra1 == 0] = 0
            mid_tra1_y[area_tra1 == 0] = 0

            # Compute cell's mid
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
            # border_depths = np.tile([self.depths[id_edge]], nb_nodes+1)
            vertical_depth = np.tile([self.depths[id_edge]], area.shape)
            edge_exp = 10

            is_edge = True

        else:
            mid_cells_x = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            mid_cells_y = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))

            # size_x = np.tile([nodes[1:]-nodes[:-1]], (len(edge_size_raw)-1,1))
            # size_y = np.transpose([edge_size_raw[1:]-edge_size_raw[:-1]]*nb_nodes)
            area = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            bed_distance = np.tile([np.nan], (len(edge_size_raw) - 1, nb_nodes))
            # border_depths = np.tile([np.nan], (nb_nodes+1))
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
            ##streamwise velocity : Power_power extrapolation from first ensemble
            # Mean velocity on the first valid ensemble
            streamwise_mean_valid = np.nanmean(self.extrap_streamwise_velocity[:, id_edge])
            # Compute mean velocity according power law at middle_distance position
            vp_mean =streamwise_mean_valid * (mid_cells_x / (edge_distance)) ** (1 / edge_exp)
            # Compute velocity according power law on the chosen vertical
            edge_streamwise_velocity = vp_mean * ((extrap_exp + 1) / extrap_exp) * (
                        (vertical_depth - mid_cells_y) / vertical_depth) ** (1 / extrap_exp)

            ### Vertical velocity : linear extrapolation from first ensemble vertical distribution
            vertical_vel_first = np.insert(self.extrap_vertical_velocity[:, id_edge], 0, 0)
            vertical_vel_first = np.append(vertical_vel_first, 0)
            norm_depth_first = np.insert(self.depth_cells_center[:, id_edge] / self.depths[id_edge], 0, 0)
            norm_depth_first = np.append(norm_depth_first, 1)
            norm_depth_edge = mid_cells_y / vertical_depth
            edge_vertical_velocity = sc.interpolate.griddata(norm_depth_first, vertical_vel_first, norm_depth_edge)

            ## transverse velocity : linear interpolation to 0 at edge
            transverse_vel_first = np.insert(self.extrap_transverse_velocity[:, id_edge], 0,
                                            self.transverse_velocity[0, id_edge])
            transverse_vel_first = np.append(transverse_vel_first,
                                            transverse_vel_first[np.where(~np.isnan(transverse_vel_first))[-1][-1]])
            depth_first = np.insert(self.depth_cells_center[:, id_edge], 0, 0)
            depth_first = np.append(depth_first, self.depths[id_edge])
            edge_transverse_vel_interp = sc.interpolate.griddata(depth_first, transverse_vel_first, mid_cells_y)

            edge_transverse_velocity = np.tile(np.nan, edge_streamwise_velocity.shape)

            for j in range(edge_streamwise_velocity.shape[1]):
                for i in range(np.count_nonzero(~np.isnan(edge_transverse_vel_interp[:, j]))):
                    edge_transverse_velocity[i, j] = interpolation(np.array([bed_distance[i, j], edge_distance]),
                                                                  np.array([0, edge_transverse_vel_interp[i, j]]),
                                                                  mid_cells_x[i, j])

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
        """ Compute discharge
        
        Parameters
        ----------
        extrap_option: bool
            Option to define if extrapolation should be applied
        left_direction: str
            Direction of the first ensemble applied to edge
        right_direction: list(float)
            Direction of the last ensemble applied to edge
        left_area: np.array
            Left area of edge's cells
        right_area: float
            Right area of edge's cells
        """

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

    # =============================================================================
    #     PLOT
    # =============================================================================

    def plot_projected_data(self, alpha, beta, x_raw_coordinates, y_raw_coordinates, x_projected, y_projected,
                            path_results,
                            name_meas, q_ens):
        x_boundaries0 = [min([min(l) for l in x_projected]), max([max(l) for l in x_projected])]
        x_boundaries1 = [min([min(l) for l in x_raw_coordinates]), max([max(l) for l in x_raw_coordinates])]
        x_boundaries = [min([x_boundaries0[0], x_boundaries1[0]]), max([x_boundaries0[1], x_boundaries1[1]])]
        y_boundaries0 = [min([min(l) for l in y_projected]), max([max(l) for l in y_projected])]
        y_boundaries1 = [min([min(l) for l in y_raw_coordinates]), max([max(l) for l in y_raw_coordinates])]
        y_boundaries = [min([y_boundaries0[0], y_boundaries1[0]]), max([y_boundaries0[1], y_boundaries1[1]])]
        x_mean = np.nanmean(x_boundaries)
        y_mean = np.nanmean(y_boundaries)
        x2 = abs(x_boundaries[1] - x_boundaries[0]) / 2
        y2 = abs(y_boundaries[1] - y_boundaries[0]) / 2
        dist = np.nanmax([x2, y2])
        fig = plt.figure(figsize=(8, 6))
        plt.plot(x_boundaries, [alpha * l + beta for l in x_boundaries], color='firebrick', linewidth=2,
                 label='Average cross-section', zorder=2)
        for i in range(len(x_raw_coordinates)):
            # plt.scatter(x_raw_coordinates[i], y_raw_coordinates[i], s=q_ens[i], color='grey', linewidth=1)
            plt.plot(x_raw_coordinates[i], y_raw_coordinates[i], color='grey', linewidth=1)
        plt.plot(np.nan, np.nan, color='grey', linewidth=1, label='Transect boat track')
        plt.xlabel('Distance East (m)')
        plt.ylabel('Distance North (m)')
        plt.ylim(y_mean - dist - 0.5, y_mean + dist + 0.5)
        plt.xlim(x_mean - dist - 0.5, x_mean + dist + 0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc='upper left')
        #fig.savefig(path_results + '\\MAP_Average_cross_section_' + name_meas + '.png', dpi=240, bbox_inches='tight')
        fig.clear

    def plot_profile(self, borders_ens_raw, nodes_depth_raw, left_mid_cells_x, right_mid_cells_x,
                     left_mid_cells_y, right_mid_cells_y, path_results, name_meas, nb_max, plot_extrap=True):

        """ Affiche la cartographie des vitesses en créant des 'patches' pour pouvoir obtenir
            une sortie graphique similaire à ce que l'on peut faire à partir des excels
            fournis par Matlab
            On a besoin des coordonnées de chaque noeud du maillage pour créer les patches
        """
        left_distance = self.left_distance
        left_coef = self.left_coef
        right_coef = self.right_coef
        borders_ens = self.borders_ens

        # Define attributs
        plot_data = np.c_[self.left_streamwise_velocity, self.extrap_streamwise_velocity, self.right_streamwise_velocity]
        distance = left_distance + (borders_ens[1:] + borders_ens[:-1]) / 2
        x_axis = np.copy(distance)
        vertical_nodes = nodes_depth_raw

        depths_plt = np.copy(self.depths)
        if left_coef == 0.3535:
            depths_plt = np.insert(depths_plt, 0, [0, self.depths[0]])
            x_axis = np.insert(x_axis, 0, self.left_borders[[0, -1]])
        elif left_coef == 0.91:
            depths_plt = np.insert(depths_plt, 0, [0, self.depths[0], self.depths[0]])
            x_axis = np.insert(x_axis, 0, self.left_borders[[0, 0, -1]])
        else:
            depths_plt = np.insert(depths_plt, 0, [self.depths[0]])
            x_axis = np.insert(x_axis, 0, 0)

        if right_coef == 0.3535:
            depths_plt = np.append(depths_plt, [self.depths[-1], 0])
            x_axis = np.append(x_axis, self.left_distance + borders_ens[-1] + self.right_borders[[0, -1]])
        elif right_coef == 0.91:
            depths_plt = np.append(depths_plt, [self.depths[-1], self.depths[-1], 0])
            x_axis = np.append(x_axis, self.left_distance + borders_ens[-1] + self.right_borders[[0, -1, -1]])
        else:
            depths_plt = np.append(depths_plt, self.depths[-1])
            x_axis = np.append(x_axis, distance[-1] + (distance[-1] - distance[-2]) / 2)

        # x_axis = np.append(np.insert(distance, 0, self.left_borders),
        #                     self.right_borders+borders_ens[-1]+self.left_distance)
        depth_cells_border = self.depth_cells_border
        middle_cells = np.c_[left_mid_cells_y, (depth_cells_border[1:] + depth_cells_border[:-1]) / 2,
                             right_mid_cells_y]

        v = np.c_[self.left_transverse_velocity, np.c_[self.extrap_transverse_velocity, self.right_transverse_velocity]]
        w = np.c_[self.left_vertical_velocity, np.c_[self.extrap_vertical_velocity, self.right_vertical_velocity]]

        quiver_distance = np.tile([distance], (plot_data.shape[0], 1))
        quiver_distance = np.c_[left_mid_cells_x, quiver_distance, \
                                left_distance + borders_ens[-1] + right_mid_cells_x]

        quiver_scale = np.nanmax([0.05, np.nanquantile(np.sqrt(v ** 2 + w ** 2), 0.95)])
        self.quiver_scale = round_it((quiver_scale)*5/(np.nanmax(depths_plt)),2)
        
        

        mid_dist = np.append(np.insert(borders_ens[1:] + left_distance, 0, self.left_borders), self.right_borders[1:] +
                             borders_ens[-1] + left_distance)

        max_limit = 0

        x_plt = np.tile(np.nan, (2 * plot_data.shape[0], 2 * (plot_data.shape[1])))
        x_pand = np.array([val for val in mid_dist for _ in (0, 1)][1:-1])
        for n in range(len(x_pand)):
            x_plt[:, n] = x_pand[n]

        cell_plt = np.tile(np.nan, (2 * plot_data.shape[0], 2 * (plot_data.shape[1])))
        cell_pand = np.array([val for val in vertical_nodes for _ in (0, 1)][1:-1])
        for p in range(cell_pand.shape[0]):
            cell_plt[p, :] = cell_pand[p]

        speed_xpand = np.tile(np.nan, (plot_data.shape[0], 2 * (plot_data.shape[1])))

        for j in range(plot_data.shape[1]):
            speed_xpand[:, 2 * j] = plot_data[:, j]
            speed_xpand[:, 2 * j + 1] = plot_data[:, j]

        speed_plt = np.repeat(speed_xpand, 2, axis=0)

        # extrap_limit = self.vertical_nodes[self.idx_bot+1]
        min_limit = np.nanmin(speed_plt)
        if max_limit == 0:
            if np.sum(speed_plt[speed_plt > -900]) > 0:
                max_limit = np.percentile(speed_plt[speed_plt > -900], 99)
            else:
                max_limit = 1
        x_fill = np.insert(x_axis, 0, -1)
        x_fill = np.append(x_fill, x_fill[-1] + 1)
        depth_fill = np.insert(depths_plt, 0, 0)
        depth_fill = np.append(depth_fill, 0)

        # Main parameters
        plt.rcParams.update({'font.size': 14})
        
        
        
        ### Création du maillage
        
        patches = []
        list_poly_vertices = []
        vect_vel = []
        
        # On définit chaque maille en tant que polygone que l'on ajouter à la liste des patches
        
        for i in range(int(len(cell_plt[:,1])/2)):
            for j in range(int(len(cell_plt[1,:])/2)):
                list_poly_vertices.append([[x_plt[2*i,2*j],cell_plt[2*i,2*j]],[x_plt[2*i,2*j+1],cell_plt[2*i,2*j]],[x_plt[2*i,2*j+1],cell_plt[2*i+1,2*j]],[x_plt[2*i,2*j],cell_plt[2*i+1,2*j]]])
                vect_vel.append(speed_plt[2*i,2*j])
        for poly_vertices in list_poly_vertices:
            patches.append(Polygon(poly_vertices, closed=True))
            
        vect_vel_without_nan = [x for x in vect_vel if isnan(x) == False]    
        sorted_vel = sorted(vect_vel_without_nan)
        norm_vel = [i - sorted_vel[0] for i in vect_vel]/(sorted_vel[-1] - sorted_vel[0])
        poly_colors = []
        line_width = []
        
        for vel in norm_vel:
            poly_colors.append(self.jet(vel))
            if isnan(vel):
                line_width.append(0)
            else :
                line_width.append(1)

        patchColl = PatchCollection(patches,facecolors='white') #,edgecolors='black',linewidths = line_width) #ajouter pour afficher les mailles
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.fill_between(x_fill, np.nanmax(depths_plt) + 2, depth_fill, color='w')
        fig.ax.plot(x_axis, depths_plt, color='k', linewidth=1.5)
        fig.ax.plot(x_axis, [0 for i in depths_plt], color='b', linewidth=2)
        
        # Ajout des flèches au centre de chaque cellule
        
        middle_poly = [np.sum(i,0)/len(i) for i in list_poly_vertices]
        middle_poly_X = [i[0] for i in middle_poly]
        middle_poly_Y = [i[1] for i in middle_poly]
        
        q = fig.ax.quiver(middle_poly_X, middle_poly_Y, v, w, units='xy',
                          scale=self.quiver_scale,width=0.015 * np.nanmax(depths_plt))  # ,width=quiver_width)
        fig.ax.quiverkey(q, X=1, Y=-0.03, U=np.round(self.quiver_scale, 2), label=str(self.quiver_scale) + 'm/s',
                          labelpos='S', coordinates='axes', fontproperties={'size': 16})
        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(depths_plt) * 1.02)
        lower_limit = mid_dist[0] - 0.5
        upper_limit = mid_dist[-1] + 0.5
        
        
        # X_sc et Y_sc ne correspondent à rien, ils servent juste à afficher la colorbar
        X_sc = [i for i in range(len(norm_vel))]
        Y_sc= [i+30 for i in range(len(norm_vel))]
        
        
        #### On mets de côtés les valeurs aberrantes pour recentrer l'échelle de couleur
        # grâce à l'utilisation du 'zscore'
        
        v_vel_0 = [0 if isnan(x) else x for x in vect_vel]

        z_vel = np.abs(sc.stats.zscore(v_vel_0))


        vmax=np.mean(v_vel_0)
        vmin=np.mean(v_vel_0)
         
        for i in range(len(vect_vel)):
            if z_vel[i]<3:              ##Critère d'identification de valeurs aberrantes
                if vect_vel[i]>vmax:
                    vmax=vect_vel[i]
                elif vect_vel[i]<vmin:
                    vmin=vect_vel[i]
                    
        self.vmin = vmin*(1-0.1*np.sign(vmin))
        
        self.vmax = vmax*(1+0.1*np.sign(vmax))
        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=self.vmin, vmax=self.vmax))
        cb.ax.set_ylabel(canvas.tr('Streamwise velocity / Méthode Classique (m/s)'))
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        
        canvas.draw()
        show_figure(fig)
        
        plt.title("Méthode Classique")
        
        plt.savefig("Images_ppt/Classic_Streamwise.png",bbox_inches='tight',pad_inches=0.1)
        
        
        ############ Plot info par cellule ###############
        
        nb_vel =self.info_cell
        
        nb_vel = nb_vel.flatten()
        
        
        
        self.nb_max = nb_max
        
        norm_vel = nb_vel/self.nb_max
        poly_colors = []


        for vel in norm_vel:
            if vel > 0:
                poly_colors.append(self.jet(vel))
            else:
                poly_colors.append([1,1,1])

        patchColl = PatchCollection(patches,facecolors='white')
        patchColl.set_color(poly_colors)
        
        main_wt_contour_canvas = MplCanvas(width=15, height=9, dpi=240)

        canvas = main_wt_contour_canvas
        fig = canvas.fig
        # Configure axis
        fig.ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, bottom=0.2, right=1, top=0.97, wspace=0.1, hspace=0)

        fig.ax.add_collection(patchColl)
        fig.ax.autoscale()
        
        fig.ax.invert_yaxis()
        fig.ax.fill_between(x_fill, np.nanmax(depths_plt) + 2, depth_fill, color='w')
        fig.ax.plot(x_axis, depths_plt, color='k', linewidth=1.5)
        fig.ax.plot(x_axis, [0 for i in depths_plt], color='b', linewidth=2)
        

        fig.ax.set_xlabel(canvas.tr('Distance (m)'))
        fig.ax.set_ylabel(canvas.tr('Profondeur (m)'))
        fig.ax.xaxis.label.set_fontsize(18)
        fig.ax.yaxis.label.set_fontsize(18)
        fig.ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        fig.ax.set_ylim(top=-0.1, bottom=np.nanmax(depths_plt) * 1.02)
        lower_limit = mid_dist[0] - 0.5
        upper_limit = mid_dist[-1] + 0.5
        
        X_sc = [i+10 for i in range(len(norm_vel))]
        Y_sc= [i+100 for i in range(len(norm_vel))]
        

        
        cb = fig.colorbar(fig.ax.scatter(x=X_sc, y=Y_sc, c=vect_vel,cmap = 'jet', vmin=0, vmax=self.nb_max))
       
        cb.ax.set_ylabel(canvas.tr("Nombre d'informations par cellule"))                 
        cb.ax.yaxis.label.set_fontsize(16)
        cb.ax.tick_params(labelsize=16)
                        
        fig.ax.set_xlim(left=lower_limit, right=upper_limit)
        canvas.draw()
                        
        show_figure(fig)           
        plt.title("Nombre d'informations par cellule pour la méthode classique")
        

        plt.savefig('Images_ppt/Info_cell_classique.png',bbox_inches='tight',pad_inches=0.1)
        
    def jet(self,x):
        """ Transformer un nombre entre 0 et 1 en un vecteur RGB selon l'échelle de couleur "jet"
        Parameters
        ----------
        x : float
        Nombre compris entre 0 et 1 (idéalement)
        """
        if x<0:
            r = 0
            v = 0
            b = 0.5
        elif 0.125>x:
            r = 0
            v = 0
            b = 0.5+4*x
        elif 0.375>x>=0.125:
            r=0
            v=4*(x-0.125)
            b=1
        elif 0.625>x>=0.375:
            r=4*(x-0.375)
            v=1
            b=-4*(x-0.625)
        elif 0.875>x>=0.625:
            r=1
            v=-4*(x-0.875)
            b=0
        elif isnan(x):
            r=1
            v=1
            b=1
        elif 1>x>=0.875:
            r = -4*(x-1.125)
            b=0
            v=0
        else:
            r=0.5
            b=0
            v=0
        return [r,v,b]
        
