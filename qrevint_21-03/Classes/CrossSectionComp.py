import math
import os

import matplotlib.pyplot as plt
import numpy as np
import utm

# from qrev.Classes.BoatStructure import BoatStructure
from Classes.BoatStructure import BoatStructure

# ToDo: Add stats to show variability between transects ie Area and
# mean depth variance.


class CrossSectionComp(object):
    """Creates average cross-section.

    Attributes
    ----------
        cross_section: list
            Transect cross-sections stored as np.arrays.
        checked_idx: np.array
            Array of checked transect indices
        gps: bool
            True is GPS data available
        rec_spacing: float
            Recommended spacing for horizontal cross-section
        unproj_xs: list of np.array
            List of unprojected cross-sections

    """

    def __init__(self, transects, path=None):
        """Initiate attributes"""

        self.cross_section = []
        self.checked_idx = []
        self.gps = True
        self.rec_spacing = 0
        self.unproj_xs = []
        self.zone_number = None
        self.zone_letter = None

        self.create_cross_sections(transects)

        # Export CSV and PDF plots for testing
        if path is not None:
            self.export_csv(path)
            self.export_plots(path)

    def create_cross_sections(self, transects):
        """Create the axes and lines for the figure.

        Parameters
        ----------
        transects: TransectData
            Object of TransectData containing boat speeds

        """

        # Initialize variables
        self.cross_section = []
        self.checked_idx = []
        left_edge = []
        start_edge = []
        lon_list = []
        lat_list = []
        x_list = []
        y_list = []
        depth_list = []
        self.gps = True
        station_list = []

        # Process each transect
        for n, transect in enumerate(transects):
            if transect.checked:
                # self.checked_idx = np.append(checked_idx, n)
                self.checked_idx.append(n)

                # Compute boat track properties
                boat_track = BoatStructure.compute_boat_track(transect)

                if np.logical_not(np.all(np.isnan(boat_track["track_x_m"]))):
                    # get x/y boat track data
                    unit_x = boat_track["track_x_m"]
                    unit_y = boat_track["track_y_m"]

                    # if the start bank is right then flip the x/y coords.
                    if transect.start_edge == "Right":
                        unit_x = (np.amax(unit_x) - unit_x - (0 - np.amin(unit_x))) * -1
                        unit_y = (np.amax(unit_y) - unit_y - (0 - np.amin(unit_y))) * -1
                        edge = [transect.start_edge, transect.edges.left.distance_m]
                    else:
                        edge = [transect.start_edge, transect.edges.left.distance_m]

                    start_edge.append(edge)
                    x_list.append(unit_x)
                    y_list.append(unit_y)
                    left_edge.append(transect.edges.left.distance_m)

                    # Get selected depth object
                    depth = getattr(transect.depths, transect.depths.selected)
                    depth_a = np.copy(depth.depth_processed_m)
                    depth_a[np.isnan(depth_a)] = 0
                    depth_list.append(depth_a)

                    # pull GPS coords if available. If not fill with NANs
                    if hasattr(transect.gps, "gga_lon_ens_deg"):
                        if transect.boat_vel.selected == "vtg_vel" or "gga_vel":
                            try:
                                lon = transect.gps.gga_lon_ens_deg
                                lat = transect.gps.gga_lat_ens_deg

                                # replace nan values with 0 to avoid utm crash
                                lon_nan = np.argwhere(np.isnan(lon))
                                lat_nan = np.argwhere(np.isnan(lat))

                                if len(lon_nan) > 0:
                                    lon = np.nan_to_num(lon, nan=0)

                                if len(lat_nan) > 0:
                                    lat = np.nan_to_num(lat, nan=0)

                                # convert lat/lon to UTM coords.
                                lat_lon = utm.from_latlon(lat, lon)
                                lat = lat_lon[0]
                                lon = lat_lon[1]
                                self.zone_number = lat_lon[2]
                                self.zone_letter = lat_lon[3]

                                # replace 0 values with nan
                                if len(lat_nan) > 0:
                                    for idx in lat_nan:
                                        lat[idx] = np.nan
                                if len(lon_nan) > 0:
                                    for idx in lon_nan:
                                        lon[idx] = np.nan

                                self.gps = True
                                lon_list.append(lon)
                                lat_list.append(lat)

                            except (ValueError, TypeError):
                                lat, lon = self.create_empty_gps(unit_x)
                                self.gps = False
                                lon_list.append(lon)
                                lat_list.append(lat)
                        else:
                            lat, lon = self.create_empty_gps(unit_x)
                            self.gps = False
                            lon_list.append(lon)
                            lat_list.append(lat)

                    else:
                        lat, lon = self.create_empty_gps(unit_x)
                        self.gps = False
                        lon_list.append(lon)
                        lat_list.append(lat)

                    unprojected_xs = np.array([lon, lat, unit_x, unit_y, depth_a]).T
                    self.unproj_xs.append(unprojected_xs)

        if self.gps:
            (
                lon_list,
                lat_list,
                gps_slope,
                gps_intercept,
            ) = self.create_projected_cross_section(lon_list, lat_list)

        x_list, y_list, xy_slope, xy_intercept = self.create_projected_cross_section(
            x_list, y_list
        )

        x_list, y_list = self.adjust_xy_distances(x_list, y_list, xy_slope, start_edge)

        for xs in range(len(x_list)):
            station = np.sqrt(x_list[xs] ** 2 + y_list[xs] ** 2)
            station_list.append(station)

            xs = np.array(
                [
                    lon_list[xs],
                    lat_list[xs],
                    x_list[xs],
                    y_list[xs],
                    station_list[xs],
                    depth_list[xs],
                ]
            ).T

            self.cross_section.append(xs)

        # set recommended spacing
        self.compute_auto_spacing(station_list)

        # compute average cross section and append
        avg_cs = self.average_cross_section(self.cross_section)
        self.cross_section.append(avg_cs)

    @staticmethod
    def create_empty_gps(unit_x):
        array_size = unit_x.shape
        lon = np.empty(array_size)
        lon[:] = np.nan
        lat = np.empty(array_size)
        lat[:] = np.nan

        return lat, lon

    @staticmethod
    def create_projected_cross_section(x_list, y_list):
        """Computes projected ship track to convert transects to.

        Parameters
        ----------
        x_list: lst of np.arrays
            arrays are either long or x data.
        y_list: lst of np.arrays
            arrays are either lat or y data.

        Returns
        -------
        projected_x_list: lst of np.arrays
            arrays are either long or x data.
        projected_y_list: lst of np.arrays
            arrays are either lat or y data.

        """

        x_array = np.concatenate(x_list)
        y_array = np.concatenate(y_list)

        # remove nan values to avoid convergence crash
        idx = np.argwhere(np.isnan(x_array))
        x_array = np.delete(x_array, idx)
        y_array = np.delete(y_array, idx)

        idx = np.argwhere(np.isnan(y_array))
        x_array = np.delete(x_array, idx)
        y_array = np.delete(y_array, idx)

        # Find ranges and extremes
        x_w = np.amin(x_array)
        x_e = np.amax(x_array)
        y_s = np.amin(y_array)
        y_n = np.amax(y_array)

        x_rng = x_e - x_w
        y_rng = y_n - y_s

        # Use the Least squares polynomial fit
        if x_rng >= y_rng:
            model = np.polyfit(x_array, y_array, 1)
            # predict_function = np.poly1d(model)

            slope = model[0]
            intercept = model[1]

        else:
            model = np.polyfit(y_array, x_array, 1)
            # predict_function = np.poly1d(model)

            # slope and intercept in terms of x
            slope = 1 / model[0]
            intercept = -model[1] / model[0]

        # map the ensembles to the mean cross-section line
        projected_x_list = []
        projected_y_list = []
        for transect in range(len(x_list)):
            projected_x = (
                x_list[transect] - (slope * intercept) + (slope * y_list[transect])
            ) / (slope**2 + 1)

            projected_y = (
                intercept + (slope * x_list[transect]) + (slope**2 * y_list[transect])
            ) / (slope**2 + 1)

            projected_x_list.append(projected_x)
            projected_y_list.append(projected_y)

        return projected_x_list, projected_y_list, slope, intercept

    @staticmethod
    def adjust_xy_distances(x_list, y_list, slope, start_edge):
        """Adjusts projected x/y based on start edge distance.

        Parameters
        ----------
        x_list: lst of np.arrays
            arrays are either long or x data.
        y_list: lst of np.arrays
            arrays are either lat or y data.
        slope: float
            slope of projection
        start_edge: list
            Start edges and distances

        Returns
        -------
        projected_x_list: lst of np.arrays
                    arrays are either long or x data.
        projected_y_list: lst of np.arrays
                    arrays are either lat or y data.
        """

        theta = math.atan(slope)

        projected_x_list = []
        projected_y_list = []
        for transect in range(len(x_list)):
            # adjust the x/y lists using the left edge distance
            if start_edge[transect][0] == "Left":
                dist_x = start_edge[transect][1] * (math.cos(theta))
                dist_y = start_edge[transect][1] * (math.sin(theta))
                projected_x = x_list[transect] - dist_x
                projected_y = y_list[transect] - dist_y
            else:
                dist_x = start_edge[transect][1] * (math.cos(theta))
                dist_y = start_edge[transect][1] * (math.sin(theta))

                projected_x = x_list[transect] - dist_x
                projected_y = y_list[transect] - dist_y

            projected_x_list.append(projected_x)
            projected_y_list.append(projected_y)

        return projected_x_list, projected_y_list

    @staticmethod
    def create_gps_xs(stations, x_start, y_start, slope):
        """Adjusts projected x/y based on start edge distance.

        Parameters
        ----------
        stations: np.arrays
            mean cross-section.
        x_start: float
            starting point of transect on x-axis
        y_start: float
            starting point of transect on y-axis
        slope: float
            slope of projection

        Returns
        -------
        x_array: np.arrays
            mean cross-section x values
        y_array: np.arrays
            mean cross-section y values
        """

        # compute the geographic angle of the mean cross-section
        theta = math.atan(slope)

        # create x and y arrays
        x_array = (stations * (math.cos(theta))) + x_start
        y_array = (stations * (math.sin(theta))) + y_start

        return x_array, y_array

    def average_cross_section(self, cross_section, hor_spacing="Auto"):
        """Compute average cross-section.

        Parameters
        ----------
        cross_section: list of np.arrays
            list of cross-section
        hor_spacing: 'Auto' or float
            spacing for stationing. Defaults to 'Auto'

        Returns
        -------
        average_cs: np.array
            Coordinates of average cross-section

        """

        x_list = []
        y_list = []
        depth_list = []
        length_list = []
        start_list = []
        end_list = []
        lon_end_lst = []
        lat_end_lst = []
        lon_start_lst = []
        lat_start_lst = []

        # find max and min for the all transects to create mean xs
        for transect in cross_section:
            trans_end = np.max(np.array(transect[:, 4], dtype="f"), axis=0)
            trans_end_idx = np.argmax(np.array(transect[:, 4], dtype="f"), axis=0)

            trans_start = np.min(np.array(transect[:, 4], dtype="f"), axis=0)
            trans_start_idx = np.argmin(np.array(transect[:, 4], dtype="f"), axis=0)

            lat_start_lst.append(transect[:, 1][trans_start_idx])
            lon_start_lst.append(transect[:, 0][trans_start_idx])
            lat_end_lst.append(transect[:, 1][trans_end_idx])
            lon_end_lst.append(transect[:, 0][trans_end_idx])

            rows = transect.shape[0]

            start_list.append(trans_start)
            end_list.append(trans_end)
            length_list.append(rows)

        max_trans_start = np.min(start_list)
        idx = np.argmin(start_list)
        lat_start = lat_start_lst[idx]
        lon_start = lon_start_lst[idx]
        min_trans_end = np.max(end_list)
        idx = np.argmax(start_list)
        lat_end = lat_end_lst[idx]
        lon_end = lon_end_lst[idx]

        # set horizontal spacing. Default to 0.1m for xs widths > 10m.
        if hor_spacing == "Auto":
            spacing = self.rec_spacing
        else:
            spacing = hor_spacing

        # create mean cross-section stationing line
        station_line = np.arange(max_trans_start, min_trans_end, spacing)

        # create nan array
        num_pnts = station_line.shape[0]
        nan_array = np.empty(num_pnts)
        nan_array[:] = np.nan

        # create GPS array for mcs
        if not self.gps:
            lon_array = nan_array.copy()
            lat_array = nan_array.copy()
        else:
            lon_array = np.linspace(lon_start, lon_end, num_pnts)
            lat_array = np.linspace(lat_start, lat_end, num_pnts)

        for transect in cross_section:
            # sort and separate individual arrays and set d-type.
            sort_array = transect[transect[:, 4].argsort()].copy()

            station_array = np.array(sort_array[:, 4], dtype="f")
            depth_array = np.array(sort_array[:, 5], dtype="f")
            x_array = np.array(sort_array[:, 2], dtype="f")
            y_array = np.array(sort_array[:, 3], dtype="f")

            # get index for stations outside current transect
            station_max = station_array.max()
            station_min = station_array.min()
            greater = np.argwhere(station_line >= station_max)
            less = np.argwhere(station_line <= station_min)

            # Create stationing for transect
            one_m_station = np.interp(station_line, station_array, station_array)

            # create array of intervals and interpolate x/y and depth
            new_x = np.interp(one_m_station, station_array, x_array)
            new_y = np.interp(one_m_station, station_array, y_array)
            new_depth = np.interp(one_m_station, station_array, depth_array)

            # fill extents of transect with NANs as to avoid repeating values
            if len(greater) > 0:
                for idx in greater:
                    new_depth[idx] = np.nan
                    new_x[idx] = np.nan
                    new_y[idx] = np.nan
            if len(less) > 0:
                for idx in less:
                    new_depth[idx] = np.nan
                    new_x[idx] = np.nan
                    new_y[idx] = np.nan

            x_list.append(new_x)
            y_list.append(new_y)
            depth_list.append(new_depth)

        # create new arrays from list of arrays for each element
        x_array = np.array(x_list, dtype="f")
        y_array = np.array(y_list, dtype="f")
        depth_array = np.array(depth_list, dtype="f")

        # create mean array of each element
        depth_avg = np.nanmean(depth_array, axis=0)
        x_avg = np.nanmean(x_array, axis=0)
        y_avg = np.nanmean(y_array, axis=0)

        # create new array
        average_cs = np.array(
            [lon_array, lat_array, x_avg, y_avg, station_line, depth_avg]
        ).T

        return average_cs

    def compute_auto_spacing(self, station_list):
        """Compute recommended horizontal spacing.

        Parameters
        ----------
        station_list: list of np.arrays
            list of stations

        """

        stations = np.concatenate(station_list)

        # remove nan values to avoid convergence crash
        stations = stations[~np.isnan(stations)]

        station_dif = np.diff(stations)
        diff_med = np.median(station_dif)
        diff_std = np.std(station_dif)

        self.rec_spacing = diff_med + diff_std

    def export_csv(self, file_name):
        """Exports CSV file for each checked transect.

        Parameters
        ----------
        file_name: str
            path to save files
        """
        # Todo add comment lines at the top of the file for metadata.
        for n in range(len(self.cross_section)):
            if n == (len(self.cross_section) - 1):
                f_name = "cross_section_mean"
            else:
                f_name = "transect_" + str(self.checked_idx[n])

            path = file_name[:-8] + f_name + "_QRev" + ".csv"
            path = os.path.join(os.getcwd(), path)

            np.savetxt(path, self.cross_section[n], delimiter=",", fmt="%s")

    def export_plots(self, file_name):
        """Exports PDF file of cross-section plots. This method is used for
         troubleshooting.

        Parameters
        ----------
        file_name: str
            path to save files
        """

        mean_survey = self.cross_section[-1].T

        try:
            # XY plot
            fig = plt.figure()
            ax_1 = fig.add_subplot(1, 1, 1)

            for index, xs in enumerate(self.cross_section[:-1]):
                survey = xs.T

                ax_1.plot(survey[2], survey[3], ".", label=index)

            ax_1.plot(
                mean_survey[2], mean_survey[3], ".", label="Projected Cross-Section"
            )

            # set axis labels and legend
            ax_1.set_xlabel("X")
            ax_1.set_ylabel("Y")
            ax_1.legend()

            # save plot to PDF
            path = file_name[:-8] + "plots" + "_QRev" + ".pdf"
            path = os.path.join(os.getcwd(), path)
            fig.savefig(path)

        except BaseException:
            pass

        try:
            # mean depth plot
            fig_2 = plt.figure()
            ax_2 = fig_2.add_subplot(1, 1, 1)

            for index, xs in enumerate(self.cross_section[:-1]):
                survey = xs.T

                ax_2.plot(survey[4], survey[5], "-", label=index)

            ax_2.plot(
                mean_survey[4], mean_survey[5], "-", label="Average Cross-Section"
            )

            ax_2.set_xlabel("Station")
            ax_2.set_ylabel("Depth")
            ax_2.invert_yaxis()
            ax_2.legend()

            path_2 = file_name[:-8] + "plots_2" + "_QRev" + ".pdf"
            path_2 = os.path.join(os.getcwd(), path_2)
            fig_2.savefig(path_2)

        except BaseException:
            pass

        try:
            # mean GPS plot
            fig_3 = plt.figure()
            ax_3 = fig_3.add_subplot(1, 1, 1)

            for index, xs in enumerate(self.cross_section[:-1]):
                survey = xs.T

                ax_3.plot(survey[0], survey[1], ".", label=index)

            ax_3.plot(
                mean_survey[0], mean_survey[1], ".", label="Projected Cross-Section"
            )

            ax_3.set_xlabel("Long UTM")
            ax_3.set_ylabel("Lat UTM")
            ax_3.legend()

            path_3 = file_name[:-8] + "plots_3" + "_QRev" + ".pdf"
            path_3 = os.path.join(os.getcwd(), path_3)
            fig_3.savefig(path_3)

        except BaseException:
            pass

        try:
            # unprojected XY with mean xs
            fig_4 = plt.figure()
            ax_4 = fig_4.add_subplot(1, 1, 1)
            for index, xs in enumerate(self.unproj_xs):
                survey = xs.T

                ax_4.plot(survey[2], survey[3], "-", label=index)

            ax_4.plot(
                mean_survey[2], mean_survey[3], "-", label="Projected Cross-Section"
            )

            ax_4.set_xlabel("X")
            ax_4.set_ylabel("Y")
            ax_4.legend()

            path_4 = file_name[:-8] + "plots_4" + "_QRev" + ".pdf"
            path_4 = os.path.join(os.getcwd(), path_4)
            fig_4.savefig(path_4)

        except BaseException:
            pass

        try:
            # unprojected GPS with mean xs
            fig_5 = plt.figure()
            ax_5 = fig_5.add_subplot(1, 1, 1)
            for index, xs in enumerate(self.unproj_xs):
                survey = xs.T

                ax_5.plot(survey[0], survey[1], "-", label=index)

            ax_5.plot(
                mean_survey[0], mean_survey[1], "-", label="Projected Cross-Section"
            )

            ax_5.set_xlabel("Long UTM")
            ax_5.set_ylabel("Lat UTM")
            ax_5.legend()

            path_5 = file_name[:-8] + "plots_5" + "_QRev" + ".pdf"
            path_5 = os.path.join(os.getcwd(), path_5)
            fig_5.savefig(path_5)

        except BaseException:
            pass
