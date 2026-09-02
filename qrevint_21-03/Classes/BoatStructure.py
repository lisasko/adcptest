import numpy as np
from Classes.BoatData import BoatData


class BoatStructure(object):
    """This class organizes the various sources for boat velocity into
    a single structured class and establishes a selected property that
    contains the select source for velocity and discharge computations.

    Attributes
    ----------
    selected: str
        Name of BoatData object to be used for discharge computations.
    bt_vel: BoatData
        BoatData object for bottom track velocity
    gga_vel: BoatData
        BoatData object for gga velocity
    vtg_vel: BoatData
        BoatData object for vtg velocity
    composite: str
        Setting to use ("On") or not ("Off") composite tracks.
    """

    def __init__(self):

        self.selected = None  # Name of BoatData object to be used for discharge computations
        self.bt_vel = None  # BoatData object for bottom track velocity
        self.gga_vel = None  # BoatData object for gga velocity
        self.vtg_vel = None  # BoatData object for vtg velocity

        # Composite track information is not currently provided by the manufacturers.
        # Future versions may try to determine this setting from SonTek data
        self.composite = 'Off'  # Setting for compositir tracks

    def add_boat_object(self, source, vel_in, freq_in=None, coord_sys_in=None, nav_ref_in=None,
                        min_beams=3, bottom_mode='Variable'):
        """Adds a BoatData object to the appropriate property

        Parameters
        ----------
        source: str
            Name of manufacturer.
        vel_in: np.array
            Boat velocity array.
        freq_in: np.array or float
            Acoustic frequency
        coord_sys_in: str
            Coordinate system of velocity data.
        nav_ref_in: str
            Source of boat velocity data
        min_beams: int
            Setting to allow 3 beam solutions or require 4 beam solutions or set to Auto (-1)
        bottom_mode: str
            Bottom mode used
        """

        if nav_ref_in == 'BT':
            self.bt_vel = BoatData()
            self.bt_vel.populate_data(source, vel_in, freq_in, coord_sys_in, nav_ref_in, min_beams, bottom_mode)
        if nav_ref_in == 'GGA':
            self.gga_vel = BoatData()
            self.gga_vel.populate_data(source, vel_in, freq_in, coord_sys_in, nav_ref_in)
        if nav_ref_in == 'VTG':
            self.vtg_vel = BoatData()
            self.vtg_vel.populate_data(source, vel_in, freq_in, coord_sys_in, nav_ref_in)

    def set_nav_reference(self, reference):
        """This function will set the navigation reference property to the specified object reference.

        Parameters
        ----------
        reference: str
            Navigation reference, BT, GGA, or VTG
        """

        if reference == 'BT':
            self.selected = 'bt_vel'
        elif reference == 'GGA':
            self.selected = 'gga_vel'
        elif reference == 'VTG':
            self.selected = 'vtg_vel'

    def change_nav_reference(self, reference, transect):
        """This function changes the navigation reference to the specified object reference and recomputes
        the composite tracks, if necessary.

        Parameters
        ----------
        reference: str
            New navigation reference, BT, GGA, or VTG.
        transect: TransectData
            Object of TransectData.
        """

        if reference == 'BT':
            self.selected = 'bt_vel'
        elif reference == 'GGA':
            self.selected = 'gga_vel'
        elif reference == 'VTG':
            self.selected = 'vtg_vel'
        elif reference == 'bt_vel':
            self.selected = 'bt_vel'
        elif reference == 'gga_vel':
            self.selected = 'gga_vel'
        elif reference == 'vtg_vel':
            self.selected = 'vtg_vel'

        self.composite_tracks(transect)

    def change_coord_sys(self, new_coord_sys, sensors, adcp):
        """This function will change the coordinate system of the boat velocity reference.
        
        Parameters
        ----------
        new_coord_sys: str
            Specified new coordinate system.
        sensors: Sensors
            Object of Sensors.
        adcp: InstrumentData
            Object of InstrumentData.
        """

        # Change coordinate system for all available boat velocity sources
        if self.bt_vel is not None:
            self.bt_vel.change_coord_sys(new_coord_sys, sensors, adcp)
        if self.gga_vel is not None:
            self.gga_vel.change_coord_sys(new_coord_sys, sensors, adcp)
        if self.vtg_vel is not None:
            self.vtg_vel.change_coord_sys(new_coord_sys, sensors, adcp)

    def composite_tracks(self, transect, setting=None):
        """If new composite setting is provided it is used, if not the setting saved in the object is used
        
        Parameters
        ----------
        transect: TransectData
            Object of TransectData.
        setting: bool
            New setting for composite tracks
        """

        if setting is None:
            setting = self.composite
        else:
            # New Setting
            self.composite = setting

        # Composite depths turned on
        if setting == 'On':
            # Initialize variables
            u_bt = np.array([])
            v_bt = np.array([])
            u_gga = np.array([])
            v_gga = np.array([])
            u_vtg = np.array([])
            v_vtg = np.array([])

            # Prepare bt data
            if self.bt_vel is not None:
                u_bt = self.bt_vel.u_processed_mps
                v_bt = self.bt_vel.v_processed_mps
                # Set to invalid all interpolated velocities
                valid_bt = self.bt_vel.valid_data[0, :]
                u_bt[valid_bt == False] = np.nan
                v_bt[valid_bt == False] = np.nan

            # Prepare gga data
            if self.gga_vel is not None:
                # Get gga velocities
                u_gga = self.gga_vel.u_processed_mps
                v_gga = self.gga_vel.v_processed_mps
                # Set to invalid all interpolated velocities
                valid_gga = self.gga_vel.valid_data[0, :]
                u_gga[valid_gga == False] = np.nan
                v_gga[valid_gga == False] = np.nan
            elif self.bt_vel is not None:
                u_gga = np.tile([np.nan], u_bt.shape)
                v_gga = np.tile([np.nan], v_bt.shape)

            # Prepare vtg data
            if self.vtg_vel is not None:
                # Get vtg velocities
                u_vtg = self.vtg_vel.u_processed_mps
                v_vtg = self.vtg_vel.v_processed_mps
                # Set to invalid all interpolated velocities
                valid_vtg = self.vtg_vel.valid_data[0, :]
                u_vtg[valid_vtg == False] = np.nan
                v_vtg[valid_vtg == False] = np.nan
            elif self.bt_vel is not None:
                u_vtg = np.tile([np.nan], u_bt.shape)
                v_vtg = np.tile([np.nan], v_bt.shape)

            # Process bt as primary
            if self.selected == 'bt_vel':
                # Initialize composite source
                comp_source = np.tile(np.nan, u_bt.shape)

                # Process u velocity component
                u_comp = u_bt
                comp_source[np.isnan(u_comp) == False] = 1

                # If BT data are not valid try VTG and set composite source (BUG HERE DSM)
                u_comp[np.isnan(u_comp)] = u_vtg[np.isnan(u_comp)]
                comp_source[np.logical_and(np.isnan(u_comp) == False, np.isnan(comp_source))] = 3

                # If there are still invalid boat velocities, try GGA and set composite source
                u_comp[np.isnan(u_comp)] = u_gga[np.isnan(u_comp)]
                comp_source[np.logical_and(np.isnan(u_comp) == False, np.isnan(comp_source))] = 2

                # If there are still invalid boat velocities, use interpolated
                # values if present and set composite source
                comp_source[np.logical_and(np.isnan(u_comp) == False, np.isnan(comp_source))] = 0

                # Set composite source to invalid for all remaining invalid boat velocity data
                comp_source[np.isnan(comp_source)] = -1

                # Process v velocity component.  Assume that the composite source is the same
                # as the u component
                v_comp = v_bt
                v_comp[np.isnan(v_comp)] = v_vtg[np.isnan(v_comp)]
                v_comp[np.isnan(v_comp)] = v_gga[np.isnan(v_comp)]
                v_comp[np.isnan(v_comp)] = self.bt_vel.v_processed_mps[np.isnan(v_comp)]

                # Apply the composite settings to the bottom track Boatdata objects
                self.bt_vel.apply_composite(u_comp, v_comp, comp_source)
                self.bt_vel.interpolate_composite(transect)

            # Process gga as primary
            elif self.selected == 'gga_vel':
                # Initialize the composite source
                comp_source = np.tile([np.nan], u_bt.shape)

                # Process the u velocity component
                u_comp = u_gga
                comp_source[np.isnan(u_comp) == False] = 2

                # If GGA data are not valid try VTG and set composite source
                u_comp[np.isnan(u_comp)] = u_vtg[np.isnan(u_comp)]
                comp_source[np.logical_and(np.isnan(u_comp) == False, np.isnan(comp_source))] = 3

                # If there are still invalid boar velocities, try BT and set composite source
                u_comp[np.isnan(u_comp)] = u_bt[np.isnan(u_comp)]
                comp_source[np.logical_and(np.isnan(u_comp) == False, np.isnan(comp_source))] = 1

                # If there are still invalid boat velocities, use interpolated values,
                # if present and set composite source
                comp_source[np.logical_and(np.isnan(u_comp) == False, np.isnan(comp_source))] = 0

                # Set composite source to invalid for all remaining invalid boat velocity data
                comp_source[np.isnan(comp_source)] = -1

                # Process v velocity component.  Assume that the composite source is the
                # same as the u component
                v_comp = v_gga
                v_comp[np.isnan(v_comp)] = v_vtg[np.isnan(v_comp)]
                v_comp[np.isnan(v_comp)] = v_bt[np.isnan(v_comp)]
                # v_comp[np.isnan(v_comp)] = self.gga_vel.v_processed_mps[np.isnan(v_comp)]

                # Apply the composite settings to the gga BoatData object
                # For the situation where the transect has no GGA data but other transects do and composite tracks
                # has been turned on, create the gga_vel object and populate only the u and v processed, comp_source,
                # and valid_data attributes.
                if self.gga_vel is None:
                    self.gga_vel = BoatData()
                    self.gga_vel.processed_source = np.array([''] * comp_source.shape[0], dtype=object)
                    self.gga_vel.valid_data = np.full((6, comp_source.shape[0]), False)
                self.gga_vel.apply_composite(u_comp, v_comp, comp_source)
                self.gga_vel.interpolate_composite(transect)

            # Process vtg as primary
            elif self.selected == 'vtg_vel':
                # Initialize the composite source
                comp_source = np.tile([np.nan], u_bt.shape)

                # Process the u velocity component
                u_comp = u_vtg
                comp_source[np.isnan(u_comp) == False] = 3

                # If VTG data are not valid try GGA and set composite source
                u_comp[np.isnan(u_comp)] = u_gga[np.isnan(u_comp)]
                comp_source[np.logical_and(np.isnan(u_comp) == False, np.isnan(comp_source))] = 2

                # If there are still invalid boat velocities, try BT and set composite source
                u_comp[np.isnan(u_comp)] = u_bt[np.isnan(u_comp)]
                comp_source[np.logical_and(np.isnan(u_comp) == False, np.isnan(comp_source))] = 1

                # If there are still invalid boat velocities, use interpolated values,
                # if present and set composite source
                comp_source[np.logical_and(np.isnan(u_comp) == False, np.isnan(comp_source))] = 0

                # Set composite source to invalid for all remaining invalid boat velocity data
                comp_source[np.isnan(comp_source)] = -1

                # Process v velocity component.  Assume that the composite source is the
                # same as the u component
                v_comp = v_vtg
                # DSM wrong in Matlab version 1/29/2018 v_comp[np.isnan(v_comp)] = v_vtg[np.isnan(v_comp)]
                v_comp[np.isnan(v_comp)] = v_gga[np.isnan(v_comp)]
                v_comp[np.isnan(v_comp)] = v_bt[np.isnan(v_comp)]
                # v_comp[np.isnan(v_comp)] = self.vtg_vel.v_processed_mps[np.isnan(v_comp)]

                # Apply the composite settings to the gga BoatData object
                # For the situation where the transect has no GGA data but other transects do and composite tracks
                # has been turned on, create the gga_vel object and populate only the u and v processed, comp_source,
                # and valid_data attributes.
                if self.vtg_vel is None:
                    self.vtg_vel = BoatData()
                    self.vtg_vel.processed_source = np.array([''] * comp_source.shape[0], dtype=object)
                    self.vtg_vel.valid_data = np.full((6, comp_source.shape[0]), False)
                self.vtg_vel.apply_composite(u_comp, v_comp, comp_source)
                self.vtg_vel.interpolate_composite(transect)
        else:
            # Composite tracks off

            # Use only interpolations for bt
            if self.bt_vel is not None:
                self.bt_vel.apply_interpolation(transect=transect,
                                                interpolation_method=transect.boat_vel.bt_vel.interpolate)
                comp_source = np.tile(np.nan, self.bt_vel.u_processed_mps.shape)
                comp_source[self.bt_vel.valid_data[0, :]] = 1
                comp_source[np.logical_and(np.isnan(comp_source),
                                           (np.isnan(self.bt_vel.u_processed_mps) == False))] = 0
                comp_source[np.isnan(comp_source)] = -1
                self.bt_vel.apply_composite(u_composite=self.bt_vel.u_processed_mps,
                                            v_composite=self.bt_vel.v_processed_mps,
                                            composite_source=comp_source)

            # Use only interpolations for gga
            if self.gga_vel is not None:
                # This if statement handles the situation where there is no GPS data for a transect but there is GPS
                # data for other transects and the user has turned on / off composite tracks.
                if self.gga_vel.u_mps is not None:
                    self.gga_vel.apply_interpolation(transect=transect,
                                                     interpolation_method=transect.boat_vel.gga_vel.interpolate)
                    comp_source = np.tile(np.nan, self.gga_vel.u_processed_mps.shape)
                    comp_source[self.gga_vel.valid_data[0, :]] = 2
                    comp_source[np.logical_and(np.isnan(comp_source),
                                               (np.isnan(self.gga_vel.u_processed_mps) == False))] = 0
                    comp_source[np.isnan(comp_source)] = -1
                    self.gga_vel.apply_composite(u_composite=self.gga_vel.u_processed_mps,
                                                 v_composite=self.gga_vel.v_processed_mps,
                                                 composite_source=comp_source)
                else:
                    self.gga_vel = None

            # Use only interpolations for vtg
            if self.vtg_vel is not None:
                # This if statement handles the situation where there is no GPS data for a transect but there is GPS
                # data for other transects and the user has turned on / off composite tracks.
                if self.vtg_vel.u_mps is not None:
                    self.vtg_vel.apply_interpolation(transect=transect,
                                                     interpolation_method=transect.boat_vel.vtg_vel.interpolate)
                    comp_source = np.tile(np.nan, self.vtg_vel.u_processed_mps.shape)
                    comp_source[self.vtg_vel.valid_data[0, :]] = 3
                    comp_source[np.logical_and(np.isnan(comp_source),
                                               (np.isnan(self.vtg_vel.u_processed_mps) == False))] = 0
                    comp_source[np.isnan(comp_source)] = -1
                    self.vtg_vel.apply_composite(u_composite=self.vtg_vel.u_processed_mps,
                                                 v_composite=self.vtg_vel.v_processed_mps,
                                                 composite_source=comp_source)
                else:
                    self.vtg_vel = None

    @staticmethod
    def compute_boat_track(transect, ref=None):
        """Computes the shiptrack coordinates, along track distance, and distance made
        good for the selected boat reference.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        ref: str
            Setting to determine what navigation reference should be used. In None use selected.

        Returns
        -------
        boat_track: dict
            Dictionary containing shiptrack coordinates (track_x_m, track_y_m), along track distance (distance_m),
            and distance made good (dmg_m)
        """

        # Initialize dictionary
        boat_track = {'track_x_m': np.nan, 'track_y_m': np.nan, 'distance_m': np.nan, 'dmg_m': np.nan}

        # Compute incremental track coordinates
        if ref is None:
            boat_vel_selected = getattr(transect.boat_vel, transect.boat_vel.selected)
        else:
            if ref == 'gps_vel':
                if getattr(transect.boat_vel, 'gga_vel') is not None:
                    ref_resolved = 'gga_vel'
                elif getattr(transect.boat_vel, 'vtg_vel') is not None:
                    ref_resolved = 'vtg_vel'
                else:
                    ref_resolved = 'bt_vel'
            else:
                ref_resolved = ref

            boat_vel_selected = getattr(transect.boat_vel, ref_resolved)

        if boat_vel_selected is None:
            boat_vel_selected = getattr(transect.boat_vel, 'bt_vel')
            
        track_x = boat_vel_selected.u_processed_mps[transect.in_transect_idx] * \
            transect.date_time.ens_duration_sec[transect.in_transect_idx]
        track_y = boat_vel_selected.v_processed_mps[transect.in_transect_idx] * \
            transect.date_time.ens_duration_sec[transect.in_transect_idx]

        # Check for any valid data
        idx = np.where(np.logical_not(np.isnan(track_x)))
        if idx[0].size > 1:
            # Compute variables
            boat_track['distance_m'] = np.nancumsum(np.sqrt(track_x ** 2 + track_y ** 2))
            boat_track['track_x_m'] = np.nancumsum(track_x)
            boat_track['track_y_m'] = np.nancumsum(track_y)
            boat_track['dmg_m'] = np.sqrt(boat_track['track_x_m'] ** 2 + boat_track['track_y_m'] ** 2)

        return boat_track

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        transect: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        if hasattr(transect, 'boatVel'):
            if hasattr(transect.boatVel, 'btVel'):
                if hasattr(transect.boatVel.btVel, 'u_mps'):
                    self.bt_vel = BoatData()
                    self.bt_vel.populate_from_qrev_mat(transect.boatVel.btVel)
            if hasattr(transect.boatVel, 'ggaVel'):
                if hasattr(transect.boatVel.ggaVel, 'u_mps'):
                    self.gga_vel = BoatData()
                    self.gga_vel.populate_from_qrev_mat(transect.boatVel.ggaVel)
            if hasattr(transect.boatVel, 'vtgVel'):
                if hasattr(transect.boatVel.vtgVel, 'u_mps'):
                    self.vtg_vel = BoatData()
                    self.vtg_vel.populate_from_qrev_mat(transect.boatVel.vtgVel)
            nav_dict = {'btVel':'bt_vel', 'bt_vel':'bt_vel',
                        'ggaVel':'gga_vel', 'gga_vel':'gga_vel',
                        'vtgVel':'vtg_vel', 'vtg_vel':'vtg_vel'}
            self.selected = nav_dict[transect.boatVel.selected]

