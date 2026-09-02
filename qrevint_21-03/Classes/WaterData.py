import copy
import numpy as np
from numpy.matlib import repmat
from scipy import interpolate
from Classes.BoatData import BoatData
from MiscLibs.common_functions import cart2pol, pol2cart, iqr
from MiscLibs.robust_loess import rloess
from MiscLibs.abba_2d_interpolation import abba_idw_interpolation


class WaterData(object):
    """Class to process and store water velocity data.

    Attributes
    ----------
    Original data provided to the class:
        raw_vel_mps: np.array(float)
            Contains the raw unfiltered velocity in m/s.  1st index 1-4 are beams 1,2,3,4 if beam or
            u,v,w,d if otherwise.
        frequency: np.array(float)
            Defines ADCP frequency used for velocity measurement, in kHz.
        orig_coord_sys: str
            Defines the original raw velocity coordinate system "Beam", "Inst", "Ship", "Earth".
        orig_nav_ref: str
            Defines the original taw data naviagation reference: "None", "BT", "GGA", "VTG".
        corr: np.array(float)
            Correlation values for WT, if available.
        rssi: np.array(float)
            Returned acoustic signal strength.
        rssi_units: str
            Units for returned acoustic signal strength: "Counts" "dB", "SNR".
        water_mode: str
            WaterMode for TRDI or 'Variable' for SonTek.
        blanking_distance_m: float
            Distance below transducer where data are marked invalid due to potential ringing.
        cells_above_sl: np.array(bool)
            Logical array of depth cells above sidelobe cutoff based on selected depth reference.
        cells_above_sl_bt: np.array(bool)
            Logical array of depth cells above the sidelobe cutoff based on BT
        sl_lag_effect_m: np.array(float)
            Side lobe distance due to lag and transmit length

    Data computed in this class:
        u_earth_no_ref_mps: np.array(float)
            Horizontal velocity in x-direction with no boat reference applied, in m/s.
        v_earth_no_ref_mps: np.array(float)
            Horizontal velocity in y-direction with no boat reference applied, in m/s.
        u_mps: np.array(float)
            Horizontal velocity in x-direction, earth coord, nav referenced, in m/s.
        v_mps: np.array(float)
            Horizontal velocity in y-direction, earth coord, nav referenced, in m/s.
        u_processed_mps: np.array(float)
            Horizontal velocity in x-direction, earth coord, nav ref, filtered, and interpolated.
        v_processed_mps: np.array(float)
            Horizontal veloctiy in y-direction, earth coord, nav ref, filtered, and interpolated.
        w_mps: np.array(float)
            Vertical velocity (+ up), in m/s.
        d_mps: np.array(float)
            Difference in vertical velocities compute from opposing beam pairs, in m/s.
        invalid_index: np.array(bool)
            Index of ensembles with no valid raw velocity data.
        num_invalid: float
            Estimated number of depth cells in ensembles with no valid raw velocity data.
        valid_data: np.array(float)
            3-D logical array of valid data
                Dim1 0 - composite
                Dim1 1 - original, cells above side lobe
                Dim1 2 - dfilter
                Dim1 3 - wfilter
                Dim1 4 - smoothFilter
                Dim1 5 - beamFilter
                Dim1 6 - excluded
                Dim1 7 - snrFilter
                Dim1 8 - validDepthFilter

    Processing settings:
        beam_filter: int
            Set 3 for 3-beam solutions, 4 for 4-beam solutions.
        d_filter: str
            Set difference velocity filter "On", "Off".
        d_filter_threshold: float
            Threshold for difference velocity filter.
        w_filter: str
            Set vertical velocity filter "On", "Off".
        w_filter_threshold: float
            Threshold for vertical velocity filter.
        excluded_dist_m: float
            Distance below transucer for which data are excluded or marked invalid, in m.
        orig_excluded_dist_m: float
            Original distance below transucer for which data are excluded or marked invalid, in m.
        smooth_filter: str
            Set filter based on smoothing function "On", "Off".
        smooth_speed: np.array(float)
            Smoothed mean water speed, in m/s.
        smooth_upper_limit: np.array(float)
            Smooth function upper limit of window, in m/s.
        smooth_lower_limit: np.array(float)
            Smooth funciton lower limit of window, in m/s.
        snr_filter: str
            Set SNR filter for SonTek data "On", "Off".
        snr_rng: np.array(float)
            Range of beam averaged SNR
        wt_depth_filter: np.array(bool)
            WT in ensembles with invalid depths are marked invalid.
        interpolate_ens: str
            Type of interpolation: "None", "TRDI", "Linear", 'abba'.
        interpolate_cells: str
            Type of cell interpolation: "None", "TRDI", "Linear", 'abba'
        coord_sys: str
            Defines the velocity coordinate system "Beam", "Inst", "Ship", "Earth"
        nav_ref: str
            Defines the navigation reference: "None", "BT", "GGA", "VTG"
        sl_cutoff_percent: float
            Percent cutoff defined by cos(angle)
        sl_cutoff_number: float
            User specified number of cells to cutoff from SonTek, not implemented, undefined
        sl_cutoff_type: str
            Type of cutoff method "Percent" or "Number".
    """

    def __init__(self):
        """Initialize instance variables.
        """

        # Data input to this class
        self.raw_vel_mps = None
        self.frequency = None
        self.orig_coord_sys = None
        self.orig_nav_ref = None
        self.corr = None
        self.rssi = None
        self.rssi_units = None
        self.water_mode = None
        self.blanking_distance_m = None
        self.cells_above_sl = None
        self.cells_above_sl_bt = None
        self.sl_lag_effect_m = None
        
        # Data computed in this class
        self.u_earth_no_ref_mps = None
        self.v_earth_no_ref_mps = None
        self.u_mps = None
        self.v_mps = None
        self.u_processed_mps = None
        self.v_processed_mps = None
        self.w_mps = None
        self.d_mps = None
        self.invalid_index = None
        self.num_invalid = []
        self.valid_data = None
        self.valid_vel_sum = None
                                
        # Settings
        self.beam_filter = None
        self.d_filter = None
        self.d_filter_threshold = None
        self.w_filter = None
        self.w_filter_threshold = None
        self.excluded_dist_m = None
        self.orig_excluded_dist_m = None
        self.smooth_filter = None
        self.smooth_speed = None
        self.smooth_upper_limit = None
        self.smooth_lower_limit = None
        self.snr_filter = 'Off'
        self.snr_rng = []
        self.wt_depth_filter = None
        self.interpolate_ens = None
        self.interpolate_cells = None
        self.coord_sys = None
        self.nav_ref = None
        self.sl_cutoff_percent = None
        self.sl_cutoff_number = None
        self.sl_cutoff_type = None
        self.sl_cutoff_m = None

    def populate_data(self, vel_in, freq_in, coord_sys_in, nav_ref_in, rssi_in, rssi_units_in,
                      excluded_dist_in, cells_above_sl_in, sl_cutoff_per_in, sl_cutoff_num_in,
                      sl_cutoff_type_in, sl_lag_effect_in, wm_in, blank_in, corr_in=None,
                      surface_vel_in=None, surface_rssi_in=None, surface_corr_in=None, sl_cutoff_m=None,
                      surface_num_cells_in=0):
        
        """Populates the variables with input, computed, or default values.

        Parameters
        ----------
        vel_in: np.array(float)
            Contains the raw unfiltered velocity data in m/s.
            Rows 1-4 are beams 1,2,3,4 if beam or u,v,w,d if otherwise.
        freq_in: np.array(float)
            Defines ADCP frequency used for velocity measurement.
        coord_sys_in: str
            Defines the original raw  velocity coordinate system "Beam", "Inst", "Ship", "Earth".
        nav_ref_in: str
            Defines the original raw data navigation reference: "None", "BT", "GGA", "VTG".
        rssi_in: np.array(float)
            Returned acoustic signal strength.
        rssi_units_in: str
            Units for returned acoustic signal strength: "Counts", "dB", "SNR".
        excluded_dist_in: float
            Distance below transducer for which data are excluded or marked invalid.
        cells_above_sl_in: np.array(bool)
            Bool array of depth cells above the sidelobe cutoff based on selected depth reference.
        sl_cutoff_per_in: float
            Percent cutoff defined by cos(angle).
        sl_cutoff_num_in: float
            User specified number of cells to cutoff above sl_cutoff.
        sl_cutoff_type_in: str
            Method used to compute cutoff "Percent" or "Number".
        sl_lag_effect_in: np.array(float)
            Lag effect for each ensemble, in m.
        wm_in: str
            Watermode for TRDI or 'Variable' for SonTek.
        blank_in: float
            Blanking distance, in m.
        corr_in: np.array(float)
            Correlation values for water track. Optional.
        surface_vel_in: np.array(float)
            Surface velocity data for RiverRay, RiverPro, RioPro. Optional.
        surface_rssi_in: np.array(float)
            Returned acoust signal strength for RiverRay, RiverPro, RioPro. Optional.
        surface_corr_in: np.array(float)
            Surface velocity correlations for RiverRay, RiverPro, RioPro. Optional.
        surface_num_cells_in: np.array(float)
            Number of surface cells in each ensemble for RiverRay, RiverPro, RioPro. Optional.
        sl_cutoff_m: np.array(float)
            Depth in meters of side lobe cutoff to center of cells.
        """
 
        # Set object properties from input data standard for all ADCPs
        self.frequency = freq_in
        self.orig_coord_sys = coord_sys_in
        self.coord_sys = coord_sys_in
        self.orig_nav_ref = nav_ref_in
        self.nav_ref = nav_ref_in
        self.water_mode = wm_in
        self.excluded_dist_m = excluded_dist_in
        self.rssi_units = rssi_units_in

        # Set object properties that depend on the presence or absence of surface cells
        if np.sum(surface_num_cells_in) > 0:
            surface_num_cells_in[np.isnan(surface_num_cells_in)] = 0
            max_cells = cells_above_sl_in.shape[0]
            num_ens = cells_above_sl_in.shape[1]
            num_reg_cells = vel_in.shape[1]
            max_surf_cells = max_cells - num_reg_cells

            # Combine surface velocity bins and regular velocity bins into one matrix
            self.raw_vel_mps = np.tile([np.nan], [4, max_cells, num_ens])
            self.rssi = np.tile([np.nan], [4, max_cells, num_ens])
            self.corr = np.tile([np.nan], [4, max_cells, num_ens])

            if max_surf_cells > 0:
                self.raw_vel_mps[:, :max_surf_cells, :] = surface_vel_in[:, :max_surf_cells, :]
                self.rssi[:, :max_surf_cells, :] = surface_rssi_in[:, :max_surf_cells, :]
                self.corr[:, :max_surf_cells, :] = surface_corr_in[:, :max_surf_cells, :]

            for i_ens in range(num_ens):
                self.raw_vel_mps[:,
                                 int(surface_num_cells_in[i_ens]):int(surface_num_cells_in[i_ens])
                                 + num_reg_cells, i_ens] = vel_in[:, :num_reg_cells, i_ens]
                self.rssi[:,
                          int(surface_num_cells_in[i_ens]):int(surface_num_cells_in[i_ens])
                          + num_reg_cells, i_ens] = rssi_in[:, :num_reg_cells, i_ens]
                self.corr[:,
                          int(surface_num_cells_in[i_ens]):int(surface_num_cells_in[i_ens])
                          + num_reg_cells, i_ens] = corr_in[:, :num_reg_cells, i_ens]
        else:
            # No surface cells
            self.raw_vel_mps = vel_in
            self.rssi = rssi_in
            if corr_in.any():
                self.corr = corr_in
            else:
                # No correlations input
                self.corr = np.tile(np.nan, rssi_in.shape)

        self.u_mps = np.copy(self.raw_vel_mps)[0, :, :]
        self.v_mps = np.copy(self.raw_vel_mps)[1, :, :]
        self.w_mps = np.copy(self.raw_vel_mps)[2, :, :]
        self.d_mps = np.copy(self.raw_vel_mps)[3, :, :]

        self.water_mode = wm_in
        self.excluded_dist_m = excluded_dist_in
        self.orig_excluded_dist_m = excluded_dist_in

        # In some rare situations the blank is empty so it is set to the excluded_dist_in
        try:
            blank_in = float(blank_in)
            self.blanking_distance_m = blank_in
        except ValueError:
            self.blanking_distance_m = excluded_dist_in
            
        self.cells_above_sl = cells_above_sl_in
        self.cells_above_sl_bt = cells_above_sl_in
        self.sl_cutoff_percent = sl_cutoff_per_in
        self.sl_cutoff_number = sl_cutoff_num_in
        self.sl_cutoff_type = sl_cutoff_type_in
        self.sl_lag_effect_m = sl_lag_effect_in
        self.sl_cutoff_m = sl_cutoff_m
        
        # Set filter defaults to no filtering and no interruption
        self.beam_filter = 3
        self.d_filter = 'Off'
        self.d_filter_threshold = 99
        self.w_filter = 'Off'
        self.w_filter_threshold = 99
        self.smooth_filter = False
        self.interpolate_ens = 'None'
        self.interpolate_cells = 'None'
        
        # Determine original valid

        # Initialize valid data property
        self.valid_data = np.tile(self.cells_above_sl, [9, 1, 1])
        
        # Find invalid raw data
        valid_vel = np.tile(self.cells_above_sl, [4, 1, 1])
        valid_vel[np.isnan(self.raw_vel_mps)] = False
        

            
        # Identify invalid velocity data (less than 3 valid beams)
        valid_vel_sum = np.sum(valid_vel, axis=0)
        valid_data2 = np.copy(self.cells_above_sl)
        valid_data2[valid_vel_sum < 3] = False
        
        self.valid_vel_sum = valid_vel_sum
        self.valid_vel_sum[valid_vel_sum < 3] = 0
        

        
        # Set valid_data property for original data
        self.valid_data[1, :, :] = valid_data2
        
        # Combine all filter data to composite valid data
        self.all_valid_data()
        
        # Estimate the number of cells in invalid ensembles using
        # Adjacent valid ensembles
        valid_data_2_sum = np.nansum(self.valid_data[1], 0)
        self.invalid_index = np.where(valid_data_2_sum == 0)[0]
        n_invalid = len(self.invalid_index)
        for n in range(n_invalid):
            # Find first valid ensemble
            idx1 = np.where(valid_data_2_sum[:self.invalid_index[n]] > 0)[0]
            if len(idx1) > 0:
                idx1 = idx1[0]
            else:
                idx1 = self.invalid_index[n]
                
            # Find next valid ensemble
            idx2 = np.where(valid_data_2_sum[:self.invalid_index[n]] > 0)[0]
            if len(idx2) > 0:
                idx2 = idx2[-1]
            else:
                idx2 = self.invalid_index[n]
                
            # Estimate number of cells in invalid ensemble
            self.num_invalid.append(np.floor((valid_data_2_sum[idx1]+valid_data_2_sum[idx2]) / 2))
            
        # Set processed data to non-interpolated valid data
        self.u_processed_mps = np.copy(self.u_mps)
        self.v_processed_mps = np.copy(self.v_mps)
        self.u_processed_mps[self.valid_data[0] == False] = np.nan
        self.v_processed_mps[self.valid_data[0] == False] = np.nan
        
        # Compute SNR range if SNR data is provided
        if rssi_units_in == 'SNR':
            self.compute_snr_rng()

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        transect: mat_struct
            Matlab data structure obtained from sio.loadmat
        """

        # Data requiring manipulation (special case for 1 ensemble)
        if len(transect.wVel.rawVel_mps.shape) == 2:
            self.raw_vel_mps = np.moveaxis(transect.wVel.rawVel_mps, 1, 0)
            self.raw_vel_mps = self.raw_vel_mps.reshape(self.raw_vel_mps.shape[0], self.raw_vel_mps.shape[1], 1)
            self.corr = np.moveaxis(transect.wVel.corr, 1, 0)
            self.corr = self.corr.reshape(self.corr.shape[0], self.corr.shape[1], 1)
            self.rssi = np.moveaxis(transect.wVel.rssi, 1, 0)
            self.rssi = self.rssi.reshape(self.rssi.shape[0], self.rssi.shape[1], 1)
            self.valid_data = np.moveaxis(transect.wVel.validData, 1, 0)
            self.valid_data = self.valid_data.reshape(self.valid_data.shape[0], self.valid_data.shape[1], 1)
            self.u_earth_no_ref_mps = transect.wVel.uEarthNoRef_mps
            self.u_earth_no_ref_mps = self.u_earth_no_ref_mps.reshape(self.u_earth_no_ref_mps.shape[0], 1)
            self.v_earth_no_ref_mps = transect.wVel.vEarthNoRef_mps
            self.v_earth_no_ref_mps = self.v_earth_no_ref_mps.reshape(self.v_earth_no_ref_mps.shape[0], 1)
            self.u_mps = transect.wVel.u_mps
            self.u_mps = self.u_mps.reshape(self.u_mps.shape[0], 1)
            self.v_mps = transect.wVel.v_mps
            self.v_mps = self.v_mps.reshape(self.v_mps.shape[0], 1)
            self.u_processed_mps = transect.wVel.uProcessed_mps
            self.u_processed_mps = self.u_processed_mps.reshape(self.u_processed_mps.shape[0], 1)
            self.v_processed_mps = transect.wVel.vProcessed_mps
            self.v_processed_mps = self.v_processed_mps.reshape(self.v_processed_mps.shape[0], 1)
            self.w_mps = transect.wVel.w_mps
            self.w_mps = self.w_mps.reshape(self.w_mps.shape[0], 1)
            self.d_mps = transect.wVel.d_mps
            self.d_mps = self.d_mps.reshape(self.d_mps.shape[0], 1)
            self.snr_rng = transect.wVel.snrRng
            self.snr_rng = self.snr_rng.reshape(self.snr_rng.shape[0], 1)
            self.cells_above_sl = transect.wVel.cellsAboveSL.astype(bool)
            self.cells_above_sl = self.cells_above_sl.reshape(self.cells_above_sl.shape[0], 1)
            self.cells_above_sl_bt = transect.wVel.cellsAboveSLbt.astype(bool)
            self.cells_above_sl_bt = self.cells_above_sl_bt.reshape(self.cells_above_sl_bt.shape[0], 1)
            self.sl_lag_effect_m = np.array([transect.wVel.slLagEffect_m])

        else:
            self.raw_vel_mps = np.moveaxis(transect.wVel.rawVel_mps, 2, 0)
            self.corr = np.moveaxis(transect.wVel.corr, 2, 0)
            self.rssi = np.moveaxis(transect.wVel.rssi, 2, 0)
            self.valid_data = np.moveaxis(transect.wVel.validData, 2, 0)
            self.u_earth_no_ref_mps = transect.wVel.uEarthNoRef_mps
            self.v_earth_no_ref_mps = transect.wVel.vEarthNoRef_mps
            self.u_mps = transect.wVel.u_mps
            self.v_mps = transect.wVel.v_mps
            self.u_processed_mps = transect.wVel.uProcessed_mps
            self.v_processed_mps = transect.wVel.vProcessed_mps
            self.w_mps = transect.wVel.w_mps
            self.d_mps = transect.wVel.d_mps
            self.snr_rng = transect.wVel.snrRng
            self.cells_above_sl = transect.wVel.cellsAboveSL.astype(bool)
            self.cells_above_sl_bt = transect.wVel.cellsAboveSLbt.astype(bool)
            self.sl_lag_effect_m = transect.wVel.slLagEffect_m

        self.valid_data = self.valid_data.astype(bool)
        # Fix for moving-bed transects that did not have 3D array indices adjusted properly when saved
        # if self.valid_data.shape[0] == self.u_processed_mps.shape[1]:
        #     self.valid_data = np.moveaxis(self.valid_data, 0, 2)
        #     self.raw_vel_mps = np.moveaxis(self.raw_vel_mps, 0, 2)
        #     self.corr = np.moveaxis(self.corr, 0, 2)
        #     self.rssi = np.moveaxis(self.rssi, 0, 2)
        self.frequency = transect.wVel.frequency
        self.orig_coord_sys = transect.wVel.origCoordSys
        self.orig_nav_ref = transect.wVel.origNavRef
        self.rssi_units = transect.wVel.rssiUnits
        self.water_mode = transect.wVel.waterMode
        self.blanking_distance_m = transect.wVel.blankingDistance_m
        self.invalid_index = transect.wVel.invalidIndex
        if type(transect.wVel.numInvalid) is np.ndarray:
            self.num_invalid = transect.wVel.numInvalid.tolist()
        else:
            self.num_invalid = transect.wVel.numInvalid

        # Settings
        self.beam_filter = transect.wVel.beamFilter
        self.d_filter = transect.wVel.dFilter
        self.d_filter_threshold = transect.wVel.dFilterThreshold
        self.w_filter = transect.wVel.wFilter
        self.w_filter_threshold = transect.wVel.wFilterThreshold
        self.excluded_dist_m = transect.wVel.excludedDist
        if hasattr(transect.wVel, 'orig_excludedDist'):
            self.orig_excluded_dist_m = transect.wVel.orig_excludedDist
        else:
            self.orig_excluded_dist_m = transect.wVel.excludedDist
        self.smooth_filter = transect.wVel.smoothFilter
        self.smooth_speed = transect.wVel.smoothSpeed
        self.smooth_upper_limit = transect.wVel.smoothUpperLimit
        self.smooth_lower_limit = transect.wVel.smoothLowerLimit
        self.snr_filter = transect.wVel.snrFilter
        self.wt_depth_filter = transect.wVel.wtDepthFilter
        self.interpolate_ens = transect.wVel.interpolateEns
        self.interpolate_cells = transect.wVel.interpolateCells
        self.coord_sys = transect.wVel.coordSys
        self.nav_ref = transect.wVel.navRef
        self.sl_cutoff_percent = transect.wVel.slCutoffPer
        self.sl_cutoff_number = transect.wVel.slCutoffNum
        self.sl_cutoff_type = transect.wVel.slCutoffType

    def change_coord_sys(self, new_coord_sys, sensors, adcp):
        """This function allows the coordinate system to be changed.

        Current implementation is only to allow a change to a higher order
        coordinate system Beam - Inst - Ship - Earth

        Parameters
        ----------
        new_coord_sys: str
            New coordinate system (Beam, Inst, Ship, Earth)
        sensors: Sensors
            Object of Sensors
        adcp: InstrumentData
            Object of instrument data
        """
        if type(self.orig_coord_sys) is list:
            o_coord_sys = self.orig_coord_sys[0].strip()
        else:
            o_coord_sys = self.orig_coord_sys.strip()

        orig_sys = None
        new_sys = None

        if o_coord_sys != new_coord_sys:
            
            # Assign the transformation matrix and retrieve the sensor data
            t_matrix = copy.deepcopy(adcp.t_matrix.matrix)
            t_matrix_freq = copy.deepcopy(adcp.frequency_khz)

            p = getattr(sensors.pitch_deg, sensors.pitch_deg.selected).data
            r = getattr(sensors.roll_deg, sensors.roll_deg.selected).data
            h = getattr(sensors.heading_deg, sensors.heading_deg.selected).data
            
            # Modify the transformation matrix and heading, pitch
            # and roll values based on the original coordinate
            # system so that only the needed values ar used in
            # computing the new coordinate system.
            if o_coord_sys.strip() == 'Beam':
                orig_sys = 1
            elif o_coord_sys.strip() == 'Inst':
                orig_sys = 2
            elif o_coord_sys.strip() == 'Ship':
                orig_sys = 3
                p = np.zeros(h.shape)
                r = np.zeros(h.shape)
                t_matrix = np.eye(len(t_matrix))
            elif o_coord_sys.strip() == 'Earth':
                orig_sys = 4

            # Assign a value to the new coordinate system
            if new_coord_sys.strip() == 'Beam':
                new_sys = 1
            elif new_coord_sys.strip() == 'Inst':
                new_sys = 2
            elif new_coord_sys.strip() == 'Ship':
                new_sys = 3
            elif new_coord_sys.strip() == 'Earth':
                new_sys = 4
                
            # Check to ensure the new coordinate system is a higher order than the original system
            if new_sys - orig_sys > 0:
                
                # Compute trig function for heaing, pitch and roll
                ch = np.cos(np.deg2rad(h))
                sh = np.sin(np.deg2rad(h))
                cp = np.cos(np.deg2rad(p))
                sp = np.sin(np.deg2rad(p))
                cr = np.cos(np.deg2rad(r))
                sr = np.sin(np.deg2rad(r))

                n_ens = self.raw_vel_mps.shape[2]
                
                for ii in range(n_ens):
                    
                    # Compute matrix for heading, pitch, and roll
                    hpr_matrix = np.array([[((ch[ii] * cr[ii]) + (sh[ii]*sp[ii] * sr[ii])),
                                            (sh[ii] * cp[ii]),
                                            ((ch[ii] * sr[ii]) - sh[ii]*sp[ii] * cr[ii])],
                                           [(-1 * sh[ii] * cr[ii]) + (ch[ii] * sp[ii] * sr[ii]),
                                            ch[ii] * cp[ii],
                                            (-1 * sh[ii] * sr[ii])-(ch[ii] * sp[ii] * cr[ii])],
                                           [(-1.*cp[ii] * sr[ii]),
                                            sp[ii],
                                            cp[ii] * cr[ii]]])
                    
                    # Transform beam coordinates
                    if o_coord_sys == 'Beam':
                        
                        # Determine frequency index for transformation
                        if len(t_matrix.shape) > 2:
                            idx_freq = np.where(t_matrix_freq == self.frequency[ii])
                            t_mult = np.copy(t_matrix[:, :, idx_freq])
                        else:
                            t_mult = np.copy(t_matrix)
                            
                        # Get velocity data
                        vel_beams = np.copy(self.raw_vel_mps[:, :, ii])
                        
                        # Apply transformation matrix for 4 beam solutions
                        temp_t = t_mult.dot(vel_beams)
                        
                        # Apply hpr_matrix
                        temp_thpr = hpr_matrix.dot(temp_t[:3])
                        temp_thpr = np.vstack([temp_thpr, temp_t[3]])
                        
                        # Check for invalid beams
                        invalid_idx = np.isnan(vel_beams)
                        
                        # Identify rows requiring 3 beam solutions
                        n_invalid_col = np.sum(invalid_idx, axis=0)
                        col_idx = np.where(n_invalid_col == 1)[0]
                        
                        # Compute 3 beam solution, if necessary
                        if len(col_idx) > 0:
                            for i3 in range(len(col_idx)):
                                
                                # Id invalid beam
                                vel_3_beam = vel_beams[:, col_idx[i3]]
                                idx_3_beam = np.where(np.isnan(vel_3_beam))[0]
                        
                                # 3 beam solution for non-RiverRay
                                vel_3_beam_zero = vel_3_beam
                                vel_3_beam_zero[np.isnan(vel_3_beam)] = 0
                                vel_error = t_mult[3, :].dot(vel_3_beam_zero)
                                vel_3_beam[idx_3_beam] = -1 * vel_error / t_mult[3, idx_3_beam]
                                temp_t = t_mult.dot(vel_3_beam)
                                
                                # Apply transformation matrix for 3
                                # beam solutions
                                temp_thpr[0:3, col_idx[i3]] = hpr_matrix.dot(temp_t[:3])
                                temp_thpr[3, col_idx[i3]] = np.nan
                            
                    else:
                        # Get velocity data
                        vel_raw = np.copy(np.squeeze(self.raw_vel_mps[:, :, ii]))
                        temp_thpr = np.array(hpr_matrix).dot(vel_raw[:3, :])
                        temp_thpr = np.vstack([temp_thpr, vel_raw[3, :]])
                        
                    # Update object
                    temp_thpr = temp_thpr.T
                    self.u_mps[:, ii] = temp_thpr[:, 0]
                    self.v_mps[:, ii] = temp_thpr[:, 1]
                    self.w_mps[:, ii] = temp_thpr[:, 2]
                    self.d_mps[:, ii] = temp_thpr[:, 3]

                # Because of padded arrays with zeros and RR has a variable number of bins,
                # the raw data may be padded with zeros.  The next 4 statements changes
                # those to nan
                self.u_mps[self.u_mps == 0] = np.nan
                self.v_mps[self.v_mps == 0] = np.nan
                self.w_mps[self.w_mps == 0] = np.nan
                self.d_mps[self.d_mps == 0] = np.nan
                
                # Assign processed object properties
                self.u_processed_mps = np.copy(self.u_mps)
                self.v_processed_mps = np.copy(self.v_mps)
                
                # Assign coordinate system and reference properties
                self.coord_sys = new_coord_sys
                self.nav_ref = self.orig_nav_ref
                    
            else:
                
                # Reset velocity properties to raw values
                self.u_mps = np.copy(self.raw_vel_mps[0])
                self.v_mps = np.copy(self.raw_vel_mps[1])
                self.w_mps = np.copy(self.raw_vel_mps[2])
                self.d_mps = np.copy(self.raw_vel_mps[3])
                
                if adcp.manufacturer == 'TRDI':
                    self.u_mps[self.u_mps == 0] = np.nan
                    self.v_mps[self.v_mps == 0] = np.nan
                    self.w_mps[self.w_mps == 0] = np.nan
                    self.d_mps[self.d_mps == 0] = np.nan
                    
                # Assign processed properties
                self.u_processed_mps = np.copy(self.u_mps)
                self.v_processed_mps = np.copy(self.v_mps)
                
        else:
            
            # Reset velocity properties to raw values
            self.u_mps = np.copy(self.raw_vel_mps[0])
            self.v_mps = np.copy(self.raw_vel_mps[1])
            self.w_mps = np.copy(self.raw_vel_mps[2])
            self.d_mps = np.copy(self.raw_vel_mps[3])
            
            if adcp.manufacturer == 'TRDI':
                self.u_mps[self.u_mps == 0] = np.nan
                self.v_mps[self.v_mps == 0] = np.nan
                self.w_mps[self.w_mps == 0] = np.nan
                self.d_mps[self.d_mps == 0] = np.nan
                
            # Assign processed properties
            self.u_processed_mps = np.copy(self.u_mps)
            self.v_processed_mps = np.copy(self.v_mps)
            
        if new_coord_sys == 'Earth':
            self.u_earth_no_ref_mps = np.copy(self.u_mps)
            self.v_earth_no_ref_mps = np.copy(self.v_mps)
                
    def set_nav_reference(self, boat_vel):           
        """This function sets the navigation reference.

        The current reference is first removed from the velocity and then the
        selected reference is applied.

        Parameters
        ----------
        boat_vel: BoatStructure
            Object of BoatStructure
        """
        
        # Apply selected navigation reference
        boat_select = getattr(boat_vel, boat_vel.selected)
        if boat_select is not None:
            self.u_mps = np.add(self.u_earth_no_ref_mps, boat_select.u_processed_mps)
            self.v_mps = np.add(self.v_earth_no_ref_mps, boat_select.v_processed_mps)
            self.nav_ref = boat_select.nav_ref
        else:
            self.u_mps = repmat([np.nan],
                                self.u_earth_no_ref_mps.shape[0],
                                self.u_earth_no_ref_mps.shape[1])
            self.v_mps = repmat([np.nan],
                                self.v_earth_no_ref_mps.shape[0],
                                self.v_earth_no_ref_mps.shape[1])
            if boat_vel.selected == 'bt_vel':
                self.nav_ref = 'BT'
            elif boat_vel.selected == 'gga_vel':
                self.nav_ref = 'GGA'
            elif boat_vel.selected == 'vtg_vel':
                self.nav_ref = 'VTG'
        
        valid_data2 = np.copy(self.cells_above_sl)
        valid_data2[np.isnan(self.u_mps)] = False
        self.valid_data[1] = valid_data2
        
        # Duplicate original to other filters that have yet to be applied
        self.valid_data[2:] = np.tile(self.valid_data[1], [7, 1, 1])
        
        # Combine all filter data and update processed properties
        self.all_valid_data()
        
    def change_heading(self, boat_vel, heading_chng):
        """Adjusts the velocity vectors for a change in heading due change in
        magnetic variation or heading offset.

        Parameters
        ----------
        boat_vel: BoatData
            Object of BoatData
        heading_chng: float
            Heading change due to change in magvar or offset, in degrees.
        """
        u_nr = self.u_earth_no_ref_mps
        v_nr = self.v_earth_no_ref_mps
        direction, mag = cart2pol(u_nr, v_nr)
        u_nr_rotated, v_nr_rotated = pol2cart(direction - np.deg2rad(heading_chng), mag)
        self.u_earth_no_ref_mps = u_nr_rotated
        self.v_earth_no_ref_mps = v_nr_rotated

        # Reprocess water data to get navigation reference corrected velocities
        self.set_nav_reference(boat_vel)
        
    def change_heading_source(self, boat_vel, heading):
        """Applies changes to water velocity when the heading source is changed.

        Typically called when the heading source is changed between external and internal.

        Parameters
        ----------
        boat_vel: BoatData
            Object of BoatData
        heading: np.array(float)
            New heading data, in degrees
        """
        u_nr = self.u_earth_no_ref_mps
        v_nr = self.v_earth_no_ref_mps
        direction, mag = cart2pol(u_nr, v_nr)
        u_nr_rotated, v_nr_rotated = pol2cart(direction
                                              - np.deg2rad(repmat(heading, len(mag), 1)), mag)
        self.u_earth_no_ref_mps = u_nr_rotated
        self.v_earth_no_ref_mps = v_nr_rotated

        self.set_nav_reference(boat_vel)
            
    def apply_interpolation(self, transect, ens_interp='None', cells_interp='None'):
        """Coordinates the application of water velocity interpolation.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        ens_interp: str
            Specifies type of interpolation for ensembles
        cells_interp: str
            Specifies type of interpolation for cells
        """

        self.u_processed_mps = np.tile([np.nan], self.u_mps.shape)
        self.v_processed_mps = np.tile([np.nan], self.v_mps.shape)
        self.u_processed_mps[self.valid_data[0]] = self.u_mps[self.valid_data[0]]
        self.v_processed_mps[self.valid_data[0]] = self.v_mps[self.valid_data[0]]
        
        # Determine interpolation methods to apply
        if ens_interp == 'None':
            ens_interp = self.interpolate_ens
        else:
            self.interpolate_ens = ens_interp

        if cells_interp == 'None':
            cells_interp = self.interpolate_cells
        else:
            self.interpolate_cells = cells_interp

        if ens_interp == 'abba' or cells_interp == 'abba':
            self.interpolate_ens = 'abba'
            self.interpolate_cells = 'abba'
            self.interpolate_abba(transect)
        else:
            if ens_interp == 'None':
                # Sets invalid data to nan with no interpolation
                self.interpolate_ens_none()
            elif ens_interp == 'ExpandedT':
                # Sets interpolate to None as the interpolation is done in class QComp
                self.interpolate_ens_next()
            elif ens_interp == 'Hold9':
                # Interpolates using SonTek's method of holding last valid for up to 9 samples
                self.interpolate_ens_hold_last_9()
            elif ens_interp == 'Hold':
                # Interpolates by holding last valid indefinitely
                self.interpolate_ens_hold_last()
            elif ens_interp == 'Linear':
                # Interpolates using linear interpolation
                self.interpolate_ens_linear(transect)
            elif ens_interp == 'TRDI':
                # TRDI is applied in discharge
                self.interpolate_ens_none()
                self.interpolate_ens = ens_interp

            # Apply specified cell interpolation method
            if cells_interp == 'None':
                # Sets invalid data to nan with no interpolation
                self.interpolate_cells_none()
            elif cells_interp == 'TRDI':
                # Use TRDI method to interpolate invalid interior cells
                self.interpolate_cells_trdi(transect)
            elif cells_interp == 'Linear':
                # Uses linear interpolation to interpolate velocity for all
                # invalid bins including those in invalid ensembles
                # up to 9 samples
                self.interpolate_cells_linear(transect)
        
    def apply_filter(self, transect, beam=None, difference=None, difference_threshold=None, vertical=None,
                     vertical_threshold=None, other=None, excluded=None, snr=None, wt_depth=None):
        """Coordinates application of specified filters and subsequent interpolation.
        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        beam: int
            Setting for beam filter (3, 4, or -1)
        difference: str
            Setting for difference filter (Auto, Off, Manual)
        difference_threshold: float
            Threshold value for Manual setting.
        vertical: str
            Setting for vertical filter (Auto, Off, Manual)
        vertical_threshold: float
            Threshold value for Manual setting.
        other:
            Setting for other filters (Off, Auto)
        excluded:
            Excluded distance below the transducer, in m
        snr: str
            SNR filter setting (Auto, Off)
        wt_depth: bool
            Setting for marking water data invalid if no available depth
        """


        # Determine filters to apply
        if len({beam, difference, difference_threshold, vertical, vertical_threshold, other, excluded, snr,
                wt_depth}) > 1:

            if difference is not None:
                if difference == 'Manual':
                    self.filter_diff_vel(setting=difference, threshold=difference_threshold)
                else:
                    self.filter_diff_vel(setting=difference)
            if vertical is not None:
                if vertical == 'Manual':
                    self.filter_vert_vel(setting=vertical, threshold=vertical_threshold)
                else:
                    self.filter_vert_vel(setting=vertical)
            if other is not None:
                self.filter_smooth(transect=transect, setting=other)
            if excluded is not None:
                self.filter_excluded(transect=transect, setting=excluded)
            if snr is not None:
                self.filter_snr(setting=snr)
            if wt_depth is not None:
                self.filter_wt_depth(transect=transect, setting=wt_depth)
            if beam is not None:
                self.filter_beam(setting=beam, transect=transect)
        else:
            self.filter_diff_vel(setting=self.d_filter, threshold=self.d_filter_threshold)
            self.filter_vert_vel(setting=self.w_filter, threshold=self.w_filter_threshold)
            self.filter_smooth(transect=transect, setting=self.smooth_filter)
            self.filter_excluded(transect=transect, setting=self.excluded_dist_m)
            self.filter_snr(setting=self.snr_filter)
            self.filter_beam(setting=self.beam_filter, transect=transect)

        # After filters have been applied, interpolate to estimate values for invalid data.
        # self.apply_interpolation(transect=transect)
        
    def sos_correction(self, ratio):
        """Corrects water velocities for a change in speed of sound.

        Parameters
        ----------
        ratio: float
            Ratio of new speed of sound to old speed of sound
        """

        # Correct water velocities
        self.u_mps = self.u_mps * ratio
        self.v_mps = self.v_mps * ratio
        self.u_earth_no_ref_mps = self.u_earth_no_ref_mps * ratio
        self.v_earth_no_ref_mps = self.v_earth_no_ref_mps * ratio

    def adjust_side_lobe(self, transect):
        """Adjust the side lobe cutoff for vertical beam and interpolated depths.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """


        selected = transect.depths.selected
        depth_selected = getattr(transect.depths, transect.depths.selected)
        cells_above_slbt = np.copy(self.cells_above_sl_bt)
        
        # Compute cutoff for vertical beam depths
        if selected == 'vb_depths':
            sl_cutoff_vb = (depth_selected.depth_processed_m - depth_selected.draft_use_m) \
                 * np.cos(np.deg2rad(transect.adcp.beam_angle_deg)) \
                - self.sl_lag_effect_m + depth_selected.draft_use_m
            cells_above_slvb = np.round(depth_selected.depth_cell_depth_m, 2) < np.round(sl_cutoff_vb, 2)
            idx = np.where(transect.depths.bt_depths.valid_data == False)
            cells_above_slbt[:, idx] = cells_above_slvb[:, idx]
            cells_above_sl = np.logical_and(cells_above_slbt, cells_above_slvb)
        else:
            cells_above_sl = cells_above_slbt

        # Compute cutoff from interpolated depths
        # Find ensembles with no valid beam depths
        idx = np.where(np.nansum(depth_selected.valid_beams, 0) == 0)[0]

        # Determine side lobe cutoff for ensembles with no valid beam depths
        if len(idx) > 0:
            if len(self.sl_lag_effect_m) > 1:
                sl_lag_effect_m = self.sl_lag_effect_m[idx]
            else:
                sl_lag_effect_m = self.sl_lag_effect_m
                
            sl_cutoff_int = (depth_selected.depth_processed_m[idx] - depth_selected.draft_use_m) \
                * np.cos(np.deg2rad(transect.adcp.beam_angle_deg)) - sl_lag_effect_m + \
                depth_selected.draft_use_m
            for i in range(len(idx)):
                cells_above_sl[:, idx[i]] = np.less(depth_selected.depth_cell_depth_m[:, idx[i]], sl_cutoff_int[i])
            
        # Find ensembles with at least 1 invalid beam depth
        idx = np.where(np.nansum(depth_selected.valid_beams, 0) < 4)[0]
        if len(idx) > 0:
            if len(self.sl_lag_effect_m) > 1:
                sl_lag_effect_m = self.sl_lag_effect_m[idx]
            else:
                sl_lag_effect_m = self.sl_lag_effect_m
                
            sl_cutoff_int = (depth_selected.depth_processed_m[idx] - depth_selected.draft_use_m)\
                * np.cos(np.deg2rad(transect.adcp.beam_angle_deg)) \
                - sl_lag_effect_m + depth_selected.draft_use_m
            cells_above_sl_int = np.tile(True, cells_above_sl.shape)

            for i in range(len(idx)):
                cells_above_sl_int[:, idx[i]] = np.less(depth_selected.depth_cell_depth_m[:, idx[i]], sl_cutoff_int[i])
            
            cells_above_sl[cells_above_sl_int == 0] = 0
        
        self.cells_above_sl = np.copy(cells_above_sl)
        valid_vel = np.logical_not(np.isnan(self.u_mps))
        self.valid_data[1, :, :] = self.cells_above_sl * valid_vel
        self.all_valid_data()
        self.compute_snr_rng()
        self.apply_filter(transect)
        # self.apply_interpolation(transect)

    def all_valid_data(self):
        """Combines the results of all filters to determine a final set of valid data"""

        n_filters = len(self.valid_data[1:, 0, 0])
        sum_filters = np.nansum(self.valid_data[1:, :, :], 0) / n_filters
        valid = np.tile([True], self.cells_above_sl.shape)
        valid[sum_filters < 1] = False
        self.valid_data[0] = valid
        
    def filter_beam(self, setting, transect=None):
        """Applies beam filter to water velocity data.

        The determination of invalid data depends on whether
        3-beam or 4-beam solutions are acceptable.  This function can be applied by
        specifying 3 or 4 beam solutions and setting self.beam_filter to -1
        which will trigger an automatic mode.  The automatic mode will find all 3 beam
        solutions and them compare the velocity of the 3 beam solutions to nearest 4
        beam solutions.  If the 3 beam solution is within 50% of the average of the
        neighboring 4 beam solutions the data are deemed valid, if not they are marked
        invalid.  Thus in automatic mode only those data from 3 beam solutions
        that are sufficiently different from  the 4 beam solutions are marked invalid.
        If the number of beams is specified manually, it is applied
        uniformly for the whole transect.

        Parameters
        ----------
        setting: int
            Setting for beam filter (3, 4, or -1)
        transect: TransectData
            Object of TransectData
        """
        
        
        self.beam_filter = setting
        
        # In manual mode (3 or 4) determine number of raw invalid and number of 2 beam solutions
        if self.beam_filter > 0:
            
            # Find invalid raw data
            valid_vel = np.array([self.cells_above_sl] * 4)
            valid_vel[np.isnan(self.raw_vel_mps)] = 0
            
            # Determine how many beams or transformed coordinates are valid
            valid_vel_sum = np.sum(valid_vel, 0)
            valid = copy.deepcopy(self.cells_above_sl)
            
            # Compare number of valid beams or velocity coordinates to filter value
            valid[np.logical_and((valid_vel_sum < self.beam_filter), (valid_vel_sum > 2))] = False
            
            # Save logical of valid data to object
            self.valid_data[5, :, :] = valid
        
        else:

            # Apply automatic filter
            self.automatic_beam_filter_abba_interpolation(transect)

    def automatic_beam_filter_abba_interpolation(self, transect):
        """Applies abba interpolation to allow comparison of interpolated and 3-beam solutions.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """


        # Create array indicating which cells do not have 4-beam solutions and all cells below side lobe are nan
        temp = copy.deepcopy(self)
        temp.filter_beam(4)
        valid_bool = temp.valid_data[5, :, :]
        valid = valid_bool.astype(float)
        valid[temp.cells_above_sl == False] = np.nan

        # Initialize processed velocity data variables
        temp.u_processed_mps = copy.deepcopy(temp.u_mps)
        temp.v_processed_mps = copy.deepcopy(temp.v_mps)

        # Set invalid data to nan in processed velocity data variables
        temp.u_processed_mps[np.logical_not(valid)] = np.nan
        temp.v_processed_mps[np.logical_not(valid)] = np.nan

        # Find indices of cells with 3 beams solutions
        rows_3b, cols_3b = np.where(np.abs(valid) == 0)

        # Check for presence of 3-beam solutions
        if len(rows_3b) > 0:
            interpolated_data = self.compute_abba_interpolation(wt_data=temp,
                                                                valid=temp.valid_data[5, :, :],
                                                                transect=transect)

            if interpolated_data is not None:
                # Compute interpolated to measured ratios and apply filter criteria
                for n in range(len(interpolated_data[0])):
                    u_ratio = (temp.u_mps[interpolated_data[0][n][0]] / interpolated_data[0][n][1]) - 1
                    v_ratio = (temp.v_mps[interpolated_data[1][n][0]] / interpolated_data[1][n][1]) - 1
                    if np.abs(u_ratio) < 0.5 and np.abs(v_ratio) < 0.5:
                        valid_bool[interpolated_data[0][n][0]] = True
                    else:
                        valid_bool[interpolated_data[0][n][0]] = False
                    # n += 1

                # Update object with filter results
                self.valid_data[5, :, :] = valid_bool
            else:
                self.valid_data[5, :, :] = temp.valid_data[5, :, :]
        else:
            self.valid_data[5, :, :] = temp.valid_data[5, :, :]

        # Combine all filter data and update processed properties
        self.all_valid_data()

    def filter_diff_vel(self, setting, threshold=None):
        """Applies filter to difference velocity.

        Applies either manual or automatic filtering of the difference (error)
        velocity.  The automatic mode is based on the following:  This filter is
        based on the assumption that the water error velocity should follow a gaussian
        distribution.  Therefore, 5 standard deviations should encompass all of the
        valid data.  The standard deviation and limits (multiplier*std dev) are computed
        in an iterative process until filtering out additional data does not change the
        computed standard deviation.

        Parameters
        ----------
        setting: str
            Filter setting (Auto, Off, Manual)
        threshold: float
            Threshold value for Manual setting.
        """
        

        # Set difference filter properties
        self.d_filter = setting
        if threshold is not None:
            self.d_filter_threshold = threshold

        # Set multiplier
        multiplier = 5

        # Get difference data from object
        d_vel = copy.deepcopy(self.d_mps)

        d_vel_min_ref = None
        d_vel_max_ref = None

        # Apply selected method
        if self.d_filter == 'Manual':
            d_vel_max_ref = np.abs(self.d_filter_threshold)
            d_vel_min_ref = -1 * d_vel_max_ref
        elif self.d_filter == 'Off':
            d_vel_max_ref = np.nanmax(np.nanmax(d_vel)) + 1
            d_vel_min_ref = np.nanmin(np.nanmin(d_vel)) - 1
        elif self.d_filter == 'Auto':
            # Initialize variables
            d_vel_filtered = copy.deepcopy(d_vel)
            std_diff = 1
            i = -1
            # Loop until no additional data are removed
            while std_diff != 0 and i < 1000:
                i = i+1

                # Compute standard deviation
                d_vel_std = iqr(d_vel_filtered)

                # Compute maximum and minimum thresholds
                d_vel_max_ref = np.nanmedian(d_vel_filtered) + multiplier * d_vel_std
                d_vel_min_ref = np.nanmedian(d_vel_filtered) - multiplier * d_vel_std

                # Identify valid and invalid data
                d_vel_bad_rows, d_vel_bad_cols = np.where(np.logical_or
                                                          (np.greater(d_vel_filtered, d_vel_max_ref),
                                                           np.less(d_vel_filtered, d_vel_min_ref)))

                # Update filtered data array
                d_vel_filtered[d_vel_bad_rows, d_vel_bad_cols] = np.nan

                # Determine differences due to last filter iteration
                if len(d_vel_filtered) > 0:
                    d_vel_std2 = iqr(d_vel_filtered)
                    std_diff = d_vel_std2 - d_vel_std
                else:
                    std_diff = 0

        # Set valid data row 2 for difference velocity filter results
        bad_idx_rows, bad_idx_cols = np.where(np.logical_or(np.greater(d_vel, d_vel_max_ref),
                                              np.less(d_vel, d_vel_min_ref)))
        valid = copy.deepcopy(self.cells_above_sl)
        if len(bad_idx_rows) > 0:
            valid[bad_idx_rows, bad_idx_cols] = False
        # TODO Seems like if the difference velocity doesn't exist due to a 3-beam solution it shouldn't be
        #  flagged as invalid however this is the way it was in Matlab. May change this in future.
        # valid[np.isnan(self.d_mps)] = True
        self.valid_data[2, :, :] = valid

        # Set threshold property
        if np.ma.is_masked(d_vel_max_ref):
            self.d_filter_threshold = np.nan
        else:
            self.d_filter_threshold = d_vel_max_ref

        # Combine all filter data and update processed properties
        self.all_valid_data()

    def filter_vert_vel(self, setting, threshold=None):
        """Applies filter to vertical velocity.

        Applies either manual or automatic filter of the difference (error) velocity.  The automatic
        mode is based on the following: This filter is based on the assumption that the water error
        velocity should follow a gaussian distribution.  Therefore, 4 standard deviations should
        encompass all of the valid data.  The standard deviation and limits (multplier * standard deviation)
        are computed in an iterative process until filtering out additional data does not change
        the computed standard deviation.

        Parameters
        ---------
        setting: str
            Filter setting (Auto, Off, Manual)
        threshold: float
            Threshold value for Manual setting."""
            
        
        # Set vertical velocity filter properties
        self.w_filter = setting
        if threshold is not None:
            self.w_filter_threshold = threshold

        # Set multiplier
        multiplier = 5

        # Get difference data from object
        w_vel = copy.deepcopy(self.w_mps)

        w_vel_min_ref = None
        w_vel_max_ref = None

        # Apply selected method
        if self.w_filter == 'Manual':
            w_vel_max_ref = np.abs(self.w_filter_threshold)
            w_vel_min_ref = -1 * w_vel_max_ref
        elif self.w_filter == 'Off':
            w_vel_max_ref = np.nanmax(np.nanmax(w_vel)) + 1
            w_vel_min_ref = np.nanmin(np.nanmin(w_vel)) - 1
        elif self.w_filter == 'Auto':
            # Initialize variables
            w_vel_filtered = copy.deepcopy(w_vel[:])
            std_diff = 1
            i = 0
            # Loop until no additional data are removed
            while std_diff != 0 and i < 1000:

                # Computed standard deviation
                w_vel_std = iqr(w_vel_filtered)

                # Compute maximum and minimum thresholds
                w_vel_max_ref = np.nanmedian(w_vel_filtered) + multiplier * w_vel_std
                w_vel_min_ref = np.nanmedian(w_vel_filtered) - multiplier * w_vel_std

                # Identify valid and invalid data
                w_vel_bad_rows, w_vel_bad_cols = np.where(np.logical_or(np.greater(w_vel_filtered, w_vel_max_ref),
                                                                        np.less(w_vel_filtered, w_vel_min_ref)))

                # Update filtered data array
                w_vel_filtered[w_vel_bad_rows, w_vel_bad_cols] = np.nan

                # Determine differences due to last filter iteration
                if len(w_vel_filtered) > 0:
                    w_vel_std2 = iqr(w_vel_filtered)
                    std_diff = w_vel_std2 - w_vel_std
                else:
                    std_diff = 0
                    
        # Set valid data row 3 for difference velocity filter results
        bad_idx_rows, bad_idx_cols = np.where(np.logical_or(np.greater(w_vel, w_vel_max_ref),
                                              np.less(w_vel, w_vel_min_ref)))
        valid = copy.deepcopy(self.cells_above_sl)
        if len(bad_idx_rows) > 0:
            valid[bad_idx_rows, bad_idx_cols] = False
        self.valid_data[3, :, :] = valid

        # Set threshold property
        if np.ma.is_masked(w_vel_max_ref):
            self.w_filter_threshold = np.nan
        else:
            self.w_filter_threshold = w_vel_max_ref

        # Combine all filter data and update processed properties
        self.all_valid_data()
                
    def filter_smooth(self, transect, setting):
        """Filter water speed using a smooth filter.

        Running Standard Deviation filter for water speed
        This filter employs a running trimmed standard deviation filter to
        identify and mark spikes in the water speed. First a robust Loess
        smooth is fitted to the water speed time series and residuals between
        the raw data and the smoothed line are computed. The trimmed standard
        eviation is computed by selecting the number of residuals specified by
        "halfwidth" before the target point and after the target point, but not
        including the target point. These values are then sorted, and the points
        with the highest and lowest values are removed from the subset, and the
        standard deviation of the trimmed subset is computed. The filter
        criteria are determined by multiplying the standard deviation by a user
        specified multiplier. This criteria defines a maximum and minimum
        acceptable residual. Data falling outside the criteria are set to nan.
          
        Recommended filter settings are:
        filter_width = 10
        half_width = 10
        multiplier = 9

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        setting: str
            Set filter (Auto, Off)
        """
        
        self.smooth_filter = setting
        upper_limit = None
        lower_limit = None
        wt_bad_idx = None
        
        # Compute ens_time
        ens_time = np.nancumsum(transect.date_time.ens_duration_sec)
        
        # Determine if smooth filter should be applied
        if self.smooth_filter == 'Auto':
            
            # Boat velocity components
            w_vele = self.u_mps
            w_veln = self.v_mps
            
            # Set filter parameters
            filter_width = 10
            half_width = 10
            multiplier = 9
            cycles = 3

            # Compute mean speed and direction of water
            w_vele_avg = np.nanmean(w_vele, 0)
            w_veln_avg = np.nanmean(w_veln, 0)
            _, speed = cart2pol(w_vele_avg, w_veln_avg)
            
            # Compute residuals from a robust Loess smooth
            speed_smooth = rloess(ens_time, speed, filter_width)
            speed_res = speed - speed_smooth
            
            # Apply a trimmed standard deviation filter multiple times
            for i in range(cycles):
                fill_array = BoatData.run_std_trim(half_width, speed_res.T)
                
                # Compute filter bounds
                upper_limit = speed_smooth + multiplier * fill_array
                lower_limit = speed_smooth - multiplier * fill_array
                
                # Apply filter to residuals
                wt_bad_idx = np.where((speed > upper_limit) or (speed < lower_limit))[0]
                speed_res[wt_bad_idx] = np.nan
            
            valid = np.copy(self.cells_above_sl)
            
            valid[:, wt_bad_idx] = False
            self.valid_data[4, :, :] = valid
            self.smooth_upper_limit = upper_limit
            self.smooth_lower_limit = lower_limit
            self.smooth_speed = speed_smooth
        
        else:
            # No filter applied
            self.valid_data[4, :, :] = np.copy(self.cells_above_sl)
            self.smooth_upper_limit = np.nan
            self.smooth_lower_limit = np.nan
            self.smooth_speed = np.nan
            
        self.all_valid_data()
     
    def filter_snr(self, setting):
        """Filters SonTek data based on SNR.

        Computes the average SNR for all cells above the side lobe cutoff for each beam in
        each ensemble. If the range in average SNR in an ensemble is greater than 12 dB the
        water velocity in that ensemble is considered invalid.

        Parameters
        ----------
        setting: str
            Setting for filter (Auto, Off)
        """
        

        self.snr_filter = setting  
        
        if setting == 'Auto':
            if self.snr_rng is not None:
                bad_snr_idx = np.greater(self.snr_rng, 12)
                valid = np.copy(self.cells_above_sl)
                
                bad_snr_array = np.tile(bad_snr_idx, (valid.shape[0], 1))
                valid[bad_snr_array] = False
                self.valid_data[7, :, :] = valid

                # Combine all filter data and update processed properties
                self.all_valid_data()
        else:
            self.valid_data[7, :, :] = np.copy(self.cells_above_sl)
            self.all_valid_data()
        
    def filter_wt_depth(self, transect, setting):
        """Marks water velocity data invalid if there is no valid or interpolated average depth.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        setting: bool
            Setting for filter (True, False)
        """
        
        
        self.wt_depth_filter = setting
        valid = np.copy(self.cells_above_sl)
        
        if setting:
            trans_select = getattr(transect.depths, transect.depths.selected)
            valid[:, np.isnan(trans_select.depth_processed_m)] = False
        self.valid_data[8, :, :] = valid
        
        self.all_valid_data()
        
    def filter_excluded(self, transect, setting):
        """Marks all data invalid that are closer to the transducer than the setting.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        setting: float
            Range from the transducer, in m
        """

        # Initialize variables
        trans_select = getattr(transect.depths, transect.depths.selected)
        cell_depth = trans_select.depth_cell_depth_m
        cell_size = trans_select.depth_cell_size_m
        draft = trans_select.draft_use_m
        top_cell_depth = cell_depth - 0.5 * cell_size
        threshold = np.round((setting+draft), 3)

        # Apply filter
        exclude = np.round(top_cell_depth, 3) <= threshold
        valid = np.copy(self.cells_above_sl)
        valid[exclude] = False
        self.valid_data[6, :, :] = valid
        
        # Set threshold property
        self.excluded_dist_m = setting
        
        self.all_valid_data()

    def interpolate_abba(self, transect, search_loc=['above', 'below', 'before', 'after']):
        """" Interpolates all data marked invalid using the abba interpolation algorithm.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """

        # Set properties
        self.interpolate_cells = 'abba'
        self.interpolate_ens = 'abba'

        # Get valid data based on all filters applied
        valid = self.valid_data[0, :, :]

        # Initialize processed velocity data variables
        self.u_processed_mps = copy.deepcopy(self.u_mps)
        self.v_processed_mps = copy.deepcopy(self.v_mps)

        # Set invalid data to nan in processed velocity data variables
        self.u_processed_mps[np.logical_not(valid)] = np.nan
        self.v_processed_mps[np.logical_not(valid)] = np.nan

        interpolated_data = self.compute_abba_interpolation(wt_data=self,
                                                            valid=valid,
                                                            transect=transect,
                                                            search_loc=search_loc)

        if interpolated_data is not None:
            # Incorporate interpolated values in processed data
            for n in range(len(interpolated_data[0])):
                self.u_processed_mps[interpolated_data[0][n][0]] = \
                    interpolated_data[0][n][1]
                self.v_processed_mps[interpolated_data[1][n][0]] = \
                    interpolated_data[1][n][1]

    @staticmethod
    def compute_abba_interpolation(wt_data, valid, transect, search_loc=['above', 'below', 'before', 'after']):
        """Computes the interpolated values for invalid cells using the abba method.

        Parameters
        ----------
        wt_data: WaterData
            Object of WaterData
        valid: np.ndarray(bool)
            Array indicating valid to be used for interpolation
        transect: TransectData
            Object of TransectData

        Returns
        -------
        interpolated_data: np.ndarray(float)
            Array of interpolated data
        """
        # Find cells with invalid data
        valid_cells = wt_data.valid_data[0, :, :]
        boat_selected = getattr(transect.boat_vel, transect.boat_vel.selected)
        if boat_selected is not None:
            boat_valid = boat_selected.valid_data[0]
        else:
            boat_valid = 0

        if not np.all(valid_cells) and np.nansum(boat_valid) > 1:
            # Compute distance along shiptrack to be used in interpolation
            distance_along_shiptrack = transect.boat_vel.compute_boat_track(transect)['distance_m']

            # Where there is invalid boat speed at beginning or end of transect mark the distance nan to avoid
            # interpolating velocities that won't be used for discharge
            if type(distance_along_shiptrack) is np.ndarray:
                distance_along_shiptrack[0:np.argmax(boat_valid == True)] = np.nan
                end_nan = np.argmax(np.flip(boat_valid) == True)
                if end_nan > 0:
                    distance_along_shiptrack[-1 * end_nan:] = np.nan
            # if type(distance_along_shiptrack) is np.ndarray:
                depth_selected = getattr(transect.depths, transect.depths.selected)

                # Interpolate values for  invalid cells with from neighboring data
                interpolated_data = abba_idw_interpolation(data_list=[wt_data.u_processed_mps, wt_data.v_processed_mps],
                                                           valid_data=valid,
                                                           cells_above_sl=wt_data.valid_data[6, :, :],
                                                           y_centers=depth_selected.depth_cell_depth_m,
                                                           y_cell_size=depth_selected.depth_cell_size_m,
                                                           y_depth=depth_selected.depth_processed_m,
                                                           x_shiptrack=distance_along_shiptrack,
                                                           search_loc=search_loc,
                                                           normalize=True)
                return interpolated_data
            else:
                return None
        else:
            return None

    def interpolate_ens_next(self):
        """Applies data from the next valid ensemble for ensembles with invalid water velocities.
        """

        # Set interpolation property for ensembles
        self.interpolate_ens = 'ExpandedT'
        
        # Set processed data to nan for all invalid data
        valid = self.valid_data[0]
        self.u_processed_mps = np.copy(self.u_mps)
        self.v_processed_mps = np.copy(self.v_mps)
        self.u_processed_mps[valid == False] = np.nan
        self.v_processed_mps[valid == False] = np.nan
        
        # Identifying ensembles with no valid data
        valid_ens = np.any(valid, axis=0)
        n_ens = len(valid_ens)
        
        # Set the invalid ensembles to the data in the next valid ensemble
        for n in np.arange(0, n_ens-1)[::-1]:
            if not valid_ens[n]:
                self.u_processed_mps[:, n] = self.u_processed_mps[:, n+1]
                self.v_processed_mps[:, n] = self.v_processed_mps[:, n+1]
                
    def interpolate_ens_hold_last(self):
        """Interpolates velocity data for invalid ensembles by repeating the
        the last valid data until new valid data is found
        """
        
        self.interpolate_ens = 'HoldLast'
        
        valid = self.valid_data[0]
        
        # Initialize processed velocity data variables
        self.u_processed_mps = np.copy(self.u_mps)
        self.v_processed_mps = np.copy(self.v_mps)
        
        # Set invalid data to nan in processed velocity data variables
        self.u_processed_mps[valid == False] = np.nan
        self.v_processed_mps[valid == False] = np.nan
        
        # Determine ensembles with valid data
        valid_ens = np.any(valid, axis=0)
        
        # Process each ensemble beginning with the second ensemble
        n_ens = len(valid_ens)
        
        for n in np.arange(1, n_ens):
            # If ensemble is invalid fill in with previous ensemble
            if not valid_ens[n]:
                self.u_processed_mps[:, n] = self.u_processed_mps[:, n-1]
                self.v_processed_mps[:, n] = self.v_processed_mps[:, n-1]

    def interpolate_ens_hold_last_9(self):
        """Apply SonTek's approach to invalid data.

        Interpolates velocity data for invalid ensembles by repeating the
        last valid data for up to 9 ensembles or until new valid data is
        found. If more the 9 consecutive ensembles are invalid the
        ensembles beyond the 9th remain invalid. This is for
        compatibility with SonTek RiverSurveyor Live.
        """
        
        self.interpolate_ens = 'Hold9'
        
        valid = self.valid_data[0]
        
        # Initialize processed velocity data variables
        self.u_processed_mps = np.copy(self.u_mps)
        self.v_processed_mps = np.copy(self.v_mps)
        
        # Set invalid data to nan in processed velocity data variables
        self.u_processed_mps[valid == False] = np.nan
        self.v_processed_mps[valid == False] = np.nan
        
        # Determine ensembles with valid data
        valid_ens = np.any(valid, axis=0)
        
        # Process each ensemble beginning with the second ensemble
        n_ens = len(valid_ens)
        n_invalid = 0
        
        for n in np.arange(1, n_ens):
            # If ensemble is invalid fill in with previous ensemble
            if valid_ens[n] == False and n_invalid < 10:
                n_invalid += 1
                self.u_processed_mps[:, n] = self.u_processed_mps[:, n-1]
                self.v_processed_mps[:, n] = self.v_processed_mps[:, n-1]
            else:
                n_invalid = 0

    def interpolate_ens_none(self):
        """Applies no interpolation for invalid ensembles."""
        
        self.interpolate_ens = 'None'
        
        valid = self.valid_data[0]
        
        # Initialize processed velocity data variables
        self.u_processed_mps = np.copy(self.u_mps)
        self.v_processed_mps = np.copy(self.v_mps)
        
        # Set invalid data to nan in processed velocity data variables
        self.u_processed_mps[valid == False] = np.nan
        self.v_processed_mps[valid == False] = np.nan

    def interpolate_cells_none(self):
        """Applies no interpolation for invalid cells that are not part of
        an invalid ensemble."""

        self.interpolate_cells = 'None'
        
        valid = self.valid_data[0]

        # Determine ensembles with valid data
        valid_ens = np.any(valid, axis=0)

        # Process each ensemble beginning with the second ensemble
        n_ens = len(valid_ens)

        # Initialize processed velocity data variables
        self.u_processed_mps = np.copy(self.u_mps)
        self.v_processed_mps = np.copy(self.v_mps)
        
        for n in range(n_ens):
            # If ensemble is invalid fill in with previous ensemble
            if valid_ens[n]:
                invalid_cells = np.logical_not(valid[:, n])
                self.u_processed_mps[invalid_cells,
                                     n] = np.nan
                self.v_processed_mps[invalid_cells,
                                     n] = np.nan
        
    def interpolate_ens_linear(self, transect):
        """Uses 2D linear interpolation to estimate values for invalid ensembles.

        Use linear interpolation as computed by scipy's interpolation
        function to interpolated velocity data for ensembles with no valid velocities.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """

        self.interpolate_ens = 'Linear'
         
        valid = self.valid_data[0, :, :]

        # Initialize processed velocity data variables
        self.u_processed_mps = np.copy(self.u_mps)
        self.v_processed_mps = np.copy(self.v_mps)

        # Determine ensembles with valid data
        valid_ens = np.any(valid, 0)
        
        if np.sum(valid_ens) > 1:
            # Determine the number of ensembles
            # n_ens = len(valid_ens)
            
            trans_select = getattr(transect.depths, transect.depths.selected)
            # Compute z
            z = np.divide(np.subtract(trans_select.depth_processed_m, trans_select.depth_cell_depth_m),
                          trans_select.depth_processed_m)
            
            # Create position array
            boat_select = getattr(transect.boat_vel, transect.boat_vel.selected)
            if boat_select is not None:
                if np.nansum(boat_select.valid_data[0]) > 0:
                    boat_vel_x = boat_select.u_processed_mps
                    boat_vel_y = boat_select.v_processed_mps
                    track_x = boat_vel_x * transect.date_time.ens_duration_sec
                    track_y = boat_vel_y * transect.date_time.ens_duration_sec
                    track = np.nancumsum(np.sqrt(track_x**2 + track_y**2))
                    track_array = np.tile(track, (self.u_processed_mps.shape[0], 1))
                    
                    # Determine index of all valid data
                    valid_z = np.isnan(z) == False
                    valid_combined = np.logical_and(valid, valid_z)

                    u = interpolate.griddata(np.vstack((z[valid_combined], track_array[valid_combined])).T,
                                             self.u_processed_mps[valid_combined],
                                             (z, track_array))
                    
                    v = interpolate.griddata(np.vstack((z[valid_combined], track_array[valid_combined])).T,
                                             self.v_processed_mps[valid_combined],
                                             (z, track_array))

                    self.u_processed_mps = np.tile(np.nan, self.u_mps.shape)
                    self.u_processed_mps = np.tile(np.nan, self.u_mps.shape)
                    processed_valid_cells = self.estimate_processed_valid_cells(transect)
                    self.u_processed_mps[processed_valid_cells] = u[processed_valid_cells]
                    self.v_processed_mps[processed_valid_cells] = v[processed_valid_cells]

    def interpolate_cells_linear(self, transect):
        """Uses 2D linear interpolation to estimate values for invalid cells.

        Use linear interpolation as computed by scipy's interpolation
        function to interpolated velocity data for cells with no valid velocities.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """

        self.interpolate_ens = 'Linear'

        valid = self.valid_data[0, :, :]

        # Initialize processed velocity data variables
        self.u_processed_mps = np.copy(self.u_mps)
        self.v_processed_mps = np.copy(self.v_mps)

        trans_select = getattr(transect.depths, transect.depths.selected)

        # Compute z
        z = np.divide(np.subtract(trans_select.depth_processed_m, trans_select.depth_cell_depth_m),
                      trans_select.depth_processed_m)

        # Create position array
        boat_select = getattr(transect.boat_vel, transect.boat_vel.selected)
        if boat_select is not None:
            if np.nansum(boat_select.valid_data[0]) > 0:
                boat_vel_x = boat_select.u_processed_mps
                boat_vel_y = boat_select.v_processed_mps
                track_x = boat_vel_x * transect.date_time.ens_duration_sec
                track_y = boat_vel_y * transect.date_time.ens_duration_sec
                track = np.nancumsum(np.sqrt(track_x ** 2 + track_y ** 2))
                track_array = np.tile(track, (self.u_processed_mps.shape[0], 1))

                # Determine index of all valid data
                valid_z = np.isnan(z) == False
                valid_combined = np.logical_and(valid, valid_z)

                u = interpolate.griddata(np.array([z[valid_combined].ravel(),
                                                   track_array[valid_combined].ravel()]).T,
                                         self.u_processed_mps[valid_combined].ravel(),
                                         (z, track_array))

                v = interpolate.griddata(np.array([z[valid_combined].ravel(),
                                                   track_array[valid_combined].ravel()]).T,
                                         self.v_processed_mps[valid_combined].ravel(),
                                         (z, track_array))

                self.u_processed_mps = np.tile(np.nan, self.u_mps.shape)
                self.u_processed_mps = np.tile(np.nan, self.u_mps.shape)
                processed_valid_cells = self.estimate_processed_valid_cells(transect)
                self.u_processed_mps[processed_valid_cells] = u[processed_valid_cells]
                self.v_processed_mps[processed_valid_cells] = v[processed_valid_cells]

    def interpolate_cells_trdi(self, transect):
        """Interpolates values for invalid cells using methods similar to WinRiver II.

        This function computes the velocity for the invalid cells using
        the methods in WinRiver II, but applied to velocity components.
        Although WinRiver II applies to discharge which theoretically is
        more correct, mathematically applying to discharge or velocity
        components is identical. By applying to velocity components the
        user can see the velocity data interpolated.
        Power fit uses the power fit equation and no slip uses linear interpolation.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """

        # Set property
        self.interpolate_cells = 'TRDI'

        # Construct variables
        depths = getattr(transect.depths, transect.depths.selected)
        valid = self.valid_data[0]
        cell_depth = depths.depth_cell_depth_m
        z_all = np.subtract(depths.depth_processed_m, cell_depth)
        z = np.copy(z_all)
        z[np.isnan(self.u_processed_mps)] = np.nan
        z_adj = np.tile(np.nan, z.shape)
        n_cells, n_ens = self.u_processed_mps.shape
        cell_size = depths.depth_cell_size_m
        exponent = transect.extrap.exponent
        bot_method = transect.extrap.bot_method

        for n in range(n_ens):

            # Identify first and last valid depth cell
            idx = np.where(valid[:, n] == True)[0]
            if len(idx) > 0:
                idx_first = idx[0]
                idx_last = idx[-1]
                idx_middle = np.where(valid[idx_first:idx_last + 1, n] == False)[0]

                # For invalid middle depth cells perform interpolation based on bottom method
                if len(idx_middle) > 0:
                    idx_middle = idx_middle + idx_first
                    z_adj[idx_middle, n] = z_all[idx_middle, n]

                    # Interpolate velocities using power fit
                    if bot_method == 'Power':
                        # Compute interpolated u-velocities
                        z2 = z[:, n] - (0.5 * cell_size[:, n])
                        z2[z2 < 0] = np.nan
                        coef = ((exponent + 1) * np.nansum(self.u_processed_mps[:, n] * cell_size[:, n], 0)) / \
                            np.nansum(((z[:, n] + 0.5 * cell_size[:, n]) ** (exponent + 1)) - (z2 ** (exponent + 1)), 0)

                        temp = coef * z_adj[:, n] ** exponent
                        self.u_processed_mps[idx_middle, n] = temp[idx_middle]
                        # Compute interpolated v-Velocities
                        coef = ((exponent + 1) * np.nansum(self.v_processed_mps[:, n] * cell_size[:, n])) / \
                            np.nansum(((z[:, n] + 0.5 * cell_size[:, n]) ** (exponent + 1)) - (z2 ** (exponent + 1)))
                        temp = coef * z_adj[:, n] ** exponent
                        self.v_processed_mps[idx_middle, n] = temp[idx_middle]

                    # Interpolate velocities using linear interpolation
                    elif bot_method == 'No Slip':
                        self.u_processed_mps[idx_middle, n] = np.interp(x=cell_depth[idx_middle, n],
                                                                        xp=cell_depth[valid[:, n], n],
                                                                        fp=self.u_processed_mps[valid[:, n], n])
                        self.v_processed_mps[idx_middle, n] = np.interp(x=cell_depth[idx_middle, n],
                                                                        xp=cell_depth[valid[:, n], n],
                                                                        fp=self.v_processed_mps[valid[:, n], n])

    def estimate_processed_valid_cells(self, transect):
        """Estimate the number of valid cells for invalid ensembles.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData

        Returns
        -------
        processed_valid_cells: np.ndarray(bool)
           Estimated valid cells
        """

        processed_valid_cells = np.copy(self.valid_data[0])
        valid_data_sum = np.nansum(processed_valid_cells, 0)
        invalid_ens_idx = np.where(valid_data_sum == 0)[0]
        n_invalid = len(invalid_ens_idx)
        depth_cell_depth = transect.depths.bt_depths.depth_cell_depth_m
        for n in range(n_invalid):

            # Find nearest valid ensembles on either side of invalid ensemble
            idx1 = np.where(valid_data_sum[:invalid_ens_idx[n]] > 0)[0]
            if len(idx1) > 0:
                idx1 = idx1[-1]
                # Find the last cell in the neighboring valid ensembles
                idx1_cell = np.where(processed_valid_cells[:, idx1] == True)[0][-1]
                # Determine valid cells for invalid ensemble
                idx1_cell_depth = depth_cell_depth[idx1_cell, idx1]
            else:
                idx1_cell_depth = 0

            idx2 = np.where(valid_data_sum[invalid_ens_idx[n]:] > 0)[0]
            if len(idx2) > 0:
                idx2 = idx2[0]
                idx2 = invalid_ens_idx[n] + idx2
                # Find the last cell in the neighboring valid ensembles
                idx2_cell = np.where(processed_valid_cells[:, idx2] == True)[0][-1]
                # Determine valid cells for invalid ensemble
                idx2_cell_depth = depth_cell_depth[idx2_cell, idx2]
            else:
                idx2_cell_depth = 0

            cutoff = np.nanmax([idx1_cell_depth, idx2_cell_depth])
            processed_valid_cells[depth_cell_depth[:, invalid_ens_idx[n]] < cutoff, invalid_ens_idx[n]] = True

            # Apply excluded distance
            processed_valid_cells = processed_valid_cells * self.valid_data[6, :, :]

        return processed_valid_cells

    def compute_snr_rng(self):
        """Computes the range between the average snr for all beams.
        The average is computed using only data above the side lobe cutoff.
        """
        if self.rssi_units == 'SNR':
            cells_above_sl = np.copy(self.cells_above_sl.astype(float))
            cells_above_sl[cells_above_sl < 0.5] = np.nan
            snr_adjusted = self.rssi * cells_above_sl
            snr_average = np.nanmean(snr_adjusted, 1)
            self.snr_rng = np.nanmax(snr_average, 0) - np.nanmin(snr_average, 0)

    def automated_beam_filter_old(self):
        """Older version of automatic beam filter. Not currently used.
        """

        # Create array indicating which cells do not have 4-beam solutions and all cells below side lobe are nan
        temp = copy.deepcopy(self)
        temp.filter_beam(4)
        valid_bool = temp.valid_data[5, :, :]
        valid = valid_bool.astype(float)
        valid[temp.cells_above_sl == False] = np.nan

        # Find cells with 3 beams solutions
        rows_3b, cols_3b = np.where(np.abs(valid) == 0)
        if len(rows_3b) > 0:
            # Find cells with 4 beams solutions
            valid_rows, valid_cols = np.where(valid == 1)

            valid_u = temp.u_mps[valid == 1]
            valid_v = temp.v_mps[valid == 1]
            # Use interpolate water velocity of cells with 3 beam solutions

            # The following code duplicates Matlab scatteredInterpolant which seems to only estimate along columns
            # as long as there is data in the ensemble above and below the value being estimated.
            row_numbers = np.linspace(0, valid.shape[0] - 1, valid.shape[0])
            n = 0
            for col in cols_3b:
                # If the cell has valid data above and below it linearly interpolate using data in that ensemble.
                # If not, use other means of interpolation.
                if np.any(valid_bool[rows_3b[n] + 1::, col]) and np.any(valid_bool[0:rows_3b[n], col]):
                    est_u = np.interp(x=rows_3b[n],
                                      xp=row_numbers[valid_bool[:, col]],
                                      fp=temp.u_mps[valid_bool[:, col], col])

                    est_v = np.interp(x=rows_3b[n],
                                      xp=row_numbers[valid_bool[:, col]],
                                      fp=temp.v_mps[valid_bool[:, col], col])
                else:
                    est_u = interpolate.griddata(np.array((valid_rows, valid_cols)).T, valid_u, (col, rows_3b[n]))
                    est_v = interpolate.griddata(np.array((valid_cols, valid_rows)).T, valid_v, (col, rows_3b[n]))

                u_ratio = (temp.u_mps[rows_3b[n], col] / est_u) - 1
                v_ratio = (temp.v_mps[rows_3b[n], col] / est_v) - 1
                if np.abs(u_ratio) < 0.5 or np.abs(v_ratio) < 0.5:
                    valid_bool[rows_3b[n], col] = True
                else:
                    valid_bool[rows_3b[n], col] = False
                n += 1
            self.valid_data[5, :, :] = valid_bool
        else:
            self.valid_data[5, :, :] = temp.valid_data[5, :, :]

        # Combine all filter data and update processed properties

        self.all_valid_data()

# Code from Aurelien
    def interpolate_cells_above(self, transect):
        """Interpolates values for invalid cells using below valid cell
        Written by Aurelien Despax
        Modified by dsm

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """

        # Set property
        self.interpolate_cells = 'Above'

        # Construct variables

        valid = self.valid_data[0]
        n_cells, n_ens = self.u_processed_mps.shape

        for n in range(n_ens):

            # Identify first and last valid depth cell
            idx = np.where(valid[:, n] == True)[0]
            if len(idx) > 0:
                idx_first = idx[0]
                idx_last = idx[-1]
                idx_middle = np.where(valid[idx_first:idx_last + 1, n] == False)[0]

                # For invalid middle depth cells assign value of shallower valid depth cell
                # TODO this assigns the value of the shallowest depth cell not the next valid depth cell
                if len(idx_middle) > 0:
                    idx_middle = idx_middle + idx_first
                    self.u_processed_mps[idx_middle, n] = self.u_processed_mps[idx_first, n]
                    self.v_processed_mps[idx_middle, n] = self.v_processed_mps[idx_first, n]

    def interpolate_cells_below(self, transect):
        """Interpolates values for invalid cells using above valid cell
        Written by Aurelien Despax
        Modified by dsm

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """

        # Set property
        self.interpolate_cells = 'Below'

        # Construct variables
        valid = self.valid_data[0]
        n_cells, n_ens = self.u_processed_mps.shape

        for n in range(n_ens):

            # Identify first and last valid depth cell
            idx = np.where(valid[:, n] == True)[0]
            if len(idx) > 0:
                idx_first = idx[0]
                idx_last = idx[-1]
                idx_middle = np.where(valid[idx_first:idx_last + 1, n] == False)[0]

                # For invalid middle depth cells assign the value of the next deeper valid depth cells
                # TODO this assigns the value of the shallowest depth cell not the next valid depth cell
                if len(idx_middle) > 0:
                    idx_middle = idx_middle + idx_first
                    self.u_processed_mps[idx_middle, n] = self.u_processed_mps[idx_last, n]
                    self.v_processed_mps[idx_middle, n] = self.v_processed_mps[idx_last, n]

    def interpolate_cells_before(self, transect):
        """Interpolates values for invalid cells using above valid cell
        Written by Aurelien Despax

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """

        # Set property
        self.interpolate_cells = 'Before'

        # Construct variables
        depths = getattr(transect.depths, transect.depths.selected)
        valid = self.valid_data[0]
        cell_depth = depths.depth_cell_depth_m
        z_all = np.subtract(depths.depth_processed_m, cell_depth)
        z = np.copy(z_all)
        z[np.isnan(self.u_processed_mps)] = np.nan
        z_adj = np.tile(np.nan, z.shape)
        n_cells, n_ens = self.u_processed_mps.shape

        for n in range(n_ens):

            # Identify first and last valid depth cell
            idx = np.where(valid[:, n] == True)[0]
            if len(idx) > 0:
                idx_first = idx[0]
                idx_last = idx[-1]
                idx_middle = np.where(valid[idx_first:idx_last + 1, n] == False)[0]

                # For invalid middle depth cells perform interpolation based on bottom method
                if len(idx_middle) > 0:
                    idx_middle = idx_middle + idx_first
                    z_adj[idx_middle, n] = z_all[idx_middle, n]

                    # Interpolate velocities using linear interpolation
                    self.u_processed_mps[idx_middle, n] = self.u_processed_mps[idx_middle, n - 1]
                    self.v_processed_mps[idx_middle, n] = self.v_processed_mps[idx_middle, n - 1]

    def interpolate_cells_after(self, transect):
        """Interpolates values for invalid cells using above valid cell
        Written by Aurelien Despax

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        """

        # Set property
        self.interpolate_cells = 'After'

        # Construct variables
        depths = getattr(transect.depths, transect.depths.selected)
        valid = self.valid_data[0]
        cell_depth = depths.depth_cell_depth_m
        z_all = np.subtract(depths.depth_processed_m, cell_depth)
        z = np.copy(z_all)
        z[np.isnan(self.u_processed_mps)] = np.nan
        z_adj = np.tile(np.nan, z.shape)
        n_cells, n_ens = self.u_processed_mps.shape

        for n in list(reversed(list(range(n_ens)))):

            # Identify first and last valid depth cell
            idx = np.where(valid[:, n] == True)[0]
            if len(idx) > 0:
                idx_first = idx[0]
                idx_last = idx[-1]
                idx_middle = np.where(valid[idx_first:idx_last + 1, n] == False)[0]

                # For invalid middle depth cells perform interpolation based on bottom method
                if len(idx_middle) > 0:
                    idx_middle = idx_middle + idx_first
                    z_adj[idx_middle, n] = z_all[idx_middle, n]

                    # Interpolate velocities using linear interpolation
                    if (n_ens > (n + 1)):
                        self.u_processed_mps[idx_middle, n] = self.u_processed_mps[idx_middle, n + 1]
                        self.v_processed_mps[idx_middle, n] = self.v_processed_mps[idx_middle, n + 1]