import copy
import numpy as np
from numpy.matlib import repmat
from MiscLibs.common_functions import iqr
from MiscLibs.robust_loess import rloess
from MiscLibs.non_uniform_savgol import non_uniform_savgol


class DepthData(object):
    """Process and store depth data.
    Supported sources include bottom track
    vertical beam, and external depth sounder.

    Attributes
    ----------
        depth_orig_m: np.array
            Original multi-beam depth data from transect file (includes draft_orig) in meters.
        depth_beams_m: np.array
            Depth data from transect file adjusted for any draft changes, in meters.
        depth_processed_m: np.array
            Depth data filtered and interpolated.
        depth_freq_kHz: float
            Defines acoustic frequency used to measure depth.
        depth_invalid_index:
            Index of depths marked invalid.
        depth_source: str
            Source of depth data ("BT", "VB", "DS").
        depth_source_ens: np.array(object)
            Source of each depth value ("BT", "VB", "DS", "IN").
        draft_orig_m: float
            Original draft from data files, in meters.
        draft_use_m: float
            Draft used in computation of depth_beams_m and depth_cell_depths_m.
        depth_cell_depth_orig_m: np.array
            Depth to centerline of depth cells in raw data, in meters.
        depth_cell_depth_m: np.array
            Depth to centerline of depth cells adjusted for draft or speed of sound changes, in meters.
        depth_cell_size_orig_m: np.array
            Size of depth cells in meters from raw data, in meters.
        depth_cell_size_m:
            Size of depth cells adjusted for draft or speed of sound changes, in meters.
        smooth_depth: np.array
            Smoothed beam depth, in meters.
        smooth_upper_limit: np.array
            Smooth function upper limit of window, in meters.
        smooth_lower_limit: np.array
            Smooth function lower limit or window, in meters.
        avg_method:str
            Defines averaging method: "Simple", "IDW", only applicable to bottom track.
        filter_type: str
            Type of filter: "None", "TRDI", "Smooth".
        interp_type: str
            Type of interpolation: "None", "Linear", "Smooth".
        valid_data_method: str
            QRev or TRDI.
        valid_data: np.array
            Logical array of valid mean depth for each ensemble.
        valid_beams: np.array
            Logical array, 1 row for each beam identifying valid data.
    """
    
    def __init__(self):
        """Initialize attributes.
        """

        self.depth_orig_m = None  # Original multi-beam depth data from transect file (includes draft_orig) in meters
        self.depth_beams_m = None  # Depth data from transect file adjusted for any draft changes, in meters
        self.depth_processed_m = None  # Depth data filtered and interpolated
        self.depth_freq_kHz = None  # Defines ADCP frequency used of each raw data point
        self.depth_invalid_index = None  # Index of depths marked invalid
        self.depth_source = None  # Source of depth data ("BT", "VB", "DS")
        self.depth_source_ens = None  # Source of each depth value ("BT", "VB", "DS", "IN")
        self.draft_orig_m = None  # Original draft from data files, in meters
        self.draft_use_m = None  # Draft used in computation of depth_beams_m and depth_cell_depths_m
        self.depth_cell_depth_orig_m = None  # Depth cell range from the transducer, in meters
        self.depth_cell_depth_m = None  # Depth to centerline of depth cells, in meters
        self.depth_cell_size_orig_m = None  # Size of depth cells in meters from raw data
        self.depth_cell_size_m = None  # Size of depth cells in meters
        self.smooth_depth = None  # Smoothed beam depth
        self.smooth_upper_limit = None  # Smooth function upper limit of window
        self.smooth_lower_limit = None  # Smooth function lowerl limit or window
        self.avg_method = None  # Defines averaging method: "Simple", "IDW"
        self.filter_type = None  # Type of filter: "None", "TRDI", "Smooth"
        self.interp_type = None  # Type of interpolation: "None", "Linear", "Smooth"
        self.valid_data_method = None  # QRev or TRDI
        self.valid_data = None  # Logical array of valid mean depth for each ensemble
        self.valid_beams = None  # Logical array, 1 row for each beam identifying valid data
        
    def populate_data(self, depth_in, source_in, freq_in, draft_in, cell_depth_in, cell_size_in):
        """Stores data in DepthData.

        Parameters
        ----------
        depth_in: np.array
            Raw depth data, in meters.
        source_in: str
            Source of raw depth data.
        freq_in: float
            Acoustic frequency used to measure depths, in kHz.
        draft_in: float
            Draft of transducer used to measure depths, in meters.
        cell_depth_in: np.array
            Depth to centerline of each depth cell, in meters. If source does not have depth cells the depth cell depth
            from bottom track should be used.
        cell_size_in: np.array
            Size of each depth cell, in meters. If source does not have depth cells the depth cell size
            from bottom track should be used.
        """

        self.depth_orig_m = depth_in
        self.depth_beams_m = depth_in
        self.depth_source = source_in
        self.depth_source_ens = np.array([source_in] * depth_in.shape[-1], dtype=object)
        self.depth_freq_kHz = freq_in
        self.draft_orig_m = draft_in
        self.draft_use_m = draft_in
        self.filter_type = 'None'
        self.interp_type = 'None'
        self.valid_data_method = 'QRev'
        
        # For BT data set method to average multiple beam depths
        if source_in == 'BT':
            self.avg_method = 'IDW'
        else:
            self.avg_method = 'None'

        # Store cell data
        self.depth_cell_depth_orig_m = cell_depth_in
        self.depth_cell_size_orig_m = cell_size_in
        self.depth_cell_size_m = cell_size_in
        self.depth_cell_depth_m = cell_depth_in

        # Remove all filters to initialize data
        self.apply_filter('dummy', filter_type='Off')

    def populate_from_qrev_mat(self, mat_data):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        self.depth_processed_m = mat_data.depthProcessed_m
        self.depth_freq_kHz = mat_data.depthFreq_Hz

        # Support for older files that may not have had invalid index
        if len(mat_data.depthInvalidIndex) > 0:
            self.depth_invalid_index = mat_data.depthInvalidIndex
        else:
            self.depth_invalid_index = None

        self.depth_source = mat_data.depthSource
        self.depth_source_ens = mat_data.depthSourceEns
        self.draft_orig_m = mat_data.draftOrig_m
        self.draft_use_m = mat_data.draftUse_m
        self.depth_cell_depth_orig_m = mat_data.depthCellDepthOrig_m
        self.depth_cell_depth_m = mat_data.depthCellDepth_m
        if hasattr(mat_data, "depthCellSizeOrig_m"):
            self.depth_cell_size_orig_m = mat_data.depthCellSizeOrig_m
        else:
            self.depth_cell_size_orig_m =  mat_data.depthCellSize_m
        self.depth_cell_size_m = mat_data.depthCellSize_m

        # Configure arrays properly for VB and DS
        if mat_data.depthSource == 'BT':
            self.depth_beams_m = mat_data.depthBeams_m
            self.depth_orig_m = mat_data.depthOrig_m
            self.smooth_depth = mat_data.smoothDepth
            self.smooth_upper_limit = mat_data.smoothUpperLimit
            self.smooth_lower_limit = mat_data.smoothLowerLimit
        else:
            self.depth_beams_m = mat_data.depthBeams_m.reshape(1, -1)
            self.depth_orig_m = mat_data.depthOrig_m.reshape(1, -1)
            self.smooth_depth = mat_data.smoothDepth.reshape(1, -1)
            self.smooth_upper_limit = mat_data.smoothUpperLimit.reshape(1, -1)
            self.smooth_lower_limit = mat_data.smoothLowerLimit.reshape(1, -1)

        self.avg_method = mat_data.avgMethod
        self.filter_type = mat_data.filterType
        self.interp_type = mat_data.interpType
        self.valid_data_method = mat_data.validDataMethod
        if type(mat_data.validData) is int:
            self.valid_data = np.array([mat_data.validData]).astype(bool)
        else:
            self.valid_data = mat_data.validData.astype(bool)

        # Reshape array for vertical beam and depth sounder
        if len(mat_data.validBeams.shape) < 2:
            self.valid_beams = mat_data.validBeams.reshape(1, -1)
        else:
            self.valid_beams = mat_data.validBeams

        self.valid_beams = self.valid_beams.astype(bool)

        if len(self.depth_beams_m.shape) == 1:
            self.depth_beams_m = self.depth_beams_m.reshape(self.depth_beams_m.shape[0], 1)
            self.depth_cell_depth_m = self.depth_cell_depth_m.reshape(self.depth_cell_depth_m.shape[0], 1)
            self.depth_cell_depth_orig_m = self.depth_cell_depth_orig_m.reshape(
                self.depth_cell_depth_orig_m.shape[0], 1)
            self.depth_cell_size_m = self.depth_cell_size_m.reshape(self.depth_cell_size_m.shape[0], 1)
            self.depth_cell_size_orig_m = self.depth_cell_size_orig_m.reshape(self.depth_cell_size_orig_m.shape[0], 1)
            self.depth_orig_m = self.depth_orig_m.reshape(self.depth_orig_m.shape[0], 1)
            self.depth_processed_m = np.array([self.depth_processed_m])
            self.smooth_depth = self.smooth_depth.reshape(self.smooth_depth.shape[0], 1)
            self.smooth_lower_limit = self.smooth_lower_limit.reshape(self.smooth_lower_limit.shape[0], 1)
            self.smooth_upper_limit = self.smooth_upper_limit.reshape(self.smooth_upper_limit.shape[0], 1)
            self.valid_data = np.array([self.valid_data])
            self.depth_source_ens = np.array([mat_data.depthSourceEns])

    def change_draft(self, draft):
        """Changes the draft for object
        
        draft: new draft for object
        """
        # Compute draft change
        draft_change = draft - self.draft_use_m
        self.draft_use_m = draft
        
        # Apply draft to ensemble depths if BT or VB
        if self.depth_source != 'DS':
            self.depth_beams_m = self.depth_beams_m + draft_change
            self.depth_processed_m = self.depth_processed_m + draft_change 
            
        # Apply draft to depth cell locations
        if len(self.depth_cell_depth_m) > 0:
            self.depth_cell_depth_m = self.depth_cell_depth_m + draft_change

    def add_cell_data(self, bt_depths):
        """Adds cell data to depth objects with no cell data
        such as the vertical beam and depth sounder.  This allows
        a single object to contain all the required depth data

        Parameters
        ----------
        bt_depths: DepthData
            Object of DepthData with bottom track depths
        """
        
        self.depth_cell_depth_orig_m = bt_depths.depth_cell_depth_orig_m
        self.depth_cell_size_m = bt_depths.depth_cell_size_m
        self.depth_cell_depth_m = bt_depths.depth_cell_depth_m

    def compute_avg_bt_depth(self, method=None):
        """Computes average depth for BT_Depths

        Parameters
        ----------
        method: str
            Averaging method (Simple or IDW)
        """

        if method is not None:
            self.avg_method = method

        # Get valid depths
        depth = np.copy(self.depth_beams_m)
        depth[self.valid_beams == False] = np.nan

        # Compute average depths
        self.depth_processed_m = DepthData.average_depth(depth, self.draft_use_m, self.avg_method)

        # Set depths to nan if depth are not valid beam depths
        self.depth_processed_m[np.equal(self.valid_data, False)] = np.nan

    def apply_filter(self, transect, filter_type=None):
        """Coordinate the application of depth filters.

        Parameters
        ----------
        transect: TransectData
            Object of transect data.
        filter_type: str
            Type of filter to apply (None, Smooth, TRDI).
        """

        # Compute selected filter
        if filter_type == 'Off' or filter_type is None:
            # No filter
            self.filter_none()
            # Savitzky-Golay
        elif filter_type == 'SavGol':
            self.filter_savgol(transect)
        elif filter_type == 'Smooth':
            # Smooth filter
            self.filter_smooth(transect)
        elif filter_type == 'TRDI' and self.depth_source == 'BT':
            # TRDI filter for multiple returns
            self.filter_trdi()
            self.filter_type = 'TRDI'
            
        self.valid_mean_data()

        # Update processed depth with filtered results
        if self.depth_source == 'BT':
            # Multiple beams require averaging to obtain 1-D array
            self.compute_avg_bt_depth()
        else:
            # Single beam (VB or DS) save to 1-D array
            self.depth_processed_m = np.array(self.depth_beams_m[0, :])
            self.depth_processed_m[np.squeeze(np.equal(self.valid_data, 0))] = np.nan
            
    def apply_interpolation(self, transect, method=None):
        """Coordinates application of interpolations

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        method: str
            Type of interpolation to apply (None, HoldLast, Smooth, Linear)
        """
        
        # Determine interpolation to apply
        if method is None:
            method = self.interp_type
            
        # Apply selected interpolation
        self.interp_type = method
        # No filtering
        if method == 'None':
            self.interpolate_none()

        # Hold last valid depth indefinitely
        elif method == 'HoldLast':
            self.interpolate_hold_last()

        # Use values form a Loess smooth
        elif method == 'Smooth':
            self.interpolate_smooth()

        # Linear interpolation
        else:
            self.interpolate_linear(transect=transect)
            
        # Identify ensembles with interpolated depths
        idx = np.where(self.valid_data[:] == False)
        if len(idx[0]) > 0:
            idx = idx[0]
            idx2 = np.where(np.isnan(self.depth_processed_m[idx]) == False)
            if len(idx2) > 0:
                idx2 = idx2[0]
                self.depth_source_ens[idx[idx2]] = 'IN'
        
    def apply_composite(self, comp_depth, comp_source):
        """Applies the data from CompDepth computed in DepthStructure
        to DepthData object

        Parameters
        ----------
        comp_depth: np.array(float)
            Composite depth computed in DepthStructure
        comp_source: str
            Source of composite depth (BT, VB, DS)
        """
        
        # Assign composite depth to property
        self.depth_processed_m = comp_depth
        
        # Assign appropriate composite source for each ensemble
        self.depth_source_ens[comp_source == 1] = 'BT'
        self.depth_source_ens[comp_source == 2] = 'VB'
        self.depth_source_ens[comp_source == 3] = 'DS'
        self.depth_source_ens[comp_source == 4] = 'IN'
        self.depth_source_ens[comp_source == 0] = 'NA'
        
    def sos_correction(self, ratio):
        """Correct depth for new speed of sound setting

        Parameters
        ----------
        ratio: float
            Ratio of new to old speed of sound value
        """
        
        # Correct unprocessed depths
        self.depth_beams_m = self.draft_use_m+np.multiply(self.depth_beams_m-self.draft_use_m, ratio)
        
        # Correct processed depths
        self.depth_processed_m = self.draft_use_m+np.multiply(self.depth_processed_m-self.draft_use_m, ratio)
        
        # Correct cell size and location
        self.depth_cell_size_m = np.multiply(self.depth_cell_size_m, ratio)
        self.depth_cell_depth_m = self.draft_use_m + np.multiply(self.depth_cell_depth_m - self.draft_use_m, ratio)
        
    def valid_mean_data(self):
        """Determines if raw data are sufficient to compute a valid depth without interpolation.
        """
        
        if self.depth_source == 'BT':
            self.valid_data = np.tile(True, self.valid_beams.shape[1])
            nvalid = np.sum(self.valid_beams, axis=0)
            
            if self.valid_data_method == 'TRDI':
                self.valid_data[nvalid < 3] = False
            else:
                self.valid_data[nvalid < 2] = False
        else:
            self.valid_data = self.valid_beams[0, :]
            
    def filter_none(self):
        """Applies no filter to depth data. Removes filter if one was applied.
        """
        
        # Set all ensembles to have valid data
        if len(self.depth_beams_m.shape) > 1:
            self.valid_beams = np.tile(True, self.depth_beams_m.shape)
        else:
            self.valid_beams = np.tile(True, (1, self.depth_beams_m.shape[0]))
        
        # Set ensembles with no depth data to invalid
        self.valid_beams[self.depth_beams_m == 0] = False
        self.valid_beams[np.isnan(self.depth_beams_m)] = False
        
        self.filter_type = 'None'
        
    def filter_smooth(self, transect):
        """This filter uses a moving InterQuartile Range filter on residuals from a
        robust Loess smooth of the depths in each beam to identify unnatural spikes in the depth
        measurements from each beam.  Each beam is filtered independently.  The filter
        criteria are set to be the maximum of the IQR filter, 5% of the measured depth, or 0.1 meter

        Parameters
        ----------
        transect: TransectData
            Object of TransectData

        Notes
        -----
        half_width - number of points to each side of target point used in computing IQR.
            This is the raw number of points actual points used may be less if some are bad.

        multiplier - number multiplied times the IQR to determine the filter criteria
        
        """

        # If the smoothed depth has not been computed
        if self.smooth_depth is None:
            
            # Set filter characteristics
            self.filter_type = 'Smooth'
            cycles = 3
            half_width = 10
            multiplier = 15
            
            # Determine number of beams
            if len(self.depth_orig_m.shape) > 1:
                n_beams, n_ensembles = self.depth_orig_m.shape[0], self.depth_orig_m.shape[1]
                depth_raw = np.copy(self.depth_orig_m)
            else:
                n_beams = 1
                n_ensembles = self.depth_orig_m.shape[0]
                depth_raw = np.copy(np.reshape(self.depth_orig_m, (1, n_ensembles)))

            # Set bad depths to nan
            depth = repmat([np.nan], n_beams, n_ensembles)

            # Arrays initialized
            depth_smooth = repmat([np.nan], n_beams, n_ensembles)
            depth_res = repmat([np.nan], n_beams, n_ensembles)
            upper_limit = repmat([np.nan], n_beams, n_ensembles)
            lower_limit = repmat([np.nan], n_beams, n_ensembles)
            depth_filtered = depth
            depth[depth_raw > 0] = depth_raw[depth_raw > 0]

            # Create position array
            boat_vel_selected = getattr(transect.boat_vel, transect.boat_vel.selected)
            if boat_vel_selected is not None:
                track_x = boat_vel_selected.u_processed_mps * transect.date_time.ens_duration_sec
                track_y = boat_vel_selected.v_processed_mps * transect.date_time.ens_duration_sec
            else:
                track_x = np.nan
                track_y = np.nan

            idx = np.where(np.isnan(track_x))
            if len(idx[0]) < 2:
                x = np.nancumsum(np.sqrt(track_x**2+track_y**2))
            else:
                x = np.nancumsum(transect.date_time.ens_duration_sec)
                
            # Loop for each beam, smooth is applied to each beam
            for j in range(n_beams):
                # At least 50% of the data in a beam must be valid to apply the smooth
                if np.nansum((np.isnan(depth_filtered[j]) == False) / depth_filtered.shape[0]) > .5:
                    # Compute residuals based on robust loess smooth
                    if len(x) > 1:
                        # Fit smooth
                        try:
                            smooth_fit = rloess(x, depth_filtered[j, :], 20)
                            depth_smooth[j, :] = smooth_fit
                        except ValueError:
                            depth_smooth[j, :] = depth_filtered[j, :]
                    else:
                        depth_smooth[j, :] = depth_filtered[j, :]
                    
                    depth_res[j, :] = depth[j, :] - depth_smooth[j, :]
                    
                    # Run the filter multiple times
                    for n in range(cycles - 1):
                        
                        # Compute inner quartile range
                        fill_array = DepthData.run_iqr(half_width, depth_res[j, :])

                        # Compute filter criteria and apply appropriate
                        criteria = multiplier * fill_array
                        idx = np.where(criteria < np.max(np.vstack((depth[j, :] * .05,
                                                                    np.ones(depth.shape) / 10)), 0))[0]
                        if len(idx) > 0:
                            criteria[idx] = np.max(np.vstack((depth[j, idx] * .05, np.ones(idx.shape) / 10)), 0)

                        # Compute limits
                        upper_limit[j] = depth_smooth[j, :] + criteria
                        lower_limit[j] = depth_smooth[j, :] - criteria

                        bad_idx = np.where(
                           np.logical_or(np.greater(depth[j], upper_limit[j]), np.less(depth[j], lower_limit[j])))[0]
                        # Update depth matrix
                        depth_res[j, bad_idx] = np.nan
                        
                else:
                    depth_smooth[j] = np.nan
                    upper_limit[j] = np.nan
                    lower_limit[j] = np.nan
                    
            # Save smooth results to avoid recomputing them if needed later
            self.smooth_depth = depth_smooth
            self.smooth_upper_limit = upper_limit
            self.smooth_lower_limit = lower_limit

        # Reset valid data
        self.filter_none()
        
        # Set filter type
        self.filter_type = 'Smooth'
        
        # Determine number of beams
        if len(self.depth_orig_m.shape) > 1:
            n_beams, n_ensembles = self.depth_orig_m.shape[0], self.depth_orig_m.shape[1]
            depth_raw = np.copy(self.depth_orig_m)
        else:
            n_beams = 1
            n_ensembles = self.depth_orig_m.shape[0]
            depth_raw = np.reshape(self.depth_orig_m, (1, n_ensembles))

        depth_res = repmat([np.nan], n_beams, n_ensembles)

        # Set bad depths to nan
        depth = repmat(np.nan, depth_raw.shape[0], depth_raw.shape[1])
        depth[depth_raw > 0] = depth_raw[depth_raw > 0]
        
        # Apply filter
        for j in range(n_beams):
            if np.nansum(self.smooth_upper_limit[j]) > 0:
                bad_idx = np.where(
                    np.logical_or(np.greater(depth[j], self.smooth_upper_limit[j]),
                                  np.less(depth[j], self.smooth_lower_limit[j])))[0]
                # Update depth matrix
                depth_res[j, bad_idx] = np.nan

            else:
                bad_idx = np.isnan(depth[j])

            # Update valid data matrix
            self.valid_beams[j, bad_idx] = False

    def filter_savgol(self, transect):
        """This filter uses a moving InterQuartile Range filter on residuals from a
        a Savitzky-Golay filter on y with non-uniform spaced x
        of the depths in each beam to identify unnatural spikes in the depth
        measurements from each beam.  Each beam is filtered independently.  The filter
        criteria are set to be the maximum of the IQR filter, 5% of the measured depth, or 0.1 meter

        Parameters
        ----------
        transect: TransectData
            Object of TransectData

        Notes
        -----
        half_width - number of points to each side of target point used in computing IQR.
            This is the raw number of points actual points used may be less if some are bad.

        multiplier - number multiplied times the IQR to determine the filter criteria

        """

        # Determine number of beams
        if len(self.depth_orig_m.shape) > 1:
            # For slant beams
            n_beams, n_ensembles = self.depth_orig_m.shape[0], self.depth_orig_m.shape[1]
            depth_raw = np.copy(self.depth_orig_m)
        else:
            # For vertical beam or depth sounder
            n_beams = 1
            n_ensembles = self.depth_orig_m.shape[0]
            depth_raw = np.copy(np.reshape(self.depth_orig_m, (1, n_ensembles)))

        # Set bad depths to nan
        depth = repmat([np.nan], n_beams, n_ensembles)
        depth[depth_raw > 0] = depth_raw[depth_raw > 0]

        # If the smoothed depth has not been computed
        if self.smooth_depth is None:

            # Set filter characteristics
            self.filter_type = 'SavGol'
            cycles = 3
            half_width = 10
            multiplier = 15

            # Arrays initialized
            depth_smooth = repmat([np.nan], n_beams, n_ensembles)
            depth_res = repmat([np.nan], n_beams, n_ensembles)
            upper_limit = repmat([np.nan], n_beams, n_ensembles)
            lower_limit = repmat([np.nan], n_beams, n_ensembles)

            # Create position array. If there are insufficient track data use elapsed time
            boat_vel_selected = getattr(transect.boat_vel, transect.boat_vel.selected)
            if boat_vel_selected is not None and \
                    np.nansum(np.isnan(boat_vel_selected.u_processed_mps)) < 2:
                track_x = boat_vel_selected.u_processed_mps * transect.date_time.ens_duration_sec
                track_y = boat_vel_selected.v_processed_mps * transect.date_time.ens_duration_sec
                x = np.nancumsum(np.sqrt(track_x ** 2 + track_y ** 2))
            else:
                x = np.nancumsum(transect.date_time.ens_duration_sec)

            # Loop for each beam, smooth is applied to each beam
            for j in range(n_beams):
                # At least 50% of the data in a beam must be valid to apply the smooth
                if np.nansum((np.isnan(depth[j, :]) == False) / depth.shape[0]) > .5:

                    # Compute residuals based on non-uniform Savitzky-Golay
                    try:
                        valid_depth_idx = np.logical_not(np.isnan(depth[j, :]))
                        x_fit = x[valid_depth_idx]
                        y_fit = depth[j, valid_depth_idx]
                        smooth_fit = non_uniform_savgol(x_fit, y_fit, 15, 3)
                        depth_smooth[j, valid_depth_idx] = smooth_fit
                    except ValueError:
                        depth_smooth[j, :] = depth[j, :]

                    depth_res[j, :] = depth[j, :] - depth_smooth[j, :]

                    # Run the filter multiple times
                    for n in range(cycles - 1):

                        # Compute inner quartile range
                        fill_array = DepthData.run_iqr(half_width, depth_res[j, :])

                        # Compute filter criteria
                        criteria = multiplier * fill_array

                        # Adjust criteria so that it is never less than 5% of depth or 0.1 m which ever is greater
                        idx = np.where(criteria < np.max(np.vstack((depth[j, :] * .05,
                                                                    np.ones(depth.shape) / 10)), 0))[0]
                        if len(idx) > 0:
                            criteria[idx] = np.max(np.vstack((depth[j, idx] * .05, np.ones(idx.shape) / 10)), 0)

                        # Compute limits
                        upper_limit[j] = depth_smooth[j, :] + criteria
                        lower_limit[j] = depth_smooth[j, :] - criteria

                        bad_idx = np.where(np.logical_or(np.greater(depth[j], upper_limit[j]),
                                                         np.less(depth[j], lower_limit[j])))[0]
                        # Update residual matrix
                        depth_res[j, bad_idx] = np.nan

                else:
                    depth_smooth[j] = np.nan
                    upper_limit[j] = np.nan
                    lower_limit[j] = np.nan

            # Save smooth results to avoid recomputing them if needed later
            self.smooth_depth = depth_smooth
            self.smooth_upper_limit = upper_limit
            self.smooth_lower_limit = lower_limit

        # Reset valid data
        self.filter_none()

        # Set filter type
        self.filter_type = 'SavGol'

        # Apply filter
        for j in range(n_beams):
            if np.nansum(self.smooth_upper_limit[j]) > 0:
                bad_idx = np.where(
                    np.logical_or(np.greater(depth[j], self.smooth_upper_limit[j]),
                                  np.less(depth[j], self.smooth_lower_limit[j])))[0]
            else:
                bad_idx = np.isnan(depth[j])

            # Update valid data matrix
            self.valid_beams[j, bad_idx] = False

    def interpolate_none(self):
        """Applies no interpolation.
        """
        
        # Compute processed depth without interpolation
        if self.depth_source == 'BT':
            # Bottom track methods
            self.compute_avg_bt_depth()
        else:
            # Vertical beam or depth sounder depths
            self.depth_processed_m = self.depth_beams_m[0, :]
            
        self.depth_processed_m[np.squeeze(np.equal(self.valid_data, False))] = np.nan
        
        # Set interpolation type
        self.interp_type = 'None'
        
    def interpolate_hold_last(self):
        """This function holds the last valid value until the next valid data point.
        """
        
        # Get number of ensembles
        n_ensembles = len(self.depth_processed_m)
        
        # Process data by ensemble
        for n in range(1, n_ensembles):
            
            # If current ensemble's depth is invalid assign depth from previous example
            if not self.valid_data[n]:
                self.depth_processed_m[n] = self.depth_processed_m[n-1]

    def interpolate_next(self):
        """This function back fills with the next valid value.
        """

        # Get number of ensembles
        n_ens = len(self.depth_processed_m)

        # Process data by ensemble
        for n in np.arange(0, n_ens-1)[::-1]:

            # If current ensemble's depth is invalid assign depth from previous example
            if not self.valid_data[n]:
                self.depth_processed_m[n] = self.depth_processed_m[n + 1]

    def interpolate_smooth(self):
        """Apply interpolation based on the robust loess smooth
        """
        
        self.interp_type = 'Smooth'
        
        # Get depth data from object
        depth_new = self.depth_beams_m
        
        # Update depth data with interpolated depths
        depth_new[not self.valid_beams] = self.smooth_depth[not self.valid_beams]
        
        # Compute processed depths with interpolated values
        if self.depth_source == 'BT':
            # Temporarily change self.depth_beams_m to compute average
            # for bottom track based depths
            temp_save = copy.deepcopy(self.depth_beams_m)
            self.depth_beams_m = depth_new
            self.compute_avg_bt_depth()
            self.depth_beams_m = temp_save

        else:
            # Assignment for VB or DS
            self.depth_processed_m = depth_new[0, :]
            
    def interpolate_linear(self, transect):
        """Apply linear interpolation
        """
        
        # Set interpolation type
        self.interp_type = 'Linear'

        # Create position array
        select = getattr(transect.boat_vel, transect.boat_vel.selected)
        if select is not None:
            boat_vel_x = select.u_processed_mps
            boat_vel_y = select.v_processed_mps
            track_x = boat_vel_x * transect.date_time.ens_duration_sec
            track_y = boat_vel_y * transect.date_time.ens_duration_sec
        else:
            select = getattr(transect.boat_vel, 'bt_vel')
            track_x = np.tile(np.nan, select.u_processed_mps.shape)
            track_y = np.tile(np.nan, select.v_processed_mps.shape)
              
        idx = np.where(np.isnan(track_x[1:]))
        
        # If the navigation reference has no gaps use it for interpolation, if not use time
        if len(idx[0]) < 1:
            x = np.nancumsum(np.sqrt(track_x**2 + track_y**2))
        else:
            # Compute accumulated time
            x = np.nancumsum(transect.date_time.ens_duration_sec)
            
        # Determine number of beams
        n_beams = self.depth_beams_m.shape[0]
        
#       Create strict monotonic arrays for depth and track by identifying duplicate
#       track values.  The first track value is used and the remaining duplicates
#       are set to nan.  The depth assigned to that first track value is the average
#       of all duplicates.  The depths for the duplicates are then set to nan.  Only
#       valid strictly monotonic track and depth data are used for the input in to linear
#       interpolation.   Only the interpolated data for invalid depths are added
#       to the valid depth data to create depth_new
        
        depth_mono = copy.deepcopy(self.depth_beams_m)
        x_mono = x
        
        idx0 = np.where(np.diff(x) == 0)[0]
        if len(idx0) > 0:
            if len(idx0) > 1:
                # Split array into subarrays in proper sequence e.g [[2,3,4],[7,8,9]] etc.
                idx1 = np.add(np.where(np.diff(idx0) != 1)[0], 1)
                group = np.split(idx0, idx1)

            else:
                # Group of only 1 point
                group = np.array([idx0])

            # Replace repeated values with mean
            n_group = len(group)
            for k in range(n_group):
                indices = group[k]
                indices = np.append(indices, indices[-1] + 1)
                depth_avg = np.nanmean(depth_mono[:, indices], axis=1)
                depth_mono[:, indices[0]] = depth_avg
                depth_mono[:, indices[1:]] = np.nan
                x[indices[1:]] = np.nan
                
        # Interpolate each beam
        depth_new = copy.deepcopy(self.depth_beams_m)
        for n in range(n_beams):
            # Determine ensembles with valid depth data
            valid_depth_mono = np.logical_not(np.isnan(depth_mono[n]))
            valid_x_mono = np.logical_not(np.isnan(x_mono))
            valid_data = copy.deepcopy(self.valid_beams[n])
            valid = np.vstack([valid_depth_mono, valid_x_mono, valid_data])
            valid = np.all(valid, 0)

            if np.sum(valid) > 1:
                # Compute interpolation function from all valid data
                depth_int = np.interp(x_mono, x_mono[valid], depth_mono[n, valid], left=np.nan, right=np.nan)
                # Fill in invalid data with interpolated data
                depth_new[n, np.logical_not(self.valid_beams[n])] = depth_int[np.logical_not(self.valid_beams[n])]

        if self.depth_source == 'BT':
            # Bottom track depths
            self.depth_processed_m = self.average_depth(depth_new, self.draft_use_m, self.avg_method)
        else:
            # Vertical beam or depth sounder depths
            self.depth_processed_m = np.copy(depth_new[0, :])

    @staticmethod
    def average_depth(depth, draft, method):
        """Compute average depth from bottom track beam depths.

        Parameters
        ----------
        depth: np.array(float)
            Individual beam depths for each beam in each ensemble including the draft
        draft: float
            Draft of ADCP
        method: str
            Averaging method (Simple, IDW)
        
        Returns
        -------
        avg_depth: np.array(float)
            Average depth for each ensemble
        
        """
        if method == 'Simple':
            avg_depth = np.nanmean(depth, 0)
        else:
            # Compute inverse weighted mean depth
            rng = depth - draft
            w = 1 - np.divide(rng, np.nansum(rng, 0))
            avg_depth = draft+np.nansum(np.divide((rng * w), np.nansum(w, 0)), 0)
            avg_depth[avg_depth == draft] = np.nan

        return avg_depth

    @staticmethod
    def run_iqr(half_width, data):
        """Computes a running Innerquartile Range
        The routine accepts a column vector as input.  "halfWidth" number of data
        points for computing the Innerquartile Range are selected before and
        after the target data point, but no including the target data point.
        Near the ends of the series the number of points before or after are reduced.
        Nan in the data are counted as points.  The IQR is computed on the slected
        subset of points.  The process occurs for each point in the provided column vector.
        A column vector with the computed IQR at each point is returned.

        Parameters
        ----------
        half_width: int
            Number of ensembles before and after current ensemble which are used to compute the IQR
        data: np.array(float)
            Data for which the IQR is computed
        """
        npts = len(data)
        half_width = int(half_width)

        if npts < 20:
            half_width = int(np.floor(npts / 2))

        iqr_array = []

        # Compute IQR for each point
        for n in range(npts):

            # Sample selection for 1st point
            if n == 0:
                sample = data[1:1 + half_width]

            # Sample selection a end of data set
            elif n + half_width > npts:
                sample = np.hstack([data[n - half_width - 1:n - 1], data[n:npts]])

            # Sample selection at beginning of data set
            elif half_width >= n + 1:
                sample = np.hstack([data[0:n], data[n + 1:n + half_width + 1]])

            # Sample selection in body of data set
            else:
                sample = np.hstack([data[n - half_width:n], data[n + 1:n + half_width + 1]])

            iqr_array.append(iqr(sample))

        return np.array(iqr_array)

    def filter_trdi(self):
        """Filter used by TRDI to filter out multiple reflections that get digitized as depth.
        """

        # Assign raw depth data to local variable
        depth_raw = np.copy(self.depth_orig_m)

        # Determine number of beams
        n_beams = depth_raw.shape[0]

        # Reset filters to none
        self.filter_none()

        # Set filter type to TRDI
        self.filter_type = 'TRDI'

        for n in range(n_beams):
            depth_ratio = depth_raw[n, :] / depth_raw
            exceeded = depth_ratio > 1.75
            exceeded_ens = np.nansum(exceeded, 0)
            self.valid_beams[n, exceeded_ens > 0] = False
