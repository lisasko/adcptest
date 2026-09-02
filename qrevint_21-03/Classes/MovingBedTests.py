import copy
import numpy as np
from Classes.TransectData import adjusted_ensemble_duration
from Classes.TransectData import TransectData
from Classes.QComp import QComp
from Classes.MatSonTek import MatSonTek
from MiscLibs.common_functions import cart2pol, sind, pol2cart, rad2azdeg


class MovingBedTests(object):
    """Stores and processes moving-bed tests.

    Attributes
    ----------
    type: str
        Loop or Stationary
    transect: TransectData
        Object of TransectData
    duration_sec: float
        Duration of test, in secs
    percent_invalid_bt: float
        Percent of invalid bottom track
    compass_diff_deg: float
        Difference in heading for out and back of loop
    flow_dir: float
        Mean flow direction from loop test
    mb_dir: float
        Moving bed or closure error direction
    dist_us_m: float
        Distance moved upstream, in m
    flow_spd_mps: float
        Magnitude of water velocity, in mps
    mb_spd_mps: float
        Magnitude of moving=bed velocity, in mps
    percent_mb: float
        Potential error due to moving bed, in percent
    moving_bed: str
        Moving-bed determined ("Yes" or "No")
    user_valid: bool
        Boolean to allow user to determine if test should be considered a valid test (True or False)
    test_quality: str
        Quality of test, 'Valid' 'Warnings' 'Errors'
    use_2_correct: bool
        Use this test to correct discharge (True or False)
    selected: bool
        Selected as valid moving-bed test to use for correction or determine moving-bed condition
    messages: list
        List of strings for warning and error messages based on data processing
    near_bed_speed_mps: float
        Mean near-bed water speed for test, in mps
    stationary_us_track: np.array(float)
        Upstream component of the bottom track referenced ship track
    stationary_cs_track: np.array(float)
        Cross=stream component of the bottom track referenced ship track
    stationary_mb_vel: np.array(float)
        Moving-bed velocity by ensemble, m/s
    ref: str
        Identifies reference used to compute moving bed
    bt_percent_mb: float
        Percent moving-bed using only BT
    bt_dist_us_m: float
        Distance upstream using only BT
    bt_mb_dir: float
        Moving-bed direction using only BT
    bt_mb_spd_mps: float
        Moving-bed speed using only BT
    bt_flow_spd_mps: float
        Corrected flow speed using only BT
    gps_percent_mb: float
        Percent moving-bed using BT and GPS
    gps_dist_us_m: float
        Distance upstream using BT and GPS
    gps_mb_dir: float
        Moving-bed direction using BT and GPS
    gps_mb_spd_mps: float
        Moving-bed speed using BT and GPS
    gps_flow_spd_mps: float
        Corrected flow speed using BT and GPS
    """
    
    def __init__(self):
        """Initialize class and instance variables."""

        self.type = None  # Loop or Stationary
        self.transect = None  # Object of TransectData
        self.duration_sec = np.nan  # Duration of test in secs
        self.percent_invalid_bt = np.nan  # Percent of invalid bottom track
        self.compass_diff_deg = np.nan  # Difference in heading for out and back of loop
        self.flow_dir = np.nan  # Mean flow direction from loop test
        self.mb_dir = np.nan  # Moving bed or closure error direction
        self.dist_us_m = np.nan  # Distance moved upstream in m
        self.flow_spd_mps = np.nan  # Magnitude of water velocity in mps
        self.mb_spd_mps = np.nan  # Magnitude of moving=bed velocity in mps
        self.percent_mb = np.nan  # Potential error due to moving bed in percent
        self.moving_bed = np.nan  # Moving-bed determined 'Yes' 'No'
        self.user_valid = True  # Logical to allow user to determine if test should be considered a valid test
        self.test_quality = None  # Quality of test 'Valid' 'Warnings' 'Errors'
        self.use_2_correct = None  # Use this test to correct discharge
        self.selected = None  # Selected valid moving-bed test to use for correction or determine moving-bed condition
        self.messages = None  # Cell array of warning and error messages based on data processing
        self.near_bed_speed_mps = np.nan  # Mean near-bed water speed for test in mps
        self.stationary_us_track = np.array([])  # Upstream component of the bottom track referenced ship track
        self.stationary_cs_track = np.array([])  # Cross=stream component of the bottom track referenced ship track
        self.stationary_mb_vel = np.array([])  # Moving-bed velocity by ensemble
        self.ref = 'BT'
        self.bt_percent_mb = np.nan
        self.bt_dist_us_m = np.nan
        self.bt_mb_dir = np.nan
        self.bt_mb_spd_mps = np.nan
        self.bt_flow_spd_mps = np.nan
        self.gps_percent_mb = np.nan
        self.gps_dist_us_m = np.nan
        self.gps_mb_dir = np.nan
        self.gps_mb_spd_mps = np.nan
        self.gps_flow_spd_mps = np.nan
        
    def populate_data(self, source, file=None, test_type=None):
        """Process and store moving-bed test data.

        Parameters
        ----------
        source: str
            Manufacturer of ADCP, SonTek or TRDI
        file: TransectData or str
            Object of TransectData for TRDI and str of filename for SonTek
        test_type: str
            Type of moving-bed test (Loop or Stationary)
        """

        if source == 'TRDI':
            self.mb_trdi(file, test_type)
        else:
            self.mb_sontek(file, test_type)

        self.process_mb_test(source)

    def process_mb_test(self, source):
        
        # Convert to earth coordinates and set the navigation reference to BT
        # for both boat and water data
        self.transect.boat_vel.bt_vel.apply_interpolation(transect=self.transect, interpolation_method='Linear')
        self.transect.change_coord_sys(new_coord_sys='Earth')
        self.transect.change_nav_reference(update=True, new_nav_ref='BT')
            
        # Adjust data for default manufacturer specific handling of invalid data
        delta_t = adjusted_ensemble_duration(self.transect, 'mbt')
        
        if self.type == 'Loop':
            if source == 'TRDI':
                self.loop_test(delta_t)
            else:
                self.loop_test()
        elif self.type == 'Stationary':
            self.stationary_test()
        else:
            raise ValueError('Invalid moving-bed test identifier specified.')

    @staticmethod
    def qrev_mat_in(meas_struct):
        """Processes the Matlab data structure to obtain a list of TransectData objects containing transect
           data from the Matlab data structure.

       Parameters
       ----------
       meas_struct: mat_struct
           Matlab data structure obtained from sio.loadmat

       Returns
       -------
       mb_tests: list
           List of MovingBedTests objects
       """

        mb_tests = []
        if hasattr(meas_struct, 'mbTests'):
            try:
                # If there are multiple test the Matlab structure will be an array
                if type(meas_struct.mbTests) == np.ndarray:
                    for test in meas_struct.mbTests:
                        temp = MovingBedTests()
                        temp.populate_from_qrev_mat(test)
                        mb_tests.append(temp)
                # If only one test, that test is not stored in an array
                else:
                    temp = MovingBedTests()
                    temp.populate_from_qrev_mat(meas_struct.mbTests)
                    mb_tests.append(temp)
            except (TypeError, AttributeError):
                pass
        return mb_tests

    def populate_from_qrev_mat(self, mat_data):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        self.type = mat_data.type
        self.transect = TransectData()
        self.transect.populate_from_qrev_mat(mat_data.transect)

        # If QRev.mat may return and empty array instead of a float
        self.duration_sec = self.return_float(mat_data.duration_sec)
        self.percent_invalid_bt = self.return_float(mat_data.percentInvalidBT)
        self.compass_diff_deg = self.return_float(mat_data.compassDiff_deg)
        self.flow_dir = self.return_float(mat_data.flowDir_deg)
        self.mb_dir = self.return_float(mat_data.mbDir_deg)
        self.dist_us_m = self.return_float(mat_data.distUS_m)
        self.flow_spd_mps = self.return_float(mat_data.flowSpd_mps)
        self.mb_spd_mps = self.return_float(mat_data.mbSpd_mps)
        self.percent_mb = self.return_float(mat_data.percentMB)
        self.near_bed_speed_mps = self.return_float(mat_data.nearBedSpeed_mps)

        self.moving_bed = mat_data.movingBed
        self.user_valid = bool(mat_data.userValid)
        self.test_quality = mat_data.testQuality
        self.use_2_correct = bool(mat_data.use2Correct)
        self.selected = bool(mat_data.selected)

        # Handle situation for one or more messages
        if type(mat_data.messages) == np.ndarray:
            self.messages = mat_data.messages.tolist()
        else:
            self.messages = [mat_data.messages]

        self.stationary_us_track = mat_data.stationaryUSTrack
        self.stationary_cs_track = mat_data.stationaryCSTrack
        self.stationary_mb_vel = mat_data.stationaryMBVel

        # Feature that can use GPS for moving-bed tests
        if hasattr(mat_data, 'bt_percent_mb'):
            self.bt_percent_mb = self.return_float(mat_data.bt_percent_mb)
            self.bt_dist_us_m = self.return_float(mat_data.bt_dist_us_m)
            self.bt_mb_dir = self.return_float(mat_data.bt_mb_dir)
            self.bt_mb_spd_mps = self.return_float(mat_data.bt_mb_spd_mps)
            self.bt_flow_spd_mps = self.return_float(mat_data.bt_flow_spd_mps)
            self.gps_percent_mb = self.return_float(mat_data.gps_percent_mb)
            self.gps_dist_us_m = self.return_float(mat_data.gps_dist_us_m)
            self.gps_mb_dir = self.return_float(mat_data.gps_mb_dir)
            self.gps_mb_spd_mps = self.return_float(mat_data.gps_mb_spd_mps)
        else:
            self.bt_percent_mb = self.percent_mb
            self.bt_dist_us_m = self.dist_us_m
            self.bt_mb_dir = self.mb_dir
            self.bt_mb_spd_mps = self.mb_spd_mps
            self.bt_flow_spd_mps = self.flow_spd_mps
            self.compute_mb_gps()

    @staticmethod
    def return_float(data):
        """Handles situation where the Matlab file is an empty array

        Parameters
        ----------
        data: float or np.ndarray
            Any variable
        """
        if type(data) is float:
            return data
        elif type(data) is np.ndarray:
            if data.size == 0:
                return np.nan
            else:
                return MovingBedTests.make_list(data)
        else:
            return np.nan

    @staticmethod
    def make_list(array_in):
        """Method to make list from several special cases that can occur in the Matlab data.

        Parameters
        ----------
        array_in: np.ndarray
            Input that needs to be convert to a list
        """

        # This traps messages with the associated codes
        if array_in.size > 3:
            list_out = array_in.tolist()
        else:
            # Create a list of lists
            temp = array_in.tolist()
            if len(temp) > 0:
                internal_list = []
                for item in temp:
                    internal_list.append(item)
                list_out = [internal_list]
            else:
                list_out = np.nan
        return list_out

    def mb_trdi(self, transect, test_type):
        """Function to create object properties for TRDI moving-bed tests

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        test_type: str
            Type of moving-bed test."""
        
        self.transect = transect
        self.user_valid = True
        self.type = test_type

    def mb_sontek(self, file_name, test_type):
        """Function to create object properties for SonTek moving-bed tests

        Parameters
        ----------
        file_name: str
            Name of moving-bed test data file
        test_type: str
            Type of moving-bed test."""
        self.type = test_type

        # Read Matlab file for moving-bed test
        rsdata = MatSonTek(file_name)

        # Create transect objects for each discharge transect
        self.transect = TransectData()
        self.transect.sontek(rsdata, file_name)
        
    def loop_test(self, ens_duration=None, ref='BT'):
        """Process loop moving bed test.

        Parameters
        ----------
        ens_duration: np.array(float)
            Duration of each ensemble, in sec
        ref: str
            Reference used to compare distance moved
        """

        # Assign data from transect to local variables
        self.transect.boat_interpolations(update=False, target='BT', method='Linear')
        self.transect.boat_interpolations(update=False, target='GPS', method='Linear')
        trans_data = copy.deepcopy(self.transect)
        in_transect_idx = trans_data.in_transect_idx
        n_ensembles = len(in_transect_idx)
        bt_valid = trans_data.boat_vel.bt_vel.valid_data[0, in_transect_idx]

        # Set variables to defaults
        self.messages = []
        vel_criteria = 0.012

        # Check that there is some valid BT data
        if np.nansum(bt_valid) > 1:
            wt_u = trans_data.w_vel.u_processed_mps[:, in_transect_idx]
            wt_v = trans_data.w_vel.v_processed_mps[:, in_transect_idx]
            if ens_duration is None:
                ens_duration = trans_data.date_time.ens_duration_sec[in_transect_idx]

            bt_u = trans_data.boat_vel.bt_vel.u_processed_mps[in_transect_idx]
            bt_v = trans_data.boat_vel.bt_vel.v_processed_mps[in_transect_idx]
            bin_size = trans_data.depths.bt_depths.depth_cell_size_m[:, in_transect_idx]

            # Compute closure distance and direction
            bt_x = np.nancumsum(bt_u * ens_duration)
            bt_y = np.nancumsum(bt_v * ens_duration)
            direct, self.bt_dist_us_m = cart2pol(bt_x[-1], bt_y[-1])
            self.bt_mb_dir = rad2azdeg(direct)

            # Compute duration of test
            self.duration_sec = np.nansum(ens_duration)

            # Compute the moving-bed velocity
            self.bt_mb_spd_mps = self.bt_dist_us_m / self.duration_sec

            # Compute discharge weighted mean velocity components for the
            # purposed of computing the mean flow direction
            xprod = QComp.cross_product(transect=trans_data)
            q = QComp.discharge_middle_cells(xprod, trans_data, ens_duration)
            wght = np.abs(q)
            se = np.nansum(np.nansum(wt_u * wght)) / np.nansum(np.nansum(wght))
            sn = np.nansum(np.nansum(wt_v * wght)) / np.nansum(np.nansum(wght))
            direct, flow_speed_q = cart2pol(se, sn)

            # Compute flow speed and direction
            self.flow_dir = rad2azdeg(direct)
            
            # Compute the area weighted mean velocity components for the
            # purposed of computing the mean flow speed. Area weighting is used for flow speed instead of
            # discharge so that the flow speed is not included in the weighting used to compute the mean flow speed.
            wght_area = np.multiply(np.multiply(np.sqrt(bt_u ** 2 + bt_v ** 2), bin_size), ens_duration)
            idx = np.where(np.isnan(wt_u) == False)
            se = np.nansum(np.nansum(wt_u[idx] * wght_area[idx])) / np.nansum(np.nansum(wght_area[idx]))
            sn = np.nansum(np.nansum(wt_v[idx] * wght_area[idx])) / np.nansum(np.nansum(wght_area[idx]))
            dir_a, self.bt_flow_spd_mps = cart2pol(se, sn)
            self.bt_flow_spd_mps = self.bt_flow_spd_mps + self.bt_mb_spd_mps

            # Compute potential error in BT referenced discharge
            self.bt_percent_mb = (self.bt_mb_spd_mps / self.bt_flow_spd_mps) * 100

            # Compute test with GPS
            self.compute_mb_gps()

            # Store selected test characteristics
            if ref == 'BT':
                self.mb_spd_mps = self.bt_mb_spd_mps
                self.dist_us_m = self.bt_dist_us_m
                self.percent_mb = self.bt_percent_mb
                self.mb_dir = self.bt_mb_dir
                self.flow_spd_mps = self.bt_flow_spd_mps
            elif not np.isnan(self.gps_percent_mb):
                self.mb_spd_mps = self.gps_mb_spd_mps
                self.dist_us_m = self.gps_dist_us_m
                self.percent_mb = self.gps_percent_mb
                self.mb_dir = self.gps_mb_dir
                self.flow_spd_mps = self.bt_flow_spd_mps

            # Assess invalid bottom track
            # Compute percent invalid bottom track
            self.percent_invalid_bt = (np.nansum(bt_valid == False) / len(bt_valid)) * 100

            # Determine if more than 9 consecutive seconds of invalid BT occurred
            consect_bt_time = np.zeros(n_ensembles)
            for n in range(1, n_ensembles):
                if bt_valid[n]:
                    consect_bt_time[n] = 0
                else:
                    consect_bt_time[n] = consect_bt_time[n - 1] + ens_duration[n]

            max_consect_bt_time = np.nanmax(consect_bt_time)

            # Evaluate compass calibration based on flow direction

            # Find apex of loop adapted from
            # http://www.mathworks.de/matlabcentral/newsreader/view_thread/164048
            loop_out = np.array([bt_x[0], bt_y[0], 0])
            loop_return = np.array([bt_x[-1], bt_y[-1], 0])

            distance = np.zeros(n_ensembles)
            for n in range(n_ensembles):
                p = np.array([bt_x[n], bt_y[n], 0])
                distance[n] = np.linalg.norm(np.cross(loop_return - loop_out, p - loop_out))  \
                    / np.linalg.norm(loop_return - loop_out)

            dmg_idx = np.where(distance == np.nanmax(distance))[0][0]

            # Compute flow direction on outgoing part of loop
            u_out = wt_u[:, :dmg_idx + 1]
            v_out = wt_v[:, :dmg_idx + 1]
            wght = np.abs(q[:, :dmg_idx+1])
            se = np.nansum(u_out * wght) / np.nansum(wght)
            sn = np.nansum(v_out * wght) / np.nansum(wght)
            direct, _ = cart2pol(se, sn)
            flow_dir1 = rad2azdeg(direct)

            # Compute unweighted flow direction in each cell
            direct, _ = cart2pol(u_out, v_out)
            flow_dir_cell = rad2azdeg(direct)

            # Compute difference from mean and correct to +/- 180
            v_dir_corr = flow_dir_cell - flow_dir1
            v_dir_idx = v_dir_corr > 180
            v_dir_corr[v_dir_idx] = 360-v_dir_corr[v_dir_idx]
            v_dir_idx = v_dir_corr < -180
            v_dir_corr[v_dir_idx] = 360 + v_dir_corr[v_dir_idx]

            # Number of invalid weights
            idx2 = np.where(np.isnan(wght) == False)
            nwght = len(idx2[0])

            # Compute 95% uncertainty using weighted standard deviation
            uncert1 = 2. * np.sqrt(np.nansum(np.nansum(wght * v_dir_corr**2))
                                   / (((nwght - 1) * np.nansum(np.nansum(wght))) / nwght)) / np.sqrt(nwght)

            # Compute flow direction on returning part of loop
            u_ret = wt_u[:, dmg_idx + 1:]
            v_ret = wt_v[:, dmg_idx + 1:]
            wght = np.abs(q[:, dmg_idx+1:])
            se = np.nansum(u_ret * wght) / np.nansum(wght)
            sn = np.nansum(v_ret * wght) / np.nansum(wght)
            direct, _ = cart2pol(se, sn)
            flow_dir2 = rad2azdeg(direct)

            # Compute unweighted flow direction in each cell
            direct, _ = cart2pol(u_ret, v_ret)
            flow_dir_cell = rad2azdeg(direct)

            # Compute difference from mean and correct to +/- 180
            v_dir_corr = flow_dir_cell - flow_dir2
            v_dir_idx = v_dir_corr > 180
            v_dir_corr[v_dir_idx] = 360 - v_dir_corr[v_dir_idx]
            v_dir_idx = v_dir_corr < -180
            v_dir_corr[v_dir_idx] = 360 + v_dir_corr[v_dir_idx]

            # Number of valid weights
            idx2 = np.where(np.isnan(wght) == False)
            nwght = len(idx2[0])

            # Compute 95% uncertainty using weighted standard deviation
            uncert2 = 2.*np.sqrt(np.nansum(np.nansum(wght * v_dir_corr**2))
                                 / (((nwght-1)*np.nansum(np.nansum(wght))) / nwght)) / np.sqrt(nwght)

            # Compute and report difference in flow direction
            diff_dir = np.abs(flow_dir1 - flow_dir2)
            if diff_dir > 180:
                diff_dir = diff_dir - 360
            self.compass_diff_deg = diff_dir
            uncert = uncert1 + uncert2

            # Compute potential compass error
            idx = np.where(np.isnan(bt_x) == False)
            if len(idx[0]) > 0:
                idx = idx[0][-1]
            width = np.sqrt((bt_x[dmg_idx] - bt_x[idx] / 2) ** 2 + (bt_y[dmg_idx] - bt_y[idx] / 2) ** 2)
            compass_error = (2 * width * sind(diff_dir / 2) * 100) / (self.duration_sec * self.flow_spd_mps)

            # Initialize message counter
            self.test_quality = 'Good'

            # Low water velocity
            if self.flow_spd_mps < 0.25:
                self.messages.append('WARNING: The water velocity is less than recommended minimum for '
                                     + 'this test and could cause the loop method to be inaccurate. '
                                     + 'CONSIDER USING A STATIONARY TEST TO CHECK MOVING-BED CONDITIONS')
                self.test_quality = 'Warnings'

            # Percent invalid bottom track
            if self.percent_invalid_bt > 20:
                self.messages.append('ERROR: Percent invalid bottom track exceeds 20 percent. '
                                     + 'THE LOOP IS NOT ACCURATE. TRY A STATIONARY MOVING-BED TEST.')
                self.test_quality = 'Errors'
            elif self.percent_invalid_bt > 5:
                self.messages.append('WARNING: Percent invalid bottom track exceeds 5 percent. '
                                     + 'Loop may not be accurate. PLEASE REVIEW DATA.')
                self.test_quality = 'Warnings'

            # More than 9 consecutive seconds of invalid BT
            if max_consect_bt_time > 9:
                self.messages.append('ERROR: Bottom track is invalid for more than 9 consecutive seconds.'
                                     + 'THE LOOP IS NOT ACCURATE. TRY A STATIONARY MOVING-BED TEST.')
                self.test_quality = 'Errors'

            if np.abs(compass_error) > 5 and np.abs(diff_dir) > 3 and np.abs(diff_dir) > uncert:
                self.messages.append('ERROR: Difference in flow direction between out and back sections of '
                                     + 'loop could result in a 5 percent or greater error in final discharge. '
                                     + 'REPEAT LOOP AFTER COMPASS CAL. OR USE A STATIONARY MOVING-BED TEST.')
                self.test_quality = 'Errors'

        else:
            self.messages.append('ERROR: Loop has no valid bottom track data. '
                                 + 'REPEAT OR USE A STATIONARY MOVING-BED TEST.')
            self.test_quality = 'Errors'

        # If loop is valid then evaluate moving-bed condition
        if self.test_quality != 'Errors':

            # Check minimum moving-bed velocity criteria
            if self.mb_spd_mps > vel_criteria:
                # Check that closure error is in upstream direction
                if 135 < np.abs(self.flow_dir - self.mb_dir) < 225:
                    # Check if moving-bed is greater than 1% of the mean flow speed
                    if self.percent_mb > 1:
                        self.messages.append('Loop Indicates a Moving Bed -- Use GPS as reference. If GPS is '
                                             + 'unavailable or invalid use the loop method to correct the '
                                             + 'final discharge.')
                        self.moving_bed = 'Yes'
                    else:
                        self.messages.append('Moving Bed Velocity < 1% of Mean Velocity -- No Correction Recommended')
                        self.moving_bed = 'No'
                else:
                    self.messages.append('ERROR: Loop closure error not in upstream direction. '
                                         + 'REPEAT LOOP or USE STATIONARY TEST')
                    self.test_quality = 'Errors'
                    self.moving_bed = 'Unknown'
            else:
                self.messages.append('Moving-bed velocity < Minimum moving-bed velocity criteria '
                                     + '-- No correction recommended')
                self.moving_bed = 'No'

            # Notify of differences in results of test between BT and GPS
            if not np.isnan(self.gps_percent_mb):
                if np.abs(self.bt_percent_mb - self.gps_percent_mb) > 2:
                    self.messages.append('WARNING - Bottom track and GPS results differ by more than 2%.')
                    self.test_quality = 'Warnings'

                if np.logical_xor(self.bt_percent_mb >= 1,  self.gps_percent_mb >= 1):
                    self.messages.append('WARNING - Bottom track and GPS results do not agree.')
                    self.test_quality = 'Warnings'

        else:
            self.messages.append('ERROR: Due to ERRORS noted above this loop is NOT VALID. '
                                 + 'Please consider suggestions.')
            self.moving_bed = 'Unknown'

    def stationary_test(self, ref='BT'):
        """Processed the stationary moving-bed tests.
        """

        # Assign data from transect to local variables
        trans_data = copy.deepcopy(self.transect)
        in_transect_idx = trans_data.in_transect_idx
        bt_valid = trans_data.boat_vel.bt_vel.valid_data[0, in_transect_idx]

        # Check to see that there is valid bottom track data
        self.messages = []
        if np.nansum(bt_valid) > 0:
            # Assign data to local variables
            wt_u = trans_data.w_vel.u_processed_mps[:, in_transect_idx]
            wt_v = trans_data.w_vel.v_processed_mps[:, in_transect_idx]
            ens_duration = trans_data.date_time.ens_duration_sec[in_transect_idx]
            bt_u = trans_data.boat_vel.bt_vel.u_processed_mps[in_transect_idx]
            bt_v = trans_data.boat_vel.bt_vel.v_processed_mps[in_transect_idx]

            # Use only data with valid bottom track
            valid_bt = trans_data.boat_vel.bt_vel.valid_data[0, in_transect_idx]
            wt_u[:, valid_bt == False] = np.nan
            wt_v[:, valid_bt == False] = np.nan
            bt_u[valid_bt == False] = np.nan
            bt_v[valid_bt == False] = np.nan

            u_water = np.nanmean(wt_u)
            v_water = np.nanmean(wt_v)
            self.flow_dir = np.arctan2(u_water, v_water) * 180 / np.pi
            if self.flow_dir < 0:
                self.flow_dir = self.flow_dir + 360

            bin_depth = trans_data.depths.bt_depths.depth_cell_depth_m[:, in_transect_idx]
            trans_select = getattr(trans_data.depths, trans_data.depths.selected)
            depth_ens = trans_select.depth_processed_m[in_transect_idx]

            nb_u, nb_v, unit_nbu, unit_nbv = self.near_bed_velocity(wt_u, wt_v, depth_ens, bin_depth)
            
            # Compute bottom track parallel to water velocity
            unit_nb_vel = np.vstack([unit_nbu, unit_nbv])
            bt_vel = np.vstack([bt_u, bt_v])
            bt_vel_up_strm = -1 * np.sum(bt_vel * unit_nb_vel, 0)
            bt_up_strm_dist = bt_vel_up_strm * ens_duration
            bt_up_strm_dist_cum = np.nancumsum(bt_up_strm_dist)
            self.bt_dist_us_m = bt_up_strm_dist_cum[-1]

            # Compute bottom track perpendicular to water velocity
            nb_vel_ang, _ = cart2pol(unit_nbu, unit_nbv)
            nb_vel_unit_cs1, nb_vel_unit_cs2 = pol2cart(nb_vel_ang + np.pi / 2, 1)
            nb_vel_unit_cs = np.vstack([nb_vel_unit_cs1, nb_vel_unit_cs2])
            bt_vel_cs = np.sum(bt_vel * nb_vel_unit_cs, 0)
            bt_cs_strm_dist = bt_vel_cs * ens_duration
            bt_cs_strm_dist_cum = np.nancumsum(bt_cs_strm_dist)
            
            # Compute cumulative mean moving bed velocity
            valid_bt_vel_up_strm = np.isnan(bt_vel_up_strm) == False

            mb_vel = np.nancumsum(bt_vel_up_strm) / np.nancumsum(valid_bt_vel_up_strm)

            # Compute the average ensemble velocities corrected for moving bed
            if mb_vel[-1] > 0:
                u_corrected = np.add(wt_u, (unit_nb_vel[0, :]) * bt_vel_up_strm)
                v_corrected = np.add(wt_v, (unit_nb_vel[1, :]) * bt_vel_up_strm)
            else:
                u_corrected = wt_u
                v_corrected = wt_v
                
            # Compute the mean of the ensemble magnitudes

            # Mean is computed using magnitudes because if a Streampro with no compass is the data source the change
            # in direction could be either real change in water direction or an uncompensated turn of the floating
            # platform. This approach is the best compromise when there is no compass or the compass is unreliable,
            # which is often why the stationary method is used. A weighted average is used to account for the possible
            # change in cell size within and ensemble for the RiverRay and RiverPro.

            mag = np.sqrt(u_corrected**2 + v_corrected**2)
            depth_cell_size = trans_data.depths.bt_depths.depth_cell_size_m[:, in_transect_idx]
            depth_cell_size[np.isnan(mag)] = np.nan
            mag_w = mag * depth_cell_size
            self.bt_flow_spd_mps = np.nansum(mag_w) / np.nansum(depth_cell_size)
            self.bt_mb_spd_mps = mb_vel[-1]
            self.bt_percent_mb = (self.bt_mb_spd_mps / self.bt_flow_spd_mps) * 100
            if self.bt_percent_mb < 0:
                self.bt_percent_mb = 0

            # Compute percent invalid bottom track
            self.percent_invalid_bt = (np.nansum(bt_valid == False) / len(bt_valid)) * 100
            self.duration_sec = np.nansum(ens_duration)

            # Compute test using GPS
            self.compute_mb_gps()

            # Store selected test characteristics
            if ref == 'BT':
                self.mb_spd_mps = self.bt_mb_spd_mps
                self.dist_us_m = self.bt_dist_us_m
                self.percent_mb = self.bt_percent_mb
                self.mb_dir = self.bt_mb_dir
                self.flow_spd_mps = self.bt_flow_spd_mps
            elif not np.isnan(self.gps_percent_mb):
                self.mb_spd_mps = self.gps_mb_spd_mps
                self.dist_us_m = self.gps_dist_us_m
                self.percent_mb = self.gps_percent_mb
                self.mb_dir = self.gps_mb_dir
                self.flow_spd_mps = self.bt_flow_spd_mps

            self.near_bed_speed_mps = np.sqrt(np.nanmean(nb_u)**2 + np.nanmean(nb_v)**2)
            self.stationary_us_track = bt_up_strm_dist_cum
            self.stationary_cs_track = bt_cs_strm_dist_cum
            self.stationary_mb_vel = mb_vel

            # Quality check
            self.test_quality = 'Good'
            # Check duration
            if self.duration_sec < 300:
                self.messages.append('WARNING - Duration of stationary test is less than 5 minutes')
                self.test_quality = 'Warnings'
                
            # Check validity of mean moving-bed velocity
            if self.duration_sec > 60:
                mb_vel_std = np.nanstd(mb_vel[-30:], ddof=1)
                cov = mb_vel_std / mb_vel[-1]
                if cov > 0.25 and mb_vel_std > 0.03:
                    self.messages.append('WARNING - Moving-bed velocity may not be consistent. '
                                         + 'Average maybe inaccurate.')
                    self.test_quality = 'Warnings'
                    
            # Check percentage of invalid BT data
            if np.nansum(ens_duration[valid_bt_vel_up_strm]) <= 120:
                
                self.messages.append('ERROR - Total duration of valid BT data is insufficient for a valid test.')
                self.test_quality = 'Errors'
                self.moving_bed = 'Unknown'
            elif self.percent_invalid_bt > 10:
                self.messages.append('WARNING - Number of ensembles with invalid bottom track exceeds 10%')
                self.test_quality = 'Warnings'
                
            # Determine if the test indicates a moving bed
            if self.test_quality != 'Errors':
                if self.percent_mb >= 1:
                    self.moving_bed = 'Yes'
                else:
                    self.moving_bed = 'No'

            # Notify of differences in results of test between BT and GPS
            if not np.isnan(self.gps_percent_mb):
                if np.abs(self.bt_percent_mb - self.gps_percent_mb) > 2:
                    self.messages.append('WARNING - Bottom track and GPS results differ by more than 2%.')
                    self.test_quality = 'Warnings'

                if np.logical_xor(self.bt_percent_mb >= 1,  self.gps_percent_mb >= 1):
                    self.messages.append('WARNING - Bottom track and GPS results do not agree.')
                    self.test_quality = 'Warnings'

        else:
            self.messages.append('ERROR - Stationary moving-bed test has no valid bottom track data.')
            self.test_quality = 'Errors'
            self.moving_bed = 'Unknown'
            self.duration_sec = np.nansum(trans_data.date_time.ens_duration_sec[in_transect_idx])
            self.percent_invalid_bt = 100

    def compute_mb_gps(self):
        """Computes moving-bed data using GPS.
        """
        if np.isnan(self.flow_dir):
            u_water = np.nanmean(self.transect.w_vel.u_processed_mps[:, self.transect.in_transect_idx])
            v_water = np.nanmean(self.transect.w_vel.v_processed_mps[:, self.transect.in_transect_idx])
            self.flow_dir = np.arctan2(u_water, v_water) * 180 / np.pi
            if self.flow_dir < 0:
                self.flow_dir = self.flow_dir + 360

        gps_bt = None
        # Use GGA data if available and VTG is GGA is not available
        if self.transect.boat_vel.gga_vel is not None:
            gps_bt = TransectData.compute_gps_bt(self.transect, gps_ref='gga_vel')
        elif self.transect.boat_vel.vtg_vel is not None:
            gps_bt = TransectData.compute_gps_bt(self.transect, gps_ref='vtg_vel')
        if gps_bt is not None and len(gps_bt) > 0:
            self.gps_dist_us_m = gps_bt['mag']
            self.gps_mb_dir = gps_bt['dir']
            self.gps_mb_spd_mps = self.gps_dist_us_m / self.duration_sec
            self.gps_flow_spd_mps = self.bt_flow_spd_mps - self.bt_mb_spd_mps + self.gps_mb_spd_mps
            self.gps_percent_mb = (self.gps_mb_spd_mps / self.gps_flow_spd_mps) * 100

    def magvar_change(self, magvar, old_magvar):
        """Adjust moving-bed test for change in magvar.

        Parameters
        ----------
        magvar: float
            New magvar
        old_magvar: float
            Existing magvar
        """

        if self.transect.sensors.heading_deg.selected == 'internal':
            magvar_change = magvar - old_magvar
            self.bt_mb_dir = self.bt_mb_dir + magvar_change
            self.flow_dir = self.flow_dir + magvar_change

            # Recompute moving-bed tests with GPS and set results using existing reference
            self.compute_mb_gps()
            self.change_ref(self.ref)

    def h_offset_change(self, h_offset, old_h_offset):
        """Adjust moving-bed test for change in h_offset for external compass.

        Parameters
        ----------
        h_offset: float
            New h_offset
        old_h_offset: float
            Existing h_offset
        """

        if self.transect.sensors.heading_deg.selected == 'external':
            h_offset_change = h_offset - old_h_offset
            self.bt_mb_dir = self.bt_mb_dir + h_offset_change
            self.flow_dir = self.flow_dir + h_offset_change

            # Recompute moving-bed tests with GPS and set results using existing reference
            self.compute_mb_gps()
            self.change_ref(self.ref)

    def change_ref(self, ref):
        """Change moving-bed test fixed reference.

        Parameters
        ----------
        ref: str
            Defines specified reference (BT or GPS)
        """

        if ref == 'BT':
            self.mb_spd_mps = self.bt_mb_spd_mps
            self.dist_us_m = self.bt_dist_us_m
            self.percent_mb = self.bt_percent_mb
            self.mb_dir = self.bt_mb_dir
            self.flow_spd_mps = self.bt_flow_spd_mps
            self.ref = 'BT'
            check_mb = True
            if self.test_quality != 'Errors':
                if self.type == 'Loop':
                    if self.mb_spd_mps <= 0.012:
                        check_mb = False
                        self.moving_bed = 'No'
                    else:
                        if 135 < np.abs(self.flow_dir - self.mb_dir) < 225:
                            check_mb = True
                        else:
                            check_mb = False
                            self.moving_bed = 'Unknown'
                if check_mb:
                    if self.percent_mb > 1:
                        self.moving_bed = 'Yes'
                    else:
                        self.moving_bed = 'No'
            else:
                self.moving_bed = 'Unknown'
        elif ref == 'GPS':
            self.mb_spd_mps = self.gps_mb_spd_mps
            self.dist_us_m = self.gps_dist_us_m
            self.percent_mb = self.gps_percent_mb
            self.mb_dir = self.gps_mb_dir
            self.flow_spd_mps = self.gps_flow_spd_mps
            self.ref = 'GPS'
            check_mb = True
            if self.test_quality != 'Errors':
                if self.type == 'Loop':
                    if self.mb_spd_mps <= 0.012:
                        check_mb = False
                        self.moving_bed = 'No'
                    else:
                        if 135 < np.abs(self.flow_dir - self.mb_dir) < 225:
                            check_mb = True
                        else:
                            check_mb = False
                            self.messages.append('ERROR: GPS Loop closure error not in upstream direction. '
                                                 + 'REPEAT LOOP or USE STATIONARY TEST')
                            self.moving_bed = 'Unknown'
                if check_mb:
                    if self.percent_mb > 1:
                        self.moving_bed = 'Yes'
                    else:
                        self.moving_bed = 'No'
            else:
                self.moving_bed = 'Unknown'

    @staticmethod
    def near_bed_velocity(u, v, depth, bin_depth):
        """Compute near bed velocities.

        Parameters
        ----------
        u: np.array(float)
            u water velocity component
        v: np.array(float)
            v water velocity component
        depth: np.array(float)
            Water depth for each ensemble
        bin_depth: np.array(float)
            Depth to centerline of each bin

        Returns
        -------
        nb_u: np.array(float)
            u near-bed velocity component
        nb_v: np.array(float)
            v near-bed velocity component
        unit_nbu: np.array(float)
            u component of the near-bed unit vector
        unit_nbv: np.array(float)
            v component of the near-bed unit vector
        """

        # Compute z near bed as 10% of depth
        z_near_bed = depth * 0.1

        # Initialize variables
        n_ensembles = u.shape[1]
        nb_u = np.tile(np.nan, n_ensembles)
        nb_v = np.tile(np.nan, n_ensembles)
        unit_nbu = np.tile(np.nan, n_ensembles)
        unit_nbv = np.tile(np.nan, n_ensembles)
        z_depth = np.tile(np.nan, n_ensembles)
        u_mean = np.tile(np.nan, n_ensembles)
        v_mean = np.tile(np.nan, n_ensembles)
        speed_near_bed = np.tile(np.nan, n_ensembles)

        # Compute near bed velocity for each ensemble
        for n in range(n_ensembles):
            idx = np.where(np.isnan(u[:, n]) == False)
            if len(idx[-1]) > 0:
                if len(idx[-1]) > 0:
                    idx = idx[-1][-2::]
                else:
                    idx = idx[-1][-1]
                # Compute near-bed velocity
                z_depth[n] = depth[n] - np.nanmean(bin_depth[idx, n], 0)
                u_mean[n] = np.nanmean(u[idx, n], 0)
                v_mean[n] = np.nanmean(v[idx, n], 0)
                nb_u[n] = (u_mean[n] / z_depth[n] ** (1. / 6.)) * (z_near_bed[n] ** (1. / 6.))
                nb_v[n] = (v_mean[n] / z_depth[n] ** (1. / 6.)) * (z_near_bed[n] ** (1. / 6.))
                speed_near_bed[n] = np.sqrt(nb_u[n] ** 2 + nb_v[n] ** 2)
                unit_nbu[n] = nb_u[n] / speed_near_bed[n]
                unit_nbv[n] = nb_v[n] / speed_near_bed[n]

        return nb_u, nb_v, unit_nbu, unit_nbv

    @staticmethod
    def auto_use_2_correct(moving_bed_tests, boat_ref=None):
        """Apply logic to determine which moving-bed tests should be used
        for correcting bottom track referenced discharges with moving-bed conditions.

        Parameters
        ----------
        moving_bed_tests: list
            List of MovingBedTests objects.
        boat_ref: str
            Boat velocity reference.

        Returns
        -------
        moving_bed_tests: list
            List of MovingBedTests objects.
        """

        if len(moving_bed_tests) != 0:
            # Initialize variables
            lidx_user = []
            lidx_no_errors = []
            test_type = []
            lidx_stationary = []
            lidx_loop = []
            flow_speed = []
            for test in moving_bed_tests:
                test.use_2_correct = False
                test.selected = False
                # Valid test according to user
                lidx_user.append(test.user_valid == True)
                # Valid test according to quality assessment
                lidx_no_errors.append(test.test_quality != 'Errors')
                # Identify type of test
                test_type.append(test.type)
                lidx_stationary.append(test.type == 'Stationary')
                lidx_loop.append(test.type == 'Loop')
                flow_speed.append(test.flow_spd_mps)

            # Combine
            lidx_valid_loop = np.all(np.vstack((lidx_user, lidx_no_errors, lidx_loop)), 0)
            lidx_valid_stationary = np.all(np.vstack((lidx_user, lidx_no_errors, lidx_stationary)), 0)

            # Check flow speed
            lidx_flow_speed = np.array(flow_speed) > 0.25

            # Determine if there are valid loop tests
            # This is the code in matlab but I don't think it is correct. I the valid loop should also have a valid
            # flow speed, if not then a stationary test, if available could be used.
            lidx_loops_2_select = np.all(np.vstack((lidx_flow_speed, lidx_valid_loop)), 0)
            if np.any(lidx_loops_2_select):
                # Select last loop
                idx_select = np.where(lidx_loops_2_select)[0][-1]
                test_select = moving_bed_tests[idx_select]
                test_select.selected = True

                if test_select.moving_bed == 'Yes':
                    test_select.use_2_correct = True

            # If there are no valid loop look for valid stationary tests
            elif np.any(lidx_valid_stationary):
                moving_bed = []
                for n, lidx in enumerate(lidx_valid_stationary):
                    if lidx:
                        moving_bed_tests[n].selected = True
                        # Determine if any stationary test resulted in a moving bed
                        if moving_bed_tests[n].moving_bed == 'Yes':
                            moving_bed.append(True)
                        else:
                            moving_bed.append(False)
                # If any stationary test shows a moving-bed use all valid stationary test to correct BT discharge
                if any(moving_bed) > 0:
                    for n, test in enumerate(moving_bed_tests):
                        if lidx_valid_stationary[n]:
                            test.use_2_correct = True

            # If the flow speed is too low but there are not valid stationary tests use the last loop test.
            elif np.any(lidx_valid_loop):
                # Select last loop
                idx_select = np.where(lidx_valid_loop)[0][-1]
                moving_bed_tests[idx_select].selected = True
                if moving_bed_tests[idx_select].moving_bed == 'Yes':
                    moving_bed_tests[idx_select].use_2_correct = True

            # If the navigation reference for discharge computations is set
            # GPS then none of test should be used for correction. The
            # selected test should be used to determine if there is a valid
            # moving-bed and a moving-bed condition.
            if boat_ref is None:
                ref = 'BT'
            else:
                ref = boat_ref

            if ref != 'BT':
                for test in moving_bed_tests:
                    test.use_2_correct = False
        return moving_bed_tests
