import copy
import numpy as np
from Classes.Uncertainty import Uncertainty
from Classes.QComp import QComp
from Classes.MovingBedTests import MovingBedTests
from Classes.TransectData import TransectData


class QAData(object):
    """Evaluates and stores quality assurance characteristics and messages.

    Attributes
    ----------
    q_run_threshold_caution: int
        Caution threshold for interpolated discharge for a run of invalid ensembles, in percent.
    q_run_threshold_warning: int
        Warning threshold for interpolated discharge for a run of invalid ensembles, in percent.
    q_total_threshold_caution: int
        Caution threshold for total interpolated discharge for invalid ensembles, in percent.
    q_total_threshold_warning: int
        Warning threshold for total interpolated discharge for invalid ensembles, in percent.
    transects: dict
        Dictionary of quality assurance checks for transects
    system_tst: dict
        Dictionary of quality assurance checks on the system test(s)
    compass: dict
        Dictionary of quality assurance checks on compass calibration and evaluations
    temperature: dict
        Dictionary of quality assurance checks on temperature comparions and variation
    movingbed: dict
        Dictionary of quality assurance checks on moving-bed tests
    user: dict
        Dictionary of quality assurance checks on user input data
    boat: dict
        Dictionary of quality assurance checks on boat velocities
    bt_vel: dict
        Dictionary of quality assurance checks on bottom track velocities
    gga_vel: dict
        Dictionary of quality assurance checks on gga boat velocities
    vtg_vel: dict
        Dictionary of quality assurance checks on vtg boat velocities
    w_vel: dict
        Dictionary of quality assurance checks on water track velocities
    extrapolation: dict
        Dictionary of quality assurance checks on extrapolations
    edges: dict
        Dictionary of quality assurance checks on edges
    """

    def __init__(self, meas, mat_struct=None, compute=True):
        """Checks the measurement for all quality assurance issues.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        # Set default thresholds
        self.q_run_threshold_caution = 3
        self.q_run_threshold_warning = 5
        self.q_total_threshold_caution = 10
        self.q_total_threshold_warning = 25

        # Initialize instance variables
        self.transects = dict()
        self.system_tst = dict()
        self.compass = dict()
        self.temperature = dict()
        self.movingbed = dict()
        self.user = dict()
        self.depths = dict()
        self.boat = dict()
        self.bt_vel = dict()
        self.gga_vel = dict()
        self.vtg_vel = dict()
        self.w_vel = dict()
        self.extrapolation = dict()
        self.edges = dict()
        self.settings_dict = dict()
        self.settings_dict['tab_compass'] = 'Default'
        self.settings_dict['tab_tempsal'] = 'Default'
        self.settings_dict['tab_mbt'] = 'Default'
        self.settings_dict['tab_bt'] = 'Default'
        self.settings_dict['tab_gps'] = 'Default'
        self.settings_dict['tab_depth'] = 'Default'
        self.settings_dict['tab_wt'] = 'Default'
        self.settings_dict['tab_extrap'] = 'Default'
        self.settings_dict['tab_edges'] = 'Default'

        if compute:
            # Apply QA checks
            self.transects_qa(meas)
            self.system_tst_qa(meas)
            self.compass_qa(meas)
            self.temperature_qa(meas)
            self.moving_bed_qa(meas)
            self.user_qa(meas)
            self.depths_qa(meas)
            self.boat_qa(meas)
            self.water_qa(meas)
            self.extrapolation_qa(meas)
            self.edges_qa(meas)
            self.check_bt_setting(meas)
            self.check_wt_settings(meas)
            self.check_depth_settings(meas)
            self.check_gps_settings(meas)
            self.check_edge_settings(meas)
            self.check_extrap_settings(meas)
            self.check_tempsal_settings(meas)
            self.check_mbt_settings(meas)
            self.check_compass_settings(meas)
        else:
            self.populate_from_qrev_mat(meas, mat_struct)

    def populate_from_qrev_mat(self, meas, meas_struct):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        meas: Measurement
            Object of Measurement
        meas_struct: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        # Generate a new QA object using the measurement data and the current QA code.
        # When QA checks from the current QA are not available from old QRev files, these
        # checks will be included to supplement the old QRev file data.
        new_qa = QAData(meas)
        if hasattr(meas_struct, 'qa'):
            # Set default thresholds
            self.q_run_threshold_caution = meas_struct.qa.qRunThresholdCaution
            self.q_run_threshold_warning = meas_struct.qa.qRunThresholdWarning
            if hasattr(meas_struct.qa, 'qTotalThresholdCaution'):
                self.q_total_threshold_caution = meas_struct.qa.qTotalThresholdCaution
            else:
                self.q_total_threshold_caution = 10
            self.q_total_threshold_warning = meas_struct.qa.qTotalThresholdWarning

            # Initialize instance variables
            self.transects = dict()
            self.transects['duration'] = meas_struct.qa.transects.duration
            self.transects['messages'] = self.make_list(meas_struct.qa.transects.messages)
            self.transects['number'] = meas_struct.qa.transects.number
            self.transects['recip'] = meas_struct.qa.transects.recip
            self.transects['sign'] = meas_struct.qa.transects.sign
            self.transects['status'] = meas_struct.qa.transects.status
            self.transects['uncertainty'] = meas_struct.qa.transects.uncertainty
            self.system_tst = dict()
            self.system_tst['messages'] = self.make_list(meas_struct.qa.systemTest.messages)
            self.system_tst['status'] = meas_struct.qa.systemTest.status
            self.compass = dict()
            self.compass['messages'] = self.make_list(meas_struct.qa.compass.messages)
            self.compass['status'] = meas_struct.qa.compass.status
            if hasattr(meas_struct.qa.compass, 'status1'):
                self.compass['status1'] = meas_struct.qa.compass.status1
                self.compass['status2'] = meas_struct.qa.compass.status2
            else:
                self.compass['status1'] = 'good'
                self.compass['status2'] = 'good'

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.compass, 'magvar'):
                self.compass['magvar'] = meas_struct.qa.compass.magvar
            else:
                self.compass['magvar'] = new_qa.compass['magvar']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.compass, 'magvarIdx'):
                self.compass['magvar_idx'] = self.make_array(meas_struct.qa.compass.magvarIdx)
            else:
                self.compass['magvar_idx'] = new_qa.compass['magvar_idx']

            # Changed mag_error_idx from bool to int array in QRevPy
            self.compass['mag_error_idx'] = new_qa.compass['mag_error_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.compass, 'pitchMeanWarningIdx'):
                self.compass['pitch_mean_warning_idx'] = self.make_array(meas_struct.qa.compass.pitchMeanWarningIdx)
            else:
                self.compass['pitch_mean_warning_idx'] = new_qa.compass['pitch_mean_warning_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.compass, 'rollMeanWarningIdx'):
                self.compass['roll_mean_warning_idx'] = self.make_array(meas_struct.qa.compass.rollMeanWarningIdx)
            else:
                self.compass['roll_mean_warning_idx'] = new_qa.compass['roll_mean_warning_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.compass, 'pitchMeanCautionIdx'):
                self.compass['pitch_mean_caution_idx'] = self.make_array(meas_struct.qa.compass.pitchMeanCautionIdx)
            else:
                self.compass['pitch_mean_caution_idx'] = new_qa.compass['pitch_mean_caution_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.compass, 'rollMeanCautionIdx'):
                self.compass['roll_mean_caution_idx'] = self.make_array(meas_struct.qa.compass.rollMeanCautionIdx)
            else:
                self.compass['roll_mean_caution_idx'] = new_qa.compass['roll_mean_caution_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.compass, 'pitchStdCautionIdx'):
                self.compass['pitch_std_caution_idx'] = self.make_array(meas_struct.qa.compass.pitchStdCautionIdx)
            else:
                self.compass['pitch_std_caution_idx'] = new_qa.compass['pitch_std_caution_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.compass, 'rollStdCautionIdx'):
                self.compass['roll_std_caution_idx'] = self.make_array(meas_struct.qa.compass.rollStdCautionIdx)
            else:
                self.compass['roll_std_caution_idx'] = new_qa.compass['roll_std_caution_idx']

            self.temperature = dict()
            self.temperature['messages'] = self.make_list(meas_struct.qa.temperature.messages)
            self.temperature['status'] = meas_struct.qa.temperature.status
            self.movingbed = dict()
            self.movingbed['messages'] = self.make_list(meas_struct.qa.movingbed.messages)
            self.movingbed['status'] = meas_struct.qa.movingbed.status
            self.movingbed['code'] = meas_struct.qa.movingbed.code
            self.user = dict()
            self.user['messages'] = self.make_list(meas_struct.qa.user.messages)
            self.user['sta_name'] = bool(meas_struct.qa.user.staName)
            self.user['sta_number'] = bool(meas_struct.qa.user.staNumber)
            self.user['status'] = meas_struct.qa.user.status

            # If QA check not available, get check from new QA
            self.depths = self.create_qa_dict(self, meas_struct.qa.depths)
            if 'draft' not in self.depths:
                self.depths['draft'] = new_qa.depths['draft']

            if 'all_invalid' not in self.depths:
                self.depths['all_invalid'] = new_qa.depths['all_invalid']

            # If QA check not available, get check from new QA
            self.bt_vel = self.create_qa_dict(self, meas_struct.qa.btVel, ndim=2)
            if 'all_invalid' not in self.bt_vel:
                self.bt_vel['all_invalid'] = new_qa.bt_vel['all_invalid']

            # If QA check not available, get check from new QA
            self.gga_vel = self.create_qa_dict(self, meas_struct.qa.ggaVel, ndim=2)
            if 'all_invalid' not in self.gga_vel:
                self.gga_vel['all_invalid'] = new_qa.gga_vel['all_invalid']

            # If QA check not available, get check from new QA
            self.vtg_vel = self.create_qa_dict(self, meas_struct.qa.vtgVel, ndim=2)
            if 'all_invalid' not in self.vtg_vel:
                self.vtg_vel['all_invalid'] = new_qa.vtg_vel['all_invalid']

            # If QA check not available, get check from new QA
            self.w_vel = self.create_qa_dict(self, meas_struct.qa.wVel, ndim=2)
            if 'all_invalid' not in self.w_vel:
                self.w_vel['all_invalid'] = new_qa.w_vel['all_invalid']

            self.extrapolation = dict()
            self.extrapolation['messages'] = self.make_list(meas_struct.qa.extrapolation.messages)
            self.extrapolation['status'] = meas_struct.qa.extrapolation.status
            self.edges = dict()
            self.edges['messages'] = self.make_list(meas_struct.qa.edges.messages)
            self.edges['status'] = meas_struct.qa.edges.status
            self.edges['left_q'] = meas_struct.qa.edges.leftQ
            self.edges['right_q'] = meas_struct.qa.edges.rightQ
            self.edges['left_sign'] = meas_struct.qa.edges.leftSign
            self.edges['right_sign'] = meas_struct.qa.edges.rightSign
            self.edges['left_zero'] = meas_struct.qa.edges.leftzero
            self.edges['right_zero'] = meas_struct.qa.edges.rightzero
            self.edges['left_type'] = meas_struct.qa.edges.leftType
            self.edges['right_type'] = meas_struct.qa.edges.rightType

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.edges, 'rightDistMovedIdx'):
                self.edges['right_dist_moved_idx'] = self.make_array(meas_struct.qa.edges.rightDistMovedIdx)
            else:
                self.edges['right_dist_moved_idx'] = new_qa.edges['right_dist_moved_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.edges, 'leftDistMovedIdx'):
                self.edges['left_dist_moved_idx'] = self.make_array(meas_struct.qa.edges.leftDistMovedIdx)
            else:
                self.edges['left_dist_moved_idx'] = new_qa.edges['left_dist_moved_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.edges, 'leftQIdx'):
                self.edges['left_q_idx'] = self.make_array(meas_struct.qa.edges.leftQIdx)
            else:
                self.edges['left_q_idx'] = new_qa.edges['left_q_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.edges, 'rightQIdx'):
                self.edges['right_q_idx'] = self.make_array(meas_struct.qa.edges.rightQIdx)
            else:
                self.edges['right_q_idx'] = new_qa.edges['right_q_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.edges, 'leftZeroIdx'):
                self.edges['left_zero_idx'] = self.make_array(meas_struct.qa.edges.leftZeroIdx)
            else:
                self.edges['left_zero_idx'] = new_qa.edges['left_zero_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.edges, 'rightZeroIdx'):
                self.edges['right_zero_idx'] = self.make_array(meas_struct.qa.edges.rightZeroIdx)
            else:
                self.edges['right_zero_idx'] = new_qa.edges['right_zero_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.edges, 'invalid_transect_left_idx'):
                self.edges['invalid_transect_left_idx'] = \
                    self.make_array(meas_struct.qa.edges.invalid_transect_left_idx)
            elif hasattr(meas_struct.qa.edges, 'invalidTransLeftIdx'):
                self.edges['invalid_transect_left_idx'] = \
                    self.make_array(meas_struct.qa.edges.invalidTransLeftIdx)
            else:
                self.edges['invalid_transect_left_idx'] = new_qa.edges['invalid_transect_left_idx']

            # If QA check not available, get check from new QA
            if hasattr(meas_struct.qa.edges, 'invalid_transect_right_idx'):
                self.edges['invalid_transect_right_idx'] = \
                    self.make_array(meas_struct.qa.edges.invalid_transect_right_idx)
            elif hasattr(meas_struct.qa, 'invalidTransRightIdx'):
                self.edges['invalid_transect_right_idx'] = \
                    self.make_array(meas_struct.qa.edges.invalidTransRightIdx)
            else:
                self.edges['invalid_transect_right_idx'] = new_qa.edges['invalid_transect_right_idx']

            if hasattr(meas_struct.qa, 'settings_dict'):
                self.settings_dict = dict()
                self.settings_dict['tab_compass'] = meas_struct.qa.settings_dict.tab_compass
                self.settings_dict['tab_tempsal'] = meas_struct.qa.settings_dict.tab_tempsal
                self.settings_dict['tab_mbt'] = meas_struct.qa.settings_dict.tab_mbt
                self.settings_dict['tab_bt'] = meas_struct.qa.settings_dict.tab_bt
                self.settings_dict['tab_gps'] = meas_struct.qa.settings_dict.tab_gps
                self.settings_dict['tab_depth'] = meas_struct.qa.settings_dict.tab_depth
                self.settings_dict['tab_wt'] = meas_struct.qa.settings_dict.tab_wt
                self.settings_dict['tab_extrap'] = meas_struct.qa.settings_dict.tab_extrap
                self.settings_dict['tab_edges'] = meas_struct.qa.settings_dict.tab_edges

    @staticmethod
    def create_qa_dict(self, mat_data, ndim=1):
        """Creates the dictionary used to store QA checks associated with the percent of discharge estimated
        by interpolation. This dictionary is used by BT, GPS, Depth, and WT.

        Parameters
        ----------
        self: QAData
            Object of QAData
        mat_data: mat_struct
            Matlab data from QRev file
        """

        # Initialize dictionary
        qa_dict = dict()

        # Populate dictionary from Matlab data
        qa_dict['messages'] = QAData.make_list(mat_data.messages)

        # allInvalid not available in older QRev data
        if hasattr(mat_data, 'allInvalid'):
            qa_dict['all_invalid'] = self.make_array(mat_data.allInvalid, 1).astype(bool)

        qa_dict['q_max_run_caution'] = self.make_array(mat_data.qRunCaution, ndim).astype(bool)
        qa_dict['q_max_run_warning'] = self.make_array(mat_data.qRunWarning, ndim).astype(bool)
        if hasattr(mat_data, 'qTotalCaution'):
            qa_dict['q_total_caution'] = self.make_array(mat_data.qTotalCaution, ndim).astype(bool)
        else:
            qa_dict['q_total_caution'] = self.make_array(mat_data.qTotalWarning, ndim).astype(bool)
        qa_dict['q_total_warning'] = self.make_array(mat_data.qTotalWarning, ndim).astype(bool)
        qa_dict['status'] = mat_data.status

        # q_max_run and q_total not available in older QRev data
        try:
            qa_dict['q_max_run'] = self.make_array(mat_data.qMaxRun, ndim)
            qa_dict['q_total'] = self.make_array(mat_data.qTotal, ndim)
        except AttributeError:
            qa_dict['q_max_run'] = np.tile(np.nan, (len(mat_data.qRunCaution), 6))
            qa_dict['q_total'] = np.tile(np.nan, (len(mat_data.qRunCaution), 6))
        return qa_dict

    @staticmethod
    def make_array(num_in, ndim=1):
        """Ensures that num_in is an array and if not makes it an array.

        num_in: any
            Any value or array
        """

        if type(num_in) is np.ndarray:
            if len(num_in.shape) < 2 and ndim > 1:
                num_in = np.reshape(num_in, (1, num_in.shape[0]))
                return num_in
            else:
                return num_in
        else:
            return np.array([num_in])

    @staticmethod
    def make_list(array_in):
        """Converts a string or array to a list.

        Parameters
        ----------
        array_in: any
            Data to be converted to list.

        Returns
        -------
        list_out: list
            List of array_in data
        """

        list_out = []
        # Convert string to list
        if type(array_in) is str:
            list_out = [array_in]
        else:
            # Empty array
            if array_in.size == 0:
                list_out = []
            # Single message with integer codes at end
            elif array_in.size == 3:
                if type(array_in[1]) is int or len(array_in[1].strip()) == 1:
                    temp = array_in.tolist()
                    if len(temp) > 0:
                        internal_list = []
                        for item in temp:
                            internal_list.append(item)
                        list_out = [internal_list]
                else:
                    list_out = array_in.tolist()
            # Either multiple messages with or without integer codes
            else:
                list_out = array_in.tolist()

        return list_out

    def transects_qa(self, meas):
        """Apply quality checks to transects

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        # Assume good results
        self.transects['status'] = 'good'

        # Initialize keys
        self.transects['messages'] = []
        self.transects['recip'] = 0
        self.transects['sign'] = 0
        self.transects['duration'] = 0
        self.transects['number'] = 0
        self.transects['uncertainty'] = 0

        # Initialize lists
        checked = []
        discharges = []
        start_edge = []

        # Populate lists
        for n in range(len(meas.transects)):
            checked.append(meas.transects[n].checked)
            if meas.transects[n].checked:
                discharges.append(meas.discharge[n])
                start_edge.append(meas.transects[n].start_edge)

        num_checked = np.nansum(np.asarray(checked))

        # Check duration
        total_duration = 0
        if num_checked >= 1:
            for transect in meas.transects:
                if transect.checked:
                    total_duration += transect.date_time.transect_duration_sec

        # Check duration against USGS policy
        if total_duration < 720:
            self.transects['status'] = 'caution'
            self.transects['messages'].append(
                ['Transects: Duration of selected transects is less than 720 seconds;', 2, 0])
            self.transects['duration'] = 1

        # Check transects for missing ensembles
        for transect in meas.transects:
            if transect.checked:

                # Determine number of missing ensembles
                if transect.adcp.manufacturer == 'SonTek':
                    # Determine number of missing ensembles for SonTek data
                    idx_missing = np.where(transect.date_time.ens_duration_sec > 1.5)[0]
                    if len(idx_missing) > 0:
                        average_ensemble_duration = (np.nansum(transect.date_time.ens_duration_sec)
                                                     - np.nansum(transect.date_time.ens_duration_sec[idx_missing])) \
                                                     / (len(transect.date_time.ens_duration_sec) - len(idx_missing))
                        num_missing = np.round(np.nansum(transect.date_time.ens_duration_sec[idx_missing])
                                               / average_ensemble_duration) - len(idx_missing)
                    else:
                        num_missing = 0
                else:
                    # Determine number of lost ensembles for TRDI data
                    idx_missing = np.where(np.isnan(transect.date_time.ens_duration_sec) == True)[0]
                    num_missing = len(idx_missing) - 1

                # Save caution message
                if num_missing > 0:
                    self.transects['messages'].append(['Transects: ' + str(transect.file_name) + ' is missing '
                                                       + str(int(num_missing)) + ' ensembles;', 2, 0])
                    self.transects['status'] = 'caution'

        # Check number of transects checked
        if num_checked == 0:
            # No transects selected
            self.transects['status'] = 'warning'
            self.transects['messages'].append(['TRANSECTS: No transects selected;', 1, 0])
            self.transects['number'] = 2
        elif num_checked == 1:
            # Only one transect selected
            self.transects['status'] = 'caution'
            self.transects['messages'].append(['Transects: Only one transect selected;', 2, 0])
            self.transects['number'] = 2
        else:
            self.transects['number'] = num_checked
            if num_checked == 2:
                # Only 2 transects selected
                cov, _ = Uncertainty.uncertainty_q_random(discharges, 'total')
                # Check uncertainty
                if cov > 2:
                    self.transects['status'] = 'caution'
                    self.transects['messages'].append(
                        ['Transects: Uncertainty would be reduced by additional transects;', 2, 0])

            # Check for consistent sign
            q_positive = []
            for q in discharges:
                if q.total >= 0:
                    q_positive.append(True)
                else:
                    q_positive.append(False)
            if len(np.unique(q_positive)) > 1:
                self.transects['status'] = 'warning'
                self.transects['messages'].append(
                    ['TRANSECTS: Sign of total Q is not consistent. One or more start banks may be incorrect;', 1, 0])

            # Check for reciprocal transects
            num_left = start_edge.count('Left')
            num_right = start_edge.count('Right')

            if not num_left == num_right:
                self.transects['status'] = 'warning'
                self.transects['messages'].append(['TRANSECTS: Transects selected are not reciprocal transects;', 1, 0])

        # Check for zero discharge transects
        q_zero = False
        for q in discharges:
            if q.total == 0:
                q_zero = True
        if q_zero:
            self.transects['status'] = 'warning'
            self.transects['messages'].append(['TRANSECTS: One or more transects have zero Q;', 1, 0])

    def system_tst_qa(self, meas):
        """Apply QA checks to system test.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.system_tst['messages'] = []
        self.system_tst['status'] = 'good'

        # Determine if a system test was recorded
        if not meas.system_tst:
            # No system test data recorded
            self.system_tst['status'] = 'warning'
            self.system_tst['messages'].append(['SYSTEM TEST: No system test;', 1, 3])
        else:

            pt3_fail = False
            num_tests_with_failure = 0

            for test in meas.system_tst:
                if hasattr(test, 'result'):

                    # Check for presence of pt3 test
                    if 'pt3' in test.result and test.result['pt3'] is not None:

                        # Check hard_limit, high gain, wide bandwidth
                        if 'hard_limit' in test.result['pt3']:
                            if 'high_wide' in test.result['pt3']['hard_limit']:
                                corr_table = test.result['pt3']['hard_limit']['high_wide']['corr_table']
                                if len(corr_table) > 0:
                                    # All lags past lag 2 should be less than 50% of lag 0
                                    qa_threshold = corr_table[0, :] * 0.5
                                    all_lag_check = np.greater(corr_table[3::, :], qa_threshold)

                                    # Lag 7 should be less than 25% of lag 0
                                    lag_7_check = np.greater(corr_table[7, :], corr_table[0, :] * 0.25)

                                    # If either condition is met for any beam the test fails
                                    if np.sum(np.sum(all_lag_check)) + np.sum(lag_7_check) > 1:
                                        pt3_fail = True

                    if test.result['sysTest']['n_failed'] is not None and test.result['sysTest']['n_failed'] > 0:
                        num_tests_with_failure += 1

            # pt3 test failure message
            if pt3_fail:
                self.system_tst['status'] = 'caution'
                self.system_tst['messages'].append(
                    ['System Test: One or more PT3 tests in the system test indicate potential EMI;', 2, 3])

            # Check for failed tests
            if num_tests_with_failure == len(meas.system_tst):
                # All tests had a failure
                self.system_tst['status'] = 'warning'
                self.system_tst['messages'].append(
                    ['SYSTEM TEST: All system test sets have at least one test that failed;', 1, 3])
            elif num_tests_with_failure > 0:
                self.system_tst['status'] = 'caution'
                self.system_tst['messages'].append(
                    ['System Test: One or more system test sets have at least one test that failed;', 2, 3])

    def compass_qa(self, meas):
        """Apply QA checks to compass calibration and evaluation.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.compass['messages'] = []

        checked = []
        for transect in meas.transects:
            checked.append(transect.checked)

        if np.any(checked):
            heading = np.unique(meas.transects[checked.index(1)].sensors.heading_deg.internal.data)
        else:
            heading = np.array([0])

        # Initialize variable as if ADCP has no compass
        self.compass['status'] = 'inactive'
        self.compass['status1'] = 'good'
        self.compass['status2'] = 'good'
        self.compass['magvar'] = 0
        self.compass['magvar_idx'] = []
        self.compass['mag_error_idx'] = []
        self.compass['pitch_mean_warning_idx'] = []
        self.compass['pitch_mean_caution_idx'] = []
        self.compass['pitch_std_caution_idx'] = []
        self.compass['roll_mean_warning_idx'] = []
        self.compass['roll_mean_caution_idx'] = []
        self.compass['roll_std_caution_idx'] = []

        if len(heading) > 1 and np.any(np.not_equal(heading, 0)):
            # ADCP has a compass
            # A compass calibration is required if a loop test or GPS are used

            # Check for loop test
            loop = False
            for test in meas.mb_tests:
                if test.type == 'Loop':
                    loop = True

            # Check for GPS data
            gps = False
            if meas.transects[checked.index(True)].boat_vel.gga_vel is not None or \
                    meas.transects[checked.index(True)].boat_vel.vtg_vel is not None:
                gps = True

            if gps or loop:
                # Compass calibration is required

                # Determine the ADCP manufacturer
                if meas.transects[checked.index(True)].adcp.manufacturer == 'SonTek':
                    # SonTek ADCP
                    if len(meas.compass_cal) == 0:
                        # No compass calibration
                        self.compass['status1'] = 'warning'
                        self.compass['messages'].append(['COMPASS: No compass calibration;', 1, 4])
                    elif meas.compass_cal[-1].result['compass']['error'] == 'N/A':
                        # If the error cannot be decoded from the calibration assume the calibration is good
                        self.compass['status1'] = 'good'
                    else:
                        if meas.compass_cal[-1].result['compass']['error'] <= 0.2:
                            self.compass['status1'] = 'good'
                        else:
                            self.compass['status1'] = 'caution'
                            self.compass['messages'].append(['Compass: Calibration result > 0.2 deg;', 2, 4])

                elif meas.transects[checked.index(True)].adcp.manufacturer == 'TRDI':
                    # TRDI ADCP
                    if len(meas.compass_cal) == 0:
                        # No compass calibration
                        if len(meas.compass_eval) == 0:
                            # No calibration or evaluation
                            self.compass['status1'] = 'warning'
                            self.compass['messages'].append(['COMPASS: No compass calibration or evaluation;', 1, 4])
                        else:
                            # No calibration but an evaluation was completed
                            self.compass['status1'] = 'caution'
                            self.compass['messages'].append(['Compass: No compass calibration;', 2, 4])
                    else:
                        # Compass was calibrated
                        if len(meas.compass_eval) == 0:
                            # No compass evaluation
                            self.compass['status1'] = 'caution'
                            self.compass['messages'].append(['Compass: No compass evaluation;', 2, 4])
                        else:
                            # Check results of evaluation
                            try:
                                if float(meas.compass_eval[-1].result['compass']['error']) <= 1:
                                    self.compass['status1'] = 'good'
                                else:
                                    self.compass['status1'] = 'caution'
                                    self.compass['messages'].append(['Compass: Evaluation result > 1 deg;', 2, 4])
                            except ValueError:
                                self.compass['status1'] = 'good'
            else:
                # Compass not required
                if len(meas.compass_cal) == 0 and len(meas.compass_eval) == 0:
                    # No compass calibration or evaluation
                    self.compass['status1'] = 'default'
                else:
                    # Compass was calibrated and evaluated
                    self.compass['status1'] = 'good'

            # Check for consistent magvar and pitch and roll mean and variation
            magvar = []
            align = []
            mag_error_exceeded = []
            pitch_mean = []
            pitch_std = []
            pitch_exceeded = []
            roll_mean = []
            roll_std = []
            roll_exceeded = []
            transect_idx = []
            for n, transect in enumerate(meas.transects):
                if transect.checked:
                    transect_idx.append(n)
                    heading_source_selected = getattr(
                        transect.sensors.heading_deg, transect.sensors.heading_deg.selected)
                    pitch_source_selected = getattr(transect.sensors.pitch_deg, transect.sensors.pitch_deg.selected)
                    roll_source_selected = getattr(transect.sensors.roll_deg, transect.sensors.roll_deg.selected)

                    magvar.append(transect.sensors.heading_deg.internal.mag_var_deg)
                    if transect.sensors.heading_deg.external is not None:
                        align.append(transect.sensors.heading_deg.external.align_correction_deg)

                    pitch_mean.append(np.nanmean(pitch_source_selected.data))
                    pitch_std.append(np.nanstd(pitch_source_selected.data, ddof=1))
                    roll_mean.append(np.nanmean(roll_source_selected.data))
                    roll_std.append(np.nanstd(roll_source_selected.data, ddof=1))

                    # SonTek G3 compass provides pitch, roll, and magnetic error parameters that can be checked
                    if transect.adcp.manufacturer == 'SonTek':
                        if heading_source_selected.pitch_limit is not None:
                            # Check for bug in SonTek data where pitch and roll was n x 3 use n x 1
                            if len(pitch_source_selected.data.shape) == 1:
                                pitch_data = pitch_source_selected.data
                            else:
                                pitch_data = pitch_source_selected.data[:, 0]
                            idx_max = np.where(pitch_data > heading_source_selected.pitch_limit[0])[0]
                            idx_min = np.where(pitch_data < heading_source_selected.pitch_limit[1])[0]
                            if len(idx_max) > 0 or len(idx_min) > 0:
                                pitch_exceeded.append(True)
                            else:
                                pitch_exceeded.append(False)

                        if heading_source_selected.roll_limit is not None:
                            if len(roll_source_selected.data.shape) == 1:
                                roll_data = roll_source_selected.data
                            else:
                                roll_data = roll_source_selected.data[:, 0]
                            idx_max = np.where(roll_data > heading_source_selected.pitch_limit[0])[0]
                            idx_min = np.where(roll_data < heading_source_selected.pitch_limit[1])[0]
                            if len(idx_max) > 0 or len(idx_min) > 0:
                                roll_exceeded.append(True)
                            else:
                                roll_exceeded.append(False)

                        if heading_source_selected.mag_error is not None:
                            idx_max = np.where(heading_source_selected.mag_error > 2)[0]
                            if len(idx_max) > 0:
                                mag_error_exceeded.append(n)
            # Check magvar consistency
            if len(np.unique(magvar)) > 1:
                self.compass['status2'] = 'caution'
                self.compass['messages'].append(
                    ['Compass: Magnetic variation is not consistent among transects;', 2, 4])
                self.compass['magvar'] = 1

            # Check magvar consistency
            if len(np.unique(align)) > 1:
                self.compass['status2'] = 'caution'
                self.compass['messages'].append(
                    ['Compass: Heading offset is not consistent among transects;', 2, 4])
                self.compass['align'] = 1

            # Check that magvar was set if GPS data are available
            if gps:
                if 0 in magvar:
                    self.compass['status2'] = 'warning'
                    self.compass['messages'].append(
                        ['COMPASS: Magnetic variation is 0 and GPS data are present;', 1, 4])
                    self.compass['magvar'] = 2
                    self.compass['magvar_idx'] = np.where(np.array(magvar) == 0)[0].tolist()

            # Check pitch mean
            if np.any(np.asarray(np.abs(pitch_mean)) > 8):
                self.compass['status2'] = 'warning'
                self.compass['messages'].append(['PITCH: One or more transects have a mean pitch > 8 deg;', 1, 4])
                temp = np.where(np.abs(pitch_mean) > 8)[0]
                if len(temp) > 0:
                    self.compass['pitch_mean_warning_idx'] = np.array(transect_idx)[temp]
                else:
                    self.compass['pitch_mean_warning_idx'] = []

            elif np.any(np.asarray(np.abs(pitch_mean)) > 4):
                if self.compass['status2'] == 'good':
                    self.compass['status2'] = 'caution'
                self.compass['messages'].append(['Pitch: One or more transects have a mean pitch > 4 deg;', 2, 4])
                temp = np.where(np.abs(pitch_mean) > 4)[0]
                if len(temp) > 0:
                    self.compass['pitch_mean_caution_idx'] = np.array(transect_idx)[temp]
                else:
                    self.compass['pitch_mean_caution_idx'] = []

            # Check roll mean
            if np.any(np.asarray(np.abs(roll_mean)) > 8):
                self.compass['status2'] = 'warning'
                self.compass['messages'].append(['ROLL: One or more transects have a mean roll > 8 deg;', 1, 4])
                temp = np.where(np.abs(roll_mean) > 8)[0]
                if len(temp) > 0:
                    self.compass['roll_mean_warning_idx'] = np.array(transect_idx)[temp]
                else:
                    self.compass['roll_mean_warning_idx'] = []

            elif np.any(np.asarray(np.abs(roll_mean)) > 4):
                if self.compass['status2'] == 'good':
                    self.compass['status2'] = 'caution'
                self.compass['messages'].append(['Roll: One or more transects have a mean roll > 4 deg;', 2, 4])
                temp = np.where(np.abs(roll_mean) > 4)[0]
                if len(temp) > 0:
                    self.compass['roll_mean_caution_idx'] = np.array(transect_idx)[temp]
                else:
                    self.compass['roll_mean_caution_idx'] = []

            # Check pitch standard deviation
            if np.any(np.asarray(pitch_std) > 5):
                if self.compass['status2'] == 'good':
                    self.compass['status2'] = 'caution'
                self.compass['messages'].append(['Pitch: One or more transects have a pitch std dev > 5 deg;', 2, 4])
                temp = np.where(np.abs(pitch_std) > 5)[0]
                if len(temp) > 0:
                    self.compass['pitch_std_caution_idx'] = np.array(transect_idx)[temp]
                else:
                    self.compass['pitch_std_caution_idx'] = []

            # Check roll standard deviation
            if np.any(np.asarray(roll_std) > 5):
                if self.compass['status2'] == 'good':
                    self.compass['status2'] = 'caution'
                self.compass['messages'].append(['Roll: One or more transects have a roll std dev > 5 deg;', 2, 4])
                temp = np.where(np.abs(roll_std) > 5)[0]
                if len(temp) > 0:
                    self.compass['roll_std_caution_idx'] = np.array(transect_idx)[temp]
                else:
                    self.compass['roll_std_caution_idx'] = []

            # Additional checks for SonTek G3 compass
            if meas.transects[checked.index(True)].adcp.manufacturer == 'SonTek':
                # Check if pitch limits were exceeded
                if any(pitch_exceeded):
                    if self.compass['status2'] == 'good':
                        self.compass['status2'] = 'caution'
                    self.compass['messages'].append(
                        ['Compass: One or more transects have pitch exceeding calibration limits;', 2, 4])

                # Check if roll limits were exceeded
                if any(roll_exceeded):
                    if self.compass['status2'] == 'good':
                        self.compass['status2'] = 'caution'
                    self.compass['messages'].append(
                        ['Compass: One or more transects have roll exceeding calibration limits;', 2, 4])

                # Check if magnetic error was exceeded
                self.compass['mag_error_idx'] = []
                if len(mag_error_exceeded) > 0:
                    self.compass['mag_error_idx'] = np.array(mag_error_exceeded)
                    if self.compass['status2'] == 'good':
                        self.compass['status2'] = 'caution'
                    self.compass['messages'].append(
                        ['Compass: One or more transects have a change in mag field exceeding 2%;', 2, 4])

            if self.compass['status1'] == 'warning' or self.compass['status2'] == 'warning':
                self.compass['status'] = 'warning'
            elif self.compass['status1'] == 'caution' or self.compass['status2'] == 'caution':
                self.compass['status'] = 'caution'
            else:
                self.compass['status'] = 'good'

    def temperature_qa(self, meas):
        """Apply QA checks to temperature.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.temperature['messages'] = []
        check = [0, 0]

        # Create array of all temperatures
        temp = np.array([])
        checked = []
        for transect in meas.transects:
            if transect.checked:
                checked.append(transect.checked)
                temp_selected = getattr(transect.sensors.temperature_deg_c, transect.sensors.temperature_deg_c.selected)
                if len(temp) == 0:
                    temp = temp_selected.data
                else:
                    temp = np.hstack((temp, temp_selected.data))

        # Check temperature range
        if np.any(checked):
            temp_range = np.nanmax(temp) - np.nanmin(temp)
        else:
            temp_range = 0

        if temp_range > 2:
            check[0] = 3
            self.temperature['messages'].append(['TEMPERATURE: Temperature range is '
                                                 + '{:3.1f}'.format(temp_range)
                                                 + ' degrees C which is greater than 2 degrees;', 1, 5])
        elif temp_range > 1:
            check[0] = 2
            self.temperature['messages'].append(['Temperature: Temperature range is '
                                                 + '{:3.1f}'.format(temp_range)
                                                 + ' degrees C which is greater than 1 degree;', 2, 5])
        else:
            check[0] = 1

        # Check for independent temperature reading
        if 'user' in meas.ext_temp_chk:
            try:
                user = float(meas.ext_temp_chk['user'])
            except (ValueError, TypeError):
                user = None
            if user is None or np.isnan(user):
                # No independent temperature reading
                check[1] = 2
                self.temperature['messages'].append(['Temperature: No independent temperature reading;', 2, 5])
            elif not np.isnan(meas.ext_temp_chk['adcp']):
                # Compare user to manually entered ADCP temperature
                diff = np.abs(user - meas.ext_temp_chk['adcp'])
                if diff < 2:
                    check[1] = 1
                else:
                    check[1] = 3
                    self.temperature['messages'].append(
                        ['TEMPERATURE: The difference between ADCP and reference is > 2:  '
                         + '{:3.1f}'.format(diff) + ' C;', 1, 5])
            else:
                # Compare user to mean of all temperature data
                diff = np.abs(user - np.nanmean(temp))
                if diff < 2:
                    check[1] = 1
                else:
                    check[1] = 3
                    self.temperature['messages'].append(
                        ['TEMPERATURE: The difference between ADCP and reference is > 2:  '
                         + '{:3.1f}'.format(diff) + ' C;', 1, 5])

        # Assign temperature status
        max_check = max(check)
        if max_check == 1:
            self.temperature['status'] = 'good'
        elif max_check == 2:
            self.temperature['status'] = 'caution'
        elif max_check == 3:
            self.temperature['status'] = 'warning'

    def moving_bed_qa(self, meas):
        """Applies quality checks to moving-bed tests.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.movingbed['messages'] = []
        self.movingbed['code'] = 0

        # Are there moving-bed tests?
        if len(meas.mb_tests) < 1:
            # No moving-bed test
            self.movingbed['messages'].append(['MOVING-BED TEST: No moving bed test;', 1, 6])
            self.movingbed['status'] = 'warning'
            self.movingbed['code'] = 3

        else:
            # Moving-bed tests available
            mb_data = meas.mb_tests

            user_valid_test = []
            file_names = []
            idx_selected = []
            test_quality = []
            mb_tests = []
            mb = []
            mb_test_type = []
            loop = []
            gps_diff1 = False
            gps_diff2 = False

            for n, test in enumerate(mb_data):
                # Are tests valid according to the user
                if test.user_valid:
                    user_valid_test.append(True)
                    file_names.append(test.transect.file_name)
                    if test.type == 'Loop' and not test.test_quality == 'Errors':
                        loop.append(test.moving_bed)
                    if not np.isnan(test.gps_percent_mb):
                        if np.abs(test.bt_percent_mb - test.gps_percent_mb) > 2:
                            gps_diff2 = True
                        if np.logical_xor(test.bt_percent_mb >= 1, test.gps_percent_mb >= 1):
                            gps_diff1 = True
                    # Selected test
                    if test.selected:
                        idx_selected.append(n)
                        test_quality.append(test.test_quality)
                        mb_tests.append(test)
                        mb.append(test.moving_bed)
                        mb_test_type.append(test.type)
                else:
                    user_valid_test.append(False)

            if not any(user_valid_test):
                # No valid test according to user
                self.movingbed['messages'].append(['MOVING-BED TEST: No valid moving-bed test based on user input;',
                                                   1, 6])
                self.movingbed['status'] = 'warning'
                self.movingbed['code'] = 3
            else:
                # Check for duplicate valid moving-bed tests
                if len(np.unique(file_names)) < len(file_names):
                    self.movingbed['messages'].append([
                        'MOVING-BED TEST: Duplicate moving-bed test files marked valid;', 1, 6])
                    self.movingbed['status'] = 'warning'
                    self.movingbed['code'] = 3

            if self.movingbed['code'] == 0:
                # Check test quality
                if len(test_quality) > 0 and sum(np.array(test_quality) == 'Good') > 0:
                    self.movingbed['status'] = 'good'
                    self.movingbed['code'] = 1

                    # Check if there is a moving-bed
                    if 'Yes' in mb:
                        # Moving-bed present
                        self.movingbed['messages'].append(
                            ['Moving-Bed Test: A moving-bed is present, use GPS or moving-bed correction;', 2, 6])
                        self.movingbed['code'] = 2
                        self.movingbed['status'] = 'caution'

                        # Check for test type
                        if sum(np.array(mb_test_type) == 'Stationary'):
                            # Check for GPS or 3 stationary tests
                            if len(mb_tests) < 3:
                                gps = []
                                for transect in meas.transects:
                                    if transect.checked:
                                        if transect.gps is None:
                                            gps.append(False)
                                        else:
                                            gps.append(True)
                                if not all(gps):
                                    # GPS not available for all selected transects
                                    self.movingbed['messages'].append([
                                        'Moving-Bed Test: '
                                        + 'Less than 3 stationary tests available for moving-bed correction;',
                                        2, 6])

                elif len(test_quality) > 0 and sum(np.array(test_quality) == 'Warnings') > 0:
                    # Quality check has warnings
                    self.movingbed['messages'].append(['Moving-Bed Test: The moving-bed test(s) has warnings, '
                                                       + 'please review tests to determine validity;', 2, 6])
                    self.movingbed['status'] = 'caution'
                    self.movingbed['code'] = 2

                elif len(test_quality) > 0 and sum(np.array(test_quality) == 'Manual') > 0:
                    # Manual override used
                    self.movingbed['messages'].append(['MOVING-BED TEST: '
                                                       + 'The user has manually forced the use of some tests;', 1, 6])
                    self.movingbed['status'] = 'warning'
                    self.movingbed['code'] = 3

                else:
                    # Test has critical errors
                    self.movingbed['messages'].append(['MOVING-BED TEST: The moving-bed test(s) have critical errors '
                                                       + 'and will not be used;', 1, 6])
                    self.movingbed['status'] = 'warning'
                    self.movingbed['code'] = 3

                # Check multiple loops for consistency
                if len(np.unique(loop)) > 1:
                    self.movingbed['messages'].append(['Moving-Bed Test: Results of valid loops are not consistent, '
                                                       + 'review moving-bed tests;', 2, 6])
                    if self.movingbed['code'] < 3:
                        self.movingbed['code'] = 2
                        self.movingbed['status'] = 'caution'

                # Notify of differences in results of test between BT and GPS
                if gps_diff2:
                    self.movingbed['messages'].append(['Moving-Bed Test: Bottom track and '
                                                      'GPS results differ by more than 2%.', 2, 6])
                    if self.movingbed['code'] < 3:
                        self.movingbed['code'] = 2
                        self.movingbed['status'] = 'caution'

                if gps_diff1:
                    self.movingbed['messages'].append(['Moving-Bed Test: Bottom track and GPS results do not agree.',
                                                      2, 6])
                    if self.movingbed['code'] < 3:
                        self.movingbed['code'] = 2
                        self.movingbed['status'] = 'caution'

        self.check_mbt_settings(meas)

    def user_qa(self, meas):
        """Apply quality checks to user input data.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.user['messages'] = []
        self.user['status'] = 'good'

        # Check for Station Name
        self.user['sta_name'] = False
        if meas.station_name is None or len(meas.station_name.strip()) < 1:
            self.user['messages'].append(['Site Info: Station name not entered;', 2, 2])
            self.user['status'] = 'caution'
            self.user['sta_name'] = True

        # Check for Station Number
        self.user['sta_number'] = False
        try:
            if meas.station_number is None or len(meas.station_number.strip()) < 1:
                self.user['messages'].append(['Site Info: Station number not entered;', 2, 2])
                self.user['status'] = 'caution'
                self.user['sta_number'] = True
        except AttributeError:
            self.user['messages'].append(['Site Info: Station number not entered;', 2, 2])
            self.user['status'] = 'caution'
            self.user['sta_number'] = True

    def depths_qa(self, meas):
        """Apply quality checks to depth data.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        # Initialize variables
        n_transects = len(meas.transects)
        self.depths['q_total'] = np.tile(np.nan, n_transects)
        self.depths['q_max_run'] = np.tile(np.nan, n_transects)
        self.depths['q_total_caution'] = np.tile(False, n_transects)
        self.depths['q_max_run_caution'] = np.tile(False, n_transects)
        self.depths['q_total_warning'] = np.tile(False, n_transects)
        self.depths['q_max_run_warning'] = np.tile(False, n_transects)
        self.depths['all_invalid'] = np.tile(False, n_transects)
        self.depths['messages'] = []
        self.depths['status'] = 'good'
        self.depths['draft'] = 0
        checked = []
        drafts = []
        for n, transect in enumerate(meas.transects):
            checked.append(transect.checked)
            if transect.checked:
                in_transect_idx = transect.in_transect_idx

                depths_selected = getattr(transect.depths, transect.depths.selected)
                drafts.append(depths_selected.draft_use_m)

                # Determine valid measured depths
                if transect.depths.composite:
                    depth_na = depths_selected.depth_source_ens[in_transect_idx] != 'NA'
                    depth_in = depths_selected.depth_source_ens[in_transect_idx] != 'IN'
                    depth_valid = np.all(np.vstack((depth_na, depth_in)), 0)
                else:
                    depth_valid_temp = depths_selected.valid_data[in_transect_idx]
                    depth_nan = depths_selected.depth_processed_m[in_transect_idx] != np.nan
                    depth_valid = np.all(np.vstack((depth_nan, depth_valid_temp)), 0)

                if not np.any(depth_valid):
                    self.depths['all_invalid'][n] = True

                # Compute QA characteristics
                q_total, q_max_run, number_invalid_ensembles = QAData.invalid_qa(depth_valid, meas.discharge[n])
                self.depths['q_total'][n] = q_total
                self.depths['q_max_run'][n] = q_max_run

                # Compute percentage compared to total
                if meas.discharge[n].total == 0.0:
                    q_total_percent = np.nan
                    q_max_run_percent = np.nan
                else:
                    q_total_percent = np.abs((q_total / meas.discharge[n].total) * 100)
                    q_max_run_percent = np.abs((q_max_run / meas.discharge[n].total) * 100)

                # Apply total interpolated discharge threshold
                if q_total_percent > self.q_total_threshold_warning:
                    self.depths['q_total_warning'][n] = True
                elif q_total_percent > self.q_total_threshold_caution:
                    self.depths['q_total_caution'][n] = True

                # Apply interpolated discharge run thresholds
                if q_max_run_percent > self.q_run_threshold_warning:
                    self.depths['q_max_run_warning'][n] = True
                elif q_max_run_percent > self.q_run_threshold_caution:
                    self.depths['q_max_run_caution'][n] = True

        if checked:

            # Create array of all unique draft values
            draft_check = np.unique(np.round(drafts, 2))

            # Check draft consistency
            if len(draft_check) > 1:
                self.depths['status'] = 'caution'
                self.depths['draft'] = 1
                self.depths['messages'].append(['Depth: Transducer depth is not consistent among transects;', 2, 10])

            # Check for zero draft
            if np.any(np.less(draft_check, 0.01)):
                self.depths['status'] = 'warning'
                self.depths['draft'] = 2
                self.depths['messages'].append(['DEPTH: Transducer depth is too shallow, likely 0;', 1, 10])

            # Check consecutive interpolated discharge criteria
            if np.any(self.depths['q_max_run_warning']):
                self.depths['messages'].append(['DEPTH: Int. Q for consecutive invalid ensembles exceeds '
                                                + '%2.0f' % self.q_run_threshold_warning + '%;', 1, 10])
                self.depths['status'] = 'warning'
            elif np.any(self.depths['q_max_run_caution']):
                self.depths['messages'].append(['Depth: Int. Q for consecutive invalid ensembles exceeds '
                                                + '%2.0f' % self.q_run_threshold_caution + '%;', 2, 10])
                self.depths['status'] = 'caution'

            # Check total interpolated discharge criteria
            if np.any(self.depths['q_total_warning']):
                self.depths['messages'].append(['DEPTH: Int. Q for invalid ensembles in a transect exceeds '
                                                + '%2.0f' % self.q_total_threshold_warning + '%;', 1, 10])
                self.depths['status'] = 'warning'
            elif np.any(self.depths['q_total_caution']):
                self.depths['messages'].append(['Depth: Int. Q for invalid ensembles in a transect exceeds '
                                                + '%2.0f' % self.q_total_threshold_caution + '%;', 2, 10])
                self.depths['status'] = 'caution'

            # Check if all depths are invalid
            if np.any(self.depths['all_invalid']):
                self.depths['messages'].append(['DEPTH: There are no valid depths for one or more transects.', 2, 10])
                self.depths['status'] = 'warning'

        else:
            self.depths['status'] = 'inactive'

    def boat_qa(self, meas):
        """Apply quality checks to boat data.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        # Initialize variables
        n_transects = len(meas.transects)
        data_type = {'BT': {'class': 'bt_vel', 'warning': 'BT-', 'caution': 'bt-',
                            'filter': [('All: ', 0), ('Original: ', 1), ('ErrorVel: ', 2),
                                       ('VertVel: ', 3), ('Other: ', 4), ('3Beams: ', 5)]},
                     'GGA': {'class': 'gga_vel', 'warning': 'GGA-', 'caution': 'gga-',
                             'filter': [('All: ', 0), ('Original: ', 1), ('DGPS: ', 2),
                                        ('Altitude: ', 3), ('Other: ', 4), ('HDOP: ', 5)]},
                     'VTG': {'class': 'vtg_vel', 'warning': 'VTG-', 'caution': 'vtg-',
                             'filter': [('All: ', 0), ('Original: ', 1), ('Other: ', 4), ('HDOP: ', 5)]}}
        self.boat['messages'] = []

        for dt_key, dt_value in data_type.items():
            boat = getattr(self, dt_value['class'])

            # Initialize dictionaries for each data type
            boat['q_total_caution'] = np.tile(False, (n_transects, 6))
            boat['q_max_run_caution'] = np.tile(False, (n_transects, 6))
            boat['q_total_warning'] = np.tile(False, (n_transects, 6))
            boat['q_max_run_warning'] = np.tile(False, (n_transects, 6))
            boat['all_invalid'] = np.tile(False, n_transects)
            boat['q_total'] = np.tile(np.nan, (n_transects, 6))
            boat['q_max_run'] = np.tile(np.nan, (n_transects, 6))
            boat['messages'] = []
            status_switch = 0
            avg_speed_check = 0

            # Check the results of each filter
            for dt_filter in dt_value['filter']:
                boat['status'] = 'inactive'

                # Quality check each transect
                for n, transect in enumerate(meas.transects):

                    # Evaluate on transects used in the discharge computation
                    if transect.checked:

                        in_transect_idx = transect.in_transect_idx

                        # Check to see if data are available for the data_type
                        if getattr(transect.boat_vel, dt_value['class']) is not None:
                            boat['status'] = 'good'

                            # Compute quality characteristics
                            valid = getattr(transect.boat_vel, dt_value['class']).valid_data[dt_filter[1],
                                                                                             in_transect_idx]
                            q_total, q_max_run, number_invalid_ens = QAData.invalid_qa(valid, meas.discharge[n])
                            boat['q_total'][n, dt_filter[1]] = q_total
                            boat['q_max_run'][n, dt_filter[1]] = q_max_run

                            # Compute percentage compared to total
                            if meas.discharge[n].total == 0.0:
                                q_total_percent = np.nan
                                q_max_run_percent = np.nan
                            else:
                                q_total_percent = np.abs((q_total / meas.discharge[n].total) * 100)
                                q_max_run_percent = np.abs((q_max_run / meas.discharge[n].total) * 100)

                            # Check if all invalid
                            if dt_filter[1] == 0 and not np.any(valid):
                                boat['all_invalid'][n] = True

                            # Apply total interpolated discharge threshold
                            if q_total_percent > self.q_total_threshold_warning:
                                boat['q_total_warning'][n, dt_filter[1]] = True
                            elif q_total_percent > self.q_total_threshold_caution:
                                boat['q_total_caution'][n, dt_filter[1]] = True

                            # Apply interpolated discharge run thresholds
                            if q_max_run_percent > self.q_run_threshold_warning:
                                boat['q_max_run_warning'][n, dt_filter[1]] = True
                            elif q_max_run_percent > self.q_run_threshold_caution:
                                boat['q_max_run_caution'][n, dt_filter[1]] = True

                            # Check boat velocity for vtg data
                            if dt_key == 'VTG' and transect.boat_vel.selected == 'vtg_vel' and avg_speed_check == 0:
                                if transect.boat_vel.vtg_vel.u_mps is not None:
                                    avg_speed = np.nanmean((transect.boat_vel.vtg_vel.u_mps ** 2
                                                            + transect.boat_vel.vtg_vel.v_mps ** 2) ** 0.5)
                                    if avg_speed < 0.24:
                                        boat['q_total_caution'][n, 2] = True
                                        if status_switch < 1:
                                            status_switch = 1
                                        boat['messages'].append(
                                            ['vtg-AvgSpeed: VTG data may not be accurate for average boat speed '
                                             'less than' + '0.24 m/s (0.8 ft/s);', 2, 8])
                                        avg_speed_check = 1

                # Create message for consecutive invalid discharge
                if boat['q_max_run_warning'][:, dt_filter[1]].any():
                    if dt_key == 'BT':
                        module_code = 7
                    else:
                        module_code = 8
                    boat['messages'].append(
                        [dt_value['warning'] + dt_filter[0] +
                         'Int. Q for consecutive invalid ensembles exceeds ' +
                         '%3.1f' % self.q_run_threshold_warning + '%;', 1, module_code])
                    status_switch = 2
                elif boat['q_max_run_caution'][:, dt_filter[1]].any():
                    if dt_key == 'BT':
                        module_code = 7
                    else:
                        module_code = 8
                    boat['messages'].append(
                        [dt_value['caution'] + dt_filter[0] +
                         'Int. Q for consecutive invalid ensembles exceeds ' +
                         '%3.1f' % self.q_run_threshold_caution + '%;', 2, module_code])
                    if status_switch < 1:
                        status_switch = 1

                # Create message for total invalid discharge
                if boat['q_total_warning'][:, dt_filter[1]].any():
                    if dt_key == 'BT':
                        module_code = 7
                    else:
                        module_code = 8
                    boat['messages'].append(
                        [dt_value['warning'] + dt_filter[0] +
                         'Int. Q for invalid ensembles in a transect exceeds ' +
                         '%3.1f' % self.q_total_threshold_warning + '%;', 1, module_code])
                    status_switch = 2
                elif boat['q_total_caution'][:, dt_filter[1]].any():
                    if dt_key == 'BT':
                        module_code = 7
                    else:
                        module_code = 8
                    boat['messages'].append(
                        [dt_value['caution'] + dt_filter[0] +
                         'Int. Q for invalid ensembles in a transect exceeds ' +
                         '%3.1f' % self.q_total_threshold_caution + '%;', 2, module_code])
                    if status_switch < 1:
                        status_switch = 1

            # Create message for all data invalid
            if boat['all_invalid'].any():
                boat['status'] = 'warning'
                if dt_key == 'BT':
                    module_code = 7
                else:
                    module_code = 8
                boat['messages'].append(
                    [dt_value['warning'] + dt_value['filter'][0][0] +
                     'There are no valid data for one or more transects.;', 1, module_code])

            # Set status
            if status_switch == 2:
                boat['status'] = 'warning'
            elif status_switch == 1:
                boat['status'] = 'caution'

            setattr(self, dt_value['class'], boat)

        lag_gga = []
        lag_vtg = []
        for transect in meas.transects:
            gga, vtg = TransectData.compute_gps_lag(transect)
            if gga is not None:
                lag_gga.append(gga)
            if vtg is not None:
                lag_vtg.append(vtg)
        if len(lag_gga) > 0:
            if np.mean(np.abs(lag_gga)) > 10:
                self.gga_vel['messages'].append(['GGA: BT and GGA do not appear to be sychronized', 1, 8])
                if self.gga_vel['status'] != 'warning':
                    self.gga_vel['status'] = 'warning'
            elif np.mean(np.abs(lag_gga)) > 2:
                self.gga_vel['messages'].append(['gga: Lag between BT and GGA > 2 sec', 2, 8])
                if self.gga_vel['status'] != 'warning':
                    self.gga_vel['status'] = 'caution'
        if len(lag_vtg) > 0:
            if np.mean(np.abs(lag_vtg)) > 10:
                self.vtg_vel['messages'].append(['VTG: BT and VTG do not appear to be sychronized', 1, 8])
                if self.vtg_vel['status'] != 'warning':
                    self.vtg_vel['status'] = 'warning'
            elif np.mean(np.abs(lag_vtg)) > 2:
                self.vtg_vel['messages'].append(['vtg: Lag between BT and VTG > 2 sec', 2, 8])
                if self.vtg_vel['status'] != 'warning':
                    self.vtg_vel['status'] = 'caution'

    def water_qa(self, meas):
        """Apply quality checks to water data.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        # Initialize filter labels and indices
        prefix = ['All: ', 'Original: ', 'ErrorVel: ', 'VertVel: ', 'Other: ', '3Beams: ', 'SNR:']
        if meas.transects[0].adcp.manufacturer == 'TRDI':
            filter_index = [0, 1, 2, 3, 4, 5]
        else:
            filter_index = [0, 1, 2, 3, 4, 5, 7]

        n_transects = len(meas.transects)
        n_filters = len(filter_index) + 1
        # Initialize dictionaries for each data type
        self.w_vel['q_total_caution'] = np.tile(False, (n_transects, n_filters))
        self.w_vel['q_max_run_caution'] = np.tile(False, (n_transects, n_filters))
        self.w_vel['q_total_warning'] = np.tile(False, (n_transects, n_filters))
        self.w_vel['q_max_run_warning'] = np.tile(False, (n_transects, n_filters))
        self.w_vel['all_invalid'] = np.tile(False, n_transects)
        self.w_vel['q_total'] = np.tile(np.nan, (n_transects, n_filters))
        self.w_vel['q_max_run'] = np.tile(np.nan, (n_transects, n_filters))
        self.w_vel['messages'] = []
        status_switch = 0

        # TODO if meas had a property checked as list it would save creating that list multiple times
        checked = []
        for transect in meas.transects:
            checked.append(transect.checked)

        # At least one transect is being used to compute discharge
        if any(checked):
            # Loop through filters
            for prefix_idx, filter_idx in enumerate(filter_index):
                # Loop through transects
                for n, transect in enumerate(meas.transects):
                    if transect.checked:
                        valid_original = np.any(transect.w_vel.valid_data[1, :, transect.in_transect_idx].T, 0)

                        # Determine what data each filter have marked invalid. Original invalid data are excluded
                        valid = np.any(transect.w_vel.valid_data[filter_idx, :, transect.in_transect_idx].T, 0)
                        if filter_idx > 1:
                            valid_int = valid.astype(int) - valid_original.astype(int)
                            valid = valid_int != -1

                        # Check if all data are invalid
                        if filter_idx == 0:
                            if np.nansum(valid.astype(int)) < 1:
                                self.w_vel['all_invalid'][n] = True
                        # TODO seems like the rest of this should be under else of all invalid or multiple messages
                        # generated.

                        # Compute characteristics
                        q_total, q_max_run, number_invalid_ens = QAData.invalid_qa(valid, meas.discharge[n])
                        self.w_vel['q_total'][n, filter_idx] = q_total
                        self.w_vel['q_max_run'][n, filter_idx] = q_max_run

                        # Compute percentage compared to total
                        if meas.discharge[n].total == 0.0:
                            q_total_percent = np.nan
                            q_max_run_percent = np.nan
                        else:
                            q_total_percent = np.abs((q_total / meas.discharge[n].total) * 100)
                            q_max_run_percent = np.abs((q_max_run / meas.discharge[n].total) * 100)

                        # Check total invalid discharge in ensembles for warning
                        if q_total_percent > self.q_total_threshold_warning:
                            self.w_vel['q_total_warning'][n, filter_idx] = True

                        # Apply run or cluster thresholds
                        if q_max_run_percent > self.q_run_threshold_warning:
                            self.w_vel['q_max_run_warning'][n, filter_idx] = True
                        elif q_max_run_percent > self.q_run_threshold_caution:
                            self.w_vel['q_max_run_caution'][n, filter_idx] = True

                        # Compute percent discharge interpolated for both cells and ensembles
                        # This approach doesn't exclude original data
                        valid_cells = transect.w_vel.valid_data[filter_idx, :, transect.in_transect_idx].T
                        q_invalid_total = np.nansum(meas.discharge[n].middle_cells[np.logical_not(valid_cells)]) \
                            + np.nansum(meas.discharge[n].top_ens[np.logical_not(valid)]) \
                            + np.nansum(meas.discharge[n].bottom_ens[np.logical_not(valid)])
                        q_invalid_total_percent = (q_invalid_total / meas.discharge[n].total) * 100

                        if q_invalid_total_percent > self.q_total_threshold_caution:
                            self.w_vel['q_total_caution'][n, filter_idx] = True

                # Generate messages for ensemble run or clusters
                if np.any(self.w_vel['q_max_run_warning'][:, filter_idx]):
                    self.w_vel['messages'].append(['WT-' + prefix[prefix_idx]
                                                   + 'Int. Q for consecutive invalid ensembles exceeds '
                                                   + '%3.0f' % self.q_run_threshold_warning
                                                   + '%;', 1, 11])
                    status_switch = 2
                elif np.any(self.w_vel['q_max_run_caution'][:, filter_idx]):
                    self.w_vel['messages'].append(['wt-' + prefix[prefix_idx]
                                                   + 'Int. Q for consecutive invalid ensembles exceeds '
                                                   + '%3.0f' % self.q_run_threshold_caution
                                                   + '%;', 2, 11])
                    if status_switch < 1:
                        status_switch = 1

                # Generate message for total_invalid Q
                if np.any(self.w_vel['q_total_warning'][:, filter_idx]):
                    self.w_vel['messages'].append(['WT-' + prefix[prefix_idx]
                                                   + 'Int. Q for invalid cells and ensembles in a transect exceeds '
                                                   + '%3.0f' % self.q_total_threshold_warning
                                                   + '%;', 1, 11])
                    status_switch = 2
                elif np.any(self.w_vel['q_total_caution'][:, filter_idx]):
                    self.w_vel['messages'].append(['wt-' + prefix[prefix_idx]
                                                   + 'Int. Q for invalid cells and ensembles in a transect exceeds '
                                                   + '%3.0f' % self.q_total_threshold_caution
                                                   + '%;', 2, 11])
                    if status_switch < 1:
                        status_switch = 1

            # Generate message for all invalid
            if np.any(self.w_vel['all_invalid']):
                self.w_vel['messages'].append(['WT-' + prefix[0] + 'There are no valid data for one or more transects.',
                                               1, 11])
                status_switch = 2

            # Set status
            self.w_vel['status'] = 'good'
            if status_switch == 2:
                self.w_vel['status'] = 'warning'
            elif status_switch == 1:
                self.w_vel['status'] = 'caution'
        else:
            self.w_vel['status'] = 'inactive'

    def extrapolation_qa(self, meas):
        """Apply quality checks to extrapolation methods

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.extrapolation['messages'] = []

        checked = []
        discharges = []
        for n, transect in enumerate(meas.transects):
            checked.append(transect.checked)
            if transect.checked:
                discharges.append(meas.discharge[n])

        if any(checked):
            self.extrapolation['status'] = 'good'
            extrap_uncertainty = Uncertainty.uncertainty_extrapolation(meas, discharges)

            if np.abs(extrap_uncertainty) > 2:
                self.extrapolation['messages'].append(['Extrapolation: The extrapolation uncertainty is more than '
                                                       + '2 percent;', 2, 12])
                self.extrapolation['messages'].append(['    Carefully review the extrapolation;', 2, 12])
                self.extrapolation['status'] = 'caution'
        else:
            self.extrapolation['status'] = 'inactive'

    def edges_qa(self, meas):
        """Apply quality checks to edge estimates

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        # Initialize variables
        self.edges['messages'] = []
        checked = []
        left_q = []
        right_q = []
        total_q = []
        edge_dist_left = []
        edge_dist_right = []
        dist_moved_left = []
        dist_moved_right = []
        dist_made_good = []
        left_type = []
        right_type = []
        transect_idx = []

        for n, transect in enumerate(meas.transects):
            checked.append(transect.checked)

            if transect.checked:
                left_q.append(meas.discharge[n].left)
                right_q.append(meas.discharge[n].right)
                total_q.append(meas.discharge[n].total)
                dmr, dml, dmg = QAData.edge_distance_moved(transect)
                dist_moved_right.append(dmr)
                dist_moved_left.append(dml)
                dist_made_good.append(dmg)
                edge_dist_left.append(transect.edges.left.distance_m)
                edge_dist_right.append(transect.edges.right.distance_m)
                left_type.append(transect.edges.left.type)
                right_type.append(transect.edges.right.type)
                transect_idx.append(n)

        if any(checked):
            # Set default status to good
            self.edges['status'] = 'good'

            mean_total_q = np.nanmean(total_q)

            # Check left edge q > 5%
            self.edges['left_q'] = 0

            left_q_percent = (np.nanmean(left_q) / mean_total_q) * 100
            temp_idx = np.where(left_q / mean_total_q > 0.05)[0]
            if len(temp_idx) > 0:
                self.edges['left_q_idx'] = np.array(transect_idx)[temp_idx]
            else:
                self.edges['left_q_idx'] = []
            if np.abs(left_q_percent) > 5:
                self.edges['status'] = 'caution'
                self.edges['messages'].append(['Edges: Left edge Q is greater than 5%;', 1, 13])
                self.edges['left_q'] = 1
            elif len(self.edges['left_q_idx']) > 0:
                self.edges['status'] = 'caution'
                self.edges['messages'].append(
                    ['Edges: One or more transects have a left edge Q greater than 5%;', 1, 13])
                self.edges['left_q'] = 1

            # Check right edge q > 5%
            self.edges['right_q'] = 0
            right_q_percent = (np.nanmean(right_q) / mean_total_q) * 100
            temp_idx = np.where(right_q / mean_total_q > 0.05)[0]
            if len(temp_idx) > 0:
                self.edges['right_q_idx'] = np.array(transect_idx)[temp_idx]
            else:
                self.edges['right_q_idx'] = []
            if np.abs(right_q_percent) > 5:
                self.edges['status'] = 'caution'
                self.edges['messages'].append(['Edges: Right edge Q is greater than 5%;', 1, 13])
                self.edges['right_q'] = 1
            elif len(self.edges['right_q_idx']) > 0:
                self.edges['status'] = 'caution'
                self.edges['messages'].append(
                    ['Edges: One or more transects have a right edge Q greater than 5%;', 1, 13])
                self.edges['right_q'] = 1

            # Check for consistent sign
            q_positive = []
            self.edges['left_sign'] = 0
            for q in left_q:
                if q >= 0:
                    q_positive.append(True)
                else:
                    q_positive.append(False)
            if len(np.unique(q_positive)) > 1 and left_q_percent > 0.5:
                self.edges['status'] = 'caution'
                self.edges['messages'].append(['Edges: Sign of left edge Q is not consistent;', 2, 13])
                self.edges['left_sign'] = 1

            q_positive = []
            self.edges['right_sign'] = 0
            for q in right_q:
                if q >= 0:
                    q_positive.append(True)
                else:
                    q_positive.append(False)
            if len(np.unique(q_positive)) > 1 and right_q_percent > 0.5:
                self.edges['status'] = 'caution'
                self.edges['messages'].append(['Edges: Sign of right edge Q is not consistent;', 2, 13])
                self.edges['right_sign'] = 1

            # Check distance moved
            dmg_5_percent = 0.05 * np.nanmean(dist_made_good)
            avg_right_edge_dist = np.nanmean(edge_dist_right)
            right_threshold = np.nanmin([dmg_5_percent, avg_right_edge_dist])
            temp_idx = np.where(dist_moved_right > right_threshold)[0]
            if len(temp_idx) > 0:
                self.edges['right_dist_moved_idx'] = np.array(transect_idx)[temp_idx]
                self.edges['status'] = 'caution'
                self.edges['messages'].append(['Edges: Excessive boat movement in right edge ensembles;', 2, 13])
            else:
                self.edges['right_dist_moved_idx'] = []

            avg_left_edge_dist = np.nanmean(edge_dist_left)
            left_threshold = np.nanmin([dmg_5_percent, avg_left_edge_dist])
            temp_idx = np.where(dist_moved_left > left_threshold)[0]
            if len(temp_idx) > 0:
                self.edges['left_dist_moved_idx'] = np.array(transect_idx)[temp_idx]
                self.edges['status'] = 'caution'
                self.edges['messages'].append(['Edges: Excessive boat movement in left edge ensembles;', 2, 13])
            else:
                self.edges['left_dist_moved_idx'] = []

            # Check for edge ensembles marked invalid due to excluded distance
            self.edges['invalid_transect_left_idx'] = []
            self.edges['invalid_transect_right_idx'] = []
            for n, transect in enumerate(meas.transects):
                if transect.checked:
                    ens_invalid = np.nansum(transect.w_vel.valid_data[0, :, :], 0) > 0
                    ens_cells_above_sl = np.nansum(transect.w_vel.cells_above_sl, 0) > 0
                    ens_invalid = np.logical_not(np.logical_and(ens_invalid, ens_cells_above_sl))
                    if np.any(ens_invalid):
                        if transect.start_edge == 'Left':
                            invalid_left = ens_invalid[0:int(transect.edges.left.number_ensembles)]
                            invalid_right = ens_invalid[-int(transect.edges.right.number_ensembles):]
                        else:
                            invalid_right = ens_invalid[0:int(transect.edges.right.number_ensembles)]
                            invalid_left = ens_invalid[-int(transect.edges.left.number_ensembles):]
                        if len(invalid_left) > 0:
                            left_invalid_percent = sum(invalid_left) / len(invalid_left)
                        else:
                            left_invalid_percent = 0
                        if len(invalid_right) > 0:
                            right_invalid_percent = sum(invalid_right) / len(invalid_right)
                        else:
                            right_invalid_percent = 0
                        max_invalid_percent = max([left_invalid_percent, right_invalid_percent]) * 100
                        if max_invalid_percent > 25:
                            self.edges['status'] = 'caution'
                            if np.any(invalid_left):
                                self.edges['invalid_transect_left_idx'].append(n)
                            if np.any(invalid_right):
                                self.edges['invalid_transect_right_idx'].append(n)

            if len(self.edges['invalid_transect_left_idx']) > 0 or len(self.edges['invalid_transect_right_idx']) > 0:
                self.edges['messages'].append(['Edges: The percent of invalid ensembles exceeds 25% in' +
                                               ' one or more transects.', 2, 13])

            # Check edges for zero discharge
            self.edges['left_zero'] = 0
            temp_idx = np.where(np.round(left_q, 4) == 0)[0]
            if len(temp_idx) > 0:
                self.edges['left_zero_idx'] = np.array(transect_idx)[temp_idx]
                self.edges['status'] = 'warning'
                self.edges['messages'].append(['EDGES: Left edge has zero Q;', 1, 13])
                self.edges['left_zero'] = 2
            else:
                self.edges['left_zero_idx'] = []

            self.edges['right_zero'] = 0
            temp_idx = np.where(np.round(right_q, 4) == 0)[0]
            if len(temp_idx) > 0:
                self.edges['right_zero_idx'] = np.array(transect_idx)[temp_idx]
                self.edges['status'] = 'warning'
                self.edges['messages'].append(['EDGES: Right edge has zero Q;', 1, 13])
                self.edges['right_zero'] = 2
            else:
                self.edges['right_zero_idx'] = []

            # Check consistent edge type
            self.edges['left_type'] = 0
            if len(np.unique(left_type)) > 1:
                self.edges['status'] = 'warning'
                self.edges['messages'].append(['EDGES: Left edge type is not consistent;', 1, 13])
                self.edges['left_type'] = 2

            self.edges['right_type'] = 0
            if len(np.unique(right_type)) > 1:
                self.edges['status'] = 'warning'
                self.edges['messages'].append(['EDGES: Right edge type is not consistent;', 1, 13])
                self.edges['right_type'] = 2
        else:
            self.edges['status'] = 'inactive'

    @staticmethod
    def invalid_qa(valid, discharge):
        """Computes the total invalid discharge in ensembles that have invalid data. The function also computes
        the maximum run or cluster of ensembles with the maximum interpolated discharge.

        Parameters
        ----------
        valid: np.array(bool)
            Array identifying valid and invalid ensembles.
        discharge: QComp
            Object of class QComp

        Returns
        -------
        q_invalid_total: float
            Total interpolated discharge in invalid ensembles
        q_invalid_max_run: float
            Maximum interpolated discharge in a run or cluster of invalid ensembles
        ens_invalid: int
            Total number of invalid ensembles
        """

        # Create bool for invalid data
        invalid = np.logical_not(valid)
        q_invalid_total = np.nansum(discharge.middle_ens[invalid]) + np.nansum(discharge.top_ens[invalid]) \
            + np.nansum(discharge.bottom_ens[invalid])

        # Compute total number of invalid ensembles
        ens_invalid = np.sum(invalid)

        # Compute the indices of where changes occur

        valid_int = np.insert(valid.astype(int), 0, -1)
        valid_int = np.append(valid_int, -1)
        valid_run = np.where(np.diff(valid_int) != 0)[0]
        run_length = np.diff(valid_run)
        run_length0 = run_length[(valid[0] == 1)::2]

        n_runs = len(run_length0)

        if valid[0]:
            n_start = 1
        else:
            n_start = 0

        n_end = len(valid_run) - 1

        if n_runs > 1:
            m = 0
            q_invalid_run = []
            for n in range(n_start, n_end, 2):
                m += 1
                idx_start = valid_run[n]
                idx_end = valid_run[n + 1]
                q_invalid_run.append(np.nansum(discharge.middle_ens[idx_start:idx_end])
                                     + np.nansum(discharge.top_ens[idx_start:idx_end])
                                     + np.nansum(discharge.bottom_ens[idx_start:idx_end]))

            # Determine the maximum discharge in a single run
            q_invalid_max_run = np.nanmax(np.abs(q_invalid_run))

        else:
            q_invalid_max_run = 0.0

        return q_invalid_total, q_invalid_max_run, ens_invalid

    @staticmethod
    def edge_distance_moved(transect):
        """Computes the boat movement during edge ensemble collection.

        Parameters
        ----------
        transect: Transect
            Object of class Transect

        Returns
        -------
        right_dist_moved: float
            Distance in m moved during collection of right edge samples
        left_dist_moved: float
            Distance in m moved during collection of left edge samples
        dmg: float
            Distance made good for the entire transect
        """

        boat_selected = getattr(transect.boat_vel, transect.boat_vel.selected)
        ens_duration = transect.date_time.ens_duration_sec

        # Get boat velocities
        if boat_selected is not None:
            u_processed = boat_selected.u_processed_mps
            v_processed = boat_selected.v_processed_mps
        else:
            u_processed = np.tile(np.nan, transect.boat_vel.bt_vel.u_processed_mps.shape)
            v_processed = np.tile(np.nan, transect.boat_vel.bt_vel.v_processed_mps.shape)

        # Compute boat coordinates
        x_processed = np.nancumsum(u_processed * ens_duration)
        y_processed = np.nancumsum(v_processed * ens_duration)
        dmg = (x_processed[-1] ** 2 + y_processed[-1] ** 2) ** 0.5

        # Compute left distance moved
        # TODO should be a dist moved function
        left_edge_idx = QComp.edge_ensembles('left', transect)
        if len(left_edge_idx) > 0:
            boat_x = x_processed[left_edge_idx[-1]] - x_processed[left_edge_idx[0]]
            boat_y = y_processed[left_edge_idx[-1]] - y_processed[left_edge_idx[0]]
            left_dist_moved = (boat_x ** 2 + boat_y ** 2) ** 0.5
        else:
            left_dist_moved = np.nan

        # Compute right distance moved
        right_edge_idx = QComp.edge_ensembles('right', transect)
        if len(right_edge_idx) > 0:
            boat_x = x_processed[right_edge_idx[-1]] - x_processed[right_edge_idx[0]]
            boat_y = y_processed[right_edge_idx[-1]] - y_processed[right_edge_idx[0]]
            right_dist_moved = (boat_x ** 2 + boat_y ** 2) ** 0.5
        else:
            right_dist_moved = np.nan

        return right_dist_moved, left_dist_moved, dmg
    
    # check for user changes
    def check_bt_setting(self, meas):
        """Checks the bt settings to see if they are still on the default
                        settings.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.settings_dict['tab_bt'] = 'Default'

        s = meas.current_settings()
        d = meas.qrev_default_settings()

        if s['BTbeamFilter'] != d['BTbeamFilter']:
            self.bt_vel['messages'].append(['BT: User modified default beam setting.', 3, 8])
            self.settings_dict['tab_bt'] = 'Custom'

        if s['BTdFilter'] != d['BTdFilter']:
            self.bt_vel['messages'].append(['BT: User modified default error velocity filter.', 3, 8])
            self.settings_dict['tab_bt'] = 'Custom'

        if s['BTwFilter'] != d['BTwFilter']:
            self.bt_vel['messages'].append(['BT: User modified default vertical velocity filter.', 3, 8])
            self.settings_dict['tab_bt'] = 'Custom'

        if s['BTsmoothFilter'] != d['BTsmoothFilter']:
            self.bt_vel['messages'].append(['BT: User modified default smooth filter.', 3, 8])
            self.settings_dict['tab_bt'] = 'Custom'

    def check_wt_settings(self, meas):
        """Checks the wt settings to see if they are still on the default
                settings.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.settings_dict['tab_wt'] = 'Default'

        s = meas.current_settings()
        d = meas.qrev_default_settings()

        if round(s['WTExcludedDistance'], 2) != round(d['WTExcludedDistance'], 2):
            self.w_vel['messages'].append(['WT: User modified excluded distance.', 3, 11])
            self.settings_dict['tab_wt'] = 'Custom'

        if s['WTbeamFilter'] != d['WTbeamFilter']:
            self.w_vel['messages'].append(['WT: User modified default beam setting.', 3, 11])
            self.settings_dict['tab_wt'] = 'Custom'

        if s['WTdFilter'] != d['WTdFilter']:
            self.w_vel['messages'].append(['WT: User modified default error velocity filter.', 3, 11])
            self.settings_dict['tab_wt'] = 'Custom'

        if s['WTwFilter'] != d['WTwFilter']:
            self.w_vel['messages'].append(['WT: User modified default vertical velocity filter.', 3, 11])
            self.settings_dict['tab_wt'] = 'Custom'

        if s['WTsnrFilter'] != d['WTsnrFilter']:
            self.w_vel['messages'].append(['WT: User modified default SNR filter.', 3, 11])
            self.settings_dict['tab_wt'] = 'Custom'

    def check_extrap_settings(self, meas):
        """Checks the extrap to see if they are still on the default
        settings.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.settings_dict['tab_extrap'] = 'Default'

        # Check fit parameters
        if meas.extrap_fit.sel_fit[0].fit_method != 'Automatic':
            self.settings_dict['tab_extrap'] = 'Custom'
            self.extrapolation['messages'].append(['Extrapolation: User modified default automatic setting.', 3, 12])

        # Check data parameters
        if meas.extrap_fit.sel_fit[-1].data_type.lower() != 'q':
            self.settings_dict['tab_extrap'] = 'Custom'
            self.extrapolation['messages'].append(['Extrapolation: User modified data type ', 3, 12])

        if meas.extrap_fit.threshold != 20:
            self.settings_dict['tab_extrap'] = 'Custom'
            self.extrapolation['messages'].append(['Extrapolation: User modified default threshold.', 3, 12])

        if meas.extrap_fit.subsection[0] != 0 or meas.extrap_fit.subsection[1] != 100:
            self.settings_dict['tab_extrap'] = 'Custom'
            self.extrapolation['messages'].append(['Extrapolation: User modified subsectioning', 3, 12])

    def check_tempsal_settings(self, meas):
        """Checks the temp and salinity settings to see if they are still on
        the default settings.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.settings_dict['tab_tempsal'] = 'Default'

        t_source_change = False
        s_sound_change = False
        t_user_change = False
        t_adcp_change = False

        if not all(np.isnan([meas.ext_temp_chk['user'], meas.ext_temp_chk['user_orig']])):
            if meas.ext_temp_chk['user'] != meas.ext_temp_chk['user_orig']:
                t_user_change = True

        if not all(np.isnan([meas.ext_temp_chk['adcp'], meas.ext_temp_chk['adcp_orig']])):
            if meas.ext_temp_chk['adcp'] != meas.ext_temp_chk['adcp_orig']:
                t_adcp_change = True

        # Check each checked transect
        for idx in meas.checked_transect_idx:
            transect = meas.transects[idx]

            # Temperature source
            if transect.sensors.temperature_deg_c.selected != 'internal':
                t_source_change = True

            # Speed of Sound
            if transect.sensors.speed_of_sound_mps.selected != 'internal':
                s_sound_change = True

        # Report condition and messages
        if any([t_source_change, s_sound_change, t_adcp_change, t_user_change]):
            self.settings_dict['tab_tempsal'] = 'Custom'
            
            if t_source_change:
                self.temperature['messages'].append(['Temperature: User modified temperature source.', 3, 5])

            if s_sound_change:
                self.temperature['messages'].append(['Temperature: User modified speed of sound source.', 3, 5])

            if t_user_change:
                self.temperature['messages'].append(['Temperature: User modified independent temperature.', 3, 5])

            if t_adcp_change:
                self.temperature['messages'].append(['Temperature: User modified ADCP temperature.', 3, 5])
         
    def check_gps_settings(self, meas):
        """Checks the gps settings to see if they are still on the default
        settings.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        gps = False
        self.settings_dict['tab_gps'] = 'Default'

        # Check for transects with gga or vtg data
        for idx in meas.checked_transect_idx:
            transect = meas.transects[idx]
            if transect.boat_vel.gga_vel is not None or transect.boat_vel.gga_vel is not None:
                gps = True
                break

        # If gga or vtg data exist check settings
        if gps:

            s = meas.current_settings()
            d = meas.qrev_default_settings()

            if s['ggaDiffQualFilter'] != d['ggaDiffQualFilter']:
                self.gga_vel['messages'].append(['GPS: User modified default quality setting.', 3, 8])
                self.settings_dict['tab_gps'] = 'Custom'

            if s['ggaAltitudeFilter'] != d['ggaAltitudeFilter']:
                self.gga_vel['messages'].append(['GPS: User modified default altitude filter.', 3, 8])
                self.settings_dict['tab_gps'] = 'Custom'

            if s['GPSHDOPFilter'] != d['GPSHDOPFilter']:
                self.gga_vel['messages'].append(['GPS: User modified default HDOP filter.', 3, 8])
                self.settings_dict['tab_gps'] = 'Custom'

            if s['GPSSmoothFilter'] != d['GPSSmoothFilter']:
                self.gga_vel['messages'].append(['GPS: User modified default smooth filter.', 3, 8])
                self.settings_dict['tab_gps'] = 'Custom'

    def check_depth_settings(self, meas):
        """Checks the depth settings to see if they are still on the default
                settings.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.settings_dict['tab_depth'] = 'Default'

        s = meas.current_settings()
        d = meas.qrev_default_settings()

        if s['depthReference'] != d['depthReference']:
            self.depths['messages'].append(['Depths: User modified '
                                            'depth reference.', 3, 10])
            self.settings_dict['tab_depth'] = 'Custom'

        if s['depthComposite'] != d['depthComposite']:
            self.depths['messages'].append(['Depths: User modified '
                                            'depth reference.', 3, 10])
            self.settings_dict['tab_depth'] = 'Custom'

        if s['depthAvgMethod'] != d['depthAvgMethod']:
            self.depths['messages'].append(['Depths: User modified '
                                            'averaging method.', 3, 10])
            self.settings_dict['tab_depth'] = 'Custom'

        if s['depthFilterType'] != d['depthFilterType']:
            self.depths['messages'].append(['Depths: User modified '
                                            'filter type.', 3, 10])
            self.settings_dict['tab_depth'] = 'Custom'

        for idx in meas.checked_transect_idx:
            transect = meas.transects[idx]
            if transect.depths.bt_depths.draft_orig_m != transect.depths.bt_depths.draft_use_m:
                self.depths['messages'].append(['Depths: User modified '
                                                'draft.', 3, 10])
                self.settings_dict['tab_depth'] = 'Custom'
                break

    def check_edge_settings(self, meas):
        """Checks the edge settings to see if they are still on the original
                settings.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        start_edge_change = False
        left_edge_type_change = False
        left_edge_dist_change = False
        left_edge_ens_change = False
        left_edge_q_change = False
        left_edge_coef_change = False
        right_edge_type_change = False
        right_edge_dist_change = False
        right_edge_ens_change = False
        right_edge_q_change = False
        right_edge_coef_change = False

        for idx in meas.checked_transect_idx:
            transect = meas.transects[idx]

            if transect.start_edge != transect.orig_start_edge:
                start_edge_change = True

            if transect.edges.left.type != transect.edges.left.orig_type:
                left_edge_type_change = True

            if transect.edges.left.distance_m != transect.edges.left.orig_distance_m:
                left_edge_dist_change = True

            if transect.edges.left.number_ensembles != transect.edges.left.orig_number_ensembles:
                left_edge_ens_change = True

            if transect.edges.left.user_discharge_cms != transect.edges.left.orig_user_discharge_cms:
                left_edge_q_change = True

            if transect.edges.left.cust_coef != transect.edges.left.orig_cust_coef:
                left_edge_coef_change = True

            if transect.edges.right.type != transect.edges.right.orig_type:
                right_edge_type_change = True

            if transect.edges.right.distance_m != transect.edges.right.orig_distance_m:
                right_edge_dist_change = True

            if transect.edges.right.number_ensembles != transect.edges.right.orig_number_ensembles:
                right_edge_ens_change = True


            if transect.edges.right.user_discharge_cms != transect.edges.right.orig_user_discharge_cms:
                right_edge_q_change = True

            if transect.edges.right.cust_coef != transect.edges.right.orig_cust_coef:
                right_edge_coef_change = True

        if any([start_edge_change, left_edge_type_change, left_edge_dist_change, left_edge_ens_change,
                left_edge_q_change, left_edge_coef_change, right_edge_type_change, right_edge_dist_change,
                right_edge_ens_change, right_edge_q_change, right_edge_coef_change]):
            self.settings_dict['tab_edges'] = 'Custom'

            if start_edge_change:
                self.edges['messages'].append(['Edges: User modified start edge.', 3, 10])
            if left_edge_type_change:
                self.edges['messages'].append(['Edges: User modified left edge type.', 3, 10])
            if left_edge_dist_change:
                self.edges['messages'].append(['Edges: User modified left edge distance.', 3, 10])
            if left_edge_ens_change:
                self.edges['messages'].append(['Edges: User modified left number of ensembles.', 3, 10])
            if left_edge_q_change:
                self.edges['messages'].append(['Edges: User modified left user discharge.', 3, 10])
            if left_edge_coef_change:
                self.edges['messages'].append(['Edges: User modified left custom coefficient.', 3, 10])
            if right_edge_type_change:
                self.edges['messages'].append(['Edges: User modified right edge type.', 3, 10])
            if right_edge_dist_change:
                self.edges['messages'].append(['Edges: User modified right edge distance.', 3, 10])
            if right_edge_ens_change:
                self.edges['messages'].append(['Edges: User modified right number of ensembles.', 3, 10])
            if right_edge_q_change:
                self.edges['messages'].append(['Edges: User modified right user discharge.', 3, 10])
            if right_edge_coef_change:
                self.edges['messages'].append(['Edges: User modified right custom coefficient.', 3, 10])
        else:
            self.settings_dict['tab_edges'] = 'Default'

    def check_mbt_settings(self, meas):
        """Checks the mbt settings to see if they are still on the original
                settings.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        # if there are mb tests check for user changes
        if len(meas.mb_tests) >= 1:
            mbt = meas.mb_tests

            # mb_present = []
            mb_user_valid = []
            # mb_test_quality = []
            mb_used = []

            auto = copy.deepcopy(mbt)
            auto = MovingBedTests.auto_use_2_correct(auto)

            for n in range(len(mbt)):

                if mbt[n].user_valid:
                    mb_user_valid.append(False)
                else:
                    mb_user_valid.append(True)

                if mbt[n].use_2_correct != auto[n].use_2_correct:
                    mb_used.append(True)
                else:
                    mb_used.append(False)

            self.settings_dict['tab_mbt'] = 'Default'
            if any(mb_user_valid):
                self.settings_dict['tab_mbt'] = 'Custom'
                self.movingbed['messages'].append(['Moving-Bed Test: '
                                                   'User modified '
                                                   'valid test settings.', 3, 6])
            if any(mb_used):
                self.settings_dict['tab_mbt'] = 'Custom'
                self.movingbed['messages'].append(['Moving-Bed Test: '
                                                   'User modified '
                                                   'use to correct settings.', 3, 6])

    def check_compass_settings(self, meas):
        """Checks the compass settings for changes.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.settings_dict['tab_compass'] = 'Default'

        magvar_change = False
        align_change = False

        # Check each checked transect
        for idx in meas.checked_transect_idx:
            transect = meas.transects[idx]

            # Magvar
            if transect.sensors.heading_deg.internal.mag_var_deg != \
                    transect.sensors.heading_deg.internal.mag_var_orig_deg:
                magvar_change = True

            # Heading offset
            if transect.sensors.heading_deg.external is not None:
                if transect.sensors.heading_deg.external.align_correction_deg != \
                        transect.sensors.heading_deg.external.align_correction_orig_deg:
                    align_change = True

        # Report condition and messages
        if any([magvar_change, align_change]):
            self.settings_dict['tab_compass'] = 'Custom'

            if magvar_change:
                self.compass['messages'].append(['Compass: User modified magnetic variation.', 3, 4])

            if align_change:
                self.compass['messages'].append(['Compass: User modified heading offset.', 3, 4])
