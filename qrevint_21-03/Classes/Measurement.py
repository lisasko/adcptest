import os
import datetime
import numpy as np
import xml.etree.ElementTree as ETree
from xml.dom.minidom import parseString
import warnings
from Classes.MMT_TRDI import MMTtrdi
from Classes.TransectData import TransectData, allocate_transects
from Classes.PreMeasurement import PreMeasurement
from Classes.MovingBedTests import MovingBedTests
from Classes.QComp import QComp
from Classes.MatSonTek import MatSonTek
from Classes.ComputeExtrap import ComputeExtrap
from Classes.ExtrapQSensitivity import ExtrapQSensitivity
from Classes.Uncertainty import Uncertainty
from Classes.QAData import QAData
from Classes.BoatStructure import BoatStructure
from Classes.NormData import  NormData
from Classes.Oursin import Oursin
# from Classes.Oursin_orig import Oursin_orig
from MiscLibs.common_functions import cart2pol, pol2cart, rad2azdeg, nans, azdeg2rad
# from profilehooks import profile

class Measurement(object):
    """Class to hold all measurement details.

    Attributes
    ----------
    station_name: str
        Station name
    station_number: str
        Station number
    transects: list
        List of transect objects of TransectData
    mb_tests: list
        List of moving-bed test objects of MovingBedTests
    system_tst: list
        List of system test objects of PreMeasurement
    compass_cal: list
        List of compass calibration objects of PreMeasurement
    compass_eval: list
        List of compass evaluation objects of PreMeasurement
    extrap_fit: ComputeExtrap
        Object of ComputeExtrap
    processing: str
        Type of processing, default QRev
    discharge: list
        List of discharge objects of QComp
    uncertainty: Uncertainty
        Object of Uncertainty
    initial_settings: dict
        Dictionary of all initial processing settings
    qa: QAData
        Object of QAData
    user_rating: str
        Optional user rating
    comments: list
        List of all user supplied comments
    ext_temp_chk: dict
        Dictionary of external temperature readings
    use_weighted: bool
        Indicates the setting for use_weighted to be used for reprocessing
    """

    # @profile
    def __init__(self, in_file, source, proc_type='QRev', checked=False, run_oursin=False, use_weighted=False):
        """Initialize instance variables and initiate processing of measurement
        data.

        Parameters
        ----------
        in_file: str or list or dict
            String containing fullname of mmt file for TRDI data, dict for
            QRev data, or list of files for SonTek
        source: str
            Source of data. TRDI, SonTek, QRev
        proc_type: str
            Type of processing. QRev, None, Original
        checked: bool
            Boolean to determine if only checked transects should be load for
            TRDI data.
        run_oursin: bool
            Determines if the Oursin uncertainty model should be run
        use_weighted: bool
            Specifies if discharge weighted medians are used for extrapolation
        """

        self.run_oursin = run_oursin
        self.station_name = None
        self.station_number = None
        self.transects = []
        self.mb_tests = []
        self.system_tst = []
        self.compass_cal = []
        self.compass_eval = []
        self.extrap_fit = None
        self.processing = None
        self.discharge = []
        self.uncertainty = None
        self.initial_settings = None
        self.qa = None
        self.user_rating = 'Not Rated'
        self.comments = []
        self.ext_temp_chk = {'user': np.nan, 'units': 'C', 'adcp': np.nan, 'user_orig': np.nan, 'adcp_orig': np.nan}
        self.checked_transect_idx = []
        self.oursin = None
        self.use_weighted = use_weighted

        # Load data from selected source
        if source == 'QRev':
            self.load_qrev_mat(mat_data=in_file)
            if proc_type == 'QRev':
                # Apply QRev default settings
                # settings = self.qrev_default_settings()
                settings = self.current_settings()
                settings['WTEnsInterpolation'] = 'abba'
                settings['WTCellInterpolation'] = 'abba'
                settings['Processing'] = 'QRev'
                self.apply_settings(settings)

        else:
            if source == 'TRDI':
                self.load_trdi(in_file, checked=checked)

            elif source == 'SonTek':
                self.load_sontek(in_file)

            elif source == 'Nortek':
                self.load_sontek(in_file)

            # Process data
            if len(self.transects) > 0:

                # Save initial settings
                self.initial_settings = self.current_settings()

                # Process moving-bed tests
                if len(self.mb_tests) > 0:
                    # Get navigation reference
                    select = self.initial_settings['NavRef']
                    ref = None
                    if select == 'bt_vel':
                        ref = 'BT'
                    elif select == 'gga_vel':
                        ref = 'GGA'
                    elif select == 'vtg_vel':
                        ref = 'VTG'
                    self.mb_tests = MovingBedTests.auto_use_2_correct(
                        moving_bed_tests=self.mb_tests, boat_ref=ref)

                # Set processing type
                if proc_type == 'QRev':
                    # Apply QRev default settings
                    settings = self.qrev_default_settings(check_user_excluded_dist=True, use_weighted=use_weighted)
                    settings['Processing'] = 'QRev'
                    self.apply_settings(settings)

                elif proc_type == 'None':
                    # Processing with no filters and interpolation
                    settings = self.no_filter_interp_settings(self)
                    settings['Processing'] = 'None'
                    self.apply_settings(settings)

                elif proc_type == 'Original':
                    # Processing for original settings
                    # from manufacturer software
                    for transect in self.transects:
                        q = QComp()
                        q.populate_data(data_in=transect,
                                        moving_bed_data=self.mb_tests)
                        self.discharge.append(q)
                self.uncertainty = Uncertainty()
                self.uncertainty.compute_uncertainty(self)

                self.qa = QAData(self)
                if self.run_oursin:
                    self.oursin = Oursin()
                    self.oursin.compute_oursin(self)
                #
                # self.oursin_orig = Oursin_orig()
                # self.oursin_orig.compute_oursin(self)

    def load_trdi(self, mmt_file, transect_type='Q', checked=False):
        """Method to load TRDI data.

        Parameters
        ----------
        mmt_file: str
            Full pathname to mmt file.
        transect_type: str
            Type of data (Q: discharge, MB: moving-bed test
        checked: bool
            Determines if all files are loaded (False) or only checked (True)
        """

        # Read mmt file
        mmt = MMTtrdi(mmt_file)

        # Get properties if they exist, otherwise set them as blank strings
        self.station_name = str(mmt.site_info['Name'])
        self.station_number = str(mmt.site_info['Number'])

        # Initialize processing variable
        self.processing = 'WR2'

        # Create transect objects for  TRDI data
        # TODO refactor allocate_transects
        self.transects = allocate_transects(mmt=mmt,
                                            transect_type=transect_type,
                                            checked=checked)

        self.checked_transect_idx = self.checked_transects(self)

        # Create object for pre-measurement tests
        if isinstance(mmt.qaqc, dict) or isinstance(mmt.mbt_transects, list):
            self.qaqc_trdi(mmt)
        
        # Save comments from mmt file in comments
        self.comments.append('MMT Remarks: ' + mmt.site_info['Remarks'])

        for t in range(len(self.transects)):
            notes = getattr(mmt.transects[t], 'Notes')
            for note in notes:
                note_text = ' File: ' + note['NoteFileNo'] + ' ' \
                            + note['NoteDate'] + ': ' + note['NoteText']
                self.comments.append(note_text)
                
        # Get external temperature
        if type(mmt.site_info['Water_Temperature']) is float:
            self.ext_temp_chk['user'] = mmt.site_info['Water_Temperature']
            self.ext_temp_chk['units'] = 'C'
            self.ext_temp_chk['user_orig'] = mmt.site_info['Water_Temperature']

        # Initialize thresholds settings dictionary
        threshold_settings = dict()
        threshold_settings['wt_settings'] = {}
        threshold_settings['bt_settings'] = {}
        threshold_settings['depth_settings'] = {}

		# Select reference transect use first checked or if none then first transect
        if len(self.checked_transect_idx) > 0:
            ref_transect = self.checked_transect_idx[0]
        else:
            ref_transect = 0

        # Water track filter threshold settings
        threshold_settings['wt_settings']['beam'] = \
            self.set_num_beam_wt_threshold_trdi(mmt.transects[ref_transect])
        threshold_settings['wt_settings']['difference'] = 'Manual'
        threshold_settings['wt_settings']['difference_threshold'] = \
            mmt.transects[ref_transect].active_config['Proc_WT_Error_Velocity_Threshold']
        threshold_settings['wt_settings']['vertical'] = 'Manual'
        threshold_settings['wt_settings']['vertical_threshold'] = \
            mmt.transects[ref_transect].active_config['Proc_WT_Up_Vel_Threshold']

        # Bottom track filter threshold settings
        threshold_settings['bt_settings']['beam'] = \
            self.set_num_beam_bt_threshold_trdi(mmt.transects[ref_transect])
        threshold_settings['bt_settings']['difference'] = 'Manual'
        threshold_settings['bt_settings']['difference_threshold'] = \
            mmt.transects[ref_transect].active_config['Proc_BT_Error_Vel_Threshold']
        threshold_settings['bt_settings']['vertical'] = 'Manual'
        threshold_settings['bt_settings']['vertical_threshold'] = \
            mmt.transects[ref_transect].active_config['Proc_BT_Up_Vel_Threshold']

        # Depth filter and averaging settings
        threshold_settings['depth_settings']['depth_weighting'] = \
            self.set_depth_weighting_trdi(mmt.transects[ref_transect])
        threshold_settings['depth_settings']['depth_valid_method'] = 'TRDI'
        threshold_settings['depth_settings']['depth_screening'] = \
            self.set_depth_screening_trdi(mmt.transects[ref_transect])

        # Determine reference used in WR2 if available
        reference = 'BT'
        if 'Reference' in mmt.site_info.keys():
            reference = mmt.site_info['Reference']
            if reference == 'BT':
                target = 'bt_vel'
            elif reference == 'GGA':
                target = 'gga_vel'
            elif reference == 'VTG':
                target = 'vtg_vel'
            else:
                target = 'bt_vel'

            for transect in self.transects:
                if getattr(transect.boat_vel, target) is None:
                    reference = 'BT'

        # Convert to earth coordinates
        for transect_idx, transect in enumerate(self.transects):
            # Convert to earth coordinates
            transect.change_coord_sys(new_coord_sys='Earth')

            # Set navigation reference
            transect.change_nav_reference(update=False, new_nav_ref=reference)

            # Apply WR2 thresholds
            self.thresholds_trdi(transect, threshold_settings)

            # Apply boat interpolations
            transect.boat_interpolations(update=False,
                                         target='BT',
                                         method='None')
            if transect.gps is not None:
                transect.boat_interpolations(update=False,
                                             target='GPS',
                                             method='HoldLast')

            # Update water data for changes in boat velocity
            transect.update_water()

            # Filter water data
            transect.w_vel.apply_filter(transect=transect, wt_depth=True)

            # Interpolate water data
            transect.w_vel.apply_interpolation(transect=transect,
                                               ens_interp='None',
                                               cells_interp='None')

            # Apply speed of sound computations as required
            mmt_sos_method = mmt.transects[transect_idx].active_config[
                'Proc_Speed_of_Sound_Correction']

            # Speed of sound computed based on user supplied values
            if mmt_sos_method == 1:
                transect.change_sos(parameter='salinity')
            elif mmt_sos_method == 2:
                # Speed of sound set by user
                speed = mmt.transects[transect_idx].active_config[
                    'Proc_Fixed_Speed_Of_Sound']
                transect.change_sos(parameter='sosSrc',
                                    selected='user',
                                    speed=speed)

    def qaqc_trdi(self, mmt):
        """Processes qaqc test, calibrations, and evaluations
        
        Parameters
        ----------
        mmt: MMTtrdi
            Object of MMT_TRDI
        """

        # ADCP Test
        if 'RG_Test' in mmt.qaqc:
            for n in range(len(mmt.qaqc['RG_Test'])):
                p_m = PreMeasurement()
                p_m.populate_data(mmt.qaqc['RG_Test_TimeStamp'][n],
                                  mmt.qaqc['RG_Test'][n], 'TST')
                self.system_tst.append(p_m)

        # Compass calibration
        if 'Compass_Calibration' in mmt.qaqc:
            for n in range(len(mmt.qaqc['Compass_Calibration'])):
                cc = PreMeasurement()
                cc.populate_data(mmt.qaqc['Compass_Calibration_TimeStamp'][n],
                                 mmt.qaqc['Compass_Calibration'][n], 'TCC')
                self.compass_cal.append(cc)
        # else:
        #     cc = PreMeasurement()
        #     self.compass_cal.append(cc)
            
        # Compass evaluation
        if 'Compass_Evaluation' in mmt.qaqc:
            for n in range(len(mmt.qaqc['Compass_Evaluation'])):
                ce = PreMeasurement()
                ce.populate_data(mmt.qaqc['Compass_Evaluation_TimeStamp'][n],
                                 mmt.qaqc['Compass_Evaluation'][n], 'TCC')
                self.compass_eval.append(ce)
        # else:
        #     ce = PreMeasurement()
        #     self.compass_cal.append(ce)

        # Check for moving-bed tests
        if len(mmt.mbt_transects) > 0:
            
            # Create transect objects
            transects = allocate_transects(mmt, transect_type='MB')

            # Process moving-bed tests
            if len(transects) > 0:
                self.mb_tests = []
                for n in range(len(transects)):

                    # Create moving-bed test object
                    mb_test = MovingBedTests()
                    mb_test.populate_data('TRDI', transects[n],
                                          mmt.mbt_transects[n].moving_bed_type)
                    
                    # Save notes from mmt files in comments
                    notes = getattr(mmt.mbt_transects[n], 'Notes')
                    for note in notes:
                        note_text = ' File: ' + note['NoteFileNo'] + ' ' \
                                    + note['NoteDate'] + ': ' + note['NoteText']
                        self.comments.append(note_text)

                    self.mb_tests.append(mb_test)

    @staticmethod
    def thresholds_trdi(transect, settings):
        """Retrieve and apply manual filter settings from mmt file

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        settings: dict
            Threshold settings computed before processing
        """

        # Apply WT settings
        transect.w_vel.apply_filter(transect, **settings['wt_settings'])

        # Apply BT settings
        transect.boat_vel.bt_vel.apply_filter(transect, **settings[
            'bt_settings'])

        # Apply depth settings
        transect.depths.bt_depths.valid_data_method = settings[
            'depth_settings']['depth_valid_method']
        transect.depths.depth_filter(transect=transect, filter_method=settings[
            'depth_settings']['depth_screening'])
        transect.depths.bt_depths.compute_avg_bt_depth(method=settings[
            'depth_settings']['depth_weighting'])

        # Apply composite depths as per setting stored in transect
        # from TransectData
        transect.depths.composite_depths(transect)

    def load_sontek(self, fullnames):
        """Coordinates reading of all SonTek data files.

        Parameters
        ----------
        fullnames: list
            File names including path for all discharge transects converted
            to Matlab files.
        """

        # Initialize variables
        rsdata = None
        pathname = None

        for file in fullnames:
            # Read data file
            rsdata = MatSonTek(file)
            pathname, file_name = os.path.split(file)

            # Create transect objects for each discharge transect
            self.transects.append(TransectData())
            self.transects[-1].sontek(rsdata, file_name)

        # Identify checked transects
        self.checked_transect_idx = self.checked_transects(self)

        # Site information pulled from last file
        if hasattr(rsdata, 'SiteInfo'):
            if hasattr(rsdata.SiteInfo, 'Site_Name'):
                self.station_name = rsdata.SiteInfo.Site_Name
            if hasattr(rsdata.SiteInfo, 'Station_Number'):
                self.station_number = rsdata.SiteInfo.Station_Number

        self.qaqc_sontek(pathname)

        for transect in self.transects:
            transect.change_coord_sys(new_coord_sys='Earth')
            transect.change_nav_reference(
                update=False,
                new_nav_ref=self.transects[self.checked_transect_idx[0]].boat_vel.selected)
            transect.boat_interpolations(update=False,
                                         target='BT',
                                         method='Hold9')
            transect.boat_interpolations(update=False,
                                         target='GPS',
                                         method='None')
            transect.apply_averaging_method(setting='Simple')
            transect.process_depths(update=False,
                                    interpolation_method='HoldLast')
            transect.update_water()

            # Filter water data
            transect.w_vel.apply_filter(transect=transect, wt_depth=True)

            # Interpolate water data
            transect.w_vel.apply_interpolation(transect=transect,
                                               ens_interp='None',
                                               cells_interp='None')
            transect.w_vel.apply_interpolation(transect=transect,
                                               ens_interp='None',
                                               cells_interp='TRDI')

    def qaqc_sontek(self, pathname):
        """Reads and stores system tests, compass calibrations,
        and moving-bed tests.

        Parameters
        ----------
        pathname: str
            Path to discharge transect files.
        """
        # Compass Evaluation
        # ce = PreMeasurement()
        # self.compass_eval.append(ce)

        # Compass Calibration
        compass_cal_folder = os.path.join(pathname, 'CompassCal')
        time_stamp = None
        if os.path.isdir(compass_cal_folder):
            for file in os.listdir(compass_cal_folder):
                valid_file = False
                # G3 compasses
                if file.endswith('.ccal'):
                    time_stamp = file.split('_')
                    time_stamp = time_stamp[0] + '_' + time_stamp[1]
                    valid_file = True

                # G2 compasses
                elif file.endswith('.txt'):
                    prefix, _ = os.path.splitext(file)
                    time_stamp = prefix.split('l')[1]
                    valid_file = True

            # for file in compass_cal_files:
                if valid_file:
                    with open(os.path.join(compass_cal_folder, file)) as f:
                        cal_data = f.read()
                        cal = PreMeasurement()
                        cal.populate_data(time_stamp, cal_data, 'SCC')
                        self.compass_cal.append(cal)

        # System Test
        system_test_folder = os.path.join(pathname, 'SystemTest')
        if os.path.isdir(system_test_folder):
            for file in os.listdir(system_test_folder):
                # Find system test files.
                if file.startswith('SystemTest'):
                    with open(os.path.join(system_test_folder, file)) as f:
                        test_data = f.read()
                        test_data = test_data.replace('\x00', '')
                    time_stamp = file[10:24]
                    sys_test = PreMeasurement()
                    sys_test.populate_data(time_stamp=time_stamp,
                                           data_in=test_data,
                                           data_type='SST')
                    self.system_tst.append(sys_test)

        # Moving-bed tests
        self.sontek_moving_bed_tests(pathname)

    def sontek_moving_bed_tests(self, pathname):
        """Locates and processes SonTek moving-bed tests.

        Searches the pathname for Matlab files that start with Loop or SMBA.
        Processes these files as moving bed tests.

        Parameters
        ----------
        pathname: str
            Path to discharge transect files.
        """
        for file in os.listdir(pathname):
            # Find moving-bed test files.
            if file.endswith('.mat'):
                # Process Loop test
                if file.lower().startswith('loop'):
                    self.mb_tests.append(MovingBedTests())
                    self.mb_tests[-1].populate_data(source='SonTek',
                                                    file=os.path.join(pathname, file),
                                                    test_type='Loop')
                # Process Stationary test
                elif file.lower().startswith('smba'):
                    self.mb_tests.append(MovingBedTests())
                    self.mb_tests[-1].populate_data(source='SonTek',
                                                    file=os.path.join(pathname, file),
                                                    test_type='Stationary')

    def load_qrev_mat(self, mat_data):
        """Loads and coordinates the mapping of existing QRev Matlab files
        into Python instance variables.

        Parameters
        ----------
        mat_data: dict
            Dictionary containing Matlab data.
        """

        meas_struct = mat_data['meas_struct']

        # Assign data from meas_struct to associated instance variables
        # in Measurement and associated objects.
        if len(meas_struct.stationName) > 0:
            self.station_name = meas_struct.stationName
        if len(meas_struct.stationNumber) > 0:
            self.station_number = meas_struct.stationNumber
        self.processing = meas_struct.processing
        if type(meas_struct.comments) == np.ndarray:
            self.comments = meas_struct.comments.tolist()

            # Needed to handle comments with blank lines
            for n, comment in enumerate(self.comments):
                if type(comment) is not str:
                    new_comment = ''
                    for item in comment:
                        if len(item.strip()) > 0:
                            new_comment = new_comment + item
                        else:
                            new_comment = new_comment + '\n'
                    self.comments[n] = new_comment
        else:
            self.comments = [meas_struct.comments]

        # Check to make sure all comments are str
        for n, comment in enumerate(self.comments):
            if type(comment) is np.ndarray:
                # Using comment =... didn't work but self.comments[n] does
                self.comments[2] = np.array2string(comment)

        if hasattr(meas_struct, 'userRating'):
            self.user_rating = meas_struct.userRating
        else:
            self.user_rating = ''

        self.initial_settings = vars(meas_struct.initialSettings)

        # Update initial settings to agree with Python definitions
        nav_dict = {'btVel': 'bt_vel', 'ggaVel': 'gga_vel', 'vtgVel': 'vtg_vel',
                    'bt_vel': 'bt_vel', 'gga_vel': 'gga_vel', 'vtg_vel': 'vtg_vel'}
        self.initial_settings['NavRef'] = nav_dict[self.initial_settings['NavRef']]

        on_off_dict = {'Off': False, 'On': True, 0: False, 1: True}
        self.initial_settings['WTwtDepthFilter'] = on_off_dict[self.initial_settings['WTwtDepthFilter']]

        if type(self.initial_settings['WTsnrFilter']) is np.ndarray:
            self.initial_settings['WTsnrFilter'] = 'Off'

        nav_dict = {'btDepths': 'bt_depths', 'vbDepths': 'vb_depths', 'dsDepths': 'ds_depths',
                    'bt_depths': 'bt_depths', 'vb_depths': 'vb_depths', 'ds_depths': 'ds_depths'}
        self.initial_settings['depthReference'] = nav_dict[self.initial_settings['depthReference']]

        self.ext_temp_chk = {'user': meas_struct.extTempChk.user,
                             'units': meas_struct.extTempChk.units,
                             'adcp': meas_struct.extTempChk.adcp}

        if hasattr(meas_struct.extTempChk, 'user_orig'):
            self.ext_temp_chk['user_orig'] = meas_struct.extTempChk.user_orig
        else:
            self.ext_temp_chk['user_orig'] = meas_struct.extTempChk.user

        if hasattr(meas_struct.extTempChk, 'adcp_orig'):
            self.ext_temp_chk['adcp_orig'] = meas_struct.extTempChk.adcp_orig
        else:
            self.ext_temp_chk['adcp_orig'] = meas_struct.extTempChk.adcp

        if type(self.ext_temp_chk['user']) is str:
            self.ext_temp_chk['user'] = np.nan
        if type(self.ext_temp_chk['adcp']) is str:
            self.ext_temp_chk['adcp'] = np.nan
        if type(self.ext_temp_chk['user']) is np.ndarray:
            self.ext_temp_chk['user'] = np.nan
        if type(self.ext_temp_chk['adcp']) is np.ndarray:
            self.ext_temp_chk['adcp'] = np.nan
        if type(self.ext_temp_chk['user_orig']) is str:
            self.ext_temp_chk['user_orig'] = np.nan
        if type(self.ext_temp_chk['adcp_orig']) is str:
            self.ext_temp_chk['adcp_orig'] = np.nan
        if type(self.ext_temp_chk['user_orig']) is np.ndarray:
            self.ext_temp_chk['user_orig'] = np.nan
        if type(self.ext_temp_chk['adcp_orig']) is np.ndarray:
            self.ext_temp_chk['adcp_orig'] = np.nan

        self.system_tst = PreMeasurement.sys_test_qrev_mat_in(meas_struct)

        # no compass cal compassCal is mat_struct with len(data) = 0
        try:
            self.compass_cal = PreMeasurement.cc_qrev_mat_in(meas_struct)
        except AttributeError:
            self.compass_cal = []

        try:
            self.compass_eval = PreMeasurement.ce_qrev_mat_in(meas_struct)
        except AttributeError:
            self.compass_eval = []

        self.transects = TransectData.qrev_mat_in(meas_struct)
        self.mb_tests = MovingBedTests.qrev_mat_in(meas_struct)
        self.extrap_fit = ComputeExtrap()
        self.extrap_fit.populate_from_qrev_mat(meas_struct)

        # # For compatibility with files saved prior to the implementation of the discharge weighted median option
        # if self.extrap_fit.norm_data[-1].weights is None:
        #     # Compute normalized data for each transect to obtain the weights
        #     for n, transect in enumerate(self.transects):
        #         norm_data_temp = NormData()
        #         norm_data_temp.populate_data(transect=transect,
        #                                 data_type=self.extrap_fit.norm_data[n].data_type,
        #                                 threshold=self.extrap_fit.threshold,
        #                                 data_extent=self.extrap_fit.subsection,
        #                                 use_weighted=self.extrap_fit.use_weighted,
        #                                 sub_from_left=self.extrap_fit.norm_data[n].sub_from_left,
        #                                 use_q=self.extrap_fit.norm_data[n].use_q)
        #         # Update the norm_data with the newly computed weights, however, the weights are not used until the
        #         # user changes the option in the GUI
        #         self.extrap_fit.norm_data[n].weights = norm_data_temp.weights
        #
        #     # Compute composite normalized data to get the composite weights
        #     norm_data_temp = NormData()
        #     norm_data_temp.create_composite(transects=self.transects,
        #                                     norm_data=self.extrap_fit.norm_data[0:-1],
        #                                     threshold=self.extrap_fit.threshold)
        #     # Update the norm_data with newly computed weights
        #     self.extrap_fit.norm_data[-1].weights = norm_data_temp.weights

        self.discharge = QComp.qrev_mat_in(meas_struct)

        # For compatibility with older QRev.mat files that didn't have this feature
        for n in range(len(self.transects)):
            if len(self.discharge[n].left_idx) == 0:
                self.discharge[n].left_idx = self.discharge[n].edge_ensembles(edge_loc='left',
                                                                              transect=self.transects[n])

            if len(self.discharge[n].right_idx) == 0:
                self.discharge[n].right_idx = self.discharge[n].edge_ensembles(edge_loc='right',
                                                                               transect=self.transects[n])

            if type(self.discharge[n].correction_factor) is list:
                self.discharge[n].correction_factor = self.discharge[n].total / self.discharge[n].total_uncorrected

        # Identify checked transects
        self.checked_transect_idx = self.checked_transects(self)

        self.uncertainty = Uncertainty()
        self.uncertainty.populate_from_qrev_mat(meas_struct)
        self.qa = QAData(self, mat_struct=meas_struct, compute=False)

    @staticmethod
    def set_num_beam_wt_threshold_trdi(mmt_transect):
        """Get number of beams to use in processing for WT from mmt file
        
        Parameters
        ----------
        mmt_transect: MMT_Transect
            Object of MMT_Transect
        
        Returns
        -------
        num_3_beam_wt_Out: int
        """

        use_3_beam_wt = mmt_transect.active_config['Proc_Use_3_Beam_WT']
        if use_3_beam_wt == 0:
            num_beam_wt_out = 4
        else:
            num_beam_wt_out = 3
            
        return num_beam_wt_out

    @staticmethod
    def set_num_beam_bt_threshold_trdi(mmt_transect):
        """Get number of beams to use in processing for BT from mmt file

        Parameters
        ----------
        mmt_transect: MMT_Transect
            Object of MMT_Transect

        Returns
        -------
        num_3_beam_WT_Out: int
        """

        use_3_beam_bt = mmt_transect.active_config['Proc_Use_3_Beam_BT']
        if use_3_beam_bt == 0:
            num_beam_bt_out = 4
        else:
            num_beam_bt_out = 3

        return num_beam_bt_out

    @staticmethod
    def set_depth_weighting_trdi(mmt_transect):
        """Get the average depth method from mmt
        
        Parameters
        ----------
        mmt_transect: MMT_Transect
            Object of MMT_Transect
        
        Returns
        -------
        depth_weighting_setting: str
            Method to compute mean depth
        """

        depth_weighting = mmt_transect.active_config['Proc_Use_Weighted_Mean_Depth']
        
        if depth_weighting == 0:
            depth_weighting_setting = 'Simple'
        else:
            depth_weighting_setting = 'IDW'

        return depth_weighting_setting

    @staticmethod
    def set_depth_screening_trdi(mmt_transect):
        """Get the depth screening setting from mmt
        
        Parameters
        ----------
        mmt_transect: MMT_Transect
            Object of MMT_Transect
        
        Returns
        -------
        depth_screening_setting: str
            Type of depth screening to use
        """

        depth_screen = mmt_transect.active_config['Proc_Screen_Depth']
        if depth_screen == 0:
            depth_screening_setting = 'None'
        else:
            depth_screening_setting = 'TRDI'
        
        return depth_screening_setting
        
    def change_sos(self, transect_idx=None, parameter=None, salinity=None, temperature=None, selected=None, speed=None):
        """Applies a change in speed of sound to one or all transects
        and update the discharge and uncertainty computations
        
        Parameters
        ----------
        transect_idx: int
            Index of transect to change
        parameter: str
            Speed of sound parameter to be changed ('temperatureSrc', 'temperature', 'salinity', 'sosSrc')
        salinity: float
            Salinity in ppt
        temperature: float
            Temperature in deg C
        selected: str
            Selected speed of sound ('internal', 'computed', 'user') or temperature ('internal', 'user')
        speed: float
            Manually supplied speed of sound for 'user' source
        """
        
        s = self.current_settings()
        if transect_idx is None:
            # Apply to all transects
            for transect in self.transects:
                transect.change_sos(parameter=parameter,
                                    salinity=salinity,
                                    temperature=temperature,
                                    selected=selected,
                                    speed=speed)
        else:
            # Apply to a single transect
            self.transects[transect_idx].change_sos(parameter=parameter,
                                                    salinity=salinity,
                                                    temperature=temperature,
                                                    selected=selected,
                                                    speed=speed)
        # Reapply settings to newly adjusted data
        self.apply_settings(s)

    def change_magvar(self, magvar, transect_idx=None):
        """Coordinates changing the magnetic variation.

        Parameters
        ----------
        magvar: float
            Magnetic variation
        transect_idx: int
            Index of transect to which the change is applied. None is all transects.
        """

        # Get current settings
        s = self.current_settings()

        # Initialize variables
        n_transects = len(self.transects)
        recompute = False
        n = 0

        # If the internal compass is used the recompute is necessary
        while n < n_transects and recompute == False:
            if self.transects[n].sensors.heading_deg.selected == 'internal':
                recompute = True
            n += 1

        # Apply change
        if transect_idx is None:
            # Apply change to all transects
            for transect in self.transects:
                transect.change_mag_var(magvar)

            # Apply change to moving-bed tests
            if len(self.mb_tests) > 0:
                for test in self.mb_tests:
                    old_magvar = test.transect.sensors.heading_deg.internal.mag_var_deg
                    test.transect.change_mag_var(magvar)
                    test.magvar_change(magvar, old_magvar)
        else:
            self.transects[transect_idx].change_mag_var(magvar)

        # Recompute is specified
        if recompute:
            self.apply_settings(s)
        else:
            self.qa.compass_qa(self)
            self.qa.check_compass_settings(self)

    def change_h_offset(self, h_offset, transect_idx=None):
        """Coordinates changing the heading offset for external heading.

        Parameters
        ----------
        h_offset: float
            Heading offset
        transect_idx: int
            Index of transect to which the change is applied. None is all transects.
        """

        # Get current settings
        s = self.current_settings()

        # Initialize variables
        n_transects = len(self.transects)
        recompute = False
        n = 0

        # If external compass is used then a recompute is necessary
        while n < n_transects and recompute == False:
            if self.transects[n].sensors.heading_deg.selected == 'external':
                recompute = True
            n += 1

        # Apply change
        if transect_idx is None:
            for transect in self.transects:
                transect.change_offset(h_offset)

            # Apply change to moving-bed tests
            if len(self.mb_tests) > 0:
                for test in self.mb_tests:
                    old_h_offset = test.transect.sensors.heading_deg.external.align_correction_deg
                    test.transect.change_offset(h_offset)
                    test.h_offset_change(h_offset, old_h_offset)
        else:
            self.transects[transect_idx].change_offset(h_offset)

        # Rcompute is specified
        if recompute:
            self.apply_settings(s)
        else:
            self.qa.compass_qa(self)
            self.qa.check_compass_settings(self)

    def change_h_source(self, h_source, transect_idx=None):
        """Coordinates changing the heading source.

        Parameters
        ----------
        h_source: str
            Heading source (internal or external)
        transect_idx: int
            Index of transect to which the change is applied. None is all transects.
        """

        # Get current settings
        s = self.current_settings()

        # Apply change
        if transect_idx is None:
            for transect in self.transects:
                transect.change_heading_source(h_source)

            # Apply change to moving-bed tests
            if len(self.mb_tests) > 0:
                for test in self.mb_tests:
                    test.transect.change_heading_source(h_source)
                    test.process_mb_test(source=test.transect.adcp.manufacturer)
                settings = self.current_settings()
                select = settings['NavRef']
                ref = None
                if select == 'bt_vel':
                    ref = 'BT'
                elif select == 'gga_vel':
                    ref = 'GGA'
                elif select == 'vtg_vel':
                    ref = 'VTG'
                self.mb_tests = MovingBedTests.auto_use_2_correct(
                    moving_bed_tests=self.mb_tests, boat_ref=ref)

        else:
            self.transects[transect_idx].change_heading_source(h_source)

        self.apply_settings(s)

    def change_draft(self, draft, transect_idx=None):
        """Coordinates changing the ADCP draft.

        Parameters
        ----------
        draft: float
            Draft of ADCP in m
        transect_idx: int
            Index of transect to which the change is applied. None is all transects.
        """

        # Get current settings
        s = self.current_settings()

        # Apply change
        if transect_idx is None:
            for transect in self.transects:
                transect.change_draft(draft)
        else:
            self.transects[transect_idx].change_draft(draft)

        self.apply_settings(s)

    @staticmethod
    def h_external_valid(meas):
        """Determine if valid external heading data is included in the measurement.

        Parameters
        ----------
        meas: Measurement
            Object of Measurement
        """

        external = False
        for transect in meas.transects:
            if transect.sensors.heading_deg.external is not None:
                external = True
                break
        return external

    def apply_settings(self, settings, force_abba=True):
        """Applies reference, filter, and interpolation settings.
        
        Parameters
        ----------
        settings: dict
            Dictionary of reference, filter, and interpolation settings
        force_abba: bool
            Allows the above, below, before, after interpolation to be applied even when the data use another approach.
        """

        for transect in self.transects:

            # Moving-boat ensembles
            if 'Processing' in settings.keys():
                transect.change_q_ensembles(proc_method=settings['Processing'])
                self.processing = settings['Processing']

            # Navigation reference
            if transect.boat_vel.selected != settings['NavRef']:
                transect.change_nav_reference(update=False, new_nav_ref=settings['NavRef'])
                if len(self.mb_tests) > 0:
                    self.mb_tests = MovingBedTests.auto_use_2_correct(
                        moving_bed_tests=self.mb_tests,
                        boat_ref=settings['NavRef'])

            # Changing the nav reference applies the current setting for
            # Composite tracks, check to see if a change is needed
            if transect.boat_vel.composite != settings['CompTracks']:
                transect.composite_tracks(update=False, setting=settings['CompTracks'])

            # Set difference velocity BT filter
            bt_kwargs = {}
            if settings['BTdFilter'] == 'Manual':
                bt_kwargs['difference'] = settings['BTdFilter']
                bt_kwargs['difference_threshold'] = settings['BTdFilterThreshold']
            else:
                bt_kwargs['difference'] = settings['BTdFilter']

            # Set vertical velocity BT filter
            if settings['BTwFilter'] == 'Manual':
                bt_kwargs['vertical'] = settings['BTwFilter']
                bt_kwargs['vertical_threshold'] = settings['BTwFilterThreshold']
            else:
                bt_kwargs['vertical'] = settings['BTwFilter']

            # Apply beam filter
                bt_kwargs['beam'] = settings['BTbeamFilter']

            # Apply smooth filter
                bt_kwargs['other'] = settings['BTsmoothFilter']

            # Apply BT settings
            transect.boat_filters(update=False, **bt_kwargs)

            # BT Interpolation
            transect.boat_interpolations(update=False,
                                         target='BT',
                                         method=settings['BTInterpolation'])

            # GPS filter settings
            if transect.gps is not None:
                gga_kwargs = {}
                if transect.boat_vel.gga_vel is not None:
                    # GGA
                    gga_kwargs['differential'] = settings['ggaDiffQualFilter']
                    if settings['ggaAltitudeFilter'] == 'Manual':
                        gga_kwargs['altitude'] = settings['ggaAltitudeFilter']
                        gga_kwargs['altitude_threshold'] = settings['ggaAltitudeFilterChange']
                    else:
                        gga_kwargs['altitude'] = settings['ggaAltitudeFilter']

                    # Set GGA HDOP Filter
                    if settings['GPSHDOPFilter'] == 'Manual':
                        gga_kwargs['hdop'] = settings['GPSHDOPFilter']
                        gga_kwargs['hdop_max_threshold'] = settings['GPSHDOPFilterMax']
                        gga_kwargs['hdop_change_threshold'] = settings['GPSHDOPFilterChange']
                    else:
                        gga_kwargs['hdop'] = settings['GPSHDOPFilter']

                    gga_kwargs['other'] = settings['GPSSmoothFilter']
                    # Apply GGA filters
                    transect.gps_filters(update=False, **gga_kwargs)

                if transect.boat_vel.vtg_vel is not None:
                    vtg_kwargs = {}
                    if settings['GPSHDOPFilter'] == 'Manual':
                        vtg_kwargs['hdop'] = settings['GPSHDOPFilter']
                        vtg_kwargs['hdop_max_threshold'] = settings['GPSHDOPFilterMax']
                        vtg_kwargs['hdop_change_threshold'] = settings['GPSHDOPFilterChange']
                        vtg_kwargs['other'] = settings['GPSSmoothFilter']
                    else:
                        vtg_kwargs['hdop'] = settings['GPSHDOPFilter']
                        vtg_kwargs['other'] = settings['GPSSmoothFilter']

                    # Apply VTG filters
                    transect.gps_filters(update=False, **vtg_kwargs)

                transect.boat_interpolations(update=False,
                                             target='GPS',
                                             method=settings['GPSInterpolation'])

            # Set depth reference
            transect.set_depth_reference(update=False, setting=settings['depthReference'])

            transect.process_depths(update=True,
                                    filter_method=settings['depthFilterType'],
                                    interpolation_method=settings['depthInterpolation'],
                                    composite_setting=settings['depthComposite'],
                                    avg_method=settings['depthAvgMethod'],
                                    valid_method=settings['depthValidMethod'])

            # Set WT difference velocity filter
            wt_kwargs = {}
            if settings['WTdFilter'] == 'Manual':
                wt_kwargs['difference'] = settings['WTdFilter']
                wt_kwargs['difference_threshold'] = settings['WTdFilterThreshold']
            else:
                wt_kwargs['difference'] = settings['WTdFilter']

            # Set WT vertical velocity filter
            if settings['WTwFilter'] == 'Manual':
                wt_kwargs['vertical'] = settings['WTwFilter']
                wt_kwargs['vertical_threshold'] = settings['WTwFilterThreshold']
            else:
                wt_kwargs['vertical'] = settings['WTwFilter']

            wt_kwargs['beam'] = settings['WTbeamFilter']
            wt_kwargs['other'] = settings['WTsmoothFilter']
            wt_kwargs['snr'] = settings['WTsnrFilter']
            wt_kwargs['wt_depth'] = settings['WTwtDepthFilter']
            wt_kwargs['excluded'] = settings['WTExcludedDistance']

            # Data loaded from old QRev.mat files will be set to use this new interpolation method. When reprocessing
            # any data the interpolation method should be 'abba'
            if force_abba:
                transect.w_vel.interpolate_cells = 'abba'
                transect.w_vel.interpolate_ens = 'abba'
                settings['WTEnsInterpolation'] = 'abba'
                settings['WTCellInterpolation'] = 'abba'

            transect.w_vel.apply_filter(transect=transect, **wt_kwargs)

            # Edge methods
            transect.edges.rec_edge_method = settings['edgeRecEdgeMethod']
            transect.edges.vel_method = settings['edgeVelMethod']

        # Recompute extrapolations
        # NOTE: Extrapolations should be determined prior to WT
        # interpolations because the TRDI approach for power/power
        # using the power curve and exponent to estimate invalid cells.

        if settings['UseWeighted'] and not self.use_weighted:
            if self.extrap_fit.norm_data[-1].weights is None:
                # Compute normalized data for each transect to obtain the weights
                self.extrap_fit.process_profiles(self.transects, self.extrap_fit.norm_data[-1].data_type,
                                                 use_weighted=settings['UseWeighted'])
            # self.extrap_fit.norm_data = []
            # for n, transect in enumerate(self.transects):
            #     self.extrap_fit.norm_data[n] = NormData()
            #     self.extrap_fit.norm_data[n].populate_data(transect=transect,
            #                             data_type=self.extrap_fit.norm_data[n].data_type,
            #                             threshold=self.extrap_fit.threshold,
            #                             data_extent=self.extrap_fit.subsection,
            #                             use_weighted=self.extrap_fit.use_weighted,
            #                             sub_from_left=self.extrap_fit.norm_data[n].sub_from_left,
            #                             use_q=self.extrap_fit.norm_data[n].use_q)
            #
            # # Compute composite normalized data to get the composite weights
            # norm_data_temp = NormData()
            # norm_data_temp.create_composite(transects=self.transects,
            #                                 norm_data=self.extrap_fit.norm_data[0:-1],
            #                                 threshold=self.extrap_fit.threshold)
            # # Update the norm_data with newly computed weights
            # self.extrap_fit.norm_data[-1].weights = norm_data_temp.weights

        self.use_weighted = settings['UseWeighted']

        if len(self.checked_transect_idx) > 0:
            ref_transect = self.checked_transect_idx[0]
        else:
            ref_transect = 0

        if self.transects[ref_transect].w_vel.interpolate_cells == 'TRDI':
            if self.extrap_fit is None:
                self.extrap_fit = ComputeExtrap()
                self.extrap_fit.populate_data(transects=self.transects, compute_sensitivity=False,
                                              use_weighted=settings['UseWeighted'])
                self.change_extrapolation(self.extrap_fit.fit_method, compute_q=False,
                                          use_weighted=settings['UseWeighted'])
            elif self.extrap_fit.fit_method == 'Automatic':
                self.change_extrapolation(self.extrap_fit.fit_method, compute_q=False,
                                          use_weighted=settings['UseWeighted'])
            else:
                if 'extrapTop' not in settings.keys():
                    settings['extrapTop'] = self.extrap_fit.sel_fit[-1].top_method
                    settings['extrapBot'] = self.extrap_fit.sel_fit[-1].bot_method
                    settings['extrapExp'] = self.extrap_fit.sel_fit[-1].exponent

            self.change_extrapolation(self.extrap_fit.fit_method,
                                      top=settings['extrapTop'],
                                      bot=settings['extrapBot'],
                                      exp=settings['extrapExp'],
                                      compute_q=False,
                                      use_weighted=settings['UseWeighted'])

        for transect in self.transects:

            # Water track interpolations
            transect.w_vel.apply_interpolation(transect=transect,
                                               ens_interp=settings['WTEnsInterpolation'],
                                               cells_interp=settings['WTCellInterpolation'])

        if self.extrap_fit is None:
            self.extrap_fit = ComputeExtrap()
            self.extrap_fit.populate_data(transects=self.transects, compute_sensitivity=False,
                                          use_weighted=settings['UseWeighted'])
            self.change_extrapolation(self.extrap_fit.fit_method, compute_q=False,
                                      use_weighted=settings['UseWeighted'])
        elif self.extrap_fit.fit_method == 'Automatic':
            self.change_extrapolation(self.extrap_fit.fit_method, compute_q=False,
                                      use_weighted=settings['UseWeighted'])
        else:
            if 'extrapTop' not in settings.keys():
                settings['extrapTop'] = self.extrap_fit.sel_fit[-1].top_method
                settings['extrapBot'] = self.extrap_fit.sel_fit[-1].bot_method
                settings['extrapExp'] = self.extrap_fit.sel_fit[-1].exponent

        self.change_extrapolation(self.extrap_fit.fit_method,
                                  top=settings['extrapTop'],
                                  bot=settings['extrapBot'],
                                  exp=settings['extrapExp'],
                                  compute_q=False,
                                  use_weighted=settings['UseWeighted'])

        self.extrap_fit.q_sensitivity = ExtrapQSensitivity()
        self.extrap_fit.q_sensitivity.populate_data(transects=self.transects,
                                                    extrap_fits=self.extrap_fit.sel_fit)

        self.compute_discharge()

        self.uncertainty = Uncertainty()
        self.uncertainty.compute_uncertainty(self)
        self.qa = QAData(self)
        if self.run_oursin:
            self.oursin = Oursin()
            self.oursin.compute_oursin(self)

    def current_settings(self):
        """Saves the current settings for a measurement. Since all settings
        in QRev are consistent among all transects in a measurement only the
        settings from the first transect are saved
        """

        settings = {}

        if len(self.checked_transect_idx) > 0:
            ref_transect = self.checked_transect_idx[0]
        else:
            ref_transect = 0

        # Guard against empty or inconsistent transect indices
        if not isinstance(self.transects, list) or len(self.transects) == 0:
            raise ValueError("Measurement.current_settings: no transects loaded in measurement."
                             " Check the input file or the load function.")

        if ref_transect < 0 or ref_transect >= len(self.transects):
            warnings.warn(
                f"checked_transect_idx ({ref_transect}) out of range for {len(self.transects)} transects; using first transect instead.")
            ref_transect = 0

        transect = self.transects[ref_transect]
        
        # Navigation reference
        settings['NavRef'] = transect.boat_vel.selected
        
        # Composite tracks
        settings['CompTracks'] = transect.boat_vel.composite
        
        # Water track settings
        settings['WTbeamFilter'] = transect.w_vel.beam_filter
        settings['WTdFilter'] = transect.w_vel.d_filter
        settings['WTdFilterThreshold'] = transect.w_vel.d_filter_threshold
        settings['WTwFilter'] = transect.w_vel.w_filter
        settings['WTwFilterThreshold'] = transect.w_vel.w_filter_threshold
        settings['WTsmoothFilter'] = transect.w_vel.smooth_filter
        settings['WTsnrFilter'] = transect.w_vel.snr_filter
        settings['WTwtDepthFilter'] = transect.w_vel.wt_depth_filter
        settings['WTEnsInterpolation'] = transect.w_vel.interpolate_ens
        settings['WTCellInterpolation'] = transect.w_vel.interpolate_cells
        settings['WTExcludedDistance'] = transect.w_vel.excluded_dist_m
        
        # Bottom track settings
        settings['BTbeamFilter'] = transect.boat_vel.bt_vel.beam_filter
        settings['BTdFilter'] = transect.boat_vel.bt_vel.d_filter
        settings['BTdFilterThreshold'] = transect.boat_vel.bt_vel.d_filter_threshold
        settings['BTwFilter'] = transect.boat_vel.bt_vel.w_filter
        settings['BTwFilterThreshold'] = transect.boat_vel.bt_vel.w_filter_threshold
        settings['BTsmoothFilter'] = transect.boat_vel.bt_vel.smooth_filter
        settings['BTInterpolation'] = transect.boat_vel.bt_vel.interpolate
        
        # Gps Settings
        # if transect.gps is not None:

        gga_present = False
        for idx in self.checked_transect_idx:
            if self.transects[idx].boat_vel.gga_vel is not None:
                gga_present = True
                transect = self.transects[idx]
                break

        # GGA settings
        if gga_present:
            settings['ggaDiffQualFilter'] = transect.boat_vel.gga_vel.gps_diff_qual_filter
            settings['ggaAltitudeFilter'] = transect.boat_vel.gga_vel.gps_altitude_filter
            settings['ggaAltitudeFilterChange'] = \
                transect.boat_vel.gga_vel.gps_altitude_filter_change
            settings['GPSHDOPFilter'] = transect.boat_vel.gga_vel.gps_HDOP_filter
            settings['GPSHDOPFilterMax'] = transect.boat_vel.gga_vel.gps_HDOP_filter_max
            settings['GPSHDOPFilterChange'] = transect.boat_vel.gga_vel.gps_HDOP_filter_change
            settings['GPSSmoothFilter'] = transect.boat_vel.gga_vel.smooth_filter
            settings['GPSInterpolation'] = transect.boat_vel.gga_vel.interpolate
        else:
            settings['ggaDiffQualFilter'] = 1
            settings['ggaAltitudeFilter'] = 'Off'
            settings['ggaAltitudeFilterChange'] = []

            settings['ggaSmoothFilter'] = 'Off'
            if 'GPSInterpolation' not in settings.keys():
                settings['GPSInterpolation'] = 'None'
            if 'GPSHDOPFilter' not in settings.keys():
                settings['GPSHDOPFilter'] = 'Off'
                settings['GPSHDOPFilterMax'] = []
                settings['GPSHDOPFilterChange'] = []
            if 'GPSSmoothFilter' not in settings.keys():
                settings['GPSSmoothFilter'] = 'Off'

        # VTG settings
        vtg_present = False
        for idx in self.checked_transect_idx:
            if self.transects[idx].boat_vel.vtg_vel is not None:
                vtg_present = True
                transect = self.transects[idx]
                break

        if vtg_present:
            settings['GPSHDOPFilter'] = transect.boat_vel.vtg_vel.gps_HDOP_filter
            settings['GPSHDOPFilterMax'] = transect.boat_vel.vtg_vel.gps_HDOP_filter_max
            settings['GPSHDOPFilterChange'] = transect.boat_vel.vtg_vel.gps_HDOP_filter_change
            settings['GPSSmoothFilter'] = transect.boat_vel.vtg_vel.smooth_filter
            settings['GPSInterpolation'] = transect.boat_vel.vtg_vel.interpolate

        # Depth Settings
        settings['depthAvgMethod'] = transect.depths.bt_depths.avg_method
        settings['depthValidMethod'] = transect.depths.bt_depths.valid_data_method
        
        # Depth settings are always applied to all available depth sources.
        # Only those saved in the bt_depths are used here but are applied to all sources
        settings['depthFilterType'] = transect.depths.bt_depths.filter_type
        settings['depthReference'] = transect.depths.selected
        settings['depthComposite'] = transect.depths.composite
        select = getattr(transect.depths, transect.depths.selected)
        settings['depthInterpolation'] = select.interp_type
        
        # Extrap Settings
        if self.extrap_fit is None:
            settings['extrapTop'] = transect.extrap.top_method
            settings['extrapBot'] = transect.extrap.bot_method
            settings['extrapExp'] = transect.extrap.exponent
        else:
            settings['extrapTop'] = self.extrap_fit.sel_fit[-1].top_method
            settings['extrapBot'] = self.extrap_fit.sel_fit[-1].bot_method
            settings['extrapExp'] = self.extrap_fit.sel_fit[-1].exponent

        # Use of self.use_weighted allows a QRev mat file to be loaded and initially processed with the settings from
        # the QRev file but upon reprocessing the self.use_weights will be set to the options setting for use_weights
        settings['UseWeighted'] = self.use_weighted

        # Edge Settings
        settings['edgeVelMethod'] = transect.edges.vel_method
        settings['edgeRecEdgeMethod'] = transect.edges.rec_edge_method
        
        return settings

    def qrev_default_settings(self, check_user_excluded_dist=False, use_weighted=False):
        """QRev default and filter settings for a measurement.
        """

        settings = dict()

        if len(self.checked_transect_idx) > 0:
            ref_transect = self.checked_transect_idx[0]
        else:
            ref_transect = 0

        # Navigation reference
        settings['NavRef'] = self.transects[ref_transect].boat_vel.selected

        # Composite tracks
        settings['CompTracks'] = 'Off'

        # Water track filter settings
        settings['WTbeamFilter'] = -1
        settings['WTdFilter'] = 'Auto'
        settings['WTdFilterThreshold'] = np.nan
        settings['WTwFilter'] = 'Auto'
        settings['WTwFilterThreshold'] = np.nan
        settings['WTsmoothFilter'] = 'Off'

        if self.transects[ref_transect].adcp.manufacturer == 'TRDI':
            settings['WTsnrFilter'] = 'Off'
        else:
            settings['WTsnrFilter'] = 'Auto'

        if check_user_excluded_dist:
            temp = [x.w_vel for x in self.transects]
            excluded_dist = np.nanmin([x.excluded_dist_m for x in temp])
        else:
            excluded_dist = 0
        if excluded_dist < 0.158 and self.transects[ref_transect].adcp.model == 'M9':
            settings['WTExcludedDistance'] = 0.16
        elif excluded_dist < 0.248 and self.transects[ref_transect].adcp.model == 'RioPro':
            settings['WTExcludedDistance'] = 0.25
        else:
            settings['WTExcludedDistance'] = excluded_dist

        # Bottom track filter settings
        settings['BTbeamFilter'] = -1
        settings['BTdFilter'] = 'Auto'
        settings['BTdFilterThreshold'] = np.nan
        settings['BTwFilter'] = 'Auto'
        settings['BTwFilterThreshold'] = np.nan
        settings['BTsmoothFilter'] = 'Off'

        # GGA Filter settings
        settings['ggaDiffQualFilter'] = 2
        settings['ggaAltitudeFilter'] = 'Auto'
        settings['ggaAltitudeFilterChange'] = np.nan

        # VTG filter settings
        settings['vtgsmoothFilter'] = 'Off'

        # GGA and VTG filter settings
        settings['GPSHDOPFilter'] = 'Auto'
        settings['GPSHDOPFilterMax'] = np.nan
        settings['GPSHDOPFilterChange'] = np.nan
        settings['GPSSmoothFilter'] = 'Off'

        # Depth Averaging
        settings['depthAvgMethod'] = 'IDW'
        settings['depthValidMethod'] = 'QRev'

        # Depth Reference

        # Default to 4 beam depth average
        settings['depthReference'] = 'bt_depths'
        # Depth settings
        settings['depthFilterType'] = 'Smooth'
        settings['depthComposite'] = 'Off'
        for transect in self.transects:
            if transect.checked:

                if transect.depths.vb_depths is not None or transect.depths.ds_depths is not None:
                    settings['depthComposite'] = 'On'
                    break
                else:
                    settings['depthComposite'] = 'Off'
                    break

        # Interpolation settings
        settings = self.qrev_default_interpolation_methods(settings)

        # Edge settings
        settings['edgeVelMethod'] = 'MeasMag'
        settings['edgeRecEdgeMethod'] = 'Fixed'

        # Extrapolation Settings
        settings['extrapTop'] = 'Power'
        settings['extrapBot'] = 'Power'
        settings['extrapExp'] = 0.1667
        settings['UseWeighted'] = use_weighted

        return settings

    def update_qa(self):
        self.qa = QAData(self)

    @staticmethod
    def no_filter_interp_settings(self):
        """Settings to turn off all filters and interpolations.

        Returns
        -------
        settings: dict
            Dictionary of all processing settings.
        """

        settings = dict()
        if len(self.checked_transect_idx) > 0:
            ref_transect = self.checked_transect_idx[0]
        else:
            ref_transect = 0

        settings['NavRef'] = self.transects[ref_transect].boatVel.selected

        # Composite tracks
        settings['CompTracks'] = 'Off'

        # Water track filter settings
        settings['WTbeamFilter'] = 3
        settings['WTdFilter'] = 'Off'
        settings['WTdFilterThreshold'] = np.nan
        settings['WTwFilter'] = 'Off'
        settings['WTwFilterThreshold'] = np.nan
        settings['WTsmoothFilter'] = 'Off'
        settings['WTsnrFilter'] = 'Off'

        temp = [x.w_vel for x in self.transects]
        excluded_dist = np.nanmin([x.excluded_dist_m for x in temp])

        settings['WTExcludedDistance'] = excluded_dist

        # Bottom track filter settings
        settings['BTbeamFilter'] = 3
        settings['BTdFilter'] = 'Off'
        settings['BTdFilterThreshold'] = np.nan
        settings['BTwFilter'] = 'Off'
        settings['BTwFilterThreshold'] = np.nan
        settings['BTsmoothFilter'] = 'Off'

        # GGA filter settings
        settings['ggaDiffQualFilter'] = 1
        settings['ggaAltitudeFilter'] = 'Off'
        settings['ggaAltitudeFilterChange'] = np.nan

        # VTG filter settings
        settings['vtgsmoothFilter'] = 'Off'

        # GGA and VTG filter settings
        settings['GPSHDOPFilter'] = 'Off'
        settings['GPSHDOPFilterMax'] = np.nan
        settings['GPSHDOPFilterChange'] = np.nan
        settings['GPSSmoothFilter'] = 'Off'

        # Depth Averaging
        settings['depthAvgMethod'] = 'IDW'
        settings['depthValidMethod'] = 'QRev'

        # Depth Reference

        # Default to 4 beam depth average
        settings['depthReference'] = 'btDepths'
        # Depth settings
        settings['depthFilterType'] = 'None'
        settings['depthComposite'] = 'Off'

        # Interpolation settings
        settings['BTInterpolation'] = 'None'
        settings['WTEnsInterpolation'] = 'None'
        settings['WTCellInterpolation'] = 'None'
        settings['GPSInterpolation'] = 'None'
        settings['depthInterpolation'] = 'None'
        settings['WTwtDepthFilter'] = 'Off'

        # Edge Settings
        settings['edgeVelMethod'] = 'MeasMag'
        # settings['edgeVelMethod'] = 'Profile'
        settings['edgeRecEdgeMethod'] = 'Fixed'

        return settings

    def selected_transects_changed(self, selected_transects_idx):
        """Handle changes in the transects selected for computing discharge.

        Parameters
        ----------
        selected_transects_idx: list
            List of indices of the transects used to compute discharge
        """

        # Update transect settings
        for n in range(len(self.transects)):
            if n in selected_transects_idx:
                self.transects[n].checked = True
            else:
                self.transects[n].checked = False

        # Changes in the transects selected may cause a change in extrapolation.
        self.extrap_fit = ComputeExtrap()
        self.extrap_fit.populate_data(transects=self.transects, compute_sensitivity=False)
        top = self.extrap_fit.sel_fit[-1].top_method
        bot = self.extrap_fit.sel_fit[-1].bot_method
        exp = self.extrap_fit.sel_fit[-1].exponent
        self.change_extrapolation(self.extrap_fit.fit_method, top=top, bot=bot, exp=exp)

        self.extrap_fit.q_sensitivity = ExtrapQSensitivity()
        self.extrap_fit.q_sensitivity.populate_data(transects=self.transects,
                                                    extrap_fits=self.extrap_fit.sel_fit)

        # Update computations
        self.compute_discharge()
        self.uncertainty = Uncertainty()
        self.uncertainty.compute_uncertainty(self)
        self.qa = QAData(self)
        if self.run_oursin:
            self.oursin = Oursin()
            self.oursin.compute_oursin(self)

    def compute_discharge(self):
        """Computes the discharge for all transects in the measurement.
        """

        self.discharge = []
        for transect in self.transects:
            q = QComp()
            q.populate_data(data_in=transect, moving_bed_data=self.mb_tests)
            self.discharge.append(q)

    @staticmethod
    def compute_edi(meas, selected_idx, percents):
        """Computes the locations and vertical properties for the user selected transect and
        flow percentages.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        selected_idx: int
            Index of selected transect
        percents: list
            List of selected flow percents
        """

        # Get transect and discharge data
        transect = meas.transects[selected_idx]
        discharge = meas.discharge[selected_idx]

        # Sort the percents in ascending order
        percents.sort()

        # Compute cumulative discharge
        q_cum = np.nancumsum(discharge.middle_ens + discharge.top_ens + discharge.bottom_ens)

        # Adjust for moving-bed conditions
        q_cum = q_cum * discharge.correction_factor

        # Adjust q for starting edge
        if transect.start_edge == 'Left':
            q_cum = q_cum + discharge.left
            q_cum[-1] = q_cum[-1] + discharge.right
            start_dist = transect.edges.left.distance_m
        else:
            q_cum = q_cum + discharge.right
            q_cum[-1] = q_cum[-1] + discharge.left
            start_dist = transect.edges.right.distance_m

        # Determine ensemble at each percent
        ensembles = []
        q_target = []
        for percent in percents:
            q_target.append(q_cum[-1] * percent / 100)
            if q_target[-1] > 0:
                ensembles.append(np.where(q_cum > q_target[-1])[0][0])
            if q_target[-1] < 0:
                ensembles.append(np.where(q_cum < q_target[-1])[0][0])

        # Compute distance from start bank
        boat_vel_selected = getattr(transect.boat_vel, transect.boat_vel.selected)
        track_x = np.nancumsum(boat_vel_selected.u_processed_mps[transect.in_transect_idx] *
                               transect.date_time.ens_duration_sec[transect.in_transect_idx])
        track_y = np.nancumsum(boat_vel_selected.v_processed_mps[transect.in_transect_idx] *
                               transect.date_time.ens_duration_sec[transect.in_transect_idx])

        dist = np.sqrt(track_x ** 2 + track_y ** 2) + start_dist

        # Initialize variables for computing vertical data
        n_pts_in_avg = int(len(q_cum) * 0.01)
        depth_selected = getattr(transect.depths, transect.depths.selected)
        q_actual = []
        distance = []
        lat = []
        lon = []
        depth = []
        velocity = []

        # Compute data for each vertical
        for ensemble in ensembles:
            q_actual.append(q_cum[ensemble])
            distance.append(dist[ensemble])
            # Report lat and lon if available
            try:
                lat.append(transect.gps.gga_lat_ens_deg[ensemble])
                lon.append(transect.gps.gga_lon_ens_deg[ensemble])
            except (ValueError, AttributeError, TypeError):
                lat.append('')
                lon.append('')
            depth.append(depth_selected.depth_processed_m[ensemble])

            # The velocity is an average velocity for ensembles +/- 1% of the total ensembles
            # about the selected ensemble
            u = np.nanmean(transect.w_vel.u_processed_mps[:, ensemble - n_pts_in_avg: ensemble + n_pts_in_avg + 1], 1)
            v = np.nanmean(transect.w_vel.v_processed_mps[:, ensemble - n_pts_in_avg: ensemble + n_pts_in_avg + 1], 1)
            velocity.append(np.sqrt(np.nanmean(u)**2 + np.nanmean(v)**2))

        # Save computed results in a dictionary
        edi_results = {'percent': percents, 'target_q': q_target, 'actual_q': q_actual, 'distance': distance,
                       'depth': depth, 'velocity': velocity, 'lat': lat, 'lon': lon}
        return edi_results

    @staticmethod
    def qrev_default_interpolation_methods(settings):
        """Adds QRev default interpolation settings to existing settings data structure

        Parameters
        ----------
        settings: dict
            Dictionary of reference and filter settings

        Returns
        -------
        settings: dict
            Dictionary with reference, filter, and interpolation settings
        """

        settings['BTInterpolation'] = 'Linear'
        settings['WTEnsInterpolation'] = 'abba'
        settings['WTCellInterpolation'] = 'abba'
        settings['GPSInterpolation'] = 'Linear'
        settings['depthInterpolation'] = 'Linear'
        settings['WTwtDepthFilter'] = 'On'

        return settings

    def change_extrapolation(self, method, top=None, bot=None, exp=None, extents=None, threshold=None, compute_q=True,
                             use_weighted=False):
        """Applies the selected extrapolation method to each transect.

        Parameters
        ----------
        method: str
            Method of computation Automatic or Manual
        top: str
            Top extrapolation method
        bot: str
            Bottom extrapolation method
        exp: float
            Exponent for power or no slip methods
        threshold: float
            Threshold as a percent for determining if a median is valid
        extents: list
            Percent of discharge, does not account for transect direction
        compute_q: bool
            Specifies if the discharge should be computed
        use_weighted: bool
            Specifies is discharge weighting is used
        """

        if top is None:
            top = self.extrap_fit.sel_fit[-1].top_method
        if bot is None:
            bot = self.extrap_fit.sel_fit[-1].bot_method
        if exp is None:
            exp = self.extrap_fit.sel_fit[-1].exponent
        if extents is not None:
            self.extrap_fit.subsection = extents
        if threshold is not None:
            self.extrap_fit.threshold = threshold

        data_type = self.extrap_fit.norm_data[-1].data_type
        if data_type is None:
            data_type = 'q'

        if method == 'Manual':
            self.extrap_fit.fit_method = 'Manual'
            for transect in self.transects:
                transect.extrap.set_extrap_data(top=top, bot=bot, exp=exp)
            self.extrap_fit.process_profiles(transects=self.transects, data_type=data_type, use_weighted=use_weighted)
        else:
            self.extrap_fit.fit_method = 'Automatic'
            self.extrap_fit.process_profiles(transects=self.transects, data_type=data_type, use_weighted=use_weighted)
            for transect in self.transects:
                transect.extrap.set_extrap_data(top=self.extrap_fit.sel_fit[-1].top_method,
                                                bot=self.extrap_fit.sel_fit[-1].bot_method,
                                                exp=self.extrap_fit.sel_fit[-1].exponent)

        if compute_q:
            self.extrap_fit.q_sensitivity = ExtrapQSensitivity()
            self.extrap_fit.q_sensitivity.populate_data(transects=self.transects,
                                                        extrap_fits=self.extrap_fit.sel_fit)

            self.compute_discharge()

    @staticmethod
    def measurement_duration(self):
        """Computes the duration of the measurement.
        """

        duration = 0
        for transect in self.transects:
            if transect.checked:
                duration += transect.date_time.transect_duration_sec
        return duration

    @staticmethod
    def mean_discharges(self):
        """Computes the mean discharge for the measurement.
        """

        # Initialize lists
        total_q = []
        uncorrected_q = []
        top_q = []
        bot_q = []
        mid_q = []
        left_q = []
        right_q = []
        int_cells_q = []
        int_ensembles_q = []

        for n, transect in enumerate(self.transects):
            if transect.checked:
                total_q.append(self.discharge[n].total)
                uncorrected_q.append(self.discharge[n].total_uncorrected)
                top_q.append(self.discharge[n].top)
                mid_q.append(self.discharge[n].middle)
                bot_q.append(self.discharge[n].bottom)
                left_q.append(self.discharge[n].left)
                right_q.append(self.discharge[n].right)
                int_cells_q.append(self.discharge[n].int_cells)
                int_ensembles_q.append(self.discharge[n].int_ens)

        discharge = {'total_mean': np.nanmean(total_q),
                     'uncorrected_mean': np.nanmean(uncorrected_q),
                     'top_mean': np.nanmean(top_q),
                     'mid_mean': np.nanmean(mid_q),
                     'bot_mean': np.nanmean(bot_q),
                     'left_mean': np.nanmean(left_q),
                     'right_mean': np.nanmean(right_q),
                     'int_cells_mean': np.nanmean(int_cells_q),
                     'int_ensembles_mean': np.nanmean(int_ensembles_q)}

        return discharge

    @staticmethod
    def compute_measurement_properties(self):
        """Computes characteristics of the transects and measurement that assist in evaluating the consistency
        of the transects.

        Returns
        -------
        trans_prop: dict
        Dictionary of transect properties
            width: float
                width in m
            width_cov: float
                coefficient of variation of width in percent
            area: float
                cross sectional area in m**2
            area_cov: float
                coefficient of variation of are in percent
            avg_boat_speed: float
                average boat speed in mps
            avg_boat_course: float
                average boat course in degrees
            avg_water_speed: float
                average water speed in mps
            avg_water_dir: float
                average water direction in degrees
            avg_depth: float
                average depth in m
            max_depth: float
                maximum depth in m
            max_water_speed: float
                99th percentile of water speed in mps
        """

        # Initialize variables
        checked_idx = np.array([], dtype=int)
        n_transects = len(self.transects)
        trans_prop = {'width': np.array([np.nan] * (n_transects + 1)),
                      'width_cov': np.array([np.nan] * (n_transects + 1)),
                      'area': np.array([np.nan] * (n_transects + 1)),
                      'area_cov': np.array([np.nan] * (n_transects + 1)),
                      'avg_boat_speed': np.array([np.nan] * (n_transects + 1)),
                      'avg_boat_course': np.array([np.nan] * n_transects),
                      'avg_water_speed': np.array([np.nan] * (n_transects + 1)),
                      'avg_water_dir': np.array([np.nan] * (n_transects + 1)),
                      'avg_depth': np.array([np.nan] * (n_transects + 1)),
                      'max_depth': np.array([np.nan] * (n_transects + 1)),
                      'max_water_speed': np.array([np.nan] * (n_transects + 1))}

        # Process each transect
        for n, transect in enumerate(self.transects):

            # Compute boat track properties
            boat_track = BoatStructure.compute_boat_track(transect)

            # Get boat speeds
            in_transect_idx = transect.in_transect_idx
            if getattr(transect.boat_vel, transect.boat_vel.selected) is not None:
                boat_selected = getattr(transect.boat_vel, transect.boat_vel.selected)
                u_boat = boat_selected.u_processed_mps[in_transect_idx]
                v_boat = boat_selected.v_processed_mps[in_transect_idx]
            else:
                u_boat = nans(transect.boat_vel.bt_vel.u_processed_mps[in_transect_idx].shape)
                v_boat = nans(transect.boat_vel.bt_vel.v_processed_mps[in_transect_idx].shape)

            if np.logical_not(np.all(np.isnan(boat_track['track_x_m']))):

                # Compute boat course and mean speed
                [course_radians, dmg] = cart2pol(boat_track['track_x_m'][-1], boat_track['track_y_m'][-1])
                trans_prop['avg_boat_course'][n] = rad2azdeg(course_radians)
                trans_prop['avg_boat_speed'][n] = np.nanmean(np.sqrt(u_boat**2 + v_boat**2))

                # Compute width
                trans_prop['width'][n] = np.nansum([dmg, transect.edges.left.distance_m,
                                                    transect.edges.right.distance_m])

                # Project the shiptrack onto a line from the beginning to end of the transect
                unit_x, unit_y = pol2cart(course_radians, 1)
                bt = np.array([boat_track['track_x_m'], boat_track['track_y_m']]).T
                dot_prod = bt @ np.array([unit_x, unit_y])
                projected_x = dot_prod * unit_x
                projected_y = dot_prod * unit_y
                station = np.sqrt(projected_x**2 + projected_y**2)

                # Get selected depth object
                depth = getattr(transect.depths, transect.depths.selected)
                depth_a = np.copy(depth.depth_processed_m)
                depth_a[np.isnan(depth_a)] = 0
                # Compute area of the moving-boat portion of the cross section using trapezoidal integration.
                # This method is consistent with AreaComp but is different from QRev in Matlab
                area_moving_boat = np.abs(np.trapz(depth_a[in_transect_idx], station[in_transect_idx]))

                # Compute area of left edge
                edge_type = transect.edges.left.type
                coef = 1
                if edge_type == 'Triangular':
                    coef = 0.5
                elif edge_type == 'Rectangular':
                    coef = 1.0
                elif edge_type == 'Custom':
                    coef = 0.5 + (transect.edges.left.cust_coef - 0.3535)
                elif edge_type == 'User Q':
                    coef = 0.5
                edge_idx = QComp.edge_ensembles('left', transect)
                edge_depth = np.nanmean(depth.depth_processed_m[edge_idx])
                area_left = edge_depth * transect.edges.left.distance_m * coef

                # Compute area of right edge
                edge_type = transect.edges.right.type
                if edge_type == 'Triangular':
                    coef = 0.5
                elif edge_type == 'Rectangular':
                    coef = 1.0
                elif edge_type == 'Custom':
                    coef = 0.5 + (transect.edges.right.cust_coef - 0.3535)
                elif edge_type == 'User Q':
                    coef = 0.5
                edge_idx = QComp.edge_ensembles('right', transect)
                edge_depth = np.nanmean(depth.depth_processed_m[edge_idx])
                area_right = edge_depth * transect.edges.right.distance_m * coef

                # Compute total cross sectional area
                trans_prop['area'][n] = np.nansum([area_left, area_moving_boat, area_right])

                # Compute average water speed
                trans_prop['avg_water_speed'][n] = self.discharge[n].total / trans_prop['area'][n]

                # Compute flow direction using discharge weighting
                u_water = transect.w_vel.u_processed_mps[:, in_transect_idx]
                v_water = transect.w_vel.v_processed_mps[:, in_transect_idx]
                weight = np.abs(self.discharge[n].middle_cells)
                u = np.nansum(np.nansum(u_water * weight)) / np.nansum(np.nansum(weight))
                v = np.nansum(np.nansum(v_water * weight)) / np.nansum(np.nansum(weight))
                trans_prop['avg_water_dir'][n] = np.arctan2(u, v) * 180 / np.pi
                if trans_prop['avg_water_dir'][n] < 0:
                    trans_prop['avg_water_dir'][n] = trans_prop['avg_water_dir'][n] + 360

                # Compute average and max depth
                # This is a deviation from QRev in Matlab which simply averaged all the depths
                trans_prop['avg_depth'][n] = trans_prop['area'][n] / trans_prop['width'][n]
                trans_prop['max_depth'][n] = np.nanmax(depth.depth_processed_m[in_transect_idx])

                # Compute max water speed using the 99th percentile
                water_speed = np.sqrt(u_water**2 + v_water**2)
                trans_prop['max_water_speed'][n] = np.nanpercentile(water_speed, 99)
                if transect.checked:
                    checked_idx = np.append(checked_idx, n)

            # Only transects used for discharge are included in measurement properties
            if len(checked_idx) > 0:
                n = n_transects
                trans_prop['width'][n] = np.nanmean(trans_prop['width'][checked_idx])
                trans_prop['width_cov'][n] = (np.nanstd(trans_prop['width'][checked_idx], ddof=1) /
                                              trans_prop['width'][n]) * 100
                trans_prop['area'][n] = np.nanmean(trans_prop['area'][checked_idx])
                trans_prop['area_cov'][n] = (np.nanstd(trans_prop['area'][checked_idx], ddof=1) /
                                             trans_prop['area'][n]) * 100
                trans_prop['avg_boat_speed'][n] = np.nanmean(trans_prop['avg_boat_speed'][checked_idx])
                trans_prop['avg_water_speed'][n] = np.nanmean(trans_prop['avg_water_speed'][checked_idx])
                trans_prop['avg_depth'][n] = np.nanmean(trans_prop['avg_depth'][checked_idx])
                trans_prop['max_depth'][n] = np.nanmax(trans_prop['max_depth'][checked_idx])
                trans_prop['max_water_speed'][n] = np.nanmax(trans_prop['max_water_speed'][checked_idx])

                # Compute average water direction using vector coordinates to avoid the problem of averaging
                # fluctuations that cross zero degrees
                x_coord = []
                y_coord = []
                for idx in checked_idx:
                    water_dir_rad = azdeg2rad(trans_prop['avg_water_dir'][idx])
                    x, y = pol2cart(water_dir_rad, 1)
                    x_coord.append(x)
                    y_coord.append(y)
                avg_water_dir_rad, _ = cart2pol(np.mean(x_coord), np.mean(y_coord))
                trans_prop['avg_water_dir'][n] = rad2azdeg(avg_water_dir_rad)

        return trans_prop

    @staticmethod
    def checked_transects(meas):
        """Create a list of indices of the checked transects.
        """

        checked_transect_idx = []
        for n in range(len(meas.transects)):
            if meas.transects[n].checked:
                checked_transect_idx.append(n)
        return checked_transect_idx

    @staticmethod
    def compute_time_series(meas, variable=None):
        """Computes the time series using serial time for any variable.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        variable: np.ndarray()
            Data for which the time series is requested
        """

        # Initialize variables
        data = np.array([])
        serial_time = np.array([])
        idx_transects = Measurement.checked_transects(meas)

        # Process transects
        for idx in idx_transects:
            if variable == 'Temperature':
                data = np.append(data, meas.transects[idx].sensors.temperature_deg_c.internal.data)
            ens_cum_time = np.nancumsum(meas.transects[idx].date_time.ens_duration_sec)
            ens_time = meas.transects[idx].date_time.start_serial_time + ens_cum_time
            serial_time = np.append(serial_time, ens_time)

        return data, serial_time

    def xml_output(self, version, file_name):
        channel = ETree.Element('Channel', QRevFilename=os.path.basename(file_name[:-4]), QRevVersion=version)

        # (2) SiteInformation Node
        if self.station_name or self.station_number:
            site_info = ETree.SubElement(channel, 'SiteInformation')

            # (3) StationName Node
            if self.station_name:
                ETree.SubElement(site_info, 'StationName', type='char').text = self.station_name

            # (3) SiteID Node
            if type(self.station_number) is str:
                ETree.SubElement(site_info, 'SiteID', type='char').text = self.station_number
            else:
                ETree.SubElement(site_info, 'SiteID', type='char').text = str(self.station_number)

        # (2) QA Node
        qa = ETree.SubElement(channel, 'QA')

        # (3) DiagnosticTestResult Node
        if len(self.system_tst) > 0:
            last_test = self.system_tst[-1].data
            failed_idx = last_test.count('FAIL')
            if failed_idx == 0:
                test_result = 'Pass'
            else:
                test_result = str(failed_idx) + ' Failed'
        else:
            test_result = 'None'
        ETree.SubElement(qa, 'DiagnosticTestResult', type='char').text = test_result

        # (3) CompassCalibrationResult Node
        try:
            last_eval = self.compass_eval[-1]
            # StreamPro, RR
            idx = last_eval.data.find('Typical Heading Error: <')
            if idx == (-1):
                # Rio Grande
                idx = last_eval.data.find('>>> Total error:')
                if idx != (-1):
                    idx_start = idx + 17
                    idx_end = idx_start + 10
                    comp_error = last_eval.data[idx_start:idx_end]
                    comp_error = ''.join([n for n in comp_error if n.isdigit() or n == '.'])
                else:
                    comp_error = ''
            else:
                # StreamPro, RR
                idx_start = idx + 24
                idx_end = idx_start + 10
                comp_error = last_eval.data[idx_start:idx_end]
                comp_error = ''.join([n for n in comp_error if n.isdigit() or n == '.'])

            # Evaluation could not be determined
            if not comp_error:
                ETree.SubElement(qa, 'CompassCalibrationResult', type='char').text = 'Yes'
            elif comp_error == '':
                ETree.SubElement(qa, 'CompassCalibrationResult', type='char').text = 'No'
            else:
                ETree.SubElement(qa, 'CompassCalibrationResult', type='char').text = 'Max ' + comp_error

        except (IndexError, TypeError, AttributeError):
            try:
                if len(self.compass_eval) > 0:
                    ETree.SubElement(qa, 'CompassCalibrationResult', type='char').text = 'Yes'
                else:
                    ETree.SubElement(qa, 'CompassCalibrationResult', type='char').text = 'No'
            except (IndexError, TypeError):
                ETree.SubElement(qa, 'CompassCalibrationResult', type='char').text = 'No'

        # (3) MovingBedTestType Node
        if not self.mb_tests:
            ETree.SubElement(qa, 'MovingBedTestType', type='char').text = 'None'
        else:
            selected_idx = [i for (i, val) in enumerate(self.mb_tests) if val.selected is True]
            if len(selected_idx) >= 1:
                temp = self.mb_tests[selected_idx[0]].type
            else:
                temp = self.mb_tests[-1].type
            ETree.SubElement(qa, 'MovingBedTestType', type='char').text = str(temp)

            # MovingBedTestResult Node
            temp = 'Unknown'
            for idx in selected_idx:
                if self.mb_tests[idx].moving_bed == 'Yes':
                    temp = 'Yes'
                    break
                elif self.mb_tests[idx].moving_bed == 'No':
                    temp = 'No'

            ETree.SubElement(qa, 'MovingBedTestResult', type='char').text = temp

        # (3) DiagnosticTest and Text Node
        if self.system_tst:
            test_text = ''
            for test in self.system_tst:
                test_text += test.data
            diag_test = ETree.SubElement(qa, 'DiagnosticTest')
            ETree.SubElement(diag_test, 'Text', type='char').text = test_text

        # (3) CompassCalibration and Text Node
        compass_text = ''
        try:
            for each in self.compass_cal:
                if self.transects[self.checked_transect_idx[0]].adcp.manufacturer == 'SonTek':
                    idx = each.data.find('CAL_TIME')
                    compass_text += each.data[idx:]
                else:
                    compass_text += each.data
        except (IndexError, TypeError, AttributeError):
            pass
        try:
            for each in self.compass_eval:
                if self.transects[self.checked_transect_idx[0]].adcp.manufacturer == 'SonTek':
                    idx = each.data.find('CAL_TIME')
                    compass_text += each.data[idx:]
                else:
                    compass_text += each.data
        except (IndexError, TypeError, AttributeError):
            pass

        if len(compass_text) > 0:
            comp_cal = ETree.SubElement(qa, 'CompassCalibration')
            ETree.SubElement(comp_cal, 'Text', type='char').text = compass_text

        # (3) MovingBedTest Node
        if self.mb_tests:
            for each in self.mb_tests:
                mbt = ETree.SubElement(qa, 'MovingBedTest')

                # (4) Filename Node
                ETree.SubElement(mbt, 'Filename', type='char').text = each.transect.file_name

                # (4) TestType Node
                ETree.SubElement(mbt, 'TestType', type='char').text = each.type

                # (4) Duration Node
                ETree.SubElement(mbt, 'Duration', type='double',
                                 unitsCode='sec').text = '{:.2f}'.format(each.duration_sec)

                # (4) PercentInvalidBT Node
                ETree.SubElement(mbt, 'PercentInvalidBT', type='double').text = '{:.4f}'.format(each.percent_invalid_bt)

                # (4) HeadingDifference Node
                if each.compass_diff_deg:
                    temp = '{:.2f}'.format(each.compass_diff_deg)
                else:
                    temp = ''
                ETree.SubElement(mbt, 'HeadingDifference', type='double', unitsCode='deg').text = temp

                # (4) MeanFlowDirection Node
                if each.flow_dir:
                    temp = '{:.2f}'.format(each.flow_dir)
                else:
                    temp = ''
                ETree.SubElement(mbt, 'MeanFlowDirection', type='double', unitsCode='deg').text = temp

                # (4) MovingBedDirection Node
                if each.mb_dir:
                    temp = '{:.2f}'.format(each.mb_dir)
                else:
                    temp = ''
                ETree.SubElement(mbt, 'MovingBedDirection', type='double', unitsCode='deg').text = temp

                # (4) DistanceUpstream Node
                ETree.SubElement(mbt, 'DistanceUpstream', type='double', unitsCode='m').text = \
                    '{:.4f}'.format(each.dist_us_m)

                # (4) MeanFlowSpeed Node
                ETree.SubElement(mbt, 'MeanFlowSpeed', type='double', unitsCode='mps').text = \
                    '{:.4f}'.format(each.flow_spd_mps)

                # (4) MovingBedSpeed Node
                ETree.SubElement(mbt, 'MovingBedSpeed', type='double', unitsCode='mps').text = \
                    '{:.4f}'.format(each.mb_spd_mps)

                # (4) PercentMovingBed Node
                ETree.SubElement(mbt, 'PercentMovingBed', type='double').text = '{:.2f}'.format(each.percent_mb)

                # (4) TestQuality Node
                ETree.SubElement(mbt, 'TestQuality', type='char').text = each.test_quality

                # (4) MovingBedPresent Node
                ETree.SubElement(mbt, 'MovingBedPresent', type='char').text = each.moving_bed

                # (4) UseToCorrect Node
                if each.use_2_correct:
                    ETree.SubElement(mbt, 'UseToCorrect', type='char').text = 'Yes'
                else:
                    ETree.SubElement(mbt, 'UseToCorrect', type='char').text = 'No'

                # (4) UserValid Node
                if each.user_valid:
                    ETree.SubElement(mbt, 'UserValid', type='char').text = 'Yes'
                else:
                    ETree.SubElement(mbt, 'UserValid', type='char').text = 'No'

                # (4) Message Node
                if len(each.messages) > 0:
                    str_out = ''
                    for message in each.messages:
                        str_out = str_out + message + '; '
                    ETree.SubElement(mbt, 'Message', type='char').text = str_out

        # (3) TemperatureCheck Node
        temp_check = ETree.SubElement(qa, 'TemperatureCheck')

        # (4) VerificationTemperature Node
        if not np.isnan(self.ext_temp_chk['user']):
            ETree.SubElement(temp_check, 'VerificationTemperature', type='double', unitsCode='degC').text = \
                '{:.2f}'.format(self.ext_temp_chk['user'])

        # (4) InstrumentTemperature Node
        if not np.isnan(self.ext_temp_chk['adcp']):
            ETree.SubElement(temp_check, 'InstrumentTemperature', type='double',
                             unitsCode='degC').text = '{:.2f}'.format(self.ext_temp_chk['adcp'])

        # (4) TemperatureChange Node:
        temp_all = np.array([np.nan])
        for each in self.transects:
            # Check for situation where user has entered a constant temperature
            temperature_selected = getattr(each.sensors.temperature_deg_c, each.sensors.temperature_deg_c.selected)
            temperature = temperature_selected.data
            if each.sensors.temperature_deg_c.selected != 'user':
                # Temperatures for ADCP.
                temp_all = np.concatenate((temp_all, temperature))
            else:
                # User specified constant temperature.
                # Concatenate a matrix of size of internal data with repeated user values.
                user_arr = np.tile(each.sensors.temperature_deg_c.user.data,
                                   (np.size(each.sensors.temperature_deg_c.internal.data)))
                temp_all = np.concatenate((temp_all, user_arr))

        t_range = np.nanmax(temp_all) - np.nanmin(temp_all)
        ETree.SubElement(temp_check, 'TemperatureChange', type='double',
                         unitsCode='degC').text = '{:.2f}'.format(t_range)

        # (3) QRev_Message Node
        qa_check_keys = ['bt_vel', 'compass', 'depths', 'edges', 'extrapolation', 'gga_vel', 'movingbed', 'system_tst',
                         'temperature', 'transects', 'user', 'vtg_vel', 'w_vel']

        # For each qa check retrieve messages
        messages = []
        for key in qa_check_keys:
            qa_type = getattr(self.qa, key)
            if qa_type['messages']:
                for message in qa_type['messages']:
                    if type(message) is str:
                        if message[:3].isupper():
                            messages.append([message, 1])
                        else:
                            messages.append([message, 2])
                    else:
                        messages.append(message)

        # Sort messages with warning at top
        messages.sort(key=lambda x: x[1])

        if len(messages) > 0:
            temp = ''
            for message in messages:
                temp = temp + message[0]
            ETree.SubElement(qa, 'QRev_Message', type='char').text = temp

        # (2) Instrument Node
        instrument = ETree.SubElement(channel, 'Instrument')

        # (3) Manufacturer Node
        ETree.SubElement(instrument, 'Manufacturer', type='char').text = \
            self.transects[self.checked_transect_idx[0]].adcp.manufacturer

        # (3) Model Node
        ETree.SubElement(instrument, 'Model', type='char').text = \
            self.transects[self.checked_transect_idx[0]].adcp.model

        # (3) SerialNumber Node
        sn = self.transects[self.checked_transect_idx[0]].adcp.serial_num
        ETree.SubElement(instrument, 'SerialNumber', type='char').text = str(sn)

        # (3) FirmwareVersion Node
        ver = self.transects[self.checked_transect_idx[0]].adcp.firmware
        ETree.SubElement(instrument, 'FirmwareVersion', type='char').text = str(ver)

        # (3) Frequency Node
        freq = self.transects[self.checked_transect_idx[0]].adcp.frequency_khz
        if type(freq) == np.ndarray:
            freq = "Multi"
        ETree.SubElement(instrument, 'Frequency', type='char', unitsCode='kHz').text = str(freq)

        # (3) BeamAngle Node
        ang = self.transects[self.checked_transect_idx[0]].adcp.beam_angle_deg
        ETree.SubElement(instrument, 'BeamAngle', type='double', unitsCode='deg').text = '{:.1f}'.format(ang)

        # (3) BlankingDistance Node
        w_vel = []
        for each in self.transects:
            w_vel.append(each.w_vel)
        blank = []
        for each in w_vel:
            blank.append(each.blanking_distance_m)
        if isinstance(blank[0], float):
            temp = np.mean(blank)
            if self.transects[self.checked_transect_idx[0]].w_vel.excluded_dist_m > temp:
                temp = self.transects[self.checked_transect_idx[0]].w_vel.excluded_dist_m
        else:
            temp = self.transects[self.checked_transect_idx[0]].w_vel.excluded_dist_m
        ETree.SubElement(instrument, 'BlankingDistance', type='double', unitsCode='m').text = '{:.4f}'.format(temp)

        # (3) InstrumentConfiguration Node
        commands = ''
        if self.transects[self.checked_transect_idx[0]].adcp.configuration_commands is not None:
            for each in self.transects[self.checked_transect_idx[0]].adcp.configuration_commands:
                if type(each) is str:
                    commands += each + '  '
            ETree.SubElement(instrument, 'InstrumentConfiguration', type='char').text = commands

        # (2) Processing Node
        processing = ETree.SubElement(channel, 'Processing')

        # (3) SoftwareVersion Node
        ETree.SubElement(processing, 'SoftwareVersion', type='char').text = version

        # (3) Type Node
        ETree.SubElement(processing, 'Type', type='char').text = self.processing

        # (3) AreaComputationMethod Node
        ETree.SubElement(processing, 'AreaComputationMethod', type='char').text = 'Parallel'

        # (3) Navigation Node
        navigation = ETree.SubElement(processing, 'Navigation')

        # (4) Reference Node
        ETree.SubElement(navigation, 'Reference', type='char').text = \
            self.transects[self.checked_transect_idx[0]].w_vel.nav_ref

        # (4) CompositeTrack
        ETree.SubElement(navigation, 'CompositeTrack', type='char').text = \
            self.transects[self.checked_transect_idx[0]].boat_vel.composite

        # (4) MagneticVariation Node
        mag_var = self.transects[self.checked_transect_idx[0]].sensors.heading_deg.internal.mag_var_deg
        ETree.SubElement(navigation, 'MagneticVariation', type='double',
                         unitsCode='deg').text = '{:.2f}'.format(mag_var)

        # (4) BeamFilter
        nav_data = getattr(self.transects[self.checked_transect_idx[0]].boat_vel,
                           self.transects[self.checked_transect_idx[0]].boat_vel.selected)
        temp = nav_data.beam_filter
        if temp < 0:
            temp = 'Auto'
        else:
            temp = str(temp)
        ETree.SubElement(navigation, 'BeamFilter', type='char').text = temp

        # (4) ErrorVelocityFilter Node
        evf = nav_data.d_filter
        if evf == 'Manual':
            evf = '{:.4f}'.format(nav_data.d_filter_threshold)
        ETree.SubElement(navigation, 'ErrorVelocityFilter', type='char', unitsCode='mps').text = evf

        # (4) VerticalVelocityFilter Node
        vvf = nav_data.w_filter
        if vvf == 'Manual':
            vvf = '{:.4f}'.format(nav_data.w_filter_threshold)
        ETree.SubElement(navigation, 'VerticalVelocityFilter', type='char', unitsCode='mps').text = vvf

        # (4) OtherFilter Node
        o_f = nav_data.smooth_filter
        ETree.SubElement(navigation, 'OtherFilter', type='char').text = o_f

        # (4) GPSDifferentialQualityFilter Node
        temp = nav_data.gps_diff_qual_filter
        if temp:
            if isinstance(temp, int) or isinstance(temp, float):
                temp = str(temp)
            ETree.SubElement(navigation, 'GPSDifferentialQualityFilter', type='char').text = temp

        # (4) GPSAltitudeFilter Node
        temp = nav_data.gps_altitude_filter
        if temp:
            if temp == 'Manual':
                temp = self.transects[self.checked_transect_idx[0]].boat_vel.gps_altitude_filter_change
            ETree.SubElement(navigation, 'GPSAltitudeFilter', type='char', unitsCode='m').text = str(temp)

        # (4) HDOPChangeFilter
        temp = nav_data.gps_HDOP_filter
        if temp:
            if temp == 'Manual':
                temp = '{:.2f}'.format(self.transects[self.checked_transect_idx[0]].boat_vel.gps_hdop_filter_change)
            ETree.SubElement(navigation, 'HDOPChangeFilter', type='char').text = temp

        # (4) HDOPThresholdFilter
        temp = nav_data.gps_HDOP_filter
        if temp:
            if temp == 'Manual':
                temp = '{:.2f}'.format(self.transects[self.checked_transect_idx[0]].boat_vel.gps_HDOP_filter_max)
            ETree.SubElement(navigation, 'HDOPThresholdFilter', type='char').text = temp

        # (4) InterpolationType Node
        temp = nav_data.interpolate
        ETree.SubElement(navigation, 'InterpolationType', type='char').text = temp

        # (3) Depth Node
        depth = ETree.SubElement(processing, 'Depth')

        # (4) Reference Node
        if self.transects[self.checked_transect_idx[0]].depths.selected == 'bt_depths':
            temp = 'BT'
        elif self.transects[self.checked_transect_idx[0]].depths.selected == 'vb_depths':
            temp = 'VB'
        elif self.transects[self.checked_transect_idx[0]].depths.selected == 'ds_depths':
            temp = 'DS'
        ETree.SubElement(depth, 'Reference', type='char').text = temp

        # (4) CompositeDepth Node
        ETree.SubElement(depth, 'CompositeDepth', type='char').text = \
            self.transects[self.checked_transect_idx[0]].depths.composite

        # (4) ADCPDepth Node
        depth_data = getattr(self.transects[self.checked_transect_idx[0]].depths,
                             self.transects[self.checked_transect_idx[0]].depths.selected)
        temp = depth_data.draft_use_m
        ETree.SubElement(depth, 'ADCPDepth', type='double', unitsCode='m').text = '{:.4f}'.format(temp)

        # (4) ADCPDepthConsistent Node
        drafts = []
        for transect in self.transects:
            if transect.checked:
                transect_depth = getattr(transect.depths, transect.depths.selected)
                drafts.append(transect_depth.draft_use_m)
        unique_drafts = set(drafts)
        num_drafts = len(unique_drafts)
        if num_drafts > 1:
            temp = 'No'
        else:
            temp = 'Yes'
        ETree.SubElement(depth, 'ADCPDepthConsistent', type='boolean').text = temp

        # (4) FilterType Node
        temp = depth_data.filter_type
        ETree.SubElement(depth, 'FilterType', type='char').text = temp

        # (4) InterpolationType Node
        temp = depth_data.interp_type
        ETree.SubElement(depth, 'InterpolationType', type='char').text = temp

        # (4) AveragingMethod Node
        temp = depth_data.avg_method
        ETree.SubElement(depth, 'AveragingMethod', type='char').text = temp

        # (4) ValidDataMethod Node
        temp = depth_data.valid_data_method
        ETree.SubElement(depth, 'ValidDataMethod', type='char').text = temp

        # (3) WaterTrack Node
        water_track = ETree.SubElement(processing, 'WaterTrack')

        # (4) ExcludedDistance Node
        temp = self.transects[self.checked_transect_idx[0]].w_vel.excluded_dist_m
        ETree.SubElement(water_track, 'ExcludedDistance', type='double', unitsCode='m').text = '{:.4f}'.format(temp)

        # (4) BeamFilter Node
        temp = self.transects[self.checked_transect_idx[0]].w_vel.beam_filter
        if temp < 0:
            temp = 'Auto'
        else:
            temp = str(temp)
        ETree.SubElement(water_track, 'BeamFilter', type='char').text = temp

        # (4) ErrorVelocityFilter Node
        temp = self.transects[self.checked_transect_idx[0]].w_vel.d_filter
        if temp == 'Manual':
            temp = '{:.4f}'.format(self.transects[self.checked_transect_idx[0]].w_vel.d_filter_threshold)
        ETree.SubElement(water_track, 'ErrorVelocityFilter', type='char', unitsCode='mps').text = temp

        # (4) VerticalVelocityFilter Node
        temp = self.transects[self.checked_transect_idx[0]].w_vel.w_filter
        if temp == 'Manual':
            temp = '{:.4f}'.format(self.transects[self.checked_transect_idx[0]].w_vel.w_filter_threshold)
        ETree.SubElement(water_track, 'VerticalVelocityFilter', type='char', unitsCode='mps').text = temp

        # (4) OtherFilter Node
        temp = self.transects[self.checked_transect_idx[0]].w_vel.smooth_filter
        ETree.SubElement(water_track, 'OtherFilter', type='char').text = temp

        # (4) SNRFilter Node
        temp = self.transects[self.checked_transect_idx[0]].w_vel.snr_filter
        ETree.SubElement(water_track, 'SNRFilter', type='char').text = temp

        # (4) CellInterpolation Node
        temp = self.transects[self.checked_transect_idx[0]].w_vel.interpolate_cells
        ETree.SubElement(water_track, 'CellInterpolation', type='char').text = temp

        # (4) EnsembleInterpolation Node
        temp = self.transects[self.checked_transect_idx[0]].w_vel.interpolate_ens
        ETree.SubElement(water_track, 'EnsembleInterpolation', type='char').text = temp

        # (3) Edge Node
        edge = ETree.SubElement(processing, 'Edge')

        # (4) RectangularEdgeMethod Node
        temp = self.transects[self.checked_transect_idx[0]].edges.rec_edge_method
        ETree.SubElement(edge, 'RectangularEdgeMethod', type='char').text = temp

        # (4) VelocityMethod Node
        temp = self.transects[self.checked_transect_idx[0]].edges.vel_method
        ETree.SubElement(edge, 'VelocityMethod', type='char').text = temp

        # (4) LeftType Node
        typ = []
        for n in self.transects:
            if n.checked:
                typ.append(n.edges.left.type)
        unique_type = set(typ)
        num_types = len(unique_type)
        if num_types > 1:
            temp = 'Varies'
        else:
            temp = typ[0]
        ETree.SubElement(edge, 'LeftType', type='char').text = temp

        # LeftEdgeCoefficient
        if temp == 'User Q':
            temp = 'N/A'
        elif temp == 'Varies':
            temp = 'N/A'
        else:
            coef = []
            for transect in self.transects:
                if transect.checked:
                    coef.append(QComp.edge_coef('left', transect))
            num_coef = len(set(coef))
            if num_coef > 1:
                temp = 'Varies'
            else:
                temp = '{:.4f}'.format(coef[0])
        ETree.SubElement(edge, 'LeftEdgeCoefficient', type='char').text = temp

        # (4) RightType Node
        typ = []
        for n in self.transects:
            if n.checked:
                typ.append(n.edges.right.type)
        unique_type = set(typ)
        num_types = len(unique_type)
        if num_types > 1:
            temp = 'Varies'
        else:
            temp = typ[0]
        ETree.SubElement(edge, 'RightType', type='char').text = temp

        # RightEdgeCoefficient
        if temp == 'User Q':
            temp = 'N/A'
        elif temp == 'Varies':
            temp = 'N/A'
        else:
            coef = []
            for transect in self.transects:
                if transect.checked:
                    coef.append(QComp.edge_coef('right', transect))
            num_coef = len(set(coef))
            if num_coef > 1:
                temp = 'Varies'
            else:
                temp = '{:.4f}'.format(coef[0])
        ETree.SubElement(edge, 'RightEdgeCoefficient', type='char').text = temp

        # (3) Extrapolation Node
        extrap = ETree.SubElement(processing, 'Extrapolation')

        # (4) TopMethod Node
        temp = self.transects[self.checked_transect_idx[0]].extrap.top_method
        ETree.SubElement(extrap, 'TopMethod', type='char').text = temp

        # (4) BottomMethod Node
        temp = self.transects[self.checked_transect_idx[0]].extrap.bot_method
        ETree.SubElement(extrap, 'BottomMethod', type='char').text = temp

        # (4) Exponent Node
        temp = self.transects[self.checked_transect_idx[0]].extrap.exponent
        ETree.SubElement(extrap, 'Exponent', type='double').text = '{:.4f}'.format(temp)

        # (4) Discharge weighted medians
        temp = self.extrap_fit.use_weighted
        if temp:
            temp = 'Yes'
        else:
            temp = 'No'
        ETree.SubElement(extrap, 'UseWeighted', type='char').text = temp

        # (3) Sensor Node
        sensor = ETree.SubElement(processing, 'Sensor')

        # (4) TemperatureSource Node
        temp = []
        for n in self.transects:
            if n.checked:
                # k+=1
                temp.append(n.sensors.temperature_deg_c.selected)
        sources = len(set(temp))
        if sources > 1:
            temp = 'Varies'
        else:
            temp = temp[0]
        ETree.SubElement(sensor, 'TemperatureSource', type='char').text = temp

        # (4) Salinity
        temp = np.array([])
        for transect in self.transects:
            if transect.checked:
                sal_selected = getattr(transect.sensors.salinity_ppt, transect.sensors.salinity_ppt.selected)
                temp = np.append(temp, sal_selected.data)
        values = np.unique(temp)
        if len(values) > 1:
            temp = 'Varies'
        else:
            temp = '{:2.1f}'.format(values[0])
        ETree.SubElement(sensor, 'Salinity', type='char', unitsCode='ppt').text = temp

        # (4) SpeedofSound Node
        temp = []
        for n in self.transects:
            if n.checked:
                temp.append(n.sensors.speed_of_sound_mps.selected)
        sources = len(set(temp))
        if sources > 1:
            temp = 'Varies'
        else:
            temp = temp[0]
        if temp == 'internal':
            temp = 'ADCP'
        ETree.SubElement(sensor, 'SpeedofSound', type='char', unitsCode='mps').text = temp

        # (2) Transect Node
        other_prop = self.compute_measurement_properties(self)
        for n in range(len(self.transects)):
            if self.transects[n].checked:
                transect = ETree.SubElement(channel, 'Transect')

                # (3) Filename Node
                temp = self.transects[n].file_name
                ETree.SubElement(transect, 'Filename', type='char').text = temp

                # (3) StartDateTime Node
                temp = int(self.transects[n].date_time.start_serial_time)
                temp = datetime.datetime.utcfromtimestamp(temp).strftime('%m/%d/%Y %H:%M:%S')
                ETree.SubElement(transect, 'StartDateTime', type='char').text = temp

                # (3) EndDateTime Node
                temp = int(self.transects[n].date_time.end_serial_time)
                temp = datetime.datetime.utcfromtimestamp(temp).strftime('%m/%d/%Y %H:%M:%S')
                ETree.SubElement(transect, 'EndDateTime', type='char').text = temp

                # (3) Discharge Node
                t_q = ETree.SubElement(transect, 'Discharge')

                # (4) Top Node
                temp = self.discharge[n].top
                ETree.SubElement(t_q, 'Top', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

                # (4) Middle Node
                temp = self.discharge[n].middle
                ETree.SubElement(t_q, 'Middle', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

                # (4) Bottom Node
                temp = self.discharge[n].bottom
                ETree.SubElement(t_q, 'Bottom', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

                # (4) Left Node
                temp = self.discharge[n].left
                ETree.SubElement(t_q, 'Left', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

                # (4) Right Node
                temp = self.discharge[n].right
                ETree.SubElement(t_q, 'Right', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

                # (4) Total Node
                temp = self.discharge[n].total
                ETree.SubElement(t_q, 'Total', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

                # (4) MovingBedPercentCorrection Node
                temp = ((self.discharge[n].total / self.discharge[n].total_uncorrected) - 1) * 100
                ETree.SubElement(t_q, 'MovingBedPercentCorrection', type='double').text = '{:.2f}'.format(temp)

                # (3) Edge Node
                t_edge = ETree.SubElement(transect, 'Edge')

                # (4) StartEdge Node
                temp = self.transects[n].start_edge
                ETree.SubElement(t_edge, 'StartEdge', type='char').text = temp

                # (4) RectangularEdgeMethod Node
                temp = self.transects[n].edges.rec_edge_method
                ETree.SubElement(t_edge, 'RectangularEdgeMethod', type='char').text = temp

                # (4) VelocityMethod Node
                temp = self.transects[n].edges.vel_method
                ETree.SubElement(t_edge, 'VelocityMethod', type='char').text = temp

                # (4) LeftType Node
                temp = self.transects[n].edges.left.type
                ETree.SubElement(t_edge, 'LeftType', type='char').text = temp

                # (4) LeftEdgeCoefficient Node
                if temp == 'User Q':
                    temp = ''
                else:
                    temp = '{:.4f}'.format(QComp.edge_coef('left', self.transects[n]))
                ETree.SubElement(t_edge, 'LeftEdgeCoefficient', type='double').text = temp

                # (4) LeftDistance Node
                temp = '{:.4f}'.format(self.transects[n].edges.left.distance_m)
                ETree.SubElement(t_edge, 'LeftDistance', type='double', unitsCode='m').text = temp

                # (4) LeftNumberEnsembles
                temp = '{:.0f}'.format(self.transects[n].edges.left.number_ensembles)
                ETree.SubElement(t_edge, 'LeftNumberEnsembles', type='double').text = temp

                # (4) RightType Node
                temp = self.transects[n].edges.right.type
                ETree.SubElement(t_edge, 'RightType', type='char').text = temp

                # (4) RightEdgeCoefficient Node
                if temp == 'User Q':
                    temp = ''
                else:
                    temp = '{:.4f}'.format(QComp.edge_coef('right', self.transects[n]))
                ETree.SubElement(t_edge, 'RightEdgeCoefficient', type='double').text = temp

                # (4) RightDistance Node
                temp = '{:.4f}'.format(self.transects[n].edges.right.distance_m)
                ETree.SubElement(t_edge, 'RightDistance', type='double', unitsCode='m').text = temp

                # (4) RightNumberEnsembles Node
                temp = '{:.0f}'.format(self.transects[n].edges.right.number_ensembles)
                ETree.SubElement(t_edge, 'RightNumberEnsembles', type='double').text = temp

                # (3) Sensor Node
                t_sensor = ETree.SubElement(transect, 'Sensor')

                # (4) TemperatureSource Node
                temp = self.transects[n].sensors.temperature_deg_c.selected
                ETree.SubElement(t_sensor, 'TemperatureSource', type='char').text = temp

                # (4) MeanTemperature Node
                dat = getattr(self.transects[n].sensors.temperature_deg_c,
                              self.transects[n].sensors.temperature_deg_c.selected)
                temp = np.nanmean(dat.data)
                temp = '{:.2f}'.format(temp)
                ETree.SubElement(t_sensor, 'MeanTemperature', type='double', unitsCode='degC').text = temp

                # (4) MeanSalinity
                sal_data = getattr(self.transects[n].sensors.salinity_ppt,
                                   self.transects[n].sensors.salinity_ppt.selected)
                temp = '{:.0f}'.format(np.nanmean(sal_data.data))
                ETree.SubElement(t_sensor, 'MeanSalinity', type='double', unitsCode='ppt').text = temp

                # (4) SpeedofSoundSource Node
                sos_selected = getattr(self.transects[n].sensors.speed_of_sound_mps,
                                       self.transects[n].sensors.speed_of_sound_mps.selected)
                temp = sos_selected.source
                ETree.SubElement(t_sensor, 'SpeedofSoundSource', type='char').text = temp

                # (4) SpeedofSound
                sos_data = getattr(self.transects[n].sensors.speed_of_sound_mps,
                                   self.transects[n].sensors.speed_of_sound_mps.selected)
                temp = '{:.4f}'.format(np.nanmean(sos_data.data))
                ETree.SubElement(t_sensor, 'SpeedofSound', type='double', unitsCode='mps').text = temp

                # (3) Other Node
                t_other = ETree.SubElement(transect, 'Other')

                # (4) Duration Node
                temp = '{:.2f}'.format(self.transects[n].date_time.transect_duration_sec)
                ETree.SubElement(t_other, 'Duration', type='double', unitsCode='sec').text = temp

                # (4) Width
                temp = other_prop['width'][n]
                ETree.SubElement(t_other, 'Width', type='double', unitsCode='m').text = '{:.4f}'.format(temp)

                # (4) Area
                temp = other_prop['area'][n]
                ETree.SubElement(t_other, 'Area', type='double', unitsCode='sqm').text = '{:.4f}'.format(temp)

                # (4) MeanBoatSpeed
                temp = other_prop['avg_boat_speed'][n]
                ETree.SubElement(t_other, 'MeanBoatSpeed', type='double', unitsCode='mps').text = '{:.4f}'.format(temp)

                # (4) QoverA
                temp = other_prop['avg_water_speed'][n]
                ETree.SubElement(t_other, 'QoverA', type='double', unitsCode='mps').text = '{:.4f}'.format(temp)

                # (4) CourseMadeGood
                temp = other_prop['avg_boat_course'][n]
                ETree.SubElement(t_other, 'CourseMadeGood', type='double', unitsCode='deg').text = '{:.2f}'.format(temp)

                # (4) MeanFlowDirection
                temp = other_prop['avg_water_dir'][n]
                ETree.SubElement(t_other, 'MeanFlowDirection', type='double',
                                 unitsCode='deg').text = '{:.2f}'.format(temp)

                # (4) NumberofEnsembles
                temp = len(self.transects[n].boat_vel.bt_vel.u_processed_mps)
                ETree.SubElement(t_other, 'NumberofEnsembles', type='integer').text = str(temp)

                # (4) PercentInvalidBins
                valid_ens, valid_cells = TransectData.raw_valid_data(self.transects[n])
                temp = (1 - (np.nansum(np.nansum(valid_cells))
                             / np.nansum(np.nansum(self.transects[n].w_vel.cells_above_sl)))) * 100
                ETree.SubElement(t_other, 'PercentInvalidBins', type='double').text = '{:.2f}'.format(temp)

                # (4) PercentInvalidEnsembles
                temp = (1 - (np.nansum(valid_ens) / len(self.transects[n].boat_vel.bt_vel.u_processed_mps))) * 100
                ETree.SubElement(t_other, 'PercentInvalidEns', type='double').text = '{:.2f}'.format(temp)

                pitch_source_selected = getattr(self.transects[n].sensors.pitch_deg,
                                                self.transects[n].sensors.pitch_deg.selected)
                roll_source_selected = getattr(self.transects[n].sensors.roll_deg,
                                               self.transects[n].sensors.roll_deg.selected)

                # (4) MeanPitch
                temp = np.nanmean(pitch_source_selected.data)
                ETree.SubElement(t_other, 'MeanPitch', type='double', unitsCode='deg').text = '{:.2f}'.format(temp)

                # (4) MeanRoll
                temp = np.nanmean(roll_source_selected.data)
                ETree.SubElement(t_other, 'MeanRoll', type='double', unitsCode='deg').text = '{:.2f}'.format(temp)

                # (4) PitchStdDev
                temp = np.nanstd(pitch_source_selected.data, ddof=1)
                ETree.SubElement(t_other, 'PitchStdDev', type='double', unitsCode='deg').text = '{:.2f}'.format(temp)

                # (4) RollStdDev
                temp = np.nanstd(roll_source_selected.data, ddof=1)
                ETree.SubElement(t_other, 'RollStdDev', type='double', unitsCode='deg').text = '{:.2f}'.format(temp)

                # (4) ADCPDepth
                depth_source_selected = getattr(self.transects[n].depths,
                                                self.transects[n].depths.selected)
                temp = depth_source_selected.draft_use_m
                ETree.SubElement(t_other, 'ADCPDepth', type='double', unitsCode='m').text = '{:.4f}'.format(temp)

        # (2) ChannelSummary Node
        summary = ETree.SubElement(channel, 'ChannelSummary')

        # (3) Discharge Node
        s_q = ETree.SubElement(summary, 'Discharge')
        discharge = self.mean_discharges(self)

        # (4) Top
        temp = discharge['top_mean']
        ETree.SubElement(s_q, 'Top', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

        # (4) Middle
        temp = discharge['mid_mean']
        ETree.SubElement(s_q, 'Middle', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

        # (4) Bottom
        temp = discharge['bot_mean']
        ETree.SubElement(s_q, 'Bottom', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

        # (4) Left
        temp = discharge['left_mean']
        ETree.SubElement(s_q, 'Left', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

        # (4) Right
        temp = discharge['right_mean']
        ETree.SubElement(s_q, 'Right', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

        # (4) Total
        temp = discharge['total_mean']
        ETree.SubElement(s_q, 'Total', type='double', unitsCode='cms').text = '{:.5f}'.format(temp)

        # (4) MovingBedPercentCorrection
        temp = ((discharge['total_mean'] / discharge['uncorrected_mean']) - 1) * 100
        ETree.SubElement(s_q, 'MovingBedPercentCorrection', type='double').text = '{:.2f}'.format(temp)

        # (3) Uncertainty Node
        s_u = ETree.SubElement(summary, 'Uncertainty')
        uncertainty = self.uncertainty

        # (4) COV Node
        temp = uncertainty.cov
        if not np.isnan(temp):
            ETree.SubElement(s_u, 'COV', type='double').text = '{:.1f}'.format(temp)

        # (4) AutoRandom Node
        temp = uncertainty.cov_95
        if not np.isnan(temp):
            ETree.SubElement(s_u, 'AutoRandom', type='double').text = '{:.1f}'.format(temp)

        # (4) AutoInvalidData Node
        temp = uncertainty.invalid_95
        ETree.SubElement(s_u, 'AutoInvalidData', type='double').text = '{:.1f}'.format(temp)

        # (4) AutoEdge Node
        temp = uncertainty.edges_95
        ETree.SubElement(s_u, 'AutoEdge', type='double').text = '{:.1f}'.format(temp)

        # (4) AutoExtrapolation Node
        temp = uncertainty.extrapolation_95
        ETree.SubElement(s_u, 'AutoExtrapolation', type='double').text = '{:.1f}'.format(temp)

        # (4) AutoMovingBed
        temp = uncertainty.moving_bed_95
        ETree.SubElement(s_u, 'AutoMovingBed', type='double').text = '{:.1f}'.format(temp)

        # (4) AutoSystematic
        temp = uncertainty.systematic
        ETree.SubElement(s_u, 'AutoSystematic', type='double').text = '{:.1f}'.format(temp)

        # (4) AutoTotal
        temp = uncertainty.total_95
        if not np.isnan(temp):
            ETree.SubElement(s_u, 'AutoTotal', type='double').text = '{:.1f}'.format(temp)

        # (4) UserRandom Node
        user_random = uncertainty.cov_95_user
        if user_random:
            ETree.SubElement(s_u, 'UserRandom', type='double').text = '{:.1f}'.format(user_random)

        # (4) UserInvalidData Node
        user_invalid = uncertainty.invalid_95_user
        if user_invalid:
            ETree.SubElement(s_u, 'UserInvalidData', type='double').text = '{:.1f}'.format(user_invalid)

        # (4) UserEdge
        user_edge = uncertainty.edges_95_user
        if user_edge:
            ETree.SubElement(s_u, 'UserEdge', type='double').text = '{:.1f}'.format(user_edge)

        # (4) UserExtrapolation
        user_extrap = uncertainty.extrapolation_95_user
        if user_extrap:
            ETree.SubElement(s_u, 'UserExtrapolation', type='double').text = '{:.1f}'.format(user_extrap)

        # (4) UserMovingBed
        user_mb = uncertainty.moving_bed_95_user
        if user_mb:
            ETree.SubElement(s_u, 'UserMovingBed', type='double').text = '{:.1f}'.format(user_mb)

        # (4) UserSystematic
        user_systematic = uncertainty.systematic_user
        if user_systematic:
            ETree.SubElement(s_u, 'UserSystematic', type='double').text = '{:.1f}'.format(user_systematic)

        # (4) UserTotal Node
        temp = uncertainty.total_95_user
        if not np.isnan(temp):
            ETree.SubElement(s_u, 'UserTotal', type='double').text = '{:.1f}'.format(temp)

        # (4) Random
        if user_random:
            temp = user_random
        else:
            temp = uncertainty.cov_95
        if not np.isnan(temp):
            ETree.SubElement(s_u, 'Random', type='double').text = '{:.1f}'.format(temp)

        # (4) InvalidData
        if user_invalid:
            temp = user_invalid
        else:
            temp = uncertainty.invalid_95
        ETree.SubElement(s_u, 'InvalidData', type='double').text = '{:.1f}'.format(temp)

        # (4) Edge
        if user_edge:
            temp = user_edge
        else:
            temp = uncertainty.edges_95
        ETree.SubElement(s_u, 'Edge', type='double').text = '{:.1f}'.format(temp)

        # (4) Extrapolation
        if user_extrap:
            temp = user_extrap
        else:
            temp = uncertainty.extrapolation_95
        ETree.SubElement(s_u, 'Extrapolation', type='double').text = '{:.1f}'.format(temp)

        # (4) MovingBed
        if user_mb:
            temp = user_mb
        else:
            temp = uncertainty.moving_bed_95
        ETree.SubElement(s_u, 'MovingBed', type='double').text = '{:.1f}'.format(temp)

        # (4) Systematic
        if user_systematic:
            temp = user_systematic
        else:
            temp = uncertainty.systematic
        ETree.SubElement(s_u, 'Systematic', type='double').text = '{:.1f}'.format(temp)

        # (4) UserTotal Node
        temp = uncertainty.total_95_user
        if not np.isnan(temp):
            ETree.SubElement(s_u, 'Total', type='double').text = '{:.1f}'.format(temp)

        # (3) Other Node
        s_o = ETree.SubElement(summary, 'Other')

        # (4) MeanWidth
        temp = other_prop['width'][-1]
        ETree.SubElement(s_o, 'MeanWidth', type='double', unitsCode='m').text = '{:.4f}'.format(temp)

        # (4) WidthCOV
        temp = other_prop['width_cov'][-1]
        if not np.isnan(temp):
            ETree.SubElement(s_o, 'WidthCOV', type='double').text = '{:.4f}'.format(temp)

        # (4) MeanArea
        temp = other_prop['area'][-1]
        ETree.SubElement(s_o, 'MeanArea', type='double', unitsCode='sqm').text = '{:.4f}'.format(temp)

        # (4) AreaCOV
        temp = other_prop['area_cov'][-1]
        if not np.isnan(temp):
            ETree.SubElement(s_o, 'AreaCOV', type='double').text = '{:.2f}'.format(temp)

        # (4) MeanBoatSpeed
        temp = other_prop['avg_boat_speed'][-1]
        ETree.SubElement(s_o, 'MeanBoatSpeed', type='double', unitsCode='mps').text = '{:.4f}'.format(temp)

        # (4) MeanQoverA
        temp = other_prop['avg_water_speed'][-1]
        ETree.SubElement(s_o, 'MeanQoverA', type='double', unitsCode='mps').text = '{:.4f}'.format(temp)

        # (4) MeanCourseMadeGood
        temp = other_prop['avg_boat_course'][-1]
        ETree.SubElement(s_o, 'MeanCourseMadeGood', type='double', unitsCode='deg').text = '{:.2f}'.format(temp)

        # (4) MeanFlowDirection
        temp = other_prop['avg_water_dir'][-1]
        ETree.SubElement(s_o, 'MeanFlowDirection', type='double', unitsCode='deg').text = '{:.2f}'.format(temp)

        # (4) MeanDepth
        temp = other_prop['avg_depth'][-1]
        ETree.SubElement(s_o, 'MeanDepth', type='double', unitsCode='m').text = '{:.4f}'.format(temp)

        # (4) MaximumDepth
        temp = other_prop['max_depth'][-1]
        ETree.SubElement(s_o, 'MaximumDepth', type='double', unitsCode='m').text = '{:.4f}'.format(temp)

        # (4) MaximumWaterSpeed
        temp = other_prop['max_water_speed'][-1]
        ETree.SubElement(s_o, 'MaximumWaterSpeed', type='double', unitsCode='mps').text = '{:.4f}'.format(temp)

        # (4) NumberofTransects
        temp = len(self.checked_transects(self))
        ETree.SubElement(s_o, 'NumberofTransects', type='integer').text = str(temp)

        # (4) Duration
        temp = self.measurement_duration(self)
        ETree.SubElement(s_o, 'Duration', type='double', unitsCode='sec').text = '{:.2f}'.format(temp)

        # (4) LeftQPer
        temp = 100 * discharge['left_mean'] / discharge['total_mean']
        ETree.SubElement(s_o, 'LeftQPer', type='double').text = '{:.2f}'.format(temp)

        # (4) RightQPer
        temp = 100 * discharge['right_mean'] / discharge['total_mean']
        ETree.SubElement(s_o, 'RightQPer', type='double').text = '{:.2f}'.format(temp)

        # (4) InvalidCellsQPer
        temp = 100 * discharge['int_cells_mean'] / discharge['total_mean']
        ETree.SubElement(s_o, 'InvalidCellsQPer', type='double').text = '{:.2f}'.format(temp)

        # (4) InvalidEnsQPer
        temp = 100 * discharge['int_ensembles_mean'] / discharge['total_mean']
        ETree.SubElement(s_o, 'InvalidEnsQPer', type='double').text = '{:.2f}'.format(temp)

        # (4) UserRating
        if self.user_rating:
            temp = self.user_rating
        else:
            temp = 'Not Rated'
        ETree.SubElement(s_o, 'UserRating', type='char').text = temp

        # (4) DischargePPDefault
        temp = self.extrap_fit.q_sensitivity.q_pp_mean
        ETree.SubElement(s_o, 'DischargePPDefault', type='double').text = '{:.2f}'.format(temp)

        # (2) UserComment
        if len(self.comments) > 1:
            temp = ''
            for comment in self.comments:
                temp = temp + comment.replace('\n', ' |||') + ' |||'
            ETree.SubElement(channel, 'UserComment', type='char').text = temp

        # Create xml output file
        with open(file_name, 'wb') as xml_file:
            # Create binary coded output file
            et = ETree.ElementTree(channel)
            root = et.getroot()
            xml_out = ETree.tostring(root)
            # Add stylesheet instructions
            xml_out = b'<?xml-stylesheet type= "text/xsl" href="QRevStylesheet.xsl"?>' + xml_out
            # Add tabs to make output more readable and apply utf-8 encoding
            xml_out = parseString(xml_out).toprettyxml(encoding='utf-8')
            # Write file
            xml_file.write(xml_out)


if __name__ == '__main__':
    pass
