import re
import copy
import numpy as np


class PreMeasurement(object):
    """Stores tests, calibrations, and evaluations conducted prior ot measurement.

    Attributes
    ----------
    time_stamp: str
        Time and date of test
    data: str
        Raw data from test
    result: dict
        Dictionary of test results. Varies by test.
    """
    
    def __init__(self):
        """Initialize instance variables."""

        self.time_stamp = None
        self.data = None
        self.result = {}
        
    def populate_data(self, time_stamp, data_in, data_type):
        """Coordinates storing of test, calibration, and evaluation data.

        Parameters
        ----------
        time_stamp: str
            Time and date text.
        data_in: str
            Raw data from test
        data_type: str
            Type of data, C-compass, TST-TRDI test, SST-SonTek test
        """

        # Store time stamp and data
        self.time_stamp = time_stamp
        self.data = data_in

        # Process data depending on data type and store result
        if data_type[1] == 'C':
            self.compass_read()
        elif data_type == 'TST':
            self.sys_test_read()
            self.pt3_data()
        elif data_type == 'SST':
            self.sys_test_read()

    def compass_read(self):
        """Method for getting compass evaluation data"""

        # Match regex for compass evaluation error:
        splits = re.split('(Total error:|Double Cycle Errors:|Error from calibration:)', self.data)
        if len(splits) > 1:
            error = float(re.search('\d+\.*\d*', splits[-1])[0])
        else:
            error = 'N/A'
        self.result['compass'] = {'error': error}

    @staticmethod
    def cc_qrev_mat_in(meas_struct):
        """Processes the Matlab data structure to obtain a list of Premeasurement objects containing compass calibration
           data from the Matlab data structure.

       Parameters
       ----------
       meas_struct: mat_struct
           Matlab data structure obtained from sio.loadmat

       Returns
       -------
       cc: list
           List of Premeasurement data objects
       """
        cc = []
        if hasattr(meas_struct, 'compassCal'):
            if type(meas_struct.compassCal) is np.ndarray:
                for cal in meas_struct.compassCal:
                    pm = PreMeasurement()
                    pm.compass_populate_from_qrev_mat(cal)
                    cc.append(pm)
            elif len(meas_struct.compassCal.data) > 0:
                pm = PreMeasurement()
                pm.compass_populate_from_qrev_mat(meas_struct.compassCal)
                cc.append(pm)

        return cc

    @staticmethod
    def ce_qrev_mat_in(meas_struct):
        """Processes the Matlab data structure to obtain a list of Premeasurement objects containing compass evaluation
           data from the Matlab data structure.

       Parameters
       ----------
       meas_struct: mat_struct
           Matlab data structure obtained from sio.loadmat

       Returns
       -------
       ce: list
           List of Premeasurement data objects
       """
        ce = []
        if hasattr(meas_struct, 'compassEval'):
            if type(meas_struct.compassEval) is np.ndarray:
                for comp_eval in meas_struct.compassEval:
                    pm = PreMeasurement()
                    pm.compass_populate_from_qrev_mat(comp_eval)
                    ce.append(pm)
            elif len(meas_struct.compassEval.data) > 0:
                pm = PreMeasurement()
                pm.compass_populate_from_qrev_mat(meas_struct.compassEval)
                ce.append(pm)
        return ce

    def compass_populate_from_qrev_mat(self, data_in):
        """Populated Premeasurement instance variables with data from QRev Matlab file.

        Parameters
        ----------
        data_in: mat_struct
            mat_struct_object containing compass cal/eval data
        """
        self.data = data_in.data
        self.time_stamp = data_in.timeStamp
        if hasattr(data_in, 'result'):
            self.result = {'compass': {'error': data_in.result.compass.error}}
        else:
            # Match regex for compass evaluation error:
            splits = re.split('(Total error:|Double Cycle Errors:|Error from calibration:)', self.data)
            if len(splits) > 1:
                error = float(re.search('\d+\.*\d*', splits[-1])[0])
            else:
                error = 'N/A'
            self.result['compass'] = {'error': error}
            
    def sys_test_read(self):
        """Method for reading the system test data"""
        if self.data is not None:
            # Match regex for number of tests and number of failures
            num_tests = re.findall('(Fail|FAIL|F A I L|Pass|PASS|NOT DETECTED|P A S S)', self.data)
            num_fails = re.findall('(Fail|FAIL|F A I L)', self.data)

            # Store results
            self.result = {'sysTest': {'n_tests': len(num_tests)}}
            self.result['sysTest']['n_failed'] = len(num_fails)
        else:
            self.result = {'sysTest': {'n_tests': None}}
            self.result['sysTest']['n_failed'] = None

    @staticmethod
    def sys_test_qrev_mat_in(meas_struct):
        """Processes the Matlab data structure to obtain a list of Premeasurement objects containing system test data
           from the Matlab data structure.

           Parameters
           ----------
           meas_struct: mat_struct
               Matlab data structure obtained from sio.loadmat

           Returns
           -------
           system_tst: list
               List of Premeasurement data objects
           """
        system_tst = []
        if hasattr(meas_struct, 'sysTest'):
            try:
                if type(meas_struct.sysTest) == np.ndarray:
                    for test in meas_struct.sysTest:
                        tst = PreMeasurement()
                        tst.sys_tst_populate_from_qrev_mat(test)
                        system_tst.append(tst)
                elif len(meas_struct.sysTest.data) > 0:
                    tst = PreMeasurement()
                    tst.sys_tst_populate_from_qrev_mat(meas_struct.sysTest)
                    system_tst.append(tst)
            except AttributeError:
                pass
        return system_tst

    def sys_tst_populate_from_qrev_mat(self, test_in):
        """Populated Premeasurement instance variables with data from QRev Matlab file.

        Parameters
        ----------
        test_in: mat_struct
            mat_struct_object containing system test data
        """
        try:
            self.data = test_in.data
            self.time_stamp = test_in.timeStamp
            self.result = {'sysTest': {'n_failed': test_in.result.sysTest.nFailed}}
            self.result['sysTest']['n_tests'] = test_in.result.sysTest.nTests

            if hasattr(test_in.result, 'pt3'):
                data_types = {'corr_table': np.array([]), 'sdc': np.array([]), 'cdc': np.array([]),
                              'noise_floor': np.array([])}
                test_types = {'high_wide': data_types.copy(), 'high_narrow': data_types.copy(),
                              'low_wide': data_types.copy(),
                              'low_narrow': data_types.copy()}
                pt3 = {'hard_limit': copy.deepcopy(test_types), 'linear': copy.deepcopy(test_types)}
                if hasattr(test_in.result.pt3, 'hardLimit'):
                    if hasattr(test_in.result.pt3.hardLimit, 'hw'):
                        pt3['hard_limit']['high_wide']['corr_table'] = test_in.result.pt3.hardLimit.hw.corrTable
                        pt3['hard_limit']['high_wide']['sdc'] = test_in.result.pt3.hardLimit.hw.sdc
                        pt3['hard_limit']['high_wide']['cdc'] = test_in.result.pt3.hardLimit.hw.cdc
                        pt3['hard_limit']['high_wide']['noise_floor'] = test_in.result.pt3.hardLimit.hw.noiseFloor
                    if hasattr(test_in.result.pt3.hardLimit, 'lw'):
                        pt3['hard_limit']['low_wide']['corr_table'] = test_in.result.pt3.hardLimit.lw.corrTable
                        pt3['hard_limit']['low_wide']['sdc'] = test_in.result.pt3.hardLimit.lw.sdc
                        pt3['hard_limit']['low_wide']['cdc'] = test_in.result.pt3.hardLimit.lw.cdc
                        pt3['hard_limit']['low_wide']['noise_floor'] = test_in.result.pt3.hardLimit.lw.noiseFloor
                    if hasattr(test_in.result.pt3.hardLimit, 'hn'):
                        pt3['hard_limit']['high_narrow']['corr_table'] = test_in.result.pt3.hardLimit.hn.corrTable
                        pt3['hard_limit']['high_narrow']['sdc'] = test_in.result.pt3.hardLimit.hn.sdc
                        pt3['hard_limit']['high_narrow']['cdc'] = test_in.result.pt3.hardLimit.hn.cdc
                        pt3['hard_limit']['high_narrow']['noise_floor'] = test_in.result.pt3.hardLimit.hn.noiseFloor
                    if hasattr(test_in.result.pt3.hardLimit, 'ln'):
                        pt3['hard_limit']['low_narrow']['corr_table'] = test_in.result.pt3.hardLimit.ln.corrTable
                        pt3['hard_limit']['low_narrow']['sdc'] = test_in.result.pt3.hardLimit.ln.sdc
                        pt3['hard_limit']['low_narrow']['cdc'] = test_in.result.pt3.hardLimit.ln.cdc
                        pt3['hard_limit']['low_narrow']['noise_floor'] = test_in.result.pt3.hardLimit.ln.noiseFloor
                if hasattr(test_in.result.pt3, 'linear'):
                    if hasattr(test_in.result.pt3.linear, 'hw'):
                        pt3['linear']['high_wide']['corr_table'] = test_in.result.pt3.linear.hw.corrTable
                        pt3['linear']['high_wide']['noise_floor'] = test_in.result.pt3.linear.hw.noiseFloor
                    if hasattr(test_in.result.pt3.linear, 'lw'):
                        pt3['linear']['low_wide']['corr_table'] = test_in.result.pt3.linear.lw.corrTable
                        pt3['linear']['low_wide']['noise_floor'] = test_in.result.pt3.linear.lw.noiseFloor
                    if hasattr(test_in.result.pt3.linear, 'hn'):
                        pt3['linear']['high_narrow']['corr_table'] = test_in.result.pt3.linear.hn.corrTable
                        pt3['linear']['high_narrow']['noise_floor'] = test_in.result.pt3.linear.hn.noiseFloor
                    if hasattr(test_in.result.pt3.linear, 'ln'):
                        pt3['linear']['low_narrow']['corr_table'] = test_in.result.pt3.linear.ln.corrTable
                        pt3['linear']['low_narrow']['noise_floor'] = test_in.result.pt3.linear.ln.noiseFloor

                self.result['pt3'] = pt3
        except AttributeError:
            # Match regex for number of tests and number of failures
            num_tests = re.findall('(Fail|FAIL|F A I L|Pass|PASS|NOT DETECTED|P A S S)', test_in.data)
            num_fails = re.findall('(Fail|FAIL|F A I L)', test_in.data)

            # Store results
            self.result = {'sysTest': {'n_tests': len(num_tests)}}
            self.result['sysTest']['n_failed'] = len(num_fails)

    def pt3_data(self):
        """Method for processing the data in the correlation matrices."""
        try:
            data_types = {'corr_table': np.array([]), 'sdc': np.array([]), 'cdc': np.array([]),
                          'noise_floor': np.array([])}
            test_types = {'high_wide': data_types.copy(), 'high_narrow': data_types.copy(),
                          'low_wide': data_types.copy(),
                          'low_narrow': data_types.copy()}
            pt3 = {'hard_limit': copy.deepcopy(test_types), 'linear': copy.deepcopy(test_types)}

            # Match regex for correlation tables
            matches = re.findall('Lag.*?0', self.data, re.DOTALL)

            # Count the number or correlation tables to process
            correl_count = 0
            for match in matches:
                bm1_matches = re.findall('Bm1', match)
                correl_count += len(bm1_matches)

            # Correlation table match
            lag_matches = re.findall('Lag.*?^\s*$', self.data, re.MULTILINE | re.DOTALL)

            # Sin match
            sin_match = re.findall('((Sin|SIN).*?^\s*$)', self.data, re.MULTILINE | re.DOTALL)[0][0]
            sin_array = np.array(re.findall('\d+\.*\d*', sin_match), dtype=int)

            # Cos match
            cos_match = re.findall('((Cos|COS).*?^\s*$)', self.data, re.MULTILINE | re.DOTALL)[0][0]
            cos_array = np.array(re.findall('\d+\.*\d*', cos_match), dtype=int)

            # RSSI match
            rssi_array = np.array([])
            rssi_matches = re.findall('RSSI.*?^\s*$', self.data, re.MULTILINE | re.DOTALL)
            for rssi_match in rssi_matches:
                rssi_array = np.hstack((rssi_array, np.array(re.findall('\d+\.*\d*', rssi_match), dtype=int)))

            # Process each set of correlation tables
            for n, lag_match in enumerate(lag_matches):

                # Count the Bm1 string to know how many tables to read
                bm_count = len(re.findall('Bm1', lag_match))

                # Extract the table into list
                numbers = re.findall('\d+\.*\d*', lag_match)

                # Create array from data in table
                corr_data = np.array(numbers[(bm_count * 4):(bm_count * 44)],
                                     dtype=int).reshape([8, (bm_count * 4) + 1])[:, 1::]

                # Only one pt3 test. Typical of Rio Grande and Streampro
                if bm_count == 1:

                    # Assign matrix slices to corresponding variables
                    # corr_hlimit_hgain_wband = corr_data
                    pt3['hard_limit']['high_wide']['corr_table'] = corr_data
                    pt3['hard_limit']['high_wide']['sdc'] = sin_array[0:4]
                    pt3['hard_limit']['high_wide']['cdc'] = cos_array[0:4]
                    pt3['hard_limit']['high_wide']['noise_floor'] = rssi_array[0:4]

                # 4 tests arranged in groups of 2. All data are hard limited.
                elif bm_count == 2 and correl_count == 4:

                    # Hard limited wide bandwidth (n=0)
                    if n == 0:

                        pt3['hard_limit']['high_wide']['corr_table'] = corr_data[:, 0:4]
                        pt3['hard_limit']['high_wide']['sdc'] = sin_array[n * 4: (n + 1) * 4]
                        pt3['hard_limit']['high_wide']['cdc'] = cos_array[n * 4: (n + 1) * 4]
                        pt3['hard_limit']['high_wide']['noise_floor'] = rssi_array[n * 4: (n + 1) * 4]

                        pt3['hard_limit']['low_wide']['corr_table'] = corr_data[:, 4::]
                        pt3['hard_limit']['low_wide']['sdc'] = sin_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['low_wide']['cdc'] = cos_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['low_wide']['noise_floor'] = rssi_array[(n + 1) * 4: (n + 2) * 4]

                    # Hard limited narrow bandwidth (n=1)
                    elif n == 1:

                        pt3['hard_limit']['high_narrow']['corr_table'] = corr_data[:, 0:4]
                        pt3['hard_limit']['high_narrow']['sdc'] = sin_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['high_narrow']['cdc'] = cos_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['high_narrow']['noise_floor'] = rssi_array[(n + 1) * 4: (n + 2) * 4]

                        pt3['hard_limit']['low_narrow']['corr_table'] = corr_data[:, 4::]
                        pt3['hard_limit']['low_narrow']['sdc'] = sin_array[(n + 2) * 4: (n + 3) * 4]
                        pt3['hard_limit']['low_narrow']['cdc'] = cos_array[(n + 2) * 4: (n + 3) * 4]
                        pt3['hard_limit']['low_narrow']['noise_floor'] = rssi_array[(n + 2) * 4: (n + 3) * 4]

                # 8 tests arranged in sets of 2. The linear is 1st followed by the hard limit.
                elif bm_count == 2 and correl_count == 8:

                    # Hard limit bandwidth (n=0)
                    if n == 0:

                        pt3['hard_limit']['high_wide']['corr_table'] = corr_data[:, 0:4]
                        pt3['hard_limit']['high_wide']['sdc'] = sin_array[n * 4: (n + 1) * 4]
                        pt3['hard_limit']['high_wide']['cdc'] = cos_array[n * 4: (n + 1) * 4]
                        pt3['hard_limit']['high_wide']['noise_floor'] = rssi_array[n * 4: (n + 1) * 4]

                        pt3['hard_limit']['low_wide']['corr_table'] = corr_data[:, 4::]
                        pt3['hard_limit']['low_wide']['sdc'] = sin_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['low_wide']['cdc'] = cos_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['low_wide']['noise_floor'] = rssi_array[(n + 1) * 4: (n + 2) * 4]

                    # Hard limit narrow bandwidth (n=1)
                    elif n == 1:

                        pt3['hard_limit']['high_narrow']['corr_table'] = corr_data[:, 0:4]
                        pt3['hard_limit']['high_narrow']['sdc'] = sin_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['high_narrow']['cdc'] = cos_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['high_narrow']['noise_floor'] = rssi_array[(n + 1) * 4: (n + 2) * 4]

                        pt3['hard_limit']['low_narrow']['corr_table'] = corr_data[:, 4::]
                        pt3['hard_limit']['low_narrow']['sdc'] = sin_array[(n + 2) * 4: (n + 3) * 4]
                        pt3['hard_limit']['low_narrow']['cdc'] = cos_array[(n + 2) * 4: (n + 3) * 4]
                        pt3['hard_limit']['low_narrow']['noise_floor'] = rssi_array[(n + 2) * 4: (n + 3) * 4]

                    # Linear wide bandwidth (n=2)
                    elif n == 2:

                        pt3['linear']['high_wide']['corr_table'] = corr_data[:, 0:4]
                        pt3['linear']['high_wide']['noise_floor'] = rssi_array[(n + 2) * 4: (n + 3) * 4]

                        pt3['linear']['low_wide']['corr_table'] = corr_data[:, 4::]
                        pt3['linear']['low_wide']['noise_floor'] = rssi_array[(n + 3) * 4: (n + 4) * 4]

                    # Linear narrow bandwidth (n=3)
                    elif n == 3:

                        pt3['linear']['high_narrow']['corr_table'] = corr_data[:, 0:4]
                        pt3['linear']['high_narrow']['noise_floor'] = rssi_array[(n + 3) * 4: (n + 4) * 4]

                        pt3['linear']['low_narrow']['corr_table'] = corr_data[:, 4::]
                        pt3['linear']['low_narrow']['noise_floor'] = rssi_array[(n + 4) * 4: (n + 5) * 4]

                # 8 tests in groups of 4. Hard limit is the first group then the linear.
                elif bm_count == 4:

                    # Hard limit data (n=0)
                    if n == 0:

                        pt3['hard_limit']['high_wide']['corr_table'] = corr_data[:, 0:4]
                        pt3['hard_limit']['high_wide']['sdc'] = sin_array[n * 4: (n + 1) * 4]
                        pt3['hard_limit']['high_wide']['cdc'] = cos_array[n * 4: (n + 1) * 4]
                        pt3['hard_limit']['high_wide']['noise_floor'] = rssi_array[n * 4: (n + 1) * 4]

                        pt3['hard_limit']['low_wide']['corr_table'] = corr_data[:, 4:8]
                        pt3['hard_limit']['low_wide']['sdc'] = sin_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['low_wide']['cdc'] = cos_array[(n + 1) * 4: (n + 2) * 4]
                        pt3['hard_limit']['low_wide']['noise_floor'] = rssi_array[(n + 1) * 4: (n + 2) * 4]

                        pt3['hard_limit']['high_narrow']['corr_table'] = corr_data[:, 8:12]
                        pt3['hard_limit']['high_narrow']['sdc'] = sin_array[(n + 2) * 4: (n + 3) * 4]
                        pt3['hard_limit']['high_narrow']['cdc'] = cos_array[(n + 2) * 4: (n + 3) * 4]
                        pt3['hard_limit']['high_narrow']['noise_floor'] = rssi_array[(n + 2) * 4: (n + 3) * 4]

                        pt3['hard_limit']['low_narrow']['corr_table'] = corr_data[:, 12::]
                        pt3['hard_limit']['low_narrow']['sdc'] = sin_array[(n + 3) * 4: (n + 4) * 4]
                        pt3['hard_limit']['low_narrow']['cdc'] = cos_array[(n + 3) * 4: (n + 4) * 4]
                        pt3['hard_limit']['low_narrow']['noise_floor'] = rssi_array[(n + 3) * 4: (n + 4) * 4]

                    # Linear data (n=1)
                    else:
                        pt3['linear']['high_wide']['corr_table'] = corr_data[:, 0:4]
                        pt3['linear']['high_wide']['noise_floor'] = rssi_array[(n + 3) * 4: (n + 4) * 4]

                        pt3['linear']['low_wide']['corr_table'] = corr_data[:, 4:8]
                        pt3['linear']['low_wide']['noise_floor'] = rssi_array[(n + 4) * 4: (n + 5) * 4]

                        pt3['linear']['high_narrow']['corr_table'] = corr_data[:, 8:12]
                        pt3['linear']['high_narrow']['noise_floor'] = rssi_array[(n + 5) * 4: (n + 6) * 4]

                        pt3['linear']['low_narrow']['corr_table'] = corr_data[:, 12::]
                        pt3['linear']['low_narrow']['noise_floor'] = rssi_array[(n + 6) * 4: (n + 7) * 4]
            self.result['pt3'] = pt3
        except Exception:
            pass
        #     pt3 = None
        # self.result['pt3'] = pt3
