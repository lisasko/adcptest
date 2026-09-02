import numpy as np
from Classes.SelectFit import SelectFit
from Classes.ExtrapQSensitivity import ExtrapQSensitivity
from Classes.NormData import NormData


class ComputeExtrap(object):
    """Class to compute the optimized or manually specified extrapolation methods

    Attributes
    ----------
    threshold: float
        Threshold as a percent for determining if a median is valid
    subsection: list
        Percent of discharge, does not account for transect direction
    fit_method: str
        Method used to determine fit.  Automatic or manual
    norm_data: NormData
        Object of class NormData
    sel_fit: SelectFit
        Object of class SelectFit
    q_sensitivity: ExtrapQSensitivity
        Object of class ExtrapQSensitivity
    messages: str
        Variable for messages to UserWarning
    use_weighted: bool
        Specifies if discharge weighted medians are used in extrapolations
     sub_from_left: bool
            Specifies if when subsectioning the subsection should start from left to right.
    use_q: bool
        Specifies to use the discharge rather than the xprod when subsectioning

    """
    
    def __init__(self):
        """Initialize instance variables."""

        self.threshold = None  # Threshold as a percent for determining if a median is valid
        self.subsection = None  # Percent of discharge, does not account for transect direction
        self.fit_method = None  # Method used to determine fit.  Automatic or manual
        self.norm_data = []  # Object of class norm data
        self.sel_fit = []  # Object of class SelectFit
        self.q_sensitivity = None  # Object of class ExtrapQSensitivity
        self.messages = []  # Variable for messages to UserWarning
        self.use_weighted = False
        self.use_q = False
        self.sub_from_left = False
        
    def populate_data(self, transects, compute_sensitivity=True, use_weighted=False, use_q=True, sub_from_left=True):
        """Store data in instance variables.

        Parameters
        ----------
        transects: list
            List of transects of TransectData
        compute_sensitivity: bool
            Determines is sensitivity should be computed.
        use_weighted: bool
        Specifies if discharge weighted medians are used in extrapolations
        """

        self.threshold = 20
        self.subsection = [0, 100]
        self.fit_method = 'Automatic'
        self.use_weighted = use_weighted
        self.use_q = use_q
        self.sub_from_left = sub_from_left
        self.process_profiles(transects=transects, data_type='q', use_weighted=use_weighted)

        # Compute the sensitivity of the final discharge to changes in extrapolation methods
        if compute_sensitivity:
            self.q_sensitivity = ExtrapQSensitivity()
            self.q_sensitivity.populate_data(transects=transects, extrap_fits=self.sel_fit)

    def populate_from_qrev_mat(self, meas_struct):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        meas_struct: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        if hasattr(meas_struct, 'extrapFit'):
            self.threshold = meas_struct.extrapFit.threshold
            self.subsection = meas_struct.extrapFit.subsection
            self.fit_method = meas_struct.extrapFit.fitMethod

            # Check for consistency between transects and norm_data. If only checked transects were saved, the
            # normData and selfit will also include unchecked transects which must be removed prior to
            # continuing to process.

            # If only a single transect the meas_struct.transects will be structure not an array, so the len method
            # won't work.
            try:
                n_transects = len(meas_struct.transects)
            except TypeError:
                n_transects = 1

            try:
                n_data = len(meas_struct.extrapFit.normData) - 1
            except TypeError:
                n_data = 1

            if n_transects != n_data:
                # normData needs adjustment to match transects
                file_names = []
                valid_norm_data = []
                valid_sel_fit = []
                # Create list of transect filenames
                if n_transects == 1:
                    file_names.append(meas_struct.transects.fileName)
                else:
                    for transect in meas_struct.transects:
                        file_names.append(transect.fileName)
                # Create a list of norm_data and sel_fit objects that match the filenames in transects
                for n in range(len(meas_struct.extrapFit.normData) - 1):
                    if meas_struct.extrapFit.normData[n].fileName in file_names:
                        valid_norm_data.append(meas_struct.extrapFit.normData[n])
                        valid_sel_fit.append(meas_struct.extrapFit.selFit[n])
                # Append the whole measurement objects
                valid_norm_data.append(meas_struct.extrapFit.normData[-1])
                valid_sel_fit.append(meas_struct.extrapFit.selFit[-1])
                # Update meas_struct so normData and selFit match transects
                meas_struct.extrapFit.normData = np.array(valid_norm_data)
                meas_struct.extrapFit.selFit = np.array(valid_sel_fit)

            self.norm_data = NormData.qrev_mat_in(meas_struct.extrapFit)
            self.sel_fit = SelectFit.qrev_mat_in(meas_struct.extrapFit)
            self.q_sensitivity = ExtrapQSensitivity()
            self.q_sensitivity.populate_from_qrev_mat(meas_struct.extrapFit)
            if hasattr(meas_struct.extrapFit, 'use_weighted'):
                self.use_weighted = meas_struct.extrapFit.use_weighted
            else:
                self.use_weighted = False

            if hasattr(meas_struct.extrapFit, 'use_q'):
                self.use_q = meas_struct.extrapFit.use_q
            else:
                self.use_q = False

            if hasattr(meas_struct.extrapFit, 'sub_from_left'):
                self.sub_from_left = meas_struct.extrapFit.sub_from_left
            else:
                self.sub_from_left = False

            if type(meas_struct.extrapFit.messages) is str:
                self.messages = [meas_struct.extrapFit.messages]
            elif type(meas_struct.extrapFit.messages) is np.ndarray:
                self.messages = meas_struct.extrapFit.messages.tolist()

    def process_profiles(self, transects, data_type, use_weighted=None, use_q=True, sub_from_left=True):
        """Function that coordinates the fitting process.

        Parameters
        ----------
        transects: TransectData
            Object of TransectData
        data_type: str
            Type of data processing (q or v)
        sub_from_left: bool
            Specifies if when subsectioning the subsection should start from left to right.
        use_q: bool
            Specifies to use the discharge rather than the xprod when subsectioning
        """
        if use_weighted is not None:
            self.use_weighted = use_weighted
        else:
            self.use_weighted = self.norm_data[-1].use_weighted

        self.use_q = use_q
        self.sub_from_left = sub_from_left

        # Compute normalized data for each transect
        self.norm_data = []
        for transect in transects:
            norm_data = NormData()
            norm_data.populate_data(transect=transect,
                                    data_type=data_type,
                                    threshold=self.threshold,
                                    data_extent=self.subsection,
                                    use_weighted=self.use_weighted,
                                    use_q=self.use_q,
                                    sub_from_left=self.sub_from_left)
            self.norm_data.append(norm_data)

        # Compute composite normalized data
        comp_data = NormData()
        comp_data.use_q = self.norm_data[-1].use_q
        comp_data.sub_from_left = self.norm_data[-1].sub_from_left
        comp_data.create_composite(transects=transects, norm_data=self.norm_data, threshold=self.threshold)
        self.norm_data.append(comp_data)

        # Compute the fit for the selected  method
        if self.fit_method == 'Manual':
            for n in range(len(transects)):
                self.sel_fit[n].populate_data(normalized=self.norm_data[n],
                                              fit_method=self.fit_method,
                                              top=transects[n].extrap.top_method,
                                              bot=transects[n].extrap.bot_method,
                                              exponent=transects[n].extrap.exponent)
        else:
            self.sel_fit = []
            for n in range(len(self.norm_data)):
                sel_fit = SelectFit()
                sel_fit.populate_data(self.norm_data[n], self.fit_method)
                self.sel_fit.append(sel_fit)

        if self.sel_fit[-1].top_fit_r2 is not None:
            # Evaluate if there is a potential that a 3-point top method may be appropriate
            if (self.sel_fit[-1].top_fit_r2 > 0.9 or self.sel_fit[-1].top_r2 > 0.9) \
                    and np.abs(self.sel_fit[-1].top_max_diff) > 0.2:
                self.messages.append('The measurement profile may warrant a 3-point fit at the top')
                
    def update_q_sensitivity(self, transects):
        """Updates the discharge sensitivity values.

        Parameters
        ----------
        transects: list
            List of TransectData objects
        """
        self.q_sensitivity = ExtrapQSensitivity()
        self.q_sensitivity.populate_data(transects, self.sel_fit)
        
    def change_fit_method(self, transects, new_fit_method, idx, top=None, bot=None, exponent=None, compute_qsens=True):
        """Function to change the extrapolation method.

        Parameters
        ----------
        transects: list
            List of TransectData objects
        new_fit_method: str
            Identifies fit method automatic or manual
        idx: int
            Index to the specified transect or measurement in NormData
        top: str
            Specifies top fit
        bot: str
            Specifies bottom fit
        exponent: float
            Specifies exponent for power or no slip fits
        compute_qsens: bool
            Specifies if the discharge sensitivities should be recomputed
        """
        self.fit_method = new_fit_method

        self.sel_fit[idx].populate_data(self.norm_data[idx], new_fit_method,  top=top, bot=bot, exponent=exponent)
        if compute_qsens & idx == len(self.norm_data)-1:
            self.q_sensitivity = ExtrapQSensitivity()
            self.q_sensitivity.populate_data(transects, self.sel_fit)
        
    def change_threshold(self, transects, data_type, threshold):
        """Function to change the threshold for accepting the increment median as valid.  The threshold
        is in percent of the median number of points in all increments.

        Parameters
        ----------
        transects: list
            List of TransectData objects
        data_type: str
            Specifies the data type (discharge or velocity)
        threshold: float
            Percent of data that must be in a median to include the median in the fit algorithm
        """
        
        self.threshold = threshold
        self.process_profiles(transects=transects, data_type=data_type)
        self.q_sensitivity = ExtrapQSensitivity()
        self.q_sensitivity.populate_data(transects=transects, extrap_fits=self.sel_fit)
        
    def change_extents(self, transects, data_type, extents, use_q, sub_from_left):
        """Function allows the data to be subsection by specifying the percent cumulative discharge
        for the start and end points.  Currently this function does not consider transect direction.

        Parameters
        ----------
        transects: list
            List of TransectData objects
        data_type: str
            Specifies the data type (discharge or velocity)
        extents: list
            List containing two values, the minimum and maximum discharge percentages to subsectioning
        sub_from_left: bool
            Specifies if when subsectioning the subsection should start from left to right.
        use_q: bool
            Specifies to use the discharge rather than the xprod when subsectioning
        """
        
        self.subsection = extents
        self.use_q = use_q
        self.sub_from_left = sub_from_left
        self.process_profiles(transects=transects, data_type=data_type )
        self.q_sensitivity = ExtrapQSensitivity()
        self.q_sensitivity.populate_data(transects=transects, extrap_fits=self.sel_fit)
        
    def change_data_type(self, transects, data_type):
        """Changes the data type to be processed in extrap.

        Parameters
        ----------
        transects: list
            List of TransectData objects
        data_type: str
            Specifies the data type (discharge or velocity)
        """
        if data_type.lower() == 'q':
            use_weighted = self.use_weighted
        else:
            use_weighted = False

        self.process_profiles(transects=transects, data_type=data_type, use_weighted=use_weighted)
        self.q_sensitivity = ExtrapQSensitivity()
        self.q_sensitivity.populate_data(transects=transects, extrap_fits=self.sel_fit)

    def change_data_auto(self, transects):
        """Changes the data selection settings to automatic.

        Parameters
        ----------
        transects: list
            List of TransectData objects
        """
        self.threshold = 20
        self.subsection = [0, 100]
        self.process_profiles(transects=transects, data_type='q', use_weighted=self.use_weighted)

        # Compute the sensitivity of the final discharge to changes in extrapolation methods
        self.q_sensitivity = ExtrapQSensitivity()
        self.q_sensitivity.populate_data(transects=transects, extrap_fits=self.sel_fit)
