import numpy as np
from _operator import xor


class HeadingData(object):
    """This class stores and manipulates heading and associated data.

    Attributes
    ----------
    data: np.array(float)
        Corrected heading data, in degrees.
    original_data: np.array(float)
        Original uncorrected heading data, in degrees.
    source: str
        Source of heading data (internal, external).
    mag_var_deg: float
        Magnetic variation applied to get corrected data, in degrees (East +, West -).
    mag_var_orig_deg: float
        Original magnetic variation, in degrees (East +, West -).
    align_correction_deg: float
        Alignment correction to align compass with instrument (used for external heading), in degrees CW.
    align_correction_orig_deg: float
        Alignment correction to align compass with instrument (used for external heading), in degrees CW.
    mag_error: np.array(float)
        Percent change in mean magnetic field from calibration (SonTek only).`
    pitch_limit: np.array(float)
        Pitch limit of compass calibration (SonTek only), in degrees.
    roll_limit: np.array(float)
        Roll limit of compass calibration (SonTek only), in degrees.
    """
    
    def __init__(self):
        """Initialize class and set variables to None."""

        self.data = None  # Corrected self.data data
        self.original_data = None  # original uncorrected self.data data
        self.source = None  # Source of self.data data (internal, external)
        self.mag_var_deg = None  # Magnetic variation for these self.data data
        self.mag_var_orig_deg = None  # Original magnetic variation
        self.align_correction_deg = None  # Alignment correction to align compass with instrument
        self.align_correction_orig_deg = None
        self.mag_error = None  # Percent change in mean magnetic field from calibration`
        self.pitch_limit = None  # Pitch limit of compass calibration (SonTek only), in degrees.
        self.roll_limit = None  # Roll limit of compass calibration (SonTek only), in degrees.
        
    def populate_data(self, data_in, source_in, magvar=0, align=0, mag_error=None, pitch_limit=None, roll_limit=None):
        """Assigns values to instance variables.

        Parameters
        ----------
        data_in: np.array(float)
            Heading, in degrees.
        source_in: str
            Source of heading data (internal, external).
        magvar: float
            Magnetic variation, in degrees (East +, West -).
        align: float
            Alignment correction to align compass with instrument, in degrees
        mag_error: np.array(float)
            Percent change in magnetic field (SonTek only)
        pitch_limit: np.array(float)
            Pitch limit of compass calibration (SonTek only)
        roll_limit: np.array(float)
            Roll limit of compass calibration (SonTek only)
        """

        self.original_data = data_in
        self.source = source_in
        self.mag_var_deg = float(magvar)
        self.mag_var_orig_deg = float(magvar)
        self.align_correction_deg = float(align)
        self.align_correction_orig_deg = float(align)
        self.mag_error = mag_error

        if pitch_limit is not None and len(pitch_limit.shape) > 1:
            self.pitch_limit = pitch_limit[0, :]
        else:
            self.pitch_limit = pitch_limit

        if roll_limit is not None and len(roll_limit.shape) > 1:
            self.roll_limit = roll_limit[0, :]
        else:
            self.roll_limit = roll_limit

        # Correct the original data for the magvar and alignment
        if source_in == 'internal':
            self.data = self.original_data + self.mag_var_deg
        else:
            self.data = self.original_data + self.align_correction_deg
        self.fix_upper_limit()
        self.interp_heading()

    def populate_from_qrev_mat(self, mat_data):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        self.data = mat_data.data
        self.original_data = mat_data.originalData
        self.source = mat_data.source
        self.mag_var_deg = float(mat_data.magVar_deg)
        self.mag_var_orig_deg = float(mat_data.magVarOrig_deg)
        self.align_correction_deg = mat_data.alignCorrection_deg
        if hasattr(mat_data, 'align_correction_orig_deg'):
            self.align_correction_orig_deg = mat_data.align_correction_orig_deg
        else:
            self.align_correction_orig_deg = mat_data.alignCorrection_deg

        # Only available for SonTek G3 compass
        try:
            if len(mat_data.magError) > 0:
                self.mag_error = mat_data.magError

            # Only available for SonTek G3 compass
            if mat_data.pitchLimit.size > 0:
                if mat_data.pitchLimit.size > 2:
                    self.pitch_limit = mat_data.pitchLimit[0]
                else:
                    self.pitch_limit = mat_data.pitchLimit

            # Only available for SonTek G3 compass
            if mat_data.rollLimit.size > 0:
                if mat_data.rollLimit.size > 2:
                    self.roll_limit = mat_data.rollLimit[0]
                else:
                    self.roll_limit = mat_data.rollLimit
        except AttributeError:
            self.mag_error = None
            self.pitch_limit = None
            self.roll_limit = None

    def set_mag_var(self, mag_var, h_source):
        """Applies a new magvar to the object data.

        Parameters
        ----------
        mag_var: float
            Magnetic variation, in degrees
        h_source: str
            Heading source (internal or external)
        """

        self.mag_var_deg = mag_var
        if h_source == 'internal':
            self.data = self.original_data + self.mag_var_deg
            self.fix_upper_limit()
            
    def set_align_correction(self, align_correction, h_source):
        """Applies a new alignment correction to the object data.

        Parameters
        ----------
        align_correction: float
            Alignment correction, in degrees
        h_source: str
            Heading source (internal or external)
        """

        self.align_correction_deg = align_correction
        if h_source == 'external':
            self.data = self.original_data + self.align_correction_deg
            self.fix_upper_limit()

    def fix_upper_limit(self):
        """Fixes heading when magvar and or alignment are applied resulting in heading greater than 360 degrees.
        """

        idx = np.where(self.data > 360)[0]
        if len(idx) > 0:
            self.data[idx] = self.data[idx] - 360   
            
    def interp_heading(self):
        """Interpolate invalid headings. Use linear interpolation if there are
        valid values on either side of the invalid heading. If the invalid heading
        occurs at the beginning of the time series, back fill using the 1st valid.
        If the invalid heading occurs at the end of the time series, forward fill
        with the last valid self.data.
        """
        
        idx_invalid = np.where(np.isnan(self.data))[0]
        
        if len(idx_invalid) > 0:
            
            first_valid_idx = np.where(np.isnan(self.data) == False)[0][0]
            last_valid_idx = np.where(np.isnan(self.data) == False)[0][-1]
        
            # Process each invalid self.data
            for n in range(len(idx_invalid)):
                before_idx = np.where(np.isnan(self.data[0:idx_invalid[n] + 1]) == False)[0]
                after_idx = np.where(np.isnan(self.data[idx_invalid[n]:]) == False)[0]
                
                # If invalid self.data is beginning back fill
                if len(before_idx) < 1:
                    self.data[idx_invalid[n]] = self.data[first_valid_idx]

                # If invalid self.data is at end forward fill
                elif len(after_idx) < 1:
                    self.data[idx_invalid[n]] = self.data[last_valid_idx]

                # If invalid self.data is in middle interpolate
                else:
                    before_idx = before_idx[-1]
                    after_idx = after_idx[0] + idx_invalid[n]
                    
                    test1 = self.data[before_idx] > 180
                    test2 = self.data[after_idx] > 180
                    c = None
                    if not xor(test1, test2):
                        c = 0
                    elif test1:
                        c = 360
                    elif test2:
                        c = -360
                    self.data[idx_invalid[n]] = (((self.data[after_idx] - self.data[before_idx] + c) /
                                                  (before_idx - after_idx)) *
                                                 (before_idx - idx_invalid[n])) + self.data[before_idx]
                    if self.data[idx_invalid[n]] > 360:
                        self.data[idx_invalid[n]] = self.data[idx_invalid[n]] - 360
                    elif self.data[idx_invalid[n]] < 0:
                        self.data[idx_invalid[n]] = self.data[idx_invalid[n]] + 360
