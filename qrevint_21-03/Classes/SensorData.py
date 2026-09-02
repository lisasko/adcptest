import numpy as np


class SensorData(object):
    """Class stores data for pitch, roll, temperature, salinity, and speed of sound and its source.

    Attributes
    ----------
    data: np.array(float)
        Data to be used in computations.
    data_orig: np.array(float)
        Original data loaded from raw data file.
    source: str
        Source of data, examples Int. Sensor, Ext. Sensor, User
    """
    
    def __init__(self):
        """Initializes class and variables."""

        self.data = None
        self.data_orig = None
        self.source = None
        
    def populate_data(self, data_in, source_in):
        """Store data in class.

        Parameters
        ----------
        data_in: np.array(float)
            Data to be stored.
        source_in: str
            Source of data to be stored.
        """

        self.data = data_in
        self.data_orig = data_in
        self.source = source_in

    def populate_from_qrev_mat(self, mat_data):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        if np.isnan(mat_data.data).all():
            self.data = np.array([])
        else:
            if type(mat_data.data) is np.ndarray:
                self.data = mat_data.data.astype(float)
            else:
                self.data = np.array([float(mat_data.data)])
        if np.isnan(mat_data.dataOrig).all():
            self.data_orig = np.array([])
        else:
            if type(mat_data.dataOrig) is np.ndarray:
                self.data_orig = mat_data.dataOrig.astype(float)
            else:
                self.data_orig = np.array([float(mat_data.dataOrig)])
        self.source = mat_data.source
        
    def change_data(self, data_in):
        """Change data to be applied in computations.

        Parameters
        ----------
        data_in: np.array(float)
        """

        self.data = data_in
        
    def set_source(self, source_in):
        """Change source of data.

        Parameters
        ----------
        source_in: str
            Source of data.
        """
        self.source = source_in
