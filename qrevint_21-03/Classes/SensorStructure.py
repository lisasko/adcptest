import numpy as np
from Classes.HeadingData import HeadingData
from Classes.SensorData import SensorData


class SensorStructure(object):
    """Class to store sensor data from various sources.

    Attributes
    ----------
    self.selected: str
        The selected sensor reference name ('internal', 'external', 'user').
    self.internal: SensorData
        Contains the data from the internal sensor, object of SensorData
    self.external: SensorData
        Contains the data from an external sensor, object of SensorData
    self.user: SensorData
        Contains user supplied value, object of SensorData
    """
    
    def __init__(self):
        """Initialize class and set variable to None."""

        self.selected = None  # The selected sensor reference name ('internal', 'external', 'user')
        self.internal = None  # Contains the data from the internal sensor
        self.external = None  # Contains the data from an external sensor
        self.user = None  # Contains user supplied value

    def populate_from_qrev_mat(self, mat_data, heading=False):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat
        heading: bool
            Determines if mat_data is heading data
        """

        if not heading:
            # Non-heading sensors
            if not type(mat_data.external) is np.ndarray:
                self.external = SensorData()
                self.external.populate_from_qrev_mat(mat_data.external)
            if not type(mat_data.internal) is np.ndarray:
                self.internal = SensorData()
                self.internal.populate_from_qrev_mat(mat_data.internal)
            if not type(mat_data.user) is np.ndarray:
                self.user = SensorData()
                self.user.populate_from_qrev_mat(mat_data.user)
            self.selected = mat_data.selected
        else:
            # Heading data
            if not type(mat_data.external) is np.ndarray:
                self.external = HeadingData()
                self.external.populate_from_qrev_mat(mat_data.external)
            if not type(mat_data.internal) is np.ndarray:
                self.internal = HeadingData()
                self.internal.populate_from_qrev_mat(mat_data.internal)
            if not type(mat_data.user) is np.ndarray:
                self.user = HeadingData()
                self.user.populate_from_qrev_mat(mat_data.user)
            self.selected = mat_data.selected
        
    def set_selected(self, selected_name):
        """Set the selected source for the specified object

        Parameters
        ----------
        selected_name: str
            Type of data (internal, external, user).
        """
        self.selected = selected_name
