import numpy as np


class EdgeData(object):
    """Class used to store edge settings.

    Attributes
    ----------
    type: str
        Shape of edge: 'Triangular', 'Rectangular', 'Custom, 'User Q'
    distance_m: float
        Distance to shore, in m.
    cust_coef: float
        Custom coefficient provided by user.
    number_ensembles: int
        Number of ensembles to average for depth and velocities.
    user_discharge_cms: float
        Original user supplied discharge for edge, in cms.
    orig_type: str
        Original shape of edge: 'Triangular', 'Rectangular', 'Custom, 'User Q'
    orig_distance_m: float
        Original distance to shore, in m.
    orig_cust_coef: float
        Original custom coefficient provided by user.
    orig_number_ensembles: int
        Original number of ensembles to average for depth and velocities.
    orig_user_discharge_cms: float
        Original user supplied discharge for edge, in cms.
    """
    
    def __init__(self):
        """Initialize EdgeData.
        """
        
        self.type = None       # Shape of edge: 'Triangular', 'Rectangular', 'Custom, 'User Q'
        self.distance_m = None          # Distance to shore
        self.cust_coef = None     # Custom coefficient provided by user
        self.number_ensembles = None   # Number of ensembles to average for depth and velocities
        self.user_discharge_cms = None      # User supplied edge discharge.

        self.orig_type = None  # Shape of edge: 'Triangular', 'Rectangular', 'Custom, 'User Q'
        self.orig_distance_m = None  # Distance to shore
        self.orig_cust_coef = None  # Custom coefficient provided by user
        self.orig_number_ensembles = None  # Number of ensembles to average for depth and velocities
        self.orig_user_discharge_cms = None  # User supplied edge discharge.

        
    def populate_data(self, edge_type, distance=None, number_ensembles=10, coefficient=None, user_discharge=None):
        """Construct left or right edge object from provided inputs
        
        Parameters
        ----------
        edge_type: str
            Type of edge (Triangular, Rectangular, Custom, User Q)
        distance: float
            Distance to shore, in m.
        number_ensembles: int
            Number of edge ensembles for all types but User Q
        coefficient: float
            User supplied custom edge coefficient.
        user_discharge: float
            User supplied edge discharge, in cms.
        """

        # Set properties for custom coefficient
        self.type = edge_type
        self.distance_m = distance
        self.number_ensembles = number_ensembles
        self.user_discharge_cms = user_discharge
        self.cust_coef = coefficient

        if self.orig_type is None:
            self.orig_type = edge_type
            self.orig_distance_m = distance
            self.orig_number_ensembles = number_ensembles
            self.orig_user_discharge_cms = user_discharge
            self.orig_cust_coef = coefficient

    def populate_from_qrev_mat(self, mat_data):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        self.type = mat_data.type
        self.distance_m = mat_data.dist_m
        self.number_ensembles = mat_data.numEns2Avg
        if type(mat_data.userQ_cms) is float:
            if not np.isnan(mat_data.userQ_cms):
                self.user_discharge_cms = mat_data.userQ_cms
        if type(mat_data.custCoef) is float:
            self.cust_coef = mat_data.custCoef
        if hasattr(mat_data, 'orig_type'):
            self.orig_type = mat_data.orig_type
            self.orig_distance_m = mat_data.orig_distance_m
            self.orig_number_ensembles = mat_data.orig_number_ensembles
            if type(mat_data.orig_user_discharge_cms) is float:
                if not np.isnan(mat_data.orig_user_discharge_cms):
                    self.orig_user_discharge_cms = mat_data.orig_user_discharge_cms
            if type(mat_data.custCoef) is float:
                self.orig_cust_coef = mat_data.orig_cust_coef
        else:
            self.orig_type = mat_data.type
            self.orig_distance_m = mat_data.dist_m
            self.orig_number_ensembles = mat_data.numEns2Avg
            if type(mat_data.userQ_cms) is float:
                if not np.isnan(mat_data.userQ_cms):
                    self.orig_user_discharge_cms = mat_data.userQ_cms
            if type(mat_data.custCoef) is float:
                self.orig_cust_coef = mat_data.custCoef

    def change_property(self, prop, setting):
        """Change edge data property

        Parameters
        ----------
        prop: str
            Property to change.
        setting:
            New setting for property.
        """
        setattr(self, prop, setting)
