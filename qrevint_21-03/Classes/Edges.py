from Classes.EdgeData import EdgeData


class Edges(object):
    """Class to store and process edge data.

    Attributes
    ----------
    rec_edge_method: str
        Method used to determine coef for rec. edge 'Fixed', 'Variable'.
    vel_method: str
        Method used to compute the velocity used 'MeasMag', 'VectorProf'.
    left: EdgeData
        Object of EdgeData for left edge.
    right: EdgeData
        Object of EdgeData for right edge.
    """
    
    def __init__(self):
        """Initialize Edges.
        """

        self.rec_edge_method = None
        self.vel_method = None
        self.left = EdgeData()
        self.right = EdgeData()
        
    def populate_data(self, rec_edge_method, vel_method):
        """Store the general methods used for edge data.

        Parameters
        ----------
        rec_edge_method: str
            Method used to determine coef for rec. edge 'Fixed', 'Variable'.
        vel_method: str
            Method used to compute the velocity used 'MeasMag', 'VectorProf'.
        """
        self.rec_edge_method = rec_edge_method
        self.vel_method = vel_method

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        transect: mat_struct
           Matlab data structure obtained from sio.loadmat
       """

        if hasattr(transect, 'edges'):
            if hasattr(transect.edges, 'left'):
                self.left = EdgeData()
                self.left.populate_from_qrev_mat(transect.edges.left)
            if hasattr(transect.edges, 'right'):
                self.right = EdgeData()
                self.right.populate_from_qrev_mat(transect.edges.right)
            self.rec_edge_method = transect.edges.recEdgeMethod
            self.vel_method = transect.edges.velMethod

    def change_property(self, prop, setting, edge=None):
        """Change edge property
        
        Parameters
        ----------
        prop: str
            Name of property.
        setting:
            New property setting.
        edge: str
            Edge to change (left, right)
        """
        
        if edge is None:
            setattr(self, prop, setting)
        else:
            temp = getattr(self, edge)
            temp.change_property(prop, setting)
