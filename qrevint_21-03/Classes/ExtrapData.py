class ExtrapData(object):
    """Class to store both original and modified extrapolation settings.

    Attributes
    ----------
    top_method_orig: str
        Original extrapolation method for top of profile: Power, Constant, 3-Point.
    bot_method_orig: str
        Original extrapolation method for bottom of profile: Power, No Slip.
    exponent_orig: float
        Original exponent for power of no slip methods.
    top_method: str
        Applied extrapolation method for top of profile: Power, Constant, 3-Point.
    bot_method: str
        Applied extrapolation method for bottom of profile: Power, No Slip
    exponent: float
        Applied exponent for power of no slip methods
    """
    
    def __init__(self):
        """Initialize class and set defaults."""
        self.top_method_orig = None  # Extrapolation method for top of profile: Power, Constant, 3-Point
        self.bot_method_orig = None  # Extrapolation method for bottom of profile: Power, No Slip
        self.exponent_orig = None  # Exponent for power of no slip methods
        self.top_method = None  # Extrapolation method for top of profile: Power, Constant, 3-Point
        self.bot_method = None  # Extrapolation method for bottom of profile: Power, No Slip
        self.exponent = None  # Exponent for power of no slip methods
        
    def populate_data(self, top, bot, exp):
        """Store data in class variables.

        Parameters
        ----------
        top: str
            Original top method.
        bot: str
            Original bottom method.
        exp: float
            Original exponent.
        """
        self.top_method_orig = top
        self.bot_method_orig = bot
        self.top_method = top
        self.bot_method = bot
        self.exponent_orig = float(exp)
        self.exponent = float(exp)

    def populate_from_qrev_mat(self, transect):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        transect: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        if hasattr(transect, 'extrap'):
            self.top_method_orig = transect.extrap.topMethodOrig
            self.bot_method_orig = transect.extrap.botMethodOrig
            self.exponent_orig = transect.extrap.exponentOrig
            self.top_method = transect.extrap.topMethod
            self.bot_method = transect.extrap.botMethod
            self.exponent = transect.extrap.exponent

    def set_extrap_data(self, top, bot, exp):
        """Store new extrapolation settings

        Parameters
        ----------
        top: str
            New top extrapolation method.
        bot: str
            New bottom extrapolation method.
        exp: float
            New exponent.
        """
        self.top_method = top
        self.bot_method = bot
        self.exponent = exp
        
    def set_property(self, prop, setting):
        """Allows setting any property.

        Parameters
        ----------
        prop: str
            Name of property.
        setting:
            New setting.
        """
        setattr(self, prop, setting)
