import numpy as np
import scipy.stats as sp


def cosd(angle):
    """Compute cosine of angle in degrees.

    Parameters
    ----------
    angle: float
        Angle in degrees
    """
    
    return np.cos(np.pi * angle/180)


def sind(angle):
    """Compute sine of angle in degrees.

        Parameters
        ----------
        angle: float
            Angle in degrees
        """
    
    return np.sin(np.pi * angle/180)


def tand(angle):
    """Compute tangent of angle in degrees.

        Parameters
        ----------
        angle: float
            Angle in degrees
        """
    
    return np.tan(np.pi * angle/180)


def arctand(angle):
    """Compute arctangent of angle in degrees.

        Parameters
        ----------
        angle: float
            Angle in degrees
        """
    
    return np.arctan(angle) * 180/np.pi


def cart2pol(x, y):
    """Convert cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x: np.array(float)
        x coordinate
    y: np.array(float)
        y coordinate

    Returns
    -------
    phi: float
        Angle in radians
    rho: float
        Magnitude
    """
    
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    return phi, rho


def pol2cart(phi, rho):
    """Convert polar coordinates to cartesian coordinates.

        Parameters
        ----------
        phi: np.array(float)
            Angle in radians
        rho: np.array(float)
            Magnitude

        Returns
        -------
        x: float
            x coordinate
        y: float
            y coordinate

        """
    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    
    return x, y


def iqr(data):
    """This function computes the iqr consistent with Matlab

    Parameters
    ----------
    data: np.ndarray
        Data for which the statistic is required

    Returns
    -------
    sp_iqr: float
        Inner quartile range

    """

    # If 2-D array flatten to 1-D array
    if len(data.shape) > 1:
        data_1d = data.flatten()
    else:
        data_1d = data

    # Remove nan elements
    idx = np.where(np.logical_not(np.isnan(data_1d)))[0]
    data_1d = data_1d[idx]

    # Compute statistics
    q25, q50, q75 = sp.mstats.mquantiles(data_1d, alphap=0.5, betap=0.5)
    sp_iqr = q75 - q25

    return sp_iqr


def iqr_2d(data):
    """This function computes the iqr consistent with Matlab

    Parameters
    ----------
    data: np.ndarray
        Data for which the statistic is required

    Returns
    -------
    sp_iqr: float
        Inner quartile range

    """

    # Remove nan elements
    data = np.array(data)
    idx = np.where(np.logical_not(np.isnan(data)))[0]
    data = data[idx]

    # Compute statistics
    q25, q50, q75 = sp.mstats.mquantiles(data, alphap=0.5, betap=0.5)
    sp_iqr = q75 - q25
    return sp_iqr


def azdeg2rad(angle) -> float:
    """Converts an azimuth angle in degrees to radians.

    Parameters
    ----------
    angle: float, np.ndarray(float)
        Azimuth angle in degrees

    Returns
    -------
    direction: float, np.ndarray(float)
        Angle in radians
    """

    # Convert to radians
    direction = np.deg2rad(90-angle)

    # Create postive angle
    idx = np.where(direction < 0)[0]
    if len(idx) > 1:
        direction[idx] = direction[idx] + 2 * np.pi
    else:
        direction = direction + 2 * np.pi
        
    return direction


def rad2azdeg(angle) -> float:
    """Converts an angle in radians to an azimuth in degrees.

    Parameters
    ----------
    angle: float, np.ndarray(float)
        Angle in radians

    Returns
    -------
    deg: float, np.ndarray(float)
        Azimuth in degrees
    """

    if isinstance(angle, float):
        deg = np.rad2deg(angle)
        deg = 90 - deg
        if deg < 0:
            deg += 360
            
        return deg
    else:
        # Multiple values
        deg = np.rad2deg(angle)
        deg = 90 - deg
        sub_zero = np.where(nan_less(deg, 0))
        deg[sub_zero] = deg[sub_zero] + 360
        
        return deg


def nandiff(values):
    """Computes difference in consecutive values with handling of nans.

    Parameters
    ----------
    values: np.ndarray()
        1-D array of numbers

    Returns
    -------
    final_values: np.ndarray()
        1-D array of differences of consecutive non nan numbers
    """
    
    final_values = []
    for n in range(len(values) - 1):
        # Check for nan and add nan to final values
        if np.isnan(values[n]):
            final_values.append(np.nan)
        else:
            # Search for next non nan number and compute difference
            i = n + 1
            while np.isnan(values[i]) and i < len(values) - 1:
                i += 1
            
            final_values.append(values[i] - values[n])
        
    return np.array(final_values)


def valid_number(data_in) -> float:
    """Check to see if data_in can be converted to float.

    Parameters
    ----------
    data_in: str
        String to be converted to float

    Returns
    -------
    data_out: float
        Returns a float of data_in or nan if conversion is not possible
    """

    try:
        data_out = float(data_in)
    except ValueError:
        data_out = np.nan
    return data_out


def nans(shape, dtype=float):
    """Create array of nans.

    Parameters
    ----------
    shape: tuple
        Shape of array to be filled with nans
    dtype: type
        Type of array

    Returns
    -------
    a: np.ndarray(float)
        Array of nan
    """
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def checked_idx(transects):
    """Create list of transect indices of all checked transects.

    Parameters
    ----------
    transects: list
        List of TransectData objects

    Returns
    -------
    checked: list
        List of indices

    """
    checked = []
    for n, transect in enumerate(transects):
        if transect.checked:
            checked.append(n)

    return checked


def units_conversion(units_id='SI'):
    """Computes the units conversion from SI units used internally to the
    desired display units.

    Parameters
    ----------
    units_id: str
        String variable identifying units (English, SI) SI is the default.

    Returns
    -------
    units: dict
        dictionary of unit conversion and labels
    """

    if units_id == 'SI':
        units = {'L': 1,
                 'Q': 1,
                 'A': 1,
                 'V': 1,
                 'label_L': '(m)',
                 'label_Q': '(m3/s)',
                 'label_A': '(m2)',
                 'label_V': '(m/s)',
                 'ID': 'SI'}

    else:
        units = {'L': 1.0 / 0.3048,
                 'Q': (1.0 / 0.3048)**3,
                 'A': (1.0 / 0.3048)**2,
                 'V': 1.0 / 0.3048,
                 'label_L': '(ft)',
                 'label_Q': '(ft3/s)',
                 'label_A': '(ft2)',
                 'label_V': '(ft/s)',
                 'ID': 'English'}

    return units


def convert_temperature(temp_in, units_in, units_out) -> float:
    """Converts temperature from F to C or C to F.

    Parameters
    ==========
    temp_in: np.array
        temperature in units_in
    units_in: str
        C for Celcius or F for Fahrenheit
    units_out: str
        C for Celcius or F for Fahrenheit

    Returns
    =======
    temp_out: np.array
        temperature in units_out
    """

    temp_out = None
    if units_in == 'F':
        if units_out == 'C':
            temp_out = (temp_in - 32) * (5./9.)
        else:
            temp_out = temp_in

    elif units_in == 'C':
        if units_out == 'C':
            temp_out = temp_in
        else:
            temp_out = (temp_in * (9./5.)) + 32

    return temp_out


def nan_less_equal(data1, data2) -> bool:
    """Computes data1 <= data2 and sets all np.nan comparisons to False.

    Parameters
    ----------
    data1: np.array()
        Data arrray.
    data2: np.array()
        Data arrray.

    Returns
    -------
    :bool
        Result of comparison.
    """

    d3 = data2 - data1
    d3[np.isnan(d3)] = -999.
    return d3 >= 0


def nan_less(data1, data2) -> bool:
    """Computes data1 < data2 and sets all np.nan comparisons to False.

    Parameters
    ----------
    data1: np.array()
        Data arrray.
    data2: np.array()
        Data arrray.

    Returns
    -------
    :bool
        Result of comparison.
    """

    d3 = data2 - data1
    d3[np.isnan(d3)] = -999.
    return d3 > 0


def nan_greater_equal(data1, data2) -> bool:
    """Computes data1 >= data2 and sets all np.nan comparisons to False.

    Parameters
    ----------
    data1: np.array()
        Data arrray.
    data2: np.array()
        Data arrray.

    Returns
    -------
    :bool
        Result of comparison.
    """

    d3 = data1 - data2
    d3[np.isnan(d3)] = -999.
    return d3 >= 0


def nan_greater(data1, data2) -> bool:
    """Computes data1 < data2 and sets all np.nan comparisons to False.

    Parameters
    ----------
    data1: np.array()
        Data arrray.
    data2: np.array()
        Data arrray.

    Returns
    -------
    :bool
        Result of comparison.
    """

    d3 = data1 - data2
    d3[np.isnan(d3)] = -999.
    return d3 > 0
