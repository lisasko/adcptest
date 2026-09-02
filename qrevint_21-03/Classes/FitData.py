import numpy as np
# from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
from scipy.stats import t


class FitData(object):
    """Class to compute top and bottom extrapolation methods and associated statistics.

     Data required for the constructor method include data of class
     NormData, threshold for the minimum number of points for a valid
     median, top extrapolation method, bottom extrapolation method, type
     of fit, and if a manual fit, the exponent.

    Attributes
    ----------
    self.file_name: str
        Name of transect file
    top_method: str
        Top extrapolation method
    bot_method: str
        Bottom extrapolation method
    coef: float
        Power fit coefficient
    exponent: float
        Power fit exponent
    u: np.array(float)
        Fit values of the variable
    u_auto: np.array(float)
        Fit values from automatic fit
    z_auto: np.array(float)
        z values for automtic fit
    z: np.array(float)
        Distance from the streambed for fit variable
    exp_method: str
        Method to determine exponent (default, optimize, or manual)
    data_type: str
        Type of data (v, q, V, or Q)
    exponent_95_ci: float
        95% confidence intervals for optimized exponent
    residuals: np.array(float)
        Residuals from fit
    r_squared: float
        R squared of model
    """

    def __init__(self):
        """Initialize object and instance variables."""

        self.file_name = None  # Name of transect file
        self.top_method = 'Power'  # Top extrapolation method
        self.bot_method = 'Power'  # Bottom extrapolation method
        self.coef = 0  # Power fit coefficient
        self.exponent = 0.1667  # Power fit exponent
        self.u = None  # Fit values of the variable
        self.u_auto = None  # Fit values from automatic fit
        self.z_auto = None  # z values for automtic fit
        self.z = None  # Distance from the streambed for fit variable
        self.exp_method = 'Power'  # Method to determine exponent (default, optimize, or manual)
        self.data_type = None  # Type of data (velocity or unit discharge)
        self.exponent_95_ci = 0  # 95% confidence intervals for optimized exponent
        self.residuals = np.array([])  # Residuals from fit
        self.r_squared = 0  # R squared of model

    def populate_data(self, norm_data, top, bot, method, exponent=None):
        """Computes fit and stores associated data.

        Parameters
        ----------
        norm_data: NormData
            Object of NormData
        top: str
            Top extrapolation method
        bot: str
            Bottom extrapolation method
        method:
            Method used to define the exponent (default, optimize, or manual), default is 1/6.
        exponent:
            Exponent for power or no slip fit methods.
        """

        avg_z = norm_data.unit_normalized_z
        y = norm_data.unit_normalized_med
        idxz = norm_data.valid_data
        zc = np.nan

        lower_bound = [-np.inf, 0.01]
        upper_bound = [np.inf, 1]
        bounds = None
        p0 = None
        uc = np.nan

        # Process data if available
        if len(idxz) > 0:
            idx_power = idxz

            # Create arrays for data fitting
            # Select median values to use in extrapolation methods selected and create
            # methods selected and create fir output data arrays

            # If bottom is No Slip, Power at top is not allowed
            if bot == 'No Slip':
                if top == 'Power':
                    top = 'Constant'

            fit_combo = ''.join([top, bot])
            if fit_combo == 'PowerPower':
                self.z = np.arange(0, 1.01, 0.01)
                zc = np.nan
                uc = np.nan
            elif fit_combo == 'ConstantPower':
                self.z = np.arange(0, np.max(avg_z[idxz]), 0.01)
                self.z = np.hstack([self.z, np.nan])
                zc = np.arange(np.max(avg_z[idxz]) + 0.01, 1.0, 0.01)
                uc = np.tile(y[idxz[0]], zc.shape)
            elif fit_combo == '3-PointPower':
                self.z = np.arange(0, np.max(avg_z[idxz]), 0.01)
                self.z = np.hstack([self.z, np.nan])
                # If less than 6 bins use constant at the top
                if len(idxz) < 6:
                    zc = np.arange(np.max(avg_z[idxz]) + 0.01, 1.0, 0.01)
                    uc = np.tile(y[idxz[0]], zc.shape)
                else:
                    p = np.polyfit(avg_z[idxz[0:3]], y[idxz[0:3]], 1)
                    zc = np.arange(np.max(avg_z[idxz]) + 0.01, 1.0, 0.01)
                    # zc = zc.T
                    uc = zc * p[0] + p[1]

            elif fit_combo == 'ConstantNo Slip':
                # Optimize constant / no slip if sufficient cells are available
                if method.lower() == 'optimize':
                    idx = idxz[int(1+len(idxz) - np.floor(len(avg_z[idxz]) / 3) - 1)::]
                    if len(idx) < 4:
                        method = 'default'

                # Compute Constant / No Slip using WinRiver II and
                # RiverSurveyor Live default cells
                else:
                    idx = np.where(avg_z[idxz] <= .2)[0]
                    if len(idx) < 1:
                        idx = idxz[-1]
                    else:
                        idx = idxz[idx]

                # Configures u and z arrays
                idxns = np.array([idx]).T
                self.z = np.arange(0, avg_z[idxns[0]], 0.01)
                self.z = np.hstack([self.z, [np.nan]])
                idx_power = idx
                zc = np.arange(np.max(avg_z[idxz]) + 0.01, 1.00, 0.01)
                uc = np.tile(y[idxz[0]], zc.shape)

            elif fit_combo == '3-PointNo Slip':
                # Optimize 3-Point / no slip if sufficient cells are available
                if method.lower() == 'optimize':
                    idx = idxz[int(1 + len(idxz) - np.floor(len(avg_z[idxz])) / 3) - 1::]
                    if len(idx) < 4:
                        method = 'default'

                # Compute 3-Point / No Slip using WinRiver II and
                # RiverSurveyor Live default cells
                else:
                    idx = np.where(avg_z[idxz] <= .2)[0]
                    if len(idx) < 1:
                        idx = idxz[-1]
                    else:
                        idx = idxz[idx]

                # Configures u and z arrays
                idxns = np.array([idx]).T
                self.z = np.arange(0, avg_z[idxns[0]], 0.01)
                self.z = np.hstack([self.z, [np.nan]])
                idx_power = idx
                # If less than 6 bins use constant at the top
                if len(idxz) < 6:
                    zc = np.arange(np.max(idxz) + 0.01, 1.0, 0.01)
                    uc = np.tile(y[idxz[0]], zc.shape)
                else:
                    p = np.polyfit(avg_z[idxz[0:3]], y[idxz[0:3]], 1)
                    zc = np.arange(np.max(avg_z[idxz]) + 0.01, 1.0, 0.01)
                    uc = zc * p[0] + p[1]

            # Compute exponent
            zfit = avg_z[idx_power]
            yfit = y[idx_power]

            # Check data validity
            ok_ = np.logical_and(np.isfinite(zfit), np.isfinite(yfit))

            self.exponent = np.nan
            self.exponent_95_ci = np.nan
            self.r_squared = np.nan
            fit_func = 'linear'

            lower_method = method.lower()

            if lower_method == 'manual':
                fit_func = 'linear'
                self.exponent = exponent
                bounds = None
                p0 = None

            elif lower_method == 'default':
                fit_func = 'linear'
                self.exponent = 1./6.
                bounds = None
                p0 = None

            elif lower_method == 'optimize':
                fit_func = 'power'
                bounds = [lower_bound, upper_bound]
                strt = yfit[ok_]
                p0 = [strt[-1], 1./6]

            fit_funcs = {
                'linear': lambda x, a: a * x**self.exponent,
                'power': lambda x, a, b: a * x**b
            }

            if ok_.size > 1:
                if bounds is not None:
                    popt, pcov = curve_fit(fit_funcs[fit_func],
                                           zfit, yfit, p0=p0, bounds=bounds)
                else:
                    popt, pcov = curve_fit(fit_funcs[fit_func],
                                           zfit, yfit, p0=p0)

                # Extract exponent and confidence intervals from fit
                if lower_method == 'optimize':
                    self.exponent = popt[1]
                    if self.exponent is None or self.exponent < 0.05:
                        self.exponent = 0.05

                if len(zfit[ok_]) > 2:

                    n = len(zfit)    # number of data points

                    t_val = t.ppf(.975, n-2)

                    # Get 95% confidence intervals
                    lower = (popt[-1] - t_val * np.sqrt(np.diag(pcov)[-1]))
                    upper = (popt[-1] + t_val * np.sqrt(np.diag(pcov)[-1]))
                    self.exponent_95_ci = np.vstack([lower, upper])

                    # Get the rsquared for the model
                    ss_tot = np.sum((y[idx_power] - np.mean(yfit))**2)
                    ss_res = np.sum((y[idx_power] - fit_funcs[fit_func](zfit, *popt))**2)
                    self.r_squared = 1 - (ss_res/ss_tot)
                else:
                    self.exponent_95_ci = np.nan
                    self.r_squared = np.nan

            # Fit power curve to appropriate data
            self.coef = ((self.exponent + 1) * 0.05 * np.nansum(y[idx_power])) / \
                np.nansum(((avg_z[idx_power] + (0.5 * 0.05))**(self.exponent + 1)
                           - ((avg_z[idx_power] - (0.5 * 0.05))**(self.exponent + 1))))

            # Compute residuals
            self.residuals = y[idx_power] - self.coef * avg_z[idx_power]**self.exponent
            if self.residuals is None:
                self.residuals = np.array([])

            # Compute values (velocity or discharge) based on exponent and compute coefficient
            self.u = self.coef * self.z**self.exponent
            if type(zc) == np.ndarray:
                self.u = np.append(self.u, uc)
                self.z = np.append(self.z, zc)

            # Assign variables to object properties
            self.file_name = norm_data.file_name
            self.top_method = top
            self.bot_method = bot
            self.exp_method = method
            self.data_type = norm_data.data_type

        else:
            # If not data are valid simply apply methods
            self.exponent = np.nan
            self.exponent_95_ci = [np.nan, np.nan]
            self.r_squared = np.nan
            self.file_name = norm_data.file_name
            self.top_method = top
            self.bot_method = bot
            self.exp_method = method
            self.data_type = norm_data.data_type
