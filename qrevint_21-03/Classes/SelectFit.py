import numpy as np
from Classes.FitData import FitData


class SelectFit(object):
    """Class automates the extrapolation method selection information.

    Attributes
    ----------
    fit_method: str
        User selected method Automatic or Manual
    top_method: str
        Top extrapolation method
    bot_method: str
        Bottom extrapolation method
    exponent: float
        Power fit exponent
    exp_method: str
        Method to determine exponent (default, optimize, or manual)
    u: np.array(float)
        Fit values of the variable
    u_auto: np.array(float)
        Fit values from automatic fit
    z: np.array(float)
        Distance from the streambed for fit variable
    z_auto: np.array(float)
        z values for automtic fit
    residuals: np.array(float)
        Residuals from fit
    coef: float
        Power fit coefficient
    bot_method_auto: str
        Selected extrapolation for top
    top_method_auto: str
        Selected extrapolation for bottom
    exponent_auto: float
        Selected exponent
    top_fit_r2: float
        Top fit custom r^2
    top_max_diff: float
        Maximum difference between power and 3-pt at top
    bot_diff: float
        Difference between power and no slop at z = 0.1
    bot_r2: float
        Bottom fit r^2
    fit_r2: float
        Selected fit of selected power/no slip fit
    ns_exponent: float
        No slip optimized exponent
    pp_exponent: float
        Power Power optimized exponent
    top_r2: float
        r^2 for linear fit of top 4 cells
    rsqr: float
        Adjusted r^2 for optimized exponent
    exponent_95_ci: np.array(float)
        95% confidence intervals for optimized exponent
    data_type: str
        Type of data (v, q, V, or Q)
    """

    def __init__(self):
        """Intialize object and instance variables."""

        self.fit_method = 'Automatic'  # User selected method Automatic or Manual
        self.top_method = 'Power'
        self.bot_method = 'Power'
        self.exponent = '0.1667'
        self.exp_method = None
        self.u = None
        self.u_auto = None
        self.z = None
        self.z_auto = None
        self.residuals = np.array([])
        self.coef = 0
        self.bot_method_auto = 'Power'  # Selected extrapolation for top
        self.top_method_auto = 'Power'  # Selected extrapolation for bottom
        self.exponent_auto = 0.1667  # Selected exponent
        self.top_fit_r2 = 0  # Top fit custom r^2
        self.top_max_diff = 0  # Maximum difference between power and 3-pt at top
        self.bot_diff = 0  # Difference between power and no slop at z = 0.1
        self.bot_r2 = 0  # Bottom fit r^2
        self.fit_r2 = 0  # Selected fit of selected power/no slip fit
        self.ns_exponent = 0.1667  # No slip optimized exponent
        self.pp_exponent = 0.1667  # Power Power optimized exponent
        self.top_r2 = 0
        self.rsqr = 0
        self.exponent_95_ci = 0
        self.data_type = 'q'

    def populate_data(self, normalized, fit_method, top=None, bot=None, exponent=None):
        """Determine selected fit.

        Parameters
        ----------
        normalized: NormData
            Object of NormData
        fit_method: str
            Fit method (Automatic or Manual)
        top: str
            Top extrapolation method
        bot: str
            Bottom extrapolation method
        exponent: float
            Exponent for extrapolation method
        """

        valid_data = np.squeeze(normalized.valid_data)

        # Store data in properties to object
        self.fit_method = fit_method
        self.data_type = normalized.data_type

        update_fd = FitData()

        if fit_method == 'Automatic':
            # Compute power fit with optimized exponent as reference to determine
            # if constant no slip will be more appropriate
            ppobj = FitData()
            ppobj.populate_data(norm_data=normalized,
                                top='Power',
                                bot='Power',
                                method='optimize')

            # Store results in object
            self.pp_exponent = ppobj.exponent
            self.residuals = ppobj.residuals
            self.rsqr = ppobj.r_squared
            self.exponent_95_ci = ppobj.exponent_95_ci

            # Begin automatic fit

            # More than 6 cells are required to compute an optimized fit.  For fewer
            # than 7 cells the default power/power fit is selected due to lack of sufficient
            # data for a good analysis
            if len(self.residuals) > 6:
                # Compute the difference between the top two cells of data and the optimized power fit
                top2 = np.nansum(normalized.unit_normalized_med[valid_data[-2:]]
                                 - ppobj.coef * normalized.unit_normalized_z[valid_data[-2:]] ** ppobj.exponent)

                # Compute the difference between the bottom two cells of data and the optimized power fit
                bot2 = np.nansum(normalized.unit_normalized_med[valid_data[:2]]
                                 - ppobj.coef * normalized.unit_normalized_z[valid_data[:2]] ** ppobj.exponent)

                # Compute the difference between the middle two cells of data and the optimized power fit
                mid1 = int(np.floor(len(np.isnan(valid_data) == False) / 2)) - 1

                mid2 = np.nansum(normalized.unit_normalized_med[valid_data[mid1:mid1 + 2]]
                                 - ppobj.coef * normalized.unit_normalized_z[valid_data[mid1:mid1 + 2]]
                                 ** ppobj.exponent)

                self.top_method_auto = 'Power'
                self.bot_method_auto = 'Power'

                # Evaluate difference in data and power fit at water surface using a linear fit through the top 4
                # median cells and save results
                y = normalized.unit_normalized_med[valid_data[:4]]
                #             x = sm.add_constant(x)
                x = normalized.unit_normalized_z[valid_data[:4]]

                coeffs = np.polyfit(x, y, 1)
                resid = y - (coeffs[0]*x + coeffs[1])
                corr = np.corrcoef(x, y)[0, 1]
                self.top_fit_r2 = 1 - (np.sum(resid ** 2) / np.mean(np.abs(resid)))
                self.top_r2 = corr**2

                # Evaluate overall fit
                # If the optimized power fit does not have an r^2 better than 0.8 or if the optimized
                # exponent if 0.1667 falls within the 95% confidence interval of the optimized fit,
                # there is insufficient justification to change the exponent from 0.1667
                if (ppobj.r_squared < 0.8) or ((0.1667 > self.exponent_95_ci[0]) and (0.1667 < self.exponent_95_ci[1])):
                    # If an optimized exponent cannot be justified the linear fit is used to determine if a constant
                    # fit at the top is a better alternative than a power fit.  If the power fit is the better
                    # alternative the exponent is set to the default 0.1667 and the data is refit
                    if np.abs(self.top_fit_r2 < 0.8 or self.top_r2 < 0.9):
                        ppobj = FitData()
                        ppobj.populate_data(norm_data=normalized,
                                            top='Power',
                                            bot='Power',
                                            method='Manual',
                                            exponent=0.1667)

                # Evaluate fit of top and bottom portions of the profile
                # Set save selected exponent and associated fit statistics
                self.exponent_auto = ppobj.exponent
                self.fit_r2 = ppobj.r_squared

                # Compute the difference at the water surface between a linear fit of the top 4 measured cells
                # and the best selected power fit of the whole profile
                self.top_max_diff = ppobj.u[-1] - np.sum(coeffs)

                # Evaluate the difference at the bottom between power using the whole profile and power using
                # only the bottom third
                ns_fd = FitData()
                ns_fd.populate_data(normalized, 'Constant', 'No Slip', 'Optimize')
                self.ns_exponent = ns_fd.exponent
                self.bot_r2 = ns_fd.r_squared
                self.bot_diff = ppobj.u[np.round(ppobj.z, 2) == 0.1][0] \
                    - ns_fd.u[np.round(ns_fd.z, 2) == 0.1][0]

                # Begin automatic selection logic
                # -----------------------------------

                # A constant no slip fit condition is selected if:
                #
                # 1)The top of the power fit doesn't fit the data well.
                # This is determined to be the situation when
                # (a) the difference at the water surface between the
                # linear fit and the power fit is greater than 10% and
                # (b) the difference is either positive or the difference
                # of the top measured cell differs from the best
                # selected power fit by more than 5%.
                top_condition = (np.abs(self.top_max_diff) > 0.1 and ((self.top_max_diff > 0)
                                 or np.abs(normalized.unit_normalized_med[valid_data[0]] - ppobj.u[-1]) > 0.05))

                # OR

                # 2) The bottom of the power fit doesn't fit the data
                # well. This is determined to be the situation when (a)
                # the difference between and optimized no slip fit
                # and the selected best power fit of the whole profile
                # is greater than 10% and (b) the optimized on slip fit has
                # an r^2 greater than 0.6.
                bottom_condition = ((np.abs(self.bot_diff) > 0.1) and self.bot_r2 > 0.6)

                # OR

                # 3) Flow is bidirectional. The sign of the top of the
                # profile is different from the sign of the bottom of
                # the profile.
                bidirectional_condition = (np.sign(normalized.unit_normalized_med[valid_data[0]])
                                           != np.sign(normalized.unit_normalized_med[valid_data[-1]]))
                # OR

                # 4) The profile is C-shaped. This is determined by
                # (a) the sign of the top and bottom difference from
                # the best selected power fit being different than the
                # sign of the middle difference from the best selected
                # power fit and (b) the combined difference of the top
                # and bottom difference from the best selected power
                # fit being greater than 10%.
                c_shape_condition = (np.sign(bot2) * np.sign(top2) == np.sign(mid2) and np.abs(bot2 + top2) > 0.1)

                if top_condition or bottom_condition or bidirectional_condition or c_shape_condition:

                    # Set the bottom to no slip
                    self.bot_method_auto = 'No Slip'
                    # If the no slip fit with an optimized exponent does not have r^2 better than 0.8 use
                    # the default 0.1667 for the no slip exponent
                    if ns_fd.r_squared > 0.8:
                        self.exponent_auto = ns_fd.exponent
                        self.fit_r2 = ns_fd.r_squared
                    else:
                        self.exponent_auto = 0.1667
                        self.fit_r2 = np.nan

                    # Use the no slip 95% confidence intervals if they are available
                    if ns_fd.exponent_95_ci is not None and np.all(
                            np.isnan(ns_fd.exponent_95_ci) == False):
                        self.exponent_95_ci[0] = ns_fd.exponent_95_ci[0]
                        self.exponent_95_ci[1] = ns_fd.exponent_95_ci[1]
                    else:
                        self.exponent_95_ci[0] = np.nan
                        self.exponent_95_ci[1] = np.nan

                    # Set the top method to constant
                    self.top_method_auto = 'Constant'

                else:

                    # Leave fit power/power and set the best selected optimized exponent as the automatic fit exponent
                    self.exponent_auto = ppobj.exponent

            else:

                # If the data are insufficient for a valid analysis use the power/power fit
                # with the default 0.1667 exponent
                self.top_method_auto = 'Power'
                self.bot_method_auto = 'Power'
                self.exponent_auto = 0.1667
                self.ns_exponent = 0.1667

            # Update the fit using the automatically selected methods
            update_fd.populate_data(norm_data=normalized,
                                    top=self.top_method_auto,
                                    bot=self.bot_method_auto,
                                    method='Manual',
                                    exponent=self.exponent_auto)
            self.u = update_fd.u
            self.u_auto = update_fd.u
            self.z_auto = update_fd.z
            self.z = update_fd.z

        elif fit_method == 'Manual':

            # Identify changes in fit settings
            if top is None:
                top = self.top_method
            if bot is None:
                bot = self.bot_method
            if exponent is None:
                exponent = self.exponent

            # Update fit with manual settings
            update_fd.populate_data(norm_data=normalized,
                                    top=top,
                                    bot=bot,
                                    method=fit_method,
                                    exponent=exponent)
            self.u = update_fd.u
            self.z = update_fd.z

        # Store fit data in object
        self.top_method = update_fd.top_method
        self.bot_method = update_fd.bot_method
        self.exponent = update_fd.exponent
        self.coef = update_fd.coef
        self.exp_method = update_fd.exp_method
        self.residuals = update_fd.residuals

    @staticmethod
    def qrev_mat_in(mat_data):
        """Processes the Matlab data structure to obtain a list of NormData objects containing transect
           data from the Matlab data structure.

       Parameters
       ----------
       mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat

       Returns
       -------
       norm_data: list
           List of NormData objects
       """
        fit_data = []
        if hasattr(mat_data, 'selFit'):
            for n, data in enumerate(mat_data.selFit):
                temp = SelectFit()
                temp.populate_from_qrev_mat(data, mat_data.normData[n])
                fit_data.append(temp)
        return fit_data

    def populate_from_qrev_mat(self, mat_data, norm_data):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat
        norm_data: NormData
            Object of NormData
        """

        self.fit_method = mat_data.fitMethod
        self.top_method = mat_data.topMethod
        self.bot_method = mat_data.botMethod
        self.exponent = mat_data.exponent
        self.exp_method = mat_data.expMethod
        self.u = mat_data.u
        self.u_auto = mat_data.uAuto
        self.z = mat_data.z
        self.z_auto = mat_data.zAuto
        self.residuals = mat_data.residuals
        self.coef = mat_data.coef
        self.bot_method_auto = mat_data.botMethodAuto
        self.top_method_auto = mat_data.topMethodAuto
        self.exponent_auto = mat_data.exponentAuto
        self.top_fit_r2 = mat_data.topfitr2
        self.top_max_diff = mat_data.topmaxdiff
        self.bot_diff = mat_data.botdiff
        self.bot_r2 = mat_data.botrsqr
        self.fit_r2 = mat_data.fitrsqr
        self.ns_exponent = mat_data.nsexponent
        self.pp_exponent = mat_data.ppexponent
        self.top_r2 = mat_data.topr2
        self.rsqr = mat_data.rsqr
        self.exponent_95_ci = mat_data.exponent95confint
        self.data_type = norm_data.dataType
