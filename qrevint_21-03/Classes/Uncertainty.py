import numpy as np
from scipy.stats import t


class Uncertainty(object):
    """Computes the uncertainty of a measurement.

    Attributes
    ----------
    cov: float
        Coefficient of variation for all transects used in dicharge computation
    cov_95: float
        Coefficient of variation inflated to the 95% percent coverage
    invalid_95: float
        Estimated 95% uncertainty for dicharge in invalid bins and ensembles
    edges_95: float
        Estimated 95% uncertainty for the computed edge discharges
    extrapolation_95: float
        Estimated 95% uncertainty in discharge due to top and bottom extrapolations
    moving_bed_95: float
        Estimated 95% uncertainty due to moving-bed tests and conditions
    systematic: float
        Systematic error estimated at 1.5% at 1 sigma
    total_95: float
        Estimated 95% uncertainty in discharge using automated values
    cov_95_user: float
        User provided estimate of coefficient of variation inflated to the 95% percent coverage
    invalid_95_user: float
        User provided estimate of95% uncertainty for discharge in invalid bins and ensembles
    edges_95_user: float
        User provided estimate of 95% uncertainty for the computed edge discharges
    extrapolation_95_user: float
        User provided estimate of 95% uncertainty in discharge due to top and bottom extrapolations
    moving_bed_95_user: float
        User provided estimate of 95% uncertainty due to moving-bed tests and conditions
    systematic_user: float
        User provided estimate of systematic error estimated at 1.5% at 1 sigma
    total_95_user: float
        Estimated 95% uncertainty in discharge using user provide values to override automated values
    """

    def __init__(self):
        """Initializes the instance variables."""
        self.cov = None
        self.cov_95 = None
        self.invalid_95 = None
        self.edges_95 = None
        self.extrapolation_95 = None
        self.moving_bed_95 = None
        self.systematic = None
        self.total_95 = None
        self.cov_95_user = None
        self.invalid_95_user = None
        self.edges_95_user = None
        self.extrapolation_95_user = None
        self.moving_bed_95_user = None
        self.systematic_user = None
        self.total_95_user = None

    def compute_uncertainty(self, meas, cov_95_user=None, invalid_95_user=None, edges_95_user=None,
                            extrapolation_95_user=None, moving_bed_95_user=None, systematic_user=None):
        """Computes the uncertainty for the components of the discharge measurement
        using measurement data or user provided values.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        cov_95_user: float
            User provided estimate of coefficient of variation inflated to the 95% percent coverage
        invalid_95_user: float
            User provided estimate of95% uncertainty for discharge in invalid bins and ensembles
        edges_95_user: float
            User provided estimate of 95% uncertainty for the computed edge discharges
        extrapolation_95_user: float
            User provided estimate of 95% uncertainty in discharge due to top and bottom extrapolations
        moving_bed_95_user: float
            User provided estimate of 95% uncertainty due to moving-bed tests and conditions
        systematic_user: float
            User provided estimate of systematic error estimated at 1.5% at 1 sigma
        """

        # Use only checked discharges
        checked = []
        discharges = []
        for n in range(len(meas.transects)):
            checked.append(meas.transects[n].checked)
            if meas.transects[n].checked:
                discharges.append(meas.discharge[n])

        # Compute uncertainties from the data
        self.cov, self.cov_95 = self.uncertainty_q_random(discharges, 'total')
        self.invalid_95 = self.uncertainty_invalid_data(discharges)
        self.edges_95 = self.uncertainty_edges(discharges)
        self.extrapolation_95 = self.uncertainty_extrapolation(meas, discharges)
        self.moving_bed_95 = self.uncertainty_moving_bed(meas, checked)
        self.systematic = 1.5

        # Get user uncertainty estimates
        self.cov_95_user = cov_95_user
        self.invalid_95_user = invalid_95_user
        self.edges_95_user = edges_95_user
        self.extrapolation_95_user = extrapolation_95_user
        self.moving_bed_95_user = moving_bed_95_user
        self.systematic_user = systematic_user

        # Estimate the total measurement uncertainty
        self.estimate_total_uncertainty()

    def populate_from_qrev_mat(self, meas_struct):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        meas_struct: mat_struct
           Matlab data structure obtained from sio.loadmat
        """
        if hasattr(meas_struct, 'uncertainty'):
            self.cov = meas_struct.uncertainty.cov
            self.cov_95 = meas_struct.uncertainty.cov95
            self.invalid_95 = meas_struct.uncertainty.invalid95
            self.edges_95 = meas_struct.uncertainty.edges95
            self.extrapolation_95 = meas_struct.uncertainty.extrapolation95
            self.moving_bed_95 = meas_struct.uncertainty.movingBed95
            self.systematic = meas_struct.uncertainty.systematic
            self.total_95 = meas_struct.uncertainty.total95
            if not type(meas_struct.uncertainty.cov95User) is np.ndarray:
                self.cov_95_user = meas_struct.uncertainty.cov95User
            if not type(meas_struct.uncertainty.invalid95User) is np.ndarray:
                self.invalid_95_user = meas_struct.uncertainty.invalid95User
            if not type(meas_struct.uncertainty.edges95User) is np.ndarray:
                self.edges_95_user = meas_struct.uncertainty.edges95User
            if not type(meas_struct.uncertainty.extrapolation95User) is np.ndarray:
                self.extrapolation_95_user = meas_struct.uncertainty.extrapolation95User
            if not type(meas_struct.uncertainty.movingBed95User) is np.ndarray:
                self.moving_bed_95_user = meas_struct.uncertainty.movingBed95User
            if not type(meas_struct.uncertainty.systematicUser) is np.ndarray:
                self.systematic_user = meas_struct.uncertainty.systematicUser
            self.total_95_user = meas_struct.uncertainty.total95User

    def estimate_total_uncertainty(self):
        """Compute the uncertainty of the measurement using the automatically computed uncertainties and
        user overrides.
        """

        self.total_95 = 2.0 * ((self.cov_95 / 2)**2
                               + (self.invalid_95 / 2)**2
                               + (self.edges_95 / 2)**2
                               + (self.extrapolation_95 / 2)**2
                               + (self.moving_bed_95 / 2)**2
                               + self.systematic**2
                               )**0.5

        if self.cov_95_user is None:
            cov_95_user = self.cov_95
        else:
            cov_95_user = self.cov_95_user

        if self.invalid_95_user is None:
            invalid_95_user = self.invalid_95
        else:
            invalid_95_user = self.invalid_95_user

        if self.edges_95_user is None:
            edges_95_user = self.edges_95
        else:
            edges_95_user = self.edges_95_user

        if self.extrapolation_95_user is None:
            extrapolation_95_user = self.extrapolation_95
        else:
            extrapolation_95_user = self. extrapolation_95_user

        if self.moving_bed_95_user is None:
            moving_bed_95_user = self.moving_bed_95
        else:
            moving_bed_95_user = self.moving_bed_95_user

        if self.systematic_user is None:
            systematic_user = self.systematic
        else:
            systematic_user = self.systematic_user

        self.total_95_user = 2.0 * ((cov_95_user / 2)**2
                                    + (invalid_95_user / 2)**2
                                    + (edges_95_user / 2)**2
                                    + (extrapolation_95_user / 2)**2
                                    + (moving_bed_95_user / 2)**2
                                    + systematic_user**2
                                    )**0.5

    @staticmethod
    def get_array_attr(list_in, prop):
        """Create an array of the requested attribute from a list of objects containing the requested attribute.

        Parameters
        ----------
        list_in: list
            List of objects
        prop: str
            Attribute requested

        Returns
        -------
        data: np.ndarray()
            Array of the requested attributes from each object in list_in

        """
        # Create array of specified attribute
        data = []
        for item in list_in:
            data.append(getattr(item, prop))
        np.asarray(data)
        return data

    @staticmethod
    def uncertainty_q_random(discharges, prop):
        """Compute 95% random uncertainty for property of discharge.
        Uses simplified method for 2 transects.

        Parameters
        ----------
        discharges: list
            List of Discharge objects
        prop: str
            Attribute of Discharge objects

        Returns
        -------
        cov: float
            Coefficient of variation
        cov_95: float
            Coefficient of variation inflated to 95% value
        """
        n_max = len(discharges)
        if n_max > 0:
            # Create array of specified attribute
            data = Uncertainty.get_array_attr(discharges, prop)

            # Compute coefficient of variation
            cov = np.abs(np.nanstd(data, ddof=1) / np.nanmean(data)) * 100

            # Inflate the cov to the 95% value
            if n_max == 2:
                # Use the approximate method as taught in class to reduce the high coverage factor for 2 transects
                # and account for prior knowledge related to 720 second duration analysis
                cov_95 = cov * 3.3
            else:
                # Use Student's t to inflate COV for n > 2
                cov_95 = t.interval(0.95, n_max-1)[1] * cov / n_max**0.5
        else:
            cov = np.nan
            cov_95 = np.nan

        return cov, cov_95

    @staticmethod
    def uncertainty_edges(discharges):
        """Compute uncertainty of edge discharge. Currently assuming random plus bias
        is within 30% of actual value.

        Parameters
        ----------
        discharges: list
            List of Discharge objects

        Returns
        -------
        edge_uncertainty: float
            95% uncertainty in discharge due to edge estimates
        """

        # Compute mean discharge values for total, left, and right
        mean_q = np.nanmean(Uncertainty.get_array_attr(discharges, 'total'))
        mean_left = np.nanmean(Uncertainty.get_array_attr(discharges, 'left'))
        mean_right = np.nanmean(Uncertainty.get_array_attr(discharges, 'right'))

        # Compute combined edge uncertainty
        percent_edge = ((np.abs(mean_left) + np.abs(mean_right)) / mean_q) * 100
        edge_uncertainty = percent_edge * 0.3

        return edge_uncertainty

    @staticmethod
    def uncertainty_extrapolation(meas, discharges):
        """Compute the uncertainty of the top and bottom extrapolations.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        discharges: list
            List of Discharge objects

        Returns
        -------
        extrapolation_uncertainty: float
            95% uncertainty due to top and bottom extrapolation estimates
        """

        # Compute mean total uncorrected discharge
        q_selected = np.nanmean(Uncertainty.get_array_attr(discharges, 'total_uncorrected'))

        # Create array of discharges from the various extrapolation methods
        q_possible = np.array([meas.extrap_fit.q_sensitivity.q_pp_mean,
                               meas.extrap_fit.q_sensitivity.q_pp_opt_mean,
                               meas.extrap_fit.q_sensitivity.q_cns_mean,
                               meas.extrap_fit.q_sensitivity.q_cns_opt_mean,
                               meas.extrap_fit.q_sensitivity.q_3p_ns_mean,
                               meas.extrap_fit.q_sensitivity.q_3p_ns_opt_mean])

        # Compute difference in discharges from the selected method
        q_diff = np.abs(q_possible - q_selected)

        # Sort differences
        percent_diff = np.sort(q_diff) / q_selected

        # Estimate the uncertainty as the average of the 4 smallest differences
        extrapolation_uncertainty = np.nanmean(percent_diff[1:5]) * 100

        return extrapolation_uncertainty

    @staticmethod
    def uncertainty_invalid_data(discharges):
        """Computes an estimate of the uncertainty for the discharge computed for invalid bins and ensembles.

        Parameters
        ----------
        discharges: list
            List of Discharge objects

        Returns
        -------
        invalid_data_uncertainty: float
            95% uncertainty due to estimating invalid data
        """

        # Compute mean discharges
        q_mean = np.nanmean(Uncertainty.get_array_attr(discharges, 'total'))
        q_cells = np.nanmean(Uncertainty.get_array_attr(discharges, 'int_cells'))
        q_ensembles = np.nanmean(Uncertainty.get_array_attr(discharges, 'int_ens'))

        # Compute percentages
        percent_cells = (q_cells / q_mean) * 100
        percent_ensembles = (q_ensembles / q_mean) * 100

        # Compute uncertainty for combined invalid cells and ensembles
        invalid_data_uncertainty = (np.abs(percent_cells) + np.abs(percent_ensembles)) * 0.2

        return invalid_data_uncertainty

    @staticmethod
    def uncertainty_moving_bed(meas, checked):
        """Estimates the 95% uncertainty of the discharge due to a moving-bed and navigation reference.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        checked: list
            Logical list of transects used to compute final discharge

        Returns
        -------
        moving_bed_uncertainty: float
            95% uncertainty associated with moving-bed conditions
        """

        if np.any(checked) and meas.transects[checked.index(1)].boat_vel.selected == 'bt_vel':
            # Boat velocity based on bottom track, moving-bed possible
            if len(meas.mb_tests) > 0:
                # Moving_bed tests recorded
                user_valid = []
                quality = []
                moving_bed = []
                used = []
                for test in meas.mb_tests:
                    user_valid.append(test.user_valid)
                    if test.test_quality == 'Errors':
                        quality.append(False)
                    else:
                        quality.append(True)
                    moving_bed.append(test.moving_bed)
                    used.append(test.use_2_correct)

                # Check to see if there are any valid tests
                if np.any(np.logical_and(np.asarray(quality), np.asarray(user_valid))):
                    # Check to see if the valid tests indicate a moving bed
                    moving_bed_bool = []
                    for result in moving_bed:
                        if result == 'Yes':
                            moving_bed_bool.append(True)
                        else:
                            moving_bed_bool.append(False)
                    valid_moving_bed = np.logical_and(quality, np.asarray(moving_bed_bool))
                    if np.any(valid_moving_bed):
                        # Check to see that a correction was used
                        if np.any(np.logical_and(valid_moving_bed, np.asarray(used))):
                            # Moving-bed exists and correction applied
                            moving_bed_uncertainty = 1.5
                        else:
                            # Moving-bed exists and no correction applied
                            moving_bed_uncertainty = 3
                    else:
                        # Valid tests indicated no moving bed
                        moving_bed_uncertainty = 1
                else:
                    moving_bed_uncertainty = 3
            else:
                # No moving bed tests
                moving_bed_uncertainty = 3
        else:
            # GPS used as boat velocity reference
            moving_bed_uncertainty = 0

        return moving_bed_uncertainty
