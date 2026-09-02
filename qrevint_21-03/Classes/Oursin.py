import pandas as pd
import copy
from Classes.BoatStructure import *
from Classes.QComp import QComp
from scipy.stats import t
# from profilehooks import profile
from MiscLibs.common_functions import cosd, sind


class Oursin(object):
    """Computes the uncertainty of a measurement using Oursin method.

    Attributes
    ----------
    bot_meth: list
        List that contains the method proposed by Extrap for each transect
    exp_95ic_min: list
        List that contains the min range of 95% interval if power-power method is used for transect
    exp_95ic_max: list
        List that contains the max range of 95% interval if power-power method is used for transect
    pp_exp: list
        List that contains the power-power exponent computed by Extrap for Power-Power transect only
    ns_exp: list
        List that contains the no-slip exponent computed by Extrap for No-Slip method transect only
    exp_pp_min: list
        Minimum power-power exponent used for simulating possible discharge
    exp_pp_max: list
        Maximum power-power exponent used for simulating possible discharge
    exp_ns_min: list
        Minimum no-slip exponent used for simulating possible discharge
    exp_ns_max: list
        Maximum no-slip exponent used for simulating possible discharge
    d_right_error_min: list
        List that contains the minimum right distance (in m) used for simulating the discharge for each transect
    d_left_error_min: list
        List that contains the minimum left distance (in m) used for simulating the discharge for each transect
    d_right_error_max: list
        List that contains the maximum right distance (in m) used for simulating the discharge for each transect
    d_left_error_max: list
        List that contains the maximum left distance (in m) used for simulating the discharge for each transect
    draft_error_list: list
        List that contains the draft (in cm) used for simulating the discharge for each transect
    u_syst_list: list
        List that contains the computed systematic uncertainty (68%) for each transect
    u_compass_list: list
        List that contains the computed uncertainty (68%) due to compass error for each transect
    u_meas_list: list
        List that contains the computed measured area uncertainty (68%) for each transect
    u_ens_list: list
       List that contains the computed uncertainty (68%) due to limited number of ensemble for each transect
    u_movbed_list: list
       List that contains the estimated uncertainty (68%) due to moving bed for each transect
    u_invalid_water_list: list
        List that contains the computed uncertainty (68%) due to invalid water velocities for each transect
    u_invalid_boat_list: list
        List that contains the computed uncertainty (68%) due to invalid boat velocities for each transect
    u_invalid_depth_list: list
       List that contains the computed uncertainty (68%) due to invalid depths for each transect
    u_top_list: list
       List that contains the computed uncertainty (68%) due to top discharge extrapolation for each transect
    u_bot_list: list
       List that contains the computed uncertainty (68%) due to bottom discharge extrapolation for each transect
    u_left_list: list
       List that contains the computed uncertainty (68%) due to left discharge extrapolation for each transect
    u_right_list: list
       List that contains the computed uncertainty (68%) due to right discharge extrapolation for each transect
               self.u_compass_user_list = []
    u_syst_mean_user_list: list
        List that contains the user specified  systematic uncertainty (68%) for each transect
    u_compass_user_list: list
        List that contains user specified uncertainty (68%) due to compass error for each transect
    u_meas_mean_user_list: list
        List that contains the user specified measured area uncertainty (68%) for each transect
    u_ens_user_list: list
       List that contains the user specified uncertainty (68%) due to limited number of ensemble for each transect
    u_movbed_user_list: list
       List that contains the user specified uncertainty (68%) due to moving bed for each transect
    u_invalid_water_user_list: list
        List that contains the user specified uncertainty (68%) due to invalid water velocities for each transect
    u_invalid_boat_user_list: list
        List that contains the user specified uncertainty (68%) due to invalid boat velocities for each transect
    u_invalid_depth_user_list: list
       List that contains the user specified uncertainty (68%) due to invalid depths for each transect
    u_top_mean_user_list: list
       List that contains the user specified uncertainty (68%) due to top discharge extrapolation for each transect
    u_bot_mean_user_list: list
       List that contains the user specified uncertainty (68%) due to bottom discharge extrapolation for each transect
    u_left_mean_user_list: list
       List that contains the user specified uncertainty (68%) due to left discharge extrapolation for each transect
    u_right_mean_user_list: list
       List that contains the user specified uncertainty (68%) due to right discharge extrapolation for each transect
    cov_68: float
       Computed uncertainty (68%) due to coefficient of variation
    sim_original: DataFrame
        Discharges (total, and subareas) computed for the processed discharge
    sim_extrap_pp_16: DataFrame
        Discharges (total, and subareas) computed using power fit with 1/6th exponent
    sim_extrap_pp_min: DataFrame
        Discharges (total, and subareas) computed using power fit with minimum exponent
    sim_extrap_pp_max: DataFrame
        Discharges (total, and subareas) computed using power fit with maximum exponent
    sim_extrap_cns_16: DataFrame
        Discharges (total, and subareas) computed using constant no slip with 1/6th exponent
    sim_extrap_cns_min: DataFrame
        Discharges (total, and subareas) computed using constant no slip with minimum exponent
    sim_extrap_cns_max: DataFrame
        Discharges (total, and subareas) computed using constant no slip with maximum exponent
    sim_extrap_3pns_16: DataFrame
        Discharges (total, and subareas) computed using 3pt no slip with 1/6the exponent
    sim_extrap_3pns_opt: DataFrame
        Discharges (total, and subareas) computed using 3pt no slip with optimized exponent
    sim_edge_min: DataFrame
        Discharges (total, and subareas) computed using minimum edge q
    sim_edge_max: DataFrame
        Discharges (total, and subareas) computed using maximum edge q
    sim_draft_min: DataFrame
        Discharges (total, and subareas) computed using minimum draft
    sim_draft_max: DataFrame
        Discharges (total, and subareas) computed using maximum draft
    sim_cells_trdi: DataFrame
        Discharges (total, and subareas) computed using TRDI method for invalid cells
    sim_cells_above: DataFrame
        Discharges (total, and subareas) computed using cells above for invalid cells
    sim_cells_below: DataFrame
        Discharges (total, and subareas) computed using cells below for invalid cells
    sim_cells_before: DataFrame
        Discharges (total, and subareas) computed for using cells before for invalid cells
    sim_cells_after: DataFrame
        Discharges (total, and subareas) computed for using cells before for invalid cells
    nb_transects: float
        Number of transects used
    checked_idx: list
        List of indices of checked transects
    user_advanced_settings: dict
        Dictionary of user specified advanced settings
        exp_pp_min_user: float
            User specified minimum exponent for power fit
        exp_pp_max_user: float
            User specified maximum exponent for power fit
        exp_ns_min_user: float
            User specified minimum exponent for no slip fit
        exp_ns_max_user: float
            User specified maximum exponent for no slip fit
        draft_error_user: float
            User specified draft error in m
        dzi_prct_user: float
            User specified percent error in depth cell size
        right_edge_dist_prct_user: float
            User specified percent error in right edge distance
        left_edge_dist_prct_user: float
            User specified percent error in left edge distance
        gga_boat_user: float
            User specified standard deviation of boat velocities based on gga in m/s
        vtg_boat_user: float
            User specified standard deviation of boat velocities based on vtg in m/s
        compass_error_user: float
            User specified compass error in degrees
    default_advanced_settings: dict
        Dictionary of default values for advanced settings
        exp_pp_min: float
            Default minimum exponent for power fit
        exp_pp_max: float
            Default maximum exponent for power fit
        exp_ns_min: float
            Default minimum exponent for no slip fit
        exp_ns_max: float
            Default maximum exponent for no slip fit
        draft_error: float
            Default draft error in m
        dzi_prct: float
            Default percent error in depth cell size
        right_edge_dist_prct: float
            Default percent error in right edge distance
        left_edge_dist_prct: float
            Default percent error in left edge distance
        gga_boat: float
            Default standard deviation of boat velocities based on gga in m/s
        vtg_boat: float
            Default standard deviation of boat velocities based on vtg in m/s
        compass_error: float
            Default compass error in degrees
    user_specified_u: dict
        Dictionary of user specified uncertainties as standard deviation in percent
        u_syst_mean_user: float
            User specified uncertianty (bias) due to the system, in percent
        u_movbed_user: float
            User specified uncertianty (bias) due to the moving-bed conditions, in percent
        u_compass_user: float
            User specified uncertianty (bias) due to the compass error, in percent
        u_ens_user: float
            User specified uncertianty (bias) due to the number of ensembles collected, in percent
        u_meas_mean_user: float
            User specified uncertianty (random) of the measured portion of the cross section, in percent
        u_top_mean_user: float
            User specified uncertianty (bias) due to the top extrapolation, in percent
        u_bot_mean_user: float
            User specified uncertianty (bias) due to the bottom extrapolation, in percent
        u_right_mean_user: float
            User specified uncertianty (bias) due to the right edge discharge estimate, in percent
        u_left_mean_user: float
            User specified uncertianty (bias) due to the left edge discharge estimate, in percent
        u_invalid_boat_user: float
            User specified uncertianty (bias) due to invalid boat velocities, in percent
        u_invalid_depth_user
            User specified uncertianty (bias) due to invalid depths, in percent
        u_invalid_water_user: float
            User specified uncertianty (bias) due to invalid water velocities, in percent
    u: DataFrame
        DataFrame containing standard deviations in percent for each transect: u_syst, u_compass, u_movbed, u_ens,
        u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, u_cov, total, and total_95
     u_contribution_meas: DataFrame
        DataFrame containing measured discharge uncertainty contribution from: boat, water, depth, and dzi
    u_measurement: DataFrame
        DataFrame containing standard deviations in percent for the whole measurement: u_syst, u_compass, u_movbed,
        u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, u_cov, total, and total_95
    u_contribution_measurement: DataFrame
        DataFrame containing uncertainty contribution in percent from: u_syst, u_compass, u_movbed,
        u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, u_cov, and total
    u_nocov: DataFrame
        DataFrame containing standard deviations in percent for each transect, without COV: u_syst, u_compass, u_movbed,
        u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, total, and total_95
    u_measurement_nocov: DataFrame
        DataFrame containing uncertainty contribution in percent from: u_syst, u_compass, u_movbed,
        u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, and total
    u_user: DataFrame
        DataFrame containing standard deviations in percent for each transect: u_syst, u_compass, u_movbed, u_ens,
        u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, u_cov, total, and total_95
    u_measurement_user: DataFrame
        DataFrame containing standard deviations in percent for the whole measurement: u_syst, u_compass, u_movbed,
        u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, u_cov, total, and total_95
    u_contribution_measurement_user: DataFrame
        DataFrame containing uncertainty contribution in percent from: u_syst, u_compass, u_movbed,
        u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, u_cov, and total
    u_nocov_user: DataFrame
        DataFrame containing standard deviations in percent for each transect, without COV: u_syst, u_compass, u_movbed,
        u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, total, and total_95
    u_measurement_nocov_user: DataFrame
        DataFrame containing uncertainty contribution in percent from: u_syst, u_compass, u_movbed,
        u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, and total
    """

    def __init__(self):
        """Initialize class and instance variables."""

        # User provided parameters
        self.user_advanced_settings = {'exp_pp_min_user': None,
                                       'exp_pp_max_user': None,
                                       'exp_ns_min_user': None,
                                       'exp_ns_max_user': None,
                                       'draft_error_user': None,
                                       'dzi_prct_user': None,
                                       'right_edge_dist_prct_user': None,
                                       'left_edge_dist_prct_user': None,
                                       'gga_boat_user': None,
                                       'vtg_boat_user': None,
                                       'compass_error_user': None}

        self.default_advanced_settings = {'exp_pp_min': 'computed',
                                          'exp_pp_max': 'computed',
                                          'exp_ns_min': 'computed',
                                          'exp_ns_max': 'computed',
                                          'draft_error_m': 'computed',
                                          'dzi_prct': 0.5,
                                          'right_edge_dist_prct': 20,
                                          'left_edge_dist_prct': 20,
                                          'gga_boat_mps': 'computed',
                                          'vtg_boat_mps': 0.05,
                                          'compass_error_deg': 1}

        self.user_specified_u = {'u_syst_mean_user': None,
                                 'u_movbed_user': None,
                                 'u_compass_user': None,
                                 'u_ens_user': None,
                                 'u_meas_mean_user': None,
                                 'u_top_mean_user': None,
                                 'u_bot_mean_user': None,
                                 'u_right_mean_user': None,
                                 'u_left_mean_user': None,
                                 'u_invalid_boat_user': None,
                                 'u_invalid_depth_user': None,
                                 'u_invalid_water_user': None}

        # Extrap results
        self.bot_meth = []
        self.exp_95ic_min = []
        self.exp_95ic_max = []
        self.pp_exp = []
        self.ns_exp = []

        # Parameters used for computing the uncertainty
        self.exp_pp_min = None
        self.exp_pp_max = None
        self.exp_ns_min = None
        self.exp_ns_max = None
        self.d_right_error_min = []
        self.d_left_error_min = []
        self.d_right_error_max = []
        self.d_left_error_max = []
        self.draft_error_list = []

        # Terms computed by transect (list at 68% level)
        self.u_syst_list = []
        self.u_compass_list = []
        self.u_meas_list = []
        self.u_ens_list = []
        self.u_movbed_list = []
        self.u_invalid_water_list = []
        self.u_invalid_boat_list = []
        self.u_invalid_depth_list = []
        self.u_top_list = []
        self.u_bot_list = []
        self.u_left_list = []
        self.u_right_list = []

        self.u_syst_mean_user_list = []
        self.u_compass_user_list = []
        self.u_movbed_user_list = []
        self.u_meas_mean_user_list = []
        self.u_ens_user_list = []
        self.u_top_mean_user_list = []
        self.u_bot_mean_user_list = []
        self.u_left_mean_user_list = []
        self.u_right_mean_user_list = []
        self.u_invalid_boat_user_list = []
        self.u_invalid_depth_user_list = []
        self.u_invalid_water_user_list = []

        # Term computed for measurement
        self.cov_68 = None

        self.nb_transects = None
        self.checked_idx = []

        # --- Store results of all simulations in DataFrame
        self.sim_original = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot', 'q_left', 'q_right', 'q_middle'])
        self.sim_extrap_pp_16 = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_extrap_pp_opt = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_extrap_pp_min = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_extrap_pp_max = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_extrap_cns_16 = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_extrap_cns_opt = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_extrap_cns_min = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_extrap_cns_max = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_extrap_3pns_16 = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_extrap_3pns_opt = pd.DataFrame(columns=['q_total', 'q_top', 'q_bot'])
        self.sim_edge_min = pd.DataFrame(columns=['q_total', 'q_left', 'q_right'])
        self.sim_edge_max = pd.DataFrame(columns=['q_total', 'q_left', 'q_right'])
        self.sim_draft_min = pd.DataFrame(columns=['q_total', 'q_top', 'q_left', 'q_right'])
        self.sim_draft_max = pd.DataFrame(columns=['q_total', 'q_top', 'q_left', 'q_right'])
        self.sim_cells_trdi = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.sim_cells_above = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.sim_cells_below = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.sim_cells_before = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.sim_cells_after = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.sim_shallow = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.sim_depth_hold = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.sim_depth_next = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.sim_boat_hold = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.sim_boat_next = pd.DataFrame(columns=['q_total', 'q_middle'])
        self.u_contribution_meas = pd.DataFrame(columns=['boat', 'water', 'depth', 'dzi'])
        self.u = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top', 'u_bot',
                                       'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water', 'u_cov', 'total',
                                       'total_95'])
        self.u_measurement = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top',
                                                   'u_bot', 'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water',
                                                   'u_cov', 'total', 'total_95'])
        self.u_contribution = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top',
                                                    'u_bot', 'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water',
                                                    'u_cov', 'total'])
        self.u_contribution_measurement = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas',
                                                                'u_top', 'u_bot', 'u_left', 'u_right', 'u_boat',
                                                                'u_depth', 'u_water', 'u_cov', 'total'])
        self.u_nocov = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top', 'u_bot',
                                             'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water', 'total', 'total_95'])
        self.u_measurement_nocov = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top',
                                                         'u_bot', 'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water',
                                                         'total', 'total_95'])
        self.u_user = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top', 'u_bot',
                                            'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water', 'u_cov', 'total',
                                            'total_95'])
        self.u_measurement_user = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top',
                                                        'u_bot', 'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water',
                                                        'u_cov', 'total', 'total_95'])
        self.u_contribution_user = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top',
                                                         'u_bot', 'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water',
                                                         'u_cov', 'total'])
        self.u_contribution_measurement_user = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens',
                                                                     'u_meas', 'u_top', 'u_bot', 'u_left', 'u_right',
                                                                     'u_boat', 'u_depth', 'u_water', 'u_cov', 'total'])
        self.u_nocov_user = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top',
                                                  'u_bot', 'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water', 'total',
                                                  'total_95'])
        self.u_measurement_nocov_user = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas',
                                                              'u_top', 'u_bot', 'u_left', 'u_right', 'u_boat',
                                                              'u_depth', 'u_water', 'total', 'total_95'])

    # @profile
    def compute_oursin(self, meas):
        """Computes the uncertainty for the components of the discharge measurement
        using measurement data or user provided values.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        # Initialize lists
        self.checked_idx = []
        self.u_syst_list = []
        self.u_meas_list = []
        self.u_ens_list = []
        self.u_movbed_list = []
        self.u_invalid_water_list = []
        self.u_invalid_boat_list = []
        self.u_invalid_depth_list = []
        self.u_top_list = []
        self.u_bot_list = []
        self.u_left_list = []
        self.u_right_list = []

        # Prep data for computations
        self.data_prep(meas)
        self.compute_measurement_cov(meas=meas)

        # 1. Systematic terms + correction terms (moving bed)
        self.uncertainty_system()
        self.uncertainty_moving_bed(meas=meas)
        self.uncertainty_compass(meas=meas)

        # 2. Measured uncertainty
        self.uncertainty_measured_discharge(meas=meas)
        self.uncertainty_number_ensembles(meas)

        # 3. Run all the simulations to compute possible discharges
        self.run_simulations(meas)

        # 4. Compute uncertainty terms based on simulations and assuming a rectangular law
        self.uncertainty_top_discharge()
        self.uncertainty_bottom_discharge()
        self.uncertainty_left_discharge()
        self.uncertainty_right_discharge()
        self.uncertainty_invalid_depth_data()
        self.uncertainty_invalid_boat_data()
        self.uncertainty_invalid_water_data()

        # 6. Compute combined uncertainty
        self.u, self.u_measurement, self.u_contribution, self.u_contribution_measurement, \
            self.u_nocov, self.u_measurement_nocov = \
            self.compute_combined_uncertainty(u_syst=self.u_syst_list,
                                              u_compass=self.u_compass_list,
                                              u_movbed=self.u_movbed_list,
                                              u_meas=self.u_meas_list,
                                              u_ens=self.u_ens_list,
                                              u_top=self.u_top_list,
                                              u_bot=self.u_bot_list,
                                              u_left=self.u_left_list,
                                              u_right=self.u_right_list,
                                              u_boat=self.u_invalid_boat_list,
                                              u_depth=self.u_invalid_depth_list,
                                              u_water=self.u_invalid_water_list,
                                              cov_68=self.cov_68)

        self.u_user, self.u_measurement_user, self.u_contribution_user, self.u_contribution_measurement_user,\
            self.u_nocov_user, self.u_measurement_nocov_user = \
            self.compute_combined_uncertainty(u_syst=self.u_syst_mean_user_list,
                                              u_compass=self.u_compass_user_list,
                                              u_movbed=self.u_movbed_user_list,
                                              u_meas=self.u_meas_mean_user_list,
                                              u_ens=self.u_ens_user_list,
                                              u_top=self.u_top_mean_user_list,
                                              u_bot=self.u_bot_mean_user_list,
                                              u_left=self.u_left_mean_user_list,
                                              u_right=self.u_right_mean_user_list,
                                              u_boat=self.u_invalid_boat_user_list,
                                              u_depth=self.u_invalid_depth_user_list,
                                              u_water=self.u_invalid_water_user_list,
                                              cov_68=self.cov_68)

    @staticmethod
    def compute_combined_uncertainty(u_syst, u_compass, u_movbed, u_meas, u_ens, u_top, u_bot, u_left, u_right,
                                     u_boat, u_depth, u_water, cov_68):
        """Combined the uncertainty for each transect and for the measurement

        Parameters
        ----------
        u_syst: list
            List of system uncertainties for each transect
        u_compass: list
            List of uncertainties due to heading error
        u_movbed: list
            List of moving-bed uncertainties for each transect
        u_meas: list
            List of uncertainties for the measured portion for each transect
        u_ens: list
            List of uncertainties due to number of ensembles in each transect
        u_top: list
            List of uncertainties due to top extrapolation in each transect
        u_bot: list
            List of uncertainties due to the bottom extrapolation in each transect
        u_left: list
            List of uncertainties due to the left edge discharge in each transect
        u_right: list
            List of uncertainties due to the right edge discharge in each transect
        u_boat: list
            List of uncertainties due to invalid boat velocities
        u_depth: list
            List of uncertainties due to invalid depth velocities
        u_water: list
            List of uncertainties due to invalid water data in each transect
        cov_68: float
            Coefficient of variation for all transects

        Returns
        -------
        u_contribution_meas: DataFrame
            DataFrame containing measured discharge uncertainty contribution from: boat, water, depth, and dzi
        u: DataFrame
            DataFrame containing standard deviations in percent for each transect: u_syst, u_compass, u_movbed, u_ens,
            u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, u_cov, total, and total_95
        u_measurement: DataFrame
            DataFrame containing standard deviations in percent for the whole measurement: u_syst, u_compass, u_movbed,
            u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, u_cov, total, and total_95
        u_contribution_measurement: DataFrame
            DataFrame containing uncertainty contribution in percent from: u_syst, u_compass, u_movbed,
            u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, u_cov, and total
        u_nocov: DataFrame
            DataFrame containing standard deviations in percent for each transect, without COV: u_syst, u_compass,
            u_movbed, u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, total, and total_95
        u_measurement_nocov: DataFrame
            DataFrame containing uncertainty contribution in percent from: u_syst, u_compass, u_movbed,
            u_ens, u_meas, u_top, u_bot, u_left, u_right, u_boat, u_depth, u_water, and total
        """

        # Create a Dataframe with all computed uncertainty for each checked transect
        u = pd.DataFrame(columns=['u_syst', 'u_compass', 'u_movbed', 'u_ens', 'u_meas', 'u_top', 'u_bot',
                                  'u_left', 'u_right', 'u_boat', 'u_depth', 'u_water', 'u_cov'])
        u['u_syst'] = u_syst
        u['u_compass'] = u_compass
        u['u_movbed'] = u_movbed
        u['u_meas'] = u_meas
        u['u_ens'] = u_ens
        u['u_water'] = u_water
        u['u_top'] = u_top
        u['u_bot'] = u_bot
        u['u_left'] = u_left
        u['u_right'] = u_right
        u['u_cov'] = cov_68
        u['u_boat'] = u_boat
        u['u_depth'] = u_depth

        n_transects = len(u_ens)

        # Computations without use of cov
        u_nocov = u.drop(['u_cov'], axis=1)
        u2_nocov = u_nocov.pow(2)
        u2_measurement_nocov = u2_nocov.mean(axis=0, skipna=False).to_frame().T
        u_nocov['total'] = (u2_nocov.sum(axis=1, skipna=False) ** 0.5)
        u_nocov['total_95'] = u_nocov['total'] * 2
        u_nocov = u_nocov.mul(100)
        u2_random_nocov = u2_measurement_nocov['u_meas']
        u2_bias_nocov = u2_measurement_nocov.drop(['u_meas'], axis=1).sum(axis=1, skipna=False)

        u2_measurement_nocov['total'] = (1 / n_transects) * u2_random_nocov + u2_bias_nocov[0]
        u_measurement_nocov = u2_measurement_nocov ** 0.5
        u_measurement_nocov['total_95'] = u_measurement_nocov['total'] * 2
        u_measurement_nocov = u_measurement_nocov * 100

        # Convert uncertainty (68% level of confidence) into variance
        # Note that only variance is additive
        u2 = u.pow(2)
        u2_measurement = u2.mean(axis=0, skipna=False).to_frame().T
        # Combined uncertainty by transect
        # Sum of variance of each component, then sqrt, then multiply by 100 for percentage
        u['total'] = (u2.sum(axis=1, skipna=False) ** 0.5)
        u['total_95'] = u['total'] * 2
        u = u.mul(100)

        # Uncertainty for the measurement
        # The only random source is from the measured area
        u2_random = u2['u_meas'].mean(skipna=False)

        # All other sources are systematic (mostly due to computation method and values from user)
        u2_bias = u2_measurement.drop(['u_meas'], axis=1).sum(axis=1, skipna=False)

        # Combined all uncertainty sources
        u2_measurement['total'] = (1 / n_transects) * u2_random + u2_bias[0]
        u_measurement = u2_measurement ** 0.5
        u_measurement['total_95'] = u_measurement['total'] * 2
        u_measurement = u_measurement * 100

        # Compute relative contributions from each source
        u_contribution_measurement = u2_measurement.copy()
        u_contribution_measurement['u_meas'] = u2_measurement['u_meas'] / (n_transects**2)
        u_contribution_measurement = u_contribution_measurement.div(u_contribution_measurement['total'], axis=0)

        u_contribution = u2.copy()
        u_contribution['u_meas'] = u2['u_meas'].div(n_transects**2, axis=0)
        u_contribution['total'] = u_contribution.sum(axis=1)
        u_contribution = u_contribution.div(u_contribution['total'], axis=0)

        return u, u_measurement, u_contribution, u_contribution_measurement, u_nocov, u_measurement_nocov

    def data_prep(self, meas):
        """Determine checked transects and max and min exponents for power and no slip extrapolation.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        # Use only checked transects
        # Extract data that are used later on (PP and NS exponents)
        for n in range(len(meas.transects)):
            if meas.transects[n].checked:
                self.checked_idx.append(n)

                # Bottom method selected using data from each transect only
                self.bot_meth.append(meas.extrap_fit.sel_fit[n].bot_method_auto)

                # Store 95 percent bounds on power fit exponent for each transect if power selected
                if meas.extrap_fit.sel_fit[n].bot_method_auto == "Power":
                    try:
                        self.exp_95ic_min.append(meas.extrap_fit.sel_fit[n].exponent_95_ci[0][0])
                    except TypeError:
                        self.exp_95ic_min.append(np.nan)
                    try:
                        self.exp_95ic_max.append(meas.extrap_fit.sel_fit[n].exponent_95_ci[1][0])
                    except TypeError:
                        self.exp_95ic_max.append(np.nan)

                    self.pp_exp.append(meas.extrap_fit.sel_fit[n].pp_exponent)

                # Store no slip exponent if no slip selected
                elif meas.extrap_fit.sel_fit[n].bot_method_auto == "No Slip":
                    self.ns_exp.append(meas.extrap_fit.sel_fit[n].ns_exponent)

        self.nb_transects = len(self.checked_idx)

    def run_simulations(self, meas):
        """Compute discharges (top, bot, right, left, total, middle)  based on possible scenarios

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        # If list have not be saved recompute q_sensitivity
        if not hasattr(meas.extrap_fit.q_sensitivity, 'q_pp_list'):
            meas.extrap_fit.q_sensitivity.populate_data(meas.transects, meas.extrap_fit.sel_fit)

        # Simulation original
        self.sim_orig(meas)

        # Simulation power / power default 1/6
        self.sim_extrap_pp_16['q_total'] = meas.extrap_fit.q_sensitivity.q_pp_list
        self.sim_extrap_pp_16['q_top'] = meas.extrap_fit.q_sensitivity.q_top_pp_list
        self.sim_extrap_pp_16['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_pp_list

        # Simulations power / power optimized
        self.sim_pp_min_max_opt(meas=meas)

        # Simulation cns default 1/6
        self.sim_extrap_cns_16['q_total'] = meas.extrap_fit.q_sensitivity.q_cns_list
        self.sim_extrap_cns_16['q_top'] = meas.extrap_fit.q_sensitivity.q_top_cns_list
        self.sim_extrap_cns_16['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_cns_list

        # Simulation cns optimized
        self.sim_cns_min_max_opt(meas=meas)

        # Simulation 3pt no slip default 1/6
        self.sim_extrap_3pns_16['q_total'] = meas.extrap_fit.q_sensitivity.q_3p_ns_list
        self.sim_extrap_3pns_16['q_top'] = meas.extrap_fit.q_sensitivity.q_top_3p_ns_list
        self.sim_extrap_3pns_16['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_3p_ns_list

        # Simulation 3pt no slip optimized
        self.sim_extrap_3pns_opt['q_total'] = meas.extrap_fit.q_sensitivity.q_3p_ns_opt_list
        self.sim_extrap_3pns_opt['q_top'] = meas.extrap_fit.q_sensitivity.q_top_3p_ns_opt_list
        self.sim_extrap_3pns_opt['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_3p_ns_opt_list

        # Simulations edge min and max
        self.sim_edge_min_max(meas=meas)

        # Simulation draft min and max
        self.sim_draft_max_min(meas=meas)

        # Simulation of invalid cells and ensembles
        self.sim_invalid_cells(meas=meas)

        # Simulation of shallow no cells
        self.sim_shallow_ens(meas=meas)

        # Simulation of invalid boat velocity
        self.sim_invalid_boat_velocity(meas=meas)

        # Simulation of invalid depths
        self.sim_invalid_depth(meas=meas)

    def uncertainty_measured_discharge(self, meas):
        """Compute the uncertainty related to the measured area.

        Parameters
        ----------
        meas: Measurement
            Object of class Measurement
        """

        self.u_contribution_meas = pd.DataFrame(columns=['boat', 'water', 'dzi'])

        # Set uncertainty of cell size
        if self.user_advanced_settings['dzi_prct_user'] is not None:
            u_dzi = self.user_advanced_settings['dzi_prct_user'] * 0.01
        else:
            u_dzi = self.default_advanced_settings['dzi_prct'] * 0.01

        # Compute the uncertainty due to the measured area
        for transect_id in self.checked_idx:
            # Relative depth error due to vertical velocity of boat
            # TODO this does not belong in the measured area, only affects bottom and edges
            # relative_error_depth = self.depth_error_boat_motion(meas.transects[transect_id])

            # Relative standard deviation of error velocity (Water Track)
            std_ev_wt_ens = self.water_std_by_error_velocity(meas.transects[transect_id])

            u_boat = np.nan
            if meas.transects[transect_id].boat_vel.selected == 'bt_vel':
                # Relative standard deviation of error velocity (Bottom Track)
                u_boat = self.boat_std_by_error_velocity(meas.transects[transect_id])
            elif meas.transects[transect_id].boat_vel.selected == 'gga_vel':
                if self.user_advanced_settings['gga_boat_user'] is None:
                    if meas.transects[transect_id].gps is not None:
                        u_boat = (np.nanstd(meas.transects[transect_id].gps.altitude_ens_m, ddof=1) / 3) / \
                                   np.nanmean(np.diff(meas.transects[transect_id].gps.gga_serial_time_ens))
                else:
                    u_boat = self.user_advanced_settings['gga_boat_mps']
            elif meas.transects[transect_id].boat_vel.selected == 'vtg_vel':
                if self.user_advanced_settings['vtg_boat_user'] is None:
                    if meas.transects[transect_id].gps is not None:
                        u_boat = self.default_advanced_settings['vtg_boat_mps']
                else:
                    u_boat = self.user_advanced_settings['vtg_boat_user']

            # Computation of u_meas
            q_2_tran = meas.discharge[transect_id].total ** 2
            q_2_ens = meas.discharge[transect_id].middle_ens ** 2
            n_cell_ens = meas.transects[transect_id].w_vel.cells_above_sl.sum(axis=0)  # number of cells by ens
            n_cell_ens = np.where(n_cell_ens == 0, np.nan, n_cell_ens)

            # Variance for each ensembles
            # u_2_meas = q_2_ens * (relative_error_depth ** 2 + u_boat ** 2 +
            #                       (1 / n_cell_ens) * (std_ev_wt_ens ** 2 + u_dzi ** 2))
            # TODO DSM I would have computed as follows
            u_2_meas = q_2_ens * (u_boat ** 2 + (1 / n_cell_ens) * (std_ev_wt_ens ** 2 + u_dzi ** 2))

            u_2_prct_meas = np.nansum(u_2_meas) / q_2_tran

            # Standard deviation
            u_prct_meas = u_2_prct_meas ** 0.5
            self.u_meas_list.append(u_prct_meas)

            # Compute the contribution of all terms to u_meas (sum of a0 to g0 =1)
            u_contrib_boat = (np.nan_to_num(q_2_ens * (u_boat ** 2)).sum() / q_2_tran) / u_2_prct_meas
            # u_contrib_depth = (np.nan_to_num(q_2_ens * (relative_error_depth ** 2)).sum()
            #                    / q_2_tran) / u_2_prct_meas
            u_contrib_water = (np.nan_to_num(q_2_ens * ((1 / n_cell_ens) * (std_ev_wt_ens ** 2))).sum()
                               / q_2_tran) / u_2_prct_meas
            u_contrib_dzi = (np.nan_to_num(q_2_ens * ((1 / n_cell_ens) * (u_dzi ** 2))).sum()
                             / q_2_tran) / u_2_prct_meas

            # self.u_contribution_meas.loc[len(self.u_contribution_meas)] = [u_contrib_boat,
            #                                                                u_contrib_water,
            #                                                                u_contrib_depth,
            #                                                                u_contrib_dzi]
            self.u_contribution_meas.loc[len(self.u_contribution_meas)] = [u_contrib_boat,
                                                                           u_contrib_water,
                                                                           u_contrib_dzi]

        # Apply user specified uncertainty
        if self.user_specified_u['u_meas_mean_user'] is not None:
            self.u_meas_mean_user_list = [0.01 * self.user_specified_u['u_meas_mean_user']] * self.nb_transects
        else:
            self.u_meas_mean_user_list = self.u_meas_list

    def uncertainty_moving_bed(self, meas):
        """Computes the moving-bed uncertainty

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        # Compute moving-bed uncertainty
        if len(self.checked_idx) and meas.transects[self.checked_idx[0]].boat_vel.selected == 'bt_vel':
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

        # Expand to list
        self.u_movbed_list = [0.01 * moving_bed_uncertainty / 2] * self.nb_transects

        # Apply user specified
        if self.user_specified_u['u_movbed_user'] is not None:
            self.u_movbed_user_list = [self.user_specified_u['u_movbed_user'] * 0.01] * self.nb_transects
        else:
            self.u_movbed_user_list = self.u_movbed_list

    def uncertainty_system(self):
        """Compute systematic uncertaint
        """

        self.u_syst_list = [0.01 * 1.31] * self.nb_transects

        if self.user_specified_u['u_syst_mean_user'] is not None:
            self.u_syst_mean_user_list = [self.user_specified_u['u_syst_mean_user'] * 0.01] * self.nb_transects
        else:
            self.u_syst_mean_user_list = self.u_syst_list

    def uncertainty_number_ensembles(self, meas):
        """Computes the uncertainty due to the number of ensembles in a transect.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        for trans_id in self.checked_idx:
            # Compute uncertainty due to limited number of ensembles (ISO 748; Le Coz et al., 2012)
            self.u_ens_list.append(0.01 * 32 * len(meas.discharge[trans_id].middle_ens) ** (-0.88))

        if self.user_specified_u['u_ens_user'] is not None:
            self.u_ens_user_list = [0.01 * self.user_specified_u['u_ens_user']] * self.nb_transects
        else:
            self.u_ens_user_list = self.u_ens_list

    def uncertainty_compass(self, meas):
        """Compute the potential bias in the measurement due to dynamic compass errors when using GPS as
        the navigation reference. The method is based on Mueller (2018,
        https://doi.org/10.1016/j.flowmeasinst.2018.10.004, equation 41.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        if meas.transects[self.checked_idx[0]].boat_vel.selected == 'bt_vel':
            self.u_compass_list = [0] * self.nb_transects
        else:
            if self.user_advanced_settings['compass_error_user'] is not None:
                compass_error = self.user_advanced_settings['compass_error_user']
            else:
                compass_error = self.default_advanced_settings['compass_error_deg']

            meas_stats = meas.compute_measurement_properties(meas)
            speed_ratio = meas_stats['avg_boat_speed'][self.checked_idx] / \
                meas_stats['avg_water_speed'][self.checked_idx]
            self.u_compass_list = np.abs(1 - (cosd(compass_error) + 0.5 * speed_ratio * sind(compass_error)))
        if self.user_specified_u['u_compass_user'] is None:
            self.u_compass_user_list = self.u_compass_list
        else:
            self.u_compass_user_list = [self.user_specified_u['u_compass_user'] * 0.01] * self.nb_transects

    def uncertainty_top_discharge(self):
        """Computes the uncertainty in the top discharge using simulations and rectangular law.
        """

        self.u_top_list = Oursin.apply_u_rect(list_sims=[self.sim_original,
                                                         self.sim_extrap_pp_opt,
                                                         self.sim_extrap_pp_min,
                                                         self.sim_extrap_pp_max,
                                                         self.sim_extrap_cns_opt,
                                                         self.sim_extrap_cns_min,
                                                         self.sim_extrap_cns_max,
                                                         self.sim_extrap_3pns_opt,
                                                         self.sim_draft_max,
                                                         self.sim_draft_min],
                                              col_name='q_top') \
            / np.abs(self.sim_original['q_total'])

        if self.user_specified_u['u_top_mean_user'] is not None:
            self.u_top_mean_user_list = [0.01 * self.user_specified_u['u_top_mean_user']] * self.nb_transects
        else:
            self.u_top_mean_user_list = self.u_top_list

    def uncertainty_bottom_discharge(self):
        """Computes uncertainty of bottom discharge using simulations and rectangular law.
        """

        self.u_bot_list = Oursin.apply_u_rect(list_sims=[self.sim_original,
                                                         self.sim_extrap_pp_opt,
                                                         self.sim_extrap_pp_min,
                                                         self.sim_extrap_pp_max,
                                                         self.sim_extrap_cns_opt,
                                                         self.sim_extrap_cns_min,
                                                         self.sim_extrap_cns_max,
                                                         self.sim_extrap_3pns_opt],
                                              col_name='q_bot') \
            / np.abs(self.sim_original['q_total'])

        if self.user_specified_u['u_bot_mean_user'] is not None:
            self.u_bot_mean_user_list = [0.01 * self.user_specified_u['u_bot_mean_user']] * self.nb_transects
        else:
            self.u_bot_mean_user_list = self.u_bot_list

    def uncertainty_left_discharge(self):
        """Computes the uncertianty of the left edge discharge using simulations and the rectangular law.
        """

        self.u_left_list = Oursin.apply_u_rect(list_sims=[self.sim_original['q_left'],
                                                          self.sim_edge_min,
                                                          self.sim_edge_max,
                                                          self.sim_draft_min,
                                                          self.sim_draft_max],
                                               col_name='q_left') \
            / np.abs(self.sim_original['q_total'])

        if self.user_specified_u['u_left_mean_user'] is not None:
            self.u_left_mean_user_list = [0.01 * self.user_specified_u['u_left_mean_user']] * self.nb_transects
        else:
            self.u_left_mean_user_list = self.u_left_list

    def uncertainty_right_discharge(self):
        """Computes the uncertainty of the right edge discharge using simulations and the rectangular law.
        """

        self.u_right_list = Oursin.apply_u_rect(list_sims=[self.sim_original['q_right'],
                                                           self.sim_edge_min,
                                                           self.sim_edge_max,
                                                           self.sim_draft_min,
                                                           self.sim_draft_max],
                                                col_name='q_right') \
            / np.abs(self.sim_original['q_total'])

        if self.user_specified_u['u_right_mean_user'] is not None:
            self.u_right_mean_user_list = [0.01 * self.user_specified_u['u_right_mean_user']] * self.nb_transects
        else:
            self.u_right_mean_user_list = self.u_right_list

    def uncertainty_invalid_depth_data(self):
        """Computes the uncertainty due to invalid depth data using simulations and the retangular law.
        """

        self.u_invalid_depth_list = Oursin.apply_u_rect(list_sims=[self.sim_original,
                                                                   self.sim_depth_hold,
                                                                   self.sim_depth_next],
                                                        col_name='q_total') \
            / np.abs(self.sim_original['q_total'])

        if self.user_specified_u['u_invalid_depth_user'] is not None:
            self.u_invalid_depth_user_list = [0.01 * self.user_specified_u[
                'u_invalid_depth_user']] * self.nb_transects
        else:
            self.u_invalid_depth_user_list = self.u_invalid_depth_list

    def uncertainty_invalid_boat_data(self):
        """Computes the uncertainty due to invalid boat data using simulations and the rectangular law.
        """

        self.u_invalid_boat_list = Oursin.apply_u_rect(list_sims=[self.sim_original,
                                                                  self.sim_boat_hold,
                                                                  self.sim_boat_next],
                                                       col_name='q_total') \
            / np.abs(self.sim_original['q_total'])

        if self.user_specified_u['u_invalid_boat_user'] is not None:
            self.u_invalid_boat_user_list = [0.01 * self.user_specified_u['u_invalid_boat_user']] * self.nb_transects
        else:
            self.u_invalid_boat_user_list = self.u_invalid_boat_list

    def uncertainty_invalid_water_data(self):
        """Computes the uncertainty due to invalid water data assuming rectangular law.
        """

        # Uncertainty due to invalid cells and ensembles
        self.u_invalid_water_list = Oursin.apply_u_rect(list_sims=[self.sim_original,
                                                                   self.sim_cells_trdi,
                                                                   self.sim_cells_above,
                                                                   self.sim_cells_below,
                                                                   self.sim_cells_before,
                                                                   self.sim_cells_after,
                                                                   self.sim_shallow],
                                                        col_name='q_total') \
            / np.abs(self.sim_original['q_total'])

        if self.user_specified_u['u_invalid_water_user'] is not None:
            self.u_invalid_water_user_list = [0.01 * self.user_specified_u['u_invalid_water_user']] \
                                             * self.nb_transects
        else:
            self.u_invalid_water_user_list = self.u_invalid_water_list

    def compute_measurement_cov(self, meas):
        """Compute the coefficient of variation of the total transect discharges used in the measurement.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        self.cov_68 = np.nan

        # Only compute for multiple transects
        if self.nb_transects > 1:
            total_q = []
            for trans_id in self.checked_idx:
                total_q.append(meas.discharge[trans_id].total)

            # Compute coefficient of variation
            cov = np.abs(np.nanstd(total_q, ddof=1) / np.nanmean(total_q))

            # Inflate the cov to the 95% value
            if len(total_q) == 2:
                # Use the approximate method as taught in class to reduce the high coverage factor for 2 transects
                # and account for prior knowledge related to 720 second duration analysis
                cov_95 = cov * 3.3
                self.cov_68 = cov_95 / 2
            else:
                # Use Student's t to inflate COV for n > 2
                cov_95 = t.interval(0.95, len(total_q) - 1)[1] * cov / len(total_q) ** 0.5
                self.cov_68 = cov_95 / 2

        # if u_cov_68_user is not None:
        #     self.u_cov_68_user = u_cov_68_user * 0.01
        #     self.u_cov_68_user_value = self.u_cov_68_user
        # else:
        # self.u_cov_68_user_value = self.cov_68

    def sim_orig(self, meas):
        """Stores original measurement results in a data frame

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """
        transect_q = dict()
        for trans_id in self.checked_idx:
            transect_q['q_total'] = meas.discharge[trans_id].total
            transect_q['q_top'] = meas.discharge[trans_id].top
            transect_q['q_bot'] = meas.discharge[trans_id].bottom
            transect_q['q_right'] = meas.discharge[trans_id].right
            transect_q['q_left'] = meas.discharge[trans_id].left
            transect_q['q_middle'] = meas.discharge[trans_id].middle
            self.sim_original = self.sim_original.append(transect_q, ignore_index=True, sort=False)

    def sim_cns_min_max_opt(self, meas):
        """Computes simulations resulting in the the min and max discharges for a constant no slip extrapolation
        fit.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        # Compute min-max no slip exponent
        skip_ns_min_max, self.exp_ns_max, self.exp_ns_min = \
            self.compute_ns_max_min(meas=meas,
                                    ns_exp=self.ns_exp,
                                    exp_ns_min_user=self.user_advanced_settings['exp_ns_min_user'],
                                    exp_ns_max_user=self.user_advanced_settings['exp_ns_max_user'])

        # Optimized
        self.sim_extrap_cns_opt['q_total'] = meas.extrap_fit.q_sensitivity.q_cns_opt_list
        self.sim_extrap_cns_opt['q_top'] = meas.extrap_fit.q_sensitivity.q_top_cns_opt_list
        self.sim_extrap_cns_opt['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_cns_opt_list

        # Max min
        if skip_ns_min_max:
            # If cns not used both max and min are equal to the optimized value
            self.sim_extrap_cns_min['q_total'] = meas.extrap_fit.q_sensitivity.q_cns_opt_list
            self.sim_extrap_cns_min['q_top'] = meas.extrap_fit.q_sensitivity.q_top_cns_opt_list
            self.sim_extrap_cns_min['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_cns_opt_list
            self.sim_extrap_cns_max['q_total'] = meas.extrap_fit.q_sensitivity.q_cns_opt_list
            self.sim_extrap_cns_max['q_top'] = meas.extrap_fit.q_sensitivity.q_top_cns_opt_list
            self.sim_extrap_cns_max['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_cns_opt_list
        else:
            # Compute q for min and max values
            q = QComp()
            self.sim_extrap_cns_min = pd.DataFrame(columns=self.sim_extrap_cns_min.columns)
            self.sim_extrap_cns_max = pd.DataFrame(columns=self.sim_extrap_cns_max.columns)

            for trans_id in self.checked_idx:
                # Compute min values
                #TODO if depth uncertianty were included a copy of transects[trans_id] would need to be made with the adjusted depth values
                q.populate_data(data_in=meas.transects[trans_id],
                                top_method='Constant',
                                bot_method='No Slip',
                                exponent=self.exp_ns_min)
                self.sim_extrap_cns_min.loc[len(self.sim_extrap_cns_min)] = [q.total, q.top, q.bottom]
                # Compute max values
                q.populate_data(data_in=meas.transects[trans_id],
                                top_method='Constant',
                                bot_method='No Slip',
                                exponent=self.exp_ns_max)
                self.sim_extrap_cns_max.loc[len(self.sim_extrap_cns_max)] = [q.total, q.top, q.bottom]

    def sim_pp_min_max_opt(self, meas):
        """Computes simulations resulting in the the min and max discharges for a power power extrapolation
        fit.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        # A power fit is not applicable to bi-directional flow
        mean_q = meas.mean_discharges(meas)
        if np.sign(mean_q['top_mean']) != np.sign(mean_q['bot_mean']):
            self.sim_extrap_pp_min = self.sim_original[['q_total', 'q_top', 'q_bot']]
            self.sim_extrap_pp_max = self.sim_original[['q_total', 'q_top', 'q_bot']]
            self.sim_extrap_pp_opt = self.sim_original[['q_total', 'q_top', 'q_bot']]

        else:
            # Compute min-max power exponent
            skip_pp_min_max, self.exp_pp_max, self.exp_pp_min = \
                self.compute_pp_max_min(meas=meas,
                                        exp_95ic_min=self.exp_95ic_min,
                                        exp_95ic_max=self.exp_95ic_max,
                                        pp_exp=self.pp_exp,
                                        exp_pp_min_user=self.user_advanced_settings['exp_pp_min_user'],
                                        exp_pp_max_user=self.user_advanced_settings['exp_pp_max_user'])

            # Optimized
            self.sim_extrap_pp_opt['q_total'] = meas.extrap_fit.q_sensitivity.q_pp_opt_list
            self.sim_extrap_pp_opt['q_top'] = meas.extrap_fit.q_sensitivity.q_top_pp_opt_list
            self.sim_extrap_pp_opt['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_pp_opt_list

            # Max min
            if skip_pp_min_max:
                self.sim_extrap_pp_min['q_total'] = meas.extrap_fit.q_sensitivity.q_pp_opt_list
                self.sim_extrap_pp_min['q_top'] = meas.extrap_fit.q_sensitivity.q_top_pp_opt_list
                self.sim_extrap_pp_min['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_pp_opt_list
                self.sim_extrap_pp_max['q_total'] = meas.extrap_fit.q_sensitivity.q_pp_opt_list
                self.sim_extrap_pp_max['q_top'] = meas.extrap_fit.q_sensitivity.q_top_pp_opt_list
                self.sim_extrap_pp_max['q_bot'] = meas.extrap_fit.q_sensitivity.q_bot_pp_opt_list
            else:
                q = QComp()
                self.sim_extrap_pp_min = pd.DataFrame(columns=self.sim_extrap_pp_min.columns)
                self.sim_extrap_pp_max = pd.DataFrame(columns=self.sim_extrap_pp_max.columns)

                for trans_id in self.checked_idx:
                    q.populate_data(data_in=meas.transects[trans_id],
                                    top_method='Power',
                                    bot_method='Power',
                                    exponent=self.exp_pp_min)
                    self.sim_extrap_pp_min.loc[len(self.sim_extrap_pp_min)] = [q.total, q.top, q.bottom]

                    q.populate_data(data_in=meas.transects[trans_id],
                                    top_method='Power',
                                    bot_method='Power',
                                    exponent=self.exp_pp_max)
                    self.sim_extrap_pp_max.loc[len(self.sim_extrap_pp_max)] = [q.total, q.top, q.bottom]

    def sim_edge_min_max(self, meas):
        """Computes simulations for the maximum and minimum edge discharges.

        Parameters
        ----------
        meas: MeasurementData
            Object of measurement data
        """

        # Clear variables
        self.d_right_error_min = []
        self.d_left_error_min = []
        self.d_right_error_max = []
        self.d_left_error_max = []
        self.sim_edge_min = pd.DataFrame(columns=self.sim_edge_min.columns)
        self.sim_edge_max = pd.DataFrame(columns=self.sim_edge_max.columns)

        # Create measurement copy to allow changes without affecting original
        meas_temp = copy.deepcopy(meas)

        # Process each checked transect
        for trans_id in self.checked_idx:
            # Compute max and min edge distances
            max_left_dist, max_right_dist, min_left_dist, min_right_dist = \
                self.compute_edge_dist_max_min(transect=meas.transects[trans_id],
                                               user_settings=self.user_advanced_settings,
                                               default_settings=self.default_advanced_settings)

            # Compute edge minimum
            self.d_right_error_min.append(min_right_dist)
            self.d_left_error_min.append(min_left_dist)
            meas_temp.transects[trans_id].edges.left.distance_m = min_left_dist
            meas_temp.transects[trans_id].edges.right.distance_m = min_right_dist
            meas_temp.transects[trans_id].edges.left.type = 'Triangular'
            meas_temp.transects[trans_id].edges.right.type = 'Triangular'
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_edge_min.loc[len(self.sim_edge_min)] = [meas_temp.discharge[trans_id].total,
                                                             meas_temp.discharge[trans_id].left,
                                                             meas_temp.discharge[trans_id].right]

            # Compute edge maximum
            self.d_right_error_max.append(max_right_dist)
            self.d_left_error_max.append(max_left_dist)
            meas_temp.transects[trans_id].edges.left.distance_m = max_left_dist
            meas_temp.transects[trans_id].edges.right.distance_m = max_right_dist
            meas_temp.transects[trans_id].edges.left.type = 'Rectangular'
            meas_temp.transects[trans_id].edges.right.type = 'Rectangular'
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_edge_max.loc[len(self.sim_edge_max)] = [meas_temp.discharge[trans_id].total,
                                                             meas_temp.discharge[trans_id].left,
                                                             meas_temp.discharge[trans_id].right]

    def sim_draft_max_min(self, meas):
        """Compute the simulations for the max and min draft errror.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        # Reset variables
        self.draft_error_list = []
        self.sim_draft_min = pd.DataFrame(columns=self.sim_draft_min.columns)
        self.sim_draft_max = pd.DataFrame(columns=self.sim_draft_max.columns)

        # Create copy of meas to avoid changing original
        meas_temp = copy.deepcopy(meas)

        for trans_id in self.checked_idx:
            # Compute max and min draft
            draft_max, draft_min, draft_error = \
                self.compute_draft_max_min(transect=meas.transects[trans_id],
                                           draft_error_user=self.user_advanced_settings['draft_error_user'])
            self.draft_error_list.append(draft_error)

            # Compute discharge for draft min
            meas_temp.transects[trans_id].change_draft(draft_min)
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_draft_min.loc[len(self.sim_draft_min)] = [meas_temp.discharge[trans_id].total,
                                                               meas_temp.discharge[trans_id].top,
                                                               meas_temp.discharge[trans_id].left,
                                                               meas_temp.discharge[trans_id].right]
            # Compute discharge for draft max
            meas_temp.transects[trans_id].change_draft(draft_max)
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_draft_max.loc[len(self.sim_draft_max)] = [meas_temp.discharge[trans_id].total,
                                                               meas_temp.discharge[trans_id].top,
                                                               meas_temp.discharge[trans_id].left,
                                                               meas_temp.discharge[trans_id].right]

    def sim_invalid_cells(self, meas):
        """Computes simulations using different methods to interpolate for invalid cells and ensembles.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        # Reset data frames
        self.sim_cells_trdi = pd.DataFrame(columns=self.sim_cells_trdi.columns)
        self.sim_cells_above = pd.DataFrame(columns=self.sim_cells_above.columns)
        self.sim_cells_below = pd.DataFrame(columns=self.sim_cells_below.columns)
        self.sim_cells_before = pd.DataFrame(columns=self.sim_cells_before.columns)
        self.sim_cells_after = pd.DataFrame(columns=self.sim_cells_after.columns)

        # Simulations for invalid cells and ensembles
        meas_temp = copy.deepcopy(meas)
        for trans_id in self.checked_idx:
            # TRDI method
            meas_temp.transects[trans_id].w_vel.interpolate_cells_trdi(meas_temp.transects[trans_id])
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_cells_trdi.loc[len(self.sim_cells_trdi)] = [meas_temp.discharge[trans_id].total,
                                                                 meas_temp.discharge[trans_id].middle]

            # Above only
            meas_temp.transects[trans_id].w_vel.interpolate_abba(meas_temp.transects[trans_id], search_loc=['above'])
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_cells_above.loc[len(self.sim_cells_above)] = [meas_temp.discharge[trans_id].total,
                                                                   meas_temp.discharge[trans_id].middle]
            # Below only
            meas_temp.transects[trans_id].w_vel.interpolate_abba(meas_temp.transects[trans_id], search_loc=['below'])
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_cells_below.loc[len(self.sim_cells_below)] = [meas_temp.discharge[trans_id].total,
                                                                   meas_temp.discharge[trans_id].middle]
            # Before only
            meas_temp.transects[trans_id].w_vel.interpolate_abba(meas_temp.transects[trans_id], search_loc=['before'])
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_cells_before.loc[len(self.sim_cells_before)] = [meas_temp.discharge[trans_id].total,
                                                                     meas_temp.discharge[trans_id].middle]
            # After only
            meas_temp.transects[trans_id].w_vel.interpolate_abba(meas_temp.transects[trans_id], search_loc=['after'])
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_cells_after.loc[len(self.sim_cells_after)] = [meas_temp.discharge[trans_id].total,
                                                                   meas_temp.discharge[trans_id].middle]

    def sim_shallow_ens(self, meas):
        """Computes simulations assuming no interpolation of discharge for ensembles where depths are too shallow
        for any valid cells.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        """

        # Reset data frame
        self.sim_shallow = pd.DataFrame(columns=self.sim_shallow.columns)

        for trans_id in self.checked_idx:
            shallow_estimate = np.nansum(meas.discharge[trans_id].middle_ens) \
                               - np.nansum(np.nansum(meas.discharge[trans_id].middle_cells))
            if np.abs(shallow_estimate) > 0:
                self.sim_shallow.loc[len(self.sim_shallow)] = [meas.discharge[trans_id].total - shallow_estimate,
                                                               meas.discharge[trans_id].middle - shallow_estimate]
            else:
                self.sim_shallow.loc[len(self.sim_shallow)] = [meas.discharge[trans_id].total,
                                                               meas.discharge[trans_id].middle]

    def sim_invalid_depth(self, meas):
        """Computes simulations using different methods to interpolate for invalid depths.

        Parameters
        ----------
        meas: MeasurementData
           Object of MeasurementData
        """

        # Reset dataframes
        self.sim_depth_hold = pd.DataFrame(columns=self.sim_depth_hold.columns)
        self.sim_depth_next = pd.DataFrame(columns=self.sim_depth_next.columns)

        # Simulations for invalid depths
        meas_temp = copy.deepcopy(meas)
        for trans_id in self.checked_idx:
            depths = getattr(meas_temp.transects[trans_id].depths, meas_temp.transects[trans_id].depths.selected)
            # Hold last
            depths.interpolate_hold_last()
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_depth_hold.loc[len(self.sim_depth_hold)] = [meas_temp.discharge[trans_id].total,
                                                                 meas_temp.discharge[trans_id].middle]
            # Fill with next
            depths.interpolate_next()
            meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                        moving_bed_data=meas_temp.mb_tests)
            self.sim_depth_next.loc[len(self.sim_depth_next)] = [meas_temp.discharge[trans_id].total,
                                                                 meas_temp.discharge[trans_id].middle]

    def sim_invalid_boat_velocity(self, meas):
        """Computes simulations using different methods to interpolate for invalid boat velocity.

        Parameters
        ----------
        meas: MeasurementData
           Object of MeasurementData
        """

        # Reset dataframes
        self.sim_boat_hold = pd.DataFrame(columns=self.sim_boat_hold.columns)
        self.sim_boat_next = pd.DataFrame(columns=self.sim_boat_next.columns)

        # Simulations for invalid boat velocity
        meas_temp = copy.deepcopy(meas)
        for trans_id in self.checked_idx:
            # Hold last
            boat_data = getattr(meas_temp.transects[trans_id].boat_vel, meas_temp.transects[trans_id].boat_vel.selected)
            if boat_data is not None:
                boat_data.interpolate_hold_last()
                meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                            moving_bed_data=meas_temp.mb_tests)
                self.sim_boat_hold.loc[len(self.sim_boat_hold)] = [meas_temp.discharge[trans_id].total,
                                                                   meas_temp.discharge[trans_id].middle]
                # Fill with next
                boat_data.interpolate_next()
                meas_temp.discharge[trans_id].populate_data(data_in=meas_temp.transects[trans_id],
                                                            moving_bed_data=meas_temp.mb_tests)
                self.sim_boat_next.loc[len(self.sim_boat_next)] = [meas_temp.discharge[trans_id].total,
                                                                   meas_temp.discharge[trans_id].middle]
            else:
                self.sim_boat_next.loc[len(self.sim_boat_next)] = [meas.discharge[trans_id].total,
                                                                   meas_temp.discharge[trans_id].middle]
                self.sim_boat_hold.loc[len(self.sim_boat_hold)] = [meas.discharge[trans_id].total,
                                                                   meas_temp.discharge[trans_id].middle]

    @staticmethod
    def compute_draft_max_min(transect, draft_error_user=None):
        """Determine the max and min values of the ADCP draft.

        Parameters
        ----------
        transect: TransectData
            Object of transect data
        draft_error_user: float
            User specified draft error in m

        Returns
        -------
        draft_max: float
            Maximum draft in m for simulations
        draft_min: float
            Minimum draft in m for simulations
        draft_error: float
            Draft error in m
        """
        depths = transect.depths.bt_depths.depth_processed_m  # depth by ens
        depth_90 = np.quantile(depths, q=0.9)  # quantile 90% to avoid spikes

        # Determine draft error value
        if draft_error_user is None:
            if depth_90 < 2.50:
                draft_error = 0.02
            else:
                draft_error = 0.05
        else:
            draft_error = draft_error_user

        # Compute draft max and min
        draft_min = transect.depths.bt_depths.draft_orig_m - draft_error
        draft_max = transect.depths.bt_depths.draft_orig_m + draft_error

        if draft_min <= 0:
            draft_min = 0.01

        return draft_max, draft_min, draft_error

    @staticmethod
    def compute_edge_dist_max_min(transect, user_settings, default_settings):
        """Compute the max and min edge distances.
        """

        init_dist_right = transect.edges.right.distance_m
        init_dist_left = transect.edges.left.distance_m

        # Select user percentage or default
        if user_settings['right_edge_dist_prct_user'] is None:
            d_right_error_prct = default_settings['right_edge_dist_prct']
        else:
            d_right_error_prct = user_settings['right_edge_dist_prct_user']

        if user_settings['left_edge_dist_prct_user'] is None:
            d_left_error_prct = default_settings['left_edge_dist_prct']
        else:
            d_left_error_prct = user_settings['left_edge_dist_prct_user']

        # Compute min distance for both edges
        min_left_dist = (1 - d_left_error_prct * 0.01) * init_dist_left
        min_right_dist = (1 - d_right_error_prct * 0.01) * init_dist_right

        if min_left_dist <= 0:
            min_left_dist = 0.10
        if min_right_dist <= 0:
            min_right_dist = 0.10

        # Compute max distance for both edges
        max_left_dist = (1 + d_left_error_prct * 0.01) * init_dist_left
        max_right_dist = (1 + d_right_error_prct * 0.01) * init_dist_right

        return max_left_dist, max_right_dist, min_left_dist, min_right_dist

    @staticmethod
    def compute_pp_max_min(meas, exp_95ic_min, exp_95ic_max, pp_exp, exp_pp_min_user, exp_pp_max_user):
        """Determine the max and min exponents for power fit.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        exp_95ic_min: list
            Minimum power fit exponent from the 95% confidence interval for each transect
        exp_95ic_max: list
            Maximum power fit exponent from the 95% confidence interval for each transect
        pp_exp: list
            Optimized power fit exponent for each transect
        exp_pp_min_user: float
            User supplied minimum power fit exponent
        exp_pp_max_user: float
            User supplied maximum power fit exponent

        Returns
        -------
        skip_pp_min_max: bool
            Boolean to identify if power fit simulations should be skipped
        exp_pp_max: float
            Maximum power fit exponent to be used in simulations
        exp_pp_min: float
            Minimum power fit exponent to be used in simulations
        """
        skip_pp_min_max = False
        if len(pp_exp) == 0:
            skip_pp_min_max = True
            min_pp = meas.extrap_fit.q_sensitivity.pp_exp
            max_pp = meas.extrap_fit.q_sensitivity.pp_exp
        else:
            if np.isnan(pp_exp).any():
                mean_pp = 0.16
            else:
                mean_pp = np.nanmean(pp_exp)

            # If all transects have confidence intervals, use the mean of the confidence interval min/max
            # Otherwise adjust average +/- 0.2
            if np.isnan(exp_95ic_min).any():
                min_pp = mean_pp - 0.2
            else:
                min_pp = np.nanmean(exp_95ic_min)

            if np.isnan(exp_95ic_max).any():
                max_pp = mean_pp + 0.2
            else:
                max_pp = np.nanmean(exp_95ic_max)

            # Diff between mean PP exponent and min/max
            if mean_pp - min_pp > 0.2:
                min_pp = mean_pp - 0.2
            if max_pp - mean_pp > 0.2:
                max_pp = mean_pp + 0.2

            # Check that 0 < exponent < 1
            if min_pp <= 0:
                min_pp = 0.01
            if max_pp >= 1:
                max_pp = 0.99

        # Set min-max exponents of user override
        if exp_pp_min_user is None:
            exp_pp_min = min_pp
        else:
            exp_pp_min = exp_pp_min_user

        if exp_pp_max_user is None:
            exp_pp_max = max_pp
        else:
            exp_pp_max = exp_pp_max_user

        return skip_pp_min_max, exp_pp_max, exp_pp_min

    @staticmethod
    def compute_ns_max_min(meas, ns_exp, exp_ns_min_user=None, exp_ns_max_user=None):
        """Determine the max and min no slip exponents.

        Parameters
        ----------
        meas: MeasurementData
            Object of MeasurementData
        ns_exp: list
            List of maximum and minimum no slip exponents.
        exp_ns_min_user: float
            User supplied minimum no slip exponent
        exp_ns_max_user: float
            User supplied maximum no slip exponent

        Returns
        -------
        skip_ns_min_max: bool
            Boolean to identify if no slip simulations should be skipped
        exp_ns_max: float
            Maximum no slip exponent to be used in simulations
        exp_ns_min: float
            Minimum no slip exponent to be used in simulations
        """
        skip_ns_min_max = False
        if len(ns_exp) == 0:
            skip_ns_min_max = True
            min_ns = meas.extrap_fit.q_sensitivity.ns_exp
            max_ns = meas.extrap_fit.q_sensitivity.ns_exp
        else:
            mean_ns = np.nanmean(ns_exp)
            if len(ns_exp) == 1:
                min_ns = ns_exp[0]-0.05
                max_ns = ns_exp[0]+0.05
            else:
                min_ns = np.nanmin(ns_exp)
                max_ns = np.nanmax(ns_exp)

            # Diff between mean NS exponent and min/max shouldn't be > 0.2
            if mean_ns - min_ns > 0.2:
                min_ns = mean_ns - 0.2
            if max_ns - mean_ns > 0.2:
                max_ns = mean_ns + 0.2

            # Check that 0 < exponent < 1
            if min_ns <= 0:
                min_ns = 0.01
            if max_ns >= 1:
                max_ns = 0.99

        # Apply user overides
        if exp_ns_min_user is None:
            exp_ns_min = min_ns
        else:
            exp_ns_min = exp_ns_min_user

        if exp_ns_max_user is None:
            exp_ns_max = max_ns
        else:
            exp_ns_max = exp_ns_max_user

        return skip_ns_min_max, exp_ns_max, exp_ns_min

    @staticmethod
    def depth_error_boat_motion(transect):
        """Relative depth error due to vertical velocity of boat
           the height [m] is vertical velocity times ensemble duration

        Parameters
        ----------
        transect: TransectData
            Object of TransectData

        Returns
        -------
        relative_error_depth: float
            Random depth error by ensemble
       """

        d_ens = transect.depths.bt_depths.depth_processed_m
        depth_vv = transect.boat_vel.bt_vel.w_mps * transect.date_time.ens_duration_sec
        relative_error_depth = np.abs(depth_vv) / d_ens
        relative_error_depth[np.isnan(relative_error_depth)] = 0.00
        return relative_error_depth

    @staticmethod
    def water_std_by_error_velocity(transect):
        """Compute the relative standard deviation of the water velocity using the fact that the error velocity is
        scaled so that the standard deviation of the error velocity is the same as the standard deviation
        of the horizontal water velocity.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData

        Returns
        -------
        std_ev_wt_ens: float
            Standard deviation of water track error velocity for each ensemble
        """

        # Computer water speed
        u_water = transect.w_vel.u_processed_mps
        v_water = transect.w_vel.v_processed_mps
        v_wa_cell_abs = np.sqrt(u_water ** 2 + v_water ** 2)

        # Use only valid error velocity data
        d_vel_filtered = np.tile([np.nan], transect.w_vel.d_mps.shape)
        d_vel_filtered[transect.w_vel.valid_data[0]] = transect.w_vel.d_mps[transect.w_vel.valid_data[0]]

        # Compute relative standard deviation of error velocity
        std_ev_wt = np.nanstd(d_vel_filtered) / np.abs(v_wa_cell_abs)
        std_ev_wt_ens = np.nanmedian(std_ev_wt, axis=0)
        # TODO consider substituting the overall std for nan rather than 0
        # all_std_ev_WT = np.nanstd(d_vel_filtered[:])
        # std_ev_wt_ens[np.isnan(std_ev_wt_ens)] = all_std_ev_WT
        std_ev_wt_ens[np.isnan(std_ev_wt_ens)] = 0.00
        return std_ev_wt_ens

    @staticmethod
    def boat_std_by_error_velocity(transect):
        """Compute the relative standard deviation of the boat velocity using the fact that the error velocity is
        scaled so that the standard deviation of the error velocity is the same as the standard deviation
        of the horizontal boat velocity.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData

        Returns
        -------
        std_ev_bt: float
            Standard deviation of bottom track error velocity
        """

        # Compute boat speed
        u_boat = transect.boat_vel.bt_vel.u_processed_mps
        v_boat = transect.boat_vel.bt_vel.v_processed_mps
        speed = np.sqrt(u_boat ** 2 + v_boat ** 2)

        # Use only valid error velocity data
        d_vel_filtered = np.tile([np.nan], transect.boat_vel.bt_vel.d_mps.shape)
        d_vel_filtered[transect.boat_vel.bt_vel.valid_data[0]] = \
            transect.boat_vel.bt_vel.d_mps[transect.boat_vel.bt_vel.valid_data[0]]

        # Compute relative standard deviation of error velocity
        all_std_ev_bt = np.nanstd(d_vel_filtered)
        std_ev_bt = np.abs(all_std_ev_bt) / speed
        # TODO Consider substituting the overall std for nan rather than 0
        # std_ev_bt[np.isnan(std_ev_bt)] = all_std_ev_bt
        std_ev_bt[np.isnan(std_ev_bt)] = 0.00

        return std_ev_bt

    @staticmethod
    def apply_u_rect(list_sims, col_name):
        """Compute the uncertainty using list of simulated discharges following a ranctangular law

        Parameters
        ----------
        list_sims: list
            List of simulation data frames to be used in the computation
        col_name: str
            Name of column in the data frames to be used in the computation

        Returns
        -------
        u_rect: float
            Result of rectangular law
        """

        # Combine data frames
        vertical_stack = pd.concat(list_sims, axis=0, sort=True)

        # Apply rectangular law
        u_rect = (vertical_stack.groupby(vertical_stack.index)[col_name].max()
                  - vertical_stack.groupby(vertical_stack.index)[col_name].min()) / (2 * (3 ** 0.5))

        return u_rect
