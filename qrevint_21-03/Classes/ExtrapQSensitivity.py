import numpy as np
from Classes.QComp import QComp


class ExtrapQSensitivity(object):
    """Class to compute the sensitivity of the discharge to various extrapolation methods.

    Attributes
    ----------
    q_pp_mean: float
        Discharge power power 1/6
    q_pp_opt_mean: float
        Discharge power power optimized
    q_cns_mean: float
        Discharge constant no RoutingSlipDelivery
    q_cns_opt_mean: float
        Discharge constant optimized no slip
    q_3p_ns_mean: float
        Discharge 3-pt no slip
    q_3p_ns_opt_mean: float
        Discharge 3-pt optimized no slip
    q_pp_per_diff: float
        Power power 1/6 difference from reference
    q_pp_opt_per_diff: float
        Power power optimized percent difference from reference
    q_cns_per_diff: float
        Constant no slip percent difference from reference
    q_cns_opt_per_diff: float
        Constant optimized no slip percent difference from reference
    q_3p_ns_per_diff: float
        3-point no skip percent difference from reference
    q_3p_ns_opt_per_diff: float
        3-point optimized no slip percent difference from reference
    pp_exp: float
        Optimized power power exponent
    ns_exp: float
        Optimized no slip Exponent
    man_top: str
        Manually specified top method
    man_bot: str
        Manually specified bottom method
    man_exp: float
        Manually specified exponent
    q_man_mean: float
        Mean discharge for manually specified extrapolations
    q_man_per_diff: float
        Manually specified extrapolations percent difference from reference
    q_pp_list: list
        List of single transect discharges base on default 1/6 power-power law
    q_pp_opt_list: list
        List of single transect discharges base on optimized power-power law
    q_cns_list: list
        List of single transect discharges base on default 1/6 constant no slip law
    q_cns_opt_list: list
        List of single transect discharges base on optimized constant no slip law
    q_3p_ns_list: list
        List of single transect discharges base on default 3pt no slip
    q_3p_ns_opt_list: list
        List of single transect discharges base on optimized 3pt no slip
    q_top_pp_list: list
        List of single transect top discharges base on default 1/6 power-power law
    q_top_pp_opt_list: list
        List of single transect top discharges base on optimized power-power law
    q_top_cns_list: list
        List of single transect top discharges base on default 1/6 constant no slip law
    q_top_cns_opt_list: list
        List of single transect top discharges base on optimized constant no slip law
    q_top_3p_ns_list: list
        List of single transect top discharges base on default 3pt no slip
    q_top_3p_ns_opt_list: list
        List of single transect top discharges base on optimized 3pt no slip
    q_bot_pp_list: list
        List of single transect bottom discharges base on default 1/6 power-power law
    q_bot_pp_opt_list: list
        List of single transect bottom discharges base on optimized power-power law
    q_bot_cns_list: list
        List of single transect bottom discharges base on default 1/6 constant no slip law
    q_bot_cns_opt_list: list
        List of single transect bottom discharges base on optimized constant no slip law
    q_bot_3p_ns_list: list
        List of single transect bottom discharges base on default 3pt no slip
    q_bot_3p_ns_opt_list: list
        List of single transect bottom discharges base on optimized 3pt no slip
    """
    
    def __init__(self):
        """Initialize object and instance variables."""

        self.q_pp_mean = None  # Discharge power power 1/6
        self.q_pp_opt_mean = None  # discharge power power optimized
        self.q_cns_mean = None  # Discharge constant no RoutingSlipDelivery
        self.q_cns_opt_mean = None  # Discharge constant optimized no slip
        self.q_3p_ns_mean = None  # Discharge 3-pt no slip
        self.q_3p_ns_opt_mean = None  # Discharge 3-pt optimized no slip
        self.q_pp_per_diff = None  # Power power 1/6 difference from reference
        self.q_pp_opt_per_diff = None  # Power power optimized percent difference from reference
        self.q_cns_per_diff = None  # Constant no slip percent difference from reference
        self.q_cns_opt_per_diff = None  # Constant optimized no slip percent difference from reference
        self.q_3p_ns_per_diff = None  # 3-point no skip percent difference from reference
        self.q_3p_ns_opt_per_diff = None  # 3-point optimized no slip percent difference from reference
        self.pp_exp = None  # Optimized power power exponent
        self.ns_exp = None  # Optimized no slip Exponent
        self.man_top = None  # Manually specified top method
        self.man_bot = None  # Manually specified bottom method
        self.man_exp = None  # Manually specified exponent
        self.q_man_mean = None  # Mean discharge for manually specified extrapolations
        self.q_man_per_diff = None  # Manually specified extrapolations percent difference from reference
        self.q_pp_list = []  # List of single transect discharges base on default 1/6 power-power law
        self.q_pp_opt_list = []  # List of single transect discharges base on optimized power-power law
        self.q_cns_list = []  # List of single transect discharges base on default 1/6 constant no slip law
        self.q_cns_opt_list = []  # List of single transect discharges base on optimized constant no slip law
        self.q_3p_ns_list = []  # List of single transect discharges base on default 3pt no slip
        self.q_3p_ns_opt_list = []  # List of single transect discharges base on optimized 3pt no slip
        self.q_top_pp_list = []  # List of single transect top discharges base on default 1/6 power-power law
        self.q_top_pp_opt_list = []  # List of single transect top discharges base on optimized power-power law
        self.q_top_cns_list = []  # List of single transect top discharges base on default 1/6 constant no slip law
        self.q_top_cns_opt_list = []  # List of single transect top discharges base on optimized constant no slip law
        self.q_top_3p_ns_list = []  # List of single transect top discharges base on default 3pt no slip
        self.q_top_3p_ns_opt_list = []  # List of single transect top discharges base on optimized 3pt no slip
        self.q_bot_pp_list = []  # List of single transect bottom discharges base on default 1/6 power-power law
        self.q_bot_pp_opt_list = []  # List of single transect bottom discharges base on optimized power-power law
        self.q_bot_cns_list = []  # List of single transect bottom discharges base on default 1/6 constant no slip law
        self.q_bot_cns_opt_list = []  # List of single transect bottom discharges base on optimized constant no slip law
        self.q_bot_3p_ns_list = []  # List of single transect bottom discharges base on default 3pt no slip
        self.q_bot_3p_ns_opt_list = []  # List of single transect bottom discharges base on optimized 3pt no slip
        
    def populate_data(self, transects, extrap_fits):
        """Compute means and percent differences.

        Parameters
        ----------
        transects: list
            List of objects of TransectData
        extrap_fits: SelectFit
            Object of SelectFit
        """
        q_pp = []
        q_pp_opt = []
        q_cns = []
        q_cns_opt = []
        q_3p_ns = []
        q_3p_ns_opt = []
        self.pp_exp = extrap_fits[-1].pp_exponent
        self.ns_exp = extrap_fits[-1].ns_exponent

        # Store top discharges
        q_pp_top = []
        q_pp_opt_top = []
        q_cns_top = []
        q_cns_opt_top = []
        q_3p_ns_top = []
        q_3p_ns_opt_top = []

        # Store bottom discharges
        q_pp_bot = []
        q_pp_opt_bot = []
        q_cns_bot = []
        q_cns_opt_bot = []
        q_3p_ns_bot = []
        q_3p_ns_opt_bot = []

        # Compute discharges for each transect for possible extrapolation combinations
        for transect in transects:
            if transect.checked:
                q = QComp()

                q.populate_data(data_in=transect, top_method='Power', bot_method='Power', exponent=0.1667)
                q_pp.append(q.total)
                q_pp_top.append(q.top)
                q_pp_bot.append(q.bottom)

                q.populate_data(data_in=transect, top_method='Power', bot_method='Power', exponent=self.pp_exp)
                q_pp_opt.append(q.total)
                q_pp_opt_top.append(q.top)
                q_pp_opt_bot.append(q.bottom)

                q.populate_data(data_in=transect, top_method='Constant', bot_method='No Slip', exponent=0.1667)
                q_cns.append(q.total)
                q_cns_top.append(q.top)
                q_cns_bot.append(q.bottom)

                q.populate_data(data_in=transect, top_method='Constant', bot_method='No Slip', exponent=self.ns_exp)
                q_cns_opt.append(q.total)
                q_cns_opt_top.append(q.top)
                q_cns_opt_bot.append(q.bottom)

                q.populate_data(data_in=transect, top_method='3-Point', bot_method='No Slip', exponent=0.1667)
                q_3p_ns.append(q.total)
                q_3p_ns_top.append(q.top)
                q_3p_ns_bot.append(q.bottom)

                q.populate_data(data_in=transect, top_method='3-Point', bot_method='No Slip', exponent=self.ns_exp)
                q_3p_ns_opt.append(q.total)
                q_3p_ns_opt_top.append(q.top)
                q_3p_ns_opt_bot.append(q.bottom)

        # Compute mean discharge for each combination
        self.q_pp_mean = np.nanmean(q_pp)
        self.q_pp_opt_mean = np.nanmean(q_pp_opt)
        self.q_cns_mean = np.nanmean(q_cns)
        self.q_cns_opt_mean = np.nanmean(q_cns_opt)
        self.q_3p_ns_mean = np.nanmean(q_3p_ns)
        self.q_3p_ns_opt_mean = np.nanmean(q_3p_ns_opt)

        # Save all single-transect discharges
        self.q_pp_list = q_pp
        self.q_pp_opt_list = q_pp_opt
        self.q_cns_list = q_cns
        self.q_cns_opt_list = q_cns_opt
        self.q_3p_ns_list = q_3p_ns
        self.q_3p_ns_opt_list = q_3p_ns_opt

        # Save all single-transect top discharges
        self.q_top_pp_list = q_pp_top
        self.q_top_pp_opt_list = q_pp_opt_top
        self.q_top_cns_list = q_cns_top
        self.q_top_cns_opt_list = q_cns_opt_top
        self.q_top_3p_ns_list = q_3p_ns_top
        self.q_top_3p_ns_opt_list = q_3p_ns_opt_top

        # Save all single-transect bottom discharges
        self.q_bot_pp_list = q_pp_bot
        self.q_bot_pp_opt_list = q_pp_opt_bot
        self.q_bot_cns_list = q_cns_bot
        self.q_bot_cns_opt_list = q_cns_opt_bot
        self.q_bot_3p_ns_list = q_3p_ns_bot
        self.q_bot_3p_ns_opt_list = q_3p_ns_opt_bot

        self.compute_percent_diff(extrap_fits=extrap_fits, transects=transects)

    def populate_from_qrev_mat(self, mat_data):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        if hasattr(mat_data, 'qSensitivity'):
            self.q_pp_mean = mat_data.qSensitivity.qPPmean
            self.q_pp_opt_mean = mat_data.qSensitivity.qPPoptmean
            self.q_cns_mean = mat_data.qSensitivity.qCNSmean
            self.q_cns_opt_mean = mat_data.qSensitivity.qCNSoptmean
            self.q_3p_ns_mean = mat_data.qSensitivity.q3pNSmean
            self.q_3p_ns_opt_mean = mat_data.qSensitivity.q3pNSoptmean

            # For compatibility with older QRev.mat files
            if hasattr(mat_data.qSensitivity, 'qPPperdiff'):
                self.q_pp_per_diff = mat_data.qSensitivity.qPPperdiff
            else:
                self.q_pp_per_diff = np.nan

            self.q_pp_opt_per_diff = mat_data.qSensitivity.qPPoptperdiff
            self.q_cns_per_diff = mat_data.qSensitivity.qCNSperdiff
            self.q_cns_opt_per_diff = mat_data.qSensitivity.qCNSoptperdiff
            self.q_3p_ns_per_diff = mat_data.qSensitivity.q3pNSperdiff
            self.q_3p_ns_opt_per_diff = mat_data.qSensitivity.q3pNSoptperdiff
            self.pp_exp = mat_data.qSensitivity.ppExponent
            self.ns_exp = mat_data.qSensitivity.nsExponent

            # If a manual fit was used
            if len(mat_data.qSensitivity.manTop) > 0:
                self.man_top = mat_data.qSensitivity.manTop
                self.man_bot = mat_data.qSensitivity.manBot
                self.man_exp = mat_data.qSensitivity.manExp
                self.q_man_mean = mat_data.qSensitivity.qManmean
                self.q_man_per_diff = mat_data.qSensitivity.qManperdiff

            # Add compatibility for Oursin uncertainty model
            if hasattr(mat_data.qSensitivity, 'q_pp_list'):
                self.q_pp_list = mat_data.qSensitivity.q_pp_list
                self.q_pp_opt_list = mat_data.qSensitivity.q_pp_opt_list
                self.q_cns_list = mat_data.qSensitivity.q_cns_list
                self.q_cns_opt_list = mat_data.qSensitivity.q_cns_opt_list
                self.q_3p_ns_list = mat_data.qSensitivity.q_3p_ns_list
                self.q_3p_ns_opt_list = mat_data.qSensitivity.q_3p_ns_opt_list
                self.q_top_pp_list = mat_data.qSensitivity.q_top_pp_list
                self.q_top_pp_opt_list = mat_data.qSensitivity.q_top_pp_opt_list
                self.q_top_cns_list = mat_data.qSensitivity.q_top_cns_list
                self.q_top_cns_opt_list = mat_data.qSensitivity.q_top_cns_opt_list
                self.q_top_3p_ns_list = mat_data.qSensitivity.q_top_3p_ns_list
                self.q_top_3p_ns_opt_list = mat_data.qSensitivity.q_top_3p_ns_opt_list
                self.q_bot_pp_list = mat_data.qSensitivity.q_bot_pp_list
                self.q_bot_pp_opt_list = mat_data.qSensitivity.q_bot_pp_opt_list
                self.q_bot_cns_list = mat_data.qSensitivity.q_bot_cns_list
                self.q_bot_cns_opt_list = mat_data.qSensitivity.q_bot_cns_opt_list
                self.q_bot_3p_ns_list = mat_data.qSensitivity.q_bot_3p_ns_list
                self.q_bot_3p_ns_opt_list = mat_data.qSensitivity.q_bot_3p_ns_opt_list
            else:
                self.q_pp_list = []
                self.q_pp_opt_list = []
                self.q_cns_list = []
                self.q_cns_opt_list = []
                self.q_3p_ns_list = []
                self.q_3p_ns_opt_list = []
                self.q_top_pp_list = []
                self.q_top_pp_opt_list = []
                self.q_top_cns_list = []
                self.q_top_cns_opt_list = []
                self.q_top_3p_ns_list = []
                self.q_top_3p_ns_opt_list = []
                self.q_bot_pp_list = []
                self.q_bot_pp_opt_list = []
                self.q_bot_cns_list = []
                self.q_bot_cns_opt_list = []
                self.q_bot_3p_ns_list = []
                self.q_bot_3p_ns_opt_list =[]

    def compute_percent_diff(self, extrap_fits, transects=None):
        """Computes the percent difference for each of the extrapolation options as compared to selected method.

        Parameters
        ----------
        extrap_fits: SelectFit
            Object of SelectFit
        transects: list
            List of TransectData objects
        """
        # Determine which mean is the reference
        if extrap_fits[-1].fit_method == 'Manual':
            self.man_top = extrap_fits[-1].top_method
            self.man_bot = extrap_fits[-1].bot_method
            self.man_exp = extrap_fits[-1].exponent

            if transects is not None:
                q_man = []
                checked = []
                # Compute discharge for each transect
                for transect in transects:
                    q = QComp()
                    checked.append(transect.checked)

                    q.populate_data(data_in=transect,
                                    top_method=self.man_top,
                                    bot_method=self.man_bot,
                                    exponent=self.man_exp)
                    q_man.append(q)
                container = []
                for index, item in enumerate(q_man):
                    if checked[index]:
                        container.append(item.total)
                self.q_man_mean = np.nanmean(container)
            reference_mean = self.q_man_mean

        else:
            if extrap_fits[-1].top_method_auto == 'Power':
                if np.abs(extrap_fits[-1].exponent_auto - 0.1667) < 0.0001:
                    reference_mean = self.q_pp_mean
                else:
                    reference_mean = self.q_pp_opt_mean
            elif extrap_fits[-1].top_method_auto == 'Constant':
                if np.abs(extrap_fits[-1].exponent_auto - 0.1667) < 0.0001:
                    reference_mean = self.q_cns_mean
                else:
                    reference_mean = self.q_cns_opt_mean
            else:
                if np.abs(extrap_fits[-1].exponent_auto - 0.1667) < 0.0001:
                    reference_mean = self.q_3p_ns_mean
                else:
                    reference_mean = self.q_3p_ns_opt_mean

        # Compute percent difference from reference
        self.q_pp_per_diff = ((self.q_pp_mean - reference_mean) / reference_mean) * 100
        self.q_pp_opt_per_diff = ((self.q_pp_opt_mean - reference_mean) / reference_mean) * 100
        self.q_cns_per_diff = ((self.q_cns_mean - reference_mean) / reference_mean) * 100
        self.q_cns_opt_per_diff = ((self.q_cns_opt_mean - reference_mean) / reference_mean) * 100
        self.q_3p_ns_per_diff = ((self.q_3p_ns_mean - reference_mean) / reference_mean) * 100
        self.q_3p_ns_opt_per_diff = ((self.q_3p_ns_opt_mean - reference_mean) / reference_mean) * 100

        if extrap_fits[-1].fit_method == 'Manual':
            self.q_man_per_diff = ((self.q_man_mean - reference_mean) / reference_mean) * 100
