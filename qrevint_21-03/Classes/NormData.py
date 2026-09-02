import numpy as np
import scipy.stats as sp
from MiscLibs.common_functions import cart2pol, pol2cart
from Classes.QComp import QComp

class NormData(object):
    """Class creates normalized depth and unit discharge or velocity.

    The constuctor method allows an object to be formed without any data.
    populate_data method creates normalized data for a single transect.
    create_composite method creates normalized data for all check transects.
    also allows only a portion of the data to be used in the
    normalization process by specifying the data extent.

    Attributes
    ----------
    file_name: str
        Name of transect file
    cell_depth_normalized: np.array(float)
        Normalized depth of cell
    unit_normalized: np.array(float)
        Normalized discharge or velocity for all depth cells
    unit_normalized_med: np.array(float)
        Median of normalized data within 5% partitions
    unit_normalized_no: np.array(int)
        Number of data points in each median
    unit_normalized_z: np.array(float)
        Relative depth for each median (5% increments)
    unit_normalized_25: np.array(float)
        Value for which 25% of normalized values are smaller
    unit_normalized_75: np.array(float)
        Value for which 75% or normalized values are larger
    data_type: str
        Type of data (v, q, V, or Q)
    data_extent: list
        Defines percent of data from start of transect to use, default [0, 100]
    valid_data: np.array(int)
        Index of median values with point count greater than threshold cutoff
    weights: np.array(float)
        Discharge based weights for computing a weighted median
    use_weights: bool
        Specifies if discharge weighted medians are to be used in the extrapolation fit
    sub_from_left: bool
        Specifies if when subsectioning the subsection should start from left to right.
    use_q: bool
        Specifies to use the discharge rather than the xprod when subsectioning
    """
    
    def __init__(self):
        """Creates object and initializes instance variables."""
        self.file_name = None  # Name of transect file
        self.cell_depth_normalized = None  # Normalized depth of cell
        self.unit_normalized = None  # Normalized discharge or velocity for all depth cells
        self.unit_normalized_med = None  # Median of normalized data within 5% partitions
        self.unit_normalized_no = None  # Number of data points in each median
        self.unit_normalized_z = None  # Relative depth for each median (5% increments)
        self.unit_normalized_25 = None  # Value for which 25% of normalized values are smaller
        self.unit_normalized_75 = None  # Value for which 75% or normalized values are larger
        self.data_type = 'q'  # Type of data (v, q, V, or Q)
        self.data_extent = None  # Defines percent of data from start of transect to use, default [0, 100]
        self.valid_data = np.array([])  # Index of median values with point count greater than threshold cutoff
        self.weights = np.array([])
        self.use_weighted = True
        self.sub_from_left = False
        self.use_q = False
        
    def populate_data(self, transect, data_type, threshold, data_extent=None, use_weighted=True, sub_from_left=True, use_q=True):
        """Computes the normalized values for a single transect.

        Parameters
        ----------
        transect: TransectData
            Object of TransectData
        data_type: str
            Type of data (v, q, V, or Q)
        threshold: int
            Number of data points in an increment for the increment to be valid.
        data_extent: list
            Defines percent of data from start of transect to use, default [0, 100]
        use_weighted: bool
            Specifies if discharge weighted medians are to be used in the extrapolation fit
        sub_from_left: bool
            Specifies if when subsectioning the subsection should start from left to right.
        use_q: bool
            Specifies to use the discharge rather than the xprod when subsectioning
        """

        # If the data extent is not defined set data_extent to zero to trigger all data to be used
        if data_extent is None:
            data_extent = [0, 100]

        self.sub_from_left = sub_from_left
        self.use_q = use_q
            
        # Get data copies to avoid changing original data
        filename = transect.file_name
        in_transect_idx = transect.in_transect_idx

        depths_selected = getattr(transect.depths, transect.depths.selected)
        cell_depth = np.copy(depths_selected.depth_cell_depth_m[:, in_transect_idx])
        cells_above_sl = transect.w_vel.cells_above_sl[:, in_transect_idx]
        cell_depth[cells_above_sl == False] = np.nan
        depth_ens = np.copy(depths_selected.depth_processed_m[in_transect_idx])

        w_vel_x = np.copy(transect.w_vel.u_processed_mps[:, in_transect_idx])
        w_vel_y = np.copy(transect.w_vel.v_processed_mps[:, in_transect_idx])

        invalid_data = np.logical_not(transect.w_vel.valid_data[0, :, in_transect_idx]).T
        w_vel_x[invalid_data] = np.nan
        w_vel_y[invalid_data] = np.nan

        boat_select = getattr(transect.boat_vel, transect.boat_vel.selected)
        if boat_select is not None:
            bt_vel_x = np.copy(boat_select.u_processed_mps[in_transect_idx])
            bt_vel_y = np.copy(boat_select.v_processed_mps[in_transect_idx])
        else:
            bt_vel_x = np.tile([np.nan], transect.boat_vel.bt_vel.u_processed_mps[in_transect_idx].shape)
            bt_vel_y = np.tile([np.nan], transect.boat_vel.bt_vel.u_processed_mps[in_transect_idx].shape)

        # Compute discharges
        xprod = np.multiply(w_vel_x, bt_vel_y) - np.multiply(w_vel_y, bt_vel_x)
        cell_size = depths_selected.depth_cell_size_m
        delta_t = transect.date_time.ens_duration_sec[in_transect_idx]
        q = np.multiply(xprod[:, in_transect_idx] * cell_size[:, in_transect_idx], delta_t)
        q_ens = np.nansum(q, 0)

        # Ensure all elements of xprod can be used to compute q (have a delta_t), first ensemble has no delta_t
        idx_invalid = np.where(np.isnan(delta_t))[0]
        xprod[:, idx_invalid] = np.nan

        if np.abs(np.nansum(abs(q_ens))) > 0:
            # Compute ensemble weights
            weight_ensemble = abs(q_ens) / np.nansum(abs(q_ens))

            # Apply weights to cells
            weights = np.tile(weight_ensemble, (cell_depth.shape[0], 1))
        else:
            weights = np.ones(w_vel_x.shape)

        # Compute normalized cell depth by average depth in each ensemble
        norm_cell_depth = np.divide(cell_depth, depth_ens)
        norm_cell_depth[norm_cell_depth < 0] = np.nan

        # If data type is discharge compute unit discharge for each cell
        if data_type.lower() == 'q':
            # Compute the cross product for each cell
            unit = xprod
        else:
            w_vel_x = np.copy(transect.w_vel.u_processed_mps[:, in_transect_idx])
            w_vel_y = np.copy(transect.w_vel.v_processed_mps[:, in_transect_idx])

            invalid_data = np.logical_not(transect.w_vel.valid_data[0, :, in_transect_idx]).T
            w_vel_x[invalid_data] = np.nan
            w_vel_y[invalid_data] = np.nan

            # Compute mean velocity components in each ensemble
            w_vel_mean_1 = np.nanmean(w_vel_x, 0)
            w_vel_mean_2 = np.nanmean(w_vel_y, 0)

            # Compute a unit vector
            direction, _ = cart2pol(w_vel_mean_1, w_vel_mean_2)
            unit_vec_1, unit_vec_2 = pol2cart(direction, 1)
            unit_vec = np.vstack([unit_vec_1, unit_vec_2])
            
            # Compute the velocity magnitude in the direction of the mean velocity of each
            # ensemble using the dot product and unit vector
            unit = np.tile([np.nan], w_vel_x.shape)
            for i in range(w_vel_x.shape[0]):
                unit[i, :] = np.sum(np.vstack([w_vel_x[i, :], w_vel_y[i, :]]) * unit_vec, 0)

            # Discharge weighting of velocity data is not permitted
            use_weighted = False

        # Adjust to positive value
        unit_total = np.nansum(np.nansum(unit), 0)
        if unit_total < 0:
            unit *= -1
            
        # Compute normalize unit values
        unit_norm = np.divide(unit, np.abs(np.nanmean(unit, 0)))

        # Apply extents if they have been specified
        if data_extent[0] != 0 or data_extent[1] != 100:
            if use_q:
                # Adjust cumulative sum direction based on start bank so that cumsum is always from left to right
                if transect.start_edge == 'Right' and sub_from_left:
                    q_ens_flipped = np.flip(q_ens)
                    q_cum = np.nancumsum(q_ens_flipped)
                    q_max = q_cum[-1]
                    q_cum = np.flip(q_cum)
                else:
                    q_cum = np.nancumsum(q_ens)
                    q_max = q_cum[-1]
            else:
                unit_ens = np.nansum(unit, 0)
                q_cum = np.nancumsum(unit_ens)
                q_max = q_cum[-1]
            # Adjust so total discharge is positive
            if q_max < 0:
                q_cum *= -1
                q_max *= -1

            # Apply extents
            unit_left = q_max * data_extent[0] / 100
            unit_right = q_max * data_extent[1] / 100
            idx_extent = np.where(np.logical_and(np.greater(q_cum, unit_left),
                                                 np.less(q_cum, unit_right)))[0]
            # if data_type.lower() == 'v':
            #     # Unit discharge is computed here because the unit norm could be based on velocity
            #     unit = np.multiply(w_vel_x, bt_vel_y) - np.multiply(w_vel_y, bt_vel_x)
            #     unit_ens = np.nansum(unit, 0)
            #     unit_total = np.nancumsum(unit_ens)
            #
            #     # Adjust so total discharge is positive
            #     if unit_total[-1] < 0:
            #         unit_total *= -1
                
            # # Apply extents
            # unit_lower = unit_total[-1] * data_extent[0] / 100
            # unit_upper = unit_total[-1] * data_extent[1] / 100
            # idx_extent = np.where(np.logical_and(np.greater(unit_total, unit_lower),
            #                                      np.less(unit_total, unit_upper)))[0]
            unit_norm = unit_norm[:, idx_extent]
            norm_cell_depth = norm_cell_depth[:, idx_extent]
            weights = weights[:, idx_extent]
            
        # If whole profile is negative make positive
        idx_neg1 = np.tile([np.nan], [unit_norm.shape[1], 1])
        idx_neg2 = np.tile([np.nan], [unit_norm.shape[1], 1])
        for c in range(unit_norm.shape[1]):
            idx_neg1[c] = len(np.where(unit_norm[:, c] < 0)[0])
            idx_neg2[c] = len(np.where(np.isnan(unit_norm[:, c]) == False)[0])
        idx_neg = np.squeeze(idx_neg1) == np.squeeze(idx_neg2)
        unit_norm[:, idx_neg] = unit_norm[:, idx_neg] * -1

        # Store results
        self.file_name = filename
        self.data_extent = data_extent
        self.data_type = data_type
        self.cell_depth_normalized = norm_cell_depth
        self.unit_normalized = unit_norm
        self.use_weighted = use_weighted
        self.weights = weights
        self.compute_stats(threshold)

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
        norm_data = []
        if hasattr(mat_data, 'normData'):
            for data in mat_data.normData:
                temp = NormData()
                temp.populate_from_qrev_mat(data)
                norm_data.append(temp)
        return norm_data

    def populate_from_qrev_mat(self, mat_data):
        """Populates the object using data from previously saved QRev Matlab file.

        Parameters
        ----------
        mat_data: mat_struct
           Matlab data structure obtained from sio.loadmat
        """

        self.file_name = mat_data.fileName
        self.cell_depth_normalized = mat_data.cellDepthNormalized
        self.unit_normalized = mat_data.unitNormalized
        self.unit_normalized_med = mat_data.unitNormalizedMed
        self.unit_normalized_no = mat_data.unitNormalizedNo
        self.unit_normalized_z = mat_data.unitNormalizedz
        self.unit_normalized_25 = mat_data.unitNormalized25
        self.unit_normalized_75 = mat_data.unitNormalized75
        self.data_type = mat_data.dataType
        self.data_extent = mat_data.dataExtent
        self.valid_data = mat_data.validData - 1
        if hasattr(mat_data, 'use_weighted'):
            self.use_weighted = mat_data.use_weighted
            self.weights = mat_data.weights
        else:
            self.use_weighted = False
            self.weights = None
        if hasattr(mat_data, 'use_q'):
            self.use_q = mat_data.use_q
        if hasattr(mat_data, 'sub_from_left'):
            self.sub_from_left = mat_data.sub_from_left

    def compute_stats(self, threshold):
        """Computes the statistics for the normalized data.

        Parameters
        ----------
        threshold: int
            Number of data points in an increment for the increment to be valid.
        """

        # Set averaging interval
        avg_interval = np.arange(0, 1.05, .05)

        # Initialize variables to nan
        unit_norm_med = np.tile([np.nan], len(avg_interval) - 1)
        unit_norm_med_no = np.tile([np.nan], len(avg_interval) - 1)
        unit_25 = np.tile([np.nan], len(avg_interval) - 1)
        unit_75 = np.tile([np.nan], len(avg_interval) - 1)
        avgz = np.tile([np.nan], len(avg_interval) - 1)

        # Process each normalized increment
        for i in range(len(avg_interval) - 1):
            condition_1 = np.greater(self.cell_depth_normalized, avg_interval[i])
            condition_2 = np.less_equal(self.cell_depth_normalized, avg_interval[i + 1])
            condition_3 = np.logical_not(np.isnan(self.unit_normalized))
            condition_all = np.logical_and(np.logical_and(condition_1, condition_2), condition_3)
            if np.any(condition_all):
                if self.data_type.lower() == 'q' and self.use_weighted:
                    results = self.weighted_quantile(self.unit_normalized[condition_all],
                                                     quantiles=[0.25, 0.5, 0.75],
                                                     sample_weight=self.weights[condition_all])
                    unit_25[i] = results[0]
                    unit_norm_med[i] = results[1]
                    unit_75[i] = results[2]
                else:
                    unit_25[i], unit_norm_med[i], unit_75[i] = sp.mstats.mquantiles(self.unit_normalized[condition_all],
                                                                                    alphap=0.5, betap=0.5)

                unit_norm_med_no[i] = np.sum(np.isnan(self.unit_normalized[condition_all]) == False)
                avgz[i] = 1 - np.nanmean(self.cell_depth_normalized[condition_all])

        # Mark increments invalid if they do not have sufficient data
        cutoff = np.nanmedian(unit_norm_med_no[unit_norm_med_no > 0]) * (threshold / 100)
        self.valid_data = np.where(unit_norm_med_no > cutoff)[0]

        self.unit_normalized_med = unit_norm_med
        self.unit_normalized_no = unit_norm_med_no
        self.unit_normalized_25 = unit_25
        self.unit_normalized_75 = unit_75
        self.unit_normalized_z = avgz

    @staticmethod
    def weighted_quantile(values, quantiles, sample_weight):
        """ Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!

        Parameters
        ----------
        values: ndarray(float)
            Array of normalized values
        quantiles: list
            List of quantiles to be computed
        sample_weight: ndarray(float)
            Weights for each value`

        Returns
        -------
        results: list
            List of values at specified quantiles

        """

        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles /= np.sum(sample_weight)

        results = []
        for quantile in quantiles:
            results.append(np.interp(quantile, weighted_quantiles, values))

        return results

    def create_composite(self, transects, norm_data, threshold):
        """Compute normalized data for measurement composite.

        Parameters
        ----------
        transects: list
            List of objects of TransectData
        norm_data: list
            List of objects of NormData
        threshold: int
            Number of data points in an increment for the increment to be valid.
        """

        # Initialize lists
        n_cells = []
        n_ens = [0]

        # Determine number of cells and ensembles for each transect
        for data in norm_data:
            n_cells.append(data.unit_normalized.shape[0])
            try:
                n_ens.append(data.unit_normalized.shape[1])
            except IndexError:
                n_ens.append(1)
        max_cells = max(n_cells)
        sum_ens = np.cumsum(n_ens)

        # Initialize normalized variables
        self.unit_normalized = np.tile([np.nan], (max_cells, sum_ens[-1]))
        self.cell_depth_normalized = np.tile([np.nan], (max_cells, sum_ens[-1]))
        self.weights = np.tile([np.nan], (max_cells, sum_ens[-1]))

        # Process each transect using data from only the checked transects
        for n in range(len(transects)):
            if transects[n].checked:
                self.unit_normalized[:n_cells[n], np.arange(sum_ens[n], sum_ens[n + 1])] \
                    = norm_data[n].unit_normalized
                self.weights[:n_cells[n], np.arange(sum_ens[n], sum_ens[n + 1])] \
                    = norm_data[n].weights
                self.cell_depth_normalized[:n_cells[n], np.arange(sum_ens[n], sum_ens[n + 1])] \
                    = norm_data[n].cell_depth_normalized
                # if self.data_extent is None:
                self.data_extent = norm_data[n].data_extent
                self.data_type = norm_data[n].data_type
                self.use_weighted = norm_data[n].use_weighted

        # Store data
        self.file_name = 'Measurement'
        self.compute_stats(threshold)
