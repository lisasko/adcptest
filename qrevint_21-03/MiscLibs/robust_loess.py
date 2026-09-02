"""robust_loess
This module computes a robust loess smooth using a quadratic model as defined by
W.S.Cleveland, (1979) "Robust Locally Weighted Regression and Smoothing Scatterplots",
Journal of the American Statistical Association, Vol 74, No. 368, pp. 829-836.
Both x and y values are required and are assumed to be 1D arrays (n,).

Example
-------

from MiscLibs.matlab_rloess import rloess

smooth_fit = rloess(x, y, span)
"""
import numpy as np

# Set constants used in multiple functions
eps = np.finfo('float').eps
seps = np.sqrt(eps)


def nearest_neighbors(num_neighbors, idx, x, valid_x):
    """Find the nearest k neighbors to x[i] that are not nan.

    Parameters
    ----------
    num_neighbors: int
        Number of neighbors to find
    idx: int
        Index for the target x value
    x: np.array
        1D array of the independent variable
    valid_x: bool
        Boolean array indicating valid x data.

    Returns
    -------
    neighbors_idx: int
        Indices for neighbors in x array
    """

    # Find neighbors
    if np.nansum(valid_x) <= num_neighbors:
        # If there are k points or fewer, then they are all neighbors
        neighbors_idx = np.where(valid_x == True)[0]
    else:
        # Find the distance to the k closest points
        distance = np.abs(x - x[idx])
        distance_sorted = np.sort(distance[valid_x])
        distance_neighbors = distance_sorted[num_neighbors - 1]

        # Find all points that are as close as or closer than the num_neighbors closest points
        close = np.array(distance <= distance_neighbors)

        # Find the indices of x that are both close and valid
        neighbors_idx = np.where(np.logical_and(close, valid_x) == True)[0]

    return neighbors_idx


def tricube_weights(distance):
    """ Convert distances into weights using tri-cubic weight function.
    Note for Matlab: This function returns the square-root of the weights.

    Parameters
    ----------
    distance: np.array
        1D array of distances

    Returns
    -------
    weights: np.array
        1D array of weights
    """

    max_distance = np.max(distance)
    if max_distance > 0:
        distance = distance / max_distance
    weights = (1 - distance ** 3) ** 1.5
    return weights


def bisquare(data):
    """Bisqure weight function which for values greater than are equal to 1 are set to zero.

    Parameters
    ----------
    data: np.array
        1D array of data used to compute weight

    Returns
    -------
    weights: np.array
        Computed weight

    """
    weights = np.zeros(data.shape)
    idx = np.abs(data) < 1
    weights[idx] = np.abs(1 - data[idx] ** 2)
    return weights


def robust_weights(residuals, max_eps):
    """Compute robust weights using residuals.

    Parameters
    ----------
    residuals: np.array
        1D array of residuals from previous fit
    max_eps: float
        Smallest value to be represented

    Returns
    -------
    weights: np.array
        1D array of computed weights
    """

    # Compute median using only valid data
    s = np.nanmax([1e8 * max_eps, np.nanmedian(np.abs(residuals))])

    # Compute weights
    weights = bisquare(residuals / (6 * s))
    weights[np.isnan(residuals)] = 0

    return weights


def compute_loess(x, y, neighbors_idx, idx, r_weights=None):
    """Computes the loess smooth for the specified point x[i]. If robust weights are specified the computed weights
    are adjusted by the robust weights.

    Parameters
    ----------
    x: np.array(float)
        1D array of independent variable
    y: np.array(float)
        1D array of dependent variable
    neighbors_idx: np.array(int)
        1D array of indices of x defining neighbors
    idx: int
        Index of x defining target
    r_weights: np.array(float)
        1D array of robust weights

    Returns
    -------
    smoothed_value: float
        Computed smoothed value for target
    """
    # Center around current point to improve conditioning
    distances = x[neighbors_idx] - x[idx]
    distances_abs = np.abs(distances)
    neighbors_y = y[neighbors_idx]

    weights = tricube_weights(distances_abs)

    # If all weights are 0, skip weighting
    if np.all(weights < seps):
        weights[:] = 1

    if r_weights is not None:
        weights = weights * r_weights[neighbors_idx]

    weighted_x_matrix = np.array([np.ones(distances.shape), distances])
    weighted_x_matrix = np.vstack((weighted_x_matrix, distances * distances))
    weighted_x_matrix = np.tile(weights, (weighted_x_matrix.shape[0], 1)) * weighted_x_matrix
    neighbors_y = weights * neighbors_y

    # Solve using least squares
    try:
        mask = ~np.isnan(weighted_x_matrix.T) & ~np.isnan(neighbors_y.T)
        smoothed_values, _, _, _ = np.linalg.lstsq(weighted_x_matrix.T[mask],
                                                   neighbors_y.T[mask], rcond=None)
    except (IndexError, ValueError):
        smoothed_values, _, _, _ = np.linalg.lstsq(weighted_x_matrix.T,
                                                   neighbors_y.T, rcond=None)

    return smoothed_values[0]


def rloess(x, y, span):
    """This function computes a robust loess smooth using a quadratic model as defined by
    W.S.Cleveland, (1979) "Robust Locally Weighted Regression and Smoothing Scatterplots",
    Journal of the American Statistical Association, Vol 74, No. 368, pp. 829-836.
    Both x and y values are required and are assumed to be 1D arrays (n,).

    Parameters
    ----------
    x: np.array
        1D array of independent variable
    y: np.array
        1D array of dependent variable
    span: int
        Number of neighbors to use in the regression
    """

    # Number of cycles of the robust fit
    cycles = 5

    n_points = len(y)
    smoothed_values = np.copy(y)

    if span > 1:

        diff_x = np.diff(x)

        # Assumes non-uniform x
        y_nan = np.isnan(y)
        any_nans = np.any(y_nan[:])
        the_diffs = np.concatenate((np.array([1]), diff_x, np.array([1])), axis=0)

        # Pre-allocate space for lower and upper indices for each fit
        lower_bound = np.zeros(n_points).astype(int)
        upper_bound = np.zeros(n_points).astype(int)

        # Compute the non-robust smooth
        for n in range(n_points):

            # if x[i] and x[i-1] are equal just use previous fit
            if the_diffs[n] == 0:

                smoothed_values[n] = smoothed_values[n-1]
                lower_bound[n] = lower_bound[n-1].astype(int)
                upper_bound[n] = upper_bound[n-1].astype(int)

            else:

                # Find nearest neighbors
                neighbors_idx = nearest_neighbors(span, n, x, np.logical_not(y_nan))
                # Store neighbors for robust loop
                lower_bound[n] = np.min(neighbors_idx).astype(int)
                upper_bound[n] = np.max(neighbors_idx).astype(int)

                if len(neighbors_idx) < 1:
                    smoothed_values[n] = np.nan
                else:
                    smoothed_values[n] = compute_loess(x, y, neighbors_idx, n)
        # Non-robust fit complete

        # Compute residual and apply robust fit
        max_absy_eps = np.max(np.abs(y)) * eps
        for cycle in range(cycles - 1):
            residuals = y - smoothed_values

            # Compute robust weights
            r_weights = robust_weights(residuals, max_absy_eps)

            # Find new value for each point
            for n in range(n_points):
                if n > 0 and x[n] == x[n-1]:
                    smoothed_values[n] = smoothed_values[n-1]
                else:
                    if not np.isnan(smoothed_values[n]):
                        neighbors_idx = np.array(list(range(lower_bound[n], upper_bound[n] + 1)))

                        if any_nans:
                            neighbors_idx = neighbors_idx[np.logical_not(y_nan[neighbors_idx])]

                        if np.any(r_weights[neighbors_idx] <= 0):
                            neighbors_idx = nearest_neighbors(span, n, x, (r_weights > 0))

                        smoothed_values[n] = compute_loess(x, y, neighbors_idx, n, r_weights)
    return smoothed_values
