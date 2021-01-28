import sys
import warnings
import time
import numpy as np

WARN_MIN_TRIPLETS = 100  # Recommended number or triplets for an optimum result

if not sys.warnoptions:
    warnings.simplefilter("always")


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'Processed {method.__name__} in {(te - ts):5.2f} s')
        return result

    return timed


@timeit
def etc(q_hat):
    """
    Extended Triple Collocation (ETC).

    ETC estimates the variance of the noise error (RMSE_u)
    and correlation coefficients (r) of three measurement systems
    (e.g., satellite, in-situ and model-based products)
    with respect to the unknown true value of the variable being measured
    (e.g., turbulent fluxes, soil moisture, ...).

    Input variables from which covariance matrix is computed do not need to be
    re-scaled.

    Parameters
    ----------
    q_hat : 2D or 3D array
        Covariance matrix of shape (3, 3) or (3, 3, f) of the three
        unscaled spatially-collocated measurement systems.
        f is the number of different collocated points.

    Returns
    -------
    stderr : array
        array of 3 standard errors for each of the measurement systems (x, y, z).
    rho : array
        array of 3 correlation coefficients for each of the measurement systems
        (x, y and z) with respect to the unknown truth.
        It is also named sensitivity of the measurement system to the target variable.

    References
    ----------
    .. [McColl_2014] McColl, K.A., J. Vogelzang, A.G. Konings, D. Entekhabi,
        M. Piles, A. Stoffelen (2014).
        Extended Triple Collocation: Estimating errors and
        correlation coefficients with respect to an unknown target.
        Geophysical Research Letters 41:6229-6236
        https://doi.org/10.1002/2014GL061322

    .. [Gruber_2016] Gruber, A., Su, C.-H., Zwieback, S., Crow, W., Dorigo, W.,
        Wagner, W., 2016.
        Recent advances in (soil moisture) triple collocation analysis.
        International Journal of Applied Earth Observation and Geoinformation 45, 200–211.
        https://doi.org/10.1016/j.jag.2015.09.002

    """

    # System sensitivity to variations in the true signal
    # Eq. 7 of [Gruber_2016]_
    sensitivity = np.array([q_hat[0, 1] * q_hat[0, 2] / q_hat[1, 2],
                            q_hat[0, 1] * q_hat[1, 2] / q_hat[0, 2],
                            q_hat[0, 2] * q_hat[1, 2] / q_hat[0, 1]])

    # Eq. 5 of [McColl_2014]_
    errvar = np.asarray((q_hat[0, 0] - sensitivity[0],
                         q_hat[1, 1] - sensitivity[1],
                         q_hat[2, 2] - sensitivity[2]))

    no_valid = errvar < 0
    if np.any(no_valid):
        warnings.warn("At least one calculated error variance is negative. "
                      "This can happen if the sample size is too small, "
                      "or if one of the assumptions of CTC is violated.",
                      RuntimeWarning)
    stderr = np.full_like(errvar, np.nan)
    stderr[~no_valid] = np.sqrt(errvar[~no_valid])

    # Eq. 9 of [McColl_2014]_
    rho = (np.sqrt(q_hat[0, 1] * q_hat[0, 2] / (q_hat[0, 0] * q_hat[1, 2])),
           np.sign(q_hat[0, 2] * q_hat[1, 2]) * np.sqrt(q_hat[0, 1]
                   * q_hat[1, 2] / (q_hat[1, 1] * q_hat[0, 2])),
           np.sign(q_hat[0, 1] * q_hat[1, 2]) * np.sqrt(q_hat[0, 2]
                   * q_hat[1, 2] / (q_hat[2, 2] * q_hat[0, 1]))
           )

    rho = np.asarray(rho)

    # Signal-to-Noise Ratio in decibels Eq. 14 of [Gruber_2016]_
    snr_db = [(q_hat[0, 0] * q_hat[1, 2] / (q_hat[0, 1] * q_hat[0, 2])) - 1,
              (q_hat[1, 1] * q_hat[0, 2] / (q_hat[1, 0] * q_hat[1, 2])) - 1,
              (q_hat[2, 2] * q_hat[0, 1] / (q_hat[2, 0] * q_hat[2, 1])) - 1]

    snr_db = np.asarray(snr_db)

    snr_db[snr_db < 0] = np.nan
    snr_db = -10 * np.log10(np.asarray(snr_db))

    return stderr, rho, snr_db, sensitivity


@timeit
def ctc(q_hat):
    """
    Correlated Triple Collocation (CTC).

    CTC estimates the variance of the noise error (RMSE_u)
    and correlation coefficient (r) between the noises in the two
    a priori correlated systems, with respect to the unknown true value of
    the variable being measured e.g., turbulent fluxes, soil moisture, ...).

    Input variables used in the covariance matrix should have been rescaled,
    so that $\alpha_{1,2}=\alpha_{1,3}=1$.

    Parameters
    ----------
    q_hat : 2D or 3D array
        Covariance matrix of shape (3, 3) or (3, 3, f) of the three
        rescaled spatially-collocated measurement systems.
        f is the number of different collocated points.

    Returns
    -------
    stderr : tuple
        tuple of 3 standard errors for each of the measurement systems,
        corresponding to measurements x, y and z.
    rho_xy: float
        correlation coefficient between the errors of x and y.

    References
    ----------
    .. [GonzalezGambau_2020] González-Gambau, V., Turiel, A., González-Haro, C.,
        Martínez, J., Olmedo, E., Oliva, R., Martín-Neira, M., 2020.
        Triple Collocation Analysis for Two Error-Correlated Datasets:
        Application to L-Band Brightness Temperatures over Land.
        Remote Sensing 12, 3381.
        https://doi.org/10.3390/rs12203381

    """

    # Linear transformation of x and y into two new variables u and v
    # with errors that are uncorrelated Eq. 6 & 7 of [GonzalezGambau_2020]_
    s_1_prime = q_hat[0, 0] + q_hat[1, 1] - 2 * q_hat[0, 1]  # Eq.7 of [GonzalezGambau_2020]_
    u = (q_hat[1, 1] - q_hat[0, 1]) / s_1_prime  # Eq.6 of [GonzalezGambau_2020]_
    v = (q_hat[0, 0] - q_hat[0, 1]) / s_1_prime  # Eq.6 of [GonzalezGambau_2020]_
    # Order-2 moments of the uncorrelated-error variables u and v
    # Eq. 7 of [GonzalezGambau_2020]_
    s_2_prime = u**2 * q_hat[0, 0] + v**2 * q_hat[1, 1] + 2 * u * v * q_hat[0, 1]
    s_23_prime = u * q_hat[0, 1] + v * q_hat[1, 2]

    # Estimates for the error variances Eq. 8 of [GonzalezGambau_2020]_
    errvar = np.asarray([v**2 * s_1_prime + s_2_prime - s_23_prime,
                         u**2 * s_1_prime + s_2_prime - s_23_prime,
                         q_hat[2, 2] - s_23_prime])

    no_valid = errvar < 0
    if np.any(no_valid):
        warnings.warn("At least one calculated error variance is negative. "
                      "This can happen if the sample size is too small, "
                      "or if one of the assumptions of CTC is violated.",
                      RuntimeWarning)
    stderr = np.full_like(errvar, np.nan)
    stderr[~no_valid] = np.sqrt(errvar[~no_valid])

    # Covariance of errors between x and y Eq. 8 of [GonzalezGambau_2020]_
    rho_xy = -u * v * s_1_prime + s_2_prime - s_23_prime
    # Convert to Corrrelation of errors between x and y
    rho_xy = rho_xy / (stderr[0] * stderr[1])
    return stderr, rho_xy

@timeit
def scaling_factors(ref, y, z):
    """
    Compute scaling factors from a reference system based on triple collocation.

    Parameters
    ----------
    ref, y, z : array_like
        Arrays of shape (N,) or shape (N, f) of spatially-collocated measurement systems.
        N is the sample size and f is the number of different collocated points.
        `ref` is the variable use as reference for the rescaling

    Returns
    -------
    factor_y : float or array
        Rescaling factor between `y` and the reference system.
        If array (f, ) returns the rescaling factor for each of the different collocated points.
    factor_z : float or array
        Rescaling factor between `z` and the reference system.
        If array (f, ) returns the rescaling factor for each of the different collocated points.

    References
    ----------
    .. [Yilmaz_2014] Yilmaz, M.T., Crow, W.T., 2014.
        Evaluation of Assumptions in Soil Moisture Triple Collocation Analysis.
        Journal of Hydrometeorology 15, 1293–1302.
        https://doi.org/10.1175/JHM-D-13-0158.1

    """
    if ref.ndim == 1:
        q_hat = covariance_matrix(ref, y, z)
    else:
        q_hat = covariance_matrix_vec(ref, y, z)

    # Eqs. 4 & 5 of [Yilmaz_2014]_
    factor_y = q_hat[0, 2] / q_hat[1, 2]
    factor_z = q_hat[0, 1] / q_hat[2, 1]

    return factor_y, factor_z


def covariance_matrix(x, y, z):
    """Compute the Triple Collocation covariance matrix

    Parameters
    ----------
    x, y, z : array_like
        Arrays of shape (N,) of spatially-collocated measurement systems.
        N is the sample size
    """

    # Remove of any existing NaN in any of the collocated systems
    valid = np.logical_and.reduce((np.isfinite(x),
                                   np.isfinite(y),
                                   np.isfinite(z)))
    if np.sum(valid) < WARN_MIN_TRIPLETS:
        warnings.warn(f"Sample size is too small (n={np.sum(valid)}, "
                      "Consider increasing your sample size for robust TC metrics",
                      RuntimeWarning)

    if np.size(np.unique(x[valid])) <= 1 or np.size(np.unique(y[valid])) <= 1 \
            or np.size(np.unique(z[valid])) <= 1:
        print("ERROR: the sample variance of each of the variables x, y, z "
                      "must be non-zero. Increase your sample size or "
                      "reconsider using Triple Collocation Analysis")
        return None
    # Estimate covariance matrix of the three measurement systems
    q_hat = np.cov(np.stack((x[valid], y[valid], z[valid]), axis=0))

    return q_hat

@timeit
def covariance_matrix_vec(x, y, z):
    """Compute the vectorized Triple Collocation covariance matrix.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of shape (N, f) elements of spatially-collocated measurement systems.
        N is the sample size, f is the number of collocated points

    Returns
    -------
    q_hat : 3D array
        Array of shape (3, 3, f) with the covariance matrix (3x3)
        for each collocated point.
    """

    # Check all triplets have either data or nan
    no_valid = np.logical_or.reduce((np.isnan(x), np.isnan(y), np.isnan(z)))
    x[no_valid] = np.nan
    y[no_valid] = np.nan
    z[no_valid] = np.nan

    q_hat = np.zeros((3, 3, x.shape[1]))
    n = np.isfinite(x)
    n_1 = np.sum(n.T, axis=1) - 1

    x_diff = x - np.nanmean(x, axis=0)
    y_diff = y - np.nanmean(y, axis=0)
    z_diff = z - np.nanmean(z, axis=0)
    del x, y, z

    # Estimate covariance matrix of the three measurement systems
    q_hat[0, 0] = np.nansum(x_diff**2, axis=0) / n_1
    q_hat[0, 1] = np.nansum(x_diff * y_diff, axis=0) / n_1
    q_hat[1, 0] = q_hat[0, 1]
    q_hat[0, 2] = np.nansum(x_diff * z_diff, axis=0) / n_1
    q_hat[2, 0] = q_hat[0, 2]
    q_hat[1, 1] = np.nansum(y_diff**2, axis=0) / n_1
    q_hat[1, 2] = np.nansum(y_diff * z_diff, axis=0) / n_1
    q_hat[2, 1] = q_hat[1, 2]
    q_hat[2, 2] = np.nansum(z_diff**2, axis=0) / n_1
    return q_hat


def apply_calibration(ref, x, factor):
    """Apply the calibration coefficient between a variable and its reference

    Parameters
    ----------
    ref : array_like
        Measurement system with shape (N,) or (N, f) used as reference for the calibration
        N is the sample size and f is the number of different collocated points.
    x : array_like
        Measurement system e ith shape (N,) or (N, f) that will be calibrated
        N is the sample size and f is the number of different collocated points.
    factor : float or 1D array
        Calibration correction factor between ``x`` and ``ref``.
        If 1D array its shape must be (f,).

    Returns
    -------
    x_prime : array_like
        Calibrated ``x`` measurement system with shape (N,) or (N, f).
        N is the sample size and f is the number of different collocated points.

    References
    ----------
    .. [Yilmaz_2013] Yilmaz, M.T., Crow, W.T., 2013.
    The Optimality of Potential Rescaling Approaches in Land Data Assimilation.
    Journal of Hydrometeorology 14, 650–660.
    https://doi.org/10.1175/JHM-D-12-052.1
    """

    # Eq. 3 of [Yilmaz_2013]_
    ref_mean = np.nanmean(ref, axis=0)
    x_mean = np.nanmean(x, axis=0)
    x_prime = ref_mean + factor * (x - x_mean)

    return x_prime

