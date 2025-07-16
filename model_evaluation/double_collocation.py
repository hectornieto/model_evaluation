import numpy as np
import scipy.stats as st
from scipy.interpolate import interpn

def descriptive_stats(obs, pre):
    """
    Calculates the typical descriptive measurements based on collocated
    observed vs. predicted measurements.

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed and 
        predicted systems.
        N is the sample size.

    Returns
    -------
    N : float
        Number of valid samples.
    mean_obs : float
        Average value of the observed.
    mean_pre : float
        Average value of the predicted.
    std_obs : float
        Standard deviation of the observed.
    std_pre : float
        Standard deviation of the predicted.

    References
    ----------
    .. [Willmott_1982] Willmott, C. J. (1982).
        Some Comments on the Evaluation of Model Performance,
        Bulletin of the American Meteorological Society, 63(11), 1309-1313.
        https://doi.org/10.1175/1520-0477(1982)063<1309:SCOTEO>2.0.CO;2
    """
    # Remove of any existing NaN in any of the collocated systems
    obs, pre = remove_nans(obs, pre)
    mean_obs = np.mean(obs)
    mean_pre = np.mean(pre)
    std_obs = np.std(obs)
    std_pre = np.std(pre)
    n = np.size(obs)

    return n, mean_obs, mean_pre, std_obs, std_pre


def error_metrics(obs, pre):
    """
    Calculates the typical error measurements based on collocated
    observed vs. predicted measurements.

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed and 
        predicted systems.
        N is the sample size.

    Returns
    -------
    mean_bias : float
        Average Error between the observed and the predicted.
    rmse : float
        Root Mean Squared Error.
    mae : float
        Mean Absolute Error

    References
    ----------
    .. [Willmott_1982] Willmott, C. J. (1982).
        Some Comments on the Evaluation of Model Performance,
        Bulletin of the American Meteorological Society, 63(11), 1309-1313.
        https://doi.org/10.1175/1520-0477(1982)063<1309:SCOTEO>2.0.CO;2
    """

    # Remove of any existing NaN in any of the collocated systems
    obs, pre = remove_nans(obs, pre)
    error = obs - pre
    mean_bias = np.mean(error)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    return mean_bias, mae, rmse


def centered_rmsd(obs, pre):
    obs, pre = remove_nans(obs, pre)
    # Calculate means
    obs_mean = np.mean(obs)
    pre_mean = np.mean(pre)

    crmsd = np.sqrt(np.mean(((pre - pre_mean) - (obs - obs_mean))**2))
    return crmsd


def agreement_metrics(obs, pre):
    """
    Calculates the typical agreement measurements based on collocated
    observed vs. predicted measurements.

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed and 
        predicted systems.
        N is the sample size.

    Returns
    -------
    cor : float
        Pearson correlation coefficient between the observed and the predicted.
    slope : float
        Slope of the linear regression between observed and predicted
    intercept : float
        Intercept of the linear regression between observed and predicted
    d : float
        Wilmott's Index of Agreement.

    References
    ----------
    .. [Willmott_1982] Willmott, C. J. (1982).
        Some Comments on the Evaluation of Model Performance,
        Bulletin of the American Meteorological Society, 63(11), 1309-1313.
        https://doi.org/10.1175/1520-0477(1982)063<1309:SCOTEO>2.0.CO;2

    """

    # Remove of any existing NaN in any of the collocated systems
    obs, pre = remove_nans(obs, pre)
    if np.size(obs) > 2:
        cor, p_value = st.pearsonr(obs, pre)
        slope, intercept = st.linregress(obs, pre)[:2]
    else:
        cor, p_value, slope, intercept = np.nan, np.nan, np.nan, np._financial_names

    obs_hat = np.mean(obs)
    pre_prime = pre - obs_hat
    obs_prime = obs - obs_hat
    d = 1 - np.sum((pre - obs)**2) / np.sum((np.abs(pre_prime)
                                              + np.abs(obs_prime))**2)

    return cor, p_value, slope, intercept, d


def rmse_wilmott_decomposition(obs, pre):
    """
    Calculates the Willmott's RMSE decomposition between its systematic
    and systematic components.

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed and 
        predicted systems.
        N is the sample size.

    Returns
    -------
    rmse_s : float
        Systematic RMSE, related to biases in the estimation
    rmse_u : float
        Unsystematic RMSE, related to random noise in the estimation

    References
    ----------
    .. [Willmott_1982] Willmott, C. J. (1982).
        Some Comments on the Evaluation of Model Performance,
        Bulletin of the American Meteorological Society, 63(11), 1309-1313.
        https://doi.org/10.1175/1520-0477(1982)063<1309:SCOTEO>2.0.CO;2
    """

    # Remove of any existing NaN in any of the collocated systems
    obs, pre = remove_nans(obs, pre)
    if np.size(obs) <= 2:
        return np.nan, np.nan

    slope, intercept = st.linregress(obs, pre)[:2]

    pre_hat = intercept + slope * obs

    rmse_s = np.sqrt(np.mean((pre_hat - obs) ** 2))
    rmse_u = np.sqrt(np.mean((pre - pre_hat) ** 2))

    return rmse_s, rmse_u


def rmse_stdev_decomposition(obs, pre):
    """
    Calculates the RMSE decomposition between its systematic
    and systematic components, based on standard-deviation-based rescaling of
    predicted values.

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed and 
        predicted systems.
        N is the sample size.

    Returns
    -------
    rmse_s : float
        Systematic RMSE, related to biases in the estimation
    rmse_u : float
        Unsystematic RMSE, related to random noise in the estimation

    References
    ----------
    .. [Willmott_1982] Willmott, C. J. (1982).
        Some Comments on the Evaluation of Model Performance,
        Bulletin of the American Meteorological Society, 63(11), 1309-1313.
        https://doi.org/10.1175/1520-0477(1982)063<1309:SCOTEO>2.0.CO;2
    """

    # Remove of any existing NaN in any of the collocated systems
    obs, pre = remove_nans(obs, pre)

    pre_hat = apply_calibration(obs, pre)
    rmse_u = np.sqrt(np.mean((pre_hat - obs) ** 2))

    return rmse_u


def create_significance_symbol(p_value, sig_thres=[(0.01, "**"), (0.05, "**")]):

    for thres, string in sig_thres:
        if p_value < thres:
            return string

    return ""


def scaling_factor(obs, pre):
    """
    Compute scaling factors from a reference system based on variances.

    Parameters
    ----------
        Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed 
        and predicted systems.
        N is the sample size.

    Returns
    -------
    factor : float
        Scaling factor for varible preds.

    References
    ----------
    .. [Yilmaz_2013] Yilmaz, M. T., & Crow, W. T. (2013).
        The Optimality of Potential Rescaling Approaches in 
        Land Data Assimilation.
        Journal of Hydrometeorology, 14(2), 650-660.
        https://doi.org/10.1175/JHM-D-12-052.1
    """

    obs, pre = remove_nans(obs, pre)

    # Eq. A15 of [Yilmaz_2013]_
    factor = np.std(obs) / np.std(pre)

    return factor


def apply_calibration(obs, pre):
    """Apply the calibration coefficient between a variable and its reference

    Parameters
    ----------
    obs : array_like
        Observed measurement system with shape (N,) or (N, f) used as reference
        for the calibration.
        N is the sample size and f is the number of different collocated points.
    pre : array_like
        Predicted measurement system e ith shape (N,) or (N, f) that will be 
        calibrated
        N is the sample size and f is the number of different collocated points.

    Returns
    -------
    pre_prime : array_like
        Calibrated ``pre`` system with shape (N,) or (N, f).
        N is the sample size and f is the number of different collocated points.

    References
    ----------
    .. [Yilmaz_2013] Yilmaz, M.T., Crow, W.T., 2013.
    The Optimality of Potential Rescaling Approaches in Land Data Assimilation.
    Journal of Hydrometeorology 14, 650–660.
    https://doi.org/10.1175/JHM-D-12-052.1
    """

    # Eq. 3 of [Yilmaz_2013]_
    n, mean_obs, mean_pre, std_obs, std_pre = descriptive_stats(obs, pre)
    scale = scaling_factor(obs, pre)
    pre_prime = mean_obs + scale * (pre - mean_pre)

    return pre_prime


def remove_nans(obs, pre):
    """Remove of any existing NaN in any of the collocated systems

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed 
        and predicted systems.
        N is the sample size.

    Returns
    -------
    obs, pre : array_like
        Observed and predicted arrays with filtered Nans.
    """
    obs, pre = obs.astype(float), pre.astype(float)
    valid = np.logical_and(np.isfinite(obs), np.isfinite(pre))
    obs = obs[valid]
    pre = pre[valid]
    return obs, pre


def density_plot(obs, pre, ax, **scatter_kwargs):
    """
    Creates a coloured density Observed vs. Predicted scatter plot on a given subplot

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed (y-axis)
        and predicted (x-axis) systems.
        N is the sample size.
    ax : Object
        Single matplotlib Axes object on which the density scatterplot will be
        displayed
    **scatter_kwargs :
        Any additional keyword arguments passed to the pyplot.scatter call.

    References
    ----------
    .. [Piñeiro2008] Piñeiro, G., Perelman, S., Guerschman, J. P.,
        & Paruelo, J. M., 2008
        How to evaluate models: observed vs. predicted or
        predicted vs. observed?.
        Ecological modelling, 216(3-4), 316-322.
        https://doi.org/10.1016/j.ecolmodel.2008.05.006
    """
    pre, obs = remove_nans(pre, obs)
    data, pre_e, obs_e = np.histogram2d(pre, obs, bins=30, density=True)
    z = interpn((0.5 * (pre_e[1:] + pre_e[:-1]),
                 0.5 * (obs_e[1:] + obs_e[:-1])),
                data,
                np.array([pre, obs]).T,
                method="splinef2d",
                bounds_error=False)

    # Observed (in the y-axis) vs. predicted (in the x-axis) (OP) regressions.
    ax.scatter(pre, obs, c=z, **scatter_kwargs)


def nse(obs, pre):
    """Nash-Sutcliffe efficiency

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed 
        and predicted systems.
        N is the sample size.

    Returns
    -------
    e : array_like
        Nash-Sutcliffe efficiency index
    """

    obs, pre = remove_nans(obs, pre)
    mean_obs = np.mean(obs)
    e = 1. - np.sum((obs - pre)**2) / np.sum((obs - mean_obs)**2)
    return e


def kge(obs, pre):
    """Kling-Gupta Efficiency

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed and 
        predicted systems.
        N is the sample size.

    Returns
    -------
    e : array_like
        Nash-Sutcliffe efficiency index
    """

    obs, pre = remove_nans(obs, pre)
    _, mean_obs, mean_pre, std_obs, std_pre = descriptive_stats(obs, pre)
    cor = agreement_metrics(obs, pre)[0]
    # Bias ratio
    beta = mean_pre / mean_obs
    # Variability ratio
    gamma = (std_pre * mean_obs) / (std_obs * mean_pre)
    e = 1 - np.sqrt((cor - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return e


def chi_squared(obs, pre, unc_obs, unc_pre):
    """Chi-squared

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed 
        and predicted systems.
        N is the sample size.
    unc_obs, unc_pre : array_like
        Arrays of N elements of spatially-collocated uncertainties
        for the observed and predicted. N is the sample size.

    Returns
    -------
    chi : array_like
        The goodness of fit between the actual and estimated uncertainties of
        measurement and validation values
    """
    valid = np.logical_and.reduce((np.isfinite(obs),
                                   np.isfinite(pre),
                                   np.isfinite(unc_obs),
                                   np.isfinite(unc_pre)))
    obs = obs[valid]
    pre = pre[valid]
    unc_obs = unc_obs[valid]
    unc_pre = unc_pre[valid]
    chi2 = np.sum(((pre - obs)**2) / (unc_pre**2 + unc_obs**2))
    p_val = st.chi2.sf(chi2, df=np.sum(valid))
    return chi2, p_val


def ora(obs_min, obs_max, pre):
    """
    Observation range adjusted

    Creates a pseudo-observation dataset that is equal to the model when
    it is within the observation range and is equal to the nearest
    observation when the model is outside the observation range

    Parameters
    ----------
    obs_min : array_like
        Minimum values of observations
    obs_max : array_like
        Maximum values of observations
    pre : array_like
        Predicted values

    Returns
    -------
    pseudo_obs : array_like
        Pseudo observation, adjusted by observation range uncertainty

    References
    ----------
    .. [Evans_2024] J P Evans and H M Imran, 2024
        The observation range adjusted method:
        a novel approach to accounting for observation uncertainty in
        model evaluation
        Environmental Research Communincations 6 (7), 071001
        https://doi.org/10.1088/2515-7620/ad5ad8
    """
    valid = np.logical_and.reduce((np.isfinite(obs_min),
                                   np.isfinite(obs_max),
                                   np.isfinite(pre)))

    pseudo_obs = pre.copy()
    pseudo_obs[~valid] = np.nan
    case = np.logical_and(valid, pre > obs_max)
    pseudo_obs[case] = obs_max[case]
    case = np.logical_and(valid, pre < obs_min)
    pseudo_obs[case] = obs_min[case]
    return pseudo_obs

