import numpy as np
import scipy.stats as st

def error_metrics(obs, pre):
    """
    Calculates the typical error measurements based on collocated
    observed vs. predicted measurements.

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed and predicted systems.
        N is the sample size.

    Returns
    -------
    mean_obs : float
        Average of the observed.
    mean_pre : float
        Average of the predicted.
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

    mean_pre = np.mean(pre)
    mean_obs = np.mean(obs)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    return mean_obs, mean_pre, mae, rmse


def agreement_metrics(obs, pre):
    """
    Calculates the typical agreement measurements based on collocated
    observed vs. predicted measurements.

    Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed and predicted systems.
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

    cor, p_value = st.pearsonr(obs, pre)
    slope, intercept = st.linregress(obs, pre)[:2]

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
        Arrays of N elements of spatially-collocated observed and predicted systems.
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

    slope, intercept = st.linregress(obs, pre)[:2]

    pre_hat = intercept + slope * obs

    rmse_s = np.sqrt(np.mean((pre_hat - obs) ** 2))
    rmse_u = np.sqrt(np.mean((pre - pre_hat) ** 2))

    return rmse_s, rmse_u


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
        Arrays of N elements of spatially-collocated observed and predicted systems.
        N is the sample size.

    Returns
    -------
    factor : float
        Scaling factor for varible preds.

    References
    ----------
    .. [Yilmaz_2013] Yilmaz, M. T., & Crow, W. T. (2013).
        The Optimality of Potential Rescaling Approaches in Land Data Assimilation.
        Journal of Hydrometeorology, 14(2), 650-660.
        https://doi.org/10.1175/JHM-D-12-052.1
    """

    obs, pre = remove_nans(obs, pre)

    # Eq. A15 of [Yilmaz_2013]_
    factor = np.std(obs) / np.std(pre)

    return factor


def remove_nans(obs, pre):
    """Remove of any existing NaN in any of the collocated systems

        Parameters
    ----------
    obs, pre : array_like
        Arrays of N elements of spatially-collocated observed and predicted systems.
        N is the sample size.

    Returns
    -------
    obs, pre : array_like
        Observed and predicted arrays with filtered Nans.
    """
    valid = np.logical_and(np.isfinite(obs), np.isfinite(pre))
    obs = obs[valid]
    pre = pre[valid]
    return obs, pre