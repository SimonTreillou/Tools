import numpy as np

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# rms_error_interpolated: Compute RMS error between observed and model data after interpolation.
# skill_score: Calculate skill score between observed and model data after interpolation.
# nrmse: Calculate normalized root mean square error between model and observations.
# skill_Willmott: Calculate Willmott skill score between model and observations.
# ---------------------------------------------------------------

def nrmse(mod, obs):
    """
    Calculates the normalized root mean square error (NRMSE) as a percentage.
    Reference: Martins et al. 2022

    Parameters:
        mod (array-like): Model predictions.
        obs (array-like): Observed values.

    Returns:
        float: NRMSE in percentage.
    """
    mean_obs = np.mean(obs)
    skill = 100 * np.sqrt(np.nansum((obs - mod)**2) / np.nansum(obs**2))
    return skill

def rms_error_interpolated(x_obs, y_obs, x_mod, y_mod):
    """
    Compute RMS error between observed (x_obs, y_obs) and model (x_mod, y_mod),
    after interpolating the model onto the observation x-locations.

    Parameters
    ----------
    x_obs : array
        Observation coordinates (1D)
    y_obs : array
        Observation values (1D)
    x_mod : array
        Model coordinates (1D)
    y_mod : array
        Model values (1D)

    Returns
    -------
    rms : float
        Root-mean-square error
    y_mod_interp : array
        Model values interpolated onto x_obs
    """
    # Interpolate model to obs positions
    y_mod_interp = np.interp(x_obs, x_mod, y_mod)

    # RMS error
    rms = np.sqrt(np.mean((y_obs - y_mod_interp)**2))

    return rms


def skill_score(x_obs, T_obs, x_mod, T_mod):
    """
    Compute skill = 1 - <(obs - mod_interp)^2> / <(obs)^2>
    where the model is interpolated to observation positions.

    Parameters
    ----------
    x_obs : array
        Observation coordinates
    T_obs : array
        Observed values
    x_mod : array
        Model coordinates
    T_mod : array
        Model values

    Returns
    -------
    skill : float
        Skill score (1 = perfect, 0 = same as zero prediction)
    T_mod_interp : array
        Model interpolated onto obs positions
    """

    # Interpolate model to obs locations
    T_mod_interp = np.interp(x_obs, x_mod, T_mod)

    # Numerator: mean square error
    mse = np.mean((T_obs - T_mod_interp)**2)

    # Denominator: mean(obs^2)
    denom = np.mean(T_obs**2)

    # Skill score
    skill = 1 - mse / denom

    return skill

def skill_Willmott(mod, obs):
    """
    Calculates the Willmott skill score between model and observations.
    Reference: Rijnsdorp et al. 2017

    Parameters:
        mod (array-like): Model predictions.
        obs (array-like): Observed values.

    Returns:
        float: Willmott skill score (1 = perfect agreement, 0 = no skill).
    """
    mean_obs = np.mean(obs)
    skill = 1 - np.sum((mod - obs)**2) / np.sum((np.abs(mod - mean_obs) + np.abs(obs - mean_obs))**2)
    return skill