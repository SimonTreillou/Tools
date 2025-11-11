import numpy as np

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
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