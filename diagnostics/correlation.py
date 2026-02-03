import numpy as np
from . import useful

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# correlation_func: Compute the correlation coefficient between two datasets, handling NaNs.
# correlation_map: Compute a spatial map of correlation coefficients between two 3D datasets, masking
# ---------------------------------------------------------------

def correlation_func(data1, data2):
    """
    Compute the Pearson correlation coefficient between two datasets, handling NaNs.
    
    Args:
        data1: Array-like data for the first variable.
        data2: Array-like data for the second variable.
    
    Returns:
        float: Correlation coefficient between -1 and 1, or NaN if computation fails.
        
    References
    ----------
    Original implementation from Emma Shie Nuss (https://github.com/emmashie/funpy)
    """
    try:    
        data1_mean = np.nanmean(data1)
        data2_mean = np.nanmean(data2)

        data1_var = np.nanvar(data1)
        data2_var = np.nanvar(data2)

        cov = np.nanmean((data1-data1_mean) * (data2-data2_mean))
        corr = cov / np.sqrt(data1_var * data2_var)
    except:
        corr = np.nan
    
    return corr


def correlation_map(landmask, data1, data2):
    """
    Compute a spatial map of correlation coefficients between two 3D datasets, masking land areas.
    
    Args:
        landmask: 2D boolean array where True indicates masked (land) regions.
        data1: 3D array (time, y, x) for the first variable.
        data2: 3D array (time, y, x) for the second variable.
    
    Returns:
        2D masked array: Correlation map with land areas masked.
    
    References
    ----------
    Original implementation from Emma Shie Nuss (https://github.com/emmashie/funpy)
    """
    [NY, NX] = landmask.shape
    corrmap = np.zeros((NY, NX))
    if landmask.any():
        corrmap = np.ma.masked_where(landmask, corrmap)
    
    [yinds, xinds] = np.where(np.invert(landmask))
    for i in range(0, len(yinds)):
        corrmap[yinds[i], xinds[i]] = correlation_func(data1[:, yinds[i], xinds[i]],
                                                       data2[:, yinds[i], xinds[i]])
    
    return corrmap