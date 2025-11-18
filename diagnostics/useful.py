import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime
from scipy.ndimage import gaussian_filter1d
import os
import glob

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# average_over_timesteps: Compute the average of a time series over consecutive segments.
# smooth: Smooth an array using a moving average filter with convolution.
# wave_average: Compute the average of a 3D array over a specified time window.
# ---------------------------------------------------------------

def average_over_timesteps(p, Navg):
    """
    Compute the average of a time series over consecutive segments of length Navg.
    
    Parameters
    ----------
    p : array_like
        Input time series (1D array).
    Navg : int
        Number of timesteps per averaging window.
    
    Returns
    -------
    pavg : ndarray
        Averaged time series (length = floor(len(p) / Navg)).
    """
    p = np.asarray(p)
    Np = len(p)
    Nblocks = Np // Navg  # full blocks only

    # Reshape into blocks and take the mean along axis 1
    return p[:Nblocks * Navg].reshape(Nblocks, Navg).mean(axis=1)

def find_file(repo_path, suffix="_his.nc"):
    """
    Search for and return the path to a file in the given directory whose name ends with
    the specified suffix.

    This is a non-recursive search: only files directly under repo_path are considered.
    The function builds a glob pattern by concatenating "*" and the provided suffix
    (os.path.join(repo_path, "*" + suffix)).

    Parameters
    ----------
    repo_path : str or os.PathLike
        Path to the directory in which to search.
    suffix : str, optional
        Filename suffix to match (default: "_his.nc"). The suffix is appended to "*"
        to form the glob pattern, so provide the trailing part of the filename
        (for example "_diags_eddy_avg.nc" or ".nc").

    Returns
    -------
    str
        Full path to the single matching file.

    Raises
    ------
    FileNotFoundError
        If no file matching the pattern is found in repo_path.
    RuntimeError
        If more than one matching file is found; this function requires a single,
        unambiguous match.

    Example
    -------
    >>> find_file("/path/to/repo", suffix="_diags_eddy_avg.nc")
    '/path/to/repo/run_diags_eddy_avg.nc'
    """
    pattern = os.path.join(repo_path, "*" + suffix)
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No file ending with '{suffix}' found in: {repo_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matching files found: {matches}. Please specify a unique repo.")

    return matches[0]

def smooth(x,L):
    """
    Smooth a time series using a moving average filter of length L.
    
    Parameters
    ----------
    x : array_like
        Input 1D array.
    L : int
        Length of the moving average filter.
    
    Returns
    -------
    res : ndarray
        Smoothed array.
    """
    res = np.convolve(x,np.ones(L)/L,'same')
    res[0]=res[1]
    res[-1]=res[-2]
    return res

def wave_average(var,dt=1,T=13):
    """
    Compute the average of a 3D array over a specified time window.

    This function takes a 3D array and computes averages over specified time periods,
    reducing the temporal dimension while maintaining spatial dimensions.

    Parameters
    ----------
    var : numpy.ndarray
        3D array with shape (time, M, L) where time is the temporal dimension
        and M, L are spatial dimensions
    dt : float, optional
        Time step between consecutive time points (default is 1)
    T : float, optional
        Length of the averaging window in the same units as dt (default is 13)

    Returns
    -------
    numpy.ndarray
        3D array with reduced time dimension, shape (n, M, L) where
        n = floor(original_time_length / (T/dt))

    Notes
    -----
    - The function will truncate the input array if the total length is not
      perfectly divisible by the averaging window size
    - Each output time point represents an average over T/dt input time points
    """
    N=int(T/dt)
    T,M,L=var.shape
    n = int(T / N)
    var = var[:n * N, :, :]
    reshaped = var.reshape(-1, N, M, L)  # reshape en 10 lignes de 10 colonnes
    var_avg = reshaped.sum(axis=1) / N
    return var_avg
