import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime
from scipy.ndimage import gaussian_filter1d

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
