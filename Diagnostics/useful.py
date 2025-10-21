import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime
from scipy.ndimage import gaussian_filter1d

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# average_over_timesteps: Compute the average of a time series over consecutive segments.
# smooth: Smooth an array using a moving average filter with convolution.
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
    N=int(T/dt)
    T,M,L=var.shape
    n = int(T / N)
    var = var[:n * N, :, :]
    reshaped = var.reshape(-1, N, M, L)  # reshape en 10 lignes de 10 colonnes
    var_avg = reshaped.sum(axis=1) / N
    return var_avg
