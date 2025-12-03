import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime
from scipy.ndimage import gaussian_filter1d
import os
import glob
from numpy.lib.stride_tricks import sliding_window_view

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# average_over_timesteps: Compute the average of a time series over consecutive segments.
# compute_coarse_grained_field: Coarse-grain a 2D field using circular top-hat convolution.
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

def compute_flux_term(var1, var2,axis=(0)):
    """
    Compute the flux term <var1' * var2'>, where var1' and var2' are the fluctuations
    of var1 and var2 from their mean values.

    Parameters
    ----------
    var1 : array_like
        First variable (2D array).
    var2 : array_like
        Second variable (2D array).

    Returns
    -------
    flux : ndarray
        Computed flux term (2D array).
    """
    var1 = np.asarray(var1)
    var2 = np.asarray(var2)

    var1_mean = np.mean(var1,axis=axis)
    var2_mean = np.mean(var2,axis=axis)
    var1_fluct = var1 - var1_mean
    var2_fluct = var2 - var2_mean

    flux = np.mean(var1_fluct * var2_fluct,axis=axis)

    return flux

def compute_coarse_grained_field(field, dx, scales):
    """
    Coarse-grain a field over x,y only using a circular top-hat convolution.
    Supports input shapes:
      - (ny, nx)
      - (t, ny, nx)
      - (t, z, ny, nx)
    Returns an array with the scale axis inserted just before the last two dims:
      e.g. (ny,nx) -> (nscale, ny, nx)
            (t,ny,nx) -> (t, nscale, ny, nx)
            (t,z,ny,nx) -> (t, z, nscale, ny, nx)

    dx can be:
      - scalar (constant grid spacing), or
      - 2D array of shape (ny, nx) (spatially varying)
    scales can be scalar or 1D array/ list.
    """

    arr = np.asarray(field)
    scales = np.atleast_1d(scales).astype(float)
    nscale = len(scales)

    if arr.ndim < 2:
        raise ValueError("field must have at least 2 dimensions (y,x).")
    ny, nx = arr.shape[-2], arr.shape[-1]
    leading_shape = arr.shape[:-2]

    # prepare grid spacing array (2D)
    if np.isscalar(dx):
        gs = np.full((ny, nx), float(dx))
    else:
        gs = np.asarray(dx)
        if gs.shape != (ny, nx):
            raise ValueError("dx must be scalar or shape (ny, nx) matching field's last two dims.")

    gs_min = float(gs.min())

    out_shape = leading_shape + (nscale, ny, nx)
    out = np.full(out_shape, np.nan, dtype=arr.dtype)

    # Helper to get/set slices for arbitrary leading dims
    if leading_shape:
        iterator = np.ndindex(*leading_shape)
    else:
        iterator = [()]

    # Precompute sliding windows for gs (same for every leading index)
    # Note: for each scale window size changes so we compute inside scale loop.
    for k, L in enumerate(scales):
        # conservative radius in pixels using smallest grid cell
        radius = int(round(L / gs_min / 2.0))
        w = 2 * radius + 1
        if w <= 1 or w > ny or w > nx:
            # window too small or too large -> skip (leave NaNs)
            continue

        # circular mask in meters: pixel distances * gs_min <= L/2
        yy, xx = np.indices((w, w))
        dist_pix = np.sqrt((yy - radius) ** 2 + (xx - radius) ** 2)
        circ = (dist_pix * gs_min) <= (L / 2.0)  # boolean mask (w,w)

        # sliding windows for gs (2D) -> shape (ny-w+1, nx-w+1, w, w)
        gsW = sliding_window_view(gs, (w, w))

        # area weight (meters^2)
        areaW = gsW ** 2  # shape (..., w, w)
        # mask area sum (shape: ny-w+1, nx-w+1)
        area_sum = np.sum(areaW * circ, axis=(-2, -1))

        # process each leading-indexed slice
        for idx in iterator:
            # select 2D slice
            if idx:
                f2d = arr[idx]
            else:
                f2d = arr

            # sliding windows of field: shape (ny-w+1, nx-w+1, w, w)
            fW = sliding_window_view(f2d, (w, w))

            # weighted sum over window
            f_weighted_sum = np.sum(fW * areaW * circ, axis=(-2, -1))

            # avoid divide by zero
            with np.errstate(invalid='ignore', divide='ignore'):
                cg_interior = np.where(area_sum > 0, f_weighted_sum / area_sum, np.nan)

            # place into output at appropriate location
            # out[leading_idx + (k, radius:ny-radius, radius:nx-radius)] = cg_interior
            if idx:
                out_idx = tuple(list(idx) + [k, slice(radius, ny - radius), slice(radius, nx - radius)])
            else:
                out_idx = (k, slice(radius, ny - radius), slice(radius, nx - radius))
            out[out_idx] = cg_interior

    return out

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
