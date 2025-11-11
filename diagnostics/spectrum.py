from scipy.signal import welch, csd
from scipy.signal.windows import hann
import numpy as np
import numpy as np
from scipy import signal
from scipy.stats import chi2
import matplotlib.pyplot as plt

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# compute_cospectrum_quadspectrum: Compute co-spectrum and quad-spectrum between two signals.
# moment: Compute the k-th spectral moment of a power spectral density.
# plot_CIbar_loglog: Plot a confidence interval bar on a log-log power spectrum plot.
# spectrum: Compute the power spectrum of a signal using Welch's method.
# welch_spectrum_CI: Compute power spectrum with confidence intervals using Welch's method.
# ---------------------------------------------------------------

def compute_cospectrum_quadspectrum(x, z, fs, nperseg=None):
    """
    Compute co-spectrum and quad-spectrum between two signals using Welch's method.

    Parameters
    ----------
    x : array-like
        First input time series.
    z : array-like
        Second input time series.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method (default: determined by scipy).

    Returns
    -------
    f : ndarray
        Array of frequency bins.
    Cxz : ndarray
        Co-spectrum (real part of cross-spectral density).
    Qxz : ndarray
        Quad-spectrum (imaginary part of cross-spectral density).
    """
    f, Pxy = csd(x, z, fs=fs, nperseg=nperseg)
    Cxz = np.real(Pxy)
    Qxz = np.imag(Pxy)
    return f, Cxz, Qxz

def moment(f, S, k):
    """
    Compute the k-th spectral moment of a power spectral density.

    Parameters
    ----------
    f : array-like
        Frequency array.
    S : array-like
        Power spectral density values corresponding to f.
    k : int or float
        Order of the moment.

    Returns
    -------
    m : float
        The k-th moment computed as the integral of f**k * S over f.
    """
    m = np.trapz(f**k * S, f)
    return m

def plot_CIbar_loglog(f, S, ci_upper, color='k', x_pos=None, y_pos=None, lw=3):
    """
    Plot a confidence interval (CI) bar on a log-log power spectrum plot.

    Parameters
    ----------
    f : array-like
        Frequency array.
    S : array-like
        Power spectral density array.
    ci_upper : float
        Upper bound multiplier for the confidence interval (from welch_spectrum_CI).
    color : str, optional
        Color of the CI bar (default: 'k' for black).
    x_pos : float, optional
        X position (frequency) for the CI bar. Defaults to 80% of max frequency.
    y_pos : float, optional
        Y position (PSD) for the CI bar. Defaults to 80% of max PSD value.
    lw : int, optional
        Line width of the CI bar (default: 3).

    Notes
    -----
    The CI bar is drawn vertically at x_pos, centered at y_pos, with height determined by ci_upper.
    Intended for use on log-log plots.
    """
    if x_pos is None:
        x_pos = f[-1] * 0.8  # near top-right, 80% of max freq
    if y_pos is None:
        y_center = max(S) * 0.8  # position around the top of PSD curve
    else:
        y_center = y_pos
    # CI width (half height in log10 units)
    ci_half = ci_upper * S[0]  # same everywhere

    # Draw the bar (vertical line)
    plt.vlines(x=x_pos, ymin=y_center - ci_half, ymax=y_center + ci_half,
               colors=color, linewidth=lw)

    # Add small horizontal “caps”
    plt.hlines([y_center - ci_half, y_center + ci_half],
               x_pos * 0.95, x_pos * 1.05, colors=color, linewidth=lw)

def spectrum(var, dt, N=256, along=False):
    """
    Compute the power spectrum of a signal (or signals) using Welch's method.

    Parameters
    ----------
    var : array-like or 2D array
        Input signal. If 2D and along=True, averages spectra along axis 1.
    dt : float
        Time step between samples (1/sampling frequency).
    N : int, optional
        Length of each segment for Welch's method (default: 256).
    along : bool, optional
        If True and var is 2D, average spectra along axis 1 (default: False).

    Returns
    -------
    f : ndarray
        Array of frequency bins.
    S : ndarray
        Power spectral density estimate.
    """
    nfft = N
    noverlap = N / 2
    win = hann(nfft, True)
    if along:
        S = 0
        for i in range(var.shape[1]):
            f, S_tmp = welch(var[:, i], 1 / dt, window=win, noverlap=noverlap, nfft=nfft, return_onesided=True)
            S = S + S_tmp
        S = S / var.shape[1]
    else:
        f, S = welch(var, 1 / dt, window=win, noverlap=noverlap, nfft=nfft, return_onesided=True)
    return f, S

def welch_spectrum_CI(x, fs, nperseg=256, noverlap=None, alpha=0.05):
    """
    Compute power spectrum with degrees of freedom (DOF) and confidence intervals using Welch's method.

    Parameters
    ----------
    x : array-like
        Time series data.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each Welch segment. Default is 256.
    noverlap : int, optional
        Number of overlapping points. Default is nperseg//2.
    alpha : float, optional
        Significance level for confidence interval (e.g., 0.05 for 95% CI).

    Returns
    -------
    f : ndarray
        Frequency array.
    Pxx : ndarray
        Power spectral density estimate.
    dof : int
        Degrees of freedom for the PSD estimate.
    ci_lower : float
        Lower bound multiplier for the confidence interval.
    ci_upper : float
        Upper bound multiplier for the confidence interval.

    Notes
    -----
    The confidence interval for the true spectrum S is:
        [ci_lower * Pxx, ci_upper * Pxx]
    where ci_lower and ci_upper are multiplicative factors based on the chi-squared distribution.
    """
    if noverlap is None:
        noverlap = nperseg // 2  # Default overlap as in scipy

    # Compute Welch's power spectral density estimate
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Estimate degrees of freedom (DOF)
    step = nperseg - noverlap
    n_segments = (len(x) - noverlap) // step
    dof = 2 * n_segments

    # Compute confidence interval multipliers using chi-squared distribution
    ci_lower = dof / chi2.ppf(1 - alpha/2, dof)
    ci_upper = dof / chi2.ppf(alpha/2, dof)

    return f, Pxx, dof, ci_lower, ci_upper
