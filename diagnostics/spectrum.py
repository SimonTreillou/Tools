from scipy.signal import welch, csd
from scipy.signal.windows import hann
import numpy as np
from scipy import signal,stats
from typing import Optional, Tuple, Dict

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# compute_cospectrum_quadspectrum: Compute co-spectrum and quad-spectrum between two signals.
# moment: Compute the k-th spectral moment of a power spectral density.
# plot_CIbar_loglog: Plot a confidence interval bar on a log-log power spectrum plot.
# spectrum: Compute the power spectrum of a signal using Welch's method.
# welch_spectrum_CI: Compute power spectrum with confidence intervals using Welch's method.
# compute_psd: Compute Power Spectral Density with confidence intervals.
# compute_csd: Compute Cross-Spectral Density.
# calculate_dof: Calculate degrees of freedom for Welch's method.
# confidence_interval: Calculate confidence intervals for spectrum estimate.
# compute_coherence: Compute magnitude-squared coherence between two signals.
# compute_phase: Compute phase spectrum between two signals.
# compute_transfer_function: Compute transfer function between two signals.
# welch_spectrum: Core Welch spectrum estimator with DOF and confidence intervals.
# ---------------------------------------------------------------

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

# ============================================================================
# CORE WELCH SPECTRUM FUNCTION
# ============================================================================

def welch_spectrum(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    nfft: Optional[int] = None,
    detrend: str = 'constant',
    confidence: Optional[float] = 0.95
) -> Dict:
    """
    Core Welch spectrum estimator with DOF and confidence intervals.
    
    This is the foundation function - all other spectral functions build on this.
    
    Parameters
    ----------
    x : np.ndarray
        First signal (or only signal for PSD)
    y : np.ndarray, optional
        Second signal (for CSD). If None, computes PSD of x
    fs : float
        Sampling frequency (Hz)
    nperseg : int, optional
        Length of each segment (default: 256 or len(x)//8)
    noverlap : int, optional
        Number of overlapping points (default: nperseg // 2)
    window : str
        Window function (default: 'hann')
    nfft : int, optional
        FFT length (default: nperseg)
    detrend : str
        Detrend type: 'constant', 'linear', or False
    confidence : float, optional
        Confidence level (0-1). Set to None to skip CI calculation
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'freqs': Frequency array
        - 'spectrum': Spectral values (PSD or CSD)
        - 'dof': Degrees of freedom
        - 'ci_lower': Lower confidence bound (PSD only)
        - 'ci_upper': Upper confidence bound (PSD only)
        - 'params': Parameters used
    
    Examples
    --------
    >>> # Simple PSD
    >>> result = welch_spectrum(signal, fs=1000)
    >>> plt.semilogy(result['freqs'], result['spectrum'])
    >>> 
    >>> # PSD with confidence intervals
    >>> result = welch_spectrum(signal, fs=1000, confidence=0.95)
    >>> plt.fill_between(result['freqs'], result['ci_lower'], result['ci_upper'], alpha=0.3)
    >>> 
    >>> # Cross-spectrum
    >>> result = welch_spectrum(x, y, fs=1000)
    """
    
    # Validate input
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be 1-dimensional")
    
    n_samples = len(x)
    
    # Set default parameters
    if nperseg is None:
        nperseg = min(256, n_samples // 8)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    
    # Compute spectrum
    if y is None:
        # Power Spectral Density
        freqs, spec = signal.welch(
            x, fs=fs, window=window, nperseg=nperseg,
            noverlap=noverlap, nfft=nfft, detrend=detrend
        )
        is_cross = False
    else:
        # Cross-Spectral Density
        y = np.asarray(y)
        if len(y) != n_samples:
            raise ValueError("Signals x and y must have same length")
        
        freqs, spec = signal.csd(
            x, y, fs=fs, window=window, nperseg=nperseg,
            noverlap=noverlap, nfft=nfft, detrend=detrend
        )
        is_cross = True
    
    # Calculate degrees of freedom
    dof = calculate_dof(n_samples, nperseg, noverlap, window)
    
    # Calculate confidence intervals (only for PSD)
    ci_lower, ci_upper = None, None
    if confidence is not None and not is_cross:
        ci_lower, ci_upper = confidence_interval(spec, dof, confidence)
    
    # Return results
    result = {
        'freqs': freqs,
        'spectrum': spec,
        'dof': dof,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'params': {
            'fs': fs,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'window': window,
            'n_samples': n_samples,
            'is_cross': is_cross
        }
    }
    
    return result


def calculate_dof(
    n_samples: int,
    nperseg: int,
    noverlap: int,
    window: str = 'hann'
) -> float:
    """
    Calculate degrees of freedom for Welch's method.
    
    DOF = 2 × n_segments × independence_factor
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    nperseg : int
        Samples per segment
    noverlap : int
        Overlap between segments
    window : str
        Window type (affects independence factor)
    
    Returns
    -------
    dof : float
        Degrees of freedom
    """
    # Number of segments
    step = nperseg - noverlap
    n_segments = (n_samples - noverlap) // step
    
    # Independence factor (correction for overlapping)
    overlap_fraction = noverlap / nperseg
    
    if overlap_fraction == 0:
        independence_factor = 1.0
    elif overlap_fraction == 0.5 and window == 'hann':
        # Optimal case: Hann window with 50% overlap
        independence_factor = 9/11  # ≈ 0.818
    else:
        # Approximate correction
        independence_factor = 1.0 - 0.5 * overlap_fraction
    
    dof = 2 * n_segments * independence_factor
    
    return dof


def confidence_interval(
    spectrum: np.ndarray,
    dof: float,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for spectrum estimate.
    
    The spectrum estimate follows a scaled chi-squared distribution.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Spectral estimate
    dof : float
        Degrees of freedom
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    ci_lower : np.ndarray
        Lower confidence bound
    ci_upper : np.ndarray
        Upper confidence bound
    """
    alpha = 1 - confidence
    
    # Chi-squared critical values
    chi2_lower = stats.chi2.ppf(alpha / 2, dof)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, dof)
    
    # Confidence intervals (to be multiplied by spectrum for plotting as shade)
    ci_lower = dof / chi2_upper
    ci_upper = dof / chi2_lower
    
    return ci_lower, ci_upper


# ============================================================================
# DERIVED SPECTRAL FUNCTIONS
# ============================================================================

def compute_psd(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    confidence: float = 0.95,
    **kwargs
) -> Dict:
    """
    Compute Power Spectral Density with confidence intervals.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    nperseg : int, optional
        Segment length
    noverlap : int, optional
        Overlap length
    confidence : float
        Confidence level (0.95 = 95%)
    **kwargs : dict
        Additional parameters for welch_spectrum
    
    Returns
    -------
    result : dict
        PSD results with confidence intervals
    
    Examples
    --------
    >>> result = compute_psd(signal, fs=1000, nperseg=512)
    >>> print(f"DOF: {result['dof']:.1f}")
    """
    return welch_spectrum(x, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                         confidence=confidence, **kwargs)


def compute_csd(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Compute Cross-Spectral Density.
    
    Parameters
    ----------
    x, y : np.ndarray
        Input signals
    fs : float
        Sampling frequency
    nperseg : int, optional
        Segment length
    noverlap : int, optional
        Overlap length
    **kwargs : dict
        Additional parameters for welch_spectrum
    
    Returns
    -------
    result : dict
        CSD results (complex-valued spectrum)
    
    Examples
    --------
    >>> result = compute_csd(x, y, fs=1000)
    >>> magnitude = np.abs(result['spectrum'])
    >>> phase = np.angle(result['spectrum'])
    """
    return welch_spectrum(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                         confidence=None, **kwargs)


def compute_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    confidence: float = 0.95,
    **kwargs
) -> Dict:
    """
    Compute magnitude-squared coherence.
    
    Coherence = |Pxy|² / (Pxx × Pyy)
    
    Parameters
    ----------
    x, y : np.ndarray
        Input signals
    fs : float
        Sampling frequency
    nperseg : int, optional
        Segment length
    noverlap : int, optional
        Overlap length
    confidence : float
        Confidence level for significance threshold
    **kwargs : dict
        Additional parameters
    
    Returns
    -------
    result : dict
        Contains:
        - 'freqs': Frequencies
        - 'coherence': Coherence values (0 to 1)
        - 'threshold': Significance threshold
        - 'dof': Degrees of freedom
    
    Examples
    --------
    >>> result = compute_coherence(x, y, fs=1000)
    >>> significant = result['coherence'] > result['threshold']
    >>> print(f"Significant frequencies: {result['freqs'][significant]}")
    """
    # Get individual PSDs and CSD
    psd_x = welch_spectrum(x, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                          confidence=None, **kwargs)
    psd_y = welch_spectrum(y, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                          confidence=None, **kwargs)
    csd_xy = welch_spectrum(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                           confidence=None, **kwargs)
    
    # Calculate coherence
    Pxx = psd_x['spectrum']
    Pyy = psd_y['spectrum']
    Pxy = csd_xy['spectrum']
    
    coh = np.abs(Pxy)**2 / (Pxx * Pyy)
    
    # Significance threshold
    # Under null hypothesis: threshold = 1 - alpha^(1/(dof/2 - 1))
    threshold = None
    if confidence is not None:
        alpha = 1 - confidence
        dof = psd_x['dof']
        threshold = 1 - alpha**(1 / (dof / 2 - 1))
    
    return {
        'freqs': psd_x['freqs'],
        'coherence': coh,
        'threshold': threshold,
        'dof': psd_x['dof'],
        'params': psd_x['params']
    }


def compute_phase(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    unwrap: bool = True,
    deg: bool = True,
    **kwargs
) -> Dict:
    """
    Compute phase spectrum between two signals.
    
    Parameters
    ----------
    x, y : np.ndarray
        Input signals
    fs : float
        Sampling frequency
    nperseg : int, optional
        Segment length
    noverlap : int, optional
        Overlap length
    unwrap : bool
        Unwrap phase to remove 2π discontinuities
    deg : bool
        Return phase in degrees (True) or radians (False)
    **kwargs : dict
        Additional parameters
    
    Returns
    -------
    result : dict
        Contains:
        - 'freqs': Frequencies
        - 'phase': Phase values
        - 'dof': Degrees of freedom
    
    Examples
    --------
    >>> result = compute_phase(x, y, fs=1000)
    >>> plt.plot(result['freqs'], result['phase'])
    >>> plt.ylabel('Phase (degrees)')
    """
    # Get CSD
    csd = welch_spectrum(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                        confidence=None, **kwargs)
    
    # Extract phase
    phase = np.angle(csd['spectrum'])
    
    if unwrap:
        phase = np.unwrap(phase)
    
    if deg:
        phase = np.degrees(phase)
    
    return {
        'freqs': csd['freqs'],
        'phase': phase,
        'dof': csd['dof'],
        'params': csd['params']
    }
    


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

    
def compute_transfer_function(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Estimate transfer function H(f) = Pxy / Pxx.
    
    Useful for system identification where x is input and y is output.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    y : np.ndarray
        Output signal
    fs : float
        Sampling frequency
    nperseg : int, optional
        Segment length
    noverlap : int, optional
        Overlap length
    **kwargs : dict
        Additional parameters
    
    Returns
    -------
    result : dict
        Contains:
        - 'freqs': Frequencies
        - 'H': Transfer function (complex)
        - 'magnitude': |H(f)|
        - 'phase': Phase of H(f) in degrees
        - 'dof': Degrees of freedom
    
    Examples
    --------
    >>> result = compute_transfer_function(input_signal, output_signal, fs=1000)
    >>> plt.subplot(211)
    >>> plt.semilogy(result['freqs'], result['magnitude'])
    >>> plt.subplot(212)
    >>> plt.plot(result['freqs'], result['phase'])
    """
    # Get PSD and CSD
    psd_x = welch_spectrum(x, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                          confidence=None, **kwargs)
    csd_xy = welch_spectrum(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, 
                           confidence=None, **kwargs)
    
    # Calculate transfer function
    H = csd_xy['spectrum'] / (psd_x['spectrum'] + 1e-12)
    
    return {
        'freqs': psd_x['freqs'],
        'H': H,
        'magnitude': np.abs(H),
        'phase': np.angle(H, deg=True),
        'dof': psd_x['dof'],
        'params': psd_x['params']
    }
    
    
# ============================================================================
# TO BE DEPRECATED FUNCTIONS
# ============================================================================

def spectrum(var, dt, N=256, along=False):
    result = compute_psd(var, fs=1/dt, nperseg=N)
    return result['freqs'], result['spectrum']

def welch_spectrum_CI(x, fs, nperseg=256, noverlap=None, alpha=0.05):
    result = compute_psd(x, fs=fs, nperseg=nperseg, noverlap=noverlap, confidence=1-alpha)
    return result['freqs'], result['spectrum'], result['dof'], result['ci_lower'], result['ci_upper']
