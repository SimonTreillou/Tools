import numpy as np
from . import spectrum

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# compute_eke: Compute eddy kinetic energy from velocity components
# energy_spectrum : Compute isotropic kinetic energy spectrum from 2D velocity field
# ---------------------------------------------------------------

def compute_eke(u, v):
    """
    Compute eddy kinetic energy (EKE) from velocity components.
    
    Removes the temporal mean (first dimension) from velocity fields and computes
    the kinetic energy of the anomalies. Handles 3D and 4D velocity fields.
    
    Parameters
    ----------
    u, v : ndarray
        Velocity components. Supported shapes:
        - (t, y, x): time, latitude, longitude
        - (t, z, y, x): time, depth, latitude, longitude
    
    Returns
    -------
    eke : ndarray
        Eddy kinetic energy [m^2/s^2]. Same shape as input.
    
    Notes
    -----
    EKE is computed as 0.5 * (u'^2 + v'^2) where u' and v' are anomalies
    from the temporal mean.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    
    # Remove temporal mean (first axis) regardless of dimensionality
    up = u - np.mean(u, axis=0, keepdims=True)
    vp = v - np.mean(v, axis=0, keepdims=True)
    
    eke = 0.5 * (up**2 + vp**2)
    
    return eke

def compute_energy_spectrum(u, v, dx, dy=None, 
                            mode='1D', 
                            component='both', 
                            nperseg=None,
                            noverlap=None,
                            window=None,
                            confidence=None,
                            isotropic_avg=False):
    """
    Compute energy wavenumber spectrum using PSD from velocity components.
    
    Parameters
    ----------
    u, v : ndarray
        Velocity components. Can be 1D or 2D arrays.
    dx, dy : float
        Grid spacing [m]. dy required for 2D, ignored for 1D.
    mode : str
        '1D' or '2D' for 1D or 2D spectrum
    component : str
        Which components to use: 'both' (default), 'u', or 'v'
    nperseg : int (optional)
        Length of each segment for PSD computation
    noverlap : int (optional)
        Number of points to overlap between segments
    window : str (optional)
        Window function to use (default: 'hann')
    confidence : float (optional)
        Confidence level for error bars (default: 0.95)
    isotropic_avg : bool (optional)
        Whether to perform isotropic averaging for 2D mode (default: False)
    
    Returns
    -------
    res : dict
        Dictionary with keys:
        - 'freqs': Wavenumbers [1/m] (1D mode) or 'freqs_x', 'freqs_y' (2D mode)
        - 'spectrum': Energy spectral density [m^3/s^2]
        - 'params': Dictionary of parameters used
    """
    u = np.asarray(u)
    v = np.asarray(v)
    if nperseg is None:
        nperseg = (u.shape[0] // 8)
    if noverlap is None:
        noverlap = nperseg // 2
    if window is None:
        window = 'hann'
    if confidence is None:
        confidence = 0.95

    args = {'nperseg': nperseg,
            'noverlap': noverlap,
            'window': window,
            'detrend': 'constant',
            'confidence': confidence}
    
    if mode == '1D':
        fs=1/dx

        res_uu = spectrum.compute_psd(u, fs, **args)
        res_vv = spectrum.compute_psd(v, fs, **args)
        
        if component == 'both':
            Seke = 0.5 * (res_uu['spectrum'] + res_vv['spectrum'])
        elif component == 'u':
            Seke = 0.5 * res_uu['spectrum']
        elif component == 'v':
            Seke = 0.5 * res_vv['spectrum']
        
        # Store results
        res = {
        'freqs': res_uu['freqs'],
        'spectrum': Seke,
        'params': {
            'fs': fs,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'window': window}
        }
        
    elif mode == '2D':
        fsx = 1/dx
        fsy = 1/dy if dy is not None else fsx

        res_uu = spectrum.compute_psd_2d(u, fs=(fsx,fsy),
                                      nperseg=(nperseg,nperseg),
                                      noverlap=(noverlap, noverlap),
                                      window=window)

        res_vv = spectrum.compute_psd_2d(v, fs=(fsx,fsy),
                                      nperseg=(nperseg,nperseg),
                                      noverlap=(noverlap, noverlap),
                                      window=window)
                
        if component == 'both':
            Seke = 0.5 * (res_uu['spectrum'] + res_vv['spectrum'])
        elif component == 'u':
            Seke = 0.5 * res_uu['spectrum']
        elif component == 'v':
            Seke = 0.5 * res_vv['spectrum']
            
        if isotropic_avg:
            # Convert to isotropic 1D spectrum
            res_iso = spectrum.convert_2Dpsd_to_1D({'freqs_x': res_uu['freqs_x'],
                                                    'freqs_y': res_uu['freqs_y'],
                                                    'spectrum': Seke}, mode='sum')
            res = {
            'freqs': res_iso['freqs'],
            'spectrum': res_iso['spectrum'],
            'std': res_iso['std'],
            'params': {
                'fs_x': fsx,
                'fs_y': fsy,
                'nperseg': nperseg,
                'noverlap': noverlap,
                'window': window}
            }
        else:
            # Store results without isotropic averaging
            res = {
            'freqs_x': res_uu['freqs_x'],
            'freqs_y': res_uu['freqs_y'],
            'spectrum': Seke,
            'params': {
                'fs_x': fsx,
                'fs_y': fsy,
                'nperseg': nperseg,
                'noverlap': noverlap,
                'window': window}
            }
        
    return res

# -------------- DEPRECATED FUNCTION --------------

def energy_spectrum(u, v, dx, dy, nbins=None, components='both'):
    """
    Compute isotropic kinetic energy spectrum E(k) from 2D velocity field.
    
    Parameters
    ----------
    u, v : 2D arrays
        Velocity components [m/s]
    dx, dy : float
        Grid spacing [m]
    nbins : int (optional)
        Number of bins for isotropic averaging
    components : str (optional)
        Which velocity components to use: 'both' (default), 'u' (cross-shore only), 
        or 'v' (longshore only)

    Returns
    -------
    k_mid : 1D array
        Wavenumber bins [rad/m]
    E_k : 1D array
        Energy spectrum density [m^3/s^2]

    Notes
    -----
    The lower bound of the wavenumber bins (`kbins`) is set to 1e-10 to avoid division by zero,
    which affects the lowest wavenumber bin.
    """

    nx, ny = u.shape

    # 2D FFTs
    u_hat = np.fft.fftshift(np.fft.fft2(u))
    v_hat = np.fft.fftshift(np.fft.fft2(v))

    # Wavenumber grids (radians per meter)
    kx = np.fft.fftfreq(ny, d=dy) * 2*np.pi
    ky = np.fft.fftfreq(nx, d=dx) * 2*np.pi
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    # Note: With indexing='xy', KX and KY have shape (ny, nx)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K = np.sqrt(KX**2 + KY**2)
    # Spectral energy density (Parseval-normalized)
    norm_factor = (nx * ny)**2
    if components == 'both':
        spec2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / norm_factor
    elif components == 'u':
        spec2d = 0.5 * np.abs(u_hat)**2 / norm_factor
    elif components == 'v':
        spec2d = 0.5 * np.abs(v_hat)**2 / norm_factor
    if nbins is None:
        nbins = min(nx, ny) // 2

    # Isotropic averaging
    kmax = np.max(K)
    if nbins is None:
        nbins = nx // 2
    kbins = np.linspace(1e-10, kmax, nbins+1)
    E_k = np.zeros(nbins)
    k_mid = 0.5*(kbins[:-1] + kbins[1:])

    for i in range(nbins):
        mask = (K >= kbins[i]) & (K < kbins[i+1])
        if np.any(mask):
            E_k[i] = np.sum(spec2d[mask])
        else:
            E_k[i] = 0.0

    # Convert to spectral density per unit k (divide by actual shell width)
    E_k = E_k / np.diff(kbins)

    return k_mid, E_k

def energy_spectrum_1d(u, dx, component='u'):
    res = compute_energy_spectrum(u, u, dx, dy=None, 
                            mode='1D', 
                            component=component)

    return res['freqs'], res['spectrum']