import numpy as np

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# compute_vorticity: Compute vorticity from velocity fields.
# enstrophy_spectrum: Compute isotropic wavenumber spectrum of enstrophy.
# ---------------------------------------------------------------

def compute_vorticity(varx,vary,dx,dy):
    """
    Compute vorticity.
    
    Parameters
    ----------
    varx,vary : 2D array
        Velocity fields in x and y directions.
    dx, dy : float
        Grid spacing in x and y directions (same units as x,y)
        
    Returns
    -------
    vort: 2D array
        Vorticity field.
    """
    if varx.ndim == 3 and vary.ndim == 3:
        vort = (vary[:, :, 1:] - vary[:, :, :-1]) / dx - (varx[:, 1:, :] - varx[:, :-1, :]) / dy
    elif varx.ndim == 4 and vary.ndim == 4:
        vort = (vary[:, :, :, 1:] - vary[:, :, :, :-1]) / dx - (varx[:, :, 1:, :] - varx[:, :, :-1, :]) / dy
    elif varx.ndim == 2 and vary.ndim == 2:
        vort = (vary[:, 1:] - vary[:, :-1]) / dx - (varx[1:, :] - varx[-1:, :]) / dy
    else:
        raise ValueError("Input arrays have incorrect dimensions (different shapes or unsupported number of dimensions).")
    return vort

def enstrophy_spectrum(omega, dx, dy):
    """
    Compute isotropic wavenumber spectrum of enstrophy.
    
    Parameters
    ----------
    omega : 2D array
        Vorticity field (already computed, real-valued)
    dx, dy : float
        Grid spacing in x and y directions (same units as x,y)
        
    Returns
    -------
    k_bins : 1D array
        Wavenumber bins
    E_k : 1D array
        Isotropic enstrophy spectrum
    """
    # Grid size
    nx, ny = omega.shape

    # Fourier transform of vorticity
    omega_hat = np.fft.fft2(omega)
    omega_hat = np.fft.fftshift(omega_hat)  # center zero freq

    # Compute wavenumbers
    kx = np.fft.fftfreq(nx, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2*np.pi
    kx, ky = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky), indexing='ij')
    kk = np.sqrt(kx**2 + ky**2)

    # Enstrophy spectral density
    enstrophy_hat = 0.5 * np.abs(omega_hat)**2 / (nx*ny)**2

    # Isotropic average (radial bins)
    kmax = np.max(kk)
    nbins = nx // 2
    k_bins = np.linspace(0, kmax, nbins+1)
    E_k = np.zeros(nbins)

    for i in range(nbins):
        mask = (kk >= k_bins[i]) & (kk < k_bins[i+1])
        E_k[i] = np.mean(enstrophy_hat[mask]) if np.any(mask) else 0.0

    # Midpoint of bins
    k_bins = 0.5 * (k_bins[:-1] + k_bins[1:])

    return k_bins, E_k
