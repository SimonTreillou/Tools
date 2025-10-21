import numpy as np

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# energy_spectrum : Compute isotropic kinetic energy spectrum from 2D velocity field
# ---------------------------------------------------------------

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
    """

    nx, ny = u.shape

    # 2D FFTs
    u_hat = np.fft.fftshift(np.fft.fft2(u))
    v_hat = np.fft.fftshift(np.fft.fft2(v))

    # Wavenumber grids (radians per meter)
    kx = np.fft.fftfreq(nx, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2*np.pi
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K = np.sqrt(KX**2 + KY**2)

    # Spectral energy density (Parseval-normalized)
    if components == 'both':
        spec2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (nx * ny)**2
    elif components == 'u':
        spec2d = 0.5 * np.abs(u_hat)**2 / (nx * ny)**2
    elif components == 'v':
        spec2d = 0.5 * np.abs(v_hat)**2 / (nx * ny)**2
    else:
        raise ValueError("components must be 'both', 'u', or 'v'")

    # Isotropic averaging
    kmax = np.max(K)
    if nbins is None:
        nbins = nx // 2
    kbins = np.linspace(0.0, kmax, nbins+1)
    E_k = np.zeros(nbins)
    k_mid = 0.5*(kbins[:-1] + kbins[1:])

    for i in range(nbins):
        mask = (K >= kbins[i]) & (K < kbins[i+1])
        if np.any(mask):
            E_k[i] = np.sum(spec2d[mask])
        else:
            E_k[i] = 0.0

    # Convert to spectral density per unit k (divide by shell width)
    dk = kbins[1] - kbins[0]
    E_k = E_k / dk

    return k_mid, E_k