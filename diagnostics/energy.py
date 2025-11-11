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
    """
    Compute 1D kinetic energy spectrum from a velocity field.

    Parameters
    ----------
    u : 1D or 2D array
        Velocity component(s) [m/s].
        If 2D, average the spectrum along the second dimension.
    dx : float
        Grid spacing [m].
    component : str
        'u' (default) or 'v' for clarity if you use both components.

    Returns
    -------
    k : 1D array
        Wavenumber [rad/m].
    E_k : 1D array
        1D energy spectral density [m^3/s^2].
    """
    u = np.asarray(u)
    
    # If 2D: compute along x and average over y
    if u.ndim == 2:
        nx, ny = u.shape
        u_hat = np.fft.fft(u, axis=0)
        u_spec = np.mean(np.abs(u_hat)**2, axis=1)
    else:
        nx = u.size
        u_hat = np.fft.fft(u)
        u_spec = np.abs(u_hat)**2

    # Wavenumbers
    k = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    k = np.fft.fftshift(k)
    u_spec = np.fft.fftshift(u_spec)

    # Parseval normalization
    # (Total energy in physical = total energy in spectral space)
    spec_density = 0.5 * u_spec / (nx**2)

    # Only keep positive wavenumbers
    mask = k > 0
    k_pos = k[mask]
    E_k = spec_density[mask]

    return k_pos, E_k