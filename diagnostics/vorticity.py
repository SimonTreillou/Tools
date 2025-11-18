import numpy as np
from netCDF4 import Dataset
from . import useful

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# compute_vorticity: Compute vorticity from velocity fields.
# compute_horizontal_shear_term: Compute RANS horizontal shear-production term.
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

def compute_horizontal_shear_term(config,
                       vel='u',          # mean velocity component: 'u','v','w'
                       flux='uv',        # Reynolds stress: 'uu','uv','uw','vv','vw','ww'
                       deriv='x',        # derivative direction: 'x','y','z'
                       dx=1.0, dy=1.0,
                       lvl=-1):
    """
    Computes a single RANS shear-production term:
        P_ij = - <u'_i u'_j> * dUi/dxj

    Parameters
    ----------
    vel : str
        Mean velocity to differentiate ('u','v','w').
    flux : str
        Reynolds flux ('uu','uv','uw','vv','vw','ww').
    deriv : str
        Derivative direction: 'x','y','z'.
    dx,dy : float
        Grid spacing.
    lvl : int
        Vertical level index.

    Returns
    -------
    shear_prod : 2D numpy array
        The shear-production term at the chosen level.
    """

    # ----------------------------------------------------------
    # Load mean velocity field
    # ----------------------------------------------------------
    fname = useful.find_file(config, suffix="_avg.nc")
    nc = Dataset(fname)
    Umean = np.mean(nc.variables[vel][..., lvl, :, :],axis=(0,1))  # assume dimensions: (time,z,y,x)
    nc.close()

    # ----------------------------------------------------------
    # Compute derivative of mean velocity
    # ----------------------------------------------------------
    if deriv == 'x':
        dU = (Umean[:, 1:] - Umean[:, :-1]) / dx
    elif deriv == 'y':
        dU = (Umean[1:, :] - Umean[:-1, :]) / dy
    elif deriv == 'z':
        raise NotImplementedError("Vertical derivative needs 3D Umean input.")
    else:
        raise ValueError("deriv must be 'x', 'y', or 'z'")

    # ----------------------------------------------------------
    # Load Reynolds stress (flux)
    # ----------------------------------------------------------
    flux_file = useful.find_file(config, suffix="_diags_eddy_avg.nc")
    nc = Dataset(flux_file)
    F = np.mean(nc.variables[flux][..., lvl, :, :],axis=(0,1))  # assume dimensions: (time,z,y,x)
    nc.close()

    # Average flux to grid of derivative (staggering)
    if deriv == 'x':
        F = 0.5 * (F[:, 1:] + F[:, :-1])
    elif deriv == 'y':
        F = 0.5 * (F[1:, :] + F[:-1, :])

    # ----------------------------------------------------------
    # Shear production term
    # ----------------------------------------------------------
    shear_prod = -F * dU

    return shear_prod

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
