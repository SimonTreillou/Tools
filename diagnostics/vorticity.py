import numpy as np
from netCDF4 import Dataset
from . import useful

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# check_budget_closure: Check vorticity budget closure.
# compute_dUdz: Compute vertical derivative dU/dz for 3D/4D fields.
# compute_rms_vorticity: Compute RMS of vorticity budget terms.
# compute_vorticity: Compute vorticity from velocity fields.
# compute_horizontal_shear_term: Compute RANS horizontal shear-production term.
# compute_Q_components: Compute Q-criterion components from velocity fields.
# enstrophy_spectrum: Compute isotropic wavenumber spectrum of enstrophy.
# load_vorticity_budget: Load vorticity budget terms from file.
# ---------------------------------------------------------------

def check_budget_closure(terms, tol=1e-4):
    """
    Check vorticity budget closure.

    The function computes the sum of all RHS terms (all keys in `terms`
    except 'rate') and compares it to the provided LHS stored under 'rate'.
    It prints a short message and returns True if the maximum absolute
    difference is below `tol`, otherwise False.

    Parameters
    ----------
    terms : dict
        Dictionary of numpy arrays representing budget terms. Must contain key 'rate'.
    tol : float, optional
        Tolerance for closure check (default 1e-4).

    Returns
    -------
    bool
        True if budget closes within tolerance, False otherwise.
    """
    if 'rate' not in terms:
        raise KeyError("terms dictionary must contain key 'rate'")

    # Sum all RHS terms into an array with the same shape as 'rate'
    rhs_sum = np.zeros_like(terms['rate'])
    for name, arr in terms.items():
        if name == 'rate':
            continue
        # ensure shapes match (will raise if incompatible)
        rhs_sum += arr

    # Difference between LHS (rate) and RHS sum
    diff = terms['rate'] - rhs_sum

    # Maximum absolute difference used for the closure check
    max_diff = float(np.max(np.abs(diff)))

    if max_diff < tol:
        print(f"Budget closure check passed: max difference = {max_diff:.2e} < tol = {tol:.2e}")
        return True
    else:
        print(f"Budget closure check failed: max difference = {max_diff:.2e} >= tol = {tol:.2e}")
        return False

def compute_dUdz(u, zr):
    """
    Compute vertical derivative dU/dz for a 3D field u(z,y,x) or 4D field u(t,z,y,x)
    using the provided zr vertical coordinates (same shape as u: (z,y,x) or (t,z,y,x)).

    Returns an array with the same shape as the input u.

    Notes:
    - zr may vary with (y,x) so gradient is computed along the vertical column
      using the actual zr coordinates for uneven spacing.
    - Time dimension is broadcast if one of u or zr has a singleton time dimension.
    """
    u = np.asarray(u)
    zr = np.asarray(zr)

    # validate dimensionality
    if u.ndim not in (3, 4):
        raise ValueError(f"u must have 3 or 4 dims (z,y,x) or (t,z,y,x); got shape {u.shape}")
    if zr.ndim not in (3, 4):
        raise ValueError(f"zr must have 3 or 4 dims (z,y,x) or (t,z,y,x); got shape {zr.shape}")

    # promote to 4D (t,z,y,x) for unified processing
    u4 = u[None, ...] if u.ndim == 3 else u.copy()
    zr4 = zr[None, ...] if zr.ndim == 3 else zr.copy()

    # ensure spatial dimensions (z,y,x) match between u and zr
    if u4.shape[1:] != zr4.shape[1:]:
        raise ValueError(f"spatial dimensions (z,y,x) must match: u has {u4.shape[1:]}, zr has {zr4.shape[1:]}")

    # broadcast time dimension if needed (allow one of them to be length 1)
    if u4.shape[0] != zr4.shape[0]:
        if u4.shape[0] == 1:
            u4 = np.repeat(u4, zr4.shape[0], axis=0)
        elif zr4.shape[0] == 1:
            zr4 = np.repeat(zr4, u4.shape[0], axis=0)
        else:
            raise ValueError(f"time dimension mismatch: u has {u4.shape[0]}, zr has {zr4.shape[0]}")

    t, nz, ny, nx = u4.shape
    # allocate output with same dtype and shape
    dudz4 = np.empty_like(u4)

    # compute vertical gradient column-by-column because zr varies with (y,x)
    for tt in range(t):
        for iy in range(ny):
            for ix in range(nx):
                # np.gradient accepts the coordinate array for uneven spacing along the axis
                # here we pass the zr values for the column (length nz)
                dudz4[tt, :, iy, ix] = np.gradient(u4[tt, :, iy, ix], zr4[tt, :, iy, ix])

    # return in the same dimensionality as the input u (3D if input was 3D, else 4D)
    return dudz4[0] if u.ndim == 3 else dudz4

def compute_rms_vorticity(terms, axis=(1, 2, 3)):
    """
    Compute RMS (root-mean-square) of vorticity budget terms.

    Computes RMS for each RHS term (all keys except 'rate') and for the 'rate'
    term over the specified axes.

    Parameters
    ----------
    terms : dict
        Dictionary of numpy arrays representing budget terms. Must contain key 'rate'.
    axis : tuple of ints, optional
        Axes over which to compute the mean-square (default (1,2,3)).

    Returns
    -------
    rms_rhs : dict
        Dictionary mapping RHS term names to their RMS arrays.
    rms_rate : ndarray
        RMS array for the 'rate' term.
    """
    if 'rate' not in terms:
        raise KeyError("terms dictionary must contain key 'rate'")

    # Exclude 'rate' from RHS terms
    RHS_terms = {k: v for k, v in terms.items() if k != 'rate'}

    # Compute RMS = sqrt(mean(square)) for each RHS term
    rms_rhs = {}
    for name, arr in RHS_terms.items():
        rms_rhs[name] = np.sqrt(np.mean(arr**2, axis=axis))

    # RMS for the rate (LHS)
    rms_rate = np.sqrt(np.mean(terms['rate']**2, axis=axis))

    return rms_rhs, rms_rate

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


def compute_Q_components(ut, vt, wt, dx, dy, dz3):
    """
    Compute Q-criterion components from velocity fields.

    The Q-criterion is used to identify vortex cores. Q is the second invariant
    of the velocity gradient tensor. Positive Q indicates vortex-dominated regions.

    Parameters
    ----------
    ut, vt, wt : 3D array
        Velocity components with shape (N, M, L) representing (z, y, x).
    dx, dy : float
        Grid spacing in x and y directions.
    dz3 : float
        Grid spacing in z direction.

    Returns
    -------
    Q : 3D array
        Q-criterion (second invariant of velocity gradient tensor).
    Qx, Qy, Qz : 3D array
        Individual components of Q in x, y, z directions.
    """
    # Compute all spatial derivatives of velocity components
    duy, duz, dux = useful.gradients_3d(ut)
    dvy, dvz, dvx = useful.gradients_3d(vt)
    dwy, dwz, dwx = useful.gradients_3d(wt)

    # Scale spatial derivatives by grid spacing to get actual derivatives
    dux = dux / dx
    dvy = dvy / dy
    dwz = dwz / dz3
    duy = duy / dy
    dvx = dvx / dx
    duz = duz / dz3
    dvz = dvz / dz3
    dwy = dwy / dy
    dwx = dwx / dx

    # Q-criterion: Q = -0.5 * (S_ij*S_ij + Omega_ij*Omega_ij)
    # where S is strain rate and Omega is rotation rate tensors
    Q = -0.5 * (
        (dux ** 2) + (dvy ** 2) + (dwz ** 2)
        + 2 * duy * dvx
        + 2 * duz * dwx
        + 2 * dvz * dwy
    )
    
    # Q-criterion components (directional contributions)
    Qx = -duz * dwx - 0.5 * (dux ** 2)
    Qy = -dvz * dwy - 0.5 * (dvy ** 2)
    Qz = -duy * dvx - 0.5 * (dwz ** 2)

    return Q, Qx, Qy, Qz

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

def load_vorticity_budget(repo_path, region=None):
    """
    Load vorticity budget terms from file.

    Parameters
    ----------
    repo_path : str
        Path or identifier passed to useful.find_file.
    region : None or tuple (iy1, iy2, ix1, ix2), optional
        If None (default) load full domain.
        If tuple provided, extract the horizontal subset [:, :, iy1:iy2, ix1:ix2]
        for all variables (assumes variables have dims (time, z, y, x)).

    Returns
    -------
    terms : dict
        Dictionary of numpy arrays containing budget terms (possibly subset).
    """
    # Build index: either full domain or [:, :, iy1:iy2, ix1:ix2]
    if region is None:
        idx = (slice(None), slice(None), slice(None), slice(None))
    else:
        iy1, iy2, ix1, ix2 = region
        idx = (slice(None), slice(None), slice(iy1, iy2), slice(ix1, ix2))

    # closing: rate = xadv+xstr+yadv+ystr+vadv+tilting+Vmix+Hmix+prsgrd+fast+nudg
    fname = useful.find_file(repo_path, suffix="_diags_vrt3d_avg.nc")
    ds = Dataset(fname)
    # extract the same indexed subset for each variable
    terms = {
        'xstr' : ds.variables['vrt_xstretch'][idx],
        'xadv' : ds.variables['vrt_xadv'][idx] - ds.variables['vrt_xstretch'][idx],
        'ystr' : ds.variables['vrt_ystretch'][idx],
        'yadv' : ds.variables['vrt_yadv'][idx] - ds.variables['vrt_ystretch'][idx],
        'vtil' : ds.variables['vrt_tilting'][idx],
        'zadv' : ds.variables['vrt_vadv'][idx] - ds.variables['vrt_tilting'][idx],
        'vmix' : ds.variables['vrt_vmix'][idx],
        'fast' : ds.variables['vrt_fast'][idx],
        'hdif' : ds.variables['vrt_hdiff'][idx],
        'hmix' : ds.variables['vrt_hmix'][idx],
        'prsg' : ds.variables['vrt_prsgrd'][idx],
        'nudg' : ds.variables['vrt_nudg'][idx],
        'rate' : ds.variables['vrt_rate'][idx],
    }
    ds.close()
    return terms
