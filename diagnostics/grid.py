import numpy as np

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# csf: Compute stretching curves for vertical coordinates.
# lonlat_to_xy: Convert lon,lat to cartesian grid x,y.
# rho2uvp: Convert variable on rho grid to u, v, and psi grids.
# u2rho: Convert variable on u grid to rho grid.
# v2rho: Convert variable on v grid to rho grid.
# xy_to_lonlat: Convert cartesian grid x,y to lon,lat.
# zlevs: Calculate vertical grid levels for ROMS model.
# ---------------------------------------------------------------

def csf(sc, theta_s, theta_b):
    """
    Compute the stretching curves used in ROMS vertical coordinate transformation.

    Parameters
    ----------
    sc : array_like
        S-coordinate values.
    theta_s : float
        Surface stretching parameter.
    theta_b : float
        Bottom stretching parameter.

    Returns
    -------
    h : array_like
        Stretching curves.
    """
    if theta_s > 0:
        csrf = (1 - np.cosh(sc * theta_s)) / (np.cosh(theta_s) - 1)
    else:
        csrf = -sc ** 2
    if theta_b > 0:
        h = (np.exp(theta_b * csrf) - 1) / (1 - np.exp(-theta_b))
    else:
        h = csrf
    return h

def lonlat_to_xy(lon,lat):
    """
        Convert lon,lat coordinates to cartesian x,y grid.
        
        Parameters
        ----------
        lon : float or ndarray
            Longitude coordinate(s).
        lat : float or ndarray
            Latitude coordinate(s).
         
        Returns
        -------
        x : float or ndarray
            X coordinate(s).
        y : float or ndarray
            Y coordinate(s).   
    """
    transformer = Transformer.from_crs("epsg:4326","epsg:32613", always_xy=True)
    x,y = transformer.transform(lon,lat)
    return x,y

def rho2uvp(rfield):
    """
    Convert a variable on the rho grid to u, v, and psi grids.

    Parameters
    ----------
    rfield : ndarray
        2D array on the rho grid (Mp x Lp).

    Returns
    -------
    ufield : ndarray
        Variable on the u grid (Mp x L).
    vfield : ndarray
        Variable on the v grid (M x Lp).
    pfield : ndarray
        Variable on the psi grid (M x L).
    """
    Mp, Lp = np.shape(rfield)  # rho grid dimensions
    M = Mp - 1
    L = Lp - 1

    vfield = 0.5 * (rfield[0:M, :] + rfield[1:Mp, :])      # v grid
    ufield = 0.5 * (rfield[:, 0:L] + rfield[:, 1:Lp])      # u grid
    pfield = 0.5 * (ufield[0:M, :] + ufield[1:Mp, :])      # psi grid

    return ufield, vfield, pfield


def u2rho(var_u):
    """
    Convert a variable on the u grid to the rho grid.

    Parameters
    ----------
    var_u : ndarray
        Variable on the u grid. Can be 2D, 3D, or 4D.

    Returns
    -------
    var_rho : ndarray
        Variable interpolated to the rho grid.
    """
    if len(var_u.shape) == 4:
        [T, N, Mp, L] = var_u.shape
        Lp = L + 1
        Lm = L - 1
        var_rho = np.zeros((T, N, Mp, Lp))
        var_rho[:, :, :, 1:L-1] = 0.5 * (var_u[:, :, :, 0:Lm-1] + var_u[:, :, :, 1:L-1])
        var_rho[:, :, :, 0] = var_rho[:, :, :, 1]
        var_rho[:, :, :, L] = var_rho[:, :, :, L-1]
    elif len(var_u.shape) == 3:
        [N, Mp, L] = var_u.shape
        Lp = L + 1
        Lm = L - 1
        var_rho = np.zeros((N, Mp, Lp))
        var_rho[:, :, 1:L-1] = 0.5 * (var_u[:, :, 0:Lm-1] + var_u[:, :, 1:L-1])
        var_rho[:, :, 0] = var_rho[:, :, 1]
        var_rho[:, :, L] = var_rho[:, :, L-1]
    elif len(var_u.shape) == 2:
        [Mp, L] = var_u.shape
        Lp = L + 1
        Lm = L - 1
        var_rho = np.zeros((Mp, Lp))
        var_rho[:, 1:L-1] = 0.5 * (var_u[:, 0:Lm-1] + var_u[:, 1:L-1])
        var_rho[:, 0] = var_rho[:, 1]
        var_rho[:, L] = var_rho[:, L-1]
    return var_rho


def v2rho(var_v):
    """
    Convert a variable on the v grid to the rho grid.

    Parameters
    ----------
    var_v : ndarray
        Variable on the v grid. Can be 2D, 3D, or 4D.

    Returns
    -------
    var_rho : ndarray
        Variable interpolated to the rho grid.
    """
    if len(var_v.shape) == 4:
        [T, N, M, Lp] = var_v.shape
        Mp = M + 1
        Mm = M - 1
        var_rho = np.zeros((T, N, Mp, Lp))
        var_rho[:, :, 1:M-1, :] = 0.5 * (var_v[:, :, 0:Mm-1, :] + var_v[:, :, 1:M-1, :])
        var_rho[:, :, 0, :] = var_rho[:, :, 1, :]
        var_rho[:, :, M, :] = var_rho[:, :, M-1, :]
    elif len(var_v.shape) == 3:
        [N, M, Lp] = var_v.shape
        Mp = M + 1
        Mm = M - 1
        var_rho = np.zeros((N, Mp, Lp))
        var_rho[:, 1:M-1, :] = 0.5 * (var_v[:, 0:Mm-1, :] + var_v[:, 1:M-1, :])
        var_rho[:, 0, :] = var_rho[:, 1, :]
        var_rho[:, M, :] = var_rho[:, M-1, :]
    elif len(var_v.shape) == 2:
        [M, Lp] = var_v.shape
        Mp = M + 1
        Mm = M - 1
        var_rho = np.zeros((Mp, Lp))
        var_rho[1:M-1, :] = 0.5 * (var_v[0:Mm-1, :] + var_v[1:M-1, :])
        var_rho[0, :] = var_rho[1, :]
        var_rho[M, :] = var_rho[M-1, :]
    return var_rho

def xy_to_lonlat(x,y):
    """
        Convert x,y coordinates to longitude and latitude.
        
        Parameters
        ----------
        x : float or ndarray
            X coordinate(s).
        y : float or ndarray
            Y coordinate(s).    
        Returns
        -------
        lon : float or ndarray
            Longitude coordinate(s).
        lat : float or ndarray
            Latitude coordinate(s).
    """
    transformer = Transformer.from_crs("epsg:32613", "epsg:4326", always_xy=True)
    lon,lat = transformer.transform(x, y)
    return lon,lat

def zlevs(h, zeta, theta_s, theta_b, hc, N, ttype, vtransform):
        """
        Calculate vertical grid levels for ROMS model.

        Parameters
        ----------
        h : ndarray
            Bathymetry (2D array, shape [M, L]).
        zeta : ndarray
            Surface elevation (2D array, shape [M, L]).
        theta_s : float
            Surface stretching parameter.
        theta_b : float
            Bottom stretching parameter.
        hc : float
            Critical depth parameter.
        N : int
            Number of vertical levels.
        ttype : str
            Type of vertical grid ('w' for w-points, otherwise rho-points).
        vtransform : int
            Vertical transformation equation (1 or 2).

        Returns
        -------
        z : ndarray
            Vertical grid levels (3D array, shape [N, M, L]).
        """
        [M, L] = np.shape(h)
        sc_r = np.zeros([N, 1])
        Cs_r = np.zeros([N, 1])
        sc_w = np.zeros([N + 1, 1])
        Cs_w = np.zeros([N + 1, 1])

        if vtransform == 2:
            ds = 1. / N
            if ttype == 'w':
                sc_w[0] = -1.0
                sc_w[-1] = 0
                Cs_w[0] = -1.0
                Cs_w[-1] = 0

                sc_w[1:-1] = ds * (np.arange(1, N) - N)
                Cs_w = csf(sc_w, theta_s, theta_b)
                N = N + 1
            else:
                sc = ds * (np.arange(1, N + 1) - N - 0.5)
                Cs_r = csf(sc, theta_s, theta_b)
                sc_r = sc

        else:
            cff1 = 1. / np.sinh(theta_s)
            cff2 = 0.5 / np.tanh(0.5 * theta_s)
            if ttype == 'w':
                sc = (np.arange(0, N + 1) - N) / N
                N = N + 1
            else:
                sc = (np.arange(1, N + 1) - N - 0.5) / N
            Cs = (1. - theta_b) * cff1 * np.sinh(theta_s * sc) + theta_b * (cff2 * np.tanh(theta_s * (sc + 0.5)) - 0.5)

        Dcrit = 0.01
        zeta[zeta < (Dcrit - h)] = Dcrit - h[zeta < (Dcrit - h)]

        z = np.zeros([N, M, L])

        if vtransform == 2:
            if ttype == 'w':
                cff1 = Cs_w
                cff2 = sc_w + 1
                sc = sc_w
            else:
                cff1 = Cs_r
                cff2 = sc_r + 1
                sc = sc_r
            h2 = (np.abs(h) + hc)
            cff = hc * sc
            h2inv = 1 / h2
            for k in range(N):
                z0 = cff[k] + cff1[k] * np.abs(h)
                z[k, :, :] = z0 * h / h2 + zeta * (1. + z0 * h2inv)
        else:
            h[h == 0] = 1.e-2
            hinv = 1. / h
            cff1 = Cs
            cff2 = sc + 1
            cff = hc * (sc - Cs)
            cff2 = sc + 1
            for k in range(N):
                z0 = cff[k] + cff1[k] * h
                z[k, :, :] = z0 + zeta * (1. + z0 * hinv)

        return z
    