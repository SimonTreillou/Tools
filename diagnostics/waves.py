import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from . import spectrum 

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# asymmetry: Compute wave asymmetry using the third standardized moment of the time derivative.
# backrefract_angle: Compute back-refracted wave angle due to depth change.
# backrefract_dirspread: Compute back-refracted directional spreading.
# backrefract_Snn: Compute back-refracted spectral density Snn0.
# compute_a1_a2_b1_b2: Compute directional Fourier components a1, a2, b1, b2.
# compute_Hs: Compute significant wave height Hs from diagnostic eddy output files.
# dispersion_relation: Compute wavenumber k for given angular frequency and water depth.
# find_theta_tseries: Estimate mean wave direction and spread from time series.
# group_velocity: Compute group velocity of water waves.
# hyd: Compute water depth from pressure sensor data.
# jonswap_spectrum: Generate a JONSWAP spectrum Snn(f).
# phase_speed: Compute phase speed of water waves.
# skewness: Compute skewness of a time series.
# test_oblique_waves: Print compatible domain lengths and angles for oblique waves.
# ---------------------------------------------------------------

def asymmetry(zeta, dt=1.0):
    """
    Compute the wave asymmetry of a time series zeta(t),
    defined using the third standardized moment of the time derivative d(zeta)/dt.

    Asymmetry quantifies the degree to which the shape of the wave profile is skewed
    in time (e.g., sharper crests or troughs).

    Parameters:
    zeta : array-like
        Time series of sea surface elevation (or similar signal)
    dt : float, optional
        Time step between samples (default=1.0)

    Returns:
    float
        Asymmetry value (dimensionless). Returns 0 if the denominator is zero.
    """
    zeta = np.asarray(zeta)
    dzdt = np.gradient(zeta, dt)
    numerator = np.mean(dzdt**3)
    denominator = np.mean(dzdt**2)**1.5
    if denominator == 0:
        return 0  # or np.nan
    return numerator / denominator

def backrefract_angle(f, thetaF, hF, hWM, fmin=0.06, fmax=0.18):
        """
        Compute the back-refracted wave angle due to a change in water depth.

        Parameters:
        - f : array-like
            Frequency array (Hz)
        - thetaF : float or array-like
            Incident wave angle in degrees (deep water or initial depth)
        - hF : float
            Initial water depth (m)
        - hWM : float
            New water depth after refraction (m)
        - fmin : float
            Minimum frequency for calculation (Hz)
        - fmax : float
            Maximum frequency for calculation (Hz)

        Returns:
        - thetaWM : array-like
            Back-refracted wave angle in degrees (shallow water or new depth)
        """
        imi = np.argmin(np.abs(f - fmin))
        ima = np.argmin(np.abs(f - fmax))

        cpF = phase_speed(2 * np.pi * f, hF)
        cpWM = phase_speed(2 * np.pi * f, hWM)

        thetaWM = np.arcsin((np.sin(thetaF * np.pi / 180) * cpWM / cpF)) * 180 / np.pi
        return thetaWM

def backrefract_dirspread(f, Snn1, h1, h0, theta1, theta0, sigma1, fmin=0.06, fmax=0.18):
    """
    Compute the back-refracted directional spreading (sigma0) due to a change in water depth.

    Parameters:
    - f : array-like
        Frequency array (Hz)
    - Snn1 : array-like
        Initial spectral density (at depth h1) [not used in calculation]
    - h1 : float
        Initial water depth (m)
    - h0 : float
        New water depth after refraction (m)
    - theta1 : array-like
        Incident wave angle in degrees (at depth h1)
    - theta0 : array-like
        Refracted wave angle in degrees (at depth h0)
    - sigma1 : array-like
        Initial directional spreading (at depth h1)
    - fmin : float
        Minimum frequency for calculation (Hz)
    - fmax : float
        Maximum frequency for calculation (Hz)

    Returns:
    - sigma0 : array-like
        Back-refracted directional spreading (at depth h0)
    """
    imi = np.argmin(np.abs(f - fmin))
    ima = np.argmin(np.abs(f - fmax))
    
    cp1 = phase_speed(2 * np.pi * f, h1)
    cp0 = phase_speed(2 * np.pi * f, h0)
    
    sigma0 = cp0 / cp1 * np.cos(np.deg2rad(theta1)) / np.cos(np.deg2rad(theta0)) * sigma1
    
    return sigma0

def backrefract_Snn(f, Snn1, h1, h0, theta1, theta0, fmin=0.06, fmax=0.18):
    """
    Compute the back-refracted spectral density Snn0 due to a change in water depth.

    Parameters:
    - f : array-like
        Frequency array (Hz)
    - Snn1 : array-like
        Initial spectral density (at depth h1)
    - h1 : float
        Initial water depth (m)
    - h0 : float
        New water depth after refraction (m)
    - theta1 : array-like
        Incident wave angle in degrees (at depth h1)
    - theta0 : array-like
        Refracted wave angle in degrees (at depth h0)
    - fmin : float
        Minimum frequency for calculation (Hz)
    - fmax : float
        Maximum frequency for calculation (Hz)

    Returns:
    - f_sel : array-like
        Selected frequency array (Hz) within [fmin, fmax]
    - Snn0 : array-like
        Back-refracted spectral density (at depth h0)
    """
    imi = np.argmin(np.abs(f - fmin))
    ima = np.argmin(np.abs(f - fmax))

    cg1 = group_velocity(2 * np.pi * f[imi:ima], h1)
    cg0 = group_velocity(2 * np.pi * f[imi:ima], h0)
    Snn0 = (cg0 * np.cos(np.deg2rad(theta0[imi:ima]))) / (cg1 * np.cos(np.deg2rad(theta1[imi:ima]))) * Snn1[imi:ima]
    return f[imi:ima], Snn0

def compute_Hs(fname, mode="diag_eddy", longshore_average=False):
    """
    Compute significant wave height Hs from diagnostic eddy output files.

    Parameters:
    - fname : str
        Base filename (path prefix) for the NetCDF files (expects fname+'avg.nc' and fname+'diags_eddy_avg.nc')
    - mode : str, optional
        Processing mode, default "diag_eddy"
    - longshore_average : bool, optional
        If True, return the longshore-averaged Hs

    Returns:
    - Hs : ndarray
        Significant wave height (m)
    """

    if mode == "diag_eddy":
        # Access averaged quantities
        nc = Dataset(fname + 'avg.nc')
        h = nc.variables['h'][:, :]
        zeta = nc.variables['zeta'][:, :, :]
        Dcrit = nc.variables['Dcrit'][0]
        nc.close()

        # Time-average of surface elevation
        z0 = np.mean(zeta, 0)
        # Adjust where depth is below critical (add slope contribution)
        z0[h < Dcrit] = z0[h < Dcrit] - h[h < Dcrit]
        z0 = z0**2  # <eta>^2

        # Access time-averaged squared elevation <eta^2>
        nc = Dataset(fname + 'diags_eddy_avg.nc')
        zz = np.mean(nc.variables['zz'][:, :, :], 0)
        nc.close()

        # Svendsen (2005) p.126: Hs = 4.0083 * sqrt(<eta^2> - <eta>^2)
        Hs = 4.0083 * np.sqrt(zz - z0)
        if longshore_average:
            Hs = np.nanmean(Hs, 0)  # longshore-average
        return Hs

    # If other modes are required later, handle here
    return None

def compute_a1_a2_b1_b2(Ezz, Exx, Eyy, Cxy, Qxz, Qyz):
    """
    Compute directional Fourier components a1, a2, b1, b2.

    Parameters:
    - Ezz: array or value of surface elevation variance density spectrum
    - Exx: array or value of x-displacement variance density spectrum
    - Eyy: array or value of y-displacement variance density spectrum
    - Cxy: array or value of x–y co-spectrum
    - Qxz: array or value of x–z quad-spectrum
    - Qyz: array or value of y–z quad-spectrum

    Returns:
    - a1, a2, b1, b2: arrays or values of directional Fourier components

    Notes:
    - a1, b1: First-order Fourier coefficients (mean wave direction)
    - a2, b2: Second-order Fourier coefficients (directional spreading)
    - The denominators ensure normalization by spectral energy.
    """
    denom_a1b1 = np.sqrt(Ezz * (Exx + Eyy))
    denom_a2b2 = Exx + Eyy

    a1 = Qxz / denom_a1b1
    b1 = Qyz / denom_a1b1
    a2 = (Exx - Eyy) / denom_a2b2
    b2 = 2 * Cxy / denom_a2b2

    return a1, a2, b1, b2

def dispersion_relation(omega, h=10):
    """
    Compute the wavenumber k for a given angular frequency omega and water depth h
    using the linear wave dispersion relation.

    Parameters:
    - omega : scalar or array-like
        Angular frequency (rad/s)
    - h : float
        Water depth (m), default is 10

    Returns:
    - k : scalar or array-like
        Wavenumber (1/m)
    """
    g = 9.81  # gravity [m/s^2]
    const = (omega ** 2) * h / g

    # If const is a scalar, convert it to a 1-element array to handle indexing properly
    if np.isscalar(const):
        const = np.array([const])
        scalar_input = True
    else:
        scalar_input = False

    # Initialize wavenumber (kh)
    kh = np.full_like(const, np.nan)

    # Handle special cases
    kh[const == 0] = 0  # Zero const returns zero wavenumber
    positive_const = const > 0

    # Initial guess for Newton-Raphson iteration
    kh[positive_const] = np.sqrt(const[positive_const])

    # Newton-Raphson iteration to solve kh * tanh(kh) = const
    tolerance = 1e-6
    max_iter = 100
    for _ in range(max_iter):
        f = kh[positive_const] * np.tanh(kh[positive_const]) - const[positive_const]
        fprime = kh[positive_const] / np.cosh(kh[positive_const]) ** 2 + np.tanh(kh[positive_const])
        kh[positive_const] -= f / fprime

        # Check for convergence
        if np.max(np.abs(f)) < tolerance:
            break

    # Compute final wavenumber
    k = kh / h

    # Return scalar if input was scalar
    if scalar_input:
        return k[0]
    return k

def find_theta_tseries(zeta, u, v, fs, fmin=0.05, fmax=0.2, N=256):
    """
    Estimate mean wave direction and directional spread from time series.

    Parameters:
    - zeta : array-like
        Surface elevation time series
    - u : array-like
        Horizontal velocity component (x-direction)
    - v : array-like
        Horizontal velocity component (y-direction)
    - fs : float
        Sampling frequency (Hz)
    - fmin : float, optional
        Minimum frequency for integration (Hz), default 0.05
    - fmax : float, optional
        Maximum frequency for integration (Hz), default 0.2
    - N : int, optional
        Number of FFT points (segment length), default 256

    Returns:
    - f : array
        Frequency array (Hz)
    - theta2 : array
        Mean wave direction (degrees) as function of frequency
    - sigma2 : array
        Directional spread (degrees) as function of frequency
    - theta2b : float
        Mean wave direction (degrees), frequency-integrated
    - sigma2b : float
        Directional spread (degrees), frequency-integrated
    """
    # Compute auto- and cross-spectra
    f, Ezz, Qzz = spectrum.compute_cospectrum_quadspectrum(zeta, zeta, fs, N)
    f, Exx, Qzz = spectrum.compute_cospectrum_quadspectrum(u, u, fs, N)
    f, Eyy, Qzz = spectrum.compute_cospectrum_quadspectrum(v, v, fs, N)
    f, Cxy, Qxy = spectrum.compute_cospectrum_quadspectrum(u, v, fs, N)
    f, Cxz, Qxz = spectrum.compute_cospectrum_quadspectrum(u, zeta, fs, N)
    f, Cyz, Qyz = spectrum.compute_cospectrum_quadspectrum(v, zeta, fs, N)

    # Compute directional Fourier coefficients
    a1, a2, b1, b2 = compute_a1_a2_b1_b2(Ezz, Exx, Eyy, Cxy, Qxz, Qyz)

    # Mean direction and spread as function of frequency
    theta2 = 180 / np.pi * np.arctan(b2 / a2) * 0.5
    sigma2 = 180 / np.pi * 0.5 * np.sqrt(1 - a2 * np.cos(2 * theta2) - b2 * np.sin(2 * theta2))

    # Frequency range indices
    ifmin = np.argmin(np.abs(f - fmin))
    ifmax = np.argmin(np.abs(f - fmax))
    df = f[1] - f[0]

    # Frequency-integrated coefficients (weighted by Ezz)
    a1b = np.trapz(a1[ifmin:ifmax] * Ezz[ifmin:ifmax], dx=df) / np.trapz(Ezz[ifmin:ifmax], dx=df)
    b1b = np.trapz(b1[ifmin:ifmax] * Ezz[ifmin:ifmax], dx=df) / np.trapz(Ezz[ifmin:ifmax], dx=df)
    a2b = np.trapz(a2[ifmin:ifmax] * Ezz[ifmin:ifmax], dx=df) / np.trapz(Ezz[ifmin:ifmax], dx=df)
    b2b = np.trapz(b2[ifmin:ifmax] * Ezz[ifmin:ifmax], dx=df) / np.trapz(Ezz[ifmin:ifmax], dx=df)

    # Frequency-integrated mean direction and spread
    theta2b = 180 / np.pi * np.arctan(b2b / a2b) * 0.5
    sigma2b = 180 / np.pi * np.sqrt(0.5 * (1 - a2b * np.cos(2 * theta2b * np.pi / 180) - b2b * np.sin(2 * theta2b * np.pi / 180)))

    return f, theta2, sigma2, theta2b, sigma2b

def group_velocity(omega, h):
    """
    Compute the group velocity of water waves for a given angular frequency and depth.

    Parameters:
    - omega : scalar or array-like
        Angular frequency (rad/s)
    - h : float
        Water depth (m)

    Returns:
    - cg : scalar or array-like
        Group velocity (m/s)
    """
    k = dispersion_relation(omega, h)
    cp = phase_speed(omega, h)
    cg = 0.5 * cp * (1 + (2 * k * h) / (np.sinh(2 * k * h)))
    return cg

def hyd(p, dm, h0, rho=1000, g=9.81, patm=1):
    """
    Compute water depth from pressure sensor data.

    Parameters:
    - p : float or array-like
        Measured pressure (kPa or same units as patm)
    - dm : float
        Sensor vertical offset above seabed (m)
    - h0 : float
        Reference height (m)
    - rho : float, optional
        Water density (kg/m^3), default 1000
    - g : float, optional
        Gravitational acceleration (m/s^2), default 9.81
    - patm : float, optional
        Atmospheric pressure (kPa or same units as p), default 1

    Returns:
    - h : float or array-like
        Computed water depth (m)
    """
    return (p - patm) / (rho * g) + dm - h0

def jonswap_spectrum(Tp, gamma, Hs, fmin=0.02, fmax=1.0, df=0.001):
    """
    Generate a JONSWAP spectrum Snn(f).
    
    Parameters:
    - Tp : Peak period (s)
    - gamma : Peak enhancement factor
    - Hs : Significant wave height (m)
    - fmin : Minimum frequency (Hz)
    - fmax : Maximum frequency (Hz)
    - df : Frequency resolution (Hz)
    
    Returns:
    - f : Frequency array (Hz)
    - Snn : Spectral density array (m^2/Hz)
    """
    # Constants
    g = 9.81  # gravitational acceleration (m/s²)
    fp = 1.0 / Tp  # peak frequency (Hz)
    alpha = 0.076 * (g**2 / (2 * np.pi)**4) * fp**(-4) * Hs**2  # PM spectrum alpha

    f = np.arange(fmin, fmax + df, df)  # frequency array
    sigma = np.where(f <= fp, 0.07, 0.09)  # spreading parameter

    r = np.exp(- ( (f - fp)**2 ) / (2 * sigma**2 * fp**2))  # peak enhancement shape

    # Pierson-Moskowitz base spectrum
    S_PM = alpha * g**2 * f**(-5) * np.exp(-1.25 * (fp / f)**4)

    # JONSWAP spectrum
    Snn = S_PM * gamma**r

    return f, Snn

def phase_speed(omega, h):
    """
    Compute the phase speed of water waves for a given angular frequency and depth.

    Parameters:
    - omega : scalar or array-like
        Angular frequency (rad/s)
    - h : float
        Water depth (m)

    Returns:
    - cp : scalar or array-like
        Phase speed (m/s)
    """
    g = 9.81  # gravitational acceleration (m/s^2)
    k = dispersion_relation(omega, h)
    cp = np.sqrt(g / k * np.tanh(k * h))
    return cp

def skewness(zeta):
    """
    Compute the skewness of a time series zeta(t).

    Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable.
    A skewness value > 0 indicates a distribution with an asymmetric tail extending toward more positive values.

    Parameters:
    zeta (array-like): Time series data.

    Returns:
    float: Skewness of the series. Returns 0 if the standard deviation is zero.
    """
    zeta = np.asarray(zeta)
    mean = np.mean(zeta)
    std = np.std(zeta)
    if std == 0:
        return 0  # or np.nan, depending on how you want to treat constant signals
    skew = np.mean((zeta - mean)**3) / std**3
    return skew

def test_oblique_waves(h, Tp, theta):
    """
    Prints compatible domain lengths and angles for oblique waves.

    Parameters:
    - h : float
        Water depth (m)
    - Tp : float
        Peak period (s)
    - theta : float
        Incident wave angle in degrees

    Prints:
    - Compatible Y-axis lengths (multiples of Ly)
    - Compatible incidence angles (multiples of theta)
    """
    g = 9.81
    omega = 2 * np.pi / Tp
    wd = theta * np.pi / 180

    k = dispersion_relation(omega, h)
    L = 2 * np.pi / k
    Ly = L / np.abs(np.sin(wd))

    a1 = wd * 180 / np.pi
    a2 = np.arcsin(np.sin(wd) / 2) * 180 / np.pi
    a3 = np.arcsin(np.sin(wd) / 3) * 180 / np.pi

    print("Compatible Y-axis length:")
    print("  " + str(np.round(Ly)) + "  " + str(np.round(Ly * 2)) + "  " + str(np.round(Ly * 3)))
    print("Compatible incidence angles:")
    print("  " + str(np.round(a1)) + "  " + str(np.round(a2)) + "  " + str(np.round(a3)))

# %%
