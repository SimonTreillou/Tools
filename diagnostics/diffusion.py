from scipy.optimize import curve_fit
import numpy as np

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# compute_Davg: Compute time-averaged diffusivity from time-dependent diffusivity array.
# compute_mu: Compute mean position μ of a tracer distribution.
# compute_sigma: Compute variance σ² of a tracer distribution.
# fit_diffusivity: Fit diffusivity model to observed variance data.
# ratio_saturated: Compute ratio of observed variance to saturated variance.
# surf_sigma_model: Model function for surface variance as a function of period.
# y_to_tp: Convert cross-shore distance to period using mean velocity.
# ---------------------------------------------------------------

def compute_Davg(tpas):
    """
    Compute time-averaged diffusivity from a time-dependent diffusivity array.

    Parameters
    ----------
    tpas : array_like
        Array of diffusivity values over time, shape (ntime, nspace) or (ntime, ...).
    Returns
    -------
    ndarray
        Time-averaged diffusivity along the time axis (mean over axis=0).
    """
    return np.mean(tpas, axis=0)


def compute_mu(x, D, xin=-13, xout=-100):
    """
    Compute the mean position μ of a tracer distribution over a specified cross-shore interval.

    The mean is computed as μ(t) = ∫ x * D(x,t) dx / ∫ D(x,t) dx over x in [xout, xin].

    Parameters
    ----------
    x : 1D array
        Cross-shore coordinates.
    D : 2D array
        Tracer distribution or weighting, shape (ntime, nspace).
    xin : float, optional
        Inner boundary position (default -13).
    xout : float, optional
        Outer boundary position (default -100).

    Returns
    -------
    ndarray
        Array of mean positions for each time step (shape (ntime,)).
    """
    dx = x[1] - x[0]
    ix2 = np.argmin(np.abs(x - xin))   # index for inner boundary
    ix1 = np.argmin(np.abs(x - xout))  # index for outer boundary

    # Weighted integral of x*D over the interval for each time (sum over space)
    tmp1 = np.sum(x[ix1:ix2] * D[:, ix1:ix2] * dx, axis=1)
    # Normalization: integral of D over the interval for each time
    tmp2 = np.sum(D[:, ix1:ix2] * dx, axis=1)

    return tmp1 / tmp2
    
def compute_sigma(x, D, xin=-13, Lx=100):
    """
    Compute time-dependent variance <x^2> of distribution D over the interval [x_out, x_in].

    Parameters
    ----------
    x : 1D array
        Cross-shore coordinates.
    D : 2D array
        Tracer distribution or weighting, shape (ntime, nspace).
    xin : float, optional
        Inner boundary position (default -13).
    Lx : float, optional
        Domain half-width; outer boundary is taken at -Lx (default 100).

    Returns
    -------
    ndarray
        Array of variances for each time step (shape (ntime,)).
    """
    dx = np.mean(np.diff(x))
    ix_out = np.argmin(np.abs(x + Lx))
    ix_in = np.argmin(np.abs(x - xin))

    # Ensure proper slice ordering (start inclusive, end exclusive)
    start = min(ix_out, ix_in)
    end = max(ix_out, ix_in)

    num = np.sum(x[start:end]**2 * D[:, start:end] * dx, axis=1)
    denom = np.sum(D[:, start:end] * dx, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        return num / denom

def fit_diffusivity(tp, sigma):
    """
    Fit σ_surf² = 2*Kxx*Tp + β using least squares.

    Parameters
    ----------
    Tp : array_like
        Period values (independent variable).
    sigma : array_like
        Surface variance σ_surf² (dependent variable).

    Returns
    -------
    Kxx : float
        Fitted diffusivity coefficient.
    beta : float
        Fitted intercept term.
    cov : 2x2 array
        Covariance matrix of the fit.
    """
    popt, pcov = curve_fit(surf_sigma_model, tp, sigma)
    Kxx, beta = popt
    return Kxx, beta, pcov

def ratio_saturated(sigma, x, xin=-13, Lx=100):
    """
    Compute the ratio of observed variance sigma to the saturated variance
    (the spatial mean of x^2 over the interval [x_out=-Lx, x_in=xin]).

    Parameters
    ----------
    sigma : scalar or ndarray
        Observed variance (can be time-dependent array).
    x : 1D ndarray
        Cross-shore coordinates.
    xin : float, optional
        Inner boundary (default -13).
    Lx : float, optional
        Domain half-width; outer boundary used is -Lx (default 100).

    Returns
    -------
    scalar or ndarray
        sigma divided by the saturated variance (same shape as sigma).
    """
    dx = np.mean(np.diff(x))
    ix_out = np.argmin(np.abs(x + Lx))
    ix_in = np.argmin(np.abs(x - xin))

    # ensure proper ordering for slicing
    start = min(ix_out, ix_in)
    end = max(ix_out, ix_in)

    # integral of x^2 over the selected interval and the interval area
    squared_integral = np.sum(x[start:end]**2 * dx)
    area = (end - start) * dx

    saturated = squared_integral / area if area != 0 else np.nan
    return sigma / saturated


def surf_sigma_model(Tp, Kxx, beta):
    """
    Model for surface variance: σ_surf²(Tp) = 2*Kxx*Tp + beta.

    Parameters
    ----------
    Tp : scalar or ndarray
        Period(s).
    Kxx : float
        Diffusivity coefficient.
    beta : float
        Intercept term (saturated variance offset).

    Returns
    -------
    scalar or ndarray
        Modeled surface variance corresponding to Tp.
    """
    return 2.0 * Kxx * Tp + beta

def y_to_tp(x, y, V, Lx):
    """
    Convert cross-shore distance(s) y to period Tp using the mean cross-shore velocity
    over the interval [x=-Lx, x=0]. The zero of x must be at the shore.

    Tp = y / mean(V[x ∈ [-Lx, 0]])

    Parameters
    ----------
    x : 1D array
        Cross-shore coordinates.
    y : scalar or array_like
        Cross-shore distance(s) to convert to period(s).
    V : 1D array
        Velocity field defined on the same grid as x.
    Lx : float
        Positive domain half-width; outer boundary used is -Lx.

    Returns
    -------
    Tp : scalar or ndarray
        Period(s) corresponding to input y (same shape as y).

    Raises
    ------
    ValueError
        If x and V have different shapes.
    ZeroDivisionError
        If the mean velocity over the selected interval is zero.
    """

    x = np.asarray(x)
    V = np.asarray(V)

    if x.shape != V.shape:
        raise ValueError("x and V must have the same shape")

    # find indices corresponding to x = -Lx and x = 0 (or nearest grid points)
    ix_out = np.argmin(np.abs(x + Lx))
    ix_in = np.argmin(np.abs(x - 0.0))

    # ensure proper slice ordering and include the end index
    start = min(ix_out, ix_in)
    end = max(ix_out, ix_in) + 1

    meanV = np.mean(V[start:end])
    if meanV == 0:
        raise ZeroDivisionError("Mean velocity over the selected interval is zero")
    print("Mean longshore current: ", meanV," m/s")
    return np.asarray(y) / meanV