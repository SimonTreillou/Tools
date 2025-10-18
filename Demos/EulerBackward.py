import numpy as np
import matplotlib.pyplot as plt

#%% --- Parameters ---
L = 3.0              # domain length
nx = 1000             # number of grid points
dx = L / nx          # spatial step
x = np.linspace(0, L, nx, endpoint=False)

# Physical parameters
nu = 0.00           # diffusion coefficient
c = 1.0              # advection speed

# Initial condition: sine wave
u0 = np.sin(2 * np.pi * x)
timesteps = [0.001, 0.01, 0.05,0.1]   # Different dt to compare
T = 1.0

# --- Helper: build matrices for implicit Euler ---
def build_matrix(dt):
    """Builds implicit matrix A for du/dt + c du/dx = nu d2u/dx2"""
    # central differences with periodic BCs
    diag = np.ones(nx)
    off = np.ones(nx - 1)

    # 1D Laplacian periodic
    Lap = -2*np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    Lap[0, -1] = 1
    Lap[-1, 0] = 1
    Lap /= dx**2

    # 1D first derivative periodic (central)
    D = np.zeros((nx, nx))
    for i in range(nx):
        D[i, (i+1)%nx] = 0.5
        D[i, (i-1)%nx] = -0.5
    D /= dx

    # Implicit Euler: (I - dt*( -nu Lap + -c D)) u^{n+1} = u^n
    A = np.eye(nx) - dt*(-nu*Lap - c*D)
    return A

#%% --- Time integration for each dt ---
plt.figure(figsize=(8,3),dpi=300)
plt.plot(x, u0, 'k-', label='Initial')

for dt in timesteps:
    A = build_matrix(dt)
    u = u0.copy()
    nsteps = int(T/dt)
    for n in range(nsteps):
        u = np.linalg.solve(A, u)
    plt.plot(x, u, label=f'dt={dt}')

plt.xlabel('x')
plt.ylabel('u')
plt.title('Implicit Euler method',loc='left')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# %% SAME WITH WENO

# --- Parameters ---
nx = 200
L = 1.0
x = np.linspace(0, L, nx, endpoint=False)
dx = L/nx
c = 1.0                     # advection speed
timesteps = [0.001, 0.01]    # compare small vs large dt
nsteps = 200

# Initial condition: high-frequency sine to make errors visible
k0 = 3
u0 = np.sin(2*np.pi*k0*x)

# ------------------------------------------------------------
# WENO5 reconstruction of derivative (flux-splitting version)
# ------------------------------------------------------------
def weno5_flux_derivative(u, c, dx):
    """Compute spatial derivative d/dx (c u) using WENO5 upwind flux splitting"""
    nx = len(u)
    flux = c * u
    f_plus = flux
    f_minus = flux

    # Upwind direction
    if c >= 0:
        # Left-biased reconstruction
        fhat = np.zeros(nx+1)
        for i in range(2, nx-2):
            f0 = f_plus[i-2]; f1 = f_plus[i-1]; f2 = f_plus[i]
            f3 = f_plus[i+1]; f4 = f_plus[i+2]

            # Smoothness indicators
            beta0 = 13/12*(f0 - 2*f1 + f2)**2 + 1/4*(f0 - 4*f1 + 3*f2)**2
            beta1 = 13/12*(f1 - 2*f2 + f3)**2 + 1/4*(f1 - f3)**2
            beta2 = 13/12*(f2 - 2*f3 + f4)**2 + 1/4*(3*f2 - 4*f3 + f4)**2

            eps = 1e-6
            alpha0 = 0.1 / (eps + beta0)**2
            alpha1 = 0.6 / (eps + beta1)**2
            alpha2 = 0.3 / (eps + beta2)**2
            w0 = alpha0 / (alpha0+alpha1+alpha2)
            w1 = alpha1 / (alpha0+alpha1+alpha2)
            w2 = alpha2 / (alpha0+alpha1+alpha2)

            p0 = (2*f0 - 7*f1 + 11*f2)/6
            p1 = (-f1 + 5*f2 + 2*f3)/6
            p2 = (2*f2 + 5*f3 - f4)/6

            fhat[i] = w0*p0 + w1*p1 + w2*p2

        # periodic BC (wrap)
        fhat[0]  = f_plus[0]
        fhat[1]  = f_plus[1]
        fhat[-1] = f_plus[0]

        dudx = (fhat[1:] - fhat[:-1]) / dx
        return dudx

    else:
        # Right-biased reconstruction (not needed here since c>0)
        raise NotImplementedError

# ------------------------------------------------------------
# Implicit Euler time stepping with WENO spatial operator
# For linear advection, we can precompute matrix A.
# ------------------------------------------------------------
def build_weno_matrix(nx, c, dx):
    """Numerical Jacobian of the WENO derivative operator (linearized)."""
    # For linear advection with positive c, WENO reduces to upwind stencil
    # Equivalent to 5th-order upwind linear derivative matrix.
    M = np.zeros((nx, nx))
    for j in range(nx):
        # periodic indexing
        jm3 = (j-3) % nx
        jm2 = (j-2) % nx
        jm1 = (j-1) % nx
        jp0 = j
        jp1 = (j+1) % nx

        # WENO5 linear weights for positive velocity (classic upwind)
        # This is actually a fixed stencil:
        # derivative ≈ (2 u_{i-3} - 15 u_{i-2} + 60 u_{i-1} - 20 u_i - 30 u_{i+1} + 3 u_{i+2}) / 60dx (not flux)
        # but easier is to use flux linearization: fifth-order upwind flux
        # Here we use known coefficients for linear WENO5:
        # derivative of flux at i = (1/dx) * ( -1/30 f_{i-3} + 1/4 f_{i-2} - 1 f_{i-1} + 1/3 f_i + 1/2 f_{i+1} - 1/20 f_{i+2})
        # but simpler: use finite-difference 5th order upwind for d/dx directly
        coeffs = np.array([1/30, -1/4, 1, -1/3, -1/2, 1/20])
        idxs = [(j-3)%nx, (j-2)%nx, (j-1)%nx, j, (j+1)%nx, (j+2)%nx]
        M[j, idxs] = c * coeffs / dx
    return M

# Precompute spatial matrix for implicit solve
M = build_weno_matrix(nx, c, dx)

# ------------------------------------------------------------
# Run for different timesteps
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(x, u0, 'k-', label='Initial')

for dt in timesteps:
    u = u0.copy()
    A = np.eye(nx) - dt*M    # Implicit Euler: (I - dt M) u^{n+1} = u^n
    for n in range(nsteps):
        u = np.linalg.solve(A, u)
    plt.plot(x, u, label=f'dt={dt}')

plt.xlabel('x')
plt.ylabel('u')
plt.title('WENO5 + Implicit Euler: Phase delay & diffusion for large dt')
plt.legend()
plt.grid()
plt.show()