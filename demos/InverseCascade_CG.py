#%%
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import signal
import pandas as pd
import sys
from matplotlib.colors import LogNorm
sys.path.append('..')
from diagnostics.grid import zlevs, u2rho, v2rho
from diagnostics.energy import energy_spectrum
from diagnostics.vorticity import compute_vorticity, enstrophy_spectrum
from diagnostics.viz import plot_3D
from diagnostics.useful import compute_coarse_grained_field
import numpy as np
from concurrent.futures import ThreadPoolExecutor
fname='/Users/simon/Code/CONFIGS/Sandbox/FLASH_RIP_1layer/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_2D_TRC_Dcrit_WMPER_test/'
#fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/'

# %% Load variables
ds = nc.Dataset(fname+'rip_avg.nc')
time_fin = ds.variables['scrum_time'][-1]
xr = ds.variables['x_rho'][0,:]
yr = ds.variables['y_rho'][:,0]
dx= xr[1]-xr[0]
h = ds.variables['h'][:]
yr = ds.variables['y_rho'][:,0]
zeta = ds.variables['zeta'][-1,:,:]
theta_s = ds.getncattr('theta_s')
theta_b = ds.getncattr('theta_b')
hc = ds.getncattr('hc')
N=10
zr=zlevs(h, zeta, theta_s, theta_b, hc, N, 'o', 2)

u = u2rho(ds.variables['u'][:,-1,:,:])
v = v2rho(ds.variables['v'][:,-1,:,:])

#%%
scale=10.0  # Coarse-graining scale in meters
scales = np.atleast_1d(scale)

def _compute_for_scale(sc):
    uL = compute_coarse_grained_field(u,dx,scale)
    uL = uL[:,0,:,:]
    vL = compute_coarse_grained_field(v,dx,scale)
    vL = vL[:,0,:,:]
    return uL,vL

# use threads (numpy ops release the GIL for heavy work). If you prefer processes,
# swap ThreadPoolExecutor -> ProcessPoolExecutor (may require top-level helper).
with ThreadPoolExecutor() as ex:
    results = list(ex.map(_compute_for_scale, scales))
uL,vL= results[0]

#%% Plot kinetic energy fields
energy = 0.5 * (u[-1,:,:]**2 + v[-1,:,:]**2)
energyl = 0.5 * (uL[-1,:,:]**2 + vL[-1,:,:]**2)
energys = 0.5 * ((u[-1,:,:]-uL[-1,:,:])**2 + (v[-1,:,:]-vL[-1,:,:])**2)

plt.figure(dpi=300,figsize=(5,5))
vmax=np.max(np.abs(energy))*1.0;vmin=1e-4
plt.pcolormesh(xr,yr,energy, cmap='Oranges', norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
plt.colorbar()
plt.title('Kinetic energy at the surface layer',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

plt.figure(dpi=300,figsize=(5,5))
energyl[np.isnan(energyl)]=0.0
vmax=np.max(np.abs(energyl))*1.0;vmin=1e-4
plt.pcolormesh(xr,yr,energyl, cmap='Oranges', norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
plt.colorbar()
plt.title('Kinetic energy - large scales',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

plt.figure(dpi=300,figsize=(5,5))
energys[np.isnan(energys)]=0.0
vmax=np.max(np.abs(energys))*1.0;vmin=1e-4
plt.pcolormesh(xr,yr,energys, cmap='Oranges', norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
plt.colorbar()
plt.title('Kinetic energy - small scales',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()
# %%

# allow `scale` to be scalar or iterable of scales
scales = np.atleast_1d(scale)

def _compute_for_scale(sc):
    uu = compute_coarse_grained_field(u * u, dx, sc)
    vv = compute_coarse_grained_field(v * v, dx, sc)
    uv = compute_coarse_grained_field(u * v, dx, sc)
    return uu, vv, uv

# use threads (numpy ops release the GIL for heavy work). If you prefer processes,
# swap ThreadPoolExecutor -> ProcessPoolExecutor (may require top-level helper).
with ThreadPoolExecutor() as ex:
    results = list(ex.map(_compute_for_scale, scales))

# stack results along a new "scale" axis: shape -> (n_scales, ...)
uuL = np.stack([r[0] for r in results], axis=0)
vvL = np.stack([r[1] for r in results], axis=0)
uvL = np.stack([r[2] for r in results], axis=0)

# if only one scale was requested, keep original single-scale layout for compatibility
if uuL.shape[0] == 1:
    uuL = uuL[0]
    vvL = vvL[0]
    uvL = uvL[0]

#%%
uuL=uuL[:,0,:,:]
vvL=vvL[:,0,:,:]
uvL=uvL[:,0,:,:]
#%%
duLdx = np.gradient(uL, dx, axis=2)
duLdy = np.gradient(uL, dx, axis=1)
dvLdx = np.gradient(vL, dx, axis=2)
dvLdy = np.gradient(vL, dx, axis=1)

#%%
pi = - ((uuL-uL**2)*duLdx + (uvL-uL*vL)*(duLdy + dvLdx) + (vvL - vL**2)*dvLdy)
# %%

plt.figure(dpi=300,figsize=(4,5))
pi[np.isnan(pi)]=0.0 
vmin=-np.max(np.abs(pi))*0.1; vmax=-vmin
plt.pcolor(xr,yr,pi[-1,:,:],cmap='RdBu_r',vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title(r'$\Pi$ at surface layer',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()
# %%
