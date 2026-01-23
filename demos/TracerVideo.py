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
from diagnostics.video import write_map_to_video
from diagnostics.vorticity import compute_vorticity
import numpy as np
import matplotlib.colors as colors
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_2D_TRC_Dcrit_WMPER_test/'
#fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/'

#%% Load variables
ds = nc.Dataset(fname+'rip_avg.nc')
time = ds.variables['scrum_time'][:]
xr = ds.variables['x_rho'][0,:]
dx = xr[1]-xr[0]
yr = ds.variables['y_rho'][:,0]
h = ds.variables['h'][:]
ix0 = np.argmin(np.abs(h[0,:]))
ixSZ = np.argmin(np.abs(xr-xr[ix0]+80))
yr = ds.variables['y_rho'][:,0]
zeta = ds.variables['zeta'][-1,:,:]
u = ds.variables['u'][:,-1,:,:]
v = ds.variables['v'][:,-1,:,:]
theta_s = ds.getncattr('theta_s')
theta_b = ds.getncattr('theta_b')
hc = ds.getncattr('hc')
N=10
zr=zlevs(h, zeta, theta_s, theta_b, hc, N, 'o', 2)
tpas = ds.variables['tpas01'][:,-1,:,:]
vort = compute_vorticity(u,v,dx,dx)

#%% First visualization
fig, ax = plt.subplots(figsize=(8,8), dpi=300)
pcm = ax.pcolor(tpas[-2], cmap='OrRd', shading="auto",norm=colors.LogNorm(vmin=1e-4, vmax=5))
cbar = fig.colorbar(pcm, ax=ax)
ax.set_xlabel(r'$x$ (m)')
ax.set_ylabel(r'$y$ (m)')
ax.vlines(ixSZ, ymin=0, ymax=tpas.shape[1], colors='k', linestyles='--')
plt.show()

#%% Vorticity
fig, ax = plt.subplots(figsize=(8,8), dpi=300)
pcm = ax.pcolor(vort[-2], cmap='RdBu', shading="auto",norm=colors.Normalize(vmin=-0.05, vmax=0.05))
cbar = fig.colorbar(pcm, ax=ax)
ax.set_xlabel(r'$x$ (m)')
ax.set_ylabel(r'$y$ (m)')
ax.vlines(ixSZ, ymin=0, ymax=vort.shape[1], colors='k', linestyles='--')
plt.show()

# %% Write video
path="/Users/simon/Code/Figures/Videos/Tracer_concentration_Treillou2025_3D_ShD"
write_map_to_video(tpas,time,path,x=None,y=None,title=r'Surface tracer concentration ($ppb$)',fps=5,norm=colors.LogNorm(vmin=1e-4, vmax=5),cmap="OrRd",label="Tracer concentration [ppb]", vline=ixSZ,xlab=r'Grid points in $x$ [-]', ylab=r'Grid points in $y$ [-]')
# %%
# %% Write vorticity video
path="/Users/simon/Code/Figures/Videos/Vorticity_Treillou2025_2D_TRC"
write_map_to_video(vort,time,path,x=None,y=None,title=r'Surface vorticity ($s^{-1}$)',fps=5,norm=colors.Normalize(vmin=-0.05, vmax=0.05),cmap="RdBu",label="Vorticity [s$^{-1}$]", vline=ixSZ,xlab=r'Grid points in $x$ [-]', ylab=r'Grid points in $y$ [-]')

# %%
