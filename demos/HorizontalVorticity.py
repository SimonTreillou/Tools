#%%
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
sys.path.append('..')
from diagnostics.grid import zlevs, u2rho, v2rho
from diagnostics.useful import regrid_uniform_z
from diagnostics.vorticity import compute_dUdz
fname='/Users/simon/Code/CONFIGS/Sandbox/FLASH_RIP_1layer/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_TRC_Dcrit_WMPER_test/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/'
#fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_ShD/'
lvl=5

# %% Load variables
ds = nc.Dataset(fname+'rip_avg.nc')
time_fin = ds.variables['scrum_time'][-1]
h = ds.variables['h'][:]
ix0 = np.argmin(np.abs(h[0,:]))
xr = ds.variables['x_rho'][0,:]
xr = xr - xr[ix0]
ixSZ = np.argmin(np.abs(xr+80))
yr = ds.variables['y_rho'][:,0]
dx = float(xr[1] - xr[0])
dy = float(yr[1] - yr[0])
zeta = ds.variables['zeta'][-1,:,:]
theta_s = ds.getncattr('theta_s')
theta_b = ds.getncattr('theta_b')
hc = ds.getncattr('hc')
N=10
zr=zlevs(h, zeta, theta_s, theta_b, hc, N, 'o', 2)
dz3 = zr[1, :, :] - zr[0, :, :]
dz3 = np.tile(dz3[None, :, :], (N, 1, 1))
it=51
zeta = ds.variables['zeta'][:,:,:]
u = u2rho(ds.variables['u'][it,:,:,:])
v = v2rho(ds.variables['v'][it,:,:,:])
w = (ds.variables['w'][it,:,:,:])
w = (w[:-1,:,:] + w[1:,:,:]) / 2.0  # average to rho-levels

# %% Compute vorticity components
# x-axis vorticity
# Defined as dw/dy - dv/dz
dwdy = np.diff(w, axis=1) / dy
# regrid dwdy to rho points
dvdz = compute_dUdz(v, zr)
dvdz = (dvdz[:,1:,:]+dvdz[:,:-1,:])/2.0
omega_x = dwdy-dvdz

# y-axis vorticity
# Defined as du/dz - dw/dx
dwdx = np.diff(w, axis=2) / dx
# regrid dwdy to rho points
dudz = compute_dUdz(u, zr)
dudz = (dudz[:,:,1:]+dudz[:,:,:-1])/2.0
omega_y = dudz-dwdx

#%% Plot x-axis vorticity
vmin=-0.05
plt.figure(dpi=300)
plt.pcolor(xr,yr[:-1], omega_x[lvl,:,:], vmin=vmin, vmax=-vmin, cmap='RdBu_r')
plt.title(r'$\omega_x = \partial w/\partial y - \partial v/\partial z$ at level '+str(lvl)+' (0: bot.)',loc='left')
plt.colorbar()
plt.xlabel(r'x [m]')
plt.ylabel(r'y [m]')
plt.axis('equal')
plt.show()

#%% Plot y-axis vorticity
vmin=-1
plt.figure(dpi=300)
plt.pcolor(xr[:-1],yr, omega_y[lvl,:,:], vmin=vmin, vmax=-vmin, cmap='RdBu_r')
plt.title(r'$\omega_y = \partial u/\partial z - \partial w/\partial x$ at level '+str(lvl)+' (0: bot.)',loc='left')
plt.colorbar()
plt.xlabel(r'x [m]')
plt.ylabel(r'y [m]')
plt.axis('equal')
plt.show()

# %% y-axis vorticity

