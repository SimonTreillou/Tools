#%%
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
sys.path.append('..')
from diagnostics.grid import zlevs, u2rho, v2rho
from diagnostics.useful import regrid_uniform_z
from diagnostics.qcriterion import *
fname='/Users/simon/Code/CONFIGS/Sandbox/FLASH_RIP_1layer/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_TRC_Dcrit_WMPER_test/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/'
#fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_ShD/'

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
it=50
zeta = ds.variables['zeta'][:,:,:]
u = u2rho(ds.variables['u'][it,:,:,:])
v = v2rho(ds.variables['v'][it,:,:,:])
w = (ds.variables['w'][it,:,:,:])
w = (w[:-1,:,:] + w[1:,:,:]) / 2.0  # average to rho-levels

#%% Compute Q-criterion
Q,Qx,Qy,Qz=compute_Q_components(u,v,w,dx,dy,dz3)
Qx = normalize_volume(Qx)
Qy = normalize_volume(Qy)
Qz = normalize_volume(Qz)

Qx_u, z_u = regrid_uniform_z(zr, Qx, Nz=N)
Qy_u, _ = regrid_uniform_z(zr, Qy, Nz=N)
Qz_u, _ = regrid_uniform_z(zr, Qz, Nz=N)

#%% Plot Qx isosurface
ix0 = np.argmin(np.abs(xr+90))
ix1 = np.argmin(np.abs(xr-0))
iy0 = np.argmin(np.abs(yr-0))
iy1 = np.argmin(np.abs(yr-110))
iso = 0.02
fig, ax, surf, XYZ, z_levels, Qreg = plot_isosurface(
    Qx[:, iy0:iy1, ix0:ix1], xr[ix0:ix1], yr[iy0:iy1], zr[:, iy0:iy1, ix0:ix1], iso,
    h=h[iy0:iy1, ix0:ix1], plot_bathy=True, bathy_stride=1,
    xmin=-90, xmax=0,
    ymin=0,ymax=110,
    zmin=-h[0,ix0]-0.05,zmax=0.5#, base_color=[12, 166, 30]
)

#%% Plot Qy isosurface
ix0 = np.argmin(np.abs(xr+90))
ix1 = np.argmin(np.abs(xr-0))
iy0 = np.argmin(np.abs(yr-0))
iy1 = np.argmin(np.abs(yr-110))
iso = 0.02
fig, ax, surf, XYZ, z_levels, Qreg = plot_isosurface(
    Qy[:, iy0:iy1, ix0:ix1], xr[ix0:ix1], yr[iy0:iy1], zr[:, iy0:iy1, ix0:ix1], iso,
    h=h[iy0:iy1, ix0:ix1], plot_bathy=True, bathy_stride=1,
    xmin=-90, xmax=0,
    ymin=0,ymax=110,
    zmin=-h[0,ix0]-0.05,zmax=0.5#, base_color=[12, 166, 30]
)

# %% Plot Qx and Qy isosurfaces together
fig, ax, surfA, surfB, z_levels, (Qxreg, Qyreg) = plot_two_isosurfaces(
     Qx[:, iy0:iy1, ix0:ix1],
     Qy[:, iy0:iy1, ix0:ix1],
     xr[ix0:ix1],
     yr[iy0:iy1],
     zr[:, iy0:iy1, ix0:ix1],
     iso_a=0.02, iso_b=0.02,
     h=h[iy0:iy1, ix0:ix1],
     xmin=-90, xmax=0, ymin=0, ymax=110,
     zmin=-h[0,ix0]-0.05, zmax=0.5,
     color_a=(189/255, 20/255, 8/255),
     color_b=(12/255, 166/255, 30/255),
     alpha_a=0.90, alpha_b=0.90
)