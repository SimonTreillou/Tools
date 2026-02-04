#%%
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
sys.path.append('..')
from diagnostics.grid import zlevs, u2rho, v2rho
from diagnostics.vorticity import compute_Helmholtz_decomposition
fname='/Users/simon/Code/CONFIGS/Sandbox/FLASH_RIP_1layer/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_TRC_Dcrit_WMPER_test/'
#fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/'

# %% Load variables
ds = nc.Dataset(fname+'rip_his.nc')
time_fin = ds.variables['scrum_time'][-1]
xr = ds.variables['x_rho'][0,:]
yr = ds.variables['y_rho'][:,0]
dx= xr[1]-xr[0]
dy= yr[1]-yr[0]
h = ds.variables['h'][:]
yr = ds.variables['y_rho'][:,0]
zeta = ds.variables['zeta'][-1,:,:]
theta_s = ds.getncattr('theta_s')
theta_b = ds.getncattr('theta_b')
hc = ds.getncattr('hc')
N=10
zr=zlevs(h, zeta, theta_s, theta_b, hc, N, 'o', 2)

it = -1  # time index
u = u2rho(ds.variables['u'][:,-1,:,:])
v = v2rho(ds.variables['v'][:,-1,:,:])

# %% Compute

u_psi, v_psi, u_phi, v_phi = compute_Helmholtz_decomposition(u,v,dx,dy)

# %% Plot
it=-1
factor=0.5
cmap='RdBu_r'

fig, axs = plt.subplots(2,3, figsize=(10,8))

im0 = axs[0,0].pcolor(xr,yr,u[it,:,:], shading='auto', cmap=cmap)
axs[0,0].set_title(r'Cross-shore velocity',loc='left')
im0.set_clim(-np.max(np.abs(u[it,:,:]))*factor, np.max(np.abs(u[it,:,:]))*factor)
plt.colorbar(im0, ax=axs[0,0])  
axs[0,0].set_ylabel(r'y [m]')

im1 = axs[0,1].pcolor(xr,yr,u_psi[it,:,:], shading='auto', cmap=cmap)
axs[0,1].set_title(r'Rotational ($u_\psi$)',loc='left')
im1.set_clim(-np.max(np.abs(u_psi[it,:,:]))*factor, np.max(np.abs(u_psi[it,:,:]))*factor)
plt.colorbar(im1, ax=axs[0,1])  

im2 = axs[0,2].pcolor(xr,yr,u_phi[it,:,:], shading='auto', cmap=cmap)
axs[0,2].set_title(r'Irrotational ($u_\phi$)',loc='left')
im2.set_clim(-np.max(np.abs(u_phi[it,:,:]))*factor, np.max(np.abs(u_phi[it,:,:]))*factor)
plt.colorbar(im2, ax=axs[0,2])

im3 = axs[1,0].pcolor(xr,yr,v[it,:,:], shading='auto', cmap=cmap)
axs[1,0].set_title(r'Longshore velocity',loc='left')
im3.set_clim(-np.max(np.abs(v[it,:,:]))*factor, np.max(np.abs(v[it,:,:]))*factor)
plt.colorbar(im3, ax=axs[1,0])  
axs[1,0].set_ylabel(r'y [m]')
axs[1,0].set_xlabel(r'x [m]')

im4 = axs[1,1].pcolor(xr,yr,v_psi[it,:,:], shading='auto', cmap=cmap)
axs[1,1].set_title(r'Rotational ($v_\psi$)',loc='left')
im4.set_clim(-np.max(np.abs(v_psi[it,:,:]))*factor, np.max(np.abs(v_psi[it,:,:]))*factor)
plt.colorbar(im4, ax=axs[1,1])
axs[1,1].set_xlabel(r'x [m]')

im5 = axs[1,2].pcolor(xr,yr,v_phi[it,:,:], shading='auto', cmap=cmap)
axs[1,2].set_title(r'Irrotational ($v_\phi$)',loc='left')
im5.set_clim(-np.max(np.abs(v_phi[it,:,:]))*factor, np.max(np.abs(v_phi[it,:,:]))*factor)
plt.colorbar(im5, ax=axs[1,2])
axs[1,2].set_xlabel(r'x [m]')

plt.tight_layout()
plt.show()

