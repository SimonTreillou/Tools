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
from diagnostics.correlation import correlation_map
fname='/Users/simon/Code/CONFIGS/Sandbox/FLASH_RIP_1layer/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_2D_TRC_Dcrit_WMPER_test/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/'

# %% Load variables
ds = nc.Dataset(fname+'rip_avg.nc')
time_fin = ds.variables['scrum_time'][-1]
time = ds.variables['scrum_time'][:]
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
zr=zlevs(h, zeta, theta_s, theta_b, hc, N+1, 'o', 2)

it = -1  # time index
u = u2rho(ds.variables['u'][:,:,:,:])
v = v2rho(ds.variables['v'][:,:,:,:])
eke = 0.5*(u[:,-1,:,:]**2 + v[:,-1,:,:]**2)

#%% Compute
shear = np.max(np.diff(u[:,:,:,:], axis=1),axis=1) / np.diff(zr, axis=0)[0,:,:][np.newaxis,:,:]
corrmap = correlation_map(h<=0, eke, shear)

#%% Plot EKE
plt.figure(dpi=300)
plt.pcolor(xr,yr,eke[it,:,:],cmap='Oranges')
plt.vlines(xr.max()-20-80, yr[0], yr[-1], colors='k', linestyles='--')
plt.colorbar()
plt.clim([0, np.max(eke[it,:,:])*0.5])
plt.xlabel(r'x [m]')
plt.ylabel(r'y [m]')
plt.title(r'Eddy Kinetic Energy [m$^2$/s$^2$]',loc='left')
plt.show()

#%% Plot vertical shear
plt.figure(dpi=300)
plt.pcolor(xr,yr,shear[it,:,:],cmap='Oranges')
plt.vlines(xr.max()-20-80, yr[0], yr[-1], colors='k', linestyles='--')
plt.colorbar()
plt.clim([0, np.max(shear[it,:,:])*0.05])
plt.xlabel(r'x [m]')
plt.ylabel(r'y [m]')
plt.title(r'Vertical Shear $\partial U/\partial z$ [m/s]',loc='left')
plt.show()

#%% Plot correlation map
point1 = (400, 350)  # y,x
point2 = (400, 400)  # y,x
plt.figure(dpi=300)
plt.pcolor(xr,yr,corrmap,cmap='RdBu_r')
plt.colorbar()
plt.clim([-1,1])
plt.vlines(xr.max()-20-80, yr[0], yr[-1], colors='k', linestyles='--')
plt.scatter([xr[point1[1]], xr[point2[1]]], [yr[point1[0]], yr[point2[0]]], color='w', marker='x')
plt.xlabel(r'x [m]')
plt.ylabel(r'y [m]')
plt.title(r'Correlation between EKE and vertical shear [-]',loc='left')
plt.show()

#%% Plot example of timeseries 
fig,axs=plt.subplots(2,1,dpi=300,sharex=True, figsize=(8,6))

axs[0].plot(time,eke[:,point1[0],point1[1]],label='EKE',color='r',alpha=0.7,linewidth=2)
axs[0].plot(time,shear[:,point1[0],point1[1]],label='Vert. shear',color='b',alpha=0.7,linewidth=2)
axs[0].set_title(r'Timeseries at point ('+str(int(xr[point1[1]]))+', '+str(int(yr[point1[0]]))+') m',loc='left')
axs[0].set_ylabel(r'EKE [m$^2$/s$^2$]')
axs[0].legend()
axs[0].grid()

axs[1].plot(time,eke[:,point2[0],point2[1]],label='EKE',color='r',alpha=0.7,linewidth=2)
axs[1].plot(time,shear[:,point2[0],point2[1]],label='Vert. shear',color='b',alpha=0.7,linewidth=2)
axs[1].set_title(r'Timeseries at point ('+str(int(xr[point2[1]]))+', '+str(int(yr[point2[0]]))+') m',loc='left')
axs[1].set_ylabel(r'EKE [m$^2$/s$^2$]')
axs[1].legend()
axs[1].grid()
axs[1].set_xlabel(r'Time [s]')
plt.show()
# %%
