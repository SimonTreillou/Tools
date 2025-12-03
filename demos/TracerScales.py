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
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_TRC_Dcrit_WMPER_test/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/'

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
tpas = ds.variables['tpas01'][:,-1,:,:]

#%% Compute coarse-grained fields
scale=10.0  # Coarse-graining scale in meters
scales = np.atleast_1d(scale)

def _compute_for_scale(sc):
    uL = compute_coarse_grained_field(u,dx,scale)
    uL = uL[:,0,:,:]
    vL = compute_coarse_grained_field(v,dx,scale)
    vL = vL[:,0,:,:]
    tpasL = compute_coarse_grained_field(tpas,dx,scale)
    tpasL = tpasL[:,0,:,:]
    return uL,vL,tpasL

# use threads (numpy ops release the GIL for heavy work). If you prefer processes,
# swap ThreadPoolExecutor -> ProcessPoolExecutor (may require top-level helper).
with ThreadPoolExecutor() as ex:
    results = list(ex.map(_compute_for_scale, scales))
uL,vL,tpasL= results[0]

#%% Compute flux terms 
flux = np.abs((u[-1,:,:]-np.mean(u,0))*(tpas[-1,:,:]-np.mean(tpas,0))) # + (v[-1,:,:]-np.mean(v,0))*(tpas[-1,:,:]-np.mean(tpas,0))
fluxL = np.abs(uL[-1,:,:]*tpasL[-1,:,:]) #+ vL[-1,:,:]*tpasL[-1,:,:]
fluxS = np.abs((u[-1,:,:]-uL[-1,:,:])*(tpas[-1,:,:]-tpasL[-1,:,:]))
dTdx = np.gradient(tpas, dx, axis=2)
dTLdx = np.gradient(tpasL, dx, axis=2)
dTSdx = np.gradient(tpas-tpasL, dx, axis=2)

#%% Show plots  
plt.figure(dpi=300,figsize=(5,5))
vmax=np.max(np.abs(flux))*1.0;vmin=1e-6
flux[flux<=0]=vmin
plt.pcolormesh(xr,yr,flux, cmap='Oranges', norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
plt.colorbar()
plt.title(f'Magnitude of flux $\overline{{u\'T\'}}$ at the surface layer',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

plt.figure(dpi=300,figsize=(5,5))
fluxL[np.isnan(fluxL)]=0.0
plt.pcolormesh(xr,yr,fluxL, cmap='Oranges', norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
plt.colorbar()
plt.title(f'Magnitude of flux $\overline{{u\'T\'}}$ - large scales',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

plt.figure(dpi=300,figsize=(5,5))
fluxS[np.isnan(fluxS)]=0.0
plt.pcolormesh(xr,yr,fluxS, cmap='Oranges', norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
plt.colorbar()
plt.title(f'Magnitude of flux $\overline{{u\'T\'}}$ - small scales',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()
# %% Show tracer gradients
plt.figure(dpi=300,figsize=(5,5))
vmin=-np.max(np.abs(dTdx))*0.1;vmax=-vmin
plt.pcolormesh(xr,yr,dTdx[-1,:,:], cmap='RdBu_r', shading='auto',vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title(f'$\partial T / \partial x$ - total',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

# %% Show diffusivity
plt.figure(dpi=300,figsize=(5,5))
Kx=-flux/dTdx[-1,:,:]
vmin=-5.0;vmax=-vmin
plt.pcolormesh(xr,yr,Kx, cmap='RdBu_r', shading='auto',vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title(f'$K_x = - \overline{{u\'T\'}}/ \\partial T / \\partial x$ - total',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

plt.figure(dpi=300,figsize=(5,5))
KxL=-fluxL/dTLdx[-1,:,:]
vmin=-5.0;vmax=-vmin
plt.pcolormesh(xr,yr,KxL, cmap='RdBu_r', shading='auto',vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title(f'$K_x = - \overline{{u\'T\'}}/ \\partial T / \\partial x$ - large scales',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

plt.figure(dpi=300,figsize=(5,5))
KxS=-fluxS/dTSdx[-1,:,:]
vmin=-5.0;vmax=-vmin
plt.pcolormesh(xr,yr,KxS, cmap='RdBu_r', shading='auto',vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title(f'$K_x = - \overline{{u\'T\'}}/ \\partial T / \\partial x$ - small scales',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

# %%
flux = ((u[-1,:,:]-np.mean(u,0))*(tpas[-1,:,:]-np.mean(tpas,0))) # + (v[-1,:,:]-np.mean(v,0))*(tpas[-1,:,:]-np.mean(tpas,0))
fluxL = (uL[-1,:,:]*tpasL[-1,:,:]) #+ vL[-1,:,:]*tpasL[-1,:,:]
fluxS = ((u[-1,:,:]-uL[-1,:,:])*(tpas[-1,:,:]-tpasL[-1,:,:]))

fluxL[np.isnan(fluxL)]=0.0
fluxS[np.isnan(fluxS)]=0.0
k,E=enstrophy_spectrum(flux,dx,dx)
k,EL=enstrophy_spectrum(fluxL,dx,dx)
k,ES=enstrophy_spectrum(fluxS,dx,dx)

plt.figure(dpi=300) 
plt.loglog(k,E,'k',label='Total ')
plt.loglog(k,EL,'b',label='Large-scale ')
plt.loglog(k,ES,'r',label='Small-scale ')
plt.loglog(k,0.0000001*k**-2,'k--',label='-2 slope')
plt.legend()
plt.xlabel('Wavenumber (rad/m)')
plt.ylabel('Spectrum')
plt.title('Spectrum of tracer flux at surface layer') 
plt.show()

# %%
