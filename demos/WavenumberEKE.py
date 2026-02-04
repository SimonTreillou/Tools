#%%
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import sys
sys.path.append('..')
from diagnostics.grid import u2rho, v2rho
from diagnostics.vorticity import compute_Helmholtz_decomposition
from diagnostics.spectrum import compute_psd

configs = ['/Users/simon/Code/CONFIGS/IB09_SZ/IB09_2D_TRC_Dcrit_WMPER_test/',
           '/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/',
           '/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_TRC_Dcrit_WMPER_test/']
names = ['2D TRC', '3D MR', '3D TRC']
colors = {'0':'red', '1':'blue', '2':'green'}
helmholtz=False
lvl = 1  # surface level
it = -1
y_target = 200
x_target = -80
N=256

# %% LONGSHORE VELOCITY ON LONGSHORE ARRAY
plt.figure(dpi=300)


for config,name in zip(configs,names):
    ds = nc.Dataset(config+'rip_avg.nc')
    v = v2rho(ds.variables['v'][:,lvl,:,:])
    h = ds.variables['h'][0,:]
    xr = ds.variables['x_rho'][0,:]
    xr = xr - xr[np.argmin(np.abs(h-0))]
    yr = ds.variables['y_rho'][:,0]
    dx= xr[1]-xr[0]
    dy= yr[1]-yr[0]
    
    ix = np.argmin(np.abs(xr - x_target))
    res = compute_psd(v[it,:,ix], fs=1/dx,
                    nperseg=N,
                    window='hann',
                    detrend='constant',
                    confidence=0.95)
    
    print('--- '+name+' ---')
    print("Variance of v is "+str(v[it,:,ix].var()))
    print("Integral of spectrum is "+str(np.trapz(res['spectrum'], res['freqs'])))
    print('Ratio is '+str(v[it,:,ix].var() / np.trapz(res['spectrum'], res['freqs'])))

    plt.loglog(res['freqs'], res['spectrum'], label=name, color=colors[str(names.index(name))],linewidth=2)
    plt.fill_between(res['freqs'], res['ci_lower'], res['ci_upper'], alpha=0.1,color=colors[str(names.index(name))],label='')

plt.fill_between([1e-1],[2,3],color='gray',alpha=0.1,label='95% CI')
plt.xlabel('Wavenumber [1/m]') 
plt.ylabel('$S_{vv}$ [$m^3/s^2$]')
plt.title('PSD of longshore velocity at x = {:.1f} m'.format(xr[ix]),loc='left')
plt.ylim([1e-8,2e1])
plt.legend()
plt.grid()
plt.show()

# %% CROSS-SHORE VELOCITY ON LONGSHORE ARRAY
plt.figure(dpi=300)


for config,name in zip(configs,names):
    ds = nc.Dataset(config+'rip_avg.nc')
    u = u2rho(ds.variables['u'][:,lvl,:,:])
    h = ds.variables['h'][0,:]
    xr = ds.variables['x_rho'][0,:]
    xr = xr - xr[np.argmin(np.abs(h-0))]
    yr = ds.variables['y_rho'][:,0]
    dx= xr[1]-xr[0]
    dy= yr[1]-yr[0]
    
    ix = np.argmin(np.abs(xr - x_target))
    res = compute_psd(u[it,:,ix], fs=1/dx,
                    nperseg=N,
                    window='hann',
                    detrend='constant',
                    confidence=0.95)

    plt.loglog(res['freqs'], res['spectrum'], label=name, color=colors[str(names.index(name))],linewidth=2)
    plt.fill_between(res['freqs'], res['ci_lower'], res['ci_upper'], alpha=0.1,color=colors[str(names.index(name))],label='')

plt.fill_between([1e-1],[2,3],color='gray',alpha=0.1,label='95% CI')
plt.xlabel('Wavenumber [1/m]') 
plt.ylabel('$S_{uu}$ [$m^3/s^2$]')
plt.title('PSD of cross-shore velocity at x = {:.1f} m'.format(xr[ix]),loc='left')
plt.ylim([1e-8,2e1])
plt.legend()
plt.grid()
plt.show()

# %% EKE ON LONGSHORE ARRAY
plt.figure(dpi=300)

for config,name in zip(configs,names):
    ds = nc.Dataset(config+'rip_avg.nc')
    u = u2rho(ds.variables['u'][:,lvl,:,:])
    v = v2rho(ds.variables['v'][:,lvl,:,:])
    h = ds.variables['h'][0,:]
    xr = ds.variables['x_rho'][0,:]
    xr = xr - xr[np.argmin(np.abs(h-0))]
    yr = ds.variables['y_rho'][:,0]
    dx= xr[1]-xr[0]
    dy= yr[1]-yr[0]
    
    ix = np.argmin(np.abs(xr - x_target))
    resu = compute_psd(u[it,:,ix], fs=1/dx,
                    nperseg=N,
                    window='hann',
                    detrend='constant',
                    confidence=0.95)
    resv = compute_psd(v[it,:,ix], fs=1/dy,
                    nperseg=N,
                    window='hann',
                    detrend='constant',
                    confidence=0.95)
    if helmholtz:
        u_psi, v_psi, u_phi, v_phi =compute_Helmholtz_decomposition(u,v,dx,dy)
        resu_psi = compute_psd(u_psi[it,:,ix], fs=1/dx,
                        nperseg=N,
                        window='hann',
                        detrend='constant',
                        confidence=0.95)
        resv_psi = compute_psd(v_psi[it,:,ix], fs=1/dy,
                        nperseg=N,
                        window='hann',
                        detrend='constant',
                        confidence=0.95)
    
    EKE = 0.5*( (u[it,:,ix]-np.mean(u[it,:,ix]))**2 + (v[it,:,ix]-np.mean(v[it,:,ix]))**2 )
    df = resu['freqs'][1]-resu['freqs'][0]
    integ_spectrum = np.sum(0.5*(resv['spectrum']+resu['spectrum']))*df
    
    print('--- '+name+' ---')
    print("Mean EKE is "+str(EKE.mean()))
    print("Mean spectrum EKE is "+str(integ_spectrum))
    print("Ratio is "+str(EKE.mean() / integ_spectrum))
    print(' ')

    plt.loglog(resu['freqs'], 0.5*(resu['spectrum']+resv['spectrum']), label=name, color=colors[str(names.index(name))],linewidth=2)
    if helmholtz:
        plt.loglog(resu['freqs'], 0.5*(resu_psi['spectrum']+resv_psi['spectrum']), label="", color=colors[str(names.index(name))],linewidth=2,linestyle=':')

    #plt.fill_between(res['freqs'], res['ci_lower'], res['ci_upper'], alpha=0.1,color=colors[str(names.index(name))],label='')

plt.loglog([1e-1],[2],color='gray',linestyle=":",label='Irrotational component')
plt.xlabel('Wavenumber [1/m]') 
plt.ylabel(r'$\frac{1}{2} (S_{uu} + S_{vv})$ [$m^3/s^2$]')
plt.title('PSD of EKE at x = {:.1f} m'.format(xr[ix]),loc='left')
plt.ylim([1e-8,2e1])
plt.legend()
plt.grid()
plt.show()

# %% LONGSHORE VELOCITY ON CROSS-SHORE ARRAY (useless)
plt.figure(dpi=300)


for config,name in zip(configs,names):
    ds = nc.Dataset(config+'rip_avg.nc')
    v = v2rho(ds.variables['v'][:,lvl,:,:])
    h = ds.variables['h'][0,:]
    xr = ds.variables['x_rho'][0,:]
    xr = xr - xr[np.argmin(np.abs(h-0))]
    yr = ds.variables['y_rho'][:,0]
    dx= xr[1]-xr[0]
    dy= yr[1]-yr[0]
    
    iy = np.argmin(np.abs(yr - y_target))
    res = compute_psd(v[it,iy,:], fs=1/dx,
                    nperseg=N,
                    window='hann',
                    detrend='constant',
                    confidence=0.95)

    plt.loglog(res['freqs'], res['spectrum'], label=name, color=colors[str(names.index(name))],linewidth=2)
    plt.fill_between(res['freqs'], res['ci_lower'], res['ci_upper'], alpha=0.1,color=colors[str(names.index(name))],label='')

plt.fill_between([1e-1],[2,3],color='gray',alpha=0.1,label='95% CI')
plt.xlabel('Wavenumber [1/m]') 
plt.ylabel('$S_{vv}$ [$m^3/s^2$]')
plt.title('PSD of longshore velocity at y = {:.1f} m'.format(yr[iy]),loc='left')
plt.ylim([1e-8,2e1])
plt.legend()
plt.grid()
plt.show()


# %%
