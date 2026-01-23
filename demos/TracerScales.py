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
from diagnostics.useful import compute_coarse_grained_field,smooth
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numpy.lib.stride_tricks import sliding_window_view
fname='/Users/simon/Code/CONFIGS/Sandbox/FLASH_RIP_1layer/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_TRC_Dcrit_WMPER_test/'
fname='/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/'

# %% Load variables
ds = nc.Dataset(fname+'rip_avg.nc')
time_fin = ds.variables['scrum_time'][-1]
h = ds.variables['h'][:]
ix0 = np.argmin(np.abs(h[0,:]))
xr = ds.variables['x_rho'][0,:]
xr = xr - xr[ix0]
ixSZ = np.argmin(np.abs(xr+80))
yr = ds.variables['y_rho'][:,0]
dx= xr[1]-xr[0]
yr = ds.variables['y_rho'][:,0]
zeta = ds.variables['zeta'][-1,:,:]
theta_s = ds.getncattr('theta_s')
theta_b = ds.getncattr('theta_b')
hc = ds.getncattr('hc')
N=10
zr=zlevs(h, zeta, theta_s, theta_b, hc, N, 'o', 2)

zeta = ds.variables['zeta'][:,:,:]
u = u2rho(ds.variables['u'][:,-1,:,:])
v = v2rho(ds.variables['v'][:,-1,:,:])
tpas = ds.variables['tpas01'][:,-1,:,:]

#%% Compute coarse-grained fields
scale=30.0  # Coarse-graining scale in meters
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
scale=10.0
tpasL = compute_coarse_grained_field(tpas[-1,:,:]-np.mean(tpas[-1,:,:]),dx,scale)
tpasL = tpasL[0,:,:]

# %%
plt.figure(dpi=300,figsize=(5,5))
vmax=1.0;vmin=0.0
plt.pcolormesh(xr,yr,tpasL, cmap='Oranges', shading='auto',vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title(f'Coarse-grained tracer concentration at scale={scale} m',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

# %%
tvar=np.nanvar(tpas[-1,:,:],axis=0)
#tvar=np.nanvar(test_padded,axis=0)
tvarL=np.nanvar(tpasL,axis=0)
tvarS=np.nanvar(tpas[-1,:,ixSZ-20:ix0+20]-tpasL,axis=0)
#tvarS=np.nanvar(test_padded-tpasL,axis=0)
integrated_tvar=np.nansum(tvar)*dx*dx
integrated_tvarL=np.nansum(tvarL)*dx*dx
integrated_tvarS=np.nansum(tvarS)*dx*dx
plt.figure(dpi=300,figsize=(6,4))
plt.plot(tvar,label='Total',color='k',linestyle='--',alpha=0.8,linewidth=2)
plt.plot(tvarL,label=f'Scales > {scale} m',color='r',alpha=0.8,linewidth=4)
plt.plot(tvarS,label=f'Scales < {scale} m',color='b',alpha=0.8,linewidth=4)
plt.vlines(xr[-1]-90, ymin=0, ymax=tvar.max(), colors='k', linestyles='--',alpha=0.5,label='Surfzone edge')
plt.legend()   
plt.xlabel('x [m]')
#plt.xlim([100,xr.max()])
plt.ylabel(r'Variance of tracer concentration [(ppb)$^2$]')
plt.show()









# %%
scale=100.0
test=tpas[-1,:,:]-np.mean(tpas[-1,:,:])
test_padded = np.pad(
    test,
    pad_width=int(scale),           # 10 rows/cols all around
    mode='constant',
    constant_values=0
)
#%%
tpasLpadded = compute_coarse_grained_field(test_padded,dx,scale)
tpasLpadded = tpasLpadded[0,:,:]
tpasL = compute_coarse_grained_field(test,dx,scale)
tpasL = tpasL[0,:,:]

# %%
plt.figure(dpi=300,figsize=(5,5))
vmax=1.0;vmin=0.0
tpasLpadded2 = tpasLpadded[int(scale):-int(scale),int(scale):-int(scale)]
plt.pcolormesh(tpasLpadded2, cmap='Oranges', shading='auto',vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title(f'Padded',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
#plt.vlines(scale, ymin=scale, ymax=tpasL.shape[0]+scale, colors='k', linestyles='--',alpha=0.5,label='Original field size')
#plt.vlines(len(xr)+scale, ymin=scale, ymax=tpasL.shape[0]+scale, colors='k', linestyles='--',alpha=0.5,label='')
#plt.hlines(scale, xmin=scale, xmax=tpasL.shape[1]+scale, colors='k', linestyles='--',alpha=0.5,label='')
#plt.hlines(len(yr)+scale, xmin=scale, xmax=tpasL.shape[1]+scale, colors='k', linestyles='--',alpha=0.5,label='')
#plt.legend()
plt.show()

#%%
plt.figure(dpi=300,figsize=(5,5))
vmax=1.0;vmin=0.0
plt.pcolormesh(tpasL, cmap='Oranges', shading='auto',vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title(f'Original',loc='left')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.vlines(0, ymin=0, ymax=tpasL.shape[0], colors='k', linestyles='--',alpha=0.5,label='Original field size')
plt.vlines(len(xr), ymin=0, ymax=tpasL.shape[0], colors='k', linestyles='--',alpha=0.5,label='')
plt.hlines(0, xmin=0, xmax=tpasL.shape[1], colors='k', linestyles='--',alpha=0.5,label='')
plt.hlines(len(yr), xmin=0, xmax=tpasL.shape[1], colors='k', linestyles='--',alpha=0.5,label='')
plt.legend()
plt.show()

#%% 
plt.figure(dpi=300,figsize=(6,4))
plt.plot(np.nanvar(tpasLpadded2,axis=0))
plt.plot(np.nanvar(tpasL,axis=0))
plt.legend(['Padded','Original'])
plt.show()






#%%

def compute_Kef_scale_dependant(u,tpas,scale,dx,constraint_kef=50.0):
    tpasL = compute_coarse_grained_field(tpas-np.mean(tpas),dx,scale)
    tpasL = tpasL[0,:,:]
    uL = compute_coarse_grained_field(u-np.mean(u),dx,scale)
    uL = uL[0,:,:]
    uS = u- uL
    tpasS = tpas - tpasL
    print(tpasL.shape)

    uSTSL = compute_coarse_grained_field(tpasS*uS,dx,scale)
    dTLdx = np.gradient(tpasL, dx, axis=1)
    Kef = - uSTSL*dTLdx / (np.abs(dTLdx)**2)
    Kef[np.abs(Kef)>constraint_kef]=np.nan
    return dTLdx

#%%
scale=50.0
tpasL = compute_coarse_grained_field(tpas[-1,:,:]-np.mean(tpas[-1,:,:]),dx,scale)
tpasL = tpasL[0,:,:]
uL = compute_coarse_grained_field(u[-1,:,:]-np.mean(u[-1,:,:]),dx,scale)
uL = uL[0,:,:]
uS = u[-1,:,:]- uL
tpasS = tpas[-1,:,:]- tpasL

uSTSL = compute_coarse_grained_field(tpasS*uS,dx,scale)
uSTSL = uSTSL[0,:,:]

#%%
scale=20.0
Kef= compute_Kef_scale_dependant(u[-1,:,:],tpas[-1,:,:],scale,dx,constraint_kef=500.0)

# %%
plt.plot(xr,np.nanmean(Kef,axis=0))
plt.hlines(0,xr[0],xr[-1],colors='k',linestyles='--')
plt.xlabel('x [m]')
plt.ylabel('Effective diffusivity [m$^2$/s]')
plt.title(f'Effective diffusivity at scale={scale} m')
plt.ylim([-1,1])
#plt.xlim(350,500)
plt.show()
#%%
plt.pcolor(Kef);plt.colorbar()
# %%
print(np.nanmean(np.abs(Kef[:,ixSZ:ix0])))

# %%
s=[2,5,10,20,50,100]
diff=[0.0135996705,0.013079347,0.012207726,0.011097647,0.0076069282,0.005221521]
diff2=[0.020886824,0.019801207,0.018698756,0.017965717,0.015501606,0.013973603]
plt.figure(dpi=300)
plt.loglog(s,diff,'o-',label='TRC and mini-rips')
plt.loglog(s,diff2,'o-',label='Mini-rips only')
plt.xlabel('Coarse-graining scale [m]')
plt.ylabel('Effective diffusivity [m$^2$/s]')
plt.legend()
plt.title('Effective diffusivity vs coarse-graining scale in the surfzone')
plt.show()
# %%



#%%
scale=70.0
tpasL = compute_coarse_grained_field(tpas,dx,scale,mode="x")
tpasL = tpasL[0,:,:]
tpasS = tpas - tpasL
uL = compute_coarse_grained_field(u[-1,:,:],dx,scale,mode="x")
uS = u[-1,:,:]- uL
flux = compute_coarse_grained_field(np.abs(uS*tpasS),dx,scale,mode="x")
scale=int(scale)

tpasT = tpas[scale:-scale,scale:-scale]
tpasL = tpasL[scale:-scale,scale:-scale]
tpasS  = tpasS[scale:-scale,scale:-scale]

plt.plot(np.var(tpasT,axis=0),label='Total',color='k',linestyle='--',alpha=0.8,linewidth=2)
plt.plot(np.var(tpasL,axis=0),label=f'Scales > {scale*dx} m',color='r',alpha=0.8,linewidth=4)
plt.plot(np.var(tpasS,axis=0),label=f'Scales < {scale*dx} m',color='b',alpha=0.8,linewidth=4)
plt.legend()
plt.xlabel('x [m]')
#plt.xlim([100,xr.max()])
plt.ylabel(r'Variance of tracer concentration [(ppb)$^2$]')
plt.show()  
# %%

for scale in [2,3,5,7,10,20,30]:
    tpasL = compute_coarse_grained_field(tpas,dx,scale,mode="y")
    tpasL = tpasL[0,:,:]
    tpasS = tpas - tpasL
    uL = compute_coarse_grained_field(u[-1,:,:],dx,scale,mode="y")
    uS = u[-1,:,:]- uL
    flux = compute_coarse_grained_field(np.abs(uS*tpasS),dx,scale,mode="y")
    flux = flux[0,0,:,:]
    plt.plot(np.nanvar(flux,0),label=f'Scale={scale} m')

plt.legend()
plt.xlabel('x [m]')
plt.xlim([400,500])
plt.ylabel(r'Mean tracer flux [ppb m/s]')
plt.show()
    

# %%


def compute_coarse_grained_field(field, dx, scales, padding=False, mode="xy"):
    """
    Coarse-grain a field using a circular (2D) or segment (1D) top-hat convolution.
    Supports input shapes:
      - (ny, nx)
      - (t, ny, nx)
      - (t, z, ny, nx)
    Returns an array with the scale axis inserted just before the last two dims:
      e.g. (ny,nx) -> (nscale, ny, nx)
            (t,ny,nx) -> (t, nscale, ny, nx)
            (t,z,ny,nx) -> (t, z, nscale, ny, nx)

    dx can be:
      - scalar (constant grid spacing), or
      - 2D array of shape (ny, nx) (spatially varying)
    scales can be scalar or 1D array/ list.

    mode:
      - "xy": 2D circular top-hat (default)
      - "x":  1D top-hat along x only
      - "y":  1D top-hat along y only
    """

    arr = np.asarray(field)
    scales = np.atleast_1d(scales).astype(float)
    nscale = len(scales)

    if arr.ndim < 2:
        raise ValueError("field must have at least 2 dimensions (y,x).")
    ny, nx = arr.shape[-2], arr.shape[-1]
    leading_shape = arr.shape[:-2]

    # prepare grid spacing array (2D)
    if np.isscalar(dx):
        gs = np.full((ny, nx), float(dx))
    else:
        gs = np.asarray(dx)
        if gs.shape != (ny, nx):
            raise ValueError("dx must be scalar or shape (ny, nx) matching field's last two dims.")

    gs_min = float(gs.min())

    out_shape = leading_shape + (nscale, ny, nx)
    out = np.full(out_shape, np.nan, dtype=arr.dtype)

    # Helper to iterate leading dims
    if leading_shape:
        iterator = np.ndindex(*leading_shape)
    else:
        iterator = [()]

    mode = str(mode).lower()
    if mode not in ("xy", "x", "y"):
        raise ValueError("mode must be one of 'xy', 'x', 'y'.")

    for k, L in enumerate(scales):
        # radius (pixels) computed conservatively from smallest grid cell
        radius = int(round(L / gs_min / 2.0))
        w = 2 * radius + 1
        if w <= 1 or w > ny and mode in ("xy", "y") or w > nx and mode in ("xy", "x"):
            # window too small or too large -> skip
            continue

        if mode == "xy":
            # circular mask in meters: pixel distances * gs_min <= L/2
            yy, xx = np.indices((w, w))
            dist_pix = np.sqrt((yy - radius) ** 2 + (xx - radius) ** 2)
            circ = (dist_pix * gs_min) <= (L / 2.0)  # boolean (w,w)

            gsW = sliding_window_view(gs, (w, w))  # (ny-w+1, nx-w+1, w, w)
            areaW = gsW ** 2
            area_sum = np.sum(areaW * circ, axis=(-2, -1))

            for idx in iterator:
                f2d = arr[idx] if idx else arr
                fW = sliding_window_view(f2d, (w, w))
                f_weighted_sum = np.sum(fW * areaW * circ, axis=(-2, -1))
                with np.errstate(invalid='ignore', divide='ignore'):
                    cg_interior = np.where(area_sum > 0, f_weighted_sum / area_sum, np.nan)
                out_idx = (tuple(list(idx) + [k, slice(radius, ny - radius), slice(radius, nx - radius)])
                           if idx else (k, slice(radius, ny - radius), slice(radius, nx - radius)))
                out[out_idx] = cg_interior

        elif mode == "x":
            # 1D window along x, length-weighted by dx
            gsW = sliding_window_view(gs, (1, w))  # (ny, nx-w+1, 1, w)
            lenW = gsW[..., 0, :]                  # (ny, nx-w+1, w)
            len_sum = np.sum(lenW, axis=-1)        # (ny, nx-w+1)

            for idx in iterator:
                f2d = arr[idx] if idx else arr
                fW = sliding_window_view(f2d, (1, w))  # (ny, nx-w+1, 1, w)
                fW = fW[..., 0, :]                     # (ny, nx-w+1, w)
                f_len_sum = np.sum(fW * lenW, axis=-1) # (ny, nx-w+1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    cg_interior = np.where(len_sum > 0, f_len_sum / len_sum, np.nan)
                # place: y unaffected, x shrinks by radius at both ends
                out_idx = (tuple(list(idx) + [k, slice(0, ny), slice(radius, nx - radius)])
                           if idx else (k, slice(0, ny), slice(radius, nx - radius)))
                out[out_idx] = cg_interior

        else:  # mode == "y"
            # 1D window along y, length-weighted by dy (assume dx ~ dy from gs)
            gsW = sliding_window_view(gs, (w, 1))  # (ny-w+1, nx, w, 1)
            lenW = gsW[..., :, 0]                  # (ny-w+1, nx, w)
            len_sum = np.sum(lenW, axis=-1)        # (ny-w+1, nx)

            for idx in iterator:
                f2d = arr[idx] if idx else arr
                fW = sliding_window_view(f2d, (w, 1))  # (ny-w+1, nx, w, 1)
                fW = fW[..., :, 0]                     # (ny-w+1, nx, w)
                f_len_sum = np.sum(fW * lenW, axis=-1) # (ny-w+1, nx)
                with np.errstate(invalid='ignore', divide='ignore'):
                    cg_interior = np.where(len_sum > 0, f_len_sum / len_sum, np.nan)
                # place: x unaffected, y shrinks by radius at both ends
                out_idx = (tuple(list(idx) + [k, slice(radius, ny - radius), slice(0, nx)])
                           if idx else (k, slice(radius, ny - radius), slice(0, nx)))
                out[out_idx] = cg_interior

    return out

# %%

k,E=enstrophy_spectrum(tpas[-200,:,:],dx,dx)
plt.loglog(k,E)
plt.loglog(k,k**-(1)*0.0001,'b--')
plt.loglog(k,k**-(5/3)*0.000001,'k--')
plt.loglog(k,k**-(6)*0.00000001,'r--')
plt.loglog(k,k**-(4)*0.00000001,'g--')
plt.loglog(k,k**-1*(np.log(k)**(-1/3))*1e-6,'c--')

plt.ylim([1e-11,1e-1])
plt.show()
# %%







#%%
import numpy as np

def compute_cospectrum_1d(u, v, dx, detrend=True):
    """
    Compute 1D cospectrum Co_uv(k) from two signals u(x), v(x).

    Parameters
    ----------
    u, v : 1D arrays
        Real-space signals
    dx : float
        Grid spacing
    detrend : bool
        Remove mean before FFT

    Returns
    -------
    k : 1D array
        Wavenumbers
    Co_uv : 1D array
        Cospectrum
    """

    if detrend:
        u = u - np.mean(u)
        v = v - np.mean(v)

    N = len(u)

    # FFT
    uhat = np.fft.rfft(u)
    vhat = np.fft.rfft(v)

    # Wavenumbers
    k = np.fft.rfftfreq(N, d=dx) * 2*np.pi   # rad/m

    # Cospectrum
    Co_uv = np.real(uhat * np.conj(vhat)) / N

    return k, Co_uv


def compute_cospectrum_2d(u, v, dx, dy, detrend=True):
    """
    Compute 2D cospectrum Co_uv(kx, ky)

    Parameters
    ----------
    u, v : 2D arrays (ny, nx)
    dx, dy : grid spacing
    """

    if detrend:
        u = u - np.mean(u)
        v = v - np.mean(v)

    ny, nx = u.shape

    uhat = np.fft.fft2(u)
    vhat = np.fft.fft2(v)

    Co_uv_2d = np.real(uhat * np.conj(vhat)) / (nx * ny)

    kx = np.fft.fftfreq(nx, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2*np.pi

    return kx, ky, Co_uv_2d

def isotropic_cospectrum(Co_uv_2d, kx, ky, nbins=50):
    """
    Radial (isotropic) average of 2D cospectrum.
    """

    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX**2 + KY**2)

    k_flat = k_mag.ravel()
    Co_flat = Co_uv_2d.ravel()

    k_bins = np.linspace(0, k_flat.max(), nbins+1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    Co_iso = np.zeros(nbins)

    for i in range(nbins):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
        if np.any(mask):
            Co_iso[i] = np.mean(Co_flat[mask])

    return k_centers, Co_iso

from scipy.integrate import cumulative_trapezoid

def cumulative_Iuv(k, Co_iso):
    return cumulative_trapezoid(Co_iso, k, initial=0.0)
# %%

fnames=['/Users/simon/Code/CONFIGS/IB09_SZ/IB09_2D_TRC_Dcrit_WMPER_test/',
        '/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_TRC_Dcrit_WMPER_test/',
        '/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/']

plt.figure(dpi=300)
for fname in fnames:
    ds = nc.Dataset(fname+'rip_avg.nc')
    time_fin = ds.variables['scrum_time'][-1]
    h = ds.variables['h'][:]
    ix0 = np.argmin(np.abs(h[0,:]))
    xr = ds.variables['x_rho'][0,:]
    xr = xr - xr[ix0]
    ixSZ = np.argmin(np.abs(xr+80))
    yr = ds.variables['y_rho'][:,0]
    dx= xr[1]-xr[0]
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

    kx, ky, Co_uv = compute_cospectrum_2d(u[-1,:,ixSZ:ix0], tpas[-1,:,ixSZ:ix0], dx, dx, detrend=True)
    k, Co_ut = isotropic_cospectrum(Co_uv, kx, ky,nbins=300)

    kx, ky, Co_uv = compute_cospectrum_2d(u[-1,:,ixSZ:ix0], u[-1,:,ixSZ:ix0], dx, dx, detrend=True)
    k, Co_uu = isotropic_cospectrum(Co_uv, kx, ky,nbins=300)

    kx, ky, Co_uv = compute_cospectrum_2d(tpas[-1,:,ixSZ:ix0], tpas[-1,:,ixSZ:ix0], dx, dx, detrend=True)
    k, Co_tt = isotropic_cospectrum(Co_uv, kx, ky,nbins=300)

    coherence=Co_ut**2/(Co_uu*Co_tt)

    #plt.semilogx(k,coherence,linewidth=0.2,color="gray",alpha=0.8)
    plt.semilogx(k,smooth(coherence,5),linewidth=3)

plt.legend(['2D TRC','3D TRC','3D MR'])
plt.xlabel('Wavenumber (rad/m)')
plt.ylabel('Coherence')
plt.title('Coherence between u and T in the surfzone')
plt.show()
# Step 3: cumulative contribution
#I_uv = cumulative_Iuv(k, Co_iso)
# %%

plt.figure(dpi=300)
for fname in fnames:
    ds = nc.Dataset(fname+'rip_avg.nc')
    time_fin = ds.variables['scrum_time'][-1]
    h = ds.variables['h'][:]
    ix0 = np.argmin(np.abs(h[0,:]))
    xr = ds.variables['x_rho'][0,:]
    xr = xr - xr[ix0]
    ixSZ = np.argmin(np.abs(xr+80))
    yr = ds.variables['y_rho'][:,0]
    dx= xr[1]-xr[0]
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


    coherence=0
    Co_uu=0
    Co_tt=0
    Co_ut=0
    for i in range(150,205):
        kx, ky, Co_uv = compute_cospectrum_2d(u[i,:,ixSZ:ix0], tpas[i,:,ixSZ:ix0], dx, dx, detrend=True)
        k, Co_uttmp = isotropic_cospectrum(Co_uv, kx, ky,nbins=300)
        Co_ut=Co_ut+Co_uttmp
        kx, ky, Co_uv = compute_cospectrum_2d(u[i,:,ixSZ:ix0], u[i,:,ixSZ:ix0], dx, dx, detrend=True)
        k, Co_uutmp = isotropic_cospectrum(Co_uv, kx, ky,nbins=300)
        Co_uu=Co_uu+Co_uutmp
        kx, ky, Co_uv = compute_cospectrum_2d(tpas[i,:,ixSZ:ix0], tpas[i,:,ixSZ:ix0], dx, dx, detrend=True)
        k, Co_tttmp = isotropic_cospectrum(Co_uv, kx, ky,nbins=300)
        Co_tt=Co_tt+Co_tttmp
        
    Co_tt=Co_tt/(205-150)
    Co_uu=Co_uu/(205-150)
    Co_ut=Co_ut/(205-150)
    coherence=Co_ut**2/(Co_uu*Co_tt)
    #plt.semilogx(k,smooth(coherence,1),linewidth=0.5,color="gray",alpha=0.8)
    plt.semilogx(k,smooth(coherence,5),linewidth=3)
    
plt.legend(['2D TRC','3D TRC','3D MR'])
plt.xlabel('Wavenumber (rad/m)')
plt.ylabel('Coherence')
plt.title('Coherence between u and T in the surfzone')
plt.ylim([0,0.5])
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Grid
# -----------------------------

x = u[i,:,ixSZ:ix0]
y = tpas[i,:,ixSZ:ix0]
dx=dx
nx, ny = np.shape(u[i,:,ixSZ:ix0])
X, Y = np.meshgrid(x, y)

# -----------------------------
# 2. Example 2D fields
# -----------------------------
A = np.sin(2*np.pi*0.05*X)
B = np.sin(2*np.pi*0.05*X + np.pi/4)

# -----------------------------
# 3. 2D FFTs
# -----------------------------
Ahat = np.fft.fft2(A)
Bhat = np.fft.fft2(B)

# -----------------------------
# 4. Cross-spectrum P_AB(kx, ky)
# -----------------------------
PAB = Ahat * np.conj(Bhat)

# -----------------------------
# 5. Wavenumber grids
# -----------------------------
kx = 2*np.pi * np.fft.fftfreq(nx, dx)
ky = 2*np.pi * np.fft.fftfreq(ny, dy)
KX, KY = np.meshgrid(kx, ky)
K = np.sqrt(KX**2 + KY**2)

# -----------------------------
# 6. Phase map arg(P_AB)
# -----------------------------
phase2D = np.angle(PAB)

# -----------------------------
# 7. Isotropic (radial) averaging
# -----------------------------
nbins = 50
kbins = np.linspace(0, K.max(), nbins+1)
kcenters = 0.5 * (kbins[:-1] + kbins[1:])

phase_iso = np.zeros(nbins)

for i in range(nbins):
    mask = (K >= kbins[i]) & (K < kbins[i+1])
    phase_iso[i] = np.nanmean(phase2D[mask])

# -----------------------------
# 8. Spatial lag Δx(k)
# -----------------------------
dx_iso = phase_iso / kcenters

# -----------------------------
# 9. Plots
# -----------------------------
plt.figure()
plt.pcolormesh(KX, KY, phase2D, shading='auto')
plt.xlabel("kx [rad/m]")
plt.ylabel("ky [rad/m]")
plt.title("2D Cross-Spectrum Phase arg(P_AB)")
plt.colorbar(label="Phase [rad]")
plt.show()

plt.figure()
plt.plot(kcenters, phase_iso)
plt.xlabel("|k| [rad/m]")
plt.ylabel("Isotropic Phase")
plt.title("Isotropic Cross-Spectrum Phase")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(kcenters, dx_iso)
plt.xlabel("|k| [rad/m]")
plt.ylabel("Spatial Lag Δx(k)")
plt.title("Phase-Derived Spatial Shift")
plt.grid(True)
plt.show()


# %%
kx, ky, Co_uv = compute_cospectrum_2d(u[u.shape[0]-101,:,:], tpas[u.shape[0]-101,:,:], dx, dx, detrend=True)
for i in range(u.shape[0]-100,u.shape[0]):
    kx, ky, Co_uvtmp = compute_cospectrum_2d(u[i,:,:], tpas[i,:,:], dx, dx, detrend=True)
    Co_uv = Co_uv + Co_uvtmp
Co_uv = Co_uv/(100)
k, Co_uutmp = isotropic_cospectrum(Co_uv, kx, ky,nbins=50)
#%%
plt.pcolormesh(kx,ky,Co_uv,shading="gouraud")
plt.colorbar()
plt.plot(-0.3*np.sqrt(-ky-0.1),ky,color="r",linestyle=":",linewidth=3)
plt.clim([-0.1,0.1])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel(r'$k_x$ [rad/m]')
plt.ylabel(r'$k_y$ [rad/m]')
plt.title(r'$k_x = -0.3 \sqrt{-k_y-0.1}$ ?? Mini-rips only here')
plt.show()
# %%
plt.semilogx(k,Co_uutmp)
plt.show()

# %%
kx, ky, Co_uv = compute_cospectrum_2d(u[u.shape[0]-101,:,:], u[u.shape[0]-101,:,:], dx, dx, detrend=True)
for i in range(u.shape[0]-100,u.shape[0]):
    kx, ky, Co_uvtmp = compute_cospectrum_2d(u[i,:,:], u[i,:,:], dx, dx, detrend=True)
    Co_uv = Co_uv + Co_uvtmp
Co_uv = Co_uv/(100)
k, Co_uutmp = isotropic_cospectrum(Co_uv, kx, ky,nbins=50)
#%%
plt.pcolormesh(kx[-352:],ky,Co_uv[:,-352:],shading="gouraud")
plt.colorbar()
#plt.plot(-0.3*np.sqrt(-ky-0.1),ky,color="r",linestyle=":",linewidth=3)
plt.clim([-0.1,0.1])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel(r'$k_x$ [rad/m]')
plt.ylabel(r'$k_y$ [rad/m]')
#plt.title(r'$k_x = -0.3 \sqrt{-k_y-0.1}$ ?? Mini-rips only here')
plt.show()

#%%
kx2D=kx;ky2d=ky;Co2D=Co_uv
#%% Diagnostic Marie

fnames=['/Users/simon/Code/CONFIGS/IB09_SZ/IB09_2D_TRC_Dcrit_WMPER_test/',
        '/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_TRC_Dcrit_WMPER_test/',
        '/Users/simon/Code/CONFIGS/IB09_SZ/IB09_3D_MR_Dcrit/']
names=['2D TRC','3D TRC','3D MR']; n=0

plt.figure(dpi=300,figsize=(6,3))
for fname in fnames:
    ds = nc.Dataset(fname+'rip_avg.nc')
    h = ds.variables['h'][:]
    ix0 = np.argmin(np.abs(h[0,:]))
    xr = ds.variables['x_rho'][0,:]
    xr = xr - xr[ix0]
    ixSZ = np.argmin(np.abs(xr+80))
    yr = ds.variables['y_rho'][:,0]
    dx= xr[1]-xr[0]
    yr = ds.variables['y_rho'][:,0]

    u = u2rho(ds.variables['u'][:,-1,:,:])
    v = v2rho(ds.variables['v'][:,-1,:,:])
    tpas = ds.variables['tpas01'][:,-1,:,:]
    istart = next((i for i, x in enumerate(np.sum(tpas,axis=(1,2))) if x), None)
    
    tpas1 = tpas[istart,:,ixSZ:ix0].ravel()
    tpas1 = tpas1/np.max(tpas1)
    tpas2 = tpas[istart+150,:,ixSZ:ix0].ravel()
    tpas2 = tpas2/np.max(tpas2)
    time_fin = np.round(ds.variables['scrum_time'][istart+170])

    initialtext="Initial"
    plt.plot(np.sort(tpas1),color='gray',linestyle='--')
    plt.plot(np.sort(tpas2),linewidth=3,alpha=0.8,label=names[n])
    n=n+1
plt.xlabel('Number of grid points [-]')
plt.ylabel('Normalized tracer concentration [-]')
plt.title(f'Sorted tracer concentration in the surf zone after {time_fin} s', loc='left')
#plt.legend(['Original tracer injection','2D TRC',"",'3D TRC',"",'3D MR'])
plt.legend()
plt.show()

# %%
