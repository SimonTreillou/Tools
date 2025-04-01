#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from netCDF4 import Dataset
import datetime

def make_crocogrid(grdname, depth, dx, dy, Lx, Ly):
    depth = depth.T
    Lm = int((Lx/dx) - 1) 
    Mm = int((Ly/dy) - 1) 
    Length_Domain = Lx
    Length_XI = Lm * dx
    Length_ETA = Ly
    Lp = Lm + 2
    Mp = Mm + 2
    xr = np.empty((Mp, Lp))
    yr = np.empty((Mp, Lp))
    # filling in
    for jj in range(0, Mp):
        for ii in range(0, Lp):
            xr[jj, ii] = dx * ((ii - 1) + 0.5)  # - Length_Domain
            yr[jj, ii] = dy * ((jj - 1) + 0.5)
    CROCO_title = f'Bathymetry'
    grid_file_path=grdname
    xu, xv, xp = rho2uvp(xr)
    yu, yv, yp = rho2uvp(yr)
    # Create the grid file
    #print('Create the grid file')

    M, L = yp.shape
    print(f'Grid dimensions: {L - 1} x {M - 1}')
    create_grid(L, M, grid_file_path, CROCO_title)

    #  Compute the metrics
    pm = 1. / dx
    pn = 1. / dy
    dndx = 0.
    dmde = 0.
    dxmax = np.max(np.max(dx / 1000))
    dxmin = np.min(np.min(dx / 1000))
    dymax = np.max(np.max(dy / 1000))
    dymin = np.min(np.min(dy / 1000))

    #  Angle between XI-axis and the direction
    #  to the EAST at RHO-points [radians].
    rotation_angle = 0
    #  Coriolis parameter
    f = np.ones(xr.shape) * 0.

    ############################################################################
    # Fill the grid file
    nc = Dataset(grid_file_path, 'r+')

    nc['pm'][:] = pm
    nc['pn'][:] = pn
    nc['dndx'][:] = dndx
    nc['dmde'][:] = dmde
    nc['x_u'][:] = xu
    nc['y_u'][:] = yu
    nc['x_v'][:] = xv
    nc['y_v'][:] = yv
    nc['x_rho'][:] = xr
    nc['y_rho'][:] = yr
    nc['x_psi'][:] = xp
    nc['y_psi'][:] = yp
    nc['angle'][:] = rotation_angle
    nc['f'][:] = f
    nc['h'][:] = depth
    nc['spherical'][:] = 'F'  # this does not work

    # Compute the mask
    maskr = np.ones(depth.shape)
    masku, maskv, maskp = uvp_mask(maskr)

    #  Write it down


    #nc['mask_u'][:] = masku
    #nc['mask_v'][:] = maskv
    #nc['mask_psi'][:] = maskp
    #nc['mask_rho'][:] = maskr

    nc.close()
    
    
    


    


def rho2uvp(rfield):
    Mp, Lp = np.shape(rfield)  # matlab use size for "shape" in python
    M = Mp - 1
    L = Lp - 1

    vfield = 0.5 * (rfield[0:M, :] + rfield[1:Mp, :])
    ufield = 0.5 * (rfield[:, 0:L] + rfield[:, 1:Lp])
    pfield = 0.5 * (ufield[0:M, :] + ufield[1:Mp, :])

    return ufield, vfield, pfield
    
def uvp_mask(rfield):
    Mp, Lp = np.shape(rfield)
    M = Mp - 1
    L = Lp - 1

    vfield = rfield[0:M, :] * rfield[1:Mp, :]
    ufield = rfield[:, 0:L] * rfield[:, 1:Lp]
    pfield = ufield[0:M, :] * ufield[1:Mp, :]
    return ufield, vfield, pfield

def get_angle(xu, yu):
    # Calculate the angle; for simplicity, this is set to zero everywhere
    return np.zeros_like(xu)

def create_grid(l, m, grid_name, title):
    """

    :param l: total number of psi points in x direction
    :param m: total number of psi points in y direction
    :param grid_name: name of the grid file
    :param title: title in the netcdf file
    :return: nc array
    """
    # L = 800
    # M = 1
    # grid_name = "test.nc"
    # title = "NT"

    Lp = l + 1
    Mp = m + 1
    # create nc file
    nw = Dataset(grid_name, "w", format="NETCDF4")

    ############################################################################
    ############################################################################
    # create dimensions
    nw.createDimension("xi_u", l)
    nw.createDimension("eta_u", Mp)
    nw.createDimension("xi_v", Lp)
    nw.createDimension("eta_v", m)
    nw.createDimension("xi_rho", Lp)
    nw.createDimension("eta_rho", Mp)
    nw.createDimension("xi_psi", l)
    nw.createDimension("eta_psi", m)
    nw.createDimension("one", 1)
    nw.createDimension("two", 2)
    nw.createDimension("four", 4)
    nw.createDimension("bath", 1)

    ############################################################################
    ############################################################################
    # create variables and attributes
    # xl
    nw.createVariable("xl", datatype="f8", dimensions=("one"))
    nw["xl"].long_name = 'domain length in the XI-direction'
    nw["xl"].units = "meters"
    # el
    nw.createVariable("el", datatype="f8", dimensions=("one"))
    nw["el"].long_name = 'domain length in the ETA-direction'
    nw["el"].units = "meters"
    # depthmin
    nw.createVariable('depthmin', datatype="f8", dimensions=("one"))
    nw['depthmin'].long_name = 'Shallow bathymetry clipping depth'
    nw['depthmin'].units = "meters"
    # depthmax
    nw.createVariable('depthmax', datatype="f8", dimensions=("one"))
    nw['depthmax'].long_name = 'Deep bathymetry clipping depth'
    nw['depthmax'].units = "meters"
    # spherical
    nw.createVariable('spherical', datatype="S1", dimensions=("one"))  # set to S1 var type for this and only this case
    nw['spherical'].long_name = 'Grid type logical switch'
    nw['spherical'].units = "meters"
    ############################################################################
    # angle
    nw.createVariable('angle', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['angle'].long_name = 'angle between xi axis and east'
    nw['angle'].units = 'radian'
    # h
    nw.createVariable('h', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['h'].long_name = 'Final bathymetry at RHO-points'
    nw['h'].units = 'meter'
    # hraw
    nw.createVariable('hraw', datatype="f8", dimensions=('bath', 'eta_rho', 'xi_rho'))
    nw['hraw'].long_name = 'Working bathymetry at RHO-points'
    nw['hraw'].units = 'meter'
    # alpha
    nw.createVariable('alpha', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['alpha'].long_name = 'Weights between coarse and fine grids at RHO-points'
    # f
    nw.createVariable('f', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['f'].long_name = 'Coriolis parameter at RHO-points'
    nw['f'].units = 'second-1'
    # pm
    nw.createVariable('pm', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['pm'].long_name = 'curvilinear coordinate metric in XI'
    nw['pm'].units = 'meter-1'
    # pn
    nw.createVariable('pn', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['pn'].long_name = 'curvilinear coordinate metric in ETA'
    nw['pn'].units = 'meter-1'
    # dndx
    nw.createVariable('dndx', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['dndx'].long_name = 'xi derivative of inverse metric factor pn'
    nw['dndx'].units = 'meter'
    # dmde
    nw.createVariable('dmde', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['dmde'].long_name = 'eta derivative of inverse metric factor pm'
    nw['dmde'].units = 'meter'
    # x_rho
    nw.createVariable('x_rho', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['x_rho'].long_name = 'x location of RHO-points'
    nw['x_rho'].units = 'meter'
    # x_u
    nw.createVariable('x_u', datatype="f8", dimensions=('eta_u', 'xi_u'))
    nw['x_u'].long_name = 'x location of U-points'
    nw['x_u'].units = 'meter'
    # x_v
    nw.createVariable('x_v', datatype="f8", dimensions=('eta_v', 'xi_v'))
    nw['x_v'].long_name = 'x location of V-points'
    nw['x_v'].units = 'meter'
    # x_psi
    nw.createVariable('x_psi', datatype="f8", dimensions=('eta_psi', 'xi_psi'))
    nw['x_psi'].long_name = 'x location of PSI-points'
    nw['x_psi'].units = 'meter'
    # y_rho
    nw.createVariable('y_rho', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['y_rho'].long_name = 'y location of RHO-points'
    nw['y_rho'].units = 'meter'
    # y_u
    nw.createVariable('y_u', datatype="f8", dimensions=('eta_u', 'xi_u'))
    nw['y_u'].long_name = 'y location of U-points'
    nw['y_u'].units = 'meter'
    # y_v
    nw.createVariable('y_v', datatype="f8", dimensions=('eta_v', 'xi_v'))
    nw['y_v'].long_name = 'y location of V-points'
    nw['y_v'].units = 'meter'
    # y_psi
    nw.createVariable('y_psi', datatype="f8", dimensions=('eta_psi', 'xi_psi'))
    nw['y_psi'].long_name = 'y location of PSI-points'
    nw['y_psi'].units = 'meter'
    # lon_rho
    nw.createVariable('lon_rho', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['lon_rho'].long_name = 'longitude of RHO-points'
    nw['lon_rho'].units = 'degree_east'
    # lon_u
    nw.createVariable('lon_u', datatype="f8", dimensions=('eta_u', 'xi_u'))
    nw['lon_u'].long_name = 'longitude of U-points'
    nw['lon_u'].units = 'degree_east'
    # lon_v
    nw.createVariable('lon_v', datatype="f8", dimensions=('eta_v', 'xi_v'))
    nw['lon_v'].long_name = 'longitude of V-points'
    nw['lon_v'].units = 'degree_east'
    # lon_psi
    nw.createVariable('lon_psi', datatype="f8", dimensions=('eta_psi', 'xi_psi'))
    nw['lon_psi'].long_name = 'longitude of PSI-points'
    nw['lon_psi'].units = 'degree_east'
    # lat_rho
    nw.createVariable('lat_rho', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['lat_rho'].long_name = 'latitude of RHO-points'
    nw['lat_rho'].units = 'degree_north'
    # lat_u
    nw.createVariable('lat_u', datatype="f8", dimensions=('eta_u', 'xi_u'))
    nw['lat_u'].long_name = 'latitude of U-points'
    nw['lat_u'].units = 'degree_north'
    # lat_v
    nw.createVariable('lat_v', datatype="f8", dimensions=('eta_v', 'xi_v'))
    nw['lat_v'].long_name = 'latitude of V-points'
    nw['lat_v'].units = 'degree_north'
    # lat_psi
    nw.createVariable('lat_psi', datatype="f8", dimensions=('eta_psi', 'xi_psi'))
    nw['lat_psi'].long_name = 'latitude of PSI-points'
    nw['lat_psi'].units = 'degree_north'
    # mask_rho
    nw.createVariable('mask_rho', datatype="f8", dimensions=('eta_rho', 'xi_rho'))
    nw['mask_rho'].long_name = 'mask on RHO-points'
    nw['mask_rho'].option_0 = 'land'
    nw['mask_rho'].option_1 = 'water'
    # mask_u
    nw.createVariable('mask_u', datatype="f8", dimensions=('eta_u', 'xi_u'))
    nw['mask_u'].long_name = 'mask on U-points'
    nw['mask_u'].option_0 = 'land'
    nw['mask_u'].option_1 = 'water'
    # mask_v
    nw.createVariable('mask_v', datatype="f8", dimensions=('eta_v', 'xi_v'))
    nw['mask_v'].long_name = 'mask on V-points'
    nw['mask_v'].option_0 = 'land'
    nw['mask_v'].option_1 = 'water'
    # mask_psi
    nw.createVariable('mask_psi', datatype="f8", dimensions=('eta_psi', 'xi_psi'))
    nw['mask_psi'].long_name = 'mask on PSI-points'
    nw['mask_psi'].option_0 = 'land'
    nw['mask_psi'].option_1 = 'water'

    ############################################################################
    ############################################################################
    # Create global attributes
    nw.title = title
    nw.date = datetime.datetime.now().isoformat()
    nw.type = 'CROCO grid file'
    nw.close()
    return nw
        
def read_grid2Dcsv(filename,wlevel=0.0):
    df = pd.read_csv(filename, header=None)
    x = df[0]
    h = -(df[1] - wlevel)
    return x,h

def planar3D(dx,Lx,dy,Ly,beta,offset):
    x = np.arange(0, Lx + dx, dx)
    y = np.arange(0, Ly + dy, dy)
    x, y = np.meshgrid(x, y)
    h = beta * (x - offset)
    return x,y,h

def create_grid_uniform_from_profile(x,h,dx,Ly):
    f = interpolate.interp1d(x, h)
    x_min = np.min(x)
    x_max = np.max(x)
    x_reg = np.arange(x_min, x_max, dx)
    h_reg = f(x_reg)
    x_reg = x_reg[1:-1]
    h_reg = np.convolve(h_reg, np.ones(3) / 3, mode='same')[1:-1]

    y = np.tile(np.arange(0, Ly, dx), (x_reg.shape[0], 1))
    x = np.tile(x_reg, (y.shape[1], 1)).T
    h = np.tile(h_reg, (y.shape[1], 1)).T
    return x,y,h

def define_dim(x,y):
    Lx=np.max(x)-np.min(x)
    Ly=np.max(y)-np.min(y)
    Mx=np.size(x,0)
    My=np.size(y,1)
    dx=x[1,0]-x[0,0]
    dy=y[0,1]-y[0,0]
    return Lx,Ly,Mx,My,dx,dy


def make_grid(repo,model,args):
    if args["Option"]=="2Ddata":
        x_csv,h_csv = read_grid2Dcsv(args["filename"],args["wlevel"])
        x,y,h = create_grid_uniform_from_profile(x_csv,h_csv,args["dx"],args["Ly"])
        Lx,Ly,Mx,My,dx,dy = define_dim(x,y)
    elif args["Option"]=="3DPlanar":
        x,y,h=planar3D(args["dx"],args["Lx"],args["dy"],args["Ly"],args["beta"],args["offset"])
        Lx,Ly,Mx,My,dx,dy = define_dim(x,y)
    elif args["Option"]=="3Ddata":
        print("TODO")
    else:
        print("TODO: 3D not planar")

    if model=="SWASH":
        # Save bathy for SWASH
        h=h.T
        with open(repo+'/bathy.bot', 'w') as file:
            # Loop through each row of the depth array
            for row in h:
                # Write each value with a width of 1.4 characters
                file.write(''.join(f"   {val:1.4f}" for val in row))
                # Newline at the end of each row
                file.write('\n')
    elif model=="CROCO":
        make_crocogrid("grid.nc", h, dx, dy, Lx, Ly)
    else:
        print("Unknown model")
    return Lx,Ly,Mx,My,dx,dy
        

      
if __name__ == "__main__":
    args_3DPlanar={"Option": "3DPlanar",
      "wlevel": 0.00,
      "offset": 20.0,
      "Lx": 150.0,
      "dx": 1.0,
      "Ly": 100.0,
      "dy": 1.0,
      "beta": 0.02
     }  
    args_2Ddata={"Option": "2Ddata",
          "wlevel": 0.75,
          "filename": 'vanderWerf.csv',
          "dx": 0.1,
          "Ly":100.0
         }
    if len(sys.argv) < 3:
        print("Usage: python make_grid.py <repository> <model> <filename> (2Ddata as args default, possibility to give filename)")
        sys.exit(1)
    elif len(sys.argv) == 3:
        repo = sys.argv[1]
        model = sys.argv[2]
        make_grid(repo,model,args_2Ddata)
    elif len(sys.argv) == 4:
        repo = sys.argv[1]
        model = sys.argv[2]
        filename = sys.argv[3]
        args_2Ddata["filename"]=filename
        make_grid(repo,model,args_2Ddata)
    else:
        print("Usage: python make_grid.py <repository> <model> <filename> (2Ddata as args default, possibility to give filename)")
        sys.exit(1)