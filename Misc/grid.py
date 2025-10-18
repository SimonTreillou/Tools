import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime



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

def getcroco_stations(x_stations,y_stations,dx):
    positions_x = np.array((x_stations) / dx).astype(int)
    positions_y = np.array(y_stations).astype(int)
    for i in range(len(positions_x)):
        for j in range(len(positions_y)):
            t1=positions_x[i]
            t2=positions_y[j]
            print("    0.0    {:03}      {:03}     10     0       0".format(t1,t2))
            