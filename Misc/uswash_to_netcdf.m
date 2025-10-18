clear all

% Load .mat file
source = '/scratch/users/treillou/';
dir = 'DUCKSWASH/DUCKSWASH_5/';
var = 'u';

swash_to_netcdf3D(source,dir,var)


var = 'zeta';

swash_to_netcdf2D(source,dir,var)