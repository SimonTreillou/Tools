import numpy as np
import scipy.io
import re
from netCDF4 import Dataset
import os

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# swash_to_netcdf2D: Converts 2D SWASH output from .mat to NetCDF.
# swash_to_netcdf3D_slice: Converts a single vertical layer of 3D SWASH output from .mat to NetCDF.
# swash_to_netcdf3D: Converts 3D SWASH output from .mat to NetCDF.
# swash_to_netcdfdomain: Converts SWASH domain data from .mat to NetCDF.
# ---------------------------------------------------------------

def swash_to_netcdf2D(source, dir_path, var):
    """
    Converts 2D SWASH output (e.g., water level or vorticity) from a .mat file to NetCDF format.

    Parameters:
        source (str): Base directory path.
        dir_path (str): Subdirectory path containing the .mat file.
        var (str): Variable name, should be 'zeta' or 'vort'.

    The function expects the .mat file to contain variables named like 'Watlev_HHMMSS_SSS' or 'vort_HHMMSS_SSS',
    where HHMMSS is the time and SSS is the subsecond.
    """
    filepath = os.path.join(source, dir_path, f"{var}.mat")
    data = scipy.io.loadmat(filepath)

    if var == 'zeta':
        swashname = 'Watlev'
    elif var == 'vort':
        swashname = 'vort'
    else:
        raise ValueError("Only 'zeta' or 'vort' is supported currently.")

    times = []
    time_map = {}

    # Parse variable names and extract time info
    for varName in data:
        match = re.match(rf"{swashname}_([0-9]{{6}})_([0-9]{{3}})$", varName)
        if match:
            hhmmss = match.group(1)
            subsec = match.group(2)

            h = int(hhmmss[0:2])
            m = int(hhmmss[2:4])
            s = int(hhmmss[4:6])
            micro = int(subsec)

            t_seconds = h * 3600 + m * 60 + s + micro / 1000.0
            times.append(t_seconds)
            time_map[t_seconds] = varName

    timesSorted = sorted(set(times))
    time_index = {t: i for i, t in enumerate(timesSorted)}

    # Get dimensions from a valid 2D numeric variable
    sample_data = None
    for v in data.values():
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            sample_data = v
            break

    if sample_data is None:
        raise ValueError("No valid 2D numerical variable found in the .mat file.")

    ySize, xSize = sample_data.shape

    # Initialize output array
    output_data = np.full((xSize, ySize, len(timesSorted)), np.nan)

    # Fill the output array with data for each time step
    for t, varName in time_map.items():
        i = time_index[t]
        try:
            output_data[:, :, i] = data[varName].T
        except KeyError:
            continue

    # Extract grid and bathymetry if available
    if hasattr(data, 'Xp'):
        Xp = data['Xp'][:]
    if hasattr(data, 'Botlev'):
        h = data['Botlev'][:]

    # Write to NetCDF
    ncFileName = os.path.join(source, dir_path, f"swash_{var}.nc")
    with Dataset(ncFileName, 'w', format='NETCDF4') as ncfile:
        ncfile.createDimension('x', xSize)
        ncfile.createDimension('y', ySize)
        ncfile.createDimension('time', len(timesSorted))

        var_nc = ncfile.createVariable(var, 'f8', ('x', 'y', 'time'))
        x_nc = ncfile.createVariable('Xp', 'f8', ('x', 'y'))
        h_nc = ncfile.createVariable('Botlev', 'f8', ('x', 'y'))
        time_nc = ncfile.createVariable('time', 'f8', ('time',))

        var_nc[:, :, :] = output_data
        if hasattr(data, 'Xp'):
            x_nc[:] = Xp
        else:
            x_nc[:] = np.tile(np.linspace(0, xSize, xSize), [1, ySize])
        if hasattr(data, 'Botlev'):
            h_nc[:] = h
        else:
            h_nc[:] = np.zeros_like(x_nc)
        time_nc[:] = timesSorted

    print(f"NetCDF file created: {ncFileName}")
    
def swash_to_netcdf3D_slice(source, dir_path, var, K):
    """
    Converts a single vertical layer (slice) of 3D SWASH output from a .mat file to NetCDF format.

    Parameters:
        source (str): Base directory path.
        dir_path (str): Subdirectory path containing the .mat file.
        var (str): Variable name, should be 'u' or 'v'.
        K (int): Vertical layer index to extract.

    The function expects the .mat file to contain variables named like 'Vksi_kX_HHMMSS_SSS' or 'Veta_kX_HHMMSS_SSS',
    where X is the vertical layer, HHMMSS is the time, and SSS is the subsecond.
    Only the specified layer K will be extracted and written to NetCDF.
    """
    print('Vertical layer: ' + str(K))
    filepath = os.path.join(source, dir_path, f"{var}.mat")
    data = scipy.io.loadmat(filepath)

    if var == 'u':
        swashname = 'Vksi'
    elif var == 'v':
        swashname = 'Veta'
    else:
        raise ValueError("Unsupported variable. Use 'u' or 'v'.")

    times = []
    layers = set()
    time_map = {}

    # Parse all variable names and extract those matching the specified layer K
    for varName in data:
        match = re.match(rf"{swashname}_k(\d+)_([0-9]{{6}})_([0-9]{{3}})$", varName)
        if match:
            k = int(match.group(1))
            hhmmss = match.group(2)
            subsec = match.group(3)

            h = int(hhmmss[0:2])
            m = int(hhmmss[2:4])
            s = int(hhmmss[4:6])
            micro = int(subsec)

            t_seconds = h * 3600 + m * 60 + s + micro / 1000.0
            times.append(t_seconds)

            if k == K:
                time_map.setdefault(t_seconds, []).append((k, varName))
            layers.add(k)

    timesSorted = sorted(set(times))
    layersSorted = sorted(layers)

    time_index = {t: i for i, t in enumerate(timesSorted)}
    layer_index = {k: i for i, k in enumerate(layersSorted)}

    # Get dimensions from a sample 2D variable
    sample_data = None
    for v in data.values():
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            sample_data = v
            break

    if sample_data is None:
        raise ValueError("No valid 2D numerical variable found in .mat file.")

    ySize, xSize = sample_data.shape

    # Initialize data array with NaNs: (x, y, 1, time)
    output_data = np.full((xSize, ySize, 1, len(timesSorted)), np.nan)

    # Fill the data array for the specified layer K
    for t in timesSorted:
        for k, varName in time_map[t]:
            try:
                value = data[varName]
                i = layer_index[K]
                j = time_index[t]
                output_data[:, :, 0, j] = value.T
            except KeyError:
                continue

    # Extract grid and bathymetry if available
    if hasattr(data, 'Xp'):
        Xp = data['Xp'][:]
    if hasattr(data, 'Botlev'):
        h = data['Botlev'][:]

    # Write to NetCDF
    ncFileName = os.path.join(source, dir_path, f"swash_{var}_{K}.nc")
    with Dataset(ncFileName, 'w', format='NETCDF4') as ncfile:
        ncfile.createDimension('x', xSize)
        ncfile.createDimension('y', ySize)
        ncfile.createDimension('time', len(timesSorted))

        var_nc = ncfile.createVariable(var, 'f8', ('x', 'y', 'time'))
        x_nc = ncfile.createVariable('Xp', 'f8', ('x', 'y'))
        h_nc = ncfile.createVariable('Botlev', 'f8', ('x', 'y'))
        time_nc = ncfile.createVariable('time', 'f8', ('time',))

        var_nc[:, :, :] = output_data[:, :, 0, :]
        if hasattr(data, 'Xp'):
            x_nc[:] = Xp
        else:
            x_nc[:] = np.tile(np.linspace(0, xSize, xSize), [1, ySize])
        if hasattr(data, 'Botlev'):
            h_nc[:] = h
        else:
            h_nc[:] = np.zeros_like(x_nc)
        time_nc[:] = timesSorted

    print(f"NetCDF file created: {ncFileName}")

def swash_to_netcdf3D(source, dir_path, var):
        """
        Converts 3D SWASH output (e.g., velocity fields) from a .mat file to NetCDF format.

        Parameters:
            source (str): Base directory path.
            dir_path (str): Subdirectory path containing the .mat file.
            var (str): Variable name, should be 'u' or 'v'.

        The function expects the .mat file to contain variables named like 'Vksi_kX_HHMMSS_SSS' or 'Veta_kX_HHMMSS_SSS',
        where X is the vertical layer, HHMMSS is the time, and SSS is the subsecond.
        """
        filepath = os.path.join(source, dir_path, f"{var}.mat")
        data = scipy.io.loadmat(filepath)

        if var == 'u':
            swashname = 'Vksi'
        elif var == 'v':
            swashname = 'Veta'
        else:
            raise ValueError("Unsupported variable. Use 'u' or 'v'.")

        times = []
        layers = set()
        time_map = {}

        # Parse all variable names to extract layer and time information
        for varName in data:
            match = re.match(rf"{swashname}_k(\d+)_([0-9]{{6}})_([0-9]{{3}})$", varName)
            if match:
                k = int(match.group(1))
                hhmmss = match.group(2)
                subsec = match.group(3)

                h = int(hhmmss[0:2])
                m = int(hhmmss[2:4])
                s = int(hhmmss[4:6])
                micro = int(subsec)

                t_seconds = h * 3600 + m * 60 + s + micro / 1000.0
                times.append(t_seconds)

                time_map.setdefault(t_seconds, []).append((k, varName))
                layers.add(k)

        timesSorted = sorted(set(times))
        layersSorted = sorted(layers)

        time_index = {t: i for i, t in enumerate(timesSorted)}
        layer_index = {k: i for i, k in enumerate(layersSorted)}

        # Get dimensions from a sample 2D variable
        sample_data = None
        for v in data.values():
            if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
                sample_data = v
                break

        if sample_data is None:
            raise ValueError("No valid 2D numerical variable found in .mat file.")

        ySize, xSize = sample_data.shape

        # Initialize data array with NaNs: (x, y, layer, time)
        output_data = np.full((xSize, ySize, len(layersSorted), len(timesSorted)), np.nan)

        # Fill the data array
        for t in timesSorted:
            for k, varName in time_map[t]:
                try:
                    value = data[varName]
                    i = layer_index[k]
                    j = time_index[t]
                    output_data[:, :, i, j] = value.T
                except KeyError:
                    continue

        # Extract grid and bathymetry if available
        if hasattr(data, 'Xp'):
            Xp = data['Xp'][:]
        if hasattr(data, 'Botlev'):
            h = data['Botlev'][:]

        # Write to NetCDF
        ncFileName = os.path.join(source, dir_path, f"swash_{var}.nc")
        with Dataset(ncFileName, 'w', format='NETCDF4') as ncfile:
            ncfile.createDimension('x', xSize)
            ncfile.createDimension('y', ySize)
            ncfile.createDimension('layer', len(layersSorted))
            ncfile.createDimension('time', len(timesSorted))

            var_nc = ncfile.createVariable(var, 'f8', ('x', 'y', 'layer', 'time'))
            x_nc = ncfile.createVariable('Xp', 'f8', ('x', 'y'))
            h_nc = ncfile.createVariable('Botlev', 'f8', ('x', 'y'))
            time_nc = ncfile.createVariable('time', 'f8', ('time',))
            layer_nc = ncfile.createVariable('layer', 'i4', ('layer',))

            var_nc[:, :, :, :] = output_data
            if hasattr(data, 'Xp'):
                x_nc[:] = Xp
            else:
                x_nc[:] = np.tile(np.linspace(0, xSize, xSize), [1, ySize])
            if hasattr(data, 'Botlev'):
                h_nc[:] = h
            else:
                h_nc[:] = np.zeros_like(x_nc)
            time_nc[:] = timesSorted
            layer_nc[:] = layersSorted

        print(f"NetCDF file created: {ncFileName}")
    
def swash_to_netcdfdomain(source, dir_path, var):
    """
    Converts SWASH domain data (e.g., bathymetry and grid) from a .mat file to NetCDF format.

    Parameters:
        source (str): Base directory path.
        dir_path (str): Subdirectory path containing the .mat file.
        var (str): Variable name, should be 'domain' for this function.

    The function expects the .mat file to contain 'Xp' (grid) and 'Botlev' (bathymetry) arrays.
    """
    filepath = os.path.join(source, dir_path, f"{var}.mat")
    data = scipy.io.loadmat(filepath)

    if var == 'domain':
        swashname = 'Botlev'
    else:
        raise ValueError("Only 'domain' is supported currently.")

    Xp = data['Xp'][:]
    h = data['Botlev'][:]
    print(h.shape)
    xSize, ySize = h.shape

    # Write to NetCDF
    ncFileName = os.path.join(source, dir_path, f"swash_{var}.nc")
    with Dataset(ncFileName, 'w', format='NETCDF4') as ncfile:
        ncfile.createDimension('x', xSize)
        ncfile.createDimension('y', ySize)

        x_nc = ncfile.createVariable('Xp', 'f8', ('x', 'y'))
        h_nc = ncfile.createVariable('Botlev', 'f8', ('x', 'y'))

        x_nc[:] = Xp
        h_nc[:] = h

    print(f"NetCDF file created: {ncFileName}")