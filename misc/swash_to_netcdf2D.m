function  swash_to_netcdf2D(source,dir,var)

    data = load([source,dir,var,'.mat']);
    if var=='zeta'
        swashname='Watlev';
    end

    % Get all variable names
    varNames = fieldnames(data);
    
    % Initialize containers
    times = [];
    parsedData = struct();
    
    % Check time length
    for i = 1:length(varNames)
        varName = varNames{i};
        tokens = regexp(varName, [swashname,'_([0-9]{6})_([0-9]{3})$'], 'tokens');
        if ~isempty(tokens)
            hhmmss = tokens{1}{1}; % '000959'
            subsec = tokens{1}{2}; % '999'
    
            % Convert hhmmss into seconds
            h = str2double(hhmmss(1:2));
            m = str2double(hhmmss(3:4));
            s = str2double(hhmmss(5:6));
            micro = str2double(subsec); % assuming milliseconds
            t_seconds = h*3600 + m*60 + s + micro/1000; % or /1e6 if microseconds
            times(end+1) = t_seconds;
        end
    end
    
    % Get unique sorted times and layers
    [timesSorted, ~] = sort(unique(times));
    
    % Parse variable names
    for i = 1:length(varNames)
        varName = varNames{i};
        tokens = regexp(varName, [swashname,'_([0-9]{6})_([0-9]{3})$'], 'tokens');
        if ~isempty(tokens)
            hhmmss = tokens{1}{1}; % '000959'
            subsec = tokens{1}{2}; % '999'
    
            % Convert hhmmss into seconds
            h = str2double(hhmmss(1:2));
            m = str2double(hhmmss(3:4));
            s = str2double(hhmmss(5:6));
            micro = str2double(subsec); % assuming milliseconds
            t_seconds = h*3600 + m*60 + s + micro/1000; % or /1e6 if microseconds
            [~,idt]=min(abs(t_seconds-timesSorted));
    
            parsedData.time(idt).name = varName;
            parsedData.time(idt).value = data.(varName);
        end
    end
    
    % Assume data size from first variable
    sampleVar = data.(varNames{1});
    [xSize, ySize] = size(sampleVar);
    
    % Create NetCDF file
    ncFileName = [source,dir,'swash_',var,'.nc'];
    ncid = netcdf.create(ncFileName, 'NETCDF4');
    
    % Define dimensions
    dimidX = netcdf.defDim(ncid, 'x', xSize);
    dimidY = netcdf.defDim(ncid, 'y', ySize);
    dimidT = netcdf.defDim(ncid, 'time', length(timesSorted));
    
    % Define variables
    varid = netcdf.defVar(ncid, var, 'double', [dimidX dimidY dimidT]);
    timeVarID = netcdf.defVar(ncid, 'time', 'double', dimidT);
    
    netcdf.endDef(ncid);
    
    % Write time and layer values
    netcdf.putVar(ncid, timeVarID, timesSorted);
    
    % Write main variable
    n=1;
    for tIdx = 1:length(timesSorted)
        t = timesSorted(tIdx);
        try
            varData = parsedData.time(n).value;
            netcdf.putVar(ncid, varid, [0, 0, tIdx-1], [xSize, ySize, 1], varData);
        catch
            % If missing, fill with NaNs
            netcdf.putVar(ncid, varid, [0, 0, tIdx-1], [xSize, ySize, 1], nan(xSize, ySize));
        end
        n=n+1;
    end
    
    netcdf.close(ncid);
    disp(['NetCDF file created: ', ncFileName]);
end