import os
import re
import sys

def extract_tstepCROCO(directory):
    directory = os.path.join('/scratch/users/treillou/',directory)
    param_file = None

    # Search for croco.in in the directory
    for root, _, files in os.walk(directory):
        if "croco.in" in files:
            param_file = os.path.join(root, "croco.in")
            break

    if not param_file:
        print("Error: croco.in file not found in the specified directory.")
        return None

    # Find the line containing "dt[sec]"
    with open(param_file, "r") as file:
        lines = file.readlines()  # Read all lines
    dt_sec = None

    for i, line in enumerate(lines):
        if "dt[sec]" in line:
            # The next line contains the values
            if i + 1 < len(lines):
                values = re.findall(r"[-+]?\d*\.\d+|\d+", lines[i + 1])  # Extract numbers
                if len(values) > 1:  # Ensure there is a second value (dt[sec])
                    dt_sec = float(values[1])  # Second value is dt[sec]
            break
    
    if dt_sec is None:
        print("Could not find dt[sec].")
    return dt_sec
        
def extract_tstepSWASH(directory):
    directory = os.path.join('/scratch/users/treillou/',directory)
    param_file = None

    # Search for croco.in in the directory
    for root, _, files in os.walk(directory):
        if "PRINT" in files:
            param_file = os.path.join(root, "PRINT")
            break

    if not param_file:
        print("Error: PRINT file not found in the specified directory.")
        return None

    with open(param_file, "r") as file:
        for line in file:
            match  = re.search(r"New\s+time\s+step:\s+(\d+\.\d+)", line)
            match2 = re.search(r"COMPUTE\s+\d+\.\d+\s+(\d+\.\d+)", line)
            
            if match:
                target_value = float(match.group(1))
                break
            elif match2:
                target_value = float(match2.group(1))

    dt_sec = target_value
    if dt_sec is not None:
        return dt_sec
    else:
        print("Could not find dt[sec].")
        return None

def extract_tstep(directory,model="CROCO"):
    if model=="CROCO":
        return extract_tstepCROCO(directory)
    elif model=="SWASH":
        return extract_tstepSWASH(directory)
    else:
        print("Error: Invalid model")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_tstep.py <directory> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 2:
        directory = sys.argv[1]
        extract_tstep(directory)
    elif len(sys.argv) ==3:
        directory = sys.argv[1]
        model = sys.argv[2]
        extract_tstep(directory,model)
    else:
        print("Usage: python extract_tstep.py <directory> OPT:<model>")
        sys.exit(1)
    
    