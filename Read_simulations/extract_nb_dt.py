import os
import re
import sys
sys.path.append('/home/users/treillou/Tools/Read_simulations')
from extract_tstep import extract_tstep


def find_last_number_in_matching_line(filepath):
    last_number = None
    # Pattern to match lines like: integer float float float float float integer
    pattern = re.compile(r'^\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)\s+\d+\s*$')

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            match = pattern.match(line)
            if match:
                last_number = int(match.group(1))

    return last_number
 
def extract_nb_dtCROCO(directory):
    directory = os.path.join('/scratch/users/treillou/',directory)
    param_file = None
    nbdt=None

    # Search for croco.in in the directory
    for root, _, files in os.walk(directory):
        if "croco.out" in files:
            param_file = os.path.join(root, "croco.out")
            break

    if not param_file:
        print("Error: croco.out file not found in the specified directory.")
        return None

    with open(param_file) as myfile:
        pattern = re.compile(r'^\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)\s+\d+\s*$')
        with open(param_file, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    nbdt=int(match.group(1))
    return nbdt
        
def extract_nb_dtSWASH(directory):
    directory = os.path.join('/scratch/users/treillou/',directory)
    param_file = None
    nbdt=None

    # Search for croco.in in the directory
    for root, _, files in os.walk(directory):
        if "PRINT" in files:
            param_file = os.path.join(root, "PRINT")
            break

    if not param_file:
        print("Error: PRINT file not found in the specified directory.")
        return None

    with open(param_file) as myfile:
        pattern = re.compile(r'Time of simulation\s+->\s+\d+\.\d+\s+in sec:\s+(\d+\.\d+)')
        with open(param_file, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    time_sim=float(match.group(1))
        dt = extract_tstep(directory,"SWASH")
        nbdt = int(time_sim/dt)
    return nbdt

def extract_nb_dt(directory,model="CROCO"):
    if model=="CROCO":
        return extract_nb_dtCROCO(directory)
    elif model=="SWASH":
        return extract_nb_dtSWASH(directory)
    else:
        print("Error: Invalid model")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_nb_dt.py <directory> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 2:
        directory = sys.argv[1]
        extract_nb_dt(directory)
    elif len(sys.argv) ==3:
        directory = sys.argv[1]
        model = sys.argv[2]
        extract_nb_dt(directory,model)
    else:
        print("Usage: python extract_nb_dt.py <directory> OPT:<model>")
        sys.exit(1)
    
    