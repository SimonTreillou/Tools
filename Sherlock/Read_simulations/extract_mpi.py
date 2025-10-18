import os
import re
import sys

def extract_mpiCROCO(directory):
    #directory = os.path.join('/scratch/users/treillou/',directory)
    #dir_comp = os.path.join(directory,'Compile')
    param_file = None

    # Search for param.h in the directory
    for root, _, files in os.walk(directory):
        if "param.h" in files:
            param_file = os.path.join(root, "param.h")
            break

    if not param_file:
        print("Error: param.h file not found in the specified directory.")
        return None

    # Regular expressions to match dimensions
    regex_patterns = {
        "NPXI" : re.compile(r"NP_XI\s*=\s*(\d+)"),
        "NPETA": re.compile(r"NP_ETA\s*=\s*(\d+)"),
    }

    dimensions = {}

    # Read the file and extract dimensions
    with open(param_file, "r") as file:
        for line in file:
            for key, pattern in regex_patterns.items():
                match = pattern.search(line)
                if match:
                    dimensions[key] = int(match.group(1))

    # Check if all dimensions were found
    if len(dimensions) == 2:
        #print(dimensions["NPXI"]*dimensions["NPETA"])
        return dimensions["NPXI"]*dimensions["NPETA"]
    else:
        print("Error")
        return None
        
def extract_mpiSWASH(directory):
    #directory = os.path.join('/scratch/users/treillou/',directory)
    param_file = None

    # Search for jobsub in the directory
    for root, _, files in os.walk(directory):
        if "jobsub" in files:
            param_file = os.path.join(root, "jobsub")
            break

    if not param_file:
        print("Error: jobsub file not found in the specified directory.")
        return None

    # Regular expressions to match dimensions
    regex_patterns = {
        "MPI" : re.compile(r"--ntasks=(\d+)"),
    }

    nb_mpi = None
    # Read the file and extract dimensions
    with open(param_file, "r") as file:
        for line in file:
            for key, pattern in regex_patterns.items():
                match = pattern.search(line)
                if match:
                    nb_mpi = int(match.group(1))

    # Check if all dimensions were found
    if nb_mpi != None:
        #print(nb_mpi)
        return nb_mpi
    else:
        print("Error")
        return None

def extract_mpi(directory,model="CROCO"):
    if model=="CROCO":
        return extract_mpiCROCO(directory)
    elif model=="SWASH":
        return extract_mpiWASH(directory)
    else:
        print("Error: Invalid model")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_MPI.py <directory> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 2:
        directory = sys.argv[1]
        extract_mpi(directory)
    elif len(sys.argv) ==3:
        directory = sys.argv[1]
        model = sys.argv[2]
        extract_mpi(directory,model)
    else:
        print("Usage: python extract_MPI.py <directory> OPT:<model>")
        sys.exit(1)