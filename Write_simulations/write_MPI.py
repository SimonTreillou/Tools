#!/usr/bin/env python3
import os
import re
import sys
import math

def best_mpi_distribution(total_cpus):
    total_cpus = int(total_cpus)
    # Find factors
    factors = [(i, total_cpus // i) for i in range(1, int(math.sqrt(total_cpus)) + 1) if total_cpus % i == 0]
    
    # Sort factors to find the most balanced pair. If tied, prefer the one with larger y.
    best_pair = min(factors, key=lambda x: (abs(x[0] - x[1]), -x[1]))
    
    return {'x_mpi': best_pair[0], 'y_mpi': best_pair[1]}

def write_MPICROCO(repo,MPI):
    lines=[]
    cpus = best_mpi_distribution(MPI)
    with open(repo+'/param.h', "r") as file:
        for num, line in enumerate(file):
            if "NP_XI=" in line:
                lines.append(num)
    with open(repo+'/param.h', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line]="      parameter (NP_XI="+str(cpus['x_mpi'])+",  NP_ETA="+str(cpus['y_mpi'])+",  NNODES=NP_XI*NP_ETA)\n"
    with open(repo+'/param.h', 'w') as file:
        file.writelines(data)
        
def write_MPI(repo,MPI,model="CROCO"):
    if model == "CROCO":
        write_MPICROCO(repo,MPI)
    elif model == "SWASH":
        print("In SWASH, no need to change the number of cores... apparemment")
        return None
    else:
        print("Unknown model.")
        return None

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 3:
        print("Usage: python write_MPI.py <repository> <number of CPUs> OPT:<model>,default=CROCO")
        sys.exit(1)
    elif len(sys.argv) == 3:
        repo = sys.argv[1]
        MPI = sys.argv[2]
        write_MPI(repo,MPI)
    elif len(sys.argv) == 4:
        repo = sys.argv[1]
        MPI = sys.argv[2]
        model = sys.argv[3]
        write_MPI(repo,MPI,model)
    else:
        print("Usage: python write_MPI.py <repository> <number of CPUs> OPT:<model>,default=CROCO")
        sys.exit(1)