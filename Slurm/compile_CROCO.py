#!/usr/bin/env python3
import sys
import os  
sys.path.append('/home/users/treillou/Tools/')
from read_parameter import read_parameter

def compile_CROCO():
    os.system('module purge; module load gcc openmpi hdf5 netcdf-fortran; ./jobcomp > comp.out')

if __name__ == "__main__":
    fixed_params,variable_params = read_parameter()
    for i, name in enumerate(variable_params['JobName']):
        print(f"Run {i}: {name}")
        os.chdir("./Prod/"+name)
        compile_CROCO()
        os.chdir("../..")