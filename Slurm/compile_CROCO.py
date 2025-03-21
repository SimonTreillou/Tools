#!/usr/bin/env python3
import sys
import os  

def compile_CROCO():
    os.system('module load hdf5 netcdf netcdf-fortran; ./jobcomp > comp.out')

if __name__ == "__main__":
    compile_CROCO()