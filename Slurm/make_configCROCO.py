#!/usr/bin/env python3
import sys
import os
sys.path.append('/home/users/treillou/Tools/')
sys.path.append('/home/users/treillou/Tools/Read_simulations')
sys.path.append('/home/users/treillou/Tools/Write_simulations')
sys.path.append('/home/users/treillou/Tools/Slurm')
sys.path.append('/home/users/treillou/Tools/Misc')
from extract_wavespectrum import extract_wavespectrum
from write_bdrag import write_bdrag
from write_MPI import write_MPI
from write_time import write_time
from write_wavespectum import write_wavespectrum
# TODO: import bathy
from read_parameter import read_parameter
from create_jobsub import create_jobsub
from compile_CROCO import compile_CROCO
#import submit_job
from copy_all_files import copy_all_files


if __name__ == "__main__":
    # Create prod directory
    if not os.path.exists("./Prod"):
        os.makedirs("Prod")
        print(f"Directory Prod created.")
    else:
        print(f"Directory Prod already exists.")
    print('**************************\n')
            
    # Find parameters
    base_params = extract_wavespectrum("./Routines")
    fixed_params,variable_params = read_parameter()
    
    
    
    for i, name in enumerate(variable_params['JobName']):
        print(f"Run {i}: {name}")
        os.makedirs("./Prod/"+name)
        copy_all_files("./Routines", "./Prod/"+name)
        
        #
        # WAVE SPECTRUM
        #
        if "Hs" in fixed_params:
            Hs=fixed_params["Hs"]
        elif "Hs" in variable_params:
            Hs=variable_params["Hs"][i]
        else:
            Hs=base_params["Hs"]
            print("No Hs provided. Default: "+str(base_params["Hs"]))
        if "Tp" in fixed_params:
            Tp=fixed_params["Tp"]
        elif "Tp" in variable_params:
            Tp=variable_params["Tp"][i]
        else:
            Tp=base_params["Tp"]
            print("No Tp provided. Default: "+str(base_params["Tp"]))
        if "Theta" in fixed_params:
            Theta=fixed_params["Theta"]
        elif "Theta" in variable_params:
            Theta=variable_params["Theta"][i]
        else:
            Theta=base_params["Theta"]
            print("No Theta provided. Default: "+str(base_params["Theta"]))
        if "Sigma" in fixed_params:
            Sigma=fixed_params["Sigma"]
        elif "Sigma" in variable_params:
            Sigma=variable_params["Sigma"][i]
        else:
            Sigma=base_params["Sigma"]
            print("No Sigma provided. Default: "+str(base_params["Sigma"]))
        if "Gamma" in fixed_params:
            Gamma=fixed_params["Gamma"]
        elif "Gamma" in variable_params:
            Gamma=variable_params["Gamma"][i]
        else:
            Gamma=base_params["Gamma"]
            print("No Gamma provided. Default: "+str(base_params["Gamma"]))
        write_wavespectrum("./Prod/"+name,Hs,Tp,Theta,Sigma,Gamma)
        
        #
        # Bathymetry and grid
        #
        # TODO
        
        #
        # TIME
        #
        if "SimTime" in fixed_params:
            SimTime=fixed_params["SimTime"]
        elif "SimTime" in variable_params:
            SimTime=variable_params["SimTime"][i]
        else:
            SimTime="3600"
            print("No simulation time provided. Default: 3600 s")
        
        if "dt" in fixed_params:
            dt=fixed_params["dt"]
        elif "dt" in variable_params:
            dt=variable_params["dt"][i]
        else:
            dt="0.05"
            print("No timestep provided. Default: 0.05 s")
            
        write_time("./Prod/"+name,SimTime,dt)
        
        #
        # MPI
        #
        if "CPUs" in fixed_params:
            CPUs=fixed_params["CPUs"]
        elif "CPUs" in variable_params:
            CPUs=variable_params["CPUs"][i]
        else:
            CPUs="16"
            print("No number of CPUs provided. Default: 16")
        
        write_MPI("./Prod/"+name,CPUs)
        
        #
        # Jobsub
        # 
        if "JobTime" in fixed_params:
            JobTime=fixed_params["JobTime"]
        elif "JobTime" in variable_params:
            JobTime=variable_params["JobTime"][i]
        else:
            JobTime="01:00:00"
            print("No jobtime provided. Default: 01:00:00")
        
        os.chdir("./Prod/"+name)
        create_jobsub(name,JobTime,CPUs,"CROCO")
        compile_CROCO()
        os.chdir("../..")
        print('**************************\n')