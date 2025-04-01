#!/usr/bin/env python3
import sys
import os
import shutil
sys.path.append('/home/users/treillou/Tools/')
sys.path.append('/home/users/treillou/Tools/Read_simulations')
sys.path.append('/home/users/treillou/Tools/Write_simulations')
sys.path.append('/home/users/treillou/Tools/Slurm')
sys.path.append('/home/users/treillou/Tools/Misc')
from extract_wavespectrum import extract_wavespectrum
from extract_mpi import extract_mpi
from write_bdrag import write_bdrag
from write_MPI import write_MPI
from write_time import write_time
from write_wavespectum import write_wavespectrum
from make_grid import make_grid
from write_grid import write_grid
# TODO: import bathy
from read_parameter import read_parameter
from create_jobsub import create_jobsub
#import submit_job
from copy_all_files import copy_all_files


def check_where(name,fixed_params,variable_params,default):
    if name in fixed_params:
        var=fixed_params[name]
    elif name in variable_params:
        var=variable_params[name][i]
    else:
        var=default
        print("No "+name+" provided, default value set to "+str(default))
    return var

if __name__ == "__main__":
    current_path = os.getcwd()
    config_name = os.path.basename(current_path)
    # Create prod directory
    if not os.path.exists("./Prod"):
        os.makedirs("Prod")
        print(f"Directory Prod created.")
    else:
        print(f"Directory Prod already exists.")
    print('**************************\n')
            
    # Find parameters
    base_params = extract_wavespectrum("./Routines","SWASH")
    fixed_params,variable_params = read_parameter()
    
    
    
    for i, name in enumerate(variable_params['JobName']):
        print(f"Run {i}: {name}")
        os.makedirs("./Prod/"+name)
        copy_all_files("./Routines", "./Prod/"+name)
        
        #
        # WAVE SPECTRUM
        #
        Hs=check_where("Hs",fixed_params,variable_params,base_params["Hs"])
        Tp=check_where("Tp",fixed_params,variable_params,base_params["Tp"])
        Theta=check_where("Theta",fixed_params,variable_params,base_params["Theta"])
        Sigma=check_where("Sigma",fixed_params,variable_params,base_params["Sigma"])
        Gamma=check_where("Gamma",fixed_params,variable_params,base_params["Gamma"])
        write_wavespectrum("./Prod/"+name,Hs,Tp,Theta,Sigma,Gamma,"SWASH")
        
        #
        # Bathymetry and grid
        #
        GridOption=check_where("GridOption",fixed_params,variable_params,None)
        if GridOption=="3DPlanar":
            wlevel=check_where("wlevel",fixed_params,variable_params,0.0)
            offset=check_where("offset",fixed_params,variable_params,20.0)
            Lx=check_where("Lx",fixed_params,variable_params,150.0)
            dx=check_where("dx",fixed_params,variable_params,1.0)
            Ly=check_where("Ly",fixed_params,variable_params,100.0)
            dy=check_where("dy",fixed_params,variable_params,1.0)
            beta=check_where("beta",fixed_params,variable_params,0.02)
            Mz=check_where("Mz",fixed_params,variable_params,10)
            args={"Option": GridOption,"wlevel": wlevel,"offset": offset,
                  "Lx": Lx,"dx": dx,"Ly": Ly,"dy": dy, "beta": beta}
            Lx,Ly,Mx,My,dx,dy=make_grid(".","SWASH",args)
            write_grid("./Prod/"+name,Lx,Ly,Mx-2,My-2,Mz,"grid.nc","SWASH")
            shutil.copyfile("./bathy.bot", "./Prod/"+name+"/bathy.bot")
        elif GridOption=="2Ddata":
            wlevel=check_where("wlevel",fixed_params,variable_params,0.75)
            filename=check_where("gridname",fixed_params,variable_params,'/home/users/treillou/Data/vanderWerf.csv')
            dx=check_where("dx",fixed_params,variable_params,0.1)
            Ly=check_where("Ly",fixed_params,variable_params,0.8)
            Mz=check_where("Mz",fixed_params,variable_params,10)
            args={"Option": GridOption,"wlevel": wlevel,"filename": filename,
                         "dx": dx, "Ly": Ly}
            Lx,Ly,Mx,My,dx,dy=make_grid(".","SWASH",args)
            print("1")
            write_grid("./Prod/"+name,Lx,Ly,Mx-2,My-2,Mz,"grid.nc","SWASH")
            shutil.copyfile("./bathy.bot", "./Prod/"+name+"/bathy.bot")
        elif GridOption==None:
            print('No grid provided. Use default Routines grid.')
        print("2")
        
        #
        # Bottom friction
        #
        bdrag=check_where("bdrag",fixed_params,variable_params,1.e-5)
        write_bdrag("./Prod/"+name,bdrag,"SWASH")
        print("3")

        #
        # TIME
        #
        SimTime=check_where("SimTime",fixed_params,variable_params,3600)
        dt=check_where("dt",fixed_params,variable_params,0.05)
            
        write_time("./Prod/"+name,SimTime,dt,"SWASH")
        print("4")
        
        #
        # MPI
        #
        CPUs=check_where("CPUs",fixed_params,variable_params,None)
        if CPUs==None:
            CPUs=extract_mpi("./Routines","SWASH")
        else:
            write_MPI("./Prod/"+name,CPUs,"SWASH")
            
        #
        # Jobsub
        # 
        JobTime=check_where("JobTime",fixed_params,variable_params,"01:00:00")
        
        os.chdir("./Prod/"+name)
        create_jobsub(name,JobTime,CPUs,config_name,"SWASH")
        os.chdir("../..")
        print('**************************\n')