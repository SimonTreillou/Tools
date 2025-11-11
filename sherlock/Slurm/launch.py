#!/usr/bin/env python3
import os
import sys
sys.path.append('/home/users/treillou/Tools/Slurm')
from submit_job import submit_jobsub

if __name__ == "__main__":
    os.chdir("./Prod/")
    for name in os.listdir():
        os.chdir(name)
        submit_jobsub()
        os.chdir("..")
    os.chdir("..")
        
        
        