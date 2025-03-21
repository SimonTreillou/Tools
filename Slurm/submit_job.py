#!/usr/bin/env python3
import sys
import os  

def submit_jobsub():
    os.system('sbatch jobsub')
    
if __name__ == "__main__":
    submit_jobsub()