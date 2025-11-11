#!/usr/bin/env python3

import sys
import os

def create_jobsubCROCO(job_name,time,cpus,config):
    with open('jobsub', 'w') as f:
        # Prepare SLURM options
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH -J {job_name}\n")
        f.write(f"#SBATCH --partition=serc\n")
        f.write(f"#SBATCH -o {job_name}-%j.out\n")
        f.write(f"#SBATCH -e {job_name}-%j.err\n")
        f.write(f"#SBATCH --time={time}\n")
        f.write(f"#SBATCH --ntasks={cpus}\n")
        f.write(f"#SBATCH --cpus-per-task=1\n")
        f.write(f"#SBATCH --mem-per-cpu=2G\n")
        f.write(f"#SBATCH --mail-user=treillou@stanford.edu\n")
        f.write(f"#SBATCH --mail-type=ALL\n")
        f.write(f"\n")
        
        # Load necessary modules
        f.write(f"module purge\n")
        f.write(f"module load gcc openmpi netcdf hdf5 netcdf-c netcdf-fortran\n")
        f.write(f"\n")
        
        # Define variables
        f.write(f"export croco=/home/groups/bakercm/CROCO/croco\n")
        f.write(f"export configs=/home/users/treillou/Configs\n")
        f.write(f"homedir=$SLURM_SUBMIT_DIR\n")
        f.write(f"workdir=$SCRATCH/{config}/{job_name}\n")
        f.write(f"\n")
        
        # Check if the name of the directory does not already exists
        f.write(f"count=1\n")
        f.write(f"while [ -d \"$workdir\" ]; do\n")
        f.write(f"  echo \"Directory $workdir already exists.\"\n")
        f.write(f"  count=$((count + 1))\n")
        f.write(f"  workdir=\"$SCRATCH/{config}/{job_name}_$count\"\n")
        f.write(f"done\n")
        f.write(f"\n")
        
        # Create directory
        f.write(f"if [ ! -d \"$SCRATCH/{config}\" ]; then\n")
        f.write(f"  mkdir $SCRATCH/{config}\n")
        f.write(f"fi\n")
        f.write(f"mkdir $workdir\n")
        f.write(f"echo \"$SLURM_JOB_ID\" > $workdir/jobid\n")
        f.write(f"\n")
        
        # Copy files
        f.write(f"cp croco*    $workdir\n")
        f.write(f"cp *.nc      $workdir\n")
        f.write(f"cp *.h       $workdir\n")
        f.write(f"cp job*      $workdir\n")
        f.write(f"cp *.F       $workdir\n")
        f.write(f"cp *.in      $workdir\n")
        f.write(f"mv ./Compile $workdir\n")
        f.write(f"cd $workdir\n")
        f.write(f"\n")
        
        # Run CROCO
        f.write(f"echo \"Launching CROCO...\"\n")
        f.write(f"mpirun -np {cpus} ./croco croco.in > croco.out\n") # To confirm
        f.write(f"echo \"... CROCO done\"\n")
        f.write(f"cp croco.out $homedir\n")
        
        return None
        
def create_jobsubSWASH(job_name,time,cpus,config):
    with open('jobsub', 'w') as f:
        # Prepare SLURM options
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH -J {job_name}\n")
        f.write(f"#SBATCH --partition=serc\n")
        f.write(f"#SBATCH -o {job_name}-%j.out\n")
        f.write(f"#SBATCH -e {job_name}-%j.err\n")
        f.write(f"#SBATCH --time={time}\n")
        f.write(f"#SBATCH --ntasks={cpus}\n")
        f.write(f"#SBATCH --cpus-per-task=1\n")
        f.write(f"#SBATCH --mem-per-cpu=6G\n")
        f.write(f"#SBATCH --mail-user=treillou@stanford.edu\n")
        f.write(f"#SBATCH --mail-type=ALL\n")
        f.write(f"\n")
        
        # Load necessary modules
        f.write(f"module purge\n")
        f.write(f"module load openmpi\n")
        f.write(f"\n")
        
        # Define variables
        f.write(f"homedir=$SLURM_SUBMIT_DIR\n")
        f.write(f"workdir=$SCRATCH/{config}/{job_name}\n")
        f.write(f"INPUT_DATA=input.sws\n")
        f.write(f"PATH=$PATH:/home/groups/bakercm/SWASH_GNU\n")
        f.write(f"\n")
        
        # Check if the name of the directory does not already exists
        f.write(f"count=1\n")
        f.write(f"while [ -d \"$workdir\" ]; do\n")
        f.write(f"  echo \"Directory $workdir already exists.\"\n")
        f.write(f"  count=$((count + 1))\n")
        f.write(f"  workdir=\"$SCRATCH/{config}/{job_name}_$count\"\n")
        f.write(f"done\n")
        f.write(f"\n")
        
        # Create directory
        f.write(f"if [ ! -d \"$SCRATCH/{config}\" ]; then\n")
        f.write(f"  mkdir $SCRATCH/{config}\n")
        f.write(f"fi\n")
        f.write(f"mkdir $workdir\n")
        f.write(f"echo \"$SLURM_JOB_ID\" > $workdir/jobid\n")
        f.write(f"\n")
        
        # Copy files
        f.write(f"cp * $workdir\n")
        f.write(f"cd $workdir\n")
        f.write(f"cp $INPUT_DATA INPUT\n")
        f.write(f"cp /home/groups/bakercm/SWASH_GNU/swash.exe .\n")
        f.write(f"cp /home/groups/bakercm/SWASH_GNU/swashrun .\n")
        f.write(f"chmod +rx swashrun\n")
        f.write(f"chmod +rx swash.exe\n")
        f.write(f"\n")
        
        # Create machine file
        f.write(f"srun hostname -s | sort -u > machinefile\n")
        
        # Run CROCO
        f.write(f"echo \"Launching SWASH...\"\n")
        f.write(f"#srun swash.exe\n") # To confirm
        f.write(f"swashrun -input input.sws -mpi {cpus}\n")
        f.write(f"echo \"... SWASH done\"\n")
        
        f.write(f"cp input.prt-001 PRINT")
        
        return None
        
def create_jobsub(job_name,time,cpus,config='',model="CROCO"):
    if model=="CROCO":
        create_jobsubCROCO(job_name,time,cpus,config)
    elif model=="SWASH":
        create_jobsubSWASH(job_name,time,cpus,config)
    else:
        print("Model not known")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 create_and_submit_job.py <job_name> <time> <number of CPUS> <config> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 5:
        job_name = sys.argv[1]
        time = sys.argv[2]
        cpus = sys.argv[3]
        config = sys.argv[4]
        create_jobsub(job_name,time,cpus,config,"CROCO")
    elif len(sys.argv) == 6:
        job_name = sys.argv[1]
        time = sys.argv[2]
        cpus = sys.argv[3]
        config = sys.argv[4]
        model = sys.argv[5]
        create_jobsub(job_name,time,cpus,config,model)
