# Tools
## My Python (and shamefully Matlab) toolbox

## Directory: Bash
### Different bash scripts to use as commands
   - check_jobs: will check the wall-time, number of CPUs, etc. of simulations present in your $SCRATCH directory
   - compile_CROCO: will compile all the CROCO configurations (i.e. directories) present under ./Prod/
   - launch: will 'sbatch jobsub' all the configurations present under ./Prod/
   - launchseveral: will 'sbatch jobsub' n times the chosen configuration (useful to benchmark computational time)
   - make_configCROCO: with specified 'fixed_params.json' and 'variable_params.json' files, will create a directory ./Prod containing different CROCO simulations
   - make_configSWASH: same but with SWASH

## Diagnostics
   - skills module

## Directory: Misc
### Diverse stuff and Matlab scripts

## Directory: Demos
### Small demonstrator for ideas, such as numerical schemes, etc.

## Directory: Read_simulations
### Different scripts useful to read the input of a given simulation
   - check_jobcpu.py: will check the wall-time, number of CPUs, etc. of simulations present in your $SCRATCH directory
   - extract_dim.py: check dimensions (Lx, Ly, Lz)
   - extract_mpi.py: check number of CPUs
   - extract_nb_dt.py: check number of timesteps
   - extract_tstep.py: check timestep
   - extract_wavespectrum.py: check parameters of the input wave spectrum

## Directory: Write_simulations
### Different scripts useful to create a simulation
   - make_grid.py: create NetCDF file for the CROCO bathymetry
   - write_MPI.py: specify the number of CPUs used
   - write_bdrag.py: specify how to model the bottom drag
   - write_dim.py: specify the dimensions (Lx,Ly,Lz)
   - write_grid.py: create the grid (for CROCO and SWASH)
   - write_output.py: specify the output frequency
   - write_time.py: specify the simulation duration
   - write_wavespectum.py: specify wave spectrum durations

## Directory: Slurm
### Different scripts useful to run a simulation
   - create_jobsub.py: create the 'jobsub' file with your indications (name of the config, duration, etc.)
   - make_configSWASH.py: with specified 'fixed_params.json' and 'variable_params.json' files, will create a directory ./Prod containing different SWASH simulations
   - make_configCROCO.py: same but with CROCO
   - compile_CROCO.py: will compile all the CROCO configurations (i.e. directories) present under ./Prod/
   - launch.py: will 'sbatch jobsub' all the configurations present under ./Prod/
                                                                                     

## Resources and recommendations
- Post-processing and analysis scripts of Emma Shie Nuss (https://github.com/emmashie/funpy#)