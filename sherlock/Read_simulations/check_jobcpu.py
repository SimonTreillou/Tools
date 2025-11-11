#!/usr/bin/env python3
import sys
sys.path.append('/home/users/treillou/Tools/Read_simulations')
import os
import re
import subprocess
import csv
from extract_dim import extract_dim
from extract_nb_dt import extract_nb_dt


def get_job_cores(jobid):
    try:
        # Command to get the number of cores with sacct
        command = ['sacct', '--format=JobID,ReqCPUS', '--noheader', '--jobs={}'.format(jobid)]
        
        # Running the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
        
        # Extracting the number of cores from the result
        output = stdout.decode().strip()
        lines = output.split('\n')
        
        for line in lines:
            parts = line.split()
            if parts[0] == str(jobid):
                return int(parts[1])
                
        raise ValueError(f"No such job with ID: {jobid}")

    except subprocess.CalledProcessError as e:
        print("An error occurred while running sacct command:", e.output.decode())
        return None

def convert_cpu_time_to_seconds(cpu_time):
    if '-' in cpu_time:
        days, time_part = cpu_time.split('-')
        days = int(days)
    else:
        days = 0
        time_part = cpu_time

    hours, minutes, seconds = map(int, time_part.split(':'))
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds

def get_job_id(directory):
    file_pattern = re.compile(r'.+\.(\d+)\.out$')
    extracted_numbers = []
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        match = file_pattern.match(filename)
        if match:
            number = match.group(1)
            extracted_numbers.append(number)
    print(len(extracted_numbers))
    if len(extracted_numbers) == 0:
        return None
    else:
        return max([int(i) for i in extracted_numbers])

def get_job_details(job_id):
    result = subprocess.run(['seff', job_id], stdout=subprocess.PIPE)
    output = result.stdout.decode()
    # Use regular expressions to extract the desired information
    cpu_time_pattern = re.compile(r'CPU Utilized: (.+)')
    cpu_efficiency_pattern = re.compile(r'CPU Efficiency: (\d+.\d+)')
    walltime_pattern = re.compile(r'Job Wall-clock time: (.+)')
    memory_utilized_pattern = re.compile(r'Memory Utilized: (.+)')
    nodes_pattern = re.compile(r'Nodes: (.+)')
    cores_per_node_pattern = re.compile(r'Cores per node: (.+)')

    cpu_time = re.search(cpu_time_pattern, output).group(1)
    cpu_efficiency = re.search(cpu_efficiency_pattern, output).group(1)
    wall_time = re.search(walltime_pattern, output).group(1)
    memory_utilized = re.search(memory_utilized_pattern, output).group(1)
    if "Nodes" in output:
        nodes = re.search(nodes_pattern, output).group(1)
        cores_per_node = re.search(cores_per_node_pattern, output).group(1)
        cores = str(int(cores_per_node)*int(nodes))
    else:
        cores = re.search(re.compile(r'Cores: (.+)'), output).group(1)
        
    return {
        "CPUTime": cpu_time,
        "Efficiency": cpu_efficiency,
        "WallTime": wall_time,
        "Memory": memory_utilized,
        "Cores": cores
    }

def find_job_ids_with_paths(directory):
    # Pattern to match files: {jobname}-{jobid}.out
    file_pattern = re.compile(r'(.+)-(\d+)\.out$')
    job_runs = {}

    # Iterate through files in the directory and its subdirectories
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for filename in files:
                match = file_pattern.match(filename)
                if match:
                    jobname = match.group(1)
                    job_id = int(match.group(2))
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(root, directory)

                    key = (relative_path, jobname)  # Use the relative path and jobname as the composite key

                    if key not in job_runs or job_id > job_runs[key]['job_id']:
                        job_runs[key] = {'job_id': job_id, 'file_path': file_path}
    
    return job_runs

def get_nb_points(job_path):
    dir = os.path.dirname(job_path)
    if 'croco' in os.listdir(dir):
        dimensions = extract_dim(dir, "CROCO")
    else:
        dimensions = extract_dim(dir, "SWASH")
    if dimensions == None:
        return None
    else:
        return int(dimensions["x"]) * int(dimensions["y"]) * int(dimensions["z"])

def get_nb_dt(job_path):
    dir = os.path.dirname(job_path)
    if 'croco' in os.listdir(dir):
        nbdt = extract_nb_dt(dir, "CROCO")
    else:
        nbdt = extract_nb_dt(dir, "SWASH")
    if nbdt == None:
        return None
    else:
        return nbdt

def write_job_details_to_csv(job_runs, csv_filename, base_path):
    existing_jobs = set()

    # Read the existing CSV file if it exists and collect existing job IDs
    if os.path.isfile(csv_filename):
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_jobs.add(int(row['JobID']))
    
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['JobID', 'FilePath', 'CPUTime', 'Efficiency', 'WallTime', 'Memory', 'Cores', 'usTime']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the CSV file is newly created (check its size)
        if os.path.getsize(csv_filename) == 0:
            writer.writeheader()

        for jobname, run_info in job_runs.items():
            print(jobname)
            job_id = run_info['job_id']
            if job_id in existing_jobs:
                continue  # Skip processing this job if it already exists in the CSV file

            job_path = run_info['file_path']
            nb_grid = get_nb_points(job_path)
            nb_dt = get_nb_dt(job_path)
            # Ensure that only base_path is removed and preserve subdirectory structure
            relative_path = os.path.relpath(os.path.dirname(job_path), base_path)

            details = get_job_details(str(job_id))
            if nb_dt is None or nb_grid is None:
                us_time = None
            else:
                us_time = convert_cpu_time_to_seconds(details['CPUTime']) / int(nb_dt) / int(nb_grid)
            
            row = {'JobID': job_id, 'FilePath': relative_path}
            row.update(details)
            print(nb_dt)
            row.update({'usTime': us_time})
            writer.writerow(row)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        base_directory = "/scratch/users/treillou"
    elif len(sys.argv) == 2:
        base_directory = "/scratch/users/treillou/"+sys.argv[1]
    else:
        print("Too much arguments.")
        
    job_runs = find_job_ids_with_paths(base_directory)
    csv_file = os.path.join(base_directory, 'job_details.csv')
    write_job_details_to_csv(job_runs, csv_file, base_directory)
    print(f"Job details have been written to {csv_file} in {base_directory}")
