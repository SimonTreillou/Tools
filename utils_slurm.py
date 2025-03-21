import re

def modify_CPUs(filename, num_CPU):
    # Read the file
    with open(filename, "r") as file:
        lines = file.readlines()
    # Modify the line containing "#SBATCH --ntasks="
    updated_lines = [
        re.sub(r"(#SBATCH --ntasks=)\d+", rf"\g<1>{num_CPU}", line)
        for line in lines
    ]
    # Write back the modified file
    with open(filename, "w") as file:
        file.writelines(updated_lines)
    
    # Read the file
    with open(filename, "r") as file:
        lines = file.readlines()
    # Modify the line containing "mpirun -np ..."
    updated_lines = [
        re.sub(r"(mpirun\s+-np\s+)\d+", rf"\g<1>{num_CPU}", line)
        for line in lines
    ]
    # Write back the modified file
    with open(filename, "w") as file:
        file.writelines(updated_lines)
    
    print(f"Updated ntasks to {num_CPU} in {filename}")

def modify_time(filename,new_time):
    # Read the file
    with open(filename, "r") as file:
        lines = file.readlines()
    # Modify the line containing "#SBATCH --time="
    updated_lines = [
        re.sub(r"(#SBATCH --time=)\d+:\d+:\d+", rf"\g<1>{new_time}", line)
        for line in lines
    ]
    # Write back the modified file
    with open(filename, "w") as file:
        file.writelines(updated_lines)
    print(f"Updated time to {new_time} in {filename}")
    
def modify_jobname(filename,new_name):
    with open(filename, "r") as file:
        lines = file.readlines()
    updated_lines = [
        re.sub(r"(#SBATCH --job-name=)\S+", rf"\g<1>{new_name}", line)
        for line in lines
    ]
    with open(filename, "w") as file:
        file.writelines(updated_lines)
    print(f"Updated jobname to {new_name} in {filename}")
    
def modify_name(filename,new_name):
    with open(filename, "r") as file:
        lines = file.readlines()
    updated_lines = [
        re.sub(r"(workdir=)\S+", rf"\g<1>{new_name}", line)
        for line in lines
    ]
    with open(filename, "w") as file:
        file.writelines(updated_lines)
    print(f"Updated name to {new_name} in {filename}")