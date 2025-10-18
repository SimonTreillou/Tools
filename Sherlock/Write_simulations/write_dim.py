import os
import re
import sys

def replace_string_in_file(file_path, target_string, replacement_string):
    # Read the content of the file
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    # Replace 'Mx' with the replacement string
    updated_content = file_content.replace(target_string, replacement_string)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_content)
        
def write_dimCROCO(repo,Lx,Ly):
    print("Not needed to write in ana_grid.F as we use now grid.nc input!")
    
def activate_gridinput(repo,active):
    lines=[]
    with open(repo+'/cppdefs.h') as myFile:
        for num, line in enumerate(myFile, 1):
            if "ANA_GRID" in line:
                lines.append(num)
    with open(repo+'/cppdefs.h', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        if active:
            data[line-1]="# undef ANA_GRID\n"
        else:
            data[line-1]="# define ANA_GRID\n"
    with open(repo+'/cppdefs.h', 'w', encoding='utf-8') as file:
        file.writelines(data)
        

def write_gridCROCO(repo,Lx,Ly,Mx,My,Mz,gridname):
    # Write grid dimensions
    lines=[]
    with open(repo+'/param.h') as myFile:
        for num, line in enumerate(myFile, 1):
            if "parameter (LLm0=" in line:
                lines.append(num)
    with open(repo+'/param.h', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line]="parameter (LLm0="+str(int(Mx))+",  MMm0="+str(int(My))+",  N="+str(int(Mz))+")\n"
    with open(repo+'/param.h', 'w', encoding='utf-8') as file:
        file.writelines(data)

    # Deactivate ANA_GRID
    activate_gridinput(repo,True)
    
    # Write grid name
    lines=[]
    with open(repo+'/croco.in') as myFile:
        for num, line in enumerate(myFile, 1):
            if "grid:" in line:
                lines.append(num)
    with open(repo+'/croco.in', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line+1]="            "+gridname+"\n"
    with open(repo+'/croco.in', 'w', encoding='utf-8') as file:
        file.writelines(data)
        
        
def write_gridSWASH(directory,Lx,Ly,Mx,My,Mz,gridname):
    replace_string_in_file(directory+"/input.sws", "Mx", str(Mx))
    replace_string_in_file(directory+"/input.sws", "My", str(My))
    replace_string_in_file(directory+"/input.sws", "Mz", str(Mz))
    replace_string_in_file(directory+"/input.sws", "Lx", str(Lx))
    replace_string_in_file(directory+"/input.sws", "Ly", str(Ly))


def write_grid(directory,Lx,Ly,Mx,My,Mz,gridname,model="CROCO"):
    if model=="CROCO":
        return write_gridCROCO(directory,Lx,Ly,Mx,My,Mz,gridname)
    elif model=="SWASH":
        return write_gridSWASH(directory,Lx,Ly,Mx,My,Mz,gridname)
    else:
        print("Error: Invalid model")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python write_grid.py <directory> <Lx> <Ly> <Mx> <My> <Mz> <gridname> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 8:
        directory = sys.argv[1]
        Lx = sys.argv[2]
        Ly = sys.argv[3]
        Mx = sys.argv[4]
        My = sys.argv[5]
        Mz = sys.argv[6]
        gridname = sys.argv[7]
        write_grid(directory,Lx,Ly,Mx,My,Mz,gridname)
    elif len(sys.argv) ==9:
        directory = sys.argv[1]
        Lx = sys.argv[2]
        Ly = sys.argv[3]
        Mx = sys.argv[4]
        My = sys.argv[5]
        Mz = sys.argv[6]
        gridname = sys.argv[7]
        model = sys.argv[8]
        write_grid(directory,Lx,Ly,Mx,My,Mz,gridname,model)
    else:
        print("Usage: python write_grid.py <directory> <Lx> <Ly> <Mx> <My> <Mz> <gridname> OPT:<model>")
        sys.exit(1)
