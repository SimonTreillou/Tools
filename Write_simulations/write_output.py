#!/usr/bin/env python3
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

def write_outputCROCO(repo,Freq):
    lines=[]
    with open(repo+'/croco.in', "r") as file:
        for num, line in enumerate(file):
            if "LDEFHIS" in line:
                lines.append(num)
    with open(repo+'/croco.in', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line+1]="            T     "+str(Freq)+"        0\n"
    with open(repo+'/croco.in', 'w') as file:
        file.writelines(data)
    lines=[]
    with open(repo+'/croco.in', "r") as file:
        for num, line in enumerate(file):
            if "NTSAVG" in line:
                lines.append(num)
    with open(repo+'/croco.in', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line+1]="            1     "+str(Freq)+"         0\n"
    with open(repo+'/croco.in', 'w') as file:
        file.writelines(data)
        
def write_outputSWASH(repo,Freq):
    replace_string_in_file(repo+"/input.sws", "Freq", str(int(Freq)))
        
def write_output(repo,Freq,model="CROCO"):
    if model == "CROCO":
        write_outputCROCO(repo,Freq)
    elif model == "SWASH":
        write_outputSWASH(repo,Freq)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python write_output.py <Freq (s)> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 3:
        repo = sys.argv[1]
        freq = sys.argv[2]
        write_output(repo,freq)
    elif len(sys.argv) == 4:
        repo = sys.argv[1]
        freq = sys.argv[2]
        model = sys.argv[3]
        write_output(repo,freq,model)
    else:
        print("Usage: python write_output.py <Freq (s)> <dt> OPT:<model>")
        sys.exit(1)
        sys.exit(1)