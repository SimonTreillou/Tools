#!/usr/bin/env python3
import os
import re
import sys

def write_bdragCROCO(repo,Zob):
    lines=[]
    with open(repo+'/croco.in', "r") as file:
        for num, line in enumerate(file):
            if "bottom_drag:" in line:
                lines.append(num)
    with open(repo+'/croco.in', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line+1]="                 0.0e-04         0.0e-3   "+str(Zob)+"      1.d-5    1.d-2\n"
    with open(repo+'/croco.in', 'w') as file:
        file.writelines(data)

def write_bdragSWASH(repo,Zob):
    lines=[]
    with open(repo+'/input.sws', "r") as file:
        for num, line in enumerate(file):
            if "FRIC" in line:
                lines.append(num)
    with open(repo+'/input.sws', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line]="FRIC LOG ROUGH "+str(Zob)+"\n"
    with open(repo+'/input.sws', 'w') as file:
        file.writelines(data)

def write_bdrag(repo,Zob,model="CROCO"):
    if model == "CROCO":
        write_bdragCROCO(repo,Zob)
    elif model == "SWASH":
        write_bdragSWASH(repo,Zob)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python write_bdrag.py <repository> <Zob (m)> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 3:
        repo = sys.argv[1]
        Zob = sys.argv[2]
        write_bdrag(repo,Zob)
    elif len(sys.argv) == 4:
        repo = sys.argv[1]
        Zob = sys.argv[2]
        model = sys.argv[3]
        write_bdrag(repo,Zob,model)
    else:
        print("Usage: python write_bdrag.py <repository> <Zob (m)> OPT:<model>")
        sys.exit(1)