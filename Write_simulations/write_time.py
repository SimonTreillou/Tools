#!/usr/bin/env python3
import os
import re
import sys

def write_timeCROCO(repo,simtime,dt):
    num_dt = int(float(simtime)/float(dt))
    lines=[]
    with open(repo+'/croco.in', "r") as file:
        for num, line in enumerate(file):
            if "time_stepping:" in line:
                lines.append(num)
    with open(repo+'/croco.in', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line+1]="               "+str(num_dt)+"     "+str(dt)+"      5      10\n"
    with open(repo+'/croco.in', 'w') as file:
        file.writelines(data)
        
def write_timeSWASH(repo,simtime,dt):
    hours = int(simtime//3600)
    mins = int((simtime-hours*3600)//60)
    secs = int(simtime-mins*60-hours*3600)
    lines=[]
    with open(repo+'/input.sws', "r") as file:
        for num, line in enumerate(file):
            if "COMPUTE" in line:
                lines.append(num)
    with open(repo+'/input.sws', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line]="COMPUTE 000000.000 "+str(dt)+" SEC {:02d}{:02d}{:02d}.000\n".format(hours,mins,secs)
    with open(repo+'/input.sws', 'w') as file:
        file.writelines(data)
        
def write_time(repo,simtime,dt,model="CROCO"):
    if model == "CROCO":
        write_timeCROCO(repo,simtime,dt)
    elif model == "SWASH":
        write_timeSWASH(repo,simtime,dt)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python write_time.py <repository> <simulation time (s)> <dt> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 4:
        repo = sys.argv[1]
        simtime = sys.argv[2]
        dt = sys.argv[3]
        write_time(repo,simtime,dt)
    elif len(sys.argv) == 5:
        repo = sys.argv[1]
        simtime = sys.argv[2]
        dt = sys.argv[3]
        model = sys.argv[4]
        write_time(repo,simtime,dt,model)
    else:
        print("Usage: python write_time.py <repository> <simulation time (s)> <dt> OPT:<model>")
        sys.exit(1)
        sys.exit(1)