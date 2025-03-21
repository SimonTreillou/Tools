#!/usr/bin/env python3
import sys
import os
sys.path.append('/home/users/treillou/Tools/Read_simulations')
import re
import extract_wavespectrum

#
# TODO: add the possibility to change the spectrum?
#

def write_wavespectrumCROCO(repo,Hs,Tp,theta,sigma,gamma):
    lines=[]
    with open(repo+'/croco.in', "r") as file:
        for num, line in enumerate(file):
            if "wave_maker:" in line:
                lines.append(num)
    with open(repo+'/croco.in', 'r', encoding='utf-8') as file:
        data = file.readlines()
    for line in lines:
        data[line+1]="             "+str(round(float(Hs)/(8**0.5),3))+"     "+str(Tp)+"       "+str(theta)+"         "+str(sigma)+"               "+str(gamma)+"\n"
    with open(repo+'/croco.in', 'w') as file:
        file.writelines(data)
        
def write_wavespectrumSWASH(repo,Hs,Tp,Theta,Sigma,Gamma):
    param_file = None

    # Search for input.sws in the directory
    for root, _, files in os.walk(repo):
        if "input.sws" in files:
            param_file = os.path.join(root, "input.sws")
            break

    # If input.sws file does not exist, create a new one
    if not param_file:
        param_file = os.path.join(repo, "input.sws")

    # Initialize patterns for gamma and spect
    gamma_pattern = re.compile(r'(BOU SHAPESPEC JONSWAP\s+)([\d.]+)')
    spect_pattern = re.compile(r"(.+SPECT )(\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+)( CYCLE 3600 SEC)")

    # Read the file content if it exists
    lines = []
    if os.path.exists(param_file):
        with open(param_file, 'r') as file:
            lines = file.readlines()

    # Update the existing lines or add new ones if not found
    gamma_found = False
    spect_found = False

    for i, line in enumerate(lines):
        gamma_match = gamma_pattern.search(line)
        if gamma_match:
            lines[i] = f"{gamma_match.group(1)}{Gamma}\n"
            gamma_found = True

        spect_match = spect_pattern.search(line)
        if spect_match:
            # Preserve the part before SPECT and update the values after SPECT
            lines[i] = f"{spect_match.group(1)}{Hs} {Tp} {Theta} {Sigma} {spect_match.group(6)}\n"
            spect_found = True

    # Add new lines if they were not found
    if not gamma_found:
        lines.append(f"BOU SHAPESPEC JONSWAP {Gamma}\n")
    if not spect_found:
        lines.append(f"BOU SIDE W CCW BTYPE WEAK SMOO 10 SEC ADDB CON SPECT {Hs} {Tp} {Theta} {Sigma}\n")

    # Write the updated lines back to the file
    with open(param_file, 'w') as file:
        file.writelines(lines)
        
        
def write_wavespectrum(repo,Hs,Tp,theta,sigma,gamma=3.3,model="CROCO"):
    if model == "CROCO":
        write_wavespectrumCROCO(repo,Hs,Tp,theta,sigma,gamma)
    elif model == "SWASH":
        write_wavespectrumSWASH(repo,Hs,Tp,theta,sigma,gamma)

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python write_wavespectrum.py <repository> <Hs> <Tp> <theta> <sigma> OPT:<gamma>")
        sys.exit(1)
    elif len(sys.argv) == 6:
        repo = sys.argv[1]
        Hs = sys.argv[2]
        Tp = sys.argv[3]
        theta = sys.argv[4]
        sigma = sys.argv[5]
        write_wavespectrum(repo,Hs,Tp,theta,sigma)
    elif len(sys.argv) == 7:
        repo = sys.argv[1]
        Hs = sys.argv[2]
        Tp = sys.argv[3]
        theta = sys.argv[4]
        sigma = sys.argv[5]
        gamma = sys.argv[6]
        write_wavespectrum(repo,Hs,Tp,theta,sigma,gamma)
    elif len(sys.argv) == 8:
        repo = sys.argv[1]
        Hs = sys.argv[2]
        Tp = sys.argv[3]
        theta = sys.argv[4]
        sigma = sys.argv[5]
        gamma = sys.argv[6]
        model = sys.argv[7]
        write_wavespectrum(repo,Hs,Tp,theta,sigma,gamma,model)
    else:
        print("Usage: python write_wavespectrum.py <repository> <Hs> <Tp> <theta> <sigma> OPT:<gamma>")
        sys.exit(1)