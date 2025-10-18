import os
import re
import sys

def extract_wavespectrumCROCO(directory):
    #directory = os.path.join('/scratch/users/treillou/',directory)
    param_file = None

    # Search for croco.in in the directory
    for root, _, files in os.walk(directory):
        if "croco.in" in files:
            param_file = os.path.join(root, "croco.in")
            break

    if not param_file:
        print("Error: croco.in file not found in the specified directory.")
        return None

    with open(param_file, 'r') as file:
        content = file.read()

    params = {}
    lines = content.split('\n')
    
    wave_maker_index = -1
    for i, line in enumerate(lines):
        if "wave_maker:" in line:
            wave_maker_index = i + 1
            break
    
    if wave_maker_index != -1:
        values = lines[wave_maker_index].split()
        params = {
            "Hs": round(float(values[0])*8**0.5,3),
            "Tp": float(values[1]),
            "Theta": float(values[2]),
            "Sigma": float(values[3]),
            "Gamma": float(values[4])
        }
    
    return params
        
def extract_wavespectrumSWASH(directory):
    param_file = None

    # Search for input.sws in the directory
    for root, _, files in os.walk(directory):
        if "input.sws" in files:
            param_file = os.path.join(root, "input.sws")
            break
    # Initialize variables to store the values
    Hs = None
    Tp = None
    Theta = None
    Sigma = None
    Gamma = None
    
    # Define regular expressions to match the lines in the file
    gamma_pattern = re.compile(r'BOU SHAPESPEC JONSWAP\s+([\d.]+)')
    spect_pattern = re.compile(r'\b\S+\s+SPECT\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')    
    
    try:
        with open(param_file, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                gamma_match = gamma_pattern.search(line)
                if gamma_match:
                    Gamma = gamma_match.group(1)
                spect_match = spect_pattern.search(line)
                if spect_match:
                    Hs, Tp, Theta, Sigma = spect_match.groups()
    except:
        print(f"No wave spectrum found")

    params = {
            "Hs": float(Hs),
            "Tp": float(Tp),
            "Theta": float(Theta),
            "Sigma": float(Sigma),
            "Gamma": float(Gamma)
        }
    return params
    
def extract_wavespectrum(directory,model="CROCO"):
    if model=="CROCO":
        return extract_wavespectrumCROCO(directory)
    elif model=="SWASH":
        return extract_wavespectrumSWASH(directory)
    else:
        print("Error: Invalid model")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_wavespectrum.py <directory> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 2:
        directory = sys.argv[1]
        params = extract_wavespectrum(directory)
    elif len(sys.argv) ==3:
        directory = sys.argv[1]
        model = sys.argv[2]
        extract_wavespectrum(directory,model)
    else:
        print("Usage: python extract_wavespectrum.py <directory> OPT:<model>")
        sys.exit(1)
