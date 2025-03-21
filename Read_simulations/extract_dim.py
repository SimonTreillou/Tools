import os
import re
import sys

def extract_dimCROCO(directory):
    directory = os.path.join('/scratch/users/treillou/',directory)
    dir_comp = os.path.join(directory,'Compile')
    param_file = None

    # Search for param.h in the directory
    for root, _, files in os.walk(dir_comp):
        if "param_.f" in files:
            param_file = os.path.join(root, "param_.f")
            break

    if param_file==None:
        print("Error: param_.f file not found in the specified directory.")
        return None
        
    # Regular expressions to match dimensions
    regex_patterns = {
        "x": re.compile(r"LLm0\s*=\s*(\d+)"),
        "y": re.compile(r"MMm0\s*=\s*(\d+)"),
        "z": re.compile(r"N\s*=\s*(\d+)")
    }

    dimensions = {}

    # Read the file and extract dimensions
    with open(param_file, "r") as file:
        for line in file:
            for key, pattern in regex_patterns.items():
                match = pattern.search(line)
                if match:
                    dimensions[key] = int(match.group(1))

    # Check if all dimensions were found
    if len(dimensions) == 3:
        #print(f"Extracted dimensions: x={dimensions['x']}, y={dimensions['y']}, z={dimensions['z']}")
        return dimensions
    else:
        print("Error: Not all dimensions (NX, NY, NZ) were found in param.h.")
        return None
        
def extract_dimSWASH(directory):
    directory = os.path.join('/scratch/users/treillou/',directory)
    param_file = None
    
    # Search for param.h in the directory
    for root, _, files in os.walk(directory):
        if "PRINT" in files:
            param_file = os.path.join(root, "PRINT")
            break

    if param_file==None:
        print("Error: PRINT file not found in the specified directory.")
        return None
    
    dimensions = {}

    with open(param_file, "r") as file:
        for line in file:
            if "CGRID REGULAR" in line:  # Check for relevant lines
                # Extract only whole numbers (integers), ignoring decimals
                matches = re.findall(r"\b\d+\b(?!\.)", line)
                dimensions["x"]=matches[0]
                dimensions["y"]=matches[1]
            elif "VERTICAL" in line:
                dimensions["z"] = re.findall(r"\b\d+\b(?!\.)", line)[0]
    
    if dimensions:
        #print(f"Extracted dimensions: x={dimensions['x']}, y={dimensions['y']}, z={dimensions['z']}")
        return dimensions
    else:
        print("No matching integers found.")
        return None

def extract_dim(directory,model="CROCO"):
    if model=="CROCO":
        return extract_dimCROCO(directory)
    elif model=="SWASH":
        return extract_dimSWASH(directory)
    else:
        print("Error: Invalid model")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_dim.py <directory> OPT:<model>")
        sys.exit(1)
    elif len(sys.argv) == 2:
        directory = sys.argv[1]
        extract_dim(directory)
    elif len(sys.argv) ==3:
        directory = sys.argv[1]
        model = sys.argv[2]
        extract_dim(directory,model)
    else:
        print("Usage: python extract_dim.py <directory> OPT:<model>")
        sys.exit(1)
