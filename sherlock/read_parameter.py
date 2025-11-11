#!/usr/bin/env python3
import json
import sys

def read_parameter(fixed='fixed_params.json',var='variable_params.json'):
    if ("json" not in fixed) | ("json" not in var) :
        print("File must be a .json")
        return None
    else:
        # Read the fixed parameters from the JSON file
        with open(fixed, 'r') as file:
            fixed_params = json.load(file)
    
        # Read the variable parameters from the JSON file
        with open(var, 'r') as file:
            variable_params = json.load(file)
        
        return fixed_params,variable_params
        
if __name__ == "__main__":
    if len(sys.argv) == 1:
        test,testv=read_parameter()
        print(test['Hs'])
        print(testv)
    elif len(sys.argv) == 2:
        print("You specified the fixed params file")
        fixed = sys.argv[1]
        test,testv=read_parameter(fixed)
        print(test)
        print(testv)
    elif len(sys.argv) == 3:
        print("You specified the fixed params file AND the variable params file")
        fixed = sys.argv[1]
        var   = sys.argv[2]
        test,testv=read_parameter(fixed,var)
        print(test['Hs'])
        print(testv)
    else:
        print("Usage: python read_parameter.py <filename>")
        sys.exit(1)
        
