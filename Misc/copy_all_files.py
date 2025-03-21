#!/usr/bin/env python3
import sys
import os
import shutil

def copy_all_files(src_dir, dst_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Iterate over all files in the source directory
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        # Only copy files (skip directories)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)