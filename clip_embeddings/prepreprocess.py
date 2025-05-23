import os
import shutil
from pathlib import Path

def get_max_tar_number(directory):
    """Get the highest tar file number in a directory."""
    max_num = -1
    for file in os.listdir(directory):
        if file.endswith('.tar'):
            num = int(file.split('.')[0])
            max_num = max(max_num, num)
    return max_num

def combine_directories(source_dirs, output_dir):
    """Combine and renumber tar files from multiple directories."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    current_number = 0
    
    for source_dir in source_dirs:
        print(f"Processing directory: {source_dir}")
        source_dir = os.path.join("./tar_audio", source_dir)
        
        # Get all tar files and sort them numerically
        tar_files = []
        for file in os.listdir(source_dir):
            if file.endswith('.tar'):
                number = int(file.split('.')[0])
                tar_files.append((number, file))
        tar_files.sort()  # Sort by number
        
        # Copy and rename files
        for _, old_name in tar_files:
            new_name = f"{current_number}.tar"
            src_path = os.path.join(source_dir, old_name)
            dst_path = os.path.join(output_dir, new_name)
            shutil.move(src_path, dst_path)
            print(f"Moved {old_name} -> {new_name}")
            current_number += 1

if __name__ == "__main__":
    # List all source directories in order
    source_dirs = [
        "video_0mil_2mil",
        "video_0mil_2mil2",
        "video_2mil_4mil",
        "video_2mil_4mil2",
        "video_4mil_6mil_0",
        "video_4mil_6mil_1",
        "video_4mil_6mil_2",
        "video_4mil_6mil_3",
        "video_6mil_8mil",
        "video_8mil_10_mil",
        "video_8mil_10mil2"
    ]
    
    output_dir = "./audio/"
    combine_directories(source_dirs, output_dir)
