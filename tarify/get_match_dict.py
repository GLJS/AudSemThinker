import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import webdataset as wds

def create_tar_mapping(directory: str) -> None:
    """
    Create a mapping between tar filenames and the mp4 files they contain using webdataset.
    Writes mapping directly to CSV file.
    
    Args:
        directory (str): Directory containing tar files
    """
    # Get all tar files
    total_length_of_files_in_dir = len(os.listdir(directory)) - 1
    tar_paths = f"{directory}{{0..{total_length_of_files_in_dir}}}.tar"
    
    # Set up output file
    output_path = os.path.join(directory, 'tar_mp4_mapping.csv')
    with open(output_path, 'w') as f:
        f.write("tar_file,mp4_name\n")
        
        dataset = wds.WebDataset(str(tar_paths))
        for idx, sample in tqdm(enumerate(dataset), total=total_length_of_files_in_dir*2048):
            try:
                tar_name = sample["__url__"].split("/")[-1]
                original_name = json.loads(sample["json"])["original_filename"]
                f.write(f"{tar_name},{original_name}\n")
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
    
    print(f"Mapping saved to {output_path}")

if __name__ == "__main__":
    create_tar_mapping("./data/video/")