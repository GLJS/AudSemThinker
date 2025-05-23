import tarfile
import os
from pathlib import Path
import csv
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_files_in_tar(tar_path: Path) -> tuple[int, int, int]:
    """Count total files, mp4 files and json files in a tar archive"""
    total_count = 0
    mp4_count = 0
    json_count = 0
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    total_count += 1
                    if member.name.endswith('.mp4'):
                        mp4_count += 1
                    elif member.name.endswith('.json'):
                        json_count += 1
        return total_count, mp4_count, json_count
    except Exception as e:
        logging.error(f"Error processing {tar_path}: {str(e)}")
        return 0, 0, 0

def main():
    input_dir = Path('./data/video_6mil_8mil')
    
    # Get list of all tar files
    tar_files = sorted(input_dir.glob('*.tar'))
    
    # Create dict to store mp4 counts
    mp4_counts = {}
            
    # Process each tar file
    for tar_path in tqdm(tar_files, desc="Processing tar files"):
        total, mp4s, jsons = count_files_in_tar(tar_path)
        assert mp4s == jsons, f"Mismatch in {tar_path}: {mp4s} != {jsons}"
        assert total == mp4s + jsons, f"Mismatch in {tar_path}: {total} != {mp4s} + {jsons}"
        
        # Store mp4 count in dict
        mp4_counts[tar_path.name] = mp4s
    
    # Sort the dictionary by numeric part of keys before saving
    sorted_counts = dict(sorted(mp4_counts.items(), key=lambda item: int(item[0].split('.')[0])))
    
    # Save sorted dict to json file
    with open('./yt_out/video_6mil_8mil/sizes2.json', 'w') as f:
        json.dump(sorted_counts, f, indent=4)
    
    logging.info("MP4 counts saved to sizes2.json")

if __name__ == "__main__":
    main()
