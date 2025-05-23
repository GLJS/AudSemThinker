import webdataset as wds
import os
from pathlib import Path
import re
import tarfile
import io
import json
from tqdm import tqdm
import time

def fix_tar_names(directory):
    """
    Fix tar file names to have proper keys when loading with webdataset and save back to tar.
    The keys should include start and end times, with video key as 'mp4' and json key as 'json'.
    """
    # Get all tar files in directory
    tar_files = sorted(Path(directory).glob('*.tar'))
    
    for tar_path in tqdm(tar_files, desc="Processing tar files"):

        # Create temporary tar file
        temp_tar_path = tar_path.with_suffix('.tar.tmp')
        
        # Open original tar for reading and new tar for writing
        with tarfile.open(tar_path, 'r') as src_tar, \
             tarfile.open(temp_tar_path, 'w') as dst_tar:

            should_replace = True
            # Process each file in the tar
            for member in src_tar.getmembers():
                # Skip if not a file
                if not member.isfile():
                    continue

                # Skip if file already has "tar/" prefix
                # if member.name.startswith('tar/'):
                #     print(f"Skipping {member.name} because it already has 'tar/' prefix")
                #     should_replace = False
                #     break
                
                # Extract original filename components
                basename = os.path.basename(member.name)
                ext = os.path.splitext(basename)[-1]
                key_match = re.match(r'(.+) \((.+)-(.+)\)', basename)
                
                if not key_match:
                    continue
                    
                video_id, start, end = key_match.groups()
                
                # Create new key format
                new_key = f"{video_id} ({start}-{end})"
                new_key = new_key.replace(".", "_")
                new_key = "tar/" + new_key + ext
                # Read file content
                f = src_tar.extractfile(member)
                content = f.read()
                
                # Create new tar info with updated name
                new_info = tarfile.TarInfo()
                new_info.name = new_key
                new_info.size = len(content)
                
                # Add to new tar
                dst_tar.addfile(new_info, io.BytesIO(content))
        
        # Replace original tar with new one only if we didn't break out
        if should_replace:
            os.replace(temp_tar_path, tar_path)

if __name__ == "__main__":
    input_dir = "./yt_out/video_6mil_8mil/"
    fix_tar_names(input_dir)
    print(f"Fixed tar files, saved to {input_dir}")

