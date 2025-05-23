import os
import json
import tarfile
import io
import csv
from pathlib import Path
import argparse
import webdataset as wds
import random
from tqdm import tqdm
import math
import subprocess
import tempfile


def load_jsonl(jsonl_path):
    """Load the JSONL file and return a mapping from cleaned file names to their JSON metadata."""
    mapping = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
            file_name = record.get("file_name", "")
            if file_name.startswith("tar/"):
                file_name = file_name[4:]
            # Remove any extension if present and strip spaces
            file_name = Path(file_name).stem.strip()
            # if file_name in mapping:
            #     print(f"Duplicate file name found: {file_name}")
            mapping[file_name] = record
    return mapping

def load_mapping_csv(csv_path):
    """Load the CSV mapping file that maps tar files to mp4 names."""
    mapping = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract the mp4 name without extension and clean it
            mp4_name = row['mp4_name']
            if mp4_name.endswith('.mp4'):
                mp4_name = mp4_name[:-4]  # Remove .mp4 extension
            if mp4_name.startswith('tar/'):
                mp4_name = mp4_name[4:]  # Remove tar/ prefix
            
            # Clean the mp4 name to match the format in the JSONL mapping
            mp4_name = mp4_name.replace('.', '_').strip()
            
            mapping.append({
                'tar_file': row['tar_file'],
                'mp4_name': mp4_name
            })
    return mapping

def process_audio_direct(mapping, csv_mapping, audio_dir, output_dir, train_keys, valid_keys, shard_size=2048):
    """Process audio files directly from tar files based on the CSV mapping.
    
    Creates two folders inside output_dir:
      - "train" for training samples (sourced from train_keys)
      - "valid" for validation samples (sourced from valid_keys)
    """
    # Create output directories for training and validation
    train_out_dir = os.path.join(output_dir, "train")
    valid_out_dir = os.path.join(output_dir, "valid")
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(valid_out_dir, exist_ok=True)

    # Create shard writers for training and validation shards
    train_pattern = os.path.join(train_out_dir, "%04d.tar")
    valid_pattern = os.path.join(valid_out_dir, "%04d.tar")
    
    # Convert csv_mapping (list of dicts) to a dictionary where mp4_name is the key
    # and the value is the dictionary with tar_file information
    csv_mapping_dict = {}
    for item in csv_mapping:
        mp4_name = item['mp4_name']
        csv_mapping_dict[mp4_name] = item["tar_file"]
    
    # Join the two dictionaries
    # Create a new joined dictionary where keys are mp4_name
    # and values contain both the metadata from mapping and tar_file from csv_mapping
    joined_mapping = {}
    for mp4_name, metadata in mapping.items():
        if mp4_name in csv_mapping_dict:
            # Create a new dictionary with all metadata from mapping
            joined_item = metadata.copy()
            # Add the tar_file information from csv_mapping
            joined_item['tar_file'] = csv_mapping_dict[mp4_name]
            joined_mapping[mp4_name] = joined_item
    
    print("len joined mapping", len(joined_mapping), "len mapping", len(mapping))
    assert len(joined_mapping) == len(mapping), "Joined mapping and mapping have different lengths"
    
    # Group the joined mapping by tar file for efficient processing
    tar_to_mp4s = {}
    for mp4_name, data in joined_mapping.items():
        tar_file = data['tar_file']
        if tar_file not in tar_to_mp4s:
            tar_to_mp4s[tar_file] = []
        tar_to_mp4s[tar_file].append(mp4_name)
    
    # Initialize counters and shard writers
    count_train = 0
    count_valid = 0
    sink_train = wds.ShardWriter(train_pattern, maxcount=shard_size)
    sink_valid = wds.ShardWriter(valid_pattern, maxcount=shard_size)
    
    # Process each tar file
    for tar_file, mp4_names in tqdm(tar_to_mp4s.items(), desc="Processing tar files", position=0):
        tar_path = os.path.join(audio_dir, tar_file)
        
        # Skip if tar file doesn't exist
        if not os.path.exists(tar_path):
            print(f"Warning: Tar file {tar_path} not found, skipping.")
            continue
        
        # Process each mp4 in the tar file
        with tarfile.open(tar_path, 'r') as tar:
            for mp4_name in mp4_names:
                # Get the full metadata for this mp4
                metadata = joined_mapping[mp4_name]
                
                # Determine if this sample should go to train or valid
                is_train = mp4_name in train_keys
                is_valid = mp4_name in valid_keys
                
                if not (is_train or is_valid):
                    continue  # Skip if not in either set
                
                member = tar.getmember("tar/" + mp4_name + ".mp4")
                if member is None:
                    print(f"Warning: Audio for {mp4_name} not found in {tar_file}")
                    continue
                
                # Skip if the member is not a file
                if not member.isfile():
                    print(f"Warning: {mp4_name} is not a file in {tar_file}")
                    continue
                
                # Extract the audio file to memory
                f = tar.extractfile(member)                
                try:
                    # Read MP4 data into memory
                    mp4_data = f.read()
                    f.close()
                    
                    # Create temporary files for MP4 and FLAC data
                    with tempfile.NamedTemporaryFile(suffix='.mp4') as mp4_temp, \
                         tempfile.NamedTemporaryFile(suffix='.flac') as flac_temp:
                        
                        # Write MP4 data to temporary file
                        mp4_temp.write(mp4_data)
                        mp4_temp.flush()
                        
                        # Use ffmpeg to convert MP4 to FLAC using temporary files (16kHz mono)
                        process = subprocess.Popen(
                            ['ffmpeg', '-i', mp4_temp.name, '-ar', '16000', '-ac', '1', '-f', 'flac', flac_temp.name],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        
                        # Wait for the conversion to complete
                        _, error = process.communicate()
                        
                        if process.returncode != 0:
                            print(f"Warning: Failed to convert {mp4_name} to FLAC: {error.decode('utf-8')}")
                            continue
                        
                        # Read the converted FLAC data
                        flac_temp.seek(0)
                        audio_data = flac_temp.read()
                        
                except Exception as e:
                    print(f"Error converting {mp4_name} to FLAC: {str(e)}")
                    continue
                                
                json_member = tar.getmember("json/" + mp4_name + ".json")
                if json_member is None:
                    print(f"Warning: JSON for {mp4_name} not found in {tar_file}")
                    continue
                json_data = tar.extractfile(json_member).read()
                json_data = json.loads(json_data.decode('utf-8'))
                
                # Create a sample with the audio data and metadata
                sample = {
                    "__key__": mp4_name,
                    "json": json.dumps(json_data).encode('utf-8'),
                    "flac": audio_data
                }
                    
                # Write to the appropriate sink
                if is_train:
                    sink_train.write(sample)
                    count_train += 1
                elif is_valid:
                    sink_valid.write(sample)
                    count_valid += 1
                    
    # Close the shard writers
    sink_train.close()
    sink_valid.close()
    
    print(f"Created {count_train} training samples and {count_valid} validation samples.")

    # -------------------------------------------------------------------
    # Create sizes.json files for train and valid shards
    # -------------------------------------------------------------------

    # For training shards:
    num_train_shards = math.ceil(count_train / shard_size) if count_train > 0 else 0
    sizes_train = {}
    for i in range(num_train_shards):
        shard_name = f"{i:04d}.tar"
        if i < num_train_shards - 1:
            sizes_train[shard_name] = shard_size
        else:
            sizes_train[shard_name] = count_train - i * shard_size
    with open(os.path.join(train_out_dir, "sizes.json"), "w") as f:
        json.dump(sizes_train, f, indent=4)
    print(f"Created sizes.json for train with {num_train_shards} shard(s).")

    # For validation shards:
    num_valid_shards = math.ceil(count_valid / shard_size) if count_valid > 0 else 0
    sizes_valid = {}
    for i in range(num_valid_shards):
        shard_name = f"{i:04d}.tar"
        if i < num_valid_shards - 1:
            sizes_valid[shard_name] = shard_size
        else:
            sizes_valid[shard_name] = count_valid - i * shard_size
    with open(os.path.join(valid_out_dir, "sizes.json"), "w") as f:
        json.dump(sizes_valid, f, indent=4)
    print(f"Created sizes.json for valid with {num_valid_shards} shard(s).")


def main():
    parser = argparse.ArgumentParser(description='Create a webdataset combining text metadata and audio samples.')
    parser.add_argument('--jsonl_path', type=str, default='./output/final_combined_outputs_filtered_0.50_3.0time_with_Qwen2.5_simple_True_0_20000.jsonl',
                        help='Path to the JSONL file with text data.')
    parser.add_argument('--audio_dir', type=str, default='./video',
                        help='Directory containing audio tar files.')
    parser.add_argument('--output_dir', type=str, default='./training_qwen2.5_simple_0_20000',
                        help='Output directory for the webdataset shards.')
    parser.add_argument('--mapping_csv', type=str, default='tar_mp4_mapping.csv',
                        help='Path to the CSV file mapping tar files to mp4 names.')
    parser.add_argument('--shard_size', type=int, default=2048,
                        help='Number of samples per shard.')
    parser.add_argument('--num_valid', type=int, default=100,
                        help='Number of samples for validation.')
    args = parser.parse_args()

    print(f"Loading JSONL data from {args.jsonl_path}")
    mapping = load_jsonl(args.jsonl_path)
    print(f"Loaded {len(mapping)} text samples from JSONL.")

    print(f"Loading CSV mapping from {args.mapping_csv}")
    mapping_csv = os.path.join(args.audio_dir, args.mapping_csv)
    csv_mapping = load_mapping_csv(mapping_csv)
    print(f"Loaded {len(csv_mapping)} mappings from CSV.")

    # Randomly sample num_valid keys for validation from the mapping keys and assign the rest for training.
    all_keys = list(mapping.keys())
    num_valid = min(args.num_valid, len(all_keys))
    valid_keys = set(random.sample(all_keys, num_valid))
    train_keys = set(all_keys) - valid_keys
    print(f"Assigned {len(train_keys)} samples to train and {len(valid_keys)} samples to validation.")

    process_audio_direct(mapping, csv_mapping, args.audio_dir, args.output_dir, train_keys, valid_keys, shard_size=args.shard_size)


if __name__ == '__main__':
    main() 