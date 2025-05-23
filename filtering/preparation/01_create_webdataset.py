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
import pandas as pd
import polars as pl
import io
import soundfile as sf
import numpy


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

def validate_audio(flac_data, mp4_name):
    """
    Validate audio data to ensure it's properly formatted and usable.
    
    Args:
        flac_data: Binary FLAC audio data
        mp4_name: Name of the file (for logging purposes)
    
    Returns:
        bool: True if audio is valid, False otherwise
        str: Error message if validation fails, None otherwise
    """
    try:
        # First basic check with soundfile
        audio, sr = sf.read(io.BytesIO(flac_data))
        
        # Check for empty audio
        if len(audio) == 0:
            return False, "audio is empty"
            
        # Check for NaN values
        if numpy.isnan(audio).any():
            return False, "audio contains NaN values"
            
        # Check for infinity values
        if numpy.isinf(audio).any():
            return False, "audio contains infinity values"
            
        # Check for reasonable length
        duration = len(audio) / sr
        if duration < 0.1:  # Less than 100ms
            return False, f"audio is too short ({duration:.2f}s)"
        if duration > 300:  # More than 5 minutes
            return False, f"audio is too long ({duration:.2f}s)"
            
        # Check audio levels
        max_amplitude = numpy.max(numpy.abs(audio))
        if max_amplitude < 0.001:  # Very low volume
            return False, f"audio is nearly silent (max amplitude: {max_amplitude})"
        
        # Try reading again to ensure it's consistently readable
        sf.read(io.BytesIO(flac_data))
        
        return True, None
        
    except Exception as e:
        return False, f"audio could not be read: {e}"

def create_question():
    """Create a question for the video based on the mapping."""
    questions = [
        "Describe the audio in detail.",
        "Please provide a detailed description of what you hear in this audio.",
        "What sounds and audio elements can you identify in this recording?",
        "Give a comprehensive breakdown of the audio content.",
        "Explain everything you hear in this audio clip.",
        "Walk me through all the sounds present in this recording.",
        "What's happening in this audio? Describe it thoroughly.",
        "Provide a detailed analysis of the audio elements you can detect.",
        "Tell me about all the audio components you can identify.",
        "Share a complete description of what this audio contains."
    ]
    return random.choice(questions)

def process_audio_direct(mapping, csv_mapping, audio_dir, output_dir, train_keys, valid_keys, shard_size=2048, semantic=False):
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
    
    # Group the mapping by tar file for efficient processing using polars
    df = pl.DataFrame(csv_mapping)
    tar_to_mp4s = df.group_by("tar_file").agg(pl.col("mp4_name").alias("mp4_names")).to_dict(as_series=False)
    
    # Initialize counters and shard writers
    count_train = 0
    count_valid = 0
    sink_train = None
    sink_valid = None
    omit_list = ["original caption", "original data", "original description", "original title"]
    
    # Process each tar file
    for item in tqdm(zip(tar_to_mp4s["tar_file"], tar_to_mp4s["mp4_names"]), desc="Processing tar files"):
        tar_file = item[0]
        mp4_names = item[1]
        tar_path = os.path.join(audio_dir, tar_file)
        
        # Skip if tar file doesn't exist
        if not os.path.exists(tar_path):
            print(f"Warning: Tar file {tar_path} not found, skipping...")
            continue
        
        extra_mapping = {}
        try:
            # Open the tar file
            with tarfile.open(tar_path, 'r') as tar:
                # Get all members of the tar file
                members = tar.getmembers()
                
                # Create a mapping from filenames to members for quick lookup
                member_dict = {}
                for member in members:
                    if member.name.endswith('.flac'):
                        # Extract the base name without extension and path
                        base_name = Path(member.name).stem
                        if base_name.startswith('tar/'):
                            base_name = base_name[4:]
                        member_dict[base_name] = member
                    elif member.name.endswith('.json'):
                        # Extract the base name without extension and path
                        content = json.load(tar.extractfile(member))
                        correct_name = content["original_filename"]
                        correct_name = Path(correct_name).stem
                        first_part, second_part = correct_name.split(" ")
                        second_part = second_part.replace(".", "_")
                        correct_name = f"{first_part} {second_part}"
                        if base_name.startswith('tar/'):
                            base_name = base_name[4:]
                        member_dict[f"{base_name}_json"] = member
                        extra_mapping[correct_name] = base_name
                
                # Process each mp4 name in this tar file
                for mp4_name in mp4_names:
                    # Skip if not in our JSONL mapping
                    if mp4_name not in mapping:
                        continue
                    
                    # Skip if caption is too long
                    if len(mapping[mp4_name]["caption"].split()) > 100:
                        print(f"Skipping {mp4_name} because caption is too long")
                        continue
                
                    if len(mapping[mp4_name]["caption"]) < 10:
                        print(f"Skipping {mp4_name} because caption is too short")
                        continue
                    if len(mapping[mp4_name]["thinking"]) < 10:
                        print(f"Skipping {mp4_name} because thinking is too short")
                        continue
                    if semantic:
                        if len(mapping[mp4_name]["semantic_elements"]) < 10:
                            print(f"Skipping {mp4_name} because semantic elements is too short")
                            continue
                    if any(omit in mapping[mp4_name]["thinking"] for omit in omit_list) \
                        or any(omit in mapping[mp4_name]["caption"] for omit in omit_list):
                        if semantic:
                            if any(omit in mapping[mp4_name]["semantic_elements"] for omit in omit_list):
                                print(f"Skipping {mp4_name} because thinking contains an omitted string")
                                continue
                    
                    # Clean caption
                    mapping[mp4_name]["caption"] = mapping[mp4_name]["caption"].replace("\n", "")
                    mapping[mp4_name]["thinking"] = mapping[mp4_name]["thinking"].replace("\n", "") 
                    mapping[mp4_name]["question"] = create_question()
                    
                    
                    # Find the corresponding flac file in the tar
                    correct_name = extra_mapping.get(mp4_name)
                    flac_member = member_dict.get(correct_name)
                    if not flac_member:
                        print(f"Skipping {mp4_name} because flac file not found")
                        continue
                    
                    # Extract the flac file
                    flac_data = tar.extractfile(flac_member).read()
                    
                    # Validate audio
                    is_valid, error = validate_audio(flac_data, mp4_name)
                    if not is_valid:
                        print(f"Skipping {mp4_name} because {error}")
                        continue
                    
                    # Create sample data
                    sample_data = {
                        "__key__": mp4_name,
                        "json": json.dumps(mapping[mp4_name]).encode("utf-8"),
                        "flac": flac_data
                    }
                    
                    # Initialize shard writers if not already done
                    if sink_train is None:
                        sink_train = wds.ShardWriter(train_pattern, maxcount=shard_size)
                    if sink_valid is None:
                        sink_valid = wds.ShardWriter(valid_pattern, maxcount=shard_size)
                    
                    # Write sample to the appropriate shard based on the key membership
                    if mp4_name in train_keys:
                        sink_train.write(sample_data)
                        count_train += 1
                        if count_train % shard_size == 0:
                            print(f"Wrote {count_train} training samples")
                    elif mp4_name in valid_keys:
                        sink_valid.write(sample_data)
                        count_valid += 1
                        if count_valid % shard_size == 0:
                            print(f"Wrote {count_valid} validation samples")
        
        except Exception as e:
            print(f"Error processing tar file {tar_path}: {e}")
            continue
    
    # Close shard writers
    if sink_train:
        sink_train.close()
    if sink_valid:
        sink_valid.close()
    
    print(f"Finished writing {count_train} training samples and {count_valid} validation samples to shards")

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
    parser.add_argument('--jsonl_path', type=str, default='./output/final_combined_outputs_filtered_0.50_3.0time_with_Qwen2.5_simple_True.jsonl',
                        help='Path to the JSONL file with text data.')
    parser.add_argument('--audio_dir', type=str, default='./audio',
                        help='Directory containing audio tar files.')
    parser.add_argument('--output_dir', type=str, default='./training_qwen2.5_simple',
                        help='Output directory for the webdataset shards.')
    parser.add_argument('--mapping_csv', type=str, default='tar_mp4_mapping.csv',
                        help='Path to the CSV file mapping tar files to mp4 names.')
    parser.add_argument('--shard_size', type=int, default=2048,
                        help='Number of samples per shard.')
    parser.add_argument('--num_valid', type=int, default=100,
                        help='Number of samples for validation.')
    parser.add_argument('--semantic', action='store_true',
                        help='Whether to use semantic elements.')
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

    process_audio_direct(mapping, csv_mapping, args.audio_dir, args.output_dir, train_keys, valid_keys, shard_size=args.shard_size, semantic=args.semantic)


if __name__ == '__main__':
    main() 