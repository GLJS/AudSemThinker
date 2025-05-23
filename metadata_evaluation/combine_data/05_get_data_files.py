import pandas as pd
import tarfile
import os
from pathlib import Path
import subprocess
from tqdm import tqdm
from glob import glob
import json
import argparse

parser = argparse.ArgumentParser(description='Extract and convert files with type selection')
parser.add_argument('--type', choices=['simple', 'semantic', 'both'], default='both',
                   help='Type of processing to perform (simple, semantic, or both)')

def extract_and_convert_files(df, tmp_dir, output_subdir):
    """
    Extract mp4 files from tar archives and convert them to mp3 audio files.
    
    Args:
        df: DataFrame containing tar_file and mp4_name columns
        tmp_dir: Directory to save extracted mp3 files
        output_subdir: Subdirectory name (simple or semantic) for output
    """
    output_dir = f'./output/sample/audio_{output_subdir}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mapping dictionary and counter for indexed filenames
    file_mapping = []
    file_counter = 1
    
    # Group by tar file
    grouped = df.groupby('tar_file')
    
    # Process each tar file
    tar_base_path = "./video"
    for tar_file, group in tqdm(grouped, desc="Processing tar files"):
        tar_file = os.path.join(tar_base_path, tar_file)
        if not os.path.exists(tar_file):
            print(f"Warning: Tar file {tar_file} not found, skipping...")
            continue
            
        try:
            # Open tar file
            with tarfile.open(tar_file, 'r') as tar:
                # Extract each mp4 file
                for _, row in group.iterrows():
                    mp4_name = "tar/" + row['file_name'] + ".mp4"
                    
                    try:
                        # Extract mp4 file to temporary location
                        tar.extract(mp4_name, path=tmp_dir)
                        
                        # Generate output mp3 filename with index
                        original_name = Path(mp4_name).stem
                        mp3_name = f"{file_counter}.mp3"
                        mp3_path = os.path.join(output_dir, mp3_name)
                        
                        # Convert mp4 to mp3 using ffmpeg
                        mp4_path = os.path.join(tmp_dir, mp4_name)
                        subprocess.run([
                            'ffmpeg', '-i', mp4_path,
                            '-vn',  # Disable video
                            '-ar', '16000',  # Sample rate
                            '-ac', '1',  # Mono audio
                            '-y',  # Overwrite output
                            mp3_path
                        ], check=True, capture_output=True)
                        
                        # Add mapping to list
                        file_mapping.append({
                            "caption_deepseek-qwen": row["caption_deepseek-qwen"],
                            "thinking_deepseek-qwen": row["thinking_deepseek-qwen"], 
                            "caption_qwen": row["caption_qwen"],
                            "thinking_qwen": row["thinking_qwen"],
                            "index": file_counter,
                            "original_name": original_name,
                            "audio": "audio/" + mp3_name
                        })
                        
                        file_counter += 1
                        
                        # Remove temporary mp4 file
                        os.remove(mp4_path)
                        
                    except Exception as e:
                        print(f"Error processing {mp4_name}: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"Error opening tar file {tar_file}: {str(e)}")
            continue
            
    # Save mapping to JSONL file
    mapping_file = os.path.join(output_dir, "file_mapping.jsonl")
    with open(mapping_file, 'w') as f:
        for item in file_mapping:
            f.write(json.dumps(item) + '\n')
    

def main():
    args = parser.parse_args()
    
    tmp_dir = "./audio_tmp/"
    
    # Create temporary extraction directory
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Process files based on argument
    if args.type in ['simple', 'both']:
        # Read simple CSV file
        df_simple = pd.read_csv(f'./output/sample/final_combined_outputs_all_models_simple.csv')
        print("Processing simple files...")
        extract_and_convert_files(df_simple, tmp_dir, 'simple')
        
        # Recreate tmp_dir if needed for semantic processing
        if args.type == 'both':
            os.makedirs(tmp_dir, exist_ok=True)
    
    if args.type in ['semantic', 'both']:
        # Read semantic CSV file
        df_semantic = pd.read_csv(f'./output/sample/final_combined_outputs_all_models_semantic.csv')
        print("Processing semantic files...")
        extract_and_convert_files(df_semantic, tmp_dir, 'semantic')

if __name__ == "__main__":
    main()
