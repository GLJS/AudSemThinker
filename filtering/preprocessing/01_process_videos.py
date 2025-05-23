import tarfile
import json
import os
import subprocess
from pathlib import Path
import re
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import glob
import polars as pl
import io
import tempfile
import shutil
import argparse
from glob import glob

# Add this near the top of the file, after the imports
BUFFER_TIME = float(os.environ.get('BUFFER_TIME', '0.5'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_and_join_data(dataset_path: str, subtitles_path: str) -> pl.DataFrame:
    logging.info(f"Reading CSV files: {dataset_path} and {subtitles_path}")
    
    dataset = pl.read_csv(dataset_path)
    subtitles = pl.read_csv(subtitles_path)
    
    # Convert start and duration to float
    dataset = dataset.with_columns([
        pl.col('start').cast(pl.Float64),
        pl.col('duration').cast(pl.Float64)
    ])
    
    subtitles = subtitles.with_columns([
        pl.col('start').cast(pl.Float64),
        pl.col('duration').cast(pl.Float64)
    ])
    
    # Join the dataframes
    joined_data = dataset.join(
        subtitles,
        on=['video_id', 'start'],
        how='left'
    ).rename({'text': 'processed_text', 'text_right': 'original_text'})
    
    logging.info("Finished reading and joining CSV files")
    return joined_data


def process_video_file(video_content: bytes, video_filename: str, dataset: pl.DataFrame, index: int, scratch_dir: Path) -> Tuple[str, str]:
    match = re.match(r'(.+) \((\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\)\.(\w+)', video_filename)
    if not match:
        logging.warning(f"Could not parse filename: {video_filename}")
        return None, None
    

    video_id, start, end, ext = match.groups()
    start = float(start)
    end = float(end)
    duration = round(end - start, 3)
    
    # Use the scratch directory for the temporary file
    with tempfile.NamedTemporaryFile(suffix=f'.{ext}', dir=scratch_dir, delete=False) as temp_file:
        temp_file.write(video_content)
        temp_file.flush()
        
        # Check actual video duration
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", temp_file.name
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            actual_duration = float(result.stdout)
        except ValueError:
            logging.warning(f"Failed to get actual duration for {video_filename}")
            os.unlink(temp_file.name)  # Remove the temporary file
            return None, None
        
        if duration < 0:
            duration = actual_duration

    if actual_duration < 1 or actual_duration > 60:
        logging.info(f"Skipping {video_filename}: duration {actual_duration} seconds is out of range")
        os.unlink(temp_file.name)  # Remove the temporary file
        return None, None

    video_id = video_id.split("/")[-1]

    # Filter the dataset for the matching row
    epsilon = 0.001  # Small tolerance for floating-point comparison
    matching_row = dataset.filter(
        (pl.col('video_id') == video_id) &
        (pl.col('start').is_between(start - epsilon, start + epsilon))
    )

    if matching_row.is_empty():
        logging.warning(f"No matching row found for {video_filename}")
        os.unlink(temp_file.name)  # Remove the temporary file
        return None, None
    else:
        row = matching_row.to_dicts()[0]

    metadata = {
        "video_id": video_id,
        "start": start,
        "end": end,
        "duration": duration,
        "actual_duration": actual_duration,
        "text": row['processed_text'],
        "original_filename": video_filename,
        "original_text": row['original_text'] if row['original_text'] else ""
    }

    flac_filename = scratch_dir / f"{index}.flac"
    json_filename = scratch_dir / f"{index}.json"

    # Calculate the start time for ffmpeg
    start_time = max(0, actual_duration - duration - BUFFER_TIME)
    
    to_time = min(duration + BUFFER_TIME, actual_duration - start_time)
    try:
        # Convert video to .flac, taking only the last part
        subprocess.run([
            "ffmpeg", "-i", temp_file.name, 
            "-y",
            "-ss", str(start_time),  # Start time
            "-t", str(to_time),  # Duration to extract
            "-ac", "1", "-ar", "48000", "-vn",
            "-sample_fmt", "s16", "-acodec", "flac", 
            str(flac_filename), "-loglevel", "error"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to convert video to .flac: {e}")
        os.unlink(temp_file.name)
        return None, None

    # Remove the temporary input file
    os.unlink(temp_file.name)

    with open(json_filename, 'w') as f:
        json.dump(metadata, f)

    return str(flac_filename), str(json_filename)

def create_tar_file(tar_index: int, files: List[Tuple[str, str]], output_dir: Path):
    tar_filename = output_dir / f"{tar_index}.tar"
    logging.info(f"Creating tar file: {tar_filename}")
    with tarfile.open(tar_filename, "w") as tar:
        for flac_file, json_file in tqdm(files, desc=f"Creating {tar_filename}"):
            try:
                tar.add(flac_file, arcname=Path(flac_file).name)
                tar.add(json_file, arcname=Path(json_file).name)
                os.remove(flac_file)
                os.remove(json_file)
            except Exception as e:
                logging.error(f"Failed to add file to tar: {e}")
    logging.info(f"Finished creating tar file: {tar_filename}")

def get_last_tar_file_count(output_dir: Path) -> Tuple[int, int]:
    tar_files = sorted([f for f in output_dir.glob("*.tar")], key=lambda x: int(x.stem))
    if not tar_files:
        return 0, 0
    last_tar = tar_files[-1]
    with tarfile.open(last_tar, "r") as tar:
        count = len(tar.getmembers()) // 2
    logging.info(f"Last tar file: {last_tar}, contains {count} pairs")
    return int(last_tar.stem), count

def process_tar_gz(tar_gz_path: str, dataset: pl.DataFrame, output_dir: Path, scratch_dir: Path, start_index: int, start_tar_index: int, start_tar_count: int):
    index = start_index
    tar_index = start_tar_index
    current_tar_files = []

    logging.info(f"Processing tar.gz file: {tar_gz_path}")
    with tarfile.open(tar_gz_path, "r|gz") as tar:
        for member in tqdm(tar, desc=f"Processing {Path(tar_gz_path).name}"):
            try:
                if not member.isfile():
                    continue

                video_content = tar.extractfile(member).read()
                video_filename = member.name

                flac_file, json_file = process_video_file(video_content, video_filename, dataset, index, scratch_dir)
            except Exception as e:
                logging.error(f"Failed to process video file: {e}")
                continue

            if flac_file and json_file:
                current_tar_files.append((flac_file, json_file))
                index += 1

                if len(current_tar_files) + start_tar_count == 4096:
                    create_tar_file(tar_index, current_tar_files, output_dir)
                    current_tar_files = []
                    tar_index += 1
                    start_tar_count = 0
                

    logging.info(f"Finished processing tar.gz file: {tar_gz_path}")
    return index, tar_index, current_tar_files

def main():
    parser = argparse.ArgumentParser(description='Process video files from a tar.gz archive')
    parser.add_argument('--tar-file', type=str, required=True, help='Path to specific tar.gz file to process')
    args = parser.parse_args()

    logging.info("Starting video processing script")
    
    joined_data = read_and_join_data("dataset.csv", "sub3_extracted_startend.csv")
    base_path = glob("./subs/*")[-1]
    input_dir = Path(base_path) / "subs"
    input_dir.mkdir(exist_ok=True)
    output_dir = Path(base_path) / "processed"
    output_dir.mkdir(exist_ok=True)
    scratch_dir = Path(base_path) / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    os.chdir(output_dir)

    start_tar_index, start_tar_count = get_last_tar_file_count(output_dir)
    index = start_tar_index * 4096 + start_tar_count + 1
    current_tar_files = []

    if args.tar_file:
        # Move tar file to input_dir directory if not already there
        if str(input_dir) not in str(args.tar_file):
            new_tar_path = input_dir / Path(args.tar_file).name
            shutil.copy2(args.tar_file, new_tar_path)
            args.tar_file = new_tar_path
            logging.info(f"Moved tar file to input directory: {args.tar_file}")

        # Process single tar file
        logging.info(f"Processing single tar.gz file: {args.tar_file}")
        index, start_tar_index, current_tar_files = process_tar_gz(
            str(args.tar_file), joined_data, output_dir, scratch_dir,
            index, start_tar_index, len(current_tar_files)
        )

    # Create the last tar file if there are remaining files
    if current_tar_files:
        create_tar_file(start_tar_index, current_tar_files, output_dir)

    logging.info("Finished processing all tar.gz files")

    # Move processed tar files to shared scratch directory
    tar_basename = Path(args.tar_file).name.split(".")[0]
    shared_output_dir = Path("./yt/out") / tar_basename
    shared_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Move all tar files from output_dir to shared output directory
    for tar_file in output_dir.glob("*.tar"):
        target_path = shared_output_dir / tar_file.name
        shutil.move(str(tar_file), str(target_path))
        logging.info(f"Moved processed tar file to shared output: {target_path}")

if __name__ == "__main__":
    main()