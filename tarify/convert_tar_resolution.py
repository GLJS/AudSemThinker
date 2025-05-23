import webdataset as wds
import os
from pathlib import Path
import tarfile
import io
import json
from tqdm import tqdm
import tempfile
import ffmpeg
import concurrent.futures
from glob import glob
import shutil

def get_scratch_dir():
    directory = glob("./tmp/*")
    if len(directory) == 0:
        raise ValueError("No scratch directory found")
    directory = os.path.join(directory[0], "tmp")
    os.makedirs(directory, exist_ok=True)
    return directory

def convert_video_360p(video_bytes, check_still_failed):
    """Converts a video to 360p, 2fps, mp4 with x264, and mono 32kHz 16bit audio."""
    try:
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(dir=get_scratch_dir(), suffix=".mp4", delete=False) as tmp_input_file, \
             tempfile.NamedTemporaryFile(dir=get_scratch_dir(), suffix=".mp4", delete=False) as tmp_output_file:
            
            # Write input video bytes to temporary file
            tmp_input_file.write(video_bytes)
            tmp_input_file.flush()
            
            # Convert video
            process = (
                ffmpeg
                .input(tmp_input_file.name)
                .output(tmp_output_file.name, 
                       format='mp4',
                       vf='scale=trunc(oh*a/2)*2:360',
                       r=2,
                       vcodec='libx264',
                       acodec='pcm_s16le',
                       ar='32000',
                       ac=1)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
                
            # Read the converted video into memory
            tmp_output_file.seek(0)
            converted_video_bytes = io.BytesIO(tmp_output_file.read())
            
            return converted_video_bytes

    except ffmpeg.Error as e:
        if check_still_failed:
            print(f"FFmpeg error: {e}")
        return None
    finally:
        # Clean up temporary files
        os.unlink(tmp_input_file.name)
        os.unlink(tmp_output_file.name)

def process_tar(tar_path):
    """Process a single tar file, converting videos to 360p."""
    print(f"Processing {tar_path.name}")
    full_path = os.path.join(os.path.dirname(tar_path), tar_path.name)
    temp_tar_path = full_path.replace('.tar', '.tar.tmp')
    
    try:
        # First pass: identify failed conversions
        failed_videos = set()
        with tarfile.open(full_path, 'r') as src_tar:
            for member in src_tar.getmembers():
                if member.isfile() and member.name.endswith('.mp4'):
                    video_content = src_tar.extractfile(member).read()
                    if convert_video_360p(video_content, False) is None:
                        failed_videos.add(member.name)
        
        # Create set of JSON files to skip
        skip_json_files = {video_name.replace('.mp4', '.json') for video_name in failed_videos}
        
        # Second pass: create new tar without failed videos and their JSONs
        with tarfile.open(full_path, 'r') as src_tar, \
             tarfile.open(temp_tar_path, 'w') as dst_tar:
            
            for member in src_tar.getmembers():
                if not member.isfile():
                    continue

                if member.name in failed_videos:
                    continue
                    
                if member.name in skip_json_files:
                    continue

                if member.name.endswith('.mp4'):
                    video_content = src_tar.extractfile(member).read()
                    converted_video = convert_video_360p(video_content, True)
                    
                    # This should always succeed since we filtered failures
                    converted_video.seek(0, os.SEEK_END)
                    member.size = converted_video.tell()
                    converted_video.seek(0)
                    dst_tar.addfile(member, converted_video)
                else:
                    # Copy non-video files directly
                    content = src_tar.extractfile(member).read()
                    dst_tar.addfile(member, io.BytesIO(content))

        # Replace original tar with new one
        os.replace(temp_tar_path, full_path)
        return True

    except Exception as e:
        print(f"Error processing {full_path}: {str(e)}")
        if os.path.exists(temp_tar_path):
            os.unlink(temp_tar_path)
        return False

def convert_tars_to_360p(directory, num_workers=None):
    """
    Convert all videos in tar files to 360p resolution.
    
    Args:
        directory (str): Directory containing tar files
        num_workers (int, optional): Number of worker processes. Defaults to CPU count - 1
    """
    if num_workers is None:
        # num_workers = 1
        num_workers = 126

    tar_files = sorted(Path(directory).glob('*.tar'))
    print(f"Found {len(tar_files)} tar files to process")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_tar, tar_files),
            total=len(tar_files),
            desc="Converting tars to 360p"
        ))

    successful = sum(1 for r in results if r)
    print(f"Successfully processed {successful} out of {len(tar_files)} tar files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert videos in tar files to 360p resolution')
    parser.add_argument('--input', type=str, default='./data/video_4mil_6mil/',
                      help='Input directory containing tar files')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    convert_tars_to_360p(args.input, args.workers) 